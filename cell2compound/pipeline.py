"""Shared response-prediction pipeline: pseudobulk loading, control-relative targets,
training-only gene selection, and the model set.

Extracted from scripts/expression_baseline.py so that any alternative evaluation
(leave-one-mechanism-class-out, response-magnitude strata, chemotype exclusions) uses the
identical data path and identical models as the headline result. An evaluation that changed
the pipeline would not be comparable to it, which is the whole point of a stress test.
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from cell2compound.chemistry import featurize
from cell2compound.compounds import load_source, map_observed_names
from cell2compound.expression import fit_mlp, predict_mlp, product_kernel, tanimoto

RIDGE_ALPHAS = [1., 10., 100., 1e3, 1e4, 1e5, 1e6]
KERNEL_ALPHAS = [1e-3, 1e-2, 1e-1, 1.]
DOSE_BANDWIDTHS = [.5, 1., 2., 1e6]
MLP_GRID = [(hidden, dropout, decay) for hidden in (64, 256) for dropout in (.2, .5) for decay in (1e-2, 1e-1)]
BASELINES = ('zero_response', 'training_mean_response', 'dose_mean_response')


@dataclass
class ResponseData:
    """Per compound/dose group: replicate deltas, features, and provenance."""
    groups: pd.DataFrame
    full_delta: np.ndarray
    rep_difference: np.ndarray
    fingerprints: np.ndarray
    features: np.ndarray
    doses: np.ndarray
    canonical: np.ndarray
    detected: np.ndarray
    cells: np.ndarray
    control_rows: np.ndarray
    gene_names: np.ndarray
    mapping: pd.DataFrame
    unmatched: list
    skipped: list

    def subset(self, keep):
        """Restrict to a boolean/index selection of groups, keeping arrays aligned."""
        keep = np.asarray(keep)
        if keep.dtype == bool:
            keep = np.flatnonzero(keep)
        return ResponseData(groups=self.groups.iloc[keep].reset_index(drop=True),
                            full_delta=self.full_delta[keep], rep_difference=self.rep_difference[keep],
                            fingerprints=self.fingerprints[keep], features=self.features[keep],
                            doses=self.doses[keep], canonical=self.canonical[keep],
                            detected=self.detected, cells=self.cells, control_rows=self.control_rows,
                            gene_names=self.gene_names, mapping=self.mapping,
                            unmatched=self.unmatched, skipped=self.skipped)


def load_response_data(path, source, control='control', control_matching='pooled', min_cells=10):
    """Build control-relative pseudobulk responses from a scripts/pseudobulk.py archive."""
    bulk = np.load(path)
    table = pd.DataFrame(bulk['groups'], columns=bulk['keys'])
    table['cells'] = bulk['cells']
    table['dose_value'] = table.dose_value.astype(float)
    sums = bulk['sums'].astype(np.float64)
    is_control = (table.perturbation == control).to_numpy()
    pooled = sums[is_control].sum(0) / table.cells[is_control].sum()
    per_plate = {}
    for plate in table.plate[is_control].unique():
        mask = is_control & (table.plate == plate).to_numpy()
        per_plate[plate] = sums[mask].sum(0) / table.cells[mask].sum()

    def reference(plate):
        return per_plate[plate] if control_matching == 'plate' else pooled

    mapping, unmatched = map_observed_names(table.perturbation[~is_control].unique(), load_source(source))
    structures = mapping[['parent_smiles']].drop_duplicates().rename(columns={'parent_smiles': 'smiles'})
    structures['compound_id'] = structures.smiles
    descriptors, fingerprints, rejected = featurize(structures)
    if len(rejected):
        raise ValueError(f'Parent structures failed RDKit parsing: {rejected.compound_id.tolist()}')
    index = {s: i for i, s in enumerate(descriptors.compound_id)}
    mapping['canonical_smiles'] = mapping.parent_smiles.map(lambda s: descriptors.canonical_smiles.iloc[index[s]])
    mapping['structure_row'] = mapping.parent_smiles.map(index)
    by_name = mapping.set_index('observed_name')

    rows, skipped = [], []
    for (drug, dose), group in table[~is_control].groupby(['perturbation', 'dose_value']):
        if drug not in by_name.index:
            continue
        rep_delta, rep_cells, rep_rows = [], [], []
        for _, part in group.groupby('replicate'):
            n = part.cells.sum()
            if n < min_cells:
                continue
            treated = sums[part.index].sum(0) / n
            control_profile = np.average([reference(p) for p in part.plate], axis=0, weights=part.cells)
            rep_delta.append(treated - control_profile); rep_cells.append(int(n)); rep_rows.append(part.index.to_numpy())
        if len(rep_delta) < 2:
            skipped.append({'compound': drug, 'dose': dose,
                            'cells_per_replicate': group.groupby('replicate').cells.sum().to_dict()})
            continue
        rows.append({'compound': drug, 'dose': dose, 'rep_delta': np.stack(rep_delta), 'rep_cells': rep_cells,
                     'table_rows': np.concatenate(rep_rows),
                     'canonical_smiles': by_name.canonical_smiles[drug], 'fp_row': by_name.structure_row[drug]})
    groups = pd.DataFrame(rows)
    doses = groups.dose.to_numpy()
    fp = fingerprints[groups.fp_row.to_numpy()]
    return ResponseData(groups=groups,
                        full_delta=np.stack([d.mean(0) for d in groups.rep_delta]),
                        rep_difference=np.stack([d[0] - d[1] for d in groups.rep_delta]),
                        fingerprints=fp, features=np.c_[fp, np.log10(doses)], doses=doses,
                        canonical=groups.canonical_smiles.to_numpy(), detected=bulk['detected'],
                        cells=table.cells.to_numpy(), control_rows=np.flatnonzero(is_control),
                        gene_names=bulk['genes'], mapping=mapping, unmatched=unmatched, skipped=skipped)


def select_genes(data, train, n_genes=1000, min_detection=.05):
    """Detected in enough training+control cells, then highest training-response variance.

    Uses only training groups, so a stress test cannot leak its held-out class into gene choice.
    """
    rows = np.r_[np.concatenate(data.groups.table_rows.iloc[train].to_list()), data.control_rows]
    detection = data.detected[rows].sum(0) / data.cells[rows].sum()
    eligible = np.flatnonzero(detection >= min_detection)
    return np.sort(eligible[np.argsort(data.full_delta[train][:, eligible].var(0))[-n_genes:]])


def nearest_neighbour(data, fit, test, y, k=5):
    """Tanimoto-weighted mean response of the k most similar fitted compounds at the same dose."""
    sims = tanimoto(data.fingerprints[test], data.fingerprints[fit])
    out = []
    for row, similarity in zip(test, sims):
        same = data.doses[fit] == data.doses[row]
        candidates = np.flatnonzero(same) if same.any() else np.arange(len(fit))
        top = candidates[np.argsort(-similarity[candidates])[:k]]
        weights = similarity[top] + 1e-6
        out.append((weights[:, None] * y[fit][top]).sum(0) / weights.sum())
    return np.stack(out)


def fit_predictions(data, y, train, valid, test, seed, device='cuda', models=None):
    """Every model from the headline evaluation. Hyperparameters chosen on `valid` only,
    then refitted on train+valid. Returns (predictions, chosen hyperparameters)."""
    fit = np.r_[train, valid]
    wanted = set(models) if models else None
    def want(name):
        return wanted is None or name in wanted
    predictions, chosen = {}, {}
    if want('zero_response'):
        predictions['zero_response'] = np.zeros((len(test), y.shape[1]))
    if want('training_mean_response'):
        predictions['training_mean_response'] = np.broadcast_to(y[fit].mean(0), (len(test), y.shape[1]))
    if want('dose_mean_response'):
        # A held-out dose cannot occur: all four doses appear in every split.
        dose_mean = {d: y[fit][data.doses[fit] == d].mean(0) for d in np.unique(data.doses[fit])}
        fallback = y[fit].mean(0)
        predictions['dose_mean_response'] = np.stack([dose_mean.get(d, fallback) for d in data.doses[test]])
    if want('tanimoto_knn'):
        predictions['tanimoto_knn'] = nearest_neighbour(data, fit, test, y)
    if want('ridge'):
        scores = []
        for alpha in RIDGE_ALPHAS:
            model = make_pipeline(StandardScaler(), Ridge(alpha=alpha)).fit(data.features[train], y[train])
            scores.append((float(np.mean((model.predict(data.features[valid]) - y[valid]) ** 2)), alpha))
        _, alpha = min(scores)
        chosen['ridge_alpha'] = alpha
        predictions['ridge'] = make_pipeline(StandardScaler(), Ridge(alpha=alpha)).fit(
            data.features[fit], y[fit]).predict(data.features[test])
    if want('tanimoto_kernel_ridge'):
        scores = []
        for bandwidth in DOSE_BANDWIDTHS:
            train_kernel = product_kernel(data.fingerprints[train], data.doses[train],
                                          data.fingerprints[train], data.doses[train], bandwidth)
            valid_kernel = product_kernel(data.fingerprints[valid], data.doses[valid],
                                          data.fingerprints[train], data.doses[train], bandwidth)
            for alpha in KERNEL_ALPHAS:
                model = KernelRidge(alpha=alpha, kernel='precomputed').fit(train_kernel, y[train])
                scores.append((float(np.mean((model.predict(valid_kernel) - y[valid]) ** 2)), alpha, bandwidth))
        _, alpha, bandwidth = min(scores)
        chosen['kernel_alpha'], chosen['dose_bandwidth'] = alpha, bandwidth
        fit_kernel = product_kernel(data.fingerprints[fit], data.doses[fit],
                                    data.fingerprints[fit], data.doses[fit], bandwidth)
        test_kernel = product_kernel(data.fingerprints[test], data.doses[test],
                                     data.fingerprints[fit], data.doses[fit], bandwidth)
        predictions['tanimoto_kernel_ridge'] = KernelRidge(alpha=alpha, kernel='precomputed').fit(
            fit_kernel, y[fit]).predict(test_kernel)
    if want('mlp'):
        scores = []
        for hidden, dropout, decay in MLP_GRID:
            _, val, epoch = fit_mlp(data.features[train], y[train], hidden, dropout, decay, 3000,
                                    device, seed, data.features[valid], y[valid], patience=150)
            scores.append((val, hidden, dropout, decay, epoch))
        _, hidden, dropout, decay, epoch = min(scores)
        chosen['mlp'] = {'hidden': hidden, 'dropout': dropout, 'weight_decay': decay, 'epochs': epoch}
        model, _, _ = fit_mlp(data.features[fit], y[fit], hidden, dropout, decay, epoch, device, seed)
        predictions['mlp'] = predict_mlp(model, data.features[test], device)
    return predictions, chosen


def replicate_floor(data, rows, genes):
    """Lower bound on attainable RMSE from replicate disagreement.

    rep1 and rep2 are separate plates. With independent, equal-variance cell noise the error of
    a two-replicate mean is half the replicate difference, so RMSE(rep1-rep2)/2 estimates the
    error of the observed target itself. No model can do better than this.
    """
    return float(np.sqrt(np.mean(data.rep_difference[rows][:, genes] ** 2)) / 2)


def attainable_fraction(model_rmse, baseline_rmse, floor_rmse, min_headroom_ratio=.05):
    """Share of the attainable improvement that a model captured.

    0 means no better than the baseline, 1 means it reached the noise floor. Returns None when
    the baseline already sits at (or within min_headroom_ratio of) the replicate floor: there the
    ratio divides by a near-zero headroom and explodes, which would be a reporting artifact
    rather than a finding. Report the median and a win count alongside the mean for the same
    reason.
    """
    headroom = baseline_rmse - floor_rmse
    if headroom <= max(1e-9, min_headroom_ratio * floor_rmse):
        return None
    return float((baseline_rmse - model_rmse) / headroom)
