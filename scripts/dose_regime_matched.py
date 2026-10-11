"""Stage 3: is the dose-dependence of model skill anything more than a measurability effect?

Sections 1 and 4 of `docs/dose_regime_findings.md` stratify by dose and by replicate-derived SNR
SEPARATELY, and observe that both orderings run the same way. That licenses the inference that the
dose effect is an SNR effect, but it does not demonstrate it: two marginal stratifications moving
together is exactly what a confound looks like. Stage 0 was run to license the matched comparison
(magnitude overlap 0.515 at the dose extremes, 10/10 bins usable) and this is that comparison.

Two readings, because binning and regression fail in different directions:

  regression  per-group improvement over the dose-mean baseline, regressed on log10 dose with and
              without SNR. If dose's coefficient collapses once SNR is in the model, the dose
              effect is mediated by measurability. Uses every group, so it is the powered test.
  crossed     attainable fraction in each (dose x SNR quartile) cell, pooled over seeds so the
              cells are populated. Readable, and shows whether the collapse is uniform.

A dose coefficient that SURVIVES conditioning on SNR would mean dose carries information beyond
measurability and that section 4 is overstated. That outcome is reportable, not avoidable.

Also scores every model under a calibrated metric that is not ours, to answer the obvious
objection that `attainable_fraction` is bespoke. The positive control is an interpolated duplicate:
one replicate shrunk per gene toward the baseline by its reliability. It predicts the OTHER
replicate, so control and target share no noise and the ceiling is reachable by construction.
"""
import json, sys
from dataclasses import replace
from pathlib import Path
import numpy as np, pandas as pd, torch
from cell2compound.measurability import (class_onehot, dynamic_range_fraction,
                                         interpolated_duplicate, noise_metrics, reliability_alpha)
from cell2compound.pipeline import (attainable_fraction, fit_predictions, load_response_data,
                                    select_genes)
from cell2compound.splits import compound_split

path = sys.argv[1] if len(sys.argv) > 1 else 'data/pseudobulk_A549_24h.npz'
label = sys.argv[2] if len(sys.argv) > 2 else 'A549_24h'
seeds = [42, 43, 44, 45, 46]
MODELS = ('dose_mean_response', 'tanimoto_kernel_ridge', 'ridge', 'mlp', 'tanimoto_knn')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
out = Path('results/dose_regime'); out.mkdir(parents=True, exist_ok=True)
data = load_response_data(path, 'data/raw/sciplex_compounds.csv')

# Split-independent SNR, identical definition to dose_regime_ablation.py so the strata match.
detection = data.detected.sum(0) / data.cells.sum()
snr_genes = np.flatnonzero(detection >= .05)
snr = noise_metrics(data.full_delta, data.rep_difference, snr_genes)['snr']
snr_edges = np.quantile(snr, [0, .25, .5, .75, 1.]); snr_edges[-1] += 1e-9

# rep1/rep2 are recoverable from the stored mean and difference, and are needed for a positive
# control whose noise is disjoint from the target it is scored against.
rep1 = data.full_delta + data.rep_difference / 2
rep2 = data.full_delta - data.rep_difference / 2

n_struct = int(data.groups.fp_row.max()) + 1
classes = pd.read_csv('results/stress_test/compound_classes.csv').set_index('compound').mechanism_class
group_class = data.groups.compound.map(classes).fillna('unannotated').to_numpy()
struct_class = np.array(['unannotated'] * n_struct, dtype=object)
for row, cls in zip(data.groups.fp_row.to_numpy(), group_class):
    struct_class[row] = cls
onehot = class_onehot(struct_class)


def arm_data(name):
    if name == 'full':
        return data
    new_fp = onehot[data.groups.fp_row.to_numpy()]
    return replace(data, fingerprints=new_fp, features=np.c_[new_fp, np.log10(data.doses)])


rows = []
for arm in ('full', 'class_onehot'):
    adata = arm_data(arm)
    for seed in seeds:
        splits = compound_split(data.canonical, seed=seed)
        train = np.flatnonzero(splits == 'train'); valid = np.flatnonzero(splits == 'validation')
        test = np.flatnonzero(splits == 'test')
        genes = select_genes(data, train, 1000)
        y = data.full_delta[:, genes]
        preds, _ = fit_predictions(adata, y, train, valid, test, seed, device, models=MODELS)

        # Reliability on TRAINING groups only, so the positive control cannot see held-out data.
        alpha = reliability_alpha(rep1[np.ix_(train, genes)], rep2[np.ix_(train, genes)])
        base_train_mean = {d: y[np.r_[train, valid]][data.doses[np.r_[train, valid]] == d].mean(0)
                           for d in np.unique(data.doses)}
        baseline_test = np.stack([base_train_mean[d] for d in data.doses[test]])
        # Target for the calibrated metric is rep1 alone; the control is built from rep2 alone.
        target1 = rep1[np.ix_(test, genes)]
        control = interpolated_duplicate(rep2[np.ix_(test, genes)], baseline_test, alpha)

        def rmse_rows(pred, target):
            return np.sqrt(((target - pred) ** 2).mean(1))

        per_group = {
            'floor': np.sqrt((data.rep_difference[np.ix_(test, genes)] ** 2).mean(1)) / 2,
            'base_rmse': rmse_rows(baseline_test, y[test]),
            'control_rmse_vs_rep1': rmse_rows(control, target1),
            'base_rmse_vs_rep1': rmse_rows(baseline_test, target1)}
        for model, pred in preds.items():
            per_group[f'rmse_{model}'] = rmse_rows(pred, y[test])
            per_group[f'rmse1_{model}'] = rmse_rows(pred, target1)

        quartile = np.digitize(snr[test], snr_edges[1:-1]) + 1
        for i, row in enumerate(test):
            rows.append({'arm': arm, 'seed': seed, 'group': int(row),
                         'compound': data.groups.compound.iloc[row], 'dose': float(data.doses[row]),
                         'log_dose': float(np.log10(data.doses[row])), 'snr': float(snr[row]),
                         'snr_quartile': int(quartile[i]), 'mechanism_class': group_class[row],
                         **{k: float(v[i]) for k, v in per_group.items()}})

frame = pd.DataFrame(rows)
frame.to_csv(out / f'stage3_raw_{label}.csv', index=False)

# --- primary: does dose predict skill once SNR is accounted for? ------------------------------
def ols(X, y):
    """Least squares with an intercept, returning coefficients for the given columns."""
    A = np.c_[np.ones(len(X)), X]
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    fitted = A @ beta
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1 - ((y - fitted) ** 2).sum() / ss_tot if ss_tot > 0 else np.nan
    return beta, float(r2)


def bootstrap_coef(X, y, index, rng, draws=2000):
    """Percentile CI for one coefficient, resampling groups."""
    vals = []
    for _ in range(draws):
        pick = rng.integers(0, len(X), len(X))
        try:
            beta, _ = ols(X[pick], y[pick])
        except np.linalg.LinAlgError:
            continue
        vals.append(beta[index])
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


MODEL = 'tanimoto_kernel_ridge'
summary = {'dataset': label, 'groups_scored': int(len(frame)), 'model': MODEL,
           'snr_quartile_edges': [float(e) for e in snr_edges]}
rng = np.random.default_rng(0)
for arm in ('full', 'class_onehot'):
    part = frame[frame.arm == arm]
    # Improvement over the dose-mean baseline, in RMSE units. Positive = model is better.
    gain = (part.base_rmse - part[f'rmse_{MODEL}']).to_numpy()
    dose_only, r2_dose = ols(part[['log_dose']].to_numpy(), gain)
    both, r2_both = ols(part[['log_dose', 'snr']].to_numpy(), gain)
    snr_only, r2_snr = ols(part[['snr']].to_numpy(), gain)
    ci_marginal = bootstrap_coef(part[['log_dose']].to_numpy(), gain, 1, rng)
    ci_conditional = bootstrap_coef(part[['log_dose', 'snr']].to_numpy(), gain, 1, rng)
    summary[arm] = {
        'dose_coefficient_marginal': float(dose_only[1]), 'dose_marginal_ci95': list(ci_marginal),
        'dose_coefficient_given_snr': float(both[1]), 'dose_given_snr_ci95': list(ci_conditional),
        'attenuation': float(1 - both[1] / dose_only[1]) if dose_only[1] else None,
        'snr_coefficient_given_dose': float(both[2]),
        'r2_dose_only': r2_dose, 'r2_snr_only': r2_snr, 'r2_both': r2_both}

# --- secondary: the crossed table -------------------------------------------------------------
cross = []
for arm in ('full', 'class_onehot'):
    for dose in sorted(frame.dose.unique()):
        for q in (1, 2, 3, 4):
            cell = frame[(frame.arm == arm) & (frame.dose == dose) & (frame.snr_quartile == q)]
            if len(cell) < 15:
                continue
            floor = float(np.sqrt((cell.floor ** 2).mean()))
            base = float(np.sqrt((cell.base_rmse ** 2).mean()))
            model = float(np.sqrt((cell[f'rmse_{MODEL}'] ** 2).mean()))
            cross.append({'arm': arm, 'dose_nM': dose, 'snr_quartile': q, 'groups': len(cell),
                          'snr_median': float(cell.snr.median()), 'floor': round(floor, 4),
                          'baseline_rmse': round(base, 4), 'model_rmse': round(model, 4),
                          'headroom': round(base - floor, 4),
                          'attainable': attainable_fraction(model, base, floor)})
crossed = pd.DataFrame(cross)
crossed.to_csv(out / f'stage3_crossed_{label}.csv', index=False)

# --- calibrated metric: Dynamic Range Fraction against an interpolated duplicate ---------------
drf = []
for arm in ('full', 'class_onehot'):
    for name, stratum in [('all', slice(None))] + [(f'snr_q{q}', q) for q in (1, 2, 3, 4)] + \
                         [(f'dose_{d:g}nM', d) for d in sorted(frame.dose.unique())]:
        part = frame[frame.arm == arm]
        if name.startswith('snr_q'):
            part = part[part.snr_quartile == stratum]
        elif name.startswith('dose_'):
            part = part[part.dose == stratum]
        if len(part) < 15:
            continue
        negative = float(np.sqrt((part.base_rmse_vs_rep1 ** 2).mean()))
        positive = float(np.sqrt((part.control_rmse_vs_rep1 ** 2).mean()))
        record = {'arm': arm, 'stratum': name, 'groups': len(part),
                  'negative_baseline_rmse': round(negative, 4),
                  'positive_control_rmse': round(positive, 4),
                  'dynamic_range': round(negative - positive, 4)}
        for model in MODELS:
            value = dynamic_range_fraction(
                float(np.sqrt((part[f'rmse1_{model}'] ** 2).mean())), negative, positive)
            record[f'drf_{model}'] = None if value is None else round(value, 4)
        drf.append(record)
calibrated = pd.DataFrame(drf)
calibrated.to_csv(out / f'stage3_calibrated_{label}.csv', index=False)

(out / f'stage3_{label}.json').write_text(json.dumps(summary, indent=1))
print(f'=== {label}: does dose survive conditioning on SNR? ({MODEL}) ===')
for arm in ('full', 'class_onehot'):
    s = summary[arm]
    print(f"  {arm:14s} dose alone {s['dose_coefficient_marginal']:+.5f} "
          f"CI [{s['dose_marginal_ci95'][0]:+.5f},{s['dose_marginal_ci95'][1]:+.5f}]  "
          f"| given snr {s['dose_coefficient_given_snr']:+.5f} "
          f"CI [{s['dose_given_snr_ci95'][0]:+.5f},{s['dose_given_snr_ci95'][1]:+.5f}]  "
          f"| attenuation {s['attenuation']:.1%}  R2 {s['r2_dose_only']:.3f}->{s['r2_both']:.3f}")
print()
print('=== attainable fraction, dose x snr quartile (arm=full) ===')
show = crossed[crossed.arm == 'full'].pivot_table(index='dose_nM', columns='snr_quartile',
                                                  values='attainable')
print(show.round(3).to_string())
print()
print('=== Dynamic Range Fraction vs interpolated duplicate (arm=full) ===')
cols = ['stratum', 'groups', 'dynamic_range'] + [f'drf_{m}' for m in MODELS]
print(calibrated[calibrated.arm == 'full'][cols].to_string(index=False))
