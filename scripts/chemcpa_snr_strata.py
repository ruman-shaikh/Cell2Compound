"""Does chemCPA's advantage survive signal-to-noise stratification?

scripts/chemcpa_compare.py scored chemCPA against our models pooled over all held-out groups and
found chemcpa 0.1016 vs our best 0.1070. But RMSE is magnitude-weighted and
docs/dose_regime_findings.md shows all demonstrable model skill sits in the top snr quartile, so a
pooled figure largely reports that quartile. This re-scores the identical predictions inside snr
strata and dose levels to find out where the margin actually comes from.

No retraining: it reuses results/chemcpa/ood_predictions.npz. The control reference, group mapping
and alignment check are the same as in chemcpa_compare.py, because a different reference would
silently shift every delta.
"""
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd, torch
from cell2compound.measurability import noise_metrics
from cell2compound.pipeline import (BASELINES, attainable_fraction, fit_predictions,
                                    load_response_data)
from cell2compound.splits import compound_split

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--predictions', type=Path, default=Path('results/chemcpa/ood_predictions.npz'))
parser.add_argument('--pseudobulk', type=Path, default=Path('data/pseudobulk_A549_24h.npz'))
parser.add_argument('--source', type=Path, default=Path('data/raw/sciplex_compounds.csv'))
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--tolerance', type=float, default=.02)
parser.add_argument('--out', type=Path, default=Path('results/chemcpa'))
args = parser.parse_args()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

bundle = np.load(args.predictions, allow_pickle=False)
chem_genes = bundle['genes'].astype(str)
categories = bundle['categories'].astype(str)
data = load_response_data(args.pseudobulk, args.source)
position = {g: i for i, g in enumerate(data.gene_names.astype(str))}
genes = np.array([position[g] for g in chem_genes])

ours = {f'A549_{c.strip().replace("+", "plus")}_{d:g}': i
        for i, (c, d) in enumerate(zip(data.groups.compound, data.doses))}
matched = [(i, ours[c]) for i, c in enumerate(categories) if c in ours]
chem_rows = np.array([m[0] for m in matched]); our_rows = np.array([m[1] for m in matched])
splits = compound_split(data.canonical, seed=args.seed)
keep = np.isin(our_rows, np.flatnonzero(splits == 'test'))
chem_rows, our_rows = chem_rows[keep], our_rows[keep]

# One pooled control reference for every model, as chemcpa_compare.py does.
bulk = np.load(args.pseudobulk)
bulk_table = pd.DataFrame(bulk['groups'], columns=bulk['keys'])
is_control = (bulk_table.perturbation == 'control').to_numpy()
control = (bulk['sums'][is_control].sum(0) / bulk['cells'][is_control].sum())[genes].astype(np.float64)

observed = data.full_delta[our_rows][:, genes]
drift = float(np.abs((bundle['observed'][chem_rows].astype(np.float64) - control) - observed).mean())
if drift > args.tolerance:
    raise SystemExit(f'alignment check failed: drift {drift:.4f} > {args.tolerance}; '
                     'the group mapping is wrong and no number below would mean anything')

# snr on a split-independent gene set, so strata are not a function of the split.
detection = data.detected.sum(0) / data.cells.sum()
snr_all = noise_metrics(data.full_delta, data.rep_difference, np.flatnonzero(detection >= .05))['snr']
snr = snr_all[our_rows]

y = data.full_delta[:, genes]
train = np.flatnonzero(splits == 'train'); valid = np.flatnonzero(splits == 'validation')
predictions, _ = fit_predictions(data, y, train, valid, our_rows, args.seed, device)
predictions['chemcpa'] = bundle['predicted'][chem_rows].astype(np.float64) - control

edges = np.quantile(snr_all, [0, .25, .5, .75, 1.]); edges[-1] += 1e-9
quart = np.digitize(snr, edges[1:-1])
strata = {'all': np.ones(len(snr), bool)}
for k in range(4):
    strata[f'snr_q{k + 1}'] = quart == k
for d in np.unique(data.doses[our_rows]):
    strata[f'dose_{d:g}nM'] = data.doses[our_rows] == d

rows = []
for name, mask in strata.items():
    if mask.sum() < 10:
        continue
    floor = float(np.sqrt((data.rep_difference[our_rows][mask][:, genes] ** 2).mean()) / 2)
    rmses = {m: float(np.sqrt(((observed[mask] - p[mask]) ** 2).mean())) for m, p in predictions.items()}
    base = min(BASELINES, key=lambda b: rmses[b])
    headroom = rmses[base] - floor
    entry = {'stratum': name, 'groups': int(mask.sum()), 'floor': round(floor, 4),
             'best_baseline': base, 'headroom': round(headroom, 4),
             'measurable': bool(headroom > .05 * floor),
             'snr_median': round(float(np.median(snr[mask])), 3)}
    for m in ('chemcpa', 'tanimoto_kernel_ridge', 'tanimoto_knn', 'dose_mean_response'):
        entry[f'rmse_{m}'] = round(rmses[m], 4)
        a = attainable_fraction(rmses[m], rmses[base], floor)
        entry[f'attain_{m}'] = None if a is None else round(a, 3)
    # Comparator is PRE-SPECIFIED as tanimoto_kernel_ridge, our best pooled model. Taking the
    # minimum over all seven of our models inside each ~36-group stratum biases the comparison in
    # our favour: with that many candidates and that few groups the minimum is selected on noise.
    # Reported alongside purely to show the size of that selection effect.
    entry['chemcpa_minus_kernel_ridge'] = round(
        rmses['chemcpa'] - rmses['tanimoto_kernel_ridge'], 5)
    entry['selection_effect_of_min_over_our_7'] = round(
        rmses['tanimoto_kernel_ridge'] - min(rmses[m] for m in rmses if m != 'chemcpa'), 5)
    rows.append(entry)

frame = pd.DataFrame(rows)
args.out.mkdir(parents=True, exist_ok=True)
frame.to_csv(args.out / 'snr_strata.csv', index=False)
(args.out / 'snr_strata.json').write_text(json.dumps(
    {'alignment_drift': round(drift, 5), 'tolerance': args.tolerance,
     'groups': int(len(our_rows)), 'genes': int(len(genes)), 'split_seed': args.seed,
     'strata': rows,
     'note': ('single chemCPA seed; its pooled margin over our best was 0.0055 against a 0.0039 '
              'five-seed sd, so pooled significance was never established'),
     'comparator': ('tanimoto_kernel_ridge, pre-specified as our best pooled model; the minimum '
                    'over all seven of our models is biased downward inside small strata')}, indent=2) + '\n')
print(f'alignment drift {drift:.5f} (tolerance {args.tolerance})  groups {len(our_rows)}\n')
cols = ['stratum', 'groups', 'snr_median', 'floor', 'headroom', 'measurable',
        'rmse_chemcpa', 'rmse_tanimoto_kernel_ridge', 'rmse_dose_mean_response',
        'attain_chemcpa', 'attain_tanimoto_kernel_ridge', 'chemcpa_minus_kernel_ridge',
        'selection_effect_of_min_over_our_7']
print(frame[cols].to_string(index=False))
