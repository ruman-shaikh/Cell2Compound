"""Score chemCPA against our models on identical targets, genes, groups and metric.

The two were not comparable as reported: chemCPA gives per-condition R2 over single cells, our
pipeline gives RMSE on control-relative pseudobulk group means. This puts both on our footing:

  * same genes      - chemCPA's 2000-gene set, which (after fixing the gene selection) is wholly
                      contained in our pseudobulk gene space
  * same groups      - the held-out (compound, dose) conditions of the same seeded compound split
  * same target      - observed group mean minus ONE pooled control reference, for both models
  * same metric      - cell2compound RMSE, top-50 DE RMSE, and the replicate-noise floor

chemCPA predicts absolute expression, so its predictions are converted to deltas against the same
pooled control reference rather than the split-specific one its own R2 uses. An alignment check
asserts that chemCPA's observed group means agree with ours for the matched groups; if they do not,
the group mapping is wrong and no number below would mean anything.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from cell2compound.expression import response_metrics
from cell2compound.pipeline import BASELINES, attainable_fraction, fit_predictions, load_response_data, replicate_floor
from cell2compound.splits import compound_split

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--predictions', type=Path, default=Path('results/chemcpa/ood_predictions.npz'))
parser.add_argument('--pseudobulk', type=Path, default=Path('data/pseudobulk_A549_24h.npz'))
parser.add_argument('--source', type=Path, default=Path('data/raw/sciplex_compounds.csv'))
parser.add_argument('--seed', type=int, default=42, help='must match the split the chemCPA input was built with')
parser.add_argument('--top-k', type=int, default=50)
parser.add_argument('--tolerance', type=float, default=.02, help='allowed |difference| on matched observed means')
parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
parser.add_argument('--out', type=Path, default=Path('results/chemcpa'))
args = parser.parse_args()
if args.device == 'cuda' and not torch.cuda.is_available():
    args.device = 'cpu'
args.out.mkdir(parents=True, exist_ok=True)

bundle = np.load(args.predictions, allow_pickle=False)
chem_genes = bundle['genes'].astype(str)
categories = bundle['categories'].astype(str)

data = load_response_data(args.pseudobulk, args.source)
gene_position = {g: i for i, g in enumerate(data.gene_names.astype(str))}
missing = [g for g in chem_genes if g not in gene_position]
if missing:
    raise ValueError(f'{len(missing)} chemCPA genes absent from the pseudobulk space, e.g. {missing[:5]}')
genes = np.array([gene_position[g] for g in chem_genes])

# chemCPA condition names had '+' substituted, so undo it to match our compound names.
def to_chemcpa_name(name):
    return name.strip().replace('+', 'plus')


# Their category is celltype_condition_dose with the dose formatted by %g.
ours = {f'A549_{to_chemcpa_name(c)}_{d:g}': i for i, (c, d) in
        enumerate(zip(data.groups.compound, data.doses))}
matched = [(i, ours[c]) for i, c in enumerate(categories) if c in ours]
if len(matched) < 10:
    raise ValueError(f'only {len(matched)} chemCPA categories matched our groups; mapping is wrong')
chem_rows = np.array([m[0] for m in matched])
our_rows = np.array([m[1] for m in matched])

splits = compound_split(data.canonical, seed=args.seed)
held_out = np.flatnonzero(splits == 'test')
keep = np.isin(our_rows, held_out)
if not keep.all():
    print(f'{int((~keep).sum())} matched categories are not in our held-out partition; dropping them', flush=True)
chem_rows, our_rows = chem_rows[keep], our_rows[keep]

# One pooled control reference for both models, as our pipeline uses. It must be computed here
# rather than taken from the prediction bundle: that bundle's reference is the mean of the 595
# TEST-split control cells chemCPA translates from, whereas our observed deltas are relative to
# all 3,726 controls pooled. Mixing the two leaves a systematic offset (it failed this script's
# own alignment check at 0.031 against a 0.02 tolerance) and would corrupt every row.
bulk = np.load(args.pseudobulk)
bulk_table = pd.DataFrame(bulk['groups'], columns=bulk['keys'])
bulk_control = (bulk_table.perturbation == 'control').to_numpy()
control = (bulk['sums'][bulk_control].sum(0) / bulk['cells'][bulk_control].sum())[genes].astype(np.float64)
split_reference = bundle['control_reference'].astype(np.float64)
print(f'pooled control reference from {int(bulk["cells"][bulk_control].sum())} cells; '
      f'the bundle\'s split reference differs by '
      f'{np.abs(control - split_reference).mean():.4f} per gene on average', flush=True)
observed_ours = data.full_delta[our_rows][:, genes]
observed_chem = bundle['observed'][chem_rows].astype(np.float64)
predicted_chem = bundle['predicted'][chem_rows].astype(np.float64)

# Alignment check: chemCPA's observed group means, re-expressed as deltas against its own control
# reference, must agree with ours. Both come from the same cells, so disagreement means the
# group mapping is wrong rather than the models differing.
observed_chem_delta = observed_chem - control
drift = np.abs(observed_chem_delta - observed_ours).mean()
aligned = bool(drift <= args.tolerance)
print(f'alignment: {len(chem_rows)} groups, mean |observed difference| {drift:.4f} '
      f'(tolerance {args.tolerance})', flush=True)

y = data.full_delta[:, genes]
train = np.flatnonzero(splits == 'train'); valid = np.flatnonzero(splits == 'validation')
predictions, chosen = fit_predictions(data, y, train, valid, our_rows, args.seed, args.device)
predictions['chemcpa'] = predicted_chem - control

floor = replicate_floor(data, our_rows, genes)
scored = {name: response_metrics(observed_ours, pred, args.top_k) for name, pred in predictions.items()}
best_baseline = min(BASELINES, key=lambda b: scored[b]['rmse'])
table = []
for name, metrics in sorted(scored.items(), key=lambda kv: kv[1]['rmse']):
    table.append({'model': name, 'rmse': round(metrics['rmse'], 4),
                  f'rmse_top{args.top_k}_de': round(metrics[f'rmse_top{args.top_k}_de'], 4),
                  f'pearson_top{args.top_k}_de': round(metrics[f'pearson_top{args.top_k}_de'], 4),
                  'attainable_fraction': (round(v, 3) if (v := attainable_fraction(
                      metrics['rmse'], scored[best_baseline]['rmse'], floor)) is not None else None)})

report = {'predictions': str(args.predictions), 'pseudobulk': str(args.pseudobulk),
          'split_seed': args.seed, 'genes': int(len(genes)),
          'held_out_groups_compared': int(len(our_rows)),
          'held_out_compounds': int(data.groups.compound.iloc[our_rows].nunique()),
          'replicate_noise_floor_rmse': round(floor, 4),
          'best_baseline': best_baseline,
          'alignment_check': {'mean_abs_observed_difference': round(float(drift), 5),
                              'tolerance': args.tolerance, 'passed': aligned},
          'control_reference': {'source': 'all control cells pooled, from the pseudobulk archive',
                                'cells': int(bulk['cells'][bulk_control].sum()),
                                'mean_abs_difference_to_chemcpa_split_reference':
                                    round(float(np.abs(control - split_reference).mean()), 5)},
          'results': table, 'our_hyperparameters': chosen,
          'comparability': [
              'Same cells, genes, groups, control reference and metric for every row.',
              'chemCPA predicts absolute expression; its predictions are converted to deltas '
              'against the same pooled control reference as every other model here.',
              'This is NOT chemCPA\'s own reported metric (per-condition R2 on single cells); it is '
              'chemCPA scored on our pseudobulk task, which is what makes the rows comparable.',
              'chemCPA was trained once at its default 201 epochs with no hyperparameter search, '
              'while our models select hyperparameters on the validation split, so this favours ours.'],
          }
(args.out / 'comparison.json').write_text(json.dumps(report, indent=2) + '\n')
pd.DataFrame(table).to_csv(args.out / 'comparison.csv', index=False)
print(json.dumps({k: report[k] for k in ('held_out_groups_compared', 'replicate_noise_floor_rmse',
                                         'alignment_check', 'results')}, indent=2))
if not aligned:
    raise SystemExit('alignment check failed: chemCPA and our observed group means disagree; '
                     'fix the group mapping before interpreting these numbers')
