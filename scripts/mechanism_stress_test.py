"""Does the response model generalise, or does it recognise a few mechanism classes?

In this library the top responders are all HDAC inhibitors and ~91 % of compounds move the
transcriptome by less than twice their own replicate noise. A Tanimoto kernel is very good at
spotting "another hydroxamate", so the headline RMSE gain could come almost entirely from a
handful of large, chemically obvious signatures. Three evaluations separate those cases:

  reference  - the standard compound-held-out evaluation, reproduced and checked against the
               recorded headline result. If this does not match, nothing below is comparable.
  loco       - leave-one-mechanism-class-out: every compound of a class is held out, so the
               model must predict a response class it has never seen.
  strata     - the reference split, split by observed response over that group's own replicate
               noise, reporting how much of the attainable improvement is captured in each
               stratum. Tells us whether anything is learned for weak responders.
  no_hdac    - the standard evaluation with HDAC inhibitors removed entirely.

Every arm uses the identical pipeline and models as the headline result (cell2compound.pipeline).
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from cell2compound.expression import response_metrics
from cell2compound.mechanism import (assign_classes, class_folds, extract_annotations,
                                     hdac_compounds, magnitude_strata)
from cell2compound.pipeline import (BASELINES, attainable_fraction, fit_predictions,
                                    load_response_data, replicate_floor, select_genes)
from cell2compound.splits import compound_split

LEARNED = ('tanimoto_knn', 'ridge', 'tanimoto_kernel_ridge', 'mlp')
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('input', type=Path, help='pseudobulk .npz from scripts/pseudobulk.py')
parser.add_argument('--raw-h5ad', type=Path, default=Path('data/raw/sciplex3.h5ad'),
                    help='only read once, to cache per-compound mechanism annotations')
parser.add_argument('--annotations', type=Path, default=Path('data/compound_annotations.csv'))
parser.add_argument('--class-source', default='target', choices=['target', 'pathway_level_1', 'pathway_level_2', 'chembl'])
parser.add_argument('--headline', type=Path, default=Path('results/expression/metrics.json'))
parser.add_argument('--tolerance', type=float, default=5e-3, help='allowed RMSE drift vs the headline result')
parser.add_argument('--source', type=Path, default=Path('data/raw/sciplex_compounds.csv'))
parser.add_argument('--mechanisms', type=Path, default=Path('results/targets/mechanisms.csv'))
parser.add_argument('--genes', type=int, default=1000)
parser.add_argument('--min-cells', type=int, default=10)
parser.add_argument('--min-detection', type=float, default=.05)
parser.add_argument('--top-k', type=int, default=50)
parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44, 45, 46])
parser.add_argument('--arms', nargs='+', default=['reference', 'strata', 'loco', 'no_hdac'])
parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
parser.add_argument('--out', type=Path, default=Path('results/stress_test'))
args = parser.parse_args()
if args.device == 'cuda' and not torch.cuda.is_available():
    raise RuntimeError('CUDA requested but unavailable; run outside sandbox or pass --device cpu')
args.out.mkdir(parents=True, exist_ok=True)

data = load_response_data(args.input, args.source, min_cells=args.min_cells)
annotations = extract_annotations(args.raw_h5ad, cache=args.annotations)
compounds = (data.groups[['compound']].drop_duplicates()
             .merge(data.mapping[['observed_name', 'parent_smiles', 'source_name']],
                    left_on='compound', right_on='observed_name', how='left'))
mechanisms = pd.read_csv(args.mechanisms) if args.mechanisms.exists() else None
compounds = assign_classes(compounds, annotations, source=args.class_source, mechanisms=mechanisms)
class_of = dict(zip(compounds.compound, compounds.mechanism_class))
hdac = hdac_compounds(compounds, annotations)
group_class = data.groups.compound.map(class_of).to_numpy()
group_is_hdac = data.groups.compound.isin(hdac).to_numpy()


def evaluate(subset, train, valid, test, seed, label):
    """Fit every model on one split and score it against the replicate noise floor."""
    genes = select_genes(subset, train, args.genes, args.min_detection)
    y = subset.full_delta[:, genes]
    predictions, chosen = fit_predictions(subset, y, train, valid, test, seed, args.device)
    floor = replicate_floor(subset, test, genes)
    scored = {name: response_metrics(y[test], pred, args.top_k) for name, pred in predictions.items()}
    best_baseline = min(BASELINES, key=lambda b: scored[b]['rmse'])
    row = {'arm': label, 'seed': seed, 'test_groups': len(test), 'train_groups': len(train),
           'test_compounds': int(subset.groups.compound.iloc[test].nunique()),
           'genes': len(genes), 'replicate_floor_rmse': floor,
           'best_baseline': best_baseline, 'best_baseline_rmse': scored[best_baseline]['rmse'],
           'hyperparameters': chosen}
    for name, metrics in scored.items():
        row[f'{name}_rmse'] = metrics['rmse']
        row[f'{name}_rmse_top{args.top_k}_de'] = metrics[f'rmse_top{args.top_k}_de']
        row[f'{name}_pearson_top{args.top_k}_de'] = metrics[f'pearson_top{args.top_k}_de']
    for name in LEARNED:
        row[f'{name}_attainable_fraction'] = attainable_fraction(
            scored[name]['rmse'], scored[best_baseline]['rmse'], floor)
    return row, y, predictions, genes, floor


rows, strata_rows = [], []
reference_cache = {}
if 'reference' in args.arms or 'strata' in args.arms:
    for seed in args.seeds:
        splits = compound_split(data.canonical, seed=seed)
        parts = [np.flatnonzero(splits == k) for k in ('train', 'validation', 'test')]
        row, y, predictions, genes, floor = evaluate(data, *parts, seed, 'reference')
        rows.append(row); reference_cache[seed] = (parts, y, predictions, genes)
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v)
                          for k, v in row.items() if k.endswith('rmse') or k in ('arm', 'seed')}), flush=True)

if 'strata' in args.arms:
    # Per stratum: is there any attainable signal, and does a model capture it?
    for seed, (parts, y, predictions, genes) in reference_cache.items():
        test = parts[2]
        response = np.sqrt(np.mean(data.full_delta[test][:, genes] ** 2, axis=1))
        noise = np.sqrt(np.mean(data.rep_difference[test][:, genes] ** 2, axis=1))
        labels, ratio, names = magnitude_strata(response, noise)
        for name in names:
            rows_in = np.flatnonzero(labels == name)
            if len(rows_in) < 5:
                continue
            floor = replicate_floor(data, test[rows_in], genes)
            scored = {model: float(np.sqrt(np.mean((y[test][rows_in] - pred[rows_in]) ** 2)))
                      for model, pred in predictions.items()}
            best_baseline = min(BASELINES, key=lambda b: scored[b])
            entry = {'seed': seed, 'stratum': name, 'groups': len(rows_in),
                     'median_response_over_noise': float(np.nanmedian(ratio[rows_in])),
                     'replicate_floor_rmse': floor, 'best_baseline': best_baseline,
                     'best_baseline_rmse': scored[best_baseline],
                     'hdac_fraction': float(group_is_hdac[test[rows_in]].mean())}
            for model, value in scored.items():
                entry[f'{model}_rmse'] = value
            for model in LEARNED:
                entry[f'{model}_attainable_fraction'] = attainable_fraction(
                    scored[model], scored[best_baseline], floor)
            strata_rows.append(entry)
        print('strata seed', seed, {n: int((labels == n).sum()) for n in names}, flush=True)

if 'loco' in args.arms:
    folds = class_folds(compounds)
    print(f'{len(folds)} mechanism classes usable as folds:',
          {k: len(v) for k, v in sorted(folds.items(), key=lambda kv: -len(kv[1]))}, flush=True)
    for label, members in folds.items():
        test = np.flatnonzero(np.isin(data.groups.compound.to_numpy(), members))
        rest = np.flatnonzero(~np.isin(data.groups.compound.to_numpy(), members))
        # Validation comes from the remaining classes, grouped by structure, never from the held-out class.
        inner = compound_split(data.canonical[rest], seed=0)
        train = rest[inner != 'test']; valid = rest[inner == 'test']
        row, *_ = evaluate(data, train, valid, test, 0, 'loco')
        row['held_out_class'] = label
        row['held_out_is_hdac'] = bool(group_is_hdac[test].mean() > .5)
        rows.append(row)
        print('loco', label, 'n_test', len(test),
              {m: round(row[f'{m}_rmse'], 4) for m in ('dose_mean_response', 'tanimoto_kernel_ridge')}, flush=True)

if 'no_hdac' in args.arms:
    keep = ~group_is_hdac
    subset = data.subset(keep)
    print(f'no_hdac: dropping {int(group_is_hdac.sum())} groups from '
          f'{len(hdac)} HDAC compounds, {len(subset.groups)} groups remain', flush=True)
    for seed in args.seeds:
        splits = compound_split(subset.canonical, seed=seed)
        parts = [np.flatnonzero(splits == k) for k in ('train', 'validation', 'test')]
        row, *_ = evaluate(subset, *parts, seed, 'no_hdac')
        rows.append(row)
        print('no_hdac', seed, {m: round(row[f'{m}_rmse'], 4)
                                for m in ('dose_mean_response', 'tanimoto_kernel_ridge')}, flush=True)

frame = pd.DataFrame(rows)
frame.drop(columns=['hyperparameters']).to_csv(args.out / 'arms.csv', index=False)
if strata_rows:
    pd.DataFrame(strata_rows).to_csv(args.out / 'strata.csv', index=False)
compounds.to_csv(args.out / 'compound_classes.csv', index=False)


def summarise(part, keys=('tanimoto_kernel_ridge', 'ridge', 'mlp', 'tanimoto_knn', 'dose_mean_response')):
    out = {'n': int(len(part)), 'replicate_floor_rmse': round(float(part.replicate_floor_rmse.mean()), 4),
           'best_baseline_rmse': round(float(part.best_baseline_rmse.mean()), 4)}
    for key in keys:
        out[f'{key}_rmse'] = round(float(part[f'{key}_rmse'].mean()), 4)
    for key in LEARNED:
        values = part[f'{key}_attainable_fraction'].dropna()
        # Median and win rate are robust; the mean is distorted by folds with little headroom.
        out[f'{key}_attainable_fraction_median'] = round(float(values.median()), 3) if len(values) else None
        out[f'{key}_attainable_fraction_mean'] = round(float(values.mean()), 3) if len(values) else None
        out[f'{key}_interpretable_folds'] = int(len(values))
        beats = (part[f'{key}_rmse'] < part.best_baseline_rmse)
        out[f'{key}_beats_baseline'] = f'{int(beats.sum())}/{int(len(part))}'
    return out


# Self-consistency: the reference arm must reproduce the recorded headline numbers, otherwise the
# pipeline has drifted and no comparison below means anything.
consistency = {'checked': False}
if 'reference' in args.arms and args.headline.exists():
    headline = json.loads(args.headline.read_text()).get('summary_over_seeds', {})
    reference = frame[frame.arm == 'reference']
    drift = {model: round(float(reference[f'{model}_rmse'].mean() - headline[model]['rmse']['mean']), 5)
             for model in headline if f'{model}_rmse' in reference}
    worst = max(abs(v) for v in drift.values()) if drift else 0.
    consistency = {'checked': True, 'headline': str(args.headline), 'rmse_drift_vs_headline': drift,
                   'max_abs_drift': worst, 'tolerance': args.tolerance, 'within_tolerance': worst <= args.tolerance}

report = {'input': str(args.input), 'class_source': args.class_source, 'seeds': args.seeds,
          'groups': int(len(data.groups)), 'compounds': int(data.groups.compound.nunique()),
          'hdac_compounds': sorted(hdac), 'classes_assigned': int(compounds.mechanism_class.notna().sum()),
          'self_consistency': consistency,
          'question': 'Does a structure-based response model generalise beyond mechanism classes it has seen?',
          'arms': {arm: summarise(part) for arm, part in frame.groupby('arm')},
          'loco_by_class': (frame[frame.arm == 'loco']
                            .set_index('held_out_class')[['test_groups', 'best_baseline_rmse',
                                                          'tanimoto_kernel_ridge_rmse',
                                                          'tanimoto_kernel_ridge_attainable_fraction',
                                                          'held_out_is_hdac']]
                            .round(4).to_dict('index') if 'loco' in frame.arm.values else {}),
          'strata': ({name: summarise(part) | {'median_response_over_noise': round(float(part.median_response_over_noise.mean()), 2),
                                               'hdac_fraction': round(float(part.hdac_fraction.mean()), 3)}
                      for name, part in pd.DataFrame(strata_rows).groupby('stratum')} if strata_rows else {}),
          'interpretation_guide': [
              'attainable_fraction is the share of the baseline-to-noise-floor gap a model closed; '
              '0 means no better than the best non-learned baseline, 1 means it reached the floor.',
              'A model that only recognises mechanism classes should score well on the reference arm '
              'and near 0 on leave-one-class-out, especially for the HDAC fold.',
              'A stratum whose baseline already sits at its replicate floor has no attainable signal, '
              'so a fraction near 0 there is expected rather than a model failure.'],
          'limitations': ['Classes come from sci-Plex/ChEMBL annotations, not from the expression data, but '
                          'they are coarse and some compounds are polypharmacological.',
                          'Leave-one-class-out changes the training set size between folds.',
                          'Strata are computed on the test groups of each reference split, so stratum '
                          'sizes vary by seed.']}
(args.out / 'summary.json').write_text(json.dumps(report, indent=2, default=str) + '\n')
print(json.dumps({k: report[k] for k in ('self_consistency', 'arms', 'strata')}, indent=2, default=str))
if consistency['checked'] and not consistency['within_tolerance']:
    raise SystemExit(f"Reference arm drifted from the headline result by {consistency['max_abs_drift']}; "
                     'fix the pipeline before interpreting the stress test')
