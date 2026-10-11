"""Aggregate the chemCPA comparison over split seeds, which is what settles significance.

The single-seed result (chemCPA 0.1016 against our 0.1071, a 0.0055 margin) was never testable:
our own models' seed-to-seed standard deviation is 0.0039, so one chemCPA run could not say whether
0.0055 was a real advantage or a lucky split. This pairs the two within each seed — same held-out
compounds, same genes, same control reference — and reports the paired margin.

Pairing matters more than it might seem. Across seeds the held-out compound sets overlap only
slightly (Jaccard 0.07-0.19 for these five), so unpaired means would be dominated by which
compounds each split happened to hold out, not by which model is better.

Reads whatever `results/chemcpa/comparison_s<seed>.json` files exist, so it can be run before
every seed has finished and will say how many it used.
"""
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--results', type=Path, default=Path('results/chemcpa'))
parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44, 45, 46])
parser.add_argument('--comparator', default='tanimoto_kernel_ridge',
                    help='pre-specified; never the per-seed minimum, which is selected on noise')
args = parser.parse_args()

rows, strata_rows, missing = [], [], []
for seed in args.seeds:
    # Seed 42 was scored before this script existed and kept the unsuffixed names.
    candidates = [args.results / f'comparison_s{seed}.json', args.results / 'comparison.json']
    path = next((p for p in candidates if p.exists() and
                 json.loads(p.read_text()).get('split_seed') == seed), None)
    if path is None:
        missing.append(seed); continue
    report = json.loads(path.read_text())
    by_model = {r['model']: r for r in report['results']}
    if 'chemcpa' not in by_model or args.comparator not in by_model:
        missing.append(seed); continue
    chem, ours = by_model['chemcpa'], by_model[args.comparator]
    best = min((r for r in report['results'] if r['model'] != 'chemcpa'), key=lambda r: r['rmse'])
    rows.append({'seed': seed, 'groups': report['held_out_groups_compared'],
                 'floor': report['replicate_noise_floor_rmse'],
                 'alignment_drift': report['alignment_check']['mean_abs_observed_difference'],
                 'alignment_passed': report['alignment_check']['passed'],
                 'chemcpa_rmse': chem['rmse'], 'ours_rmse': ours['rmse'],
                 'margin': chem['rmse'] - ours['rmse'],
                 'best_of_ours_model': best['model'], 'best_of_ours_rmse': best['rmse'],
                 'selection_effect': ours['rmse'] - best['rmse'],
                 'chemcpa_top50_de': chem.get('rmse_top50_de'),
                 'ours_top50_de': ours.get('rmse_top50_de')})
    # Seed 42 was scored before per-seed names existed, so fall back to the unsuffixed file.
    strata = next((p for p in (args.results / f'snr_strata_s{seed}.csv',
                               args.results / 'snr_strata.csv') if p.exists()), None)
    if strata is not None and (seed == 42 or strata.name.endswith(f's{seed}.csv')):
        s = pd.read_csv(strata)
        s['seed'] = seed
        strata_rows.append(s)

if not rows:
    raise SystemExit('no per-seed comparisons found; run chemcpa_predict + chemcpa_compare first')

frame = pd.DataFrame(rows).sort_values('seed')
frame.to_csv(args.results / 'seed_comparison.csv', index=False)
margin = frame.margin.to_numpy()

# Paired, so the test statistic is the mean margin over its own standard error.
mean = float(margin.mean()); sd = float(margin.std(ddof=1)) if len(margin) > 1 else float('nan')
se = sd / np.sqrt(len(margin)) if len(margin) > 1 else float('nan')
rng = np.random.default_rng(0)
boot = [rng.choice(margin, len(margin), replace=True).mean() for _ in range(10000)] \
    if len(margin) > 1 else []
summary = {'comparator': args.comparator, 'seeds_used': frame.seed.tolist(), 'seeds_missing': missing,
           'chemcpa_rmse_mean': round(float(frame.chemcpa_rmse.mean()), 5),
           'ours_rmse_mean': round(float(frame.ours_rmse.mean()), 5),
           'paired_margin_mean': round(mean, 5),
           'paired_margin_sd': None if np.isnan(sd) else round(sd, 5),
           'paired_margin_se': None if np.isnan(se) else round(se, 5),
           'margin_in_se_units': None if not se or np.isnan(se) else round(mean / se, 2),
           'margin_ci95_bootstrap': [round(float(np.percentile(boot, 2.5)), 5),
                                     round(float(np.percentile(boot, 97.5)), 5)] if boot else None,
           'chemcpa_wins_seeds': int((margin < 0).sum()), 'our_wins_seeds': int((margin > 0).sum()),
           'selection_effect_mean': round(float(frame.selection_effect.mean()), 5),
           'alignment_all_passed': bool(frame.alignment_passed.all())}
if boot:
    lo, hi = summary['margin_ci95_bootstrap']
    summary['significant_at_95'] = bool(hi < 0 or lo > 0)
(args.results / 'seed_comparison.json').write_text(json.dumps(summary, indent=1))

print('=== chemCPA vs ours, paired within split seed ===')
print(frame[['seed', 'groups', 'chemcpa_rmse', 'ours_rmse', 'margin', 'best_of_ours_model',
             'selection_effect', 'alignment_drift']].to_string(index=False))
print()
print(json.dumps(summary, indent=1))
print()
if summary.get('significant_at_95') is True:
    side = 'chemCPA is ahead' if mean < 0 else 'our model is ahead'
    print(f'VERDICT: {side}; the paired 95% bootstrap CI excludes zero.')
elif len(margin) > 1:
    print('VERDICT: not significant. The paired 95% bootstrap CI includes zero, so the '
          'single-seed margin does not survive seed variation.')
else:
    print('VERDICT: one seed only; no significance statement is possible.')

if strata_rows:
    allstrata = pd.concat(strata_rows, ignore_index=True)
    allstrata.to_csv(args.results / 'snr_strata_all_seeds.csv', index=False)
    col = next((c for c in allstrata.columns if 'advantage' in c or 'chemcpa_minus' in c), None)
    if col:
        piv = allstrata.pivot_table(index='stratum', columns='seed', values=col)
        print()
        print(f'=== {col} by stratum and seed ===')
        print(piv.round(5).to_string())
        print()
        print('mean over seeds:')
        print(piv.mean(axis=1).round(5).to_string())
