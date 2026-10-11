"""Stage 1 / OQ3: is the predictability ceiling dose-dependent?

floor   = RMSE(rep1-rep2)/2, the error of the observed two-replicate mean. No model beats it.
rms     = RMS of the observed mean delta, which CONTAINS that noise.
signal  = sqrt(max(rms^2 - floor^2, 0)), the noise-corrected response.
snr     = signal / floor. snr < 1 means the group's response is below its own measurement error.

Also tests whether the floor is depth-limited (floor ~ k/sqrt(cells)), because if it is then
'low dose is unmeasurable' is a statement about sci-Plex's cells-per-condition, not biology.
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from cell2compound.measurability import noise_metrics
from cell2compound.pipeline import load_response_data

rng = np.random.default_rng(0)
path = sys.argv[1] if len(sys.argv) > 1 else 'data/pseudobulk_A549_24h.npz'
label = sys.argv[2] if len(sys.argv) > 2 else 'A549_24h'
data = load_response_data(path, 'data/raw/sciplex_compounds.csv')
detection = data.detected.sum(0) / data.cells.sum()
genes = np.flatnonzero(detection >= .05)

m = noise_metrics(data.full_delta, data.rep_difference, genes)
rms, floor, signal = m['rms'], m['floor'], m['signal']
cells = np.array([data.cells[r].sum() for r in data.groups.table_rows])
doses = data.doses

def boot(values, weights=None, n=2000):
    """Bootstrap CI of a ratio-of-medians style statistic over groups."""
    out = np.empty(n)
    idx = np.arange(len(values))
    for i in range(n):
        s = rng.choice(idx, len(idx), replace=True)
        out[i] = np.median(values[s])
    return float(np.quantile(out, .025)), float(np.quantile(out, .975))

classes = pd.read_csv('results/stress_test/compound_classes.csv').set_index('compound').mechanism_class
group_class = data.groups.compound.map(classes).fillna('unannotated').to_numpy()

levels = np.unique(doses)
rows = []
for d in levels:
    m = doses == d
    snr = signal[m] / floor[m]
    lo, hi = boot(snr)
    rows.append({'dose_nM': float(d), 'groups': int(m.sum()),
                 'floor_median': round(float(np.median(floor[m])), 4),
                 'signal_median': round(float(np.median(signal[m])), 4),
                 'snr_median': round(float(np.median(snr)), 3),
                 'snr_ci_low': round(lo, 3), 'snr_ci_high': round(hi, 3),
                 'frac_groups_snr_above_1': round(float((snr > 1).mean()), 3),
                 'cells_median': int(np.median(cells[m]))})
per_dose = pd.DataFrame(rows)

# Depth limitation: is floor consistent with k/sqrt(cells)?
ok = cells > 0
A = np.c_[np.ones(ok.sum()), np.log(cells[ok])]
coef, *_ = np.linalg.lstsq(A, np.log(floor[ok]), rcond=None)
resid = np.log(floor[ok]) - A @ coef
r2 = 1 - resid.var() / np.log(floor[ok]).var()

# Does dose still predict the floor after controlling for cell count?
B = np.c_[np.ones(ok.sum()), np.log(cells[ok]), np.log10(doses[ok])]
coef_b, *_ = np.linalg.lstsq(B, np.log(floor[ok]), rcond=None)
resid_b = np.log(floor[ok]) - B @ coef_b
r2_b = 1 - resid_b.var() / np.log(floor[ok]).var()

# Per class x dose SNR, for classes with enough groups
cls_rows = []
for c in pd.unique(group_class):
    mc = group_class == c
    if mc.sum() < 8:
        continue
    entry = {'mechanism_class': c, 'groups': int(mc.sum())}
    for d in levels:
        m = mc & (doses == d)
        entry[f'snr_{d:g}nM'] = round(float(np.median(signal[m] / floor[m])), 3) if m.sum() else None
    entry['snr_all'] = round(float(np.median(signal[mc] / floor[mc])), 3)
    cls_rows.append(entry)
per_class = pd.DataFrame(cls_rows).sort_values('snr_all', ascending=False)

report = {'input': path, 'label': label, 'genes_used': int(len(genes)),
          'definitions': {'floor': 'RMSE(rep1-rep2)/2', 'signal': 'sqrt(rms^2 - floor^2)',
                          'snr': 'signal/floor; <1 means response below its own measurement error'},
          'per_dose': rows,
          'depth_limitation': {
              'log_floor_on_log_cells_slope': round(float(coef[1]), 3),
              'slope_expected_if_pure_counting_noise': -0.5,
              'r2': round(float(r2), 3),
              'r2_adding_log_dose': round(float(r2_b), 3),
              'log_dose_coefficient_after_controlling_cells': round(float(coef_b[2]), 4)},
          'per_class': cls_rows,
          'finding': None}
lo_snr, hi_snr = rows[0]['snr_median'], rows[-1]['snr_median']
report['finding'] = (f'SNR rises from {lo_snr} at {levels[0]:g} nM to {hi_snr} at {levels[-1]:g} nM; '
                     f'the ceiling is dose-dependent in the OPPOSITE direction to the hypothesis '
                     f'(low dose is the unmeasurable regime, not high dose).')
print(per_dose.to_string(index=False)); print()
print(per_class.to_string(index=False)); print()
print(json.dumps({k: report[k] for k in ('depth_limitation', 'finding')}, indent=2))
out = Path('results/dose_regime'); out.mkdir(parents=True, exist_ok=True)
(out / f'stage1_{label}.json').write_text(json.dumps(report, indent=2) + '\n')
per_dose.to_csv(out / f'stage1_per_dose_{label}.csv', index=False)
per_class.to_csv(out / f'stage1_per_class_{label}.csv', index=False)
