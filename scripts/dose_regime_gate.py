"""Stage 0 feasibility gate: is response magnitude separable from dose?

If the magnitude distributions at different doses do not overlap, then conditioning on
magnitude (OQ2 / rival R2) is impossible by stratification - the same collinearity that
killed the magnitude/direction idea. Also logs the R3 (cytotoxicity) and R5 (detection)
covariates per dose.

Gene set is chosen by detection alone over ALL rows, so magnitude here is split-independent.
"""
import json, sys
import numpy as np, pandas as pd
from cell2compound.measurability import noise_metrics, overlap_coefficient
from cell2compound.pipeline import load_response_data

path = sys.argv[1] if len(sys.argv) > 1 else 'data/pseudobulk_A549_24h.npz'
label = sys.argv[2] if len(sys.argv) > 2 else 'A549_24h'
data = load_response_data(path, 'data/raw/sciplex_compounds.csv')

# split-independent gene set: detected in >=5% of all cells
detection = data.detected.sum(0) / data.cells.sum()
genes = np.flatnonzero(detection >= .05)
m = noise_metrics(data.full_delta, data.rep_difference, genes)
rms, floor = m['rms'], m['floor']
cells = np.array([data.cells[r].sum() for r in data.groups.table_rows])
doses = data.doses
log_rms = np.log10(np.maximum(rms, 1e-12))

edges = np.histogram_bin_edges(log_rms, bins=30)
levels = np.unique(doses)
rows = []
for d in levels:
    m = doses == d
    rows.append({'dose_nM': float(d), 'groups': int(m.sum()),
                 'rms_median': round(float(np.median(rms[m])), 4),
                 'rms_q25': round(float(np.quantile(rms[m], .25)), 4),
                 'rms_q75': round(float(np.quantile(rms[m], .75)), 4),
                 'replicate_floor_median': round(float(np.median(floor[m])), 4),
                 'floor_over_signal': round(float(np.median(floor[m]) / np.median(rms[m])), 3),
                 'cells_median': int(np.median(cells[m])),
                 'detection_mean': round(float((data.detected[np.concatenate(
                     data.groups.table_rows[m].to_list())].sum(0) /
                     data.cells[np.concatenate(data.groups.table_rows[m].to_list())].sum()).mean()), 4)})
table = pd.DataFrame(rows)

pairs = []
for i, a in enumerate(levels):
    for b in levels[i + 1:]:
        pairs.append({'dose_a': float(a), 'dose_b': float(b),
                      'overlap_coefficient': round(overlap_coefficient(
                          log_rms[doses == a], log_rms[doses == b], edges), 3)})

# Can a magnitude-matched design be built? Bin magnitude into deciles; a bin is usable if
# at least two doses contribute >=10 groups each.
qs = np.quantile(log_rms, np.linspace(0, 1, 11)); qs[-1] += 1e-9
binned = np.digitize(log_rms, qs[1:-1])
usable, counts = 0, []
for b in range(10):
    per_dose = [(doses[(binned == b)] == d).sum() for d in levels]
    counts.append({'bin': b, 'per_dose': [int(x) for x in per_dose]})
    if sum(1 for x in per_dose if x >= 10) >= 2:
        usable += 1
matched_groups = int(sum(sum(x for x in c['per_dose'] if x >= 10)
                         for c in counts if sum(1 for x in c['per_dose'] if x >= 10) >= 2))

extreme = [p for p in pairs if p['dose_a'] == levels[0] and p['dose_b'] == levels[-1]][0]
report = {'input': path, 'label': label, 'genes_used': int(len(genes)), 'groups': int(len(rms)),
          'per_dose': rows, 'pairwise_magnitude_overlap': pairs,
          'extreme_dose_overlap': extreme['overlap_coefficient'],
          'magnitude_matched_bins_usable_of_10': usable,
          'magnitude_matched_groups_available': matched_groups,
          'bin_occupancy': counts,
          'gate': ('PASS - magnitude is separable from dose; OQ2 answerable by stratification'
                   if usable >= 4 and extreme['overlap_coefficient'] >= .2 else
                   'FAIL - magnitude too collinear with dose for stratification')}
print(table.to_string(index=False))
print()
print(json.dumps({k: report[k] for k in ('pairwise_magnitude_overlap', 'extreme_dose_overlap',
                                         'magnitude_matched_bins_usable_of_10',
                                         'magnitude_matched_groups_available', 'gate')}, indent=2))
from pathlib import Path
out = Path('results/dose_regime'); out.mkdir(parents=True, exist_ok=True)
(out / f'stage0_{label}.json').write_text(json.dumps(report, indent=2) + '\n')
table.to_csv(out / f'stage0_{label}.csv', index=False)
