"""Is the 72h SNR advantage real, or an artifact of the 47-compound panel being HDAC-enriched?

Restrict A549 24h to exactly the compounds measured at 72h, then compare like for like.
"""
import json
from pathlib import Path
import numpy as np, pandas as pd
from cell2compound.pipeline import load_response_data

def snr_table(path, keep=None):
    d = load_response_data(path, 'data/raw/sciplex_compounds.csv')
    det = d.detected.sum(0) / d.cells.sum()
    g = np.flatnonzero(det >= .05)
    rms = np.sqrt((d.full_delta[:, g] ** 2).mean(1))
    fl = np.sqrt((d.rep_difference[:, g] ** 2).mean(1)) / 2
    s = np.sqrt(np.maximum(rms ** 2 - fl ** 2, 0)) / fl
    sel = np.isin(d.groups.compound.to_numpy(), keep) if keep is not None else np.ones(len(s), bool)
    rows = []
    for dose in np.unique(d.doses):
        m = sel & (d.doses == dose)
        if not m.sum():
            continue
        cells = np.array([d.cells[r].sum() for r in d.groups.table_rows[m]])
        rows.append({'dose_nM': float(dose), 'groups': int(m.sum()),
                     'snr_median': round(float(np.median(s[m])), 3),
                     'floor_median': round(float(np.median(fl[m])), 4),
                     'cells_median': int(np.median(cells))})
    return pd.DataFrame(rows), set(d.groups.compound.unique())

t72, cmp72 = snr_table('data/pseudobulk_A549_72h.npz')
t24_all, cmp24 = snr_table('data/pseudobulk_A549_24h.npz')
shared = sorted(cmp72 & cmp24)
t24_m, _ = snr_table('data/pseudobulk_A549_24h.npz', keep=np.array(shared))

print(f'shared compounds: {len(shared)}\n')
for name, t in (('A549 24h  ALL 188 compounds', t24_all),
                (f'A549 24h  matched {len(shared)} compounds', t24_m),
                (f'A549 72h  {len(cmp72)} compounds', t72)):
    print(name); print(t.to_string(index=False)); print()

merged = t24_m.merge(t72, on='dose_nM', suffixes=('_24h', '_72h'))
merged['snr_ratio_72_over_24'] = (merged.snr_median_72h / merged.snr_median_24h).round(2)
merged['cells_ratio'] = (merged.cells_median_72h / merged.cells_median_24h).round(2)
print('=== matched comparison ===')
print(merged[['dose_nM', 'snr_median_24h', 'snr_median_72h', 'snr_ratio_72_over_24',
              'floor_median_24h', 'floor_median_72h', 'cells_ratio']].to_string(index=False))
out = Path('results/dose_regime'); out.mkdir(parents=True, exist_ok=True)
(out / 'time_matched_A549.json').write_text(json.dumps(
    {'shared_compounds': len(shared), 'matched_24h': t24_m.to_dict('records'),
     'at_72h': t72.to_dict('records'), 'all_188_at_24h': t24_all.to_dict('records'),
     'note': ('72h panel is a 47-compound subset enriched for HDAC and Aurora; the matched rows '
              'are the only valid time comparison')}, indent=2) + '\n')
