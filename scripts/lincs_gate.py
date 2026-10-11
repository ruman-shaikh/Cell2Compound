"""Gate: can LINCS L1000 support the replicate-floor method at all, before anything large is pulled?

Our measurability result needs at least two independent profiles of the SAME (compound, dose,
cell line, time) so that a replicate floor can be estimated. LINCS Level 3 is per-instance (one
profile per well), so the question is whether instances group into replicated conditions at usable
depth. If they do not, no amount of expression data makes the method applicable and the 4.3 GB
download is wasted.

Reads only the small metadata tables. Paths are passed as arguments because the download
directory holds untrusted third-party files.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd

inst_path = Path(sys.argv[1])
pert_path = Path(sys.argv[2]) if len(sys.argv) > 2 else None

inst = pd.read_csv(inst_path, sep='\t', low_memory=False)
print(f'instances: {len(inst):,}  columns: {list(inst.columns)}')

dose_col = next((c for c in ('pert_dose', 'pert_idose') if c in inst.columns), None)
time_col = next((c for c in ('pert_time', 'pert_itime') if c in inst.columns), None)
type_col = next((c for c in ('pert_type',) if c in inst.columns), None)
keys = ['pert_id', dose_col, time_col, 'cell_id']
print(f'grouping keys: {keys}')

treated = inst
if type_col:
    print(f'\npert_type counts:\n{inst[type_col].value_counts().head(8).to_string()}')
    treated = inst[inst[type_col] == 'trt_cp']
print(f'\ncompound-treated instances: {len(treated):,}')

groups = treated.groupby(keys, dropna=False).size()
print(f'distinct (compound, dose, time, cell) conditions: {len(groups):,}')
print(f'replicates per condition: median {groups.median():.0f}  '
      f'mean {groups.mean():.2f}  max {groups.max()}')
for k in (2, 3, 4):
    n = int((groups >= k).sum())
    print(f'  conditions with >= {k} replicates: {n:,} ({n / len(groups):.1%})')

usable = groups[groups >= 2]
print(f'\nUSABLE conditions (>=2 replicates): {len(usable):,}')
print(f'instances they account for: {int(usable.sum()):,}')

# Dose coverage is what makes this a test of OUR claim rather than a generic benchmark.
per_compound = (treated[treated.set_index(keys).index.isin(usable.index)]
                .groupby(['pert_id', 'cell_id'])[dose_col].nunique())
print(f'\ncompound x cell pairs with >=2-replicate conditions: {len(per_compound):,}')
for k in (2, 3, 4):
    n = int((per_compound >= k).sum())
    print(f'  pairs spanning >= {k} doses: {n:,} ({n / max(len(per_compound), 1):.1%})')
print(f'distinct compounds: {treated.pert_id.nunique():,}  cell lines: {treated.cell_id.nunique()}')
print(f'distinct doses: {treated[dose_col].nunique()}  times: {sorted(treated[time_col].unique())[:6]}')

if pert_path and pert_path.exists():
    pert = pd.read_csv(pert_path, sep='\t', low_memory=False)
    print(f'\npert_info columns: {list(pert.columns)}')
    smiles_col = next((c for c in ('canonical_smiles', 'smiles') if c in pert.columns), None)
    if smiles_col:
        have = pert[pert[smiles_col].notna() & (pert[smiles_col] != '-666')]
        print(f'compounds with usable SMILES: {len(have):,} of {len(pert):,}')

print('\nVERDICT: method applies' if len(usable) > 500 else '\nVERDICT: insufficient replication')
