"""Generate the RDKit 2D descriptor embedding parquet that chemCPA looks up by SMILES.

chemCPA indexes the embedding frame by SMILES string and, for any SMILES it cannot find,
substitutes a zero vector with only a printed warning (chemCPA/embedding.py). A string mismatch
would therefore not fail - it would quietly hand the model no drug information at all and make the
baseline look far worse than it is. So the index is built with chemCPA's own
`canonicalize_smiles`, and this script asserts every SMILES in the dataset resolves.
"""
import argparse
import json
import sys
from pathlib import Path
import anndata as ad
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors

RDLogger.DisableLog('rdApp.*')
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--dataset', type=Path,
                    default=Path('/home/ruman/repos/chemCPA/project_folder/datasets/sciplex_a549_24h_ours.h5ad'))
parser.add_argument('--chemcpa', type=Path, default=Path('/home/ruman/repos/chemCPA'))
parser.add_argument('--out', type=Path,
                    default=Path('/home/ruman/repos/chemCPA/project_folder/embeddings/rdkit/ours_rdkit2D_embedding.parquet'))
args = parser.parse_args()

sys.path.insert(0, str(args.chemcpa))
from chemCPA.helper import canonicalize_smiles  # noqa: E402  - uses their canonicalisation, not ours

data = ad.read_h5ad(args.dataset, backed='r')
smiles = sorted(set(data.obs.SMILES.astype(str)))
data.file.close()

names = [name for name, _ in Descriptors.descList]
rows, failed = [], []
for raw in smiles:
    canonical = canonicalize_smiles(raw)
    mol = Chem.MolFromSmiles(canonical) if canonical else None
    if mol is None:
        failed.append(raw); continue
    values = []
    for _, function in Descriptors.descList:
        try:
            values.append(float(function(mol)))
        except Exception:
            values.append(np.nan)
    rows.append((raw, canonical, values))
if failed:
    raise ValueError(f'RDKit could not parse: {failed}')

frame = pd.DataFrame([v for _, _, v in rows], columns=names, index=[c for _, c, _ in rows])
# Descriptors that are constant or non-finite carry no information and would break standardisation.
frame = frame.replace([np.inf, -np.inf], np.nan)
dropped_nan = frame.columns[frame.isna().any()].tolist()
frame = frame.drop(columns=dropped_nan)
dropped_constant = frame.columns[frame.std(ddof=0) == 0].tolist()
frame = frame.drop(columns=dropped_constant)
# Standardise so no single large-magnitude descriptor dominates the drug encoder.
frame = (frame - frame.mean()) / frame.std(ddof=0)
frame.index.name = None

# The dataset's SMILES must resolve to this index after chemCPA's canonicalisation, or chemCPA
# silently zero-fills. Verified here rather than discovered as a bad result later.
resolved = {canonicalize_smiles(s) for s in smiles}
missing = resolved - set(frame.index)
if missing:
    raise ValueError(f'{len(missing)} dataset SMILES absent from the embedding index: {sorted(missing)[:5]}')
raw_index_mismatch = sorted(s for s in smiles if s != canonicalize_smiles(s))

args.out.parent.mkdir(parents=True, exist_ok=True)
frame.to_parquet(args.out)
summary = {'output': str(args.out), 'compounds': int(len(frame)), 'descriptors': int(frame.shape[1]),
           'dropped_non_finite': dropped_nan, 'dropped_constant': dropped_constant,
           'standardised': True,
           'smiles_changed_by_canonicalisation': len(raw_index_mismatch),
           'note': ('index is chemCPA-canonical SMILES; every dataset SMILES verified present, '
                    'so no compound falls back to a zero embedding')}
Path(str(args.out).replace('.parquet', '.json')).write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps(summary, indent=2))
