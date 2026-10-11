"""Convert a cached ChEMBL molecule response to a provenance-backed input table."""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('input', type=Path)
parser.add_argument('--source-url', required=True)
parser.add_argument('--out', type=Path, default=Path('data/compounds.csv'))
args = parser.parse_args()
raw = args.input.read_bytes()
payload = json.loads(raw)
records = []
for molecule in payload['molecules']:
    records.append({'compound_id': molecule['molecule_chembl_id'],
                    'name': molecule.get('pref_name'),
                    'smiles': (molecule.get('molecule_structures') or {}).get('canonical_smiles'),
                    'source': 'ChEMBL'})
args.out.parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame(records).to_csv(args.out, index=False)
args.out.with_suffix('.provenance.json').write_text(json.dumps({
    'source_url': args.source_url, 'raw_file': str(args.input),
    'sha256': hashlib.sha256(raw).hexdigest(),
    'imported_utc': datetime.now(timezone.utc).isoformat(),
    'rows': len(records), 'page_meta': payload.get('page_meta'),
    'purpose': 'Initial chemistry integration smoke test; not the modelling cohort'
}, indent=2) + '\n')
print(f'Imported {len(records)} ChEMBL records to {args.out}')
