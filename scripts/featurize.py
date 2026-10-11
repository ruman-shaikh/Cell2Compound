"""Usage: python -m scripts.featurize compounds.csv --out results/chemistry"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from cell2compound.chemistry import featurize

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('input', type=Path)
parser.add_argument('--out', type=Path, default=Path('results/chemistry'))
args = parser.parse_args()
frame = pd.read_csv(args.input)
descriptors, fingerprints, rejected = featurize(frame)
args.out.mkdir(parents=True, exist_ok=True)
descriptors.to_csv(args.out / 'descriptors.csv', index=False)
np.savez_compressed(args.out / 'fingerprints.npz', fingerprints=fingerprints,
                    compound_ids=descriptors.get('compound_id', pd.Series(dtype=str)).to_numpy(dtype=str))
rejected.to_csv(args.out / 'rejected.csv', index=False)
(args.out / 'summary.json').write_text(json.dumps({'input_rows': len(frame),
    'accepted': len(descriptors), 'rejected': len(rejected),
    'fingerprint': {'kind': 'Morgan', 'radius': 2, 'bits': 2048}}, indent=2) + '\n')
print(f'Accepted {len(descriptors)}; rejected {len(rejected)}; output: {args.out}')
