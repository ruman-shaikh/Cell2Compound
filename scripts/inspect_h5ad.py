"""Inspect metadata without loading the full expression matrix."""
import argparse
import json
from pathlib import Path
import anndata as ad

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('input', type=Path)
args = parser.parse_args()
data = ad.read_h5ad(args.input, backed='r')
report = {'shape': list(data.shape), 'obs_columns': list(data.obs.columns),
          'var_columns': list(data.var.columns), 'layers': list(data.layers),
          'uns_keys': list(data.uns), 'categories': {}}
for key in data.obs.columns:
    counts = data.obs[key].value_counts(dropna=False)
    if len(counts) <= 250:
        report['categories'][key] = {str(k): int(v) for k, v in counts.items()}
Path('results').mkdir(exist_ok=True)
Path('results/single_cell_inventory.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
data.file.close()
