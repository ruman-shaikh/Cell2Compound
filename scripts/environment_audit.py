"""Run with the shared general venv; does not modify it."""
import importlib.metadata
import os
import json
import platform
import shutil
import sys
from pathlib import Path

import torch

packages = {}
for name in ['torch', 'rdkit', 'pandas', 'numpy', 'scipy', 'scikit-learn', 'transformers',
             'scanpy', 'anndata', 'h5py', 'requests', 'matplotlib', 'biopython',
             'vina', 'meeko', 'gemmi', 'pytest']:
    try:
        packages[name] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        packages[name] = None
report = {'python': sys.executable, 'python_version': platform.python_version(),
          'packages': packages, 'cuda_available': torch.cuda.is_available(),
          'cuda_device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
          'vina_cli': shutil.which('vina'), 'obabel': shutil.which('obabel'),
          'autogrid4': shutil.which('autogrid4'),
          # Meeko's CLI lives next to this interpreter; PATH may not include the venv bin.
          'mk_prepare_receptor': shutil.which('mk_prepare_receptor.py',
                                              path=str(Path(sys.executable).parent)+':'+(os.environ.get('PATH') or '')),
          'disk_free_gib': round(shutil.disk_usage('.').free / 2**30, 2)}
Path('results').mkdir(exist_ok=True)
Path('results/environment.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
