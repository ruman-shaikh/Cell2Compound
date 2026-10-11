"""Predict held-out (ood) compound responses with a trained chemCPA checkpoint and save group means.

Run this inside the chemCPA venv (it imports their package); score the output in ours with
scripts/chemcpa_compare.py. Splitting the two steps avoids mixing their pinned numpy/rdkit with
ours inside one process.

chemCPA predicts single cells and reports R2 per condition; our pipeline reports RMSE on
control-relative pseudobulk group means. Those are not the same quantity, so nothing here is
scored - this step only emits, for each held-out (compound, dose) condition:
    observed group mean, predicted group mean, and the control reference,
all on chemCPA's own gene set, using their `compute_prediction` so the forward pass is theirs and
not a reimplementation of it.
"""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--chemcpa', type=Path, default=Path('/home/ruman/repos/chemCPA'))
parser.add_argument('--checkpoint', type=Path, default=None,
                    help='last.ckpt; defaults to the most recent under training_output/')
parser.add_argument('--dataset', default='ours_sciplex_v4', help='name of the config/dataset entry used for training')
parser.add_argument('--min-cells', type=int, default=6,
                    help='their evaluate_r2 skips categories with <=5 cells; matched here')
parser.add_argument('--out', type=Path, default=Path('results/chemcpa/ood_predictions.npz'))
args = parser.parse_args()

sys.path.insert(0, str(args.chemcpa))
import os
os.chdir(args.chemcpa)  # their config paths are relative to the repo root
from omegaconf import OmegaConf
from hydra import compose, initialize_config_dir
from chemCPA.data.data import load_dataset_splits
from chemCPA.lightning_module import ChemCPA
from chemCPA.train import bool2idx, compute_prediction, repeat_n

with initialize_config_dir(version_base=None, config_dir=str(args.chemcpa / 'config')):
    config = compose(config_name='main', overrides=[f'dataset={args.dataset}', 'model=ours'])
OmegaConf.set_struct(config, False)

datasets, dataset = load_dataset_splits(**config['dataset'], return_dataset=True)
dataset_config = {'num_genes': datasets['training'].num_genes,
                  'num_drugs': datasets['training'].num_drugs,
                  'num_covariates': datasets['training'].num_covariates,
                  'use_drugs_idx': dataset.use_drugs_idx,
                  'canon_smiles_unique_sorted': dataset.canon_smiles_unique_sorted}

checkpoint = args.checkpoint
if checkpoint is None:
    candidates = sorted(Path('training_output').glob('*/last.ckpt'), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError('no training_output/*/last.ckpt found; was the run completed?')
    checkpoint = candidates[-1]
print(f'loading {checkpoint}', flush=True)
model = ChemCPA(config, dataset_config)
state = torch.load(checkpoint, map_location='cpu')
missing, unexpected = model.load_state_dict(state['state_dict'], strict=False)
if missing:
    raise RuntimeError(f'checkpoint is missing parameters: {missing[:5]}')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device).eval()

# Their datamodule sets ood_control_dataset = datasplits['test_control'] (there is no
# 'ood_control' split), so ood predictions are translated from TEST-split control cells. Matched
# here so the prediction is theirs rather than a variant of it. This is also why an empty
# test_control made both test and ood R2 return NaN.
treated = datasets['ood']
controls = datasets['test_control']
genes_control = controls.genes.to(device)
reference = controls.genes.mean(0).cpu().numpy()
print(f'ood: {len(treated.pert_categories)} cells, {len(np.unique(treated.pert_categories))} conditions; '
      f'{genes_control.shape[0]} control cells', flush=True)

index = pd.Index(treated.pert_categories, dtype='category')
categories, counts = np.unique(treated.pert_categories, return_counts=True)
rows, skipped = [], []
for category, count in zip(categories, counts):
    if count <= args.min_cells - 1:
        skipped.append({'category': str(category), 'cells': int(count)}); continue
    if 'control' in str(category).lower() or 'dmso' in str(category).lower():
        continue
    positions = bool2idx(index.get_loc(category))
    first = positions[0]
    covariates = [repeat_n(c[first], genes_control.size(0)) for c in treated.covariates]
    drugs = (repeat_n(treated.drugs_idx[first], genes_control.size(0)).squeeze(),
             repeat_n(treated.dosages[first], genes_control.size(0)).squeeze())
    with torch.no_grad():
        mean_pred, _ = compute_prediction(model.model, genes_control, drugs, covariates)
    rows.append({'category': str(category), 'cells': int(count),
                 'observed': treated.genes[positions].mean(0).cpu().numpy(),
                 'predicted': mean_pred.mean(0).cpu().numpy()})
    if len(rows) % 25 == 0:
        print(f'  {len(rows)} conditions predicted', flush=True)

if not rows:
    raise RuntimeError('no ood condition had enough cells to predict')
out = Path(args.chemcpa if args.out.is_absolute() else Path.cwd())
target = args.out if args.out.is_absolute() else Path('/home/ruman/repos/Cell2Compound') / args.out
target.parent.mkdir(parents=True, exist_ok=True)
np.savez_compressed(target,
                    categories=np.array([r['category'] for r in rows]),
                    cells=np.array([r['cells'] for r in rows]),
                    observed=np.stack([r['observed'] for r in rows]),
                    predicted=np.stack([r['predicted'] for r in rows]),
                    control_reference=reference,
                    genes=np.array(dataset.var_names, dtype=str))
summary = {'checkpoint': str(checkpoint), 'dataset': args.dataset, 'device': device,
           'conditions_predicted': len(rows), 'conditions_skipped_too_few_cells': skipped,
           'ood_control_cells': int(genes_control.shape[0]), 'genes': int(len(dataset.var_names)),
           'output': str(target),
           'note': ('group means only; chemCPA R2 is not computed here because our comparison uses '
                    'RMSE on control-relative deltas over the same genes')}
Path(str(target).replace('.npz', '.json')).write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps(summary, indent=2))
