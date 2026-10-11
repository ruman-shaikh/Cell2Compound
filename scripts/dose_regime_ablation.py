"""Stage 2: does chemical structure's predictive value track measurable signal, and is there
any regime where structure beats a class-only encoding?

Reformulated after Stage 1 inverted the original hypothesis. Four feature arms, applied at the
STRUCTURE level with the compound split held fixed on true structures, so every arm holds out
the same compounds and only the information content of the features changes:

  full                  real Morgan fingerprints
  within_class_scramble fingerprints permuted among compounds sharing a mechanism class
                        -> destroys structure, PRESERVES class. The control Bai et al.'s global
                           shuffle structurally cannot provide.
  class_onehot          one-hot mechanism class instead of a fingerprint (Tanimoto on one-hot is
                        exactly a class-identity kernel)
  global_scramble       fingerprints permuted across all compounds -> reproduces their ablation

Scored inside strata of replicate-derived SNR and of dose. A stratum whose best baseline already
sits at the replicate floor has no headroom, so no model can be shown to improve on it; those are
reported as UNMEASURABLE rather than given a misleading ratio.
"""
import json, sys
from dataclasses import replace
from pathlib import Path
import numpy as np, pandas as pd, torch
from cell2compound.measurability import class_onehot, class_permutation, noise_metrics
from cell2compound.pipeline import (BASELINES, attainable_fraction, fit_predictions,
                                    load_response_data, select_genes)
from cell2compound.splits import compound_split

path = sys.argv[1] if len(sys.argv) > 1 else 'data/pseudobulk_A549_24h.npz'
label = sys.argv[2] if len(sys.argv) > 2 else 'A549_24h'
seeds = [42, 43, 44, 45, 46]
device = 'cuda' if torch.cuda.is_available() else 'cpu'
data = load_response_data(path, 'data/raw/sciplex_compounds.csv')

# --- split-independent SNR, used only to define strata (never for gene selection) -------------
detection = data.detected.sum(0) / data.cells.sum()
snr_genes = np.flatnonzero(detection >= .05)
snr = noise_metrics(data.full_delta, data.rep_difference, snr_genes)['snr']

# --- mechanism class per structure row --------------------------------------------------------
classes = pd.read_csv('results/stress_test/compound_classes.csv').set_index('compound').mechanism_class
group_class = data.groups.compound.map(classes).fillna('unannotated').to_numpy()
n_struct = int(data.groups.fp_row.max()) + 1
struct_class = np.array(['unannotated'] * n_struct, dtype=object)
for row, cls in zip(data.groups.fp_row.to_numpy(), group_class):
    struct_class[row] = cls

rng = np.random.default_rng(0)
fp = data.fingerprints
struct_fp = np.zeros((n_struct, fp.shape[1]), dtype=fp.dtype)
for row, vec in zip(data.groups.fp_row.to_numpy(), fp):
    struct_fp[row] = vec
onehot = class_onehot(struct_class)

def arm_data(name):
    rows = data.groups.fp_row.to_numpy()
    if name == 'full':
        new_fp, displaced = fp, 0.
    elif name == 'class_onehot':
        new_fp, displaced = onehot[rows], 1.
    else:
        m = class_permutation(struct_class, name == 'within_class_scramble', rng)
        new_fp = struct_fp[m][rows]
        displaced = float((m != np.arange(n_struct)).mean())
    return replace(data, fingerprints=new_fp,
                   features=np.c_[new_fp, np.log10(data.doses)]), displaced

# --- strata -----------------------------------------------------------------------------------
snr_edges = np.quantile(snr, [0, .25, .5, .75, 1.]); snr_edges[-1] += 1e-9
def strata_of(rows):
    out = {'all': np.ones(len(rows), bool)}
    for d in np.unique(data.doses):
        out[f'dose_{d:g}nM'] = data.doses[rows] == d
    q = np.digitize(snr[rows], snr_edges[1:-1])
    for k in range(4):
        out[f'snr_q{k + 1}'] = q == k
    # Mechanism-class strata. This repository's long-running question is whether the whole
    # expression result is an HDAC detector, so HDAC is split out explicitly; held-out test sets
    # are small, so only the largest classes clear the minimum-size filter downstream.
    cls = group_class[rows]
    out['hdac'] = cls == 'HDAC'
    out['not_hdac'] = cls != 'HDAC'
    for c in ('JAK', 'Aurora Kinase'):
        out[f'class_{c.split()[0].lower()}'] = cls == c
    return out

records = []
for arm in ('full', 'within_class_scramble', 'class_onehot', 'global_scramble'):
    adata, displaced = arm_data(arm)
    for seed in seeds:
        splits = compound_split(data.canonical, seed=seed)   # fixed on TRUE structures
        train = np.flatnonzero(splits == 'train'); valid = np.flatnonzero(splits == 'validation')
        test = np.flatnonzero(splits == 'test')
        genes = select_genes(data, train, 1000)
        y = data.full_delta[:, genes]
        preds, _ = fit_predictions(adata, y, train, valid, test, seed, device)
        observed = y[test]
        for name, mask in strata_of(test).items():
            if mask.sum() < 15:
                continue
            fl = float(np.sqrt((data.rep_difference[test][mask][:, genes] ** 2).mean()) / 2)
            rmses = {m: float(np.sqrt(((observed[mask] - p[mask]) ** 2).mean()))
                     for m, p in preds.items()}
            base = min(BASELINES, key=lambda b: rmses[b])
            for m, r in rmses.items():
                records.append({'arm': arm, 'displaced_fraction': round(displaced, 3), 'seed': seed,
                                'stratum': name, 'groups': int(mask.sum()), 'model': m,
                                'rmse': r, 'floor': fl, 'baseline': base,
                                'baseline_rmse': rmses[base],
                                'headroom': rmses[base] - fl,
                                'attainable': attainable_fraction(r, rmses[base], fl)})
frame = pd.DataFrame(records)
out = Path('results/dose_regime'); out.mkdir(parents=True, exist_ok=True)
frame.to_csv(out / f'stage2_raw_{label}.csv', index=False)

agg = (frame.groupby(['stratum', 'arm', 'model'])
       .agg(rmse=('rmse', 'mean'), rmse_sd=('rmse', 'std'), floor=('floor', 'mean'),
            headroom=('headroom', 'mean'), groups=('groups', 'mean'),
            attainable=('attainable', 'mean'), wins=('attainable', lambda s: int((s > 0).sum())))
       .reset_index())
agg['measurable'] = agg.headroom > .05 * agg.floor
agg.to_csv(out / f'stage2_summary_{label}.csv', index=False)

key = 'tanimoto_kernel_ridge'
print(f'=== {label}: {key}, mean over {len(seeds)} seeds ===')
piv = agg[agg.model == key].pivot_table(index='stratum', columns='arm', values='rmse')
head = agg[(agg.model == key)][['stratum', 'headroom', 'floor', 'measurable']].drop_duplicates('stratum').set_index('stratum')
print(pd.concat([piv, head], axis=1).round(4).to_string())
print()
print('=== attainable fraction (mean), tanimoto_kernel_ridge ===')
print(agg[agg.model == key].pivot_table(index='stratum', columns='arm', values='attainable').round(3).to_string())
