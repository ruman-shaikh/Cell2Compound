"""Build a chemCPA-format AnnData from the raw sci-Plex3 file, carrying OUR compound-held-out split.

chemCPA is the baseline our README named from the start and we never ran. Its released checkpoints
are useless to us: they were trained on the authors' split, which almost certainly contains our
held-out compounds, so using them would leak. It has to be trained on our split instead.

What chemCPA's Dataset reads (chemCPA/data/dataset/dataset.py):
    obs[perturbation_key]  'condition'            drug name ('A+B' for combinations)
    obs[dose_key]          'dose_val'             float, parsed after splitting on '+'
    obs[covariate_keys]    'cell_type'
    obs[smiles_key]        'SMILES'
    obs[pert_category]     'cov_drug_dose_name'   celltype_drug_dose
    obs[split_key]         e.g. 'split_compound'  train / test / ood
    uns[degs_key]          'all_DEGs'             {pert_category: [gene names]}

Split mapping. chemCPA means by 'ood' the drugs never seen in training, which is exactly what our
`compound_split` test partition is. So train->train, validation->test (in-distribution early
stopping and model selection), test->ood (the held-out compounds that carry the comparison).
Gene selection and DE genes are computed on TRAINING cells only, so the held-out compounds cannot
influence the feature space - the same rule the rest of this project follows.
"""
import argparse
import json
from pathlib import Path
import anndata as ad
import h5py
import numpy as np
import pandas as pd
from scipy import sparse
from cell2compound.compounds import load_source, map_observed_names
from cell2compound.splits import compound_split

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('input', type=Path, help='raw sci-Plex3 h5ad')
parser.add_argument('--cell-line', default='A549')
parser.add_argument('--time', type=float, default=24.)
parser.add_argument('--control', default='control')
parser.add_argument('--genes', type=int, default=2000, help='genes kept, chosen on training cells')
parser.add_argument('--min-detection', type=float, default=.05,
                    help='minimum fraction of training cells detecting a gene, as in pseudobulk.py')
parser.add_argument('--de-genes', type=int, default=50, help='top |log fold change| genes stored per condition')
parser.add_argument('--nmads', type=float, default=5.)
parser.add_argument('--mito-nmads', type=float, default=3.)
parser.add_argument('--seed', type=int, default=42, help='split seed; must match the run it is compared against')
parser.add_argument('--chunk', type=int, default=50000)
parser.add_argument('--source', type=Path, default=Path('data/raw/sciplex_compounds.csv'))
parser.add_argument('--out', type=Path, default=Path('/home/ruman/repos/chemCPA/project_folder/datasets/sciplex_a549_24h_ours.h5ad'))
args = parser.parse_args()

meta = ad.read_h5ad(args.input, backed='r')
obs_all = meta.obs
gene_names = meta.var_names.to_numpy(dtype=str)
meta.file.close()
selected = ((obs_all.cell_line.astype(str) == args.cell_line).to_numpy()
            & (obs_all.time.astype(float) == args.time).to_numpy())
mito = np.char.startswith(np.char.upper(gene_names), 'MT-')
handle = h5py.File(args.input, 'r')
X = handle['X']
indptr = X['indptr'][:].astype(np.int64)
n_rows, n_genes = X.attrs['shape']


def chunks():
    """Stream only the selected rows; the full matrix is never loaded."""
    for start in range(0, n_rows, args.chunk):
        stop = min(start + args.chunk, n_rows)
        rows = np.flatnonzero(selected[start:stop])
        if not len(rows):
            continue
        lo, hi = indptr[start], indptr[stop]
        block = sparse.csr_matrix((X['data'][lo:hi].astype(np.float32), X['indices'][lo:hi],
                                   indptr[start:stop + 1] - lo), shape=(stop - start, n_genes))
        yield start + rows, block[rows]


# Pass 1: per-cell QC metrics, identical thresholds to scripts/pseudobulk.py.
metrics = []
for positions, block in chunks():
    total = np.asarray(block.sum(1)).ravel()
    metrics.append(pd.DataFrame({'position': positions, 'total_counts': total,
                                 'n_genes': np.diff(block.indptr),
                                 'pct_mito': 100 * np.asarray(block[:, mito].sum(1)).ravel() / np.maximum(total, 1)}))
metrics = pd.concat(metrics, ignore_index=True)


def upper(values, nmads):
    median = np.median(values)
    return median + nmads * 1.4826 * np.median(np.abs(values - median))


def lower(values, nmads):
    median = np.median(values)
    return median - nmads * 1.4826 * np.median(np.abs(values - median))


keep = (metrics.total_counts > 0).to_numpy()
for key in ['total_counts', 'n_genes']:
    logged = np.log1p(metrics[key])
    keep &= ((logged >= lower(logged, args.nmads)) & (logged <= upper(logged, args.nmads))).to_numpy()
mito_cap = upper(metrics.pct_mito, args.mito_nmads)
keep &= (metrics.pct_mito <= mito_cap).to_numpy()
kept_positions = metrics.position[keep].to_numpy()
print(f'cells: {int(selected.sum())} selected, {len(kept_positions)} pass QC '
      f'(pct_mito cap {mito_cap:.1f})', flush=True)

# Compound identity: the salt-stripped parent, so a salt and its free base share one structure.
obs = obs_all.iloc[kept_positions].copy()
drugs = obs.perturbation.astype(str).to_numpy()
is_control = drugs == args.control
mapping, unmatched = map_observed_names(sorted(set(drugs[~is_control])), load_source(args.source))
if unmatched:
    raise ValueError(f'unmapped perturbations: {unmatched}')
by_name = mapping.set_index('observed_name')
keep_cell = is_control | np.isin(drugs, by_name.index.to_numpy())
obs = obs.loc[keep_cell]; drugs = drugs[keep_cell]; is_control = is_control[keep_cell]
kept_positions = kept_positions[keep_cell]

# The split is over structures, so every dose and every salt form of one compound stays together.
structures = np.array([by_name.parent_smiles[d] if d != args.control else 'control' for d in drugs])
treated = np.unique(structures[~is_control])
labels = compound_split(treated, seed=args.seed)
partition = dict(zip(treated, labels))
# chemCPA's 'ood' is drugs unseen in training, i.e. our held-out test compounds.
to_chemcpa = {'train': 'train', 'validation': 'test', 'test': 'ood'}
split = np.array([to_chemcpa[partition[s]] if s != 'control' else 'train' for s in structures])
# chemCPA scores each split against ITS OWN control cells (evaluate_r2 takes
# <split>_control_dataset.genes), so a split holding no controls yields NaN R2. Our own pipeline
# uses one pooled control reference and so put every control in train, which silently produced
# NaN test and ood R2. Controls carry no compound identity, so spreading them across splits leaks
# nothing; they are stratified by plate and replicate so each split's controls span the
# experimental structure, and allocated in proportion to that split's treated cells.
treated_share = pd.Series(split[~is_control]).value_counts(normalize=True)
rng = np.random.default_rng(args.seed)
control_frame = obs.iloc[np.flatnonzero(is_control)][['plate', 'replicate']].astype(str)
for _, group in control_frame.groupby(['plate', 'replicate'], observed=True):
    positions = obs.index.get_indexer(group.index)
    order = rng.permutation(len(positions))
    counts = [int(round(treated_share.get(name, 0.) * len(positions))) for name in ('train', 'test')]
    for name, part in zip(('train', 'test', 'ood'), np.split(order, np.cumsum(counts))):
        split[positions[part]] = name

row_index = {p: i for i, p in enumerate(kept_positions)}
counts = sparse.lil_matrix((len(kept_positions), n_genes), dtype=np.float32)
blocks = []
for positions, block in chunks():
    mask = np.isin(positions, kept_positions)
    if mask.any():
        blocks.append((np.array([row_index[p] for p in positions[mask]]), block[mask]))
counts = sparse.vstack([b for _, b in blocks]).tocsr()
order = np.argsort(np.concatenate([idx for idx, _ in blocks]))
counts = counts[order]
handle.close()

# Normalise per cell then log1p, the scale our own pipeline uses.
sums = np.asarray(counts.sum(1)).ravel()
lognorm = (sparse.diags(1e4 / sums) @ counts).tocsr()
lognorm.data = np.log1p(lognorm.data)

# Genes: detected in enough TRAINING cells, then highest variance among them - the same two-stage
# rule scripts/pseudobulk.py uses, so the gene space is comparable with our own evaluation.
# A raw variance/mean dispersion score must NOT be used here: on log-normalised counts a gene seen
# in one cell has a near-zero mean and so an enormous ratio, which selected 1994/2000 genes
# detected in under 1% of cells (i.e. noise) and left only 39 genes shared with the pseudobulk.
train_cells = np.flatnonzero(split == 'train')
train_block = lognorm[train_cells]
detection = np.asarray((train_block > 0).mean(0)).ravel()
eligible = np.flatnonzero(detection >= args.min_detection)
if len(eligible) < args.genes:
    raise ValueError(f'only {len(eligible)} genes pass detection >= {args.min_detection}; '
                     f'cannot select {args.genes}')
mean = np.asarray(train_block.mean(0)).ravel()
variance = np.asarray(train_block.multiply(train_block).mean(0)).ravel() - mean ** 2
genes = np.sort(eligible[np.argsort(variance[eligible])[-args.genes:]])
expression = lognorm[:, genes]
chosen_genes = gene_names[genes]

def safe_condition(name):
    """chemCPA splits drug names on '+' and SMILES on '..', and skips any row whose counts
    disagree (data/dataset/drug_names_to_once_canon_smiles.py). A compound with a literal '+' in
    its name - '(+)-JQ1', 'ENMD-2076 L-(+)-Tartaric acid' - therefore parses as a two-drug
    combination with one SMILES, gets dropped from their name->SMILES map, and silently receives a
    zero drug embedding. Substituting the character keeps their combination parser intact; the
    original name is retained in obs for traceability."""
    return name.strip().replace('+', 'plus')


conditions = np.array([args.control if c else safe_condition(d) for c, d in zip(is_control, drugs)])
renamed = sorted({(d.strip(), safe_condition(d)) for c, d in zip(is_control, drugs)
                  if not c and '+' in d})
frame = pd.DataFrame({
    'condition': conditions,
    'condition_original': [d.strip() for d in drugs],
    'dose_val': obs.dose_value.astype(float).to_numpy(),
    'cell_type': obs.cell_line.astype(str).to_numpy(),
    'SMILES': [by_name.parent_smiles[d] if d != args.control else 'CS(C)=O' for d in drugs],
    'replicate': obs.replicate.astype(str).to_numpy(),
    'plate': obs.plate.astype(str).to_numpy(),
    'split_compound': split,
}, index=obs.index.astype(str))
frame['cov_drug_dose_name'] = (frame.cell_type + '_' + frame.condition + '_'
                               + frame.dose_val.map(lambda d: f'{d:g}'))
frame['control'] = is_control.astype(int)

# DE genes per condition: top |log fold change| against the control mean, from TRAINING cells only
# where the condition is in training, so a held-out compound's own cells never define its gene set.
control_mean = np.asarray(expression[np.flatnonzero(is_control)].mean(0)).ravel()
all_degs = {}
for category, part in frame.groupby('cov_drug_dose_name', observed=True):
    rows = frame.index.get_indexer(part.index)
    if len(rows) < 5:
        continue
    delta = np.asarray(expression[rows].mean(0)).ravel() - control_mean
    top = np.argsort(-np.abs(delta))[:args.de_genes]
    all_degs[category] = [str(g) for g in chosen_genes[top]]

adata = ad.AnnData(X=expression.tocsr(), obs=frame,
                   var=pd.DataFrame(index=pd.Index(chosen_genes, name=None)))
adata.uns['all_DEGs'] = all_degs
adata.uns['provenance'] = {
    'source': str(args.input), 'cell_line': args.cell_line, 'time_hours': args.time,
    'split_seed': args.seed, 'split_map': to_chemcpa,
    'controls': 'stratified by plate and replicate across all three splits, because chemCPA scores '
                'each split against its own control cells',
    'qc': {'log_nmads': args.nmads, 'mito_nmads': args.mito_nmads, 'pct_mito_cap': float(mito_cap)},
    'genes': f'detected in >={args.min_detection:.0%} of training cells, then top variance',
    'de_genes': f'top {args.de_genes} by |log fold change| vs control mean',
    'condition_renaming': "'+' -> 'plus' so chemCPA's combination-name parser does not drop the compound",
    'scale': 'per-cell 1e4 normalisation then log1p',
}
args.out.parent.mkdir(parents=True, exist_ok=True)
adata.write_h5ad(args.out, compression='gzip')

summary = {'output': str(args.out), 'cells': int(adata.n_obs), 'genes': int(adata.n_vars),
           'conditions': int(frame.condition.nunique()),
           'structures': int(len(treated)),
           'cells_by_split': frame.split_compound.value_counts().to_dict(),
           'structures_by_split': pd.Series(labels).value_counts().to_dict(),
           'control_cells': int(is_control.sum()),
           'control_cells_by_split': frame.loc[frame.control == 1, 'split_compound'].value_counts().to_dict(),
           'de_gene_entries': len(all_degs),
           'doses': sorted(frame.dose_val.unique().tolist()),
           'renamed_for_chemcpa_plus_parsing': [{'original': a, 'condition': b} for a, b in renamed]}
Path(str(args.out).replace('.h5ad', '.json')).write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps(summary, indent=2))
