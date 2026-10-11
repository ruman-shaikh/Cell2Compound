"""Stream one cell line/time point from the raw h5ad; QC every cell; sum log-normalised
expression per (compound, dose, replicate, plate). Never loads the full matrix."""
import argparse
import json
from pathlib import Path
import anndata as ad
import h5py
import numpy as np
import pandas as pd
from scipy import sparse

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('input',type=Path)
parser.add_argument('--cell-line',default='A549')
parser.add_argument('--time',type=float,default=24.)
parser.add_argument('--keys',nargs='+',default=['perturbation','dose_value','replicate','plate'])
parser.add_argument('--nmads',type=float,default=5.)
parser.add_argument('--mito-nmads',type=float,default=3.)
parser.add_argument('--min-cell-fraction',type=float,default=1e-3,help='keep genes detected in at least this fraction of kept cells')
parser.add_argument('--chunk',type=int,default=50000)
parser.add_argument('--out',type=Path,default=Path('data/pseudobulk_A549_24h.npz'))
args=parser.parse_args()

meta=ad.read_h5ad(args.input,backed='r')
obs=meta.obs; genes=meta.var_names.to_numpy(dtype=str); meta.file.close()
selected=(obs.cell_line.astype(str)==args.cell_line).to_numpy()&(obs.time.astype(float)==args.time).to_numpy()
mito=np.char.startswith(np.char.upper(genes),'MT-')
handle=h5py.File(args.input,'r'); X=handle['X']; indptr=X['indptr'][:].astype(np.int64)
n_rows,n_genes=X.attrs['shape']

def chunks():
    for start in range(0,n_rows,args.chunk):
        stop=min(start+args.chunk,n_rows); rows=np.flatnonzero(selected[start:stop])
        if not len(rows):
            continue
        lo,hi=indptr[start],indptr[stop]
        block=sparse.csr_matrix((X['data'][lo:hi].astype(np.float32),X['indices'][lo:hi],indptr[start:stop+1]-lo),
                                shape=(stop-start,n_genes))
        yield start+rows,block[rows]

# Pass 1: per-cell metrics and per-gene detection.
metrics=[]; detected=np.zeros(n_genes,np.int64)
for positions,block in chunks():
    total=np.asarray(block.sum(1)).ravel()
    metrics.append(pd.DataFrame({'position':positions,'total_counts':total,'n_genes':np.diff(block.indptr),
                                 'pct_mito':100*np.asarray(block[:,mito].sum(1)).ravel()/np.maximum(total,1)}))
    detected+=np.bincount(block.indices,minlength=n_genes)
metrics=pd.concat(metrics,ignore_index=True)
def bounds(values,nmads):
    median=np.median(values); mad=1.4826*np.median(np.abs(values-median))
    return median-nmads*mad,median+nmads*mad
keep=np.ones(len(metrics),bool)
for key in ['total_counts','n_genes']:
    lo,hi=bounds(np.log1p(metrics[key]),args.nmads); keep&=(np.log1p(metrics[key])>=lo)&(np.log1p(metrics[key])<=hi)
mito_cap=bounds(metrics.pct_mito,args.mito_nmads)[1]; keep&=(metrics.pct_mito<=mito_cap).to_numpy()&(metrics.total_counts>0).to_numpy()
kept_positions=set(metrics.position[keep])
gene_mask=detected>=args.min_cell_fraction*keep.sum()

# Pass 2: sum log-normalised profiles and detection counts per group.
labels=obs.iloc[metrics.position[keep]][args.keys].astype(str)
key_frame=labels.drop_duplicates().reset_index(drop=True)
key_index={tuple(r):i for i,r in enumerate(key_frame.itertuples(index=False))}
cell_group=dict(zip(metrics.position[keep],[key_index[tuple(r)] for r in labels.itertuples(index=False)]))
sums=np.zeros((len(key_frame),gene_mask.sum()),np.float64); hits=np.zeros_like(sums,dtype=np.int32)
cells=np.zeros(len(key_frame),np.int64)
for positions,block in chunks():
    mask=np.array([p in kept_positions for p in positions])
    if not mask.any():
        continue
    block=block[mask][:,gene_mask].tocsr(); group=np.array([cell_group[p] for p in positions[mask]])
    total=np.asarray(block.sum(1)).ravel()
    # Normalise with full-library totals, not gene-filtered totals.
    full=metrics.set_index('position').total_counts.loc[positions[mask]].to_numpy()
    lognorm=(sparse.diags(1e4/full)@block).tocsr(); lognorm.data=np.log1p(lognorm.data)
    indicator=sparse.csr_matrix((np.ones(len(group)),(group,np.arange(len(group)))),shape=(len(key_frame),len(group)))
    sums+=(indicator@lognorm).toarray(); hits+=(indicator@(block>0).astype(np.float64)).toarray().astype(np.int32)
    cells+=np.bincount(group,minlength=len(key_frame))
handle.close()
args.out.parent.mkdir(parents=True,exist_ok=True)
np.savez_compressed(args.out,sums=sums.astype(np.float32),detected=hits,cells=cells,genes=genes[gene_mask],
                    keys=np.array(args.keys),groups=key_frame.to_numpy(dtype=str))
report={'input':str(args.input),'output':str(args.out),'cell_line':args.cell_line,'time_hours':args.time,
        'cells_selected':int(selected.sum()),'cells_kept':int(keep.sum()),
        'thresholds':{'log_nmads':args.nmads,'pct_mito_cap':float(mito_cap)},
        'genes_kept':int(gene_mask.sum()),'groups':int(len(key_frame)),'group_keys':args.keys,
        'normalisation':'per-cell 1e4 by full library size, log1p; summed (divide by cells for means)'}
args.out.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
