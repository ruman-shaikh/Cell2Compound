"""Label-agnostic cell QC plus replicate/plate structure checks on raw counts."""
import argparse
import json
from pathlib import Path
import anndata as ad
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('input',type=Path)
parser.add_argument('--drug-key',default='perturbation')
parser.add_argument('--control',default='control')
parser.add_argument('--nmads',type=float,default=5.)
parser.add_argument('--mito-nmads',type=float,default=3.)
parser.add_argument('--max-mito',type=float,default=None,help='optional fixed cap; sci-Plex nuclei have high baseline mito fraction')
parser.add_argument('--out',type=Path,default=Path('data/sciplex_qc.h5ad'))
args=parser.parse_args()
data=ad.read_h5ad(args.input)
x=data.X.tocsr() if sparse.issparse(data.X) else sparse.csr_matrix(data.X)
if (x.data<0).any() or not np.allclose(x.data,np.round(x.data)):
    raise ValueError('QC expects raw counts')
obs=data.obs
mito=data.var_names.str.upper().str.startswith('MT-')
obs['total_counts']=np.asarray(x.sum(axis=1)).ravel()
obs['n_genes']=np.diff(x.indptr)
obs['pct_mito']=100*np.asarray(x[:,mito].sum(axis=1)).ravel()/obs.total_counts

def mad_outlier(values):
    values=np.log1p(values); median=np.median(values)
    mad=np.median(np.abs(values-median))*1.4826
    return (values<median-args.nmads*mad)|(values>median+args.nmads*mad)
mito_median=float(obs.pct_mito.median())
mito_cap=mito_median+args.mito_nmads*1.4826*float(np.median(np.abs(obs.pct_mito-mito_median)))
if args.max_mito is not None:
    mito_cap=min(mito_cap,args.max_mito)
# Thresholds use all cells, never drug labels; drug-specific loss is reported, not tuned.
flags=pd.DataFrame({'counts_outlier':mad_outlier(obs.total_counts),
                    'genes_outlier':mad_outlier(obs.n_genes),
                    'high_mito':obs.pct_mito>mito_cap},index=obs.index)
keep=~flags.any(axis=1).to_numpy()
drug=obs[args.drug_key].astype(str)
loss=(~pd.Series(keep,index=obs.index)).groupby(drug).mean().sort_values(ascending=False)

# Replicate/plate structure: is a compound measured on one plate, so plate effects confound it?
treated=obs.loc[keep & (drug!=args.control).to_numpy()]
plates_per_drug=treated.groupby(args.drug_key,observed=True).plate.nunique()
counts=x[keep]; sums=np.asarray(counts.sum(axis=1)).ravel()
lognorm=sparse.diags(1e4/sums)@counts; lognorm=lognorm.tocsr(); lognorm.data=np.log1p(lognorm.data)
kept_obs=obs.loc[keep]
control_mask=(kept_obs[args.drug_key].astype(str)==args.control).to_numpy()
expressed=np.asarray((counts[control_mask]>0).mean(axis=0)).ravel()>=.05
def mean_profile(mask):
    return np.asarray(lognorm[np.flatnonzero(mask)][:,expressed].mean(axis=0)).ravel()
reps=sorted(kept_obs.replicate.astype(str).unique())
control_profiles={r:mean_profile(control_mask&(kept_obs.replicate.astype(str)==r).to_numpy()) for r in reps}
rep_corr=float(np.corrcoef(*control_profiles.values())[0,1]) if len(reps)==2 else None
# Split-half control noise: the floor below which no response model can be judged.
rng=np.random.default_rng(0); idx=rng.permutation(np.flatnonzero(control_mask)); half=len(idx)//2
split_half=[mean_profile(np.isin(np.arange(len(kept_obs)),part)) for part in (idx[:half],idx[half:])]
control_plates=kept_obs.loc[control_mask].plate.astype(str).value_counts()

filtered=data[keep].copy()
nonzero=np.asarray((filtered.X>0).sum(axis=0)).ravel()>0
filtered=filtered[:,nonzero].copy()
filtered.write_h5ad(args.out,compression='gzip')
out=Path('results/qc'); out.mkdir(parents=True,exist_ok=True)
fig,axes=plt.subplots(1,3,figsize=(12,3.5))
for ax,key,log in zip(axes,['total_counts','n_genes','pct_mito'],[True,True,False]):
    values=np.log10(obs[key]) if log else obs[key]
    ax.hist(values[keep],bins=80,alpha=.7,label='kept'); ax.hist(values[~keep],bins=80,alpha=.7,label='removed')
    ax.set_xlabel(f'log10 {key}' if log else key); ax.set_ylabel('cells')
axes[0].legend(); fig.tight_layout(); fig.savefig(out/'qc_distributions.png',dpi=120)
report={'input':str(args.input),'output':str(args.out),'cells_in':int(len(obs)),'cells_kept':int(keep.sum()),
        'genes_in':int(data.n_vars),'genes_kept_nonzero':int(nonzero.sum()),'mito_genes':int(mito.sum()),
        'thresholds':{'nmads_log':args.nmads,'mito_nmads':args.mito_nmads,'pct_mito_cap':mito_cap},
        'removed_by_flag':{k:int(v) for k,v in flags.sum().items()},
        'median':{k:float(obs.loc[keep,k].median()) for k in ['total_counts','n_genes','pct_mito']},
        'control_cells_kept':int(control_mask.sum()),
        'control_cells_by_replicate':{r:int((control_mask&(kept_obs.replicate.astype(str)==r).to_numpy()).sum()) for r in reps},
        'control_plates':int(len(control_plates)),
        'control_replicate_profile_pearson':rep_corr,
        'control_split_half_rmse':float(np.sqrt(np.mean((split_half[0]-split_half[1])**2))),
        'control_profile_genes':int(expressed.sum()),
        'plates_per_compound':{'min':int(plates_per_drug.min()),'median':float(plates_per_drug.median()),'max':int(plates_per_drug.max())},
        'highest_compound_loss':{k:round(float(v),3) for k,v in loss.head(10).items()},
        'limitations':['Thresholds are global; cytotoxic compounds can lose more cells, which biases their mean response toward survivors.',
                       'Split-half RMSE is computed on control-expressed genes and is not directly comparable to model RMSE on selected genes.']}
(out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
