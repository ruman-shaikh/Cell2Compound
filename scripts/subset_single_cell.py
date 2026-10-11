"""Bounded, stratified backed-mode subsetting; requires inspected metadata keys."""
import argparse
import json
from pathlib import Path
import anndata as ad
import numpy as np

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('input',type=Path)
parser.add_argument('--cell-key',required=True)
parser.add_argument('--cell-line',required=True)
parser.add_argument('--drug-key',required=True)
parser.add_argument('--dose-key',required=True)
parser.add_argument('--per-group',type=int,default=30)
parser.add_argument('--control',default=None,help='control label; capped separately via --control-cap')
parser.add_argument('--control-cap',type=int,default=1500)
parser.add_argument('--stratify-key',default=None,help='sample evenly across e.g. replicate within each group')
parser.add_argument('--max-cells',type=int,default=30000)
parser.add_argument('--time',type=float,default=24.)
parser.add_argument('--out',type=Path,default=Path('data/sciplex_subset.h5ad'))
args=parser.parse_args()
data=ad.read_h5ad(args.input,backed='r')
keys=[args.cell_key,args.drug_key,args.dose_key]
if not set(keys).issubset(data.obs.columns):
    raise ValueError(f'Missing metadata keys; available: {list(data.obs.columns)}')
obs=data.obs.copy(); obs['_position']=np.arange(len(obs))
eligible=obs.loc[obs[args.cell_key].astype(str)==args.cell_line]
if 'time' in eligible:
    eligible=eligible.loc[eligible['time'].astype(float)==args.time]
if eligible.empty:
    raise ValueError('Requested cell line has no observations')
rng=np.random.default_rng(42); selected=[]
def sample(group,size):
    if args.stratify_key is None:
        return list(rng.choice(group['_position'].to_numpy(),size=min(size,len(group)),replace=False))
    strata=[s['_position'].to_numpy() for _,s in group.groupby(args.stratify_key,observed=True)]
    share=size//max(len(strata),1); picked=[]
    for positions in strata:
        picked.extend(rng.choice(positions,size=min(share,len(positions)),replace=False))
    return picked
for (drug,_),group in eligible.groupby([args.drug_key,args.dose_key],observed=True,dropna=False):
    cap=args.control_cap if args.control is not None and str(drug)==args.control else args.per_group
    selected.extend(sample(group,cap))
if len(selected)>args.max_cells:
    raise ValueError(f'{len(selected)} cells exceed cap; reduce per-group rather than dropping groups')
subset=data[np.sort(selected),:].to_memory()
data.file.close()
args.out.parent.mkdir(exist_ok=True,parents=True)
subset.write_h5ad(args.out,compression='gzip')
report={'input':str(args.input),'output':str(args.out),'shape':list(subset.shape),
        'cell_key':args.cell_key,'cell_line':args.cell_line,'drug_key':args.drug_key,
        'dose_key':args.dose_key,'per_group_cap':args.per_group,'seed':42,
        'control':args.control,'control_cap':args.control_cap,'stratify_key':args.stratify_key,
        'control_cells':int((subset.obs[args.drug_key].astype(str)==args.control).sum()) if args.control else None,
        'time_hours':args.time,
        'normalisation':'none; preserve input values pending count-scale inspection'}
args.out.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
