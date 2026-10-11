"""Control-relative pseudobulk response prediction for held-out compounds over several split seeds."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from cell2compound.chemistry import featurize
from cell2compound.compounds import load_source, map_observed_names
from cell2compound.expression import fit_mlp, predict_mlp, product_kernel, response_metrics, tanimoto
from cell2compound.splits import compound_split

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('input',type=Path,help='pseudobulk .npz from scripts/pseudobulk.py')
parser.add_argument('--control',default='control')
parser.add_argument('--control-matching',choices=['plate','pooled'],default='pooled',help='pooled chosen: lower replicate disagreement over all groups (0.091 vs 0.099)')
parser.add_argument('--genes',type=int,default=1000)
parser.add_argument('--min-detection',type=float,default=.05,help='gene detection fraction in training+control cells')
parser.add_argument('--min-cells',type=int,default=10,help='per replicate')
parser.add_argument('--top-k',type=int,default=50)
parser.add_argument('--seeds',type=int,nargs='+',default=[42,43,44,45,46])
parser.add_argument('--device',choices=['cpu','cuda'],default='cuda')
parser.add_argument('--source',type=Path,default=Path('data/raw/sciplex_compounds.csv'))
parser.add_argument('--out',type=Path,default=Path('results/expression'))
args=parser.parse_args()
if args.device=='cuda' and not torch.cuda.is_available():
    raise RuntimeError('CUDA requested but unavailable; run outside sandbox or pass --device cpu')

bulk=np.load(args.input)
table=pd.DataFrame(bulk['groups'],columns=bulk['keys']); table['cells']=bulk['cells']
table['dose_value']=table.dose_value.astype(float)
sums=bulk['sums'].astype(np.float64); detected=bulk['detected']; gene_names=bulk['genes']
means=sums/table.cells.to_numpy()[:,None]
is_control=(table.perturbation==args.control).to_numpy()
pooled_control=sums[is_control].sum(0)/table.cells[is_control].sum()
plate_control={plate:sums[is_control&(table.plate==plate).to_numpy()].sum(0)/table.cells[is_control&(table.plate==plate).to_numpy()].sum()
               for plate in table.plate[is_control].unique()}
def reference(plate):
    return plate_control[plate] if args.control_matching=='plate' else pooled_control

mapping,unmatched=map_observed_names(table.perturbation[~is_control].unique(),load_source(args.source))
structures=mapping[['parent_smiles']].drop_duplicates().rename(columns={'parent_smiles':'smiles'})
structures['compound_id']=structures.smiles
descriptors,fingerprints,rejected=featurize(structures)
if len(rejected):
    raise ValueError(f'Parent structures failed RDKit parsing: {rejected.compound_id.tolist()}')
structure_index={s:i for i,s in enumerate(descriptors.compound_id)}
mapping['canonical_smiles']=mapping.parent_smiles.map(lambda s:descriptors.canonical_smiles.iloc[structure_index[s]])
mapping['structure_row']=mapping.parent_smiles.map(structure_index)
by_name=mapping.set_index('observed_name')

# One row per compound/dose. Response = mean over replicates of (treated - matched control).
rows,too_few=[],[]
for (drug,dose),group in table[~is_control].groupby(['perturbation','dose_value']):
    if drug not in by_name.index:
        continue
    per_rep=group.groupby('replicate')
    rep_delta,rep_cells,rep_rows=[],[],[]
    for rep,part in per_rep:
        n=part.cells.sum()
        if n<args.min_cells:
            continue
        treated=sums[part.index].sum(0)/n
        control=np.average([reference(pl) for pl in part.plate],axis=0,weights=part.cells)
        rep_delta.append(treated-control); rep_cells.append(int(n)); rep_rows.append(part.index.to_numpy())
    if len(rep_delta)<2:
        too_few.append({'compound':drug,'dose':dose,'cells_per_replicate':per_rep.cells.sum().to_dict()}); continue
    rows.append({'compound':drug,'dose':dose,'rep_delta':np.stack(rep_delta),'rep_cells':rep_cells,
                 'table_rows':np.concatenate(rep_rows),
                 'canonical_smiles':by_name.canonical_smiles[drug],'fp_row':by_name.structure_row[drug]})
groups=pd.DataFrame(rows)
canonical=groups.canonical_smiles.to_numpy(); doses=groups.dose.to_numpy()
fp=fingerprints[groups.fp_row.to_numpy()]
features=np.c_[fp,np.log10(doses)]
full_delta=np.stack([d.mean(0) for d in groups.rep_delta])
rep_difference=np.stack([d[0]-d[1] for d in groups.rep_delta])
control_rows=np.flatnonzero(is_control)

def nearest_neighbour(fit,test,y,k=5):
    """Tanimoto-weighted mean response of the k most similar fitted compounds at the same dose."""
    sims=tanimoto(fp[test],fp[fit]); out=[]
    for row,similarity in zip(test,sims):
        same=doses[fit]==doses[row]
        candidates=np.flatnonzero(same) if same.any() else np.arange(len(fit))
        top=candidates[np.argsort(-similarity[candidates])[:k]]
        weights=similarity[top]+1e-6
        out.append((weights[:,None]*y[fit][top]).sum(0)/weights.sum())
    return np.stack(out)

all_results=[]; per_compound=None
for seed_number,seed in enumerate(args.seeds):
    splits=compound_split(canonical,seed=seed)
    tr=np.flatnonzero(splits=='train'); va=np.flatnonzero(splits=='validation'); te=np.flatnonzero(splits=='test')
    fit=np.r_[tr,va]
    # Genes: detected in enough training+control cells, then highest training-response variance.
    reference_rows=np.r_[np.concatenate(groups.table_rows.iloc[tr].to_list()),control_rows]
    detection=detected[reference_rows].sum(0)/table.cells.to_numpy()[reference_rows].sum()
    eligible=np.flatnonzero(detection>=args.min_detection)
    genes=np.sort(eligible[np.argsort(full_delta[tr][:,eligible].var(0))[-args.genes:]])
    y=full_delta[:,genes]
    predictions={'zero_response':np.zeros((len(te),len(genes))),
                 'training_mean_response':np.broadcast_to(y[fit].mean(0),(len(te),len(genes)))}
    dose_mean={d:y[fit][doses[fit]==d].mean(0) for d in np.unique(doses[fit])}
    predictions['dose_mean_response']=np.stack([dose_mean[d] for d in doses[te]])
    predictions['tanimoto_knn']=nearest_neighbour(fit,te,y)

    ridge_val=[]
    for alpha in [1.,10.,100.,1e3,1e4,1e5,1e6]:
        model=make_pipeline(StandardScaler(),Ridge(alpha=alpha)).fit(features[tr],y[tr])
        ridge_val.append((float(np.mean((model.predict(features[va])-y[va])**2)),alpha))
    _,alpha=min(ridge_val)
    predictions['ridge']=make_pipeline(StandardScaler(),Ridge(alpha=alpha)).fit(features[fit],y[fit]).predict(features[te])

    # Tanimoto-kernel ridge: the similarity structure the nearest-neighbour baseline exploits,
    # fitted jointly over all genes instead of averaging k neighbours.
    kernel_val=[]
    for bandwidth in [.5,1.,2.,1e6]:
        train_kernel=product_kernel(fp[tr],doses[tr],fp[tr],doses[tr],bandwidth)
        valid_kernel=product_kernel(fp[va],doses[va],fp[tr],doses[tr],bandwidth)
        for alpha in [1e-3,1e-2,1e-1,1.]:
            model=KernelRidge(alpha=alpha,kernel='precomputed').fit(train_kernel,y[tr])
            kernel_val.append((float(np.mean((model.predict(valid_kernel)-y[va])**2)),alpha,bandwidth))
    _,kernel_alpha,bandwidth=min(kernel_val)
    fit_kernel=product_kernel(fp[fit],doses[fit],fp[fit],doses[fit],bandwidth)
    test_kernel=product_kernel(fp[te],doses[te],fp[fit],doses[fit],bandwidth)
    predictions['tanimoto_kernel_ridge']=KernelRidge(alpha=kernel_alpha,kernel='precomputed').fit(fit_kernel,y[fit]).predict(test_kernel)

    mlp_val=[]
    for hidden in [64,256]:
        for dropout in [.2,.5]:
            for weight_decay in [1e-2,1e-1]:
                _,val,epoch=fit_mlp(features[tr],y[tr],hidden,dropout,weight_decay,3000,args.device,seed,features[va],y[va],patience=150)
                mlp_val.append((val,hidden,dropout,weight_decay,epoch))
    _,hidden,dropout,weight_decay,epoch=min(mlp_val)
    model,_,_=fit_mlp(features[fit],y[fit],hidden,dropout,weight_decay,epoch,args.device,seed)
    predictions['mlp']=predict_mlp(model,features[te],args.device)

    # Replicate noise floor: rep1 and rep2 are separate plates/experiments. With independent,
    # equal-variance errors, RMSE(replicate mean vs truth) ~ RMSE(rep1 - rep2)/2.
    floor=float(np.sqrt(np.mean(rep_difference[te][:,genes]**2))/2)
    result={'seed':seed,'train_groups':len(tr),'validation_groups':len(va),'test_groups':len(te),
            'test_compounds':int(len(set(canonical[te]))),'ridge_alpha':alpha,'ridge_validation_mse':dict((str(a),v) for v,a in ridge_val),
            'kernel_ridge':{'alpha':kernel_alpha,'dose_bandwidth':bandwidth,
                            'validation_mse':{f'a{a}_b{b}':v for v,a,b in kernel_val}},
            'mlp':{'hidden':hidden,'dropout':dropout,'weight_decay':weight_decay,'epochs':epoch,
                   'validation_mse':{f'h{h}_d{d}_wd{w}':v for v,h,d,w,_ in mlp_val}},
            'replicate_noise_floor_rmse':floor,'replicate_floor_groups':len(te),
            'test':{name:response_metrics(y[te],pred,args.top_k) for name,pred in predictions.items()}}
    all_results.append(result)
    print(json.dumps({'seed':seed,'floor':floor,**{k:round(v['rmse'],4) for k,v in result['test'].items()}}))
    if seed_number==0:
        args.out.mkdir(parents=True,exist_ok=True)
        split_frame=groups.drop(columns=['rep_delta','table_rows','fp_row']).assign(split=splits,cells=groups.rep_cells.map(sum))
        split_frame.to_csv(args.out/'splits.csv',index=False)
        np.savez_compressed(args.out/'predictions.npz',observed=y[te],genes=gene_names[genes],
                            test_compound=groups.compound.iloc[te].to_numpy(dtype=str),test_dose=doses[te],**predictions)
        floor_all_groups=float(np.sqrt(np.mean(rep_difference[:,genes]**2))/2)
        per_compound=split_frame.iloc[te].copy()
        for name in ['training_mean_response','dose_mean_response','tanimoto_knn','ridge','tanimoto_kernel_ridge','mlp']:
            per_compound[f'{name}_rmse']=np.sqrt(np.mean((y[te]-predictions[name])**2,axis=1))
        per_compound['response_norm']=np.sqrt(np.mean(y[te]**2,axis=1))
        per_compound.sort_values('response_norm',ascending=False).to_csv(args.out/'per_compound_errors.csv',index=False)
        # Observed response magnitude for every compound/dose (descriptive; no model involved).
        split_frame.assign(response_rms=np.sqrt(np.mean(y**2,axis=1)),
                           replicate_difference_rms=np.sqrt(np.mean(rep_difference[:,genes]**2,axis=1))
                           ).to_csv(args.out/'responses.csv',index=False)

models=list(all_results[0]['test'])
summary={name:{metric:{'mean':float(np.mean([r['test'][name][metric] for r in all_results])),
                       'sd':float(np.std([r['test'][name][metric] for r in all_results],ddof=1))}
               for metric in all_results[0]['test'][name]} for name in models}
mapping.drop(columns=['structure_row']).to_csv(args.out/'compound_mapping.csv',index=False)
report={'input':str(args.input),'scale':'raw counts -> per-cell 1e4 normalisation -> log1p -> pseudobulk mean',
        'control_matching':args.control_matching,'min_detection':args.min_detection,
        'groups':len(groups),'compounds':int(groups.compound.nunique()),'unique_structures':int(len(set(canonical))),
        'genes':args.genes,'control_cells':int(table.cells[is_control].sum()),'treated_cells':int(table.cells[~is_control].sum()),'features':'Morgan r2 2048 bits (salt-stripped parent) + log10 dose (nM)',
        'unmatched_names':unmatched,'groups_below_min_cells':too_few,
        'replicate_noise_floor_all_groups_first_seed_genes':floor_all_groups,
        'summary_over_seeds':summary,'per_seed':all_results,
        'limitations':['Predicts group means, not single-cell distributions.',
                       'Pooled controls by default; plate-matched ablation had higher replicate disagreement.',
                       'Cytotoxic compounds lose cells at QC; their responses describe survivors.',
                       'Seeds vary only the compound split; models are not independent replications.',
                       'No disease-rescue or significance claims.']}
(args.out/'metrics.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(summary,indent=1))
