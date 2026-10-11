"""Does an ESM-2 target embedding help predict pChEMBL for compound-target pairs?
Compares target representations (none / one-hot / ESM / shuffled ESM) under:
  scaffold   - held-out compound scaffolds, all targets seen in training
  loto       - leave-one-target-out; test compounds may appear in training with other targets
  loto_strict- leave-one-target-out and all test compounds removed from training"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit
from cell2compound.chemistry import featurize

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--data',type=Path,default=Path('data/bioactivity'))
parser.add_argument('--esm',type=Path,default=Path('results/esm_panel'))
parser.add_argument('--seeds',type=int,nargs='+',default=[0,1,2])
parser.add_argument('--device',choices=['cpu','cuda'],default='cuda')
parser.add_argument('--out',type=Path,default=Path('results/target_aware'))
args=parser.parse_args()
if args.device=='cuda' and not torch.cuda.is_available():
    raise RuntimeError('CUDA requested but unavailable; run outside sandbox or pass --device cpu')
args.out.mkdir(parents=True,exist_ok=True)

pairs=pd.read_csv(args.data/'pairs.csv')
structures=pd.DataFrame({'compound_id':pairs.parent_smiles.unique(),'smiles':pairs.parent_smiles.unique()})
descriptors,fingerprints,rejected=featurize(structures)
pairs=pairs[pairs.parent_smiles.isin(descriptors.compound_id)].reset_index(drop=True)
row_of={s:i for i,s in enumerate(descriptors.compound_id)}
compound_row=pairs.parent_smiles.map(row_of).to_numpy()
scaffold=descriptors.scaffold.to_numpy()[compound_row]
genes=sorted(pairs.gene.unique()); gene_index=pairs.gene.map({g:i for i,g in enumerate(genes)}).to_numpy()
manifest=json.loads((args.esm/'manifest.json').read_text())
esm={t['id']:np.load(args.esm/t['file'])['mean_embedding'] for t in manifest['targets']}
esm=np.stack([esm[g] for g in genes])
y=pairs.pchembl.to_numpy(np.float32)
# Fixed derangement: every target receives another target's embedding.
rng=np.random.default_rng(0)
while True:
    permutation=rng.permutation(len(genes))
    if not (permutation==np.arange(len(genes))).any():
        break

def target_features(variant,train_targets):
    if variant=='none':
        return np.zeros((len(genes),0),np.float32)
    if variant=='one_hot':
        return np.eye(len(genes),dtype=np.float32)
    table=esm if variant=='esm' else esm[permutation]
    # Standardise with training targets only; the held-out target's embedding never sets the scale.
    mean=table[train_targets].mean(0); sd=table[train_targets].std(0)+1e-6
    return ((table-mean)/sd).astype(np.float32)

def design(rows,variant,train_targets):
    return np.c_[fingerprints[compound_row[rows]].astype(np.float32),target_features(variant,train_targets)[gene_index[rows]]]

def fit_predict(train,valid,test,variant,seed):
    train_targets=np.unique(gene_index[train])
    xs=[torch.as_tensor(design(r,variant,train_targets),device=args.device) for r in (train,valid,test)]
    ys=[torch.as_tensor(y[r],device=args.device) for r in (train,valid)]
    torch.manual_seed(seed)
    model=torch.nn.Sequential(torch.nn.Linear(xs[0].shape[1],512),torch.nn.ReLU(),torch.nn.Dropout(.3),
                              torch.nn.Linear(512,256),torch.nn.ReLU(),torch.nn.Dropout(.3),torch.nn.Linear(256,1)).to(args.device)
    opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-3)
    offset=float(ys[0].mean())
    best,best_state,stale=np.inf,None,0
    for epoch in range(200):
        model.train()
        for batch in torch.randperm(len(train),device=args.device).split(512):
            opt.zero_grad(); loss=torch.nn.functional.mse_loss(model(xs[0][batch]).squeeze(1)+offset,ys[0][batch]); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            val=torch.nn.functional.mse_loss(model(xs[1]).squeeze(1)+offset,ys[1]).item()
        if val<best-1e-4:
            best,stale=val,0; best_state={k:v.clone() for k,v in model.state_dict().items()}
        else:
            stale+=1
            if stale>=10:
                break
    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        return (model(xs[2]).squeeze(1)+offset).cpu().numpy(),epoch+1

def metrics(rows,prediction):
    error=prediction-y[rows]
    per_target=[np.corrcoef(prediction[gene_index[rows]==g],y[rows][gene_index[rows]==g])[0,1]
                for g in np.unique(gene_index[rows]) if (gene_index[rows]==g).sum()>=20]
    return {'rmse':float(np.sqrt(np.mean(error**2))),'mae':float(np.mean(np.abs(error))),
            'r2':float(1-np.sum(error**2)/np.sum((y[rows]-y[rows].mean())**2)),
            'pearson':float(np.corrcoef(prediction,y[rows])[0,1]),'mean_within_target_pearson':float(np.nanmean(per_target)) if per_target else None}

def group_split(rows,groups,fraction,seed):
    a,b=next(GroupShuffleSplit(n_splits=1,test_size=fraction,random_state=seed).split(rows,groups=groups[rows]))
    return rows[a],rows[b]

results=[]
everything=np.arange(len(pairs))
for seed in args.seeds:
    trainval,test=group_split(everything,scaffold,.1,seed)
    train,valid=group_split(trainval,scaffold,.1,seed+100)
    for variant in ['none','one_hot','esm','esm_shuffled']:
        prediction,epochs=fit_predict(train,valid,test,variant,seed)
        results.append({'setting':'scaffold','seed':seed,'held_out':'scaffolds','variant':variant,'epochs':epochs,
                        'n_test':len(test),**metrics(test,prediction)})
        print(results[-1]['setting'],seed,variant,round(results[-1]['rmse'],3),flush=True)
for g,gene in enumerate(genes):
    test=np.flatnonzero(gene_index==g)
    for setting in ['loto','loto_strict']:
        pool=np.flatnonzero(gene_index!=g)
        if setting=='loto_strict':
            pool=pool[~np.isin(compound_row[pool],compound_row[test])]
        train,valid=group_split(pool,scaffold,.1,0)
        # Reference: the compound's mean pChEMBL on the training targets (shared-compound setting only).
        for variant in ['none','esm','esm_shuffled']:
            prediction,epochs=fit_predict(train,valid,test,variant,0)
            results.append({'setting':setting,'seed':0,'held_out':gene,'variant':variant,'epochs':epochs,
                            'n_test':len(test),'n_train':len(train),'coverage':1.,**metrics(test,prediction)})
            if setting=='loto':
                # Same predictions restricted to compounds the comparator can score.
                shared=np.isin(compound_row[test],np.unique(compound_row[np.flatnonzero(gene_index!=g)]))
                if shared.sum()>=20:
                    results.append({'setting':'loto_shared_compounds','seed':0,'held_out':gene,'variant':variant,
                                    'n_test':int(shared.sum()),'coverage':1.,**metrics(test[shared],prediction[shared])})
        print(setting,gene,{r['variant']:round(r['rmse'],3) for r in results[-3:]},flush=True)
    # Comparator: the compound's mean pChEMBL on the training targets. Scored on ALL test pairs
    # (unseen compounds fall back to the training mean) so it is comparable with the models,
    # and again on only the compounds that have other-target data, which is an easier subset.
    pool=np.flatnonzero(gene_index!=g)
    prior=pd.Series(y[pool]).groupby(compound_row[pool]).mean()
    known=np.isin(compound_row[test],prior.index)
    filled=np.where(known,prior.reindex(compound_row[test]).to_numpy(),y[pool].mean())
    results.append({'setting':'loto','seed':0,'held_out':gene,'variant':'compound_mean_other_targets',
                    'n_test':len(test),'coverage':float(known.mean()),**metrics(test,filled)})
    if known.sum()>=20:
        results.append({'setting':'loto_shared_compounds','seed':0,'held_out':gene,
                        'variant':'compound_mean_other_targets','n_test':int(known.sum()),'coverage':1.,
                        **metrics(test[known],prior.loc[compound_row[test[known]]].to_numpy())})

frame=pd.DataFrame(results); frame.to_csv(args.out/'results.csv',index=False)
summary=frame.groupby(['setting','variant'])[['rmse','r2','mean_within_target_pearson']].agg(['mean','std']).round(4)
settings=['loto','loto_strict']
loto=frame[frame.setting!='scaffold'].pivot_table(index='held_out',columns=['setting','variant'],values='rmse').round(3)
loto.to_csv(args.out/'loto_rmse_by_target.csv')
report={'pairs':len(pairs),'structures':int(len(descriptors)),'targets':genes,
        'esm_model':manifest['model'],'esm_pooling':'mean over residues (overlapping 1022-residue windows for long sequences)',
        'model':'MLP 512-256, dropout 0.3, AdamW 1e-3, wd 1e-3, batch 512, early stopping on scaffold-grouped validation',
        'summary':{f'{s}/{v}':{f'{m}_{a}':val for (m,a),val in row.items()} for (s,v),row in summary.iterrows()},
        'esm_beats_shuffled_targets':{s:int((loto[(s,'esm')]<loto[(s,'esm_shuffled')]).sum()) for s in settings},
        'esm_beats_none_targets':{s:int((loto[(s,'esm')]<loto[(s,'none')]).sum()) for s in settings},
        'esm_beats_compound_mean_targets':int((loto[('loto_shared_compounds','esm')]
                                               <loto[('loto_shared_compounds','compound_mean_other_targets')]).sum()),
        'comparator_note':('compound_mean_other_targets needs a compound to have data at another target; '
                           'setting loto scores it on all pairs with a training-mean fallback, '
                           'setting loto_shared_compounds scores every method on only the covered compounds.'),
        'limitations':['Mean-pooled full-length embeddings; no pocket-level features.',
                       '16 targets in two families: leave-one-target-out tests transfer within families, not to novel folds.',
                       'ChEMBL pChEMBL mixes IC50/Ki/Kd across assays; median per pair reduces but does not remove assay noise.']}
(args.out/'summary.json').write_text(json.dumps(report,indent=2,default=str)+'\n')
print(summary.to_string()); print(loto.to_string())
