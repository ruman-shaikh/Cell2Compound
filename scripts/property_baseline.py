"""Measured solubility (ESOL) prediction with scaffold-disjoint splits over several seeds.

Acyclic molecules share the empty Murcko scaffold. Treating them as one group is conservative
but makes split sizes wildly unbalanced (one group of ~90 molecules); --empty-scaffold singleton
gives each acyclic structure its own group. Both are reported."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from cell2compound.chemistry import featurize

TARGET='measured log solubility in mols per litre'
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--raw',type=Path,default=Path('data/raw/delaney.csv'))
parser.add_argument('--seeds',type=int,nargs='+',default=[42,43,44,45,46])
parser.add_argument('--empty-scaffold',choices=['group','singleton'],default='singleton')
parser.add_argument('--out',type=Path,default=Path('results/property'))
args=parser.parse_args()
args.out.mkdir(parents=True,exist_ok=True)

source=pd.read_csv(args.raw)
frame=pd.DataFrame({'compound_id':np.arange(len(source)).astype(str),'smiles':source.smiles.str.strip()})
descriptors,fingerprints,rejected=featurize(frame)
if len(rejected):
    raise ValueError('Resolve rejected benchmark molecules before training')
descriptors['target']=source[TARGET].to_numpy()
descriptors['row']=np.arange(len(source))
# Duplicate structures stay together and their repeated measurements are averaged.
unique=descriptors.groupby('canonical_smiles',sort=False).agg(
    row=('row','first'),target=('target','mean'),scaffold=('scaffold','first'),
    measurements=('target','size')).reset_index()
indices=unique.row.to_numpy()
features=descriptors[['molecular_weight','logp','tpsa','hbd','hba','rotatable_bonds']].to_numpy()[indices]
fingerprints=fingerprints[indices]
y=unique.target.to_numpy()
acyclic=unique.scaffold==''
groups=unique.scaffold.to_numpy().copy()
if args.empty_scaffold=='singleton':
    groups[acyclic.to_numpy()]=unique.canonical_smiles[acyclic].to_numpy()
unique['group']=groups

def split(rows,fraction,seed):
    a,b=next(GroupShuffleSplit(n_splits=1,test_size=fraction,random_state=seed).split(rows,groups=groups[rows]))
    return rows[a],rows[b]

def candidates():
    items=[('mean',DummyRegressor(),features)]
    for alpha in [1.,10.,100.]:
        items.append((f'ridge_descriptors_{alpha}',make_pipeline(StandardScaler(),Ridge(alpha=alpha)),features))
    for alpha in [1.,10.]:
        items.append((f'ridge_fingerprints_{alpha}',make_pipeline(StandardScaler(with_mean=False),Ridge(alpha=alpha)),fingerprints))
    for leaf in [1,2,5]:
        items.append((f'rf_descriptors_leaf{leaf}',RandomForestRegressor(n_estimators=300,min_samples_leaf=leaf,random_state=0,n_jobs=2),features))
        items.append((f'rf_fingerprints_leaf{leaf}',RandomForestRegressor(n_estimators=300,min_samples_leaf=leaf,random_state=0,n_jobs=2),fingerprints))
    return items

everything=np.arange(len(unique))
rows,predictions=[],[]
for seed in args.seeds:
    trainval,test=split(everything,.2,seed)
    train,valid=split(trainval,.2,seed+100)
    assert not (set(groups[train])&set(groups[valid])) and not (set(groups[trainval])&set(groups[test]))
    validation=[]
    for name,model,x in candidates():
        model.fit(x[train],y[train])
        validation.append((float(mean_squared_error(y[valid],model.predict(x[valid]))**.5),name,model,x))
    # Test only the model selected on validation, plus the predefined mean comparator.
    chosen=min((v for v in validation if v[1]!='mean'),key=lambda v:v[0])
    for val_rmse,name,model,x in [next(v for v in validation if v[1]=='mean'),chosen]:
        model.fit(x[trainval],y[trainval]); prediction=model.predict(x[test])
        rows.append({'seed':seed,'role':'comparator' if name=='mean' else 'selected','model':name,
                     'validation_rmse':val_rmse,'test_rmse':float(mean_squared_error(y[test],prediction)**.5),
                     'test_mae':float(mean_absolute_error(y[test],prediction)),'test_r2':float(r2_score(y[test],prediction)),
                     'n_train':len(train),'n_validation':len(valid),'n_test':len(test)})
        if name!='mean':
            predictions.append(unique.iloc[test][['canonical_smiles','target']].assign(seed=seed,model=name,prediction=prediction))
    print(json.dumps({k:rows[-1][k] for k in ['seed','model','test_rmse','test_r2','n_train','n_validation','n_test']}),flush=True)

results=pd.DataFrame(rows); results.to_csv(args.out/'results.csv',index=False)
pd.concat(predictions).to_csv(args.out/'test_predictions.csv',index=False)
unique.drop(columns=['row']).to_csv(args.out/'structures.csv',index=False)
summary={role:{metric:{'mean':float(part[metric].mean()),'sd':float(part[metric].std(ddof=1))}
               for metric in ['test_rmse','test_mae','test_r2']}
         for role,part in results.groupby('role')}
report={'source_url':'https://raw.githubusercontent.com/deepchem/deepchem/master/datasets/delaney-processed.csv',
        'sha256':hashlib.sha256(args.raw.read_bytes()).hexdigest(),'target':TARGET,
        'input_rows':len(source),'unique_structures':len(unique),
        'acyclic_structures_sharing_empty_scaffold':int(acyclic.sum()),'empty_scaffold_policy':args.empty_scaffold,
        'scaffold_groups':int(pd.Series(groups).nunique()),
        'split_sizes':{'train':int(results.n_train.mean()),'validation':int(results.n_validation.mean()),
                       'test_mean':int(results.n_test.mean()),'test_min':int(results.n_test.min()),'test_max':int(results.n_test.max())},
        'seeds':args.seeds,'models_selected':results.query("role=='selected'").model.value_counts().to_dict(),
        'summary_over_seeds':summary,'per_seed':rows,
        'limitations':['Solubility only; not comprehensive ADMET validation.',
                       'Scaffold-disjoint splits still leave analogue series split across train and test.',
                       'Split sizes vary between seeds because scaffold groups differ in size.']}
(args.out/'metrics.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:report[k] for k in ['unique_structures','scaffold_groups','acyclic_structures_sharing_empty_scaffold','split_sizes','models_selected','summary_over_seeds']},indent=2))
