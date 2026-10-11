"""Dock sci-Plex compounds into HDAC2 with the validated protocol.

Validation showed the receptor must match the ligand chemotype: benzamides cross-docked into the
vorinostat structure (4LXZ) land ~11 A from their crystal pose, because the internal pocket they
occupy is not open there. So each compound is docked into both a hydroxamate-bound receptor
(4LXZ) and a benzamide-bound one (4LY1), and results are reported per receptor and per chemotype.

Positives: compounds carrying an HDAC zinc-binding group (hydroxamate / ortho-aminoanilide);
comparators: size-limited non-ZBG sci-Plex compounds sampled with a fixed seed.
Docking, transcriptional response, and ChEMBL evidence are reported side by side, never combined."""
import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.metrics import roc_auc_score
from cell2compound.docking import (centroid, dock, extract_chain, extract_ligand, ligand_pdbqt, pose_molecule,
                                   pose_scores, prepare_receptor, receptor_zinc, zinc_contacts)

ZBG={'hydroxamate':'[CX3](=O)[NX3H][OX2H]','o_aminoanilide':'c([NH2])c[NH]C(=O)'}
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--receptors',nargs='+',default=['4LXZ:SHH:ONC(=O)CCCCCCC(=O)Nc1ccccc1:hydroxamate',
                                                    '4LY1:20Y:CC(=O)Nc1ccc(cc1)C(=O)Nc1cc(ccc1N)-c1cccs1:o_aminoanilide'],
                    help='PDB:ligand:SMILES:chemotype_the_receptor_suits')
parser.add_argument('--validation',type=Path,default=Path('results/docking/validation/summary.json'))
parser.add_argument('--comparators',type=int,default=21)
parser.add_argument('--max-mw',type=float,default=600.)
parser.add_argument('--max-rotatable',type=int,default=12)
parser.add_argument('--seeds',type=int,nargs='+',default=[1,2,3])
parser.add_argument('--exhaustiveness',type=int,default=16)
parser.add_argument('--box',type=float,default=22.)
parser.add_argument('--cpu',type=int,default=4,
                    help='threads per docking run. Changing it changes Vina trajectories, so poses '
                         'cached at a different value would not be comparable; keep it fixed per dataset.')
parser.add_argument('--workers',type=int,default=max(1,(os.cpu_count() or 4)//5),
                    help='concurrent docking runs. Compounds/receptors/seeds are independent, so this '
                         'scales near-linearly, unlike raising --cpu past --exhaustiveness.')
parser.add_argument('--refresh',action='store_true',help='re-dock even when a saved pose file exists')
parser.add_argument('--out',type=Path,default=Path('results/docking/candidates'))
args=parser.parse_args()
args.out.mkdir(parents=True,exist_ok=True); poses_dir=args.out/'poses'; poses_dir.mkdir(exist_ok=True)

validation=json.loads(args.validation.read_text())
rates=validation['success_rates_rmsd_le_2A']
scoring=validation.get('scoring_selected') or max(
    ['vina','vinardo'],key=lambda s:(rates[f'redock/{s}']['top1_success'],rates[f'redock/{s}']['top1_zinc_engaged']))

compounds=pd.read_csv('results/targets/compound_chembl.csv')
mechanisms=pd.read_csv('results/targets/mechanisms.csv')
hdac_mech=set(mechanisms[mechanisms.target_name.str.contains('deacetylase',case=False,na=False)
                         &~mechanisms.target_name.str.contains('sirtuin',case=False,na=False)].parent_smiles)
patterns={k:Chem.MolFromSmarts(v) for k,v in ZBG.items()}
rows=[]
for r in compounds.itertuples():
    mol=Chem.MolFromSmiles(r.parent_smiles)
    rows.append({'compound':r.observed_names,'parent_smiles':r.parent_smiles,'chembl_id':r.chembl_id,
                 'zbg':','.join(k for k,p in patterns.items() if mol.HasSubstructMatch(p)),
                 'chembl_hdac_mechanism':r.parent_smiles in hdac_mech,
                 'mw':Descriptors.MolWt(mol),'rotatable_bonds':Descriptors.NumRotatableBonds(mol)})
pool=pd.DataFrame(rows)
positives=pool[pool.zbg!='']
eligible=pool[(pool.zbg=='')&(pool.mw<=args.max_mw)&(pool.rotatable_bonds<=args.max_rotatable)&~pool.chembl_hdac_mechanism]
comparators=eligible.sample(n=min(args.comparators,len(eligible)),random_state=0)
library=pd.concat([positives.assign(group='zbg'),comparators.assign(group='comparator')],ignore_index=True)

results=[]
receptors={}
for spec in args.receptors:
    code,resname,smiles,suits=spec.split(':')
    path=Path('data/raw/pdb')/f'{code}.pdb'
    if not path.exists():
        reply=requests.get(f'https://files.rcsb.org/download/{code}.pdb',timeout=60); reply.raise_for_status()
        path.write_text(reply.text)
    crystal=extract_ligand(path,resname,'A' if code!='3MAX' else 'B',smiles)
    center=centroid(crystal).round(3); size=(args.box,)*3
    receptor,removed=prepare_receptor(extract_chain(path,'A' if code!='3MAX' else 'B',{'ZN'},args.out/f'{code}.pdb'),
                                      args.out/code,center,size)
    receptors[code]={'receptor':receptor,'center':center,'size':size,'zinc':receptor_zinc(receptor),
                     'removed':removed,'suits_chemotype':suits,'cognate_ligand':resname}
    print(code,'receptor ready; residues removed:',removed,flush=True)

def pose_path(compound,code,seed):
    return poses_dir/f'{code}_{library.index[library.compound==compound][0]:03d}_seed{seed}.pdbqt'


def run_one(task):
    """One docking run in a worker process. Vina objects cannot be pickled, so the receptor is
    passed as a file path and the Vina instance is built here."""
    compound,smiles,code,receptor,center,size,seed,saved=task
    try:
        if Path(saved).exists() and not args.refresh:
            poses=Path(saved).read_text(); energies=pose_scores(poses); reused=True
        else:
            poses,energies=dock(receptor,ligand_pdbqt(smiles,seed),center,size,seed,
                                args.exhaustiveness,cpu=args.cpu,scoring=scoring)
            Path(saved).write_text(poses); reused=False
        return {'compound':compound,'receptor':code,'seed':seed,'cpu':args.cpu,
                'top1_score':float(energies[0]),
                'top1_zinc_contacts':zinc_contacts(pose_molecule(poses),np.asarray(receptors[code]['zinc'])),
                'reused_saved_pose':reused}
    except Exception as error:  # record failures instead of silently dropping compounds
        return {'compound':compound,'receptor':code,'seed':seed,'cpu':args.cpu,'error':str(error)[:200]}


tasks=[(item.compound,item.parent_smiles,code,str(info['receptor']),tuple(info['center']),tuple(info['size']),
        seed,str(pose_path(item.compound,code,seed)))
       for item in library.itertuples() for code,info in receptors.items() for seed in args.seeds]
cached=sum(Path(t[-1]).exists() for t in tasks)
print(f'{len(tasks)} runs; {cached} already on disk; {len(tasks)-cached} to dock '
      f'on {args.workers} workers x {args.cpu} threads',flush=True)
# Keep BLAS single-threaded in workers: Vina manages its own threads and oversubscription
# would slow every run down.
for variable in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ.setdefault(variable,'1')
with ProcessPoolExecutor(max_workers=args.workers) as pool:
    for n,record in enumerate(pool.map(run_one,tasks),1):
        results.append(record)
        if 'error' in record:
            print('FAILED',record['receptor'],record['compound'][:34],record['error'][:80],flush=True)
        elif not record['reused_saved_pose']:
            print(f"[{n}/{len(tasks)}]",record['receptor'],record['compound'][:34],
                  round(record['top1_score'],2),flush=True)
        if n%20==0:
            pd.DataFrame(results).to_csv(args.out/'runs.partial.csv',index=False)
runs=pd.DataFrame(results); runs.to_csv(args.out/'runs.csv',index=False)
if 'error' in runs and runs.error.notna().any():
    print('runs with errors:',int(runs.error.notna().sum()),flush=True)
per=runs[runs.get('top1_score',pd.Series(dtype=float)).notna()].groupby(['compound','receptor']).agg(best_score=('top1_score','min'),score_sd=('top1_score','std'),
                 zinc_engaged_fraction=('top1_zinc_contacts',lambda v:float((v>=1).mean()))).reset_index()
table=library.merge(per,on='compound',how='left')
# Primary receptor for a compound = the one whose cognate ligand matches its zinc-binding group.
suits={code:info['suits_chemotype'] for code,info in receptors.items()}
table['receptor_matches_chemotype']=[bool(r and z and suits.get(r) in str(z).split(',')) for r,z in zip(table.receptor,table.zbg)]

# Side-by-side evidence: transcriptional response magnitude (max over doses) and ChEMBL HDAC2 potency.
responses=pd.read_csv('results/expression/responses.csv')
names=table.compound.str.split(' \\| ')
response=responses.groupby('compound').response_rms.max()
table['max_response_rms']=[max((response.get(n,np.nan) for n in ns),default=np.nan) for ns in names]
pairs_path=Path('data/bioactivity/pairs.csv')
if pairs_path.exists():
    hdac2=pd.read_csv(pairs_path).query("gene=='HDAC2'").set_index('parent_smiles').pchembl
    table['chembl_hdac2_pchembl']=table.parent_smiles.map(hdac2)
table.sort_values(['receptor','best_score']).to_csv(args.out/'candidates.csv',index=False)

ok=table.dropna(subset=['best_score'])
def contrast(part):
    """ZBG vs comparator separation within one receptor; AUROC is descriptive, not a claim."""
    label=(part.group=='zbg').astype(int)
    if label.nunique()<2:
        return None
    sizes=[Chem.MolFromSmiles(x).GetNumHeavyAtoms() for x in part.parent_smiles]
    return {'n':int(len(part)),'median_best_score':part.groupby('group').best_score.median().round(2).to_dict(),
            'zinc_engaged_fraction':part.groupby('group').zinc_engaged_fraction.mean().round(3).to_dict(),
            'auroc_zbg_vs_comparator':round(float(roc_auc_score(label,-part.best_score)),3),
            'score_vs_heavy_atoms_spearman':round(float(spearmanr(part.best_score,sizes)[0]),3)}

validation_by_chemotype=validation.get('success_rates_by_chemotype',{})
report={'receptors':{code:{'pdb':code,'cognate_ligand':info['cognate_ligand'],'suits_chemotype':info['suits_chemotype'],
                           'residues_removed':info['removed'],'box_center':info['center'].tolist()}
                     for code,info in receptors.items()},
        'scoring':scoring,'scoring_selected_by':validation.get('selection_rule'),
        'threads_per_run':args.cpu,'concurrent_runs':args.workers,
        'parallelism_note':('Compounds/receptors/seeds are independent and run concurrently; threads per run '
                            'is held at the value the cached poses were produced with, since Vina trajectories '
                            'depend on it.'),
        'box_size':args.box,'exhaustiveness':args.exhaustiveness,'seeds':args.seeds,
        'library':{'zbg':int((library.group=='zbg').sum()),'comparators':int((library.group=='comparator').sum()),
                   'failed':sorted(runs.loc[runs.get('error',pd.Series(dtype=str)).notna(),'compound'].unique().tolist()) if 'error' in runs else []},
        'per_receptor':{code:contrast(ok[ok.receptor==code]) for code in receptors},
        'chemotype_matched_receptor':contrast(ok[ok.receptor_matches_chemotype|(ok.group=='comparator')]),
        'validation_context':{'redock_success_by_chemotype':validation_by_chemotype,
                              'note':'No receptor redocked hydroxamates within 2 A, so hydroxamate rankings are weak evidence.'},
        'response_rms_median':ok.groupby('group').max_response_rms.median().round(4).to_dict(),
        'response_mannwhitney_p_zbg_greater':float(mannwhitneyu(
            ok.drop_duplicates('compound').query("group=='zbg'").max_response_rms.dropna(),
            ok.drop_duplicates('compound').query("group=='comparator'").max_response_rms.dropna(),
            alternative='greater').pvalue),
        'limitations':['Docking scores are not binding affinities; ZBG compounds are expected to score well partly by construction.',
                       'Cognate redocking never reached 2 A for hydroxamates in any receptor, including the native one.',
                       'Benzamides only redock well in a benzamide-bound receptor; a single receptor cannot serve both chemotypes.',
                       'Score correlates with ligand size; the correlation is reported beside every contrast.',
                       'Comparators are other sci-Plex drugs, not confirmed HDAC non-binders.',
                       'Transcriptional response is shown alongside, never used to rank or validate docking.']}
if 'chembl_hdac2_pchembl' in table:
    known=ok.dropna(subset=['chembl_hdac2_pchembl'])
    report['compounds_with_chembl_hdac2_potency']=int(known.compound.nunique())
    report['score_vs_hdac2_pchembl_spearman']={
        code:(round(float(spearmanr(-part.best_score,part.chembl_hdac2_pchembl)[0]),3) if len(part)>=5 else None)
        for code,part in known.groupby('receptor')}
(args.out/'summary.json').write_text(json.dumps(report,indent=2,default=str)+'\n')
print(json.dumps(report,indent=2,default=str))
