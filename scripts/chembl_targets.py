"""Resolve sci-Plex parent structures in ChEMBL by InChIKey; collect curated mechanisms.
Every API response is cached under data/raw/chembl/ with its URL; reruns are offline."""
import argparse
import json
from pathlib import Path
import pandas as pd
from rdkit import Chem
from cell2compound.chembl import CachedChEMBL
from cell2compound.compounds import first_token, load_source

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--source',type=Path,default=Path('data/raw/sciplex_compounds.csv'))
parser.add_argument('--mapping',type=Path,default=Path('results/expression/compound_mapping.csv'))
parser.add_argument('--cache',type=Path,default=Path('data/raw/chembl'))
parser.add_argument('--out',type=Path,default=Path('results/targets'))
args=parser.parse_args()
args.out.mkdir(parents=True,exist_ok=True)
get=CachedChEMBL(args.cache).get

status=get('status')
source=load_source(args.source)
observed=pd.read_csv(args.mapping)
names=observed.groupby('parent_smiles').observed_name.apply(lambda s:' | '.join(sorted(s)))
records,mechanisms=[],[]
def normal(text):
    return ''.join(ch for ch in (text or '').casefold() if ch.isalnum())
for parent,observed_names in names.items():
    mol=Chem.MolFromSmiles(parent); key=Chem.MolToInchiKey(mol)
    fields='molecule_chembl_id,pref_name,max_phase,molecule_synonyms'
    exact=get('molecule',molecule_structures__standard_inchi_key=key,only=fields)['molecules']
    # Same connectivity (first InChIKey block): racemates, other stereoisomers, isotopologues.
    related=[m for m in get('molecule',molecule_structures__standard_inchi_key__startswith=key[:14],only=fields,limit=20)['molecules']
             if m['molecule_chembl_id'] not in {e['molecule_chembl_id'] for e in exact}]
    hits=[(m,'exact_inchikey') for m in exact]+[(m,'connectivity_only') for m in related]
    hit=hits[0][0] if hits else {}
    names={normal(m.get('pref_name')) for m,_ in hits}|{normal(y.get('molecule_synonym')) for m,_ in hits for y in m.get('molecule_synonyms') or []}
    names.discard('')
    tokens={normal(first_token(n)) for n in observed_names.split(' | ')}|{normal(t) for n in observed_names.split(' | ')
                                                                           for t in n.replace(')','(').replace(',','(').split('(')}
    tokens={t for t in tokens if len(t)>=4}
    agreement='no_chembl_name' if not names else ('agree' if any(t in nm or nm in t for t in tokens for nm in names if len(nm)>=4) else 'disagree')
    records.append({'observed_names':observed_names,'parent_smiles':parent,'inchikey':key,
                    'match':hits[0][1] if hits else 'not_found','exact_records':len(exact),'connectivity_records':len(related),
                    'chembl_id':hit.get('molecule_chembl_id'),'pref_name':hit.get('pref_name'),
                    'named_record':next((m.get('pref_name') for m,_ in hits if m.get('pref_name')),None),
                    'max_phase':max([float(m['max_phase']) for m,_ in hits if m.get('max_phase') is not None],default=None),
                    'name_check':agreement})
    for m,evidence in hits:
        for mech in get('mechanism',molecule_chembl_id=m['molecule_chembl_id'],limit=100)['mechanisms']:
            mechanisms.append({'chembl_id':m['molecule_chembl_id'],'evidence':evidence,'observed_names':observed_names,
                               'parent_smiles':parent,'target_chembl_id':mech['target_chembl_id'],
                               'mechanism':mech['mechanism_of_action'],'action_type':mech['action_type'],
                               'direct_interaction':mech['direct_interaction']})
compounds=pd.DataFrame(records); mech=pd.DataFrame(mechanisms)
targets=sorted(mech.target_chembl_id.dropna().unique())
target_info=[]
for chunk in [targets[i:i+40] for i in range(0,len(targets),40)]:
    for t in get('target',target_chembl_id__in=','.join(chunk),limit=100,
                 only='target_chembl_id,pref_name,target_type,organism,target_components')['targets']:
        accessions=[c.get('accession') for c in t.get('target_components') or [] if c.get('accession')]
        target_info.append({'target_chembl_id':t['target_chembl_id'],'target_name':t['pref_name'],
                            'target_type':t['target_type'],'organism':t['organism'],
                            'uniprot':';'.join(accessions)})
mech=mech.merge(pd.DataFrame(target_info),on='target_chembl_id',how='left')
summary=(mech[mech.target_type=='SINGLE PROTEIN'].groupby(['target_chembl_id','target_name','uniprot'])
         .parent_smiles.nunique().sort_values(ascending=False).rename('compounds').reset_index())
families=(mech[mech.target_type!='SINGLE PROTEIN'].groupby(['target_chembl_id','target_name','target_type'])
          .parent_smiles.nunique().sort_values(ascending=False).rename('compounds').reset_index())
compounds.to_csv(args.out/'compound_chembl.csv',index=False)
mech.to_csv(args.out/'mechanisms.csv',index=False)
summary.to_csv(args.out/'single_protein_targets.csv',index=False)
families.to_csv(args.out/'family_targets.csv',index=False)
report={'chembl_release':status.get('chembl_db_version'),'release_date':status.get('chembl_release_date'),
        'parents':len(compounds),'match':compounds.match.value_counts().to_dict(),
        'name_check':compounds.name_check.value_counts().to_dict(),
        'name_disagreements':compounds.loc[compounds.name_check=='disagree',['observed_names','named_record','chembl_id']].to_dict('records'),
        'not_found':compounds.loc[compounds.match=='not_found','observed_names'].tolist(),
        'parents_with_mechanism':int(mech.parent_smiles.nunique()),
        'parents_with_mechanism_exact_only':int(mech[mech.evidence=='exact_inchikey'].parent_smiles.nunique()),
        'top_single_protein_targets':summary.head(15).to_dict('records'),
        'top_family_or_complex_targets':families.head(8).to_dict('records'),
        'note':'ChEMBL mechanisms are curated drug-target annotations; they do not show binding in A549 at the sci-Plex doses.'}
(args.out/'summary.json').write_text(json.dumps(report,indent=2,default=str)+'\n')
print(json.dumps(report,indent=2,default=str))
