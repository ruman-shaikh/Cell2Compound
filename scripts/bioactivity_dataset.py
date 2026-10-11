"""Build a target-aware binding dataset: ChEMBL pChEMBL values for an HDAC + kinase panel,
plus UniProt sequences for ESM. Targets come from the sci-Plex mechanism annotations."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import requests
from cell2compound.chembl import CachedChEMBL
from cell2compound.compounds import parent_smiles

PANEL={'Q13547':'HDAC1','Q92769':'HDAC2','O15379':'HDAC3','Q9UBN7':'HDAC6','Q9BY41':'HDAC8',
       'P23458':'JAK1','O60674':'JAK2','P52333':'JAK3','P29597':'TYK2','O14965':'AURKA','Q96GD4':'AURKB',
       'P00519':'ABL1','P10721':'KIT','P36888':'FLT3','P00533':'EGFR','P12931':'SRC'}
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--max-pages',type=int,default=30,help='1000 records per page per target')
parser.add_argument('--out',type=Path,default=Path('data/bioactivity'))
args=parser.parse_args()
args.out.mkdir(parents=True,exist_ok=True)
chembl=CachedChEMBL()

fasta=[]; targets=[]
for accession,gene in PANEL.items():
    path=Path('data/raw/uniprot')/f'{accession}.fasta'; path.parent.mkdir(parents=True,exist_ok=True)
    if not path.exists():
        reply=requests.get(f'https://rest.uniprot.org/uniprotkb/{accession}.fasta',timeout=60); reply.raise_for_status()
        path.write_text(reply.text)
    sequence=''.join(path.read_text().splitlines()[1:])
    fasta.append(f'>{gene} {accession}\n{sequence}')
    hit=chembl.get('target',target_components__accession=accession,target_type='SINGLE%20PROTEIN',
                   organism='Homo%20sapiens',only='target_chembl_id,pref_name')['targets']
    if len(hit)!=1:
        raise ValueError(f'{accession}: expected one human single-protein target, got {hit}')
    targets.append({'gene':gene,'accession':accession,'target_chembl_id':hit[0]['target_chembl_id'],
                    'target_name':hit[0]['pref_name'],'length':len(sequence),
                    'sequence_sha256':hashlib.sha256(sequence.encode()).hexdigest()})
(args.out/'targets.fasta').write_text('\n'.join(fasta)+'\n')
targets=pd.DataFrame(targets)

fields='molecule_chembl_id,canonical_smiles,pchembl_value,standard_type,standard_relation,data_validity_comment,assay_type,document_chembl_id'
raw=[]
for target in targets.itertuples():
    for record in chembl.pages('activity','activities',max_pages=args.max_pages,target_chembl_id=target.target_chembl_id,
                               pchembl_value__isnull='false',assay_type='B',standard_relation='%3D',limit=1000,only=fields):
        raw.append({**record,'gene':target.gene})
raw=pd.DataFrame(raw)
clean=raw[raw.data_validity_comment.isna()&raw.canonical_smiles.notna()].copy()
clean['pchembl_value']=clean.pchembl_value.astype(float)
parents={s:parent_smiles(s) for s in clean.canonical_smiles.unique()}
clean['parent_smiles']=clean.canonical_smiles.map(parents)
clean=clean[clean.parent_smiles.notna()]
# One label per (structure, target): median over assays; record spread for noise estimation.
pairs=(clean.groupby(['parent_smiles','gene']).pchembl_value
       .agg(pchembl='median',measurements='size',spread='std').reset_index())
pairs.to_csv(args.out/'pairs.csv',index=False)
targets=targets.merge(pairs.groupby('gene').agg(pairs=('pchembl','size'),median_pchembl=('pchembl','median')).reset_index(),on='gene')
targets.to_csv(args.out/'targets.csv',index=False)
report={'chembl_release':chembl.get('status').get('chembl_db_version'),'raw_records':len(raw),
        'clean_records':len(clean),'pairs':len(pairs),'structures':int(pairs.parent_smiles.nunique()),
        'filters':['assay_type=B','standard_relation==','pchembl_value present','no data_validity_comment',
                   'salt-stripped parent structure','median per structure/target'],
        'max_pages_per_target':args.max_pages,
        'replicate_sd_median':float(pairs.spread.median()),
        'targets':targets.drop(columns=['sequence_sha256']).to_dict('records')}
(args.out/'summary.json').write_text(json.dumps(report,indent=2,default=str)+'\n')
print(json.dumps(report,indent=2,default=str))
