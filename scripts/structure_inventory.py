"""Inspect experimental target chains and AlphaFold residue confidence."""
import hashlib
import json
from pathlib import Path
import numpy as np
from Bio.PDB import PDBParser

parser=PDBParser(QUIET=True)
report={}
for name,path in [('experimental',Path('data/raw/4LXZ.pdb')),
                  ('alphafold',Path('data/raw/HDAC2_alphafold.pdb'))]:
    structure=parser.get_structure(name,path)
    chains=[]
    for chain in structure[0]:
        amino=[r for r in chain if r.id[0]==' ' and 'CA' in r]
        hetero=[{'name':r.resname,'number':int(r.id[1]),'atoms':len(list(r.get_atoms()))}
                for r in chain if r.id[0].startswith('H_')]
        item={'chain':chain.id,'residues_with_CA':len(amino),'hetero_residues':hetero}
        if name=='alphafold' and amino:
            confidence=np.array([r['CA'].bfactor for r in amino])
            item['confidence']={'mean_plddt':float(confidence.mean()),
                'fraction_below_70':float((confidence<70).mean()),
                'fraction_below_50':float((confidence<50).mean())}
        chains.append(item)
    report[name]={'file':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'chains':chains}
report['limitations']=['Global confidence does not establish pocket quality.',
    'Experimental B factors are not pLDDT and are not compared as such.',
    'Pocket mapping, sequence alignment, and docking validation remain required.']
out=Path('results/structures'); out.mkdir(parents=True,exist_ok=True)
(out/'inventory.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
