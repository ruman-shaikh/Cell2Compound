"""Compare the HDAC2 active-site pocket in the experimental structure (4LXZ, vorinostat-bound)
and the AlphaFold model: residue mapping, pLDDT, pocket geometry, zinc-site geometry.
Writes an AlphaFold receptor with the crystal zinc transplanted for docking comparison."""
import argparse
import json
from pathlib import Path
import numpy as np
from Bio.PDB import PDBIO, PDBParser, Select, Superimposer
from Bio.PDB.Atom import Atom
from Bio.PDB.Residue import Residue

SYMMETRIC={('ASP','OD1'):['OD1','OD2'],('ASP','OD2'):['OD1','OD2'],('GLU','OE1'):['OE1','OE2'],('GLU','OE2'):['OE1','OE2']}
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--experimental',type=Path,default=Path('data/raw/4LXZ.pdb'))
parser.add_argument('--alphafold',type=Path,default=Path('data/raw/HDAC2_alphafold.pdb'))
parser.add_argument('--chain',default='A')
parser.add_argument('--ligand',default='SHH')
parser.add_argument('--cutoff',type=float,default=5.)
parser.add_argument('--out',type=Path,default=Path('results/structures'))
args=parser.parse_args()
args.out.mkdir(parents=True,exist_ok=True)

# Residue-number offset from DBREF (PDB start - UniProt start), then verified residue by residue.
dbref=next(l for l in args.experimental.read_text().splitlines() if l.startswith('DBREF') and l[12]==args.chain)
offset=int(dbref[14:18])-int(dbref[55:60])
pdb=PDBParser(QUIET=True)
exp_chain=pdb.get_structure('exp',args.experimental)[0][args.chain]
af_chain=next(iter(pdb.get_structure('af',args.alphafold)[0]))
af={r.id[1]:r for r in af_chain}
pairs,mismatch=[],[]
for residue in exp_chain:
    if residue.id[0]!=' ' or 'CA' not in residue:
        continue
    model=af.get(residue.id[1]-offset)
    if model is None:
        continue
    (pairs if model.resname==residue.resname else mismatch).append((residue,model))
plddt={r.id[1]:r['CA'].bfactor for r in af_chain}

ligand=[r for r in exp_chain if r.resname==args.ligand][0]
ligand_xyz=np.array([a.coord for a in ligand if a.element!='H' and a.get_altloc() in (' ','A')])
zinc=[r for r in exp_chain if r.resname=='ZN'][0]['ZN']
def near(residue):
    return min(np.linalg.norm(ligand_xyz-a.coord,axis=1).min() for a in residue)<=args.cutoff
pocket=[(e,m) for e,m in pairs if near(e)]
zinc_ligands=[(e,m) for e,m in pairs for a in e if a.element in 'NO' and np.linalg.norm(a.coord-zinc.coord)<=2.6]
zinc_ligands=list({e.id[1]:(e,m) for e,m in zinc_ligands}.values())

# Superpose AlphaFold onto the crystal using confident (pLDDT >= 70) CA atoms.
confident=[(e,m) for e,m in pairs if plddt[m.id[1]]>=70]
sup=Superimposer(); sup.set_atoms([e['CA'] for e,_ in confident],[m['CA'] for _,m in confident])
rotation,translation=sup.rotran
def moved(atom):
    return atom.coord@rotation+translation
def rmsd(atom_pairs):
    return float(np.sqrt(np.mean([np.sum((e.coord-moved(m))**2) for e,m in atom_pairs])))
def heavy_pairs(residues):
    out=[]
    for e,m in residues:
        for atom in e:
            if atom.element!='H' and atom.get_id() in m and atom.get_altloc() in (' ','A'):
                out.append((atom,m[atom.get_id()]))
    return out
pocket_local=Superimposer(); pocket_local.set_atoms([e['CA'] for e,_ in pocket],[m['CA'] for _,m in pocket])
inverse=np.linalg.inv(rotation)
zinc_in_af=(zinc.coord-translation)@inverse
coordination=[]
for e,m in zinc_ligands:
    for atom in e:
        if atom.element in 'NO' and np.linalg.norm(atom.coord-zinc.coord)<=2.6:
            # Carboxylate oxygens are chemically equivalent; take the closer one in the model.
            equivalent=SYMMETRIC.get((e.resname,atom.get_id()),[atom.get_id()])
            coordination.append({'residue':f'{e.resname}{e.id[1]}','uniprot':m.id[1],'atom':atom.get_id(),
                                 'crystal_distance':round(float(np.linalg.norm(atom.coord-zinc.coord)),2),
                                 'alphafold_distance_to_transplanted_zinc':round(float(min(np.linalg.norm(m[n].coord-zinc_in_af) for n in equivalent if n in m)),2)})

report={'experimental':str(args.experimental),'alphafold':str(args.alphafold),'chain':args.chain,
        'pdb_minus_uniprot_offset':offset,'mapped_residues':len(pairs),'name_mismatches':[f'{e.resname}{e.id[1]}/{m.resname}{m.id[1]}' for e,m in mismatch],
        'global_ca_rmsd_confident':round(float(sup.rms),3),'confident_ca_atoms':len(confident),
        'pocket_definition':f'residues with any atom within {args.cutoff} A of {args.ligand} chain {args.chain}',
        'pocket_residues':[{'pdb':f'{e.resname}{e.id[1]}','uniprot':m.id[1],'plddt':round(plddt[m.id[1]],1)} for e,m in pocket],
        'pocket_plddt':{'mean':round(float(np.mean([plddt[m.id[1]] for _,m in pocket])),2),
                        'min':round(float(np.min([plddt[m.id[1]] for _,m in pocket])),2)},
        'pocket_ca_rmsd_global_frame':round(rmsd([(e['CA'],m['CA']) for e,m in pocket]),3),
        'pocket_heavy_atom_rmsd_global_frame':round(rmsd(heavy_pairs(pocket)),3),
        'pocket_ca_rmsd_local_fit':round(float(pocket_local.rms),3),
        'zinc_coordination':coordination,
        'per_residue_heavy_rmsd':{f'{e.resname}{e.id[1]}':round(rmsd(heavy_pairs([(e,m)])),2) for e,m in pocket},
        'outputs':{'alphafold_receptor':str(args.out/'HDAC2_alphafold_zn_crystal_frame.pdb')},
        'limitations':['AlphaFold models apo HDAC2 without zinc; the zinc is transplanted, not predicted.',
                       'pLDDT measures local confidence, not ligand-induced conformational accuracy.',
                       'Residues with pLDDT < 50 are excluded from the written receptor.']}
# AlphaFold receptor in the crystal frame with the transplanted zinc (docking boxes stay comparable).
class Ordered(Select):
    def accept_residue(self,residue):
        return plddt[residue.id[1]]>=50
for atom in af_chain.get_atoms():
    atom.set_coord(moved(atom))
metal=Residue(('H_ZN',901,' '),'ZN',' '); metal.add(Atom('ZN',zinc.coord.copy(),0.,1.,' ','ZN',1,'ZN'))
af_chain.add(metal); plddt[901]=100.
io=PDBIO(); io.set_structure(af_chain.get_parent().get_parent()); io.save(str(args.out/'HDAC2_alphafold_zn_crystal_frame.pdb'),Ordered())
# Biopython drops formal charges; Meeko's zinc template needs element ZN (77-78) and charge 2+ (79-80).
receptor_path=args.out/'HDAC2_alphafold_zn_crystal_frame.pdb'
receptor_path.write_text('\n'.join(l[:66].ljust(76)+'ZN2+' if l.startswith('HETATM') and l[17:20]==' ZN' else l
                                   for l in receptor_path.read_text().splitlines())+'\n')

(args.out/'pocket_comparison.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k not in ('pocket_residues','per_residue_heavy_rmsd')},indent=2))
worst=sorted(report['per_residue_heavy_rmsd'].items(),key=lambda kv:-kv[1])[:6]
print('largest pocket residue deviations',worst)
