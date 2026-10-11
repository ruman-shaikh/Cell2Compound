"""Validate the HDAC2 docking protocol: cognate redocking and cross-docking into the 4LXZ
receptor for zinc-site ligands from several crystal structures, two scoring functions, several seeds."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import gemmi
import requests
from Bio.PDB import PDBParser, Superimposer
from rdkit import Chem
from rdkit.Chem import Descriptors
from cell2compound.docking import (centroid, dock, extract_chain, extract_ligand, ligand_pdbqt, pick_scoring, pose_scores,
                                   resname_matches,
                                   pose_molecule, pose_rmsds, prepare_receptor, receptor_zinc,
                                   summarise_runs, zinc_contacts)

# id: (ligand code, SMILES from the RCSB chemical component, chemotype)
COMPLEXES={'4LXZ':('SHH','ONC(=O)CCCCCCC(=O)Nc1ccccc1','hydroxamate'),
           '6G3O':('EL8','CC(CCCCC(=O)NO)NC1=C(C(=O)C1=O)Nc1ccc(cc1)OC(F)(F)F','hydroxamate'),
           '4LY1':('20Y','CC(=O)Nc1ccc(cc1)C(=O)Nc1cc(ccc1N)-c1cccs1','benzamide'),
           '5IWG':('IWX','Nc1ccc(cc1NC(=O)C1CCOCC1)-c1ccc(F)cc1','benzamide'),
           '3MAX':('LLX','Nc1ccc(cc1NC(=O)c1ccccc1)-c1ccccc1','benzamide'),
           '9NTB':('A1B1V','CS(=N)(=O)c1ccc(cc1)C(=O)Nc1cc(ccc1N)-c1ccc(F)cc1','benzamide')}
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--reference',default='4LXZ')
parser.add_argument('--alphafold',type=Path,default=Path('results/structures/HDAC2_alphafold_zn_crystal_frame.pdb'),
                    help='AlphaFold receptor already superposed into the reference chain A frame (pocket_comparison.py)')
parser.add_argument('--seeds',type=int,nargs='+',default=[1,2,3,4,5])
parser.add_argument('--scoring',nargs='+',default=['vina','vinardo'])
parser.add_argument('--exhaustiveness',type=int,default=16)
parser.add_argument('--box',type=float,default=22.)
parser.add_argument('--cpu',type=int,default=4)
parser.add_argument('--refresh',action='store_true',help='re-dock even when a saved pose file exists')
parser.add_argument('--out',type=Path,default=Path('results/docking/validation'))
args=parser.parse_args()
args.out.mkdir(parents=True,exist_ok=True); work=args.out/'work'; work.mkdir(exist_ok=True)
pdb_dir=Path('data/raw/pdb'); pdb_dir.mkdir(parents=True,exist_ok=True)

def fetch(code):
    """Legacy PDB format where RCSB still serves it; newer entries are converted from mmCIF."""
    path=pdb_dir/f'{code}.pdb'
    if path.exists():
        return path
    reply=requests.get(f'https://files.rcsb.org/download/{code}.pdb',timeout=60)
    if reply.status_code==404:
        cif=pdb_dir/f'{code}.cif'
        if not cif.exists():
            source=requests.get(f'https://files.rcsb.org/download/{code}.cif',timeout=60)
            source.raise_for_status(); cif.write_text(source.text)
        structure=gemmi.read_structure(str(cif))
        structure.setup_entities(); structure.assign_label_seq_id()
        structure.write_pdb(str(path))
        return path
    reply.raise_for_status(); path.write_text(reply.text)
    return path

def ligand_chain(path,resname):
    """Chain whose ligand copy sits closest to that chain's zinc (the active-site copy)."""
    structure=PDBParser(QUIET=True).get_structure('x',path)[0]; best=None
    for chain in structure:
        zinc=[a.coord for r in chain if r.resname=='ZN' for a in r]
        for residue in chain:
            if resname_matches(residue.resname,resname) and zinc:
                distance=min(np.linalg.norm(a.coord-z) for a in residue for z in zinc)
                if best is None or distance<best[0]:
                    best=(distance,chain.id)
    if best is None:
        raise ValueError(f'{path}: no {resname} in a zinc-containing chain')
    return best[1],best[0]

def superposition(mobile_path,mobile_chain,reference_path,reference_chain='A',max_rms=2.5):
    """Superpose onto the reference chain, matching residues by number after detecting the
    numbering offset. HDAC2 entries use different conventions (4LXZ is UniProt+4, 9NTB is
    UniProt), so matching on raw numbers silently pairs the wrong residues."""
    parser=PDBParser(QUIET=True)
    mobile={r.id[1]:r for r in parser.get_structure('m',mobile_path)[0][mobile_chain] if 'CA' in r and r.id[0]==' '}
    fixed={r.id[1]:r for r in parser.get_structure('f',reference_path)[0][reference_chain] if 'CA' in r and r.id[0]==' '}
    best=None
    for offset in range(-25,26):
        shared=[i for i in fixed if i-offset in mobile]
        if len(shared)<100:
            continue
        agree=sum(fixed[i].resname==mobile[i-offset].resname for i in shared)/len(shared)
        if best is None or agree>best[0]:
            best=(agree,offset,shared)
    if best is None or best[0]<.9:
        raise ValueError(f'{mobile_path}: no residue numbering offset aligns the sequences '
                         f'(best identity {best[0]:.2f} at offset {best[1]})' if best else 'no overlap')
    agree,offset,shared=best
    common=[i for i in shared if fixed[i].resname==mobile[i-offset].resname]
    sup=Superimposer(); sup.set_atoms([fixed[i]['CA'] for i in common],[mobile[i-offset]['CA'] for i in common])
    if sup.rms>max_rms:
        raise ValueError(f'{mobile_path}: CA RMSD {sup.rms:.2f} A to {reference_path} exceeds {max_rms} A; '
                         'the transformed reference ligand would be meaningless')
    return sup.rotran,float(sup.rms),len(common),offset,round(agree,3)


def transformed(mol,rotran):
    rotation,translation=rotran; mol=Chem.Mol(mol); conf=mol.GetConformer()
    xyz=conf.GetPositions()@rotation+translation
    for i,p in enumerate(xyz):
        conf.SetAtomPosition(i,p.tolist())
    return mol

reference_path=fetch(args.reference)
size=(args.box,)*3
rows=[]; structures=[]
for code,(resname,smiles,chemotype) in COMPLEXES.items():
    path=fetch(code); chain,zinc_distance=ligand_chain(path,resname)
    crystal=extract_ligand(path,resname,chain,smiles)
    own_receptor,removed=prepare_receptor(extract_chain(path,chain,{'ZN'},work/f'{code}_{chain}.pdb'),work/f'{code}_{chain}',
                                  centroid(crystal).round(3),size)
    rotran,ca_rmsd,ca_count,offset,identity=superposition(path,chain,reference_path)
    crystal_in_reference=transformed(crystal,rotran)
    structures.append({'pdb':code,'ligand':resname,'chain':chain,'chemotype':chemotype,
                       'ligand_zinc_distance':round(zinc_distance,2),'heavy_atoms':crystal.GetNumAtoms(),
                       'rotatable_bonds':Descriptors.NumRotatableBonds(crystal),
                       'ca_rmsd_to_reference':round(ca_rmsd,3),'ca_atoms':ca_count,
                       'numbering_offset_to_reference':offset,'resname_identity':identity,'receptor_residues_removed':removed,
                       'crystal_zinc_contacts':zinc_contacts(crystal,receptor_zinc(own_receptor))})
    cross_receptor=work/f'{args.reference}_A.pdbqt'; af_receptor=work/'alphafold.pdbqt'
    if not cross_receptor.exists():
        prepare_receptor(extract_chain(reference_path,'A',{'ZN'},work/f'{args.reference}_A.pdb'),
                         work/f'{args.reference}_A',centroid(crystal_in_reference).round(3),size)
    if not af_receptor.exists():
        prepare_receptor(extract_chain(args.alphafold,'A',{'ZN'},work/'alphafold.pdb'),work/'alphafold',
                         centroid(crystal_in_reference).round(3),size)
    tasks=[('redock',own_receptor,crystal),('alphafold',af_receptor,crystal_in_reference)]
    if code!=args.reference:
        tasks.append(('crossdock',cross_receptor,crystal_in_reference))
    for mode,receptor,truth in tasks:
        for scoring in args.scoring:
            for seed in args.seeds:
                # A finished run is reused from its saved pose file; Vina scores are recorded there.
                saved=work/f'{code}_{mode}_{scoring}_{seed}.pdbqt'
                if saved.exists() and not args.refresh:
                    poses=saved.read_text(); energies=pose_scores(poses); reused=True
                else:
                    poses,energies=dock(receptor,ligand_pdbqt(smiles,seed),centroid(truth),size,seed,
                                        args.exhaustiveness,cpu=args.cpu,scoring=scoring)
                    saved.write_text(poses); reused=False
                docked=pose_molecule(poses); rmsd=pose_rmsds(docked,truth); zinc=receptor_zinc(receptor)
                contacts=[zinc_contacts(docked,zinc,conf_id=c.GetId()) for c in docked.GetConformers()]
                rows.append({'pdb':code,'chemotype':chemotype,'mode':mode,'scoring':scoring,'seed':seed,
                             'top1_rmsd':rmsd[0],'best_top3_rmsd':min(rmsd[:3]),'best_any_rmsd':min(rmsd),
                             'top1_score':float(energies[0]),'top1_zinc_contacts':contacts[0],
                             'crystal_zinc_contacts':zinc_contacts(truth,zinc),'reused_saved_pose':reused})
        pd.DataFrame(rows).to_csv(args.out/'runs.partial.csv',index=False)  # survive a later failure
        print(code,mode,pd.DataFrame(rows).query('pdb==@code and mode==@mode').groupby('scoring').top1_rmsd.median().round(2).to_dict(),flush=True)

runs=pd.DataFrame(rows); runs.to_csv(args.out/'runs.csv',index=False)
print(f"reused {int(runs.reused_saved_pose.sum())} saved poses of {len(runs)} runs",flush=True)
summary=summarise_runs(runs)
per_complex=summary['per_complex']
report={'reference_receptor':args.reference,'box_angstrom':args.box,'exhaustiveness':args.exhaustiveness,'seeds':args.seeds,
        'structures':structures,
        'selection_rule':'Scoring function for candidate docking = best crossdock top-1 success (RMSD <= threshold); zinc engagement breaks ties.',
        'scoring_selected':pick_scoring(summary),
        'success_rates_rmsd_le_2A':{f'{m}/{s}':v for (m,s),v in summary['overall'].to_dict('index').items()},
        'success_rates_by_chemotype':{f'{c}/{m}/{s}':v for (c,m,s),v in summary['by_chemotype'].to_dict('index').items()},
        'median_top1_rmsd_by_chemotype':{f'{c}/{m}/{s}':v for (c,m,s),v in summary['median_top1_rmsd_by_chemotype'].items()},
        'median_top1_rmsd':{f'{p}/{m}':v for (p,m),v in per_complex.to_dict('index').items()},
        'protocol':'Chain with active-site ligand + catalytic Zn; Meeko templates (altloc A, template hydrogens); '
                   'box centred on crystal ligand; ligand from SMILES via ETKDG/MMFF (crystal pose never used as input); '
                   'heavy-atom symmetry-aware RMSD without refitting.',
        'limitations':['Box centred on the known ligand: tests pose ranking, not pocket finding.',
                       'Vina/Vinardo do not model zinc chelation explicitly (AutoDock4Zn needs autogrid4, unavailable).',
                       'Six complexes; success rates are descriptive, not a benchmark.',
                       'Zinc engagement = any ligand N/O within 2.8 A of Zn; it does not check chelation geometry.',
                       'Success rates differ by chemotype; the pooled rate should not be quoted alone.']}
(args.out/'summary.json').write_text(json.dumps(report,indent=2,default=str)+'\n')
print(summary['overall'].to_string()); print(summary['by_chemotype'].to_string()); print(per_complex.to_string())
