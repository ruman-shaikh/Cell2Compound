"""Receptor/ligand preparation, AutoDock Vina runs, and symmetry-aware pose RMSD."""
import re
import subprocess
from pathlib import Path
import numpy as np
from meeko import MoleculePreparation, PDBQTMolecule, PDBQTWriterLegacy, RDKitMolCreate
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolAlign
from vina import Vina

MK_RECEPTOR = Path('/home/ruman/venvs/general/bin/mk_prepare_receptor.py')


def extract_chain(pdb, chain, keep_hetero, out):
    """Protein atoms of one chain plus named hetero residues (e.g. catalytic metals)."""
    lines = [line for line in Path(pdb).read_text().splitlines()
             if line[21:22] == chain and (line.startswith('ATOM') or
                                          (line.startswith('HETATM') and line[17:20].strip() in keep_hetero))]
    Path(out).write_text('\n'.join(lines) + '\nEND\n')
    return out


def resname_matches(field, resname):
    """Legacy PDB holds 3 characters, so ligand codes from newer mmCIF entries arrive truncated."""
    field = field.strip()
    return field == resname or (len(resname) > 3 and field == resname[:3])


def extract_ligand(pdb, resname, chain, smiles):
    """Crystal ligand with bond orders assigned from a SMILES template.

    AssignBondOrdersFromTemplate raises if the matched atoms are not the template molecule,
    so a truncated residue code cannot silently pull in the wrong or a partial ligand.
    """
    block = '\n'.join(line for line in Path(pdb).read_text().splitlines()
                      if line.startswith('HETATM') and resname_matches(line[17:20], resname)
                      and line[21:22] == chain and line[16:17] in ' A')
    mol = Chem.MolFromPDBBlock(block + '\nEND\n', removeHs=True)
    return AllChem.AssignBondOrdersFromTemplate(Chem.MolFromSmiles(smiles), mol)


def prepare_receptor(chain_pdb, basename, center, size, required=('ZN',), protect_radius=12.):
    """Meeko template preparation. Residues failing templates (e.g. truncated side chains) are removed
    only if every atom lies farther than protect_radius from the box centre; removals are returned."""
    command = [str(MK_RECEPTOR), '--read_pdb', str(chain_pdb), '-o', str(basename), '-p', '-v',
               '--default_altloc', 'A', '--box_center', *map(str, center), '--box_size', *map(str, size)]
    env = {'_ZO_DOCTOR': '0', 'PATH': '/usr/bin:/bin'}
    run = subprocess.run(command, capture_output=True, text=True, env=env)
    removed = []
    # Some residues carry altlocs B/C with no A, so a blanket --default_altloc A is rejected.
    missing = re.search(r"Requested altlocs not found for: \[(.*?)\]", run.stdout + run.stderr)
    if run.returncode != 0 and missing:
        available = {}
        for line in Path(chain_pdb).read_text().splitlines():
            if line.startswith(('ATOM', 'HETATM')) and line[16] != ' ':
                available.setdefault(f'{line[21]}:{line[22:26].strip()}', set()).add(line[16])
        wanted = [f'{item}={sorted(available[item])[0]}' for item in
                  (i.strip(" '") for i in missing.group(1).split(',')) if item in available]
        if wanted:
            command = command + ['--wanted_altloc', ','.join(wanted)]
            run = subprocess.run(command, capture_output=True, text=True, env=env)
    if run.returncode != 0:
        failed = re.search(r"Template matching failed for: \[(.*?)\]", run.stdout + run.stderr)
        if not failed:
            raise RuntimeError(run.stdout[-2000:] + run.stderr[-2000:])
        removed = [item.strip(" '") for item in failed.group(1).split(',')]
        atoms = [line for line in Path(chain_pdb).read_text().splitlines() if line.startswith(('ATOM', 'HETATM'))]
        for item in removed:
            chain, number = item.split(':')
            xyz = np.array([[float(l[30:38]), float(l[38:46]), float(l[46:54])] for l in atoms
                            if l[21] == chain and l[22:26].strip() == number])
            if np.linalg.norm(xyz - np.asarray(center, float), axis=1).min() <= protect_radius:
                raise RuntimeError(f'Unmatched residue {item} lies within {protect_radius} A of the pocket')
        subprocess.run(command + ['--allow_bad_res'], check=True, capture_output=True, text=True, env=env)
    pdbqt = Path(f'{basename}.pdbqt'); text = pdbqt.read_text()
    for element in required:
        if f' {element} ' not in text:
            raise ValueError(f'{element} missing from prepared receptor; refusing metal-free docking')
    return pdbqt, removed


def zinc_contacts(mol, zinc, cutoff=2.8, conf_id=-1):
    """Ligand N/O atoms within cutoff of the zinc: the zinc-binding group engagement."""
    xyz = mol.GetConformer(conf_id).GetPositions()
    polar = [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() in ('N', 'O')]
    return int((np.linalg.norm(xyz[polar] - zinc, axis=1) <= cutoff).sum())


def receptor_zinc(pdbqt):
    line = next(l for l in Path(pdbqt).read_text().splitlines() if l[12:16].strip() == 'ZN')
    return np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])


def ligand_pdbqt(smiles, seed):
    """Protonation as given in the parent SMILES; one ETKDG conformer as the Vina start."""
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    if AllChem.EmbedMolecule(mol, randomSeed=seed) != 0:
        raise ValueError(f'Embedding failed: {smiles}')
    AllChem.MMFFOptimizeMolecule(mol)
    setups = MoleculePreparation().prepare(mol)
    text, ok, error = PDBQTWriterLegacy.write_string(setups[0])
    if not ok:
        raise ValueError(error)
    return text


def dock(receptor, ligand, center, size, seed, exhaustiveness=16, poses=9, cpu=4, scoring='vina'):
    vina = Vina(sf_name=scoring, cpu=cpu, seed=seed, verbosity=0)
    vina.set_receptor(str(receptor)); vina.set_ligand_from_string(ligand)
    vina.compute_vina_maps(center=list(center), box_size=list(size))
    vina.dock(exhaustiveness=exhaustiveness, n_poses=poses)
    return vina.poses(n_poses=poses), vina.energies(n_poses=poses)[:, 0]


def pose_molecule(pdbqt_text):
    """Docked poses as conformers of one heavy-atom RDKit molecule (conformer i = pose i)."""
    mol = RDKitMolCreate.from_pdbqt_mol(PDBQTMolecule(pdbqt_text, skip_typing=True))[0]
    return Chem.RemoveHs(mol)


def pose_rmsds(poses, reference):
    """Heavy-atom RMSD in the receptor frame (no superposition), symmetry-aware, for every pose."""
    return [float(rdMolAlign.CalcRMS(poses, reference, prbId=c.GetId())) for c in poses.GetConformers()]


def centroid(mol):
    return mol.GetConformer().GetPositions().mean(axis=0)


def max_extent(mol):
    xyz = mol.GetConformer().GetPositions()
    return float(np.ptp(xyz, axis=0).max())


def summarise_runs(runs, threshold=2.):
    """Success rates from a validation runs table, overall and split by ligand chemotype.

    Hydroxamates and benzamides engage the catalytic zinc differently, so a pooled success
    rate can hide a protocol that works for one chemotype and fails for the other.
    """
    import pandas as pd
    flags = runs.assign(top1_success=runs.top1_rmsd <= threshold,
                        top3_success=runs.best_top3_rmsd <= threshold,
                        sampled=runs.best_any_rmsd <= threshold,
                        top1_zinc_engaged=runs.top1_zinc_contacts >= 1)
    columns = ['top1_success', 'top3_success', 'sampled', 'top1_zinc_engaged']
    overall = flags.groupby(['mode', 'scoring'])[columns].mean().round(3)
    by_chemotype = flags.groupby(['chemotype', 'mode', 'scoring'])[columns].mean().round(3)
    median_rmsd = flags.groupby(['chemotype', 'mode', 'scoring']).top1_rmsd.median().round(2)
    per_complex = flags.groupby(['pdb', 'mode', 'scoring']).top1_rmsd.median().unstack('scoring').round(2)
    return {'overall': overall, 'by_chemotype': by_chemotype,
            'median_top1_rmsd_by_chemotype': median_rmsd, 'per_complex': per_complex}


def pick_scoring(summary, mode='crossdock'):
    """Scoring function with the best top-1 success in the given mode; zinc engagement breaks ties."""
    table = summary['overall'].xs(mode, level='mode')
    ranked = table.sort_values(['top1_success', 'top1_zinc_engaged'], ascending=False)
    return str(ranked.index[0])


def pose_scores(pdbqt_text):
    """Vina scores recorded in a saved pose file, so a finished run need not be repeated."""
    return np.array([float(line.split(':')[1].split()[0]) for line in pdbqt_text.splitlines()
                     if line.startswith('REMARK VINA RESULT')])
