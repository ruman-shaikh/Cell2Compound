from rdkit import Chem
from rdkit.Chem import AllChem
from cell2compound.docking import ligand_pdbqt, pose_rmsds


def test_rmsd_is_symmetry_aware_and_zero_for_identity():
    mol = Chem.AddHs(Chem.MolFromSmiles('OC(=O)c1ccccc1'))
    AllChem.EmbedMultipleConfs(mol, 2, randomSeed=0)
    mol = Chem.RemoveHs(mol)
    reference = Chem.Mol(mol, confId=0)
    reference.RemoveAllConformers(); reference.AddConformer(Chem.Conformer(mol.GetConformer(0)), assignId=True)
    rmsd = pose_rmsds(mol, reference)
    assert rmsd[0] < 1e-6 and len(rmsd) == 2


def test_ligand_preparation_writes_torsion_tree():
    text = ligand_pdbqt('ONC(=O)CCCCCCC(=O)Nc1ccccc1', seed=1)
    assert 'ROOT' in text and 'TORSDOF' in text
