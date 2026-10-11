"""Validated molecular representations; preserve failed input records."""
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold


def featurize(frame, bits=2048, radius=2):
    if not {'compound_id', 'smiles'}.issubset(frame.columns):
        raise ValueError('Input requires compound_id and smiles columns')
    if frame.compound_id.isna().any() or frame.compound_id.duplicated().any():
        raise ValueError('compound_id must be present and unique')
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=bits)
    records, vectors, rejected = [], [], []
    for row in frame.itertuples(index=False):
        mol = Chem.MolFromSmiles(row.smiles) if isinstance(row.smiles, str) else None
        if mol is None or mol.GetNumAtoms() == 0:
            rejected.append({'compound_id': row.compound_id, 'smiles': row.smiles,
                             'reason': 'invalid_or_empty_smiles'})
            continue
        canonical = Chem.MolToSmiles(mol, isomericSmiles=True)
        vector = np.zeros(bits, dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(generator.GetFingerprint(mol), vector)
        vectors.append(vector)
        records.append(dict(compound_id=row.compound_id, canonical_smiles=canonical,
                            scaffold=MurckoScaffold.MurckoScaffoldSmiles(mol=mol),
                            molecular_weight=Descriptors.MolWt(mol),
                            logp=Descriptors.MolLogP(mol), tpsa=Descriptors.TPSA(mol),
                            hbd=Descriptors.NumHDonors(mol), hba=Descriptors.NumHAcceptors(mol),
                            rotatable_bonds=Descriptors.NumRotatableBonds(mol)))
    return (pd.DataFrame(records), np.stack(vectors) if vectors else np.empty((0, bits), dtype=np.uint8),
            pd.DataFrame(rejected, columns=['compound_id', 'smiles', 'reason']))
