"""Map observed perturbation names to source structures; reduce salts to parents."""
import re
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize

RDLogger.DisableLog('rdApp.*')


def parent_smiles(smiles):
    """Disconnect metals, keep the largest organic fragment, neutralise charges."""
    mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
    if mol is None:
        return None
    parent = rdMolStandardize.ChargeParent(mol)
    if parent.GetNumHeavyAtoms() == 0:
        return None
    return Chem.MolToSmiles(parent, isomericSmiles=True)


def first_token(name):
    """Source table truncates names to their first word, without stereo prefixes."""
    return re.sub(r'^\([+-]\)-', '', name.strip()).split()[0]


def load_source(path):
    source = pd.read_csv(path, header=None, names=['source_name', 'smiles', 'category'])
    source = source[source.source_name != 'control']
    if (source.groupby('source_name').smiles.nunique() > 1).any():
        raise ValueError('Compound table contains conflicting structures')
    source = source.drop_duplicates('source_name').reset_index(drop=True)
    source['parent_smiles'] = source.smiles.map(parent_smiles)
    return source


def map_observed_names(observed, source):
    """Exact name first, then first-token match. Every rule must resolve uniquely."""
    by_name = dict(zip(source.source_name, source.index))
    rows, unmatched = [], []
    for name in sorted(set(observed)):
        if name in by_name:
            rule, index = 'exact', by_name[name]
        elif first_token(name) in by_name:
            rule, index = 'first_token', by_name[first_token(name)]
        else:
            unmatched.append(name)
            continue
        record = source.loc[index]
        rows.append({'observed_name': name, 'source_name': record.source_name, 'rule': rule,
                     'source_smiles': record.smiles, 'parent_smiles': record.parent_smiles,
                     'salt_stripped': record.smiles != record.parent_smiles,
                     'category': record.category})
    return pd.DataFrame(rows), unmatched
