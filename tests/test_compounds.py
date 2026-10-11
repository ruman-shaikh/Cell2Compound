import pandas as pd
from cell2compound.compounds import first_token, map_observed_names, parent_smiles


def test_salts_and_metal_salts_reduce_to_neutral_parent():
    assert parent_smiles('CCCC(CCC)C(=O)[O-].[Na+]') == parent_smiles('CCCC(CCC)C(=O)O')
    assert parent_smiles('Cl.Cl.NC1CC1c1ccccc1') == parent_smiles('NC1CC1c1ccccc1')
    assert parent_smiles('NCCCC(O)(P(=O)(O)O)P(=O)(O)O[Na].O.O.O') == 'NCCCC(O)(P(=O)(O)O)P(=O)(O)O'
    assert parent_smiles('*.Cl.N[C@@H]1C[C@H]1c1ccccc1') == 'N[C@@H]1C[C@H]1c1ccccc1'
    assert parent_smiles('not a smiles') is None


def test_name_mapping_exact_then_first_token():
    source = pd.DataFrame({'source_name': ['JQ1', 'GSK', 'GSK1070916', 'Vorinostat'],
                           'smiles': ['C', 'CC', 'CCC', 'CCCC'], 'category': ['x'] * 4})
    source['parent_smiles'] = source.smiles
    mapping, unmatched = map_observed_names(['(+)-JQ1', 'GSK J1', 'GSK1070916', 'Vorinostat', 'Unknown drug'], source)
    found = dict(zip(mapping.observed_name, mapping.source_name))
    assert found == {'(+)-JQ1': 'JQ1', 'GSK J1': 'GSK', 'GSK1070916': 'GSK1070916', 'Vorinostat': 'Vorinostat'}
    assert unmatched == ['Unknown drug']
    assert first_token('ENMD-2076 L-(+)-Tartaric acid ') == 'ENMD-2076'
