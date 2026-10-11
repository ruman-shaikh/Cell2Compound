import numpy as np
import pandas as pd
import pytest
from cell2compound.mechanism import (assign_classes, class_folds, chembl_classes, hdac_compounds,
                                     magnitude_strata, normalise, zinc_binding_group)

# Real structures: vorinostat and panobinostat are hydroxamates, entinostat is an
# ortho-aminoanilide, aspirin is neither.
VORINOSTAT = 'ONC(=O)CCCCCCC(=O)Nc1ccccc1'
PANOBINOSTAT = 'CCNCc1ccc2cc(/C=C/C(=O)NO)ccc2c1'
ENTINOSTAT = 'Nc1ccccc1NC(=O)c1ccc(CNC(=O)OCc2cccnc2)cc1'
ASPIRIN = 'CC(=O)Oc1ccccc1C(=O)O'


def test_zinc_binding_group_detects_both_chemotypes():
    assert zinc_binding_group(VORINOSTAT) == 'hydroxamate'
    assert zinc_binding_group(PANOBINOSTAT) == 'hydroxamate'
    assert zinc_binding_group(ENTINOSTAT) == 'o_aminoanilide'
    assert zinc_binding_group(ASPIRIN) is None
    assert zinc_binding_group('not a smiles') is None
    assert zinc_binding_group(None) is None


def test_normalise_treats_placeholders_as_missing():
    assert normalise('HDAC') == 'HDAC'
    assert normalise('  HDAC ') == 'HDAC'
    for missing in ['nan', 'NaN', '', 'None', 'control', 'Vehicle', 'NA']:
        assert normalise(missing) is None


def frame():
    return pd.DataFrame({'compound': ['Vorinostat', 'Panobinostat', 'Entinostat', 'Aspirin'],
                         'parent_smiles': [VORINOSTAT, PANOBINOSTAT, ENTINOSTAT, ASPIRIN]})


def annotations():
    return pd.DataFrame({'compound': ['Vorinostat', 'Panobinostat', 'Entinostat', 'Aspirin'],
                         'target': ['HDAC', 'HDAC', 'HDAC', 'nan'],
                         'pathway_level_1': ['Epigenetic regulation'] * 3 + ['Other'],
                         'pathway_level_2': ['Histone deacetylation'] * 3 + ['nan']})


def test_assign_classes_from_annotation_and_marks_missing_as_none():
    out = assign_classes(frame(), annotations(), source='target')
    assert list(out.mechanism_class) == ['HDAC', 'HDAC', 'HDAC', None]
    assert list(out.zinc_binding_group) == ['hydroxamate', 'hydroxamate', 'o_aminoanilide', None]


def test_assign_classes_rejects_unavailable_source():
    with pytest.raises(ValueError, match='not available'):
        assign_classes(frame(), annotations(), source='no_such_column')
    with pytest.raises(ValueError, match='mechanisms table'):
        assign_classes(frame(), annotations(), source='chembl')


def test_chembl_class_prefers_the_widely_shared_target():
    # HDAC1 is shared by three structures, HDAC8 by one, so HDAC1 is the usable fold label.
    mechanisms = pd.DataFrame({
        'parent_smiles': [VORINOSTAT, VORINOSTAT, PANOBINOSTAT, ENTINOSTAT, ASPIRIN],
        'target_name': ['Histone deacetylase 1', 'Histone deacetylase 8', 'Histone deacetylase 1',
                        'Histone deacetylase 1', 'Cyclooxygenase'],
        'target_type': ['SINGLE PROTEIN'] * 5})
    lookup = chembl_classes(mechanisms, min_compounds=3)
    assert lookup[VORINOSTAT] == 'Histone deacetylase 1'
    assert lookup[PANOBINOSTAT] == 'Histone deacetylase 1'
    # Cyclooxygenase has one compound, too few to form a held-out class.
    assert ASPIRIN not in lookup


def test_hdac_compounds_found_by_chemotype_or_annotation():
    out = assign_classes(frame(), annotations(), source='target')
    assert hdac_compounds(out, annotations()) == {'Vorinostat', 'Panobinostat', 'Entinostat'}
    # Without annotations, chemotype alone still catches all three.
    assert hdac_compounds(out) == {'Vorinostat', 'Panobinostat', 'Entinostat'}


def test_class_folds_require_enough_distinct_structures():
    classes = pd.DataFrame({
        'compound': ['a', 'b', 'c', 'd', 'e', 'f'],
        'parent_smiles': ['C', 'CC', 'CCC', 'CCCC', 'CCCCC', 'CCCCC'],
        'mechanism_class': ['big', 'big', 'big', 'small', 'twin', 'twin']})
    folds = class_folds(classes, min_compounds=3, min_structures=3)
    assert set(folds) == {'big'}
    assert folds['big'] == ['a', 'b', 'c']
    # 'twin' has 2 compounds sharing 1 structure: holding it out would test one scaffold.
    assert 'twin' not in folds and 'small' not in folds


def test_magnitude_strata_bins_by_response_over_own_noise():
    response = np.array([0.05, 0.15, 0.30, 0.80, 0.10])
    noise = np.array([0.10, 0.10, 0.10, 0.10, 0.0])
    labels, ratio, names = magnitude_strata(response, noise, edges=(1., 2., 4.))
    assert names == ['<1x', '1-2x', '2-4x', '>4x']
    assert list(labels[:4]) == ['<1x', '1-2x', '2-4x', '>4x']
    # Zero noise cannot give a ratio; it must not silently become a bin.
    assert labels[4] == 'unknown' and np.isnan(ratio[4])


def test_sirtuin_inhibitors_are_not_counted_as_zinc_hdac():
    # Sirtuins are class III histone deacetylases but NAD+-dependent, with no zinc pharmacophore.
    frame = pd.DataFrame({'compound': ['Sirtinol', 'CUDC-907', 'Vorinostat'],
                          'parent_smiles': [ASPIRIN, ASPIRIN, VORINOSTAT]})
    annotations = pd.DataFrame({'compound': ['Sirtinol', 'CUDC-907', 'Vorinostat'],
                                'target': ['Sirtuin', 'HDAC,PI3K', 'HDAC'],
                                'pathway_level_2': ['Histone deacetylation'] * 3})
    out = assign_classes(frame, annotations, source='target')
    assert hdac_compounds(out, annotations) == {'CUDC-907', 'Vorinostat'}
    # A dual HDAC inhibitor is kept even without a zinc-binding group in this fixture.
    assert hdac_compounds(out, annotations, include_sirtuin=True) == {'Sirtinol', 'CUDC-907', 'Vorinostat'}
