import numpy as np
import pandas as pd
import pytest
from cell2compound.chemistry import featurize


def test_equivalent_smiles_and_invalid_records():
    frame = pd.DataFrame({'compound_id': ['a', 'b', 'bad', 'empty'],
                          'smiles': ['CCO', 'OCC', 'invalid', '']})
    descriptors, fingerprints, rejected = featurize(frame)
    assert list(descriptors.compound_id) == ['a', 'b']
    assert descriptors.canonical_smiles.nunique() == 1
    assert fingerprints.shape == (2, 2048)
    assert np.array_equal(fingerprints[0], fingerprints[1])
    assert set(rejected.compound_id) == {'bad', 'empty'}
    assert descriptors.molecular_weight.iloc[0] == pytest.approx(46.069, abs=.001)


def test_duplicate_ids_fail():
    with pytest.raises(ValueError, match='unique'):
        featurize(pd.DataFrame({'compound_id': ['a', 'a'], 'smiles': ['C', 'CC']}))
