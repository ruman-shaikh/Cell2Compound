import numpy as np
from cell2compound.splits import compound_split


def test_repeated_doses_and_aliases_stay_together():
    structures=np.repeat([f'structure_{i}' for i in range(30)],4)
    labels=compound_split(structures)
    assert set(labels)=={'train','validation','test'}
    for structure in np.unique(structures):
        assert len(set(labels[structures==structure]))==1
    assert np.array_equal(labels,compound_split(structures))
