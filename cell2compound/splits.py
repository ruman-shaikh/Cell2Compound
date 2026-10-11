"""Keep each canonical molecular structure in one evaluation partition."""
import numpy as np
from sklearn.model_selection import GroupShuffleSplit


def compound_split(canonical_smiles, seed=42):
    groups=np.asarray(canonical_smiles)
    if len(np.unique(groups))<10:
        raise ValueError('At least ten distinct structures required for three-way evaluation')
    trainval,test=next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=seed).split(groups,groups=groups))
    tr,va=next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=seed+1).split(trainval,groups=groups[trainval]))
    labels=np.full(len(groups),'test',dtype=object)
    labels[trainval[tr]]='train'; labels[trainval[va]]='validation'
    return labels
