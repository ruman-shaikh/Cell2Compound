"""Mechanism-class labels for sci-Plex compounds, from three independent sources.

The question these support: does a structure-based response model generalise, or does it
recognise a few mechanism classes with large transcriptional signatures? Answering that needs
class labels that were not derived from the expression data itself.

Sources, in decreasing precision:
  target            - sci-Plex curated target family (e.g. "HDAC"), from the h5ad obs
  pathway_level_2   - sci-Plex curated process (e.g. "Histone deacetylation")
  pathway_level_1   - sci-Plex coarse pathway (e.g. "Epigenetic regulation")
  category          - the vendor category in the source SMILES table (e.g. "Epigenetics")
  chembl            - curated ChEMBL mechanism target, independent of sci-Plex
"""
import numpy as np
import pandas as pd
from rdkit import Chem

OBS_COLUMNS = ('target', 'pathway_level_1', 'pathway_level_2')
# Zinc-binding groups that define the HDAC-inhibitor chemotypes in this library.
ZINC_BINDING_GROUPS = {'hydroxamate': '[CX3](=O)[NX3H][OX2H]',
                       'o_aminoanilide': 'c([NH2])c[NH]C(=O)'}
MISSING = {'nan', 'none', '', 'na', 'control', 'vehicle'}


def extract_annotations(h5ad, cache=None, drug_key='perturbation'):
    """One row per perturbation from the h5ad obs. Cached, since this touches the raw file.

    Fails loudly if an annotation is not constant within a perturbation, rather than silently
    taking the first value.
    """
    cache = cache and pd.io.common.stringify_path(cache)
    if cache and pd.io.common.file_exists(cache):
        # Round-trip as strings so a cached read matches a fresh extraction; read_csv would
        # otherwise turn the 'nan' placeholders back into floats.
        return pd.read_csv(cache, keep_default_na=False, dtype=str)
    import anndata as ad
    handle = ad.read_h5ad(h5ad, backed='r')
    obs = handle.obs
    present = [c for c in OBS_COLUMNS if c in obs.columns]
    if not present:
        raise ValueError(f'None of {OBS_COLUMNS} in obs; available: {list(obs.columns)}')
    frame = obs[[drug_key, *present]].astype(str).drop_duplicates()
    handle.file.close()
    counts = frame.groupby(drug_key).size()
    conflicting = counts[counts > 1]
    if len(conflicting):
        raise ValueError(f'Annotations vary within a perturbation: {list(conflicting.index[:5])}')
    frame = frame.rename(columns={drug_key: 'compound'}).reset_index(drop=True)
    if cache:
        frame.to_csv(cache, index=False)
    return frame


def normalise(label):
    text = str(label).strip()
    return None if text.casefold() in MISSING else text


def zinc_binding_group(smiles):
    """Zinc-binding chemotype of a structure, or None. Independent of any annotation."""
    mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
    if mol is None:
        return None
    hits = [name for name, smarts in ZINC_BINDING_GROUPS.items()
            if mol.HasSubstructMatch(Chem.MolFromSmarts(smarts))]
    return ','.join(hits) or None


def chembl_classes(mechanisms, min_compounds=3):
    """Primary ChEMBL class per structure: its most widely shared single-protein target.

    Sharing is the tie-break so that classes are populated enough to hold out; a target seen in
    one compound only cannot form a leave-one-class-out fold.
    """
    single = mechanisms[mechanisms.target_type == 'SINGLE PROTEIN'] if 'target_type' in mechanisms else mechanisms
    if not len(single):
        return {}
    shared = single.groupby('target_name').parent_smiles.nunique()
    usable = shared[shared >= min_compounds]
    best = {}
    for smiles, part in single[single.target_name.isin(usable.index)].groupby('parent_smiles'):
        best[smiles] = usable[part.target_name.unique()].idxmax()
    return best


def assign_classes(compounds, annotations=None, source='target', mechanisms=None, category_table=None):
    """Attach a mechanism class to each row of `compounds` (needs columns compound, parent_smiles).

    Returns the frame with `mechanism_class` (None where unknown) and `zinc_binding_group`.
    """
    frame = compounds.copy()
    frame['zinc_binding_group'] = frame.parent_smiles.map(zinc_binding_group)
    if source == 'chembl':
        if mechanisms is None:
            raise ValueError('source="chembl" needs the mechanisms table')
        lookup = chembl_classes(mechanisms)
        frame['mechanism_class'] = frame.parent_smiles.map(lookup).map(normalise)
    elif source == 'category':
        if category_table is None:
            raise ValueError('source="category" needs the source SMILES table')
        lookup = dict(zip(category_table.source_name, category_table.category))
        frame['mechanism_class'] = frame.source_name.map(lookup).map(normalise) if 'source_name' in frame \
            else frame.compound.map(lookup).map(normalise)
    else:
        if annotations is None or source not in annotations.columns:
            raise ValueError(f'source={source!r} not available in the annotation table')
        lookup = dict(zip(annotations.compound, annotations[source]))
        frame['mechanism_class'] = frame.compound.map(lookup).map(normalise)
    return frame


def hdac_compounds(frame, annotations=None, include_sirtuin=False):
    """Compounds that inhibit a zinc-dependent histone deacetylase.

    Matched by zinc-binding chemotype or by an annotation naming HDAC/deacetylation. Sirtuins
    are class III histone deacetylases but are NAD+-dependent rather than zinc-dependent: they
    share none of the zinc-binding pharmacophore, they form their own mechanism fold, and they
    are not the large responders this exclusion is aimed at. They are therefore left in by
    default, and a compound naming HDAC directly (e.g. a dual HDAC/PI3K inhibitor) is kept
    regardless.
    """
    flag = frame.zinc_binding_group.notna() if 'zinc_binding_group' in frame \
        else frame.parent_smiles.map(zinc_binding_group).notna()
    if annotations is not None:
        columns = [c for c in OBS_COLUMNS if c in annotations.columns]
        text = (annotations.set_index('compound')[columns].fillna('').astype(str)
                .agg(' '.join, axis=1).str.casefold())
        mapped = frame.compound.map(text).fillna('')
        named = mapped.str.contains('hdac|histone deacetyl')
        if not include_sirtuin:
            # Drop sirtuin-only hits; keep anything that names HDAC itself.
            named &= ~(mapped.str.contains('sirtuin') & ~mapped.str.contains('hdac'))
        flag = flag | named
    return set(frame.compound[flag])


def class_folds(classes, min_compounds=3, min_structures=3):
    """Leave-one-class-out folds: class -> the compounds held out.

    A class is usable only with enough distinct structures, otherwise the fold tests memorising
    one scaffold rather than generalising to an unseen mechanism.
    """
    frame = pd.DataFrame(classes).dropna(subset=['mechanism_class'])
    folds = {}
    for label, part in frame.groupby('mechanism_class'):
        if part.compound.nunique() >= min_compounds and part.parent_smiles.nunique() >= min_structures:
            folds[label] = sorted(part.compound.unique())
    return folds


def magnitude_strata(response, noise, edges=(1., 2., 4.)):
    """Label groups by observed response relative to their own replicate noise.

    Absolute response is not the right axis: a large response measured on few cells is noisy.
    The ratio says whether there is anything above noise for a model to predict.
    """
    ratio = np.divide(np.asarray(response, float), np.asarray(noise, float),
                      out=np.full(len(response), np.nan), where=np.asarray(noise, float) > 0)
    labels = np.full(len(ratio), 'unknown', dtype=object)
    bounds = [(-np.inf, edges[0]), *zip(edges, edges[1:]), (edges[-1], np.inf)]
    names = [f'<{edges[0]:g}x'] + [f'{a:g}-{b:g}x' for a, b in zip(edges, edges[1:])] + [f'>{edges[-1]:g}x']
    for (low, high), name in zip(bounds, names):
        labels[(ratio > low) & (ratio <= high)] = name
    return labels, ratio, names
