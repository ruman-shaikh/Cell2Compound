import numpy as np
import pandas as pd
import pytest
from cell2compound.pipeline import (ResponseData, attainable_fraction, fit_predictions,
                                    replicate_floor, select_genes)


def synthetic(n_groups=12, n_genes=6, seed=0):
    """Minimal ResponseData with known structure, so leakage and alignment are checkable."""
    rng = np.random.default_rng(seed)
    doses = np.tile([10., 100., 1000., 10000.], n_groups // 4)
    fp = rng.integers(0, 2, size=(n_groups, 32)).astype(np.uint8)
    rep = rng.normal(scale=.05, size=(n_groups, 2, n_genes))
    signal = np.zeros((n_groups, n_genes)); signal[:, 0] = np.log10(doses)
    rep += signal[:, None, :]
    groups = pd.DataFrame({'compound': [f'c{i}' for i in range(n_groups)], 'dose': doses,
                           'rep_delta': list(rep), 'rep_cells': [[50, 50]] * n_groups,
                           'table_rows': [np.array([i]) for i in range(n_groups)],
                           'canonical_smiles': [f's{i}' for i in range(n_groups)]})
    return ResponseData(groups=groups, full_delta=rep.mean(1), rep_difference=rep[:, 0] - rep[:, 1],
                        fingerprints=fp, features=np.c_[fp, np.log10(doses)], doses=doses,
                        canonical=groups.canonical_smiles.to_numpy(),
                        detected=np.full((n_groups + 1, n_genes), 100), cells=np.full(n_groups + 1, 100),
                        control_rows=np.array([n_groups]), gene_names=np.array([f'g{i}' for i in range(n_genes)]),
                        mapping=pd.DataFrame(), unmatched=[], skipped=[])


def test_attainable_fraction_is_interpretable_and_guards_undefined():
    # Model halfway between baseline and the noise floor.
    assert attainable_fraction(.15, .20, .10) == pytest.approx(.5)
    assert attainable_fraction(.20, .20, .10) == pytest.approx(0.)
    assert attainable_fraction(.10, .20, .10) == pytest.approx(1.)
    # Worse than the baseline is negative, not clipped away.
    assert attainable_fraction(.25, .20, .10) == pytest.approx(-.5)
    # Baseline already at the floor: undefined, must not be a huge number.
    assert attainable_fraction(.10, .10, .10) is None
    assert attainable_fraction(.05, .10, .12) is None
    # Headroom too small to divide by: a real fold had baseline 0.0761 vs floor 0.0758 and the
    # ratio came out at -21, which is a reporting artifact rather than a finding.
    assert attainable_fraction(.0818, .0761, .0758) is None
    # Just above the threshold it is reported again.
    assert attainable_fraction(.09, .10, .09) == pytest.approx(1.)


def test_replicate_floor_is_half_the_replicate_difference():
    data = synthetic()
    genes = np.arange(data.full_delta.shape[1])
    rows = np.arange(len(data.groups))
    expected = np.sqrt(np.mean(data.rep_difference[rows][:, genes] ** 2)) / 2
    assert replicate_floor(data, rows, genes) == pytest.approx(expected)


def test_select_genes_uses_only_training_groups():
    data = synthetic(n_groups=12, n_genes=6)
    # Give gene 3 huge variance in the held-out groups only; it must not be selected.
    data.full_delta[8:, 3] += np.array([10., -10., 10., -10.])
    train = np.arange(8)
    genes = select_genes(data, train, n_genes=2, min_detection=0.)
    assert 3 not in genes
    # With those groups in training, it becomes the top-variance gene.
    genes_with = select_genes(data, np.arange(12), n_genes=2, min_detection=0.)
    assert 3 in genes_with


def test_select_genes_respects_detection_floor():
    data = synthetic()
    data.detected = np.zeros_like(data.detected)
    data.detected[:, :2] = 100
    genes = select_genes(data, np.arange(8), n_genes=6, min_detection=.5)
    assert set(genes) <= {0, 1}


def test_dose_mean_baseline_uses_only_fitted_groups():
    data = synthetic(n_groups=12, n_genes=4)
    genes = np.arange(4)
    y = data.full_delta[:, genes]
    train, valid, test = np.arange(6), np.arange(6, 8), np.arange(8, 12)
    predictions, _ = fit_predictions(data, y, train, valid, test, seed=0, device='cpu',
                                     models=['zero_response', 'training_mean_response', 'dose_mean_response'])
    fit = np.r_[train, valid]
    for row, group in enumerate(test):
        expected = y[fit][data.doses[fit] == data.doses[group]].mean(0)
        assert np.allclose(predictions['dose_mean_response'][row], expected)
    assert np.all(predictions['zero_response'] == 0)
    assert predictions['dose_mean_response'].shape == (len(test), len(genes))


def test_subset_keeps_arrays_aligned():
    data = synthetic(n_groups=12)
    keep = np.array([0, 5, 7])
    part = data.subset(keep)
    assert list(part.groups.compound) == ['c0', 'c5', 'c7']
    assert np.array_equal(part.doses, data.doses[keep])
    assert np.array_equal(part.features, data.features[keep])
    assert np.array_equal(part.full_delta, data.full_delta[keep])
    mask = np.zeros(12, bool); mask[keep] = True
    assert np.array_equal(data.subset(mask).doses, part.doses)
