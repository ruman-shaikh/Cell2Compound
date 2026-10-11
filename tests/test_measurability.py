import numpy as np
import pytest
from cell2compound.expression import tanimoto
from cell2compound.measurability import (class_onehot, class_permutation,
                                         dynamic_range_fraction, interpolated_duplicate,
                                         noise_metrics, overlap_coefficient, reliability_alpha)


def test_noise_metrics_subtracts_noise_in_quadrature():
    # rms^2 = signal^2 + floor^2 by construction: a response of 3 measured with floor 4 gives 5.
    full = np.full((1, 100), 5.); rep = np.full((1, 100), 8.)   # floor = 8/2 = 4
    m = noise_metrics(full, rep)
    assert m['floor'][0] == pytest.approx(4.)
    assert m['signal'][0] == pytest.approx(3.)
    assert m['snr'][0] == pytest.approx(.75)


def test_response_entirely_within_noise_clips_to_zero_not_negative():
    full = np.full((1, 50), 1.); rep = np.full((1, 50), 6.)     # floor 3 > rms 1
    m = noise_metrics(full, rep)
    assert m['signal'][0] == 0.
    assert m['snr'][0] == 0.


def test_noise_metrics_respects_gene_subset():
    full = np.c_[np.ones((1, 10)), np.full((1, 10), 9.)]
    rep = np.zeros((1, 20))
    assert noise_metrics(full, rep, genes=np.arange(10))['rms'][0] == pytest.approx(1.)
    assert noise_metrics(full, rep, genes=np.arange(10, 20))['rms'][0] == pytest.approx(9.)


def test_overlap_coefficient_bounds():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 20000)
    bins = np.linspace(-8, 12, 60)
    assert overlap_coefficient(a, a, bins) == pytest.approx(1., abs=.02)
    assert overlap_coefficient(a, a + 10, bins) < .01


def test_within_class_permutation_preserves_every_label():
    labels = np.array(['a', 'a', 'a', 'b', 'b', 'c'])
    mapping = class_permutation(labels, within_class=True, rng=np.random.default_rng(1))
    assert (labels[mapping] == labels).all()
    assert sorted(mapping) == list(range(len(labels)))


def test_global_permutation_is_allowed_to_break_labels():
    labels = np.array(['a'] * 10 + ['b'] * 10)
    moved = [(labels[class_permutation(labels, False, np.random.default_rng(s))] != labels).any()
             for s in range(5)]
    assert any(moved), 'a global shuffle that never crosses classes would not be a valid control'


def test_singleton_class_is_a_fixed_point():
    labels = np.array(['a', 'a', 'solo'])
    mapping = class_permutation(labels, within_class=True, rng=np.random.default_rng(3))
    assert mapping[2] == 2


def test_tanimoto_on_onehot_is_the_class_identity_kernel():
    labels = np.array(['x', 'y', 'x'])
    k = tanimoto(class_onehot(labels), class_onehot(labels))
    assert k[0, 2] == pytest.approx(1., abs=1e-5)
    assert k[0, 1] == pytest.approx(0., abs=1e-5)


def test_reliability_is_near_one_for_signal_and_near_zero_for_noise():
    rng = np.random.default_rng(0)
    truth = rng.normal(0, 3, (400, 1))                       # gene 0: large real spread
    signal_gene = np.c_[truth, np.zeros((400, 1))]           # gene 1: no real spread at all
    noise_a = rng.normal(0, .05, (400, 2)); noise_b = rng.normal(0, .05, (400, 2))
    alpha = reliability_alpha(signal_gene + noise_a, signal_gene + noise_b)
    assert alpha[0] > .99
    assert alpha[1] < .05


def test_interpolated_duplicate_collapses_to_baseline_at_zero_reliability():
    rep = np.full((3, 2), 7.); baseline = np.zeros((3, 2))
    assert interpolated_duplicate(rep, baseline, np.zeros(2)) == pytest.approx(baseline)
    assert interpolated_duplicate(rep, baseline, np.ones(2)) == pytest.approx(rep)


def test_dynamic_range_fraction_anchors_at_zero_and_one():
    # A model level with the negative baseline scores 0; level with the control scores 1.
    assert dynamic_range_fraction(.5, .5, .1) == pytest.approx(0.)
    assert dynamic_range_fraction(.1, .5, .1) == pytest.approx(1.)
    assert dynamic_range_fraction(.3, .5, .1) == pytest.approx(.5)
    assert dynamic_range_fraction(.05, .5, .1) > 1., 'beating the control must be representable'


def test_dynamic_range_fraction_is_undefined_when_references_coincide():
    # No separation between baseline and positive control: the ratio would explode.
    assert dynamic_range_fraction(.2, .5, .5) is None
    assert dynamic_range_fraction(.2, .5, .499) is None
