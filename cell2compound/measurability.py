"""Replicate-derived measurability of a perturbation response, and feature-ablation arms.

The central quantity is whether a group's response is larger than the error of the measurement
that produced it. `rms` of an observed mean delta already contains replicate noise, so comparing
it to the noise floor directly understates how much is noise; subtracting in quadrature does not.
"""
import numpy as np


def noise_metrics(full_delta, rep_difference, genes=None):
    """Per-group rms, replicate floor, noise-corrected signal and snr.

    floor  = RMSE(rep1 - rep2) / 2, the error of the two-replicate mean (no model beats it).
    signal = sqrt(max(rms^2 - floor^2, 0)); snr = signal / floor.
    snr < 1 means the response is below its own measurement error.
    """
    if genes is not None:
        full_delta = full_delta[:, genes]
        rep_difference = rep_difference[:, genes]
    rms = np.sqrt((full_delta ** 2).mean(1))
    floor = np.sqrt((rep_difference ** 2).mean(1)) / 2
    signal = np.sqrt(np.maximum(rms ** 2 - floor ** 2, 0))
    with np.errstate(divide='ignore', invalid='ignore'):
        snr = np.where(floor > 0, signal / np.maximum(floor, 1e-12), np.nan)
    return {'rms': rms, 'floor': floor, 'signal': signal, 'snr': snr}


def overlap_coefficient(a, b, bins):
    """Histogram overlap of two samples: integral of the pointwise minimum of their densities.

    1.0 means identical distributions, 0.0 means disjoint support. Used to decide whether a
    covariate can be conditioned on by stratification or is too collinear with the exposure.
    """
    width = np.diff(bins)[0]
    ha, _ = np.histogram(a, bins=bins, density=True)
    hb, _ = np.histogram(b, bins=bins, density=True)
    return float(np.minimum(ha, hb).sum() * width)


def class_permutation(labels, within_class, rng):
    """Index permutation that either shuffles inside each label block or across everything.

    Within-class keeps every item's label and changes only which member's features it receives;
    singleton classes are necessarily fixed points.
    """
    labels = np.asarray(labels)
    mapping = np.arange(len(labels))
    blocks = ([np.flatnonzero(labels == c) for c in np.unique(labels)] if within_class
              else [np.arange(len(labels))])
    for block in blocks:
        if len(block) > 1:
            mapping[block] = rng.permutation(block)
    return mapping


def reliability_alpha(rep_a, rep_b, floor_variance=None):
    """Per-gene reliability: the share of a gene's across-group variance that is real signal.

    alpha_g = s_g^2 / (s_g^2 + n_g^2), with s^2 the across-group variance of the response and n^2
    the per-gene replicate noise variance. 0 means a gene's spread is entirely noise, 1 means it is
    entirely signal. This is the standard reliability ratio.

    It is NOT the rule Miller et al. use. Theirs (`data/add_interpolated_baseline.py` in
    shiftbioscience/Perturbation-Models-Outperform-Baselines) is `alpha = 1 - p_adj`, from an
    adjusted differential-expression p-value computed per gene AND per condition, with genes absent
    from the test given p = 1 and hence alpha = 0. Two consequences of the difference:

      - ours is per gene only, shared across conditions, so it cannot credit a control for
        recovering a response pattern specific to one condition. That makes it the flatter and more
        conservative control of the two.
      - ours needs no differential-expression test, which is why it is usable here.

    The blend itself (`interpolated_duplicate`) is their formula exactly.
    """
    mean = (rep_a + rep_b) / 2
    noise = np.var(rep_a - rep_b, axis=0) / 4 if floor_variance is None else floor_variance
    total = np.var(mean, axis=0)
    signal = np.maximum(total - noise, 0)
    with np.errstate(divide='ignore', invalid='ignore'):
        alpha = np.where(signal + noise > 0, signal / np.maximum(signal + noise, 1e-12), 0.)
    return np.clip(alpha, 0., 1.)


def interpolated_duplicate(rep, baseline, alpha):
    """Positive control: a technical duplicate shrunk per gene toward the baseline prediction.

    A raw duplicate is an over-optimistic ceiling on genes whose spread is mostly noise, because it
    shares that noise with the target. Shrinking each gene by its reliability gives a control that
    is actually achievable: on a pure-noise gene it collapses to the baseline and is credited with
    no skill, on a high-reliability gene it keeps the duplicate's information.

    `rep` must be a replicate that is NOT part of the target being scored against, or the control
    and the target share noise and the resulting ceiling is unreachable by construction.
    """
    return baseline + alpha * (rep - baseline)


def dynamic_range_fraction(model_error, negative_error, positive_error, min_range_ratio=.05):
    """Where a model sits on the range between an uninformative baseline and a positive control.

    0 means no better than the negative baseline, 1 means it matched the positive control, and
    values above 1 are possible when a model beats the control. Returns None when the two
    references are too close to separate, for the same reason `attainable_fraction` does: the
    ratio's denominator approaches zero and the quantity becomes a reporting artifact.

    This is our implementation of the published calibrated-metric idea, not a port of their code.
    """
    span = negative_error - positive_error
    if span <= max(1e-9, min_range_ratio * max(positive_error, 1e-12)):
        return None
    return float((negative_error - model_error) / span)


def class_onehot(labels):
    """One-hot encoding of labels as uint8.

    A Tanimoto kernel on these rows is exactly the class-identity kernel (1 within a class,
    0 between), which is the clean 'class information only, no structure' feature set.
    """
    labels = np.asarray(labels)
    order = list(np.unique(labels))
    out = np.zeros((len(labels), len(order)), dtype=np.uint8)
    out[np.arange(len(labels)), [order.index(c) for c in labels]] = 1
    return out
