import numpy as np
from cell2compound.expression import fit_mlp, predict_mlp, response_metrics, tanimoto


def test_metrics_perfect_and_top_genes():
    rng = np.random.default_rng(0)
    observed = rng.normal(size=(4, 100))
    perfect = response_metrics(observed, observed, top_k=10)
    assert perfect['rmse'] == 0 and perfect['rmse_top10_de'] == 0
    assert np.isclose(perfect['pearson_all'], 1)
    # Errors confined to non-top genes leave the top-k metric untouched.
    predicted = observed.copy()
    weakest = np.argsort(np.abs(observed), axis=1)[:, :50]
    np.put_along_axis(predicted, weakest, 0., axis=1)
    assert response_metrics(observed, predicted, top_k=10)['rmse_top10_de'] == 0


def test_tanimoto_bounds():
    a = np.array([[1, 1, 0, 0], [0, 0, 1, 1]], dtype=np.uint8)
    similarity = tanimoto(a, a)
    assert np.allclose(np.diag(similarity), 1) and np.isclose(similarity[0, 1], 0)


def test_mlp_learns_linear_map():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(200, 5)); y = x @ rng.normal(size=(5, 3))
    model, _, _ = fit_mlp(x, y, hidden=64, dropout=0., weight_decay=0., epochs=500, device='cpu', seed=0)
    assert np.mean((predict_mlp(model, x, 'cpu') - y) ** 2) < .05 * np.var(y)


def test_product_kernel_is_psd_and_dose_aware():
    from cell2compound.expression import product_kernel
    rng = np.random.default_rng(2)
    fp = rng.integers(0, 2, size=(12, 64)).astype(np.uint8)
    dose = np.repeat([10., 100., 1000.], 4)
    kernel = product_kernel(fp, dose, fp, dose, bandwidth=1.)
    assert np.allclose(kernel, kernel.T)
    assert np.linalg.eigvalsh(kernel).min() > -1e-8
    # Same structure, different dose: similarity falls off with the dose gap.
    same = product_kernel(fp[:1], np.array([10.]), fp[:1], np.array([10.]), 1.)[0, 0]
    far = product_kernel(fp[:1], np.array([10.]), fp[:1], np.array([10000.]), 1.)[0, 0]
    assert same > far
