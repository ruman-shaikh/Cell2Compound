"""Pseudobulk response targets, response metrics, and a small torch MLP."""
import numpy as np
import torch


def response_metrics(observed, predicted, top_k=50):
    """RMSE over all genes and over each group's top-k observed |response| genes."""
    top = np.argsort(-np.abs(observed), axis=1)[:, :top_k]
    rows = np.arange(len(observed))[:, None]
    def pearson(a, b):
        a = a - a.mean(axis=1, keepdims=True); b = b - b.mean(axis=1, keepdims=True)
        denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
        return np.divide((a * b).sum(axis=1), denom, out=np.full(len(a), np.nan), where=denom > 0)
    return {'rmse': float(np.sqrt(np.mean((observed - predicted) ** 2))),
            f'rmse_top{top_k}_de': float(np.sqrt(np.mean((observed[rows, top] - predicted[rows, top]) ** 2))),
            f'pearson_top{top_k}_de': float(np.nanmean(pearson(observed[rows, top], predicted[rows, top]))),
            'pearson_all': float(np.nanmean(pearson(observed, predicted)))}


def tanimoto(a, b):
    a = a.astype(np.float32); b = b.astype(np.float32)
    inter = a @ b.T
    return inter / (a.sum(1)[:, None] + b.sum(1)[None, :] - inter + 1e-9)


def dose_kernel(a, b, bandwidth):
    """RBF over log10 dose; bandwidth in log10 units."""
    return np.exp(-((a[:, None] - b[None, :]) / bandwidth) ** 2)


def product_kernel(fp_a, dose_a, fp_b, dose_b, bandwidth):
    """Tanimoto kernel on fingerprints times an RBF on log dose; a product of PSD kernels is PSD."""
    return tanimoto(fp_a, fp_b) * dose_kernel(np.log10(dose_a), np.log10(dose_b), bandwidth)


class MLP(torch.nn.Module):
    def __init__(self, inputs, outputs, hidden, dropout):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(inputs, hidden), torch.nn.ReLU(), torch.nn.Dropout(dropout),
                                       torch.nn.Linear(hidden, hidden), torch.nn.ReLU(), torch.nn.Dropout(dropout),
                                       torch.nn.Linear(hidden, outputs))

    def forward(self, x):
        return self.net(x)


def fit_mlp(x, y, hidden, dropout, weight_decay, epochs, device, seed, x_val=None, y_val=None, patience=40):
    """Full-batch Adam. With validation data, returns the best epoch count for refitting."""
    torch.manual_seed(seed)
    model = MLP(x.shape[1], y.shape[1], hidden, dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=weight_decay)
    xt = torch.as_tensor(x, dtype=torch.float32, device=device); yt = torch.as_tensor(y, dtype=torch.float32, device=device)
    if x_val is not None:
        xv = torch.as_tensor(x_val, dtype=torch.float32, device=device); yv = torch.as_tensor(y_val, dtype=torch.float32, device=device)
    best, best_epoch, best_state = np.inf, epochs, None
    for epoch in range(1, epochs + 1):
        model.train(); opt.zero_grad()
        loss = torch.nn.functional.mse_loss(model(xt), yt); loss.backward(); opt.step()
        if x_val is not None:
            model.eval()
            with torch.no_grad():
                val = torch.nn.functional.mse_loss(model(xv), yv).item()
            if val < best:
                best, best_epoch = val, epoch
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            elif epoch - best_epoch >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, best, best_epoch


def predict_mlp(model, x, device):
    with torch.no_grad():
        return model(torch.as_tensor(x, dtype=torch.float32, device=device)).cpu().numpy()
