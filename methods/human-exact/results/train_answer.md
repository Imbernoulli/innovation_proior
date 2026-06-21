The single-epoch scaling law for language models works beautifully as long as every training token is fresh, but that assumption is about to break. Equal scaling of parameters and tokens predicts that the next generation of large models will need tens of trillions of unique tokens, while estimates of high-quality English text suggest the supply will be exhausted within a few years and most other languages are already orders of magnitude too small. Once the unique pool is fixed, the only way to spend more compute is to repeat data or over-parameterize, yet the standard law treats a re-read token as worth exactly a fresh one and a parameter as equally valuable whether it is backed by one token or one billion. That predicts the absurd outcome that looping forever over a single sentence would drive loss to the entropy floor, so the law must be extended rather than reused.

The right fix is to keep the single-epoch backbone but make it see the difference between fresh and repeated information. Think of one pass over a token as extracting most of its value, the next pass extracting a fixed fraction of what remains, and so on. That geometric-decay story gives an effective data count: the number of fresh unique tokens that would have produced the same reduction in loss. When the number of repetitions is small, effective data is essentially the total processed tokens, matching the intuition that a few repeats are nearly as good as fresh data. When repetition grows large, effective data plateaus at a finite multiple of the unique pool, because a fixed corpus can only yield so much signal. The same argument applies symmetrically to parameters: excess parameters beyond what the data can justify also relearn the same features and add diminishing value, so raw parameter count should be replaced by an effective parameter count.

The method is called Data-Constrained Scaling Laws. It generalizes the Chinchilla form L = E + A/N^α + B/D^β by substituting effective quantities N' and D' for the raw N and D. For data, let U be the unique-token pool and R_D = max(D/U - 1, 0) the number of repetitions beyond the first epoch. The effective data is D' = U + U R*_D (1 - e^{-R_D/R*_D}), where R*_D is a learned repetition budget that controls the plateau. For parameters, compute the data-justified parameter count U_N from the single-epoch compute-optimal frontier, let R_N = max(N/U_N - 1, 0), and set N' = U_N + U_N R*_N (1 - e^{-R_N/R*_N}). The resulting law is L = E + A/N'^α + B/D'^β. With no repetition and no excess parameters it collapses exactly to the single-epoch law, and if the decay constants are removed it becomes the old law for all inputs, so everything the original law got right is preserved.

Fitting follows the same log-space machinery used for the single-epoch law. The single-epoch coefficients A, B, E, α, β are frozen to their validated values, and only the two decay constants R*_N and R*_D are learned by minimizing a Huber loss between predicted and observed log-loss, using log-sum-exp for numerical stability and L-BFGS with a grid of initializations. Outliers where loss turns upward are dropped because the form models saturation, not over-training. On the C4 data-constrained runs the fit gives R*_D ≈ 15.4 and R*_N ≈ 5.3, meaning repeated tokens stay useful for around fifteen epochs while excess parameters decay about three times faster. The practical implication is that in the data-constrained regime additional compute should be tilted toward more epochs rather than more parameters, the opposite of the single-epoch equal-scaling prescription.

```python
import numpy as np
import torch

# Single-epoch coefficients fit on the corpus (alpha tied to beta).
A_LOG, B_LOG, E_LOG = 6.255414, 7.3049974, 0.6254804
ALPHA = BETA = 0.3526596
A, B, E = np.exp(A_LOG), np.exp(B_LOG), np.exp(E_LOG)  # ~521, ~1488, ~1.87
G = ((ALPHA * A) / (BETA * B)) ** (1.0 / (ALPHA + BETA))

# Decay constants learned on repeated-data runs.
R_D_STAR, R_N_STAR = 15.387756, 5.309743
PARAMS = [A_LOG, B_LOG, E_LOG, ALPHA, BETA, R_D_STAR, R_N_STAR]


def optimal_N(C):
    return G * (C / 6.0) ** (BETA / (ALPHA + BETA))


def D_to_C(D):
    return ((G * D) ** (1.0 / (ALPHA / (ALPHA + BETA)))) * 6.0


def _fit_loss(inp, params):
    a, b, e, alpha, beta, ep_star, n_star = params
    n_eff = inp[:, 0] + inp[:, 0] * n_star * (1 - torch.exp(-inp[:, 3] / n_star))
    d_eff = inp[:, 1] + inp[:, 1] * ep_star * (1 - torch.exp(-inp[:, 2] / ep_star))
    pre = torch.stack([
        a - alpha * torch.log(n_eff),
        b - beta * torch.log(d_eff),
        e.expand(inp.shape[0])
    ])
    pred = torch.logsumexp(pre, dim=0)
    return torch.nn.functional.huber_loss(
        pred, torch.log(inp[:, 4]), delta=1e-3, reduction="none"
    ).sum()


def _fit_decay_from(inp, init, steps=50):
    p = torch.nn.Parameter(torch.tensor(init, dtype=torch.float32))
    lbfgs = torch.optim.LBFGS(
        [p], lr=1e-1, history_size=10, max_iter=20, line_search_fn="strong_wolfe"
    )

    def closure():
        lbfgs.zero_grad()
        loss = _fit_loss(inp, p)
        loss.backward()
        p.grad[:5] = 0  # freeze single-epoch coefficients
        return loss

    for _ in range(steps):
        lbfgs.step(closure)
    with torch.no_grad():
        return float(_fit_loss(inp, p)), p.detach().numpy()


def fit_decay(inp, steps=50):
    best_loss, best_params = float("inf"), None
    for rd0 in np.arange(0.0, 24.0, 4.0):
        for rn0 in np.arange(0.0, 24.0, 4.0):
            init = [A_LOG, B_LOG, E_LOG, ALPHA, BETA,
                    max(rd0, 1e-6), max(rn0, 1e-6)]
            loss, params = _fit_decay_from(inp, init, steps=steps)
            if np.isfinite(loss) and loss < best_loss:
                best_loss, best_params = loss, params
    return best_params


def scaling_law(N, D, U, params=PARAMS):
    a, b, e, alpha, beta, rd_star, rn_star = params
    A_, B_, E_ = np.exp(a), np.exp(b), np.exp(e)
    R_D = np.maximum((D / U) - 1.0, 0.0)
    U_N = np.minimum(N, optimal_N(D_to_C(U)))
    R_N = np.maximum((N / U_N) - 1.0, 0.0)
    D_eff = U + U * rd_star * (1 - np.exp(-R_D / rd_star))
    N_eff = U_N + U_N * rn_star * (1 - np.exp(-R_N / rn_star))
    return E_ + A_ / N_eff ** alpha + B_ / D_eff ** beta
```
