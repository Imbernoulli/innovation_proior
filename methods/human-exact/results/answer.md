# Data-Constrained Scaling Laws, distilled

A loss law for language models that holds when unique data is finite and must be repeated. It
generalizes the single-epoch compute-optimal law `L(N, D) = E + A/N^α + B/D^β` by replacing the raw
parameter and token counts with **effective** quantities that account for the diminishing value of
repeated tokens and excess parameters. It reduces exactly to the single-epoch law when nothing is
repeated, and tells you how to allocate compute between more epochs and more parameters under a hard
data cap.

## Problem it solves

Compute-optimal scaling assumes unlimited unique text and prescribes scaling parameters `N` and tokens
`D` together. Extrapolated, it demands token counts that exceed the available high-quality text supply
(exhaustion of English ~mid-2020s; far worse for other languages). When the unique pool `U` is fixed and
compute is large, you must repeat data (multiple epochs) and/or over-parameterize relative to the data —
regimes the single-epoch law was never fit on and cannot model: it treats a re-read token as worth a
fresh one and a parameter as equally valuable regardless of how little data backs it.

## Key idea

Each pass over a token lets the model learn a fixed fraction `1 - δ` of the value remaining in it, so
the value of the `k`-th pass decays geometrically as `(1-δ)^k`. Summing the geometric series over `R_D`
repetitions of `U` unique tokens gives the **effective data** — the number of fresh unique tokens that
would have done the same job — and the symmetric argument gives **effective parameters** (excess
parameters relearn the same features and add diminishing value).

**Effective data.** With `R_D` = repetitions = epochs − 1:

```
D' = U + (1-δ)U (1 - (1-δ)^{R_D}) / δ            (exact geometric sum)
   = U + U · R*_D · (1 - e^{-R_D/R*_D})           (small-δ closed form, R*_D = (1-δ)/δ)
```

`R*_D` is the "repetition half-life": `D'` plateaus at `U(1 + R*_D)` as `R_D → ∞` (a fixed pool yields
only so much signal), at `R_D = R*_D` the repeated-token block is worth on average `1 - 1/e` of fresh,
and for `R_D ≪ R*_D`, `D' ≈ U(1 + R_D) = D` (repeats ≈ fresh). The `e^{-x}` smoothing of the integer
geometric sum costs under 2% on the loss term (because `D'` enters as `B/D'^β` with `β < 1`); the fitted
`δ ≈ 0.06` confirms the small-`δ` regime.

**Effective parameters** (symmetric), with `U_N` = compute-optimal params for the pool and
`R_N = max(N/U_N − 1, 0)`:

```
N' = U_N + U_N · R*_N · (1 - e^{-R_N/R*_N}),    U_N = min{ (U_D · G)^{β/α} · G,  N },   G = (αA/βB)^{1/(α+β)}
```

**Final law:**

```
L(U_N, U_D, R_N, R_D) = E + A / (U_N + U_N R*_N (1 - e^{-R_N/R*_N}))^α
                          + B / (U_D + U_D R*_D (1 - e^{-R_D/R*_D}))^β
```

A strict generalization: with no data repetitions and no excess parameters it recovers
`E + A/N^α + B/D^β`; with `R*_N = R*_D = ∞` it is exactly the old law for all `N` and `D`.

## Fitting and the headline result

Freeze the single-epoch coefficients `A, B, E, α, β` (so the law agrees with the validated single-epoch
fit); learn only `R*_N, R*_D` by minimizing, over all runs, a Huber loss (`δ = 1e-3`) between predicted
and observed `log L`, with the prediction formed via log-sum-exp:

```
min_{R*_N, R*_D}  Σ_i Huber_δ( LSE(a − α log N'_i,  b − β log D'_i,  e) − log L_i ),   a=log A, b=log B, e=log E
```

LSE evaluates `log(A/N'^α + B/D'^β + E)` stably; Huber resists the noisy low-compute / double-descent
runs. Optimize with L-BFGS from a grid of inits; drop outliers where loss turns *up* (the form models
plateau, not upturn — if epochs/params start to hurt, you simply stop or remove them).

Result on the C4 runs: **`R*_D ≈ 15.39 > R*_N ≈ 5.31`** — repeated data outlasts excess parameters
(decays ~3× slower). So in the data-constrained regime, allocate additional compute to **more epochs
faster than to more parameters**, the opposite of single-epoch equal scaling. Meaningful gains from
repeating extend to ~16 epochs (`R*_D`), beyond which returns diminish sharply.

C4 single-epoch coefficients (`α` tied to `β`, forced by the corpus's `a = b = 0.5`):
`A ≈ 521`, `B ≈ 1488`, `E ≈ 1.87`, `α = β ≈ 0.353`, validated against the corpus's IsoFLOP frontier
(Gopher budget `5.76e23` FLOPs → `N_opt ≈ 70B`, `D_opt ≈ 1.37T`). Numeric law:

```
L = 521 / (U_N + 5.3 U_N (1 - e^{-R_N/5.3}))^{0.35} + 1488 / (U_D + 15.4 U_D (1 - e^{-R_D/15.4}))^{0.35} + 1.87,
with U_N = min(N, 0.051 · U_D).
```

## Why these choices

- **Geometric value-decay**: "learn a fixed fraction of remaining value per pass" is the simplest
  monotone-saturating model and sums in closed form. Reparameterizing `δ → R*_D = (1-δ)/δ` turns the
  learned constant into an interpretable repetition budget (the plateau is `U(1 + R*_D)`).
- **Symmetric `N'` term**: required — fitting decay on `D` only collapses the fit; excess parameters in
  heavily-repeated runs genuinely stop helping.
- **`e^{-x}` smoothing** vs raw geometric: the raw form fits marginally better but its constant `δ` is
  less interpretable; the < 2% loss-term error makes the trade worth it.
- **Monotone-saturating, not upturn-modeling**: an exponent-decay variant that bends loss back up fits
  the overall surface worse and gets the *return* curve wrong, with no mechanistic justification.
- **Fit `R*` only, freeze the rest, in log space with LSE + Huber**: keeps single-epoch behavior pinned;
  log space because loss is power-law; Huber for robustness to outliers.

## Working code

The fit is the torch log-sum-exp/Huber/L-BFGS path, and prediction is NumPy.

```python
import numpy as np
import torch

# Single-epoch coefficients (fit once on the corpus, alpha tied to beta).
A_LOG, B_LOG, E_LOG = 6.255414, 7.3049974, 0.6254804
ALPHA = BETA = 0.3526596
A, B, E = np.exp(A_LOG), np.exp(B_LOG), np.exp(E_LOG)          # ~521, ~1488, ~1.87
G = ((ALPHA * A) / (BETA * B)) ** (1.0 / (ALPHA + BETA))

# Decay constants learned by fit_decay (grid of L-BFGS inits, outliers removed).
R_D_STAR, R_N_STAR = 15.387756, 5.309743                       # R*_D > R*_N
PARAMS = [A_LOG, B_LOG, E_LOG, ALPHA, BETA, R_D_STAR, R_N_STAR]


def optimal_N(C):
    return G * (C / 6.0) ** (BETA / (ALPHA + BETA))            # N_opt(C)


def D_to_C(D):
    return ((G * D) ** (1.0 / (ALPHA / (ALPHA + BETA)))) * 6.0  # invert D_opt(C)


def _fit_loss(inp, params):
    # inp columns: [U_N, U_D, R_D, R_N, L]; params: a,b,e,alpha,beta,R*_D,R*_N
    a, b, e, alpha, beta, ep_star, n_star = params
    n_eff = inp[:, 0] + inp[:, 0] * n_star * (1 - torch.exp(-inp[:, 3] / n_star))
    d_eff = inp[:, 1] + inp[:, 1] * ep_star * (1 - torch.exp(-inp[:, 2] / ep_star))
    pre = torch.stack([a - alpha * torch.log(n_eff),
                       b - beta * torch.log(d_eff),
                       e.expand(inp.shape[0])])
    pred = torch.logsumexp(pre, dim=0)                          # log(A/N'^alpha + B/D'^beta + E)
    return torch.nn.functional.huber_loss(pred, torch.log(inp[:, 4]),
                                          delta=1e-3, reduction="none").sum()


def _fit_decay_from(inp, init, steps=50):
    p = torch.nn.Parameter(torch.tensor(init, dtype=torch.float32))
    lbfgs = torch.optim.LBFGS([p], lr=1e-1, history_size=10, max_iter=20,
                              line_search_fn="strong_wolfe")

    def closure():
        lbfgs.zero_grad()
        l = _fit_loss(inp, p)
        l.backward()
        p.grad[:5] = 0          # freeze single-epoch coefficients; learn only R*_D, R*_N
        return l

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
    R_D = np.maximum((D / U) - 1.0, 0.0)                        # epochs - 1
    U_N = np.minimum(N, optimal_N(D_to_C(U)))                   # data-justified params
    R_N = np.maximum((N / U_N) - 1.0, 0.0)
    D_eff = U + U * rd_star * (1 - np.exp(-R_D / rd_star))
    N_eff = U_N + U_N * rn_star * (1 - np.exp(-R_N / rn_star))
    return E_ + A_ / N_eff ** alpha + B_ / D_eff ** beta
```
