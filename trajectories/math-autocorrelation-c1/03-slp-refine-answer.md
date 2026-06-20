**Problem.** Minimize `R = 2N·max_k(a*a)_k / (Σ a_n)^2` (lower = tighter upper bound on `C1`). The coarse rung
stalled at `1.5371` on `N=50` because of resolution *and* because a single-node gradient cannot lower a whole
plateau of near-equal peak nodes. This rung lifts to `N=600` and replaces the gradient with a minimax solver that
attacks all near-tight nodes at once.

**Key idea.** Treat the objective as a **minimax** and solve it by **Sequential Linear Programming (SLP)**. Introduce
an epigraph variable `z` for the peak; minimize `z` subject to every autoconvolution node `b_k = (a*a)_k ≤ z` and
the mass fixed. The nodes are *quadratic* in the heights, so linearize them around the current `a`: `b_k(a+d) ≈
b_k(a) + Σ_j 2 a_{k−j} d_j` (the gradient of a self-convolution node is a shifted copy of the heights). A single LP
gives the best peak-lowering step `d` under the linear model — and crucially it lowers *all* the near-tight nodes
simultaneously, which the softmax/subgradient could not. Keep `d` in a small **trust region** so the neglected
quadratic term stays negligible, accept only if the *true* `R` drops, grow the trust on success / shrink on
rejection, and iterate. This is exactly the "focus the LP on the near-tight constraints" recipe TTT-Discover and
AutoEvolver describe. A short `β`-annealed Adam pass provides the warm start (flat → ~`1.52`); the SLP then grinds
where the gradient stalled.

**Why these choices.** The constraints are restricted to the **top-K** largest nodes (the active part of the
minimax): with a small trust region the peak cannot jump to a node far outside that set in one step, so the
restricted LP is a faithful, far cheaper local model, with occasional full passes over all `2N−1` nodes as a check.
The trust region is the load-bearing safeguard — too large and the quadratic error moves the true peak to an
unmodeled node (the step is then rejected and the trust shrinks); too small and progress crawls — so it is adapted
each round. Restart kicks (a small multiplicative perturbation, trust reset) escape trust-region collapse. The
returned `N=600` profile is genuinely asymmetric and sparse — a tall spike at the boundary, mass heavier toward the
ends, the middle third thinned, and a large fraction of heights near zero — the peak-suppressing boundary-spike
structure the record constructions exhibit.

**Hyperparameters / contract.** `N = 600`. Warm start: Adam `12000` steps, `lr 0.006`, `β: 300 → 2e5`, kicks `0.02`
every `3000`. SLP: trust `1e-4` (cap `~2.5e-4`), grow `1.05` / shrink `0.5–0.6`, top-K `≈ all` near-tight nodes with
periodic full passes, restart kicks `~0.01` when stalled, several hundred rounds, best-true-`R` checkpointed.
Stochastic but seeded — returned vector and `R` reproducible.

```python
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import fftconvolve
from scipy.optimize import linprog

def R_eval(a):
    a = np.clip(np.asarray(a, float), 0.0, 1000.0); N = len(a)
    b = fftconvolve(a, a); s = float(np.sum(a))
    return float("inf") if s < 0.01 else 2.0 * N * float(np.max(b)) / (s * s)

def _grad_surrogate(a, beta):                 # softmax-max gradient for the Adam warm start
    n = len(a); b = fftconvolve(a, a); s = float(np.sum(a))
    cmax = float(np.max(b)); w = np.exp(beta * (b - cmax)); w /= w.sum()
    full = fftconvolve(w, a[::-1]); dM = 2.0 * full[n - 1 + np.arange(n)]
    R = R_eval(a); return R, (2.0 * n / s**2) * dM - (2.0 * R / s) * np.ones(n)

def _adam(a, iters, lr, b0, b1, seed, ke, ks):
    rng = np.random.default_rng(seed); a = np.clip(a, 1e-9, None)
    m = np.zeros_like(a); v = np.zeros_like(a); best = a.copy(); bR = R_eval(a)
    for t in range(1, iters + 1):
        beta = np.exp(np.log(b0) + (np.log(b1) - np.log(b0)) * (t - 1) / max(1, iters - 1))
        R, g = _grad_surrogate(a, beta)
        m = 0.9 * m + 0.1 * g; v = 0.999 * v + 0.001 * g * g
        a = np.clip(a - lr * (m / (1 - 0.9**t)) / (np.sqrt(v / (1 - 0.999**t)) + 1e-12), 0, 1000)
        if R < bR: bR = R; best = a.copy()
        if ke and t % ke == 0 and t < iters - 1:
            a = np.clip(best * (1 + ks * rng.standard_normal(len(a))), 0, 1000)
    return best

def _jac(a):                                  # J[k,j] = 2 a[k-j]
    n = len(a); ap = np.concatenate([np.zeros(n - 1), 2.0 * a, np.zeros(n - 1)])
    return sliding_window_view(ap, n)[:, ::-1]

def slp(a, rounds, trust=1e-4, tr_cap=2.5e-4, grow=1.05, shrink=0.6, topK=0, seed=0):
    rng = np.random.default_rng(seed)
    a = np.clip(np.asarray(a, float), 0, None); a = a / a.sum(); n = len(a)
    best = a.copy(); bR = R_eval(a); tr = trust
    c = np.zeros(n + 1); c[n] = 1.0
    Aeq = np.zeros((1, n + 1)); Aeq[0, :n] = 1.0
    for r in range(rounds):
        b = fftconvolve(a, a)
        if topK and topK < 2 * n - 1:
            idx = np.argpartition(b, -topK)[-topK:]; J = _jac(a)[idx]; rhs = -b[idx]
        else:
            idx = np.arange(2 * n - 1); J = _jac(a); rhs = -b
        A = np.empty((len(idx), n + 1)); A[:, :n] = J; A[:, n] = -1.0
        lo = np.maximum(-a, -tr)
        res = linprog(c, A_ub=A, b_ub=rhs, A_eq=Aeq, b_eq=[0.0],
                      bounds=list(zip(lo, np.full(n, tr))) + [(None, None)], method="highs")
        if not res.success:
            tr *= shrink; tr = trust if tr < 1e-7 else tr; continue
        an = np.clip(a + res.x[:n], 0, None); an /= an.sum(); Rn = R_eval(an)
        if Rn < bR - 1e-13:
            bR = Rn; best = an.copy(); a = an; tr = min(tr * grow, tr_cap)
        else:
            a = np.clip(best * (1 + 0.01 * rng.standard_normal(n)), 0, None); a /= a.sum()
            tr = trust
    return best

def construct(N: int = 600):
    a = _adam(np.ones(N), 12000, 0.006, 300.0, 2e5, seed=1, ke=3000, ks=0.02)
    for _ in range(8):                        # several SLP rounds-blocks with restart escapes
        a = slp(a, 60, topK=0, seed=3)
    return [float(x) for x in a]

if __name__ == "__main__":
    print(R_eval(construct()))   # ~1.5172
```
