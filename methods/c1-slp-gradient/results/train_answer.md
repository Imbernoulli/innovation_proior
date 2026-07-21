The first autocorrelation constant `C1` is the infimum of `R(f) = max_{|t|≤1/2}(f*f)(t)/(∫f)²` over non-negative `f` on `[-1/4,1/4]`, so any step function I write down certifies `C1 ≤ R(f)` and lower is better. A coarse simulated-anneal reached `1.5371` at `N = 50` and then stalled — partly resolution, but mostly something deeper that I had to confront. When I push the peak of the autoconvolution down, I am not moving a single node: I am flattening a whole *plateau* of near-equal peak nodes. A gradient on a softmax surrogate, or a subgradient at the single argmax, lowers one node, but the instant it does another node in the plateau becomes the new maximum and `R` does not actually drop. The true objective is the maximum over the whole near-tight set, and lowering one node at a time is like pressing one spot of a balloon. That is why the gradient warm start plateaus near `1.52`.

The fix is to treat the problem as the minimax it is and solve it with Sequential Linear Programming. I introduce an auxiliary variable `z` standing for the peak, ask to minimize `z` subject to every autoconvolution node `(a*a)_k ≤ z`, and hold the mass fixed. Each node is quadratic in the heights, so I linearize around the current `a`: `(a*a)_k(a+d) ≈ (a*a)_k(a) + Σ_j 2 a_{k−j} d_j`, where the gradient of a self-convolution node is just a shifted copy of the heights. That makes every constraint linear in the step `d`, and a single LP solve gives the best peak-lowering move under the linear model — one that lowers all the near-tight nodes at once, which is exactly what the gradient could not do. I keep the step inside a small trust region so the neglected quadratic term stays negligible, accept it only if the true `R` drops, grow the trust on success and shrink it (with a restart kick) on rejection, and iterate. This is precisely the recipe the record constructions describe: focus the optimization on the constraints close to tight — the positions where the convolution is largest — and drive them down together.

Two engineering choices make it practical. The `slp` solver takes a top-K cutoff so the LP can be restricted to the near-tight nodes (the active part of the minimax) when the full node count gets expensive — with a small trust region the peak cannot jump to a node far outside that set in one step, so the restricted LP stays a faithful local model while being far cheaper to solve. At `N = 600` the autoconvolution has only about `1200` nodes, and a full-node LP solves in a couple hundred milliseconds — cheap enough that the run shipped here leaves every round unrestricted and keeps the cutoff in reserve for when the node count climbs into the thousands at higher resolution. And the warm start is the softmax-Adam refinement itself: even though it plateaus, it cheaply takes me from the flat ceiling down to about `1.52` in a sensible asymmetric basin, and the SLP takes over from there. The trust region is the load-bearing safeguard — too large and the quadratic error moves the true peak to an unmodeled node and the step is rejected; too small and progress crawls — so it is adapted every round, and restart kicks escape trust-region collapse.

The measured outcome is `R = 1.5172` at `N = 600`: the Adam warm start lands at `1.5266`, and the trust-region SLP grinds it down to `1.5172` where the gradient stalled. The returned profile is the peak-suppressing structure the literature reports — a tall spike at the boundary (about nine times the mean at the first index), mass heavier toward both ends, the middle third thinned to about `0.70×` the mean, and roughly `215` of the `600` heights driven near zero. The rung lands about `0.012` above the `600`-piece AlphaEvolve `1.5053` and `0.014` above the `30000`-piece record `1.5028628969`. That residual gap at the same resolution is the honest signature of a single bounded constructor: a local trust-region SLP from one warm start converges into a good basin but not the global one AlphaEvolve reached with an agentic search and far more compute. The gap, and the room a finer grid would open, is exactly what motivates lifting to more pieces and grinding the SLP longer in the endpoint rung.

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
            a = np.clip(best * (1 + 0.01 * rng.standard_normal(n)), 0, None); a /= a.sum(); tr = trust
    return best

def construct(N: int = 600):
    a = _adam(np.ones(N), 12000, 0.006, 300.0, 2e5, seed=1, ke=3000, ks=0.02)
    for _ in range(8):
        a = slp(a, 60, topK=0, seed=3)
    return [float(x) for x in a]

if __name__ == "__main__":
    print(R_eval(construct()))   # ~1.5172
```
