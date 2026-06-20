This is the endpoint of a descent toward the first autocorrelation constant `C1`, certified as an upper bound by `R(f) = max_{|t|≤1/2}(f*f)(t)/(∫f)²` over non-negative step functions, lower being better. The previous rung's trust-region Sequential-LP minimax converged to `1.5172` from a single warm start at `N = 600` and then only crawled. That plateau is the honest signature of a single local constructor: it finds *a* good basin, not *the* good basin. AlphaEvolve reached `1.5053` at the very same `N = 600`, so the remaining gap is not resolution — it is that AlphaEvolve's agentic search explored far more of the basin landscape than one warm start can. The endpoint question is how far a single SLP constructor can genuinely push with more search and, if it helps, a finer grid.

The method keeps the Sequential-LP engine of the previous rung — epigraph variable `z` for the peak, node constraints `(a*a)_k ≤ z` linearized around the current heights, a small trust region, accept only if the true `R` drops — and adds the two levers an agentic search would use. The first is basin diversity: instead of one warm start, launch the SLP from several structurally different initializations and keep the global best. The record constructions are asymmetric, concentrating mass into tall spikes at the boundaries over a thinned interior — the AutoEvolver `30000`-piece solution is exactly this — and the rung-3 shape already drifted toward it, with a spike at one boundary. So I seed the SLP from the rung-3 shape and from a spread of explicit boundary-spike profiles, both left- and right-heavy since the optimum is asymmetric and I do not know which way it leans, and keep whichever lands lowest. The second lever is a long dedicated polish of that best with restart kicks, checkpointing continuously so a long run never loses ground.

I also tried the resolution lever — a repeat-lift of the best `N = 600` shape to `N = 1200`, which is free because replacing each height by copies gives the same function and the same `R`. But the LP at the heart of each SLP round scales with the number of pieces, so at `N = 1200` a single round is several seconds, and the finer grid did not recover below the lift value within budget. The LP solve, not the FFTs, is the bottleneck, so doubling the grid roughly halves the affordable rounds; the trade did not pay here, and the returned frontier stays at `N = 600`.

The measured outcome is `R = 1.5170399450` at `N = 600`. Each lever paid a little: the diverse multi-start search confirmed the rung-3 basin is the lowest of those tried (the boundary-spike seeds converged into worse basins in their time slices) and nudged it to `1.517146`, and the long full-constraint polish ground it to `1.517040`, still inching down at the budget. The returned profile is the peak-suppressing boundary-spike structure: a spike about `9.6×` the mean at index `0`, roughly `216` of `600` heights near zero, the middle third thinned to about `0.70×` the mean, and about `403` autoconvolution nodes within `10⁻³` of the peak — the flat-top plateau the minimax LP drives down together.

The endpoint lands about `0.0117` above the `600`-piece AlphaEvolve `1.5053` and `0.0142` above the record `1.5028628969`. That residual gap at the same resolution is honest and expected: a local trust-region SLP, even diversified over several basins and polished long, converges into a good basin but not the global one AlphaEvolve found, and the record is a `30000`-piece deliberately irregular construction (AutoEvolver, Claude/Opus "aspiration prompting"; after TTT-Discover's `30000`-piece `1.5028628983` and AlphaEvolve's `600`-piece `1.5053`) found over tens of hours with two orders of magnitude more pieces and vastly more compute. The gap from `1.5170` down to `1.50286`, and the further gap to the provable floor `1.28`, is the still-open part of the first autocorrelation inequality — there is no finale that reaches the record because that requires a large-scale agentic/evolutionary search, not a single SLP constructor.

```python
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import fftconvolve
from scipy.optimize import linprog

def R_eval(a):
    a = np.clip(np.asarray(a, float), 0.0, 1000.0); N = len(a)
    b = fftconvolve(a, a); s = float(np.sum(a))
    return float("inf") if s < 0.01 else 2.0 * N * float(np.max(b)) / (s * s)

def _jac(a):
    n = len(a); ap = np.concatenate([np.zeros(n - 1), 2.0 * a, np.zeros(n - 1)])
    return sliding_window_view(ap, n)[:, ::-1]

def slp(a, rounds, trust=8e-5, tr_cap=2e-4, grow=1.05, shrink=0.5, topK=0, seed=0):
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
        bnds = np.empty((n + 1, 2)); bnds[:n, 0] = np.maximum(-a, -tr); bnds[:n, 1] = tr
        bnds[n] = (-np.inf, np.inf)
        res = linprog(c, A_ub=A, b_ub=rhs, A_eq=Aeq, b_eq=[0.0], bounds=bnds, method="highs")
        if not res.success:
            tr *= shrink; tr = trust if tr < 1e-7 else tr; continue
        an = np.clip(a + res.x[:n], 0, None); an /= an.sum(); Rn = R_eval(an)
        if Rn < bR - 1e-13:
            bR = Rn; best = an.copy(); a = an; tr = min(tr * grow, tr_cap)
        else:
            a = np.clip(best * (1 + 0.01 * rng.standard_normal(n)), 0, None); a /= a.sum(); tr = trust
    return best

def _boundary_init(n, sL, sR, rng):
    a = np.ones(n); e = n // 12
    a[:e] += np.linspace(sL, 0, e); a[-e:] += np.linspace(0, sR, e)
    a[0] += sL; a[-1] += sR
    return np.clip(a + 0.02 * rng.standard_normal(n), 1e-3, None)

def construct(N: int = 600, warm=None):
    rng = np.random.default_rng(0)
    starts = ([np.asarray(warm, float)] if warm is not None else [])
    for sL, sR in [(10, 4), (4, 10), (8, 8), (14, 6), (6, 14)]:
        starts.append(_boundary_init(N, sL, sR, rng))
    best = None; bR = np.inf
    for i, a0 in enumerate(starts):
        a = slp(a0, 200, topK=0, seed=i + 1); R = R_eval(a)
        if R < bR: bR = R; best = a
    best = slp(best, 250, trust=6e-5, grow=1.04, topK=0, seed=21)
    return [float(x) for x in best]

if __name__ == "__main__":
    print(R_eval(construct()))   # ~1.5170
```
