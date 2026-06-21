The coarse annealing gave a $50$-piece profile at $R = 1.5371$, asymmetric and partly sparse with mass heavier toward the ends — the right shape — but it stops there for a precise reason. Fifty pieces is too coarse a grid to render an autoconvolution whose peak can be pushed lower, and there is a second, subtler obstruction that the softmax-Adam polish ran straight into: when I push the peak down I am not moving a single node, I am flattening a whole *plateau* of near-equal peak nodes. At the coarse optimum many nodes already sit within a hair of the maximum. A gradient on the softmax surrogate, or a subgradient at the single argmax, can lower *one* peak node, but the instant it does another node in the plateau becomes the new maximum and $R$ does not actually drop — lowering one node at a time is like pressing down one spot of a balloon. That is exactly why the surrogate gradient stalled near $1.5266$. So I keep the optimized shape, lift it to a much finer grid, and attack the right object: the *minimax*, the largest of all near-peak nodes lowered simultaneously.

The method is a **minimax Sequential Linear Program (SLP)** at $N = 600$, warm-started by the softmax-Adam pass. Minimizing a maximum of smooth functions subject to linear constraints is the classic epigraph problem: I introduce an auxiliary variable $z$ standing for the peak and ask to minimize $z$ subject to every autoconvolution node being at most $z$ and the mass held fixed,

$$\min_{a, z} \; z \quad \text{s.t.} \quad b_k(a) = (a*a)_k \le z \;\; \forall k, \quad \textstyle\sum_n a_n = \text{const}, \quad a \ge 0.$$

The catch is that each node $b_k$ is *quadratic* in the heights, so the constraints are not linear. I linearize them around the current heights: $b_k(a + d) \approx b_k(a) + (\nabla b_k)\cdot d$, where the gradient of a self-convolution node is itself a shifted copy of the heights, $\partial b_k / \partial a_j = 2\,a_{k-j}$. That makes every constraint linear in the step $d$, so a single LP solve gives the best peak-lowering move under the linearized model — and crucially it lowers *all* the near-tight nodes at once, which is exactly what the gradient could not do. The objective vector picks out $z$ alone ($c = [0,\dots,0,1]$ over the variables $(d, z)$); the constraint rows are $J\,d - z \le -b$ with $J_{kj} = 2\,a_{k-j}$, and one equality row $\sum_j d_j = 0$ holds the mass fixed so the move is a pure reshape.

The linearization is only trustworthy locally, so the load-bearing safeguard is a **trust region**: I bound each $|d_j| \le \text{tr}$ (and never below $-a_j$, to keep heights non-negative), so the neglected quadratic term, which is second order in $d$, stays negligible. I accept the step only if the *true* $R$ actually drops; on success I grow the trust region (×$1.05$, capped) to take bolder steps, and on rejection I shrink it (×$0.6$) and, when it collapses, restart-kick out with a small multiplicative perturbation and trust reset. Too large a trust region and the quadratic error moves the true peak to an unmodeled node (the step is rejected, the trust shrinks); too small and progress crawls — so it is adapted each round. This accept-true-$R$, grow/shrink discipline is what makes a linearized model converge on a genuinely non-convex objective.

Efficiency forces one more choice. The autoconvolution at $N = 600$ has about $1200$ nodes, and writing a constraint for every one each round is wasteful since the vast majority are slack. So I focus the LP on the **top-$K$** largest nodes — the active part of the minimax. With a small trust region the peak cannot jump to a node far outside that set in one step, so the restricted LP is a faithful local model and an order of magnitude cheaper to solve; I interleave occasional full passes over all $2N-1$ nodes as a check so no node sneaks above the peak unseen. This focusing is what lets the method grind for many rounds within budget — the LP solve, not the FFTs, is the cost. The warm start matters because SLP is local and its basin depends on where it begins: the $\beta$-annealed Adam pass cheaply gets from the flat ceiling down to about $1.52$ and lands in a sensible asymmetric basin, and from there the SLP takes over and keeps descending where the gradient stalled. The returned $N = 600$ profile is the peak-suppressing structure the record constructions exhibit — a tall spike at the boundary, mass heavier toward the ends, the middle third thinned, a large fraction of heights near zero. I expect this to push from $1.52$ into the $1.51$ range, toward the AlphaEvolve $600$-piece band near $1.5053$, then taper, limited by resolution and the affordable number of LP rounds — the opening for the endpoint, which lifts further and grinds longer.

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
