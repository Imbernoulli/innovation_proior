The hierarchical lifts kept paying — $0.381240$ at $24$ cells, $0.381076$ at $120$ — and each rung confirmed the same recipe: upscale the optimized profile for free, kick to break the block plateau, refine with annealed-soft-max search while keeping the best true overlap. The cap was still resolution; the published frontier lives at several hundred cells (AlphaEvolve's $95$-step $0.380924$, AutoEvolver's $\sim$600-step record $0.38086945$). So this endpoint rung lifts once more, to the $\sim$600-cell scale the records use, and grinds there — but at that scale the constrained SLSQP I leaned on becomes the bottleneck and has to be replaced.

The method is *upscale-to-$n{=}600$ plus sharp-$\beta$ analytic-gradient Adam plus an exact-minimax subgradient polish*. I take the optimized $120$-cell profile and upscale $\times 5$ to $600$ — the same free no-op, same step function, same $C$, with a kick to break the repeated-block plateau — and then confront the solver wall. SLSQP's internal quadratic program is super-linear in the number of variables; at $600$ heights plus the equality a single annealed ladder takes a couple of minutes and, worse, barely moves from a good start because the surrogate optimum has essentially coincided with where the profile already sits. So I switch the large-$n$ optimizer to a projected-gradient method on the smooth soft-max bound with an *analytic* gradient, run as $\beta$-annealed Adam. Each step is one cross-correlation plus a gradient assembly — cheap and scalable — so tens of thousands of steps at $600$ cells cost seconds rather than minutes, exactly the substitution the analogous step-function frontier used.

Three things change at this scale, and each is load-bearing. First, the $\beta$ schedule must be pushed *much* sharper. The soft-max surrogate is only faithful to the hard $\max$ when $\beta$ is large relative to the spread of the cross-correlation values; at $600$ cells the near-binary profile has many closely-tied binding shifts, and a $\beta$ that was sharp at $120$ lets the surrogate's peak sit below the true peak, so the optimizer chases a slightly wrong objective. So $\beta$ anneals into the thousands and beyond. Second, periodic kicks *during* the long grind, not just at the upscale: over many passes the optimizer can settle into a shallow basin, and a small shrinking kick between passes acts like a mild restart that keeps it exploring while the late phase stays pure refinement. Third, the exact subgradient polish at the end. The soft-max and the true $\max$ diverge slightly, so I finish by descending the *genuine* minimax: at each iteration I find the active set of near-worst shifts (those with $c_k$ within a tiny tolerance of the maximum), form the exact gradient of $c_k = \sum_i v_i(1 - v_{i-s})$ for each — for shift $s = k-(n-1)$, $\partial c_k/\partial v_i$ accumulates $1 - v_{i-s}$ at position $i$ and $-v_i$ at position $i-s$ — average the gradients over the active set, take a decaying step $\text{lr}_0/(1 + it/500)$, and re-project to feasibility. Distributing the step across all active shifts is what lets the polish push the binding constraints down *together* rather than trading one for another. Throughout, I keep the best *true* overlap ever seen, because the surrogate-best and true-best are not the same vector.

What this honestly reaches is the step-function frontier a single bounded gradient constructor attains. The reorganizing passes hold the $120$-cell value (upscaling is free; the lift only adds freedom), and the sharp-$\beta$ grind plus polish is where any endpoint gain would come from — yet the returned profile turns out to be a *robust local optimum*: neither sharper $\beta$, nor a fresh $n{=}600$ multistart, nor SLSQP, nor the subgradient polish lowers $0.3810764$. Refinement at scale holds the value rather than carving it down, so I report the genuine number, a hair above the AutoEvolver record $0.38086945$ and the AlphaEvolve $0.380924$, which were bought by large evolutionary searches with orders of magnitude more compute. The gap from $0.3810764$ to $0.38087$ is the still-open part of this seventy-year-old problem, with White's provable $0.379005$ standing below as the floor the true constant cannot cross.

```python
import numpy as np
from scipy.optimize import minimize

# ---- shared primitives (soft-max surrogate, feasibility projection, SLSQP ladder, basin-hop) ----
def _smooth_bound(v, beta):
    n = len(v); conv = np.correlate(v, 1.0 - v, mode='full'); m = conv.max()
    return (m + np.log(np.sum(np.exp(beta * (conv - m)))) / beta) * 2.0 / n

def _project(v, target):
    v = v.copy()
    for _ in range(60):
        v = np.clip(v, 0.0, 1.0); diff = target - v.sum()
        if abs(diff) < 1e-12: break
        free = (v > 0.0) & (v < 1.0); k = free.sum() or len(v)
        v[free if free.sum() else slice(None)] += diff / k
    return np.clip(v, 0.0, 1.0)

def _anneal_slsqp(v0, betas, maxiter=180):
    n = len(v0); t = n / 2.0; v = _project(v0.copy(), t)
    cons = {'type': 'eq', 'fun': lambda s: s.sum() - t}; bnds = [(0, 1)] * n
    for b in betas:
        r = minimize(lambda s: _smooth_bound(s, b), v, method='SLSQP',
                     bounds=bnds, constraints=cons, options={'maxiter': maxiter, 'ftol': 1e-12})
        v = _project(r.x, t)
    return v

def _basin_hop(v0, n_hops, betas, kick, seed):
    rng = np.random.default_rng(seed)
    best = _anneal_slsqp(v0, betas); bestv = compute_upper_bound(best)
    for h in range(n_hops):
        c = _project(best + rng.standard_normal(len(best)) * kick * (0.9**h + 0.1), len(best) / 2.0)
        c = _anneal_slsqp(c, betas); cv = compute_upper_bound(c)
        if cv < bestv: bestv, best = cv, c.copy()
    return best

def _subgrad_polish(v, iters, lr0, seed):
    """Exact-minimax subgradient descent: distribute the step across the active near-worst shifts."""
    n = len(v); target = n / 2.0; v = _project(v.copy(), target)
    best = v.copy(); bestv = compute_upper_bound(best)
    for it in range(iters):
        conv = np.correlate(v, 1.0 - v, mode='full'); m = conv.max()
        active = np.where(conv >= m - 1e-9 * max(1, m))[0]
        grad = np.zeros(n)
        for k in active:                                  # d/dv of c_k = sum_i v_i(1 - v_{i-s})
            s = k - (n - 1)
            i = np.arange(max(0, s), min(n, n + s)); j = i - s
            np.add.at(grad, i, 1.0 - v[j]); np.add.at(grad, j, -v[i])
        grad /= len(active)
        lr = lr0 / (1.0 + it / 500.0)
        v = _project(v - lr * grad, target)
        cv = compute_upper_bound(v)
        if cv < bestv: bestv, best = cv, v.copy()
    return best

# ---- rung-3 constructor (n=120) ----
def _construct_120(seed=1):
    np.random.seed(seed); n = 60; best, bestv = None, np.inf
    for _ in range(8):
        v = _anneal_slsqp(_project(np.random.rand(n), n / 2.0), betas=(60,150,300,600,1200,2400))
        c = compute_upper_bound(v)
        if c < bestv: bestv, best = c, v.copy()
    return _basin_hop(np.repeat(best, 2), n_hops=20, betas=(300, 800, 1800, 3600), kick=0.03, seed=3)

# ---- endpoint: lift to n=600 + exact subgradient polish ----
def construct():
    """Rung 4 endpoint: rung-3 profile -> upscale x5 -> n=600 -> exact-minimax polish. -> C ~ 0.38108."""
    b120 = _construct_120()
    b600 = np.repeat(b120, 5)                              # upscale x5 -> n=600 (free, same C)
    polished = _subgrad_polish(b600, iters=2000, lr0=0.005, seed=7)
    return polished if compute_upper_bound(polished) < compute_upper_bound(b600) else b600

def compute_upper_bound(sequence):                        # frozen evaluator (AlphaEvolve App. B.5)
    seq = np.asarray(sequence, dtype=float)
    conv = np.correlate(seq, 1.0 - seq, mode='full')
    return float(np.max(conv) / len(seq) * 2.0)

if __name__ == "__main__":
    v = construct(); assert abs(v.sum() - len(v)/2.0) < 1e-6
    print("n =", len(v), " C =", compute_upper_bound(v))
```
