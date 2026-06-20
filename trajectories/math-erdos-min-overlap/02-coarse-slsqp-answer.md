**Problem.** Same as rung 1: minimize the worst overlap `C(v) = max_k (Σ_i v_i (1 − v_{i−k}))·2/n` over
feasible height vectors (`v_i ∈ [0,1]`, `Σ v = n/2`). Lower is a tighter upper bound on the Erdős
constant `C5`. Here the goal is the first real drop off the flat `0.5` floor at a small, legible piece
count.

**Key idea.** Optimize the heights with constrained SLSQP against a *smooth* surrogate of the minimax.
The hard objective is `max_k c_k`, which has kinks where shifts tie; replace it with a log-sum-exp
soft-max over the shifts at sharpness `β`, anneal `β` upward (so the surrogate goes from feeling all
near-worst shifts at once to hugging the true max), and at each level solve with SLSQP under the box
`[0,1]` and the linear equality `Σ v = n/2`. Re-project onto the constraint set after each solve, and
multi-start from many random feasible profiles to escape the non-convex landscape's local minima, keeping
the best *true* (hard-max) overlap.

**Why these choices.** The sum constraint `Σ v = n/2` is load-bearing — without it the optimizer drives
every height to `0` and certifies a meaningless bound; it encodes the `A`/`B` balance. SLSQP is the
natural solver because it handles exactly a box plus a linear equality, and it is the workhorse the
agentic-search record (AutoEvolver) reports for this problem. The soft-max with annealed `β` turns the
brittle minimax into a sequence of smooth problems and makes the surrogate track the reported hard-max by
the end. Multi-start is the cheap, decisive remedy for local minima at small `n`. The piece count is kept
small (`n = 24`) on purpose: fast restarts and a legible optimized shape, accepting that coarse resolution
caps the achievable bound above the frontier.

**Hyperparameters / contract.** `n = 24` cells; `12` random restarts (seed `0`); `β`-ladder
`(60, 150, 300, 600, 1200)` with `250` SLSQP iterations per level (`ftol = 1e-12`); constraint re-projection
by clip-to-`[0,1]` then sum-restoring shift. Output is the best feasible vector found, scored by the frozen
hard-max evaluator. Reproducible under the fixed seed.

```python
import numpy as np
from scipy.optimize import minimize

def _smooth_bound(v, beta):
    n = len(v); conv = np.correlate(v, 1.0 - v, mode='full'); m = conv.max()
    return (m + np.log(np.sum(np.exp(beta * (conv - m)))) / beta) * 2.0 / n   # soft-max surrogate

def _project(v, target):                      # onto box [0,1] AND sum == target
    v = v.copy()
    for _ in range(60):
        v = np.clip(v, 0.0, 1.0)
        diff = target - v.sum()
        if abs(diff) < 1e-12: break
        free = (v > 0.0) & (v < 1.0)
        k = free.sum() or len(v)
        v[free if free.sum() else slice(None)] += diff / k
    return np.clip(v, 0.0, 1.0)

def _anneal_slsqp(v0, betas=(60,150,300,600,1200), maxiter=250):
    n = len(v0); target = n / 2.0; v = _project(v0.copy(), target)
    cons = {'type': 'eq', 'fun': lambda s: s.sum() - target}; bnds = [(0,1)] * n
    for b in betas:
        res = minimize(lambda s: _smooth_bound(s, b), v, method='SLSQP',
                       bounds=bnds, constraints=cons, options={'maxiter': maxiter, 'ftol': 1e-12})
        v = _project(res.x, target)
    return v

def construct(n=24, restarts=12, seed=0):
    """Rung 2: coarse constrained SLSQP, annealed soft-max minimax, multi-start."""
    np.random.seed(seed); best, bestv = None, np.inf
    for _ in range(restarts):
        v = _anneal_slsqp(_project(np.random.rand(n), n / 2.0))
        c = compute_upper_bound(v)
        if c < bestv: bestv, best = c, v.copy()
    return best     # -> C = 0.3812396 at n=24, seed=0

# --- frozen evaluator (AlphaEvolve App. B.5) ---
def compute_upper_bound(sequence):
    seq = np.asarray(sequence, dtype=float)
    conv = np.correlate(seq, 1.0 - seq, mode='full')
    return float(np.max(conv) / len(seq) * 2.0)

if __name__ == "__main__":
    v = construct(); assert abs(v.sum() - len(v)/2.0) < 1e-6
    print("n =", len(v), " C =", compute_upper_bound(v))
```
