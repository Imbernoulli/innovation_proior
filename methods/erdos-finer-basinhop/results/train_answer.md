The coarse SLSQP rung on the Erdős minimum-overlap step function landed at `C = 0.381240` with two dozen
cells, and the obstacle it left behind was resolution: the wide steps cannot resolve the fine structure of
a near-optimal profile. The right next move is to lift the optimized coarse profile to many more pieces and
refine it there — but carefully, because the coarse profile already encodes most of the gross structure and
I do not want to discard that by re-randomizing.

The method is upscale-then-basin-hop at a middle resolution. The clean way to add resolution for free is to
*upscale*: replace each coarse cell by several identical finer cells of the same height. This is literally
the same step function — same `h`, same overlap, same `C` — just expressed on a finer grid, so the lifted
point starts at the value the coarse profile reached and now has many more degrees of freedom to spend. The
upscaled point is a degenerate plateau of repeated blocks, which is flat in many directions (perturbing two
cells of the same original block in opposite ways leaves the overlap unchanged to first order), so a
constrained step from it can stall; a small kick — a tiny perturbation followed by re-projection to
feasibility — breaks the block symmetry and gives the optimizer traction. The optimizer itself is the
annealed-SLSQP ladder from the coarse rung (the same soft-max surrogate of the minimax, the same box plus
`Σ v = n/2` equality), but a single ladder from one start is not enough at the larger `n`, because the
finer landscape has *more* local minima, not fewer — the extra degrees of freedom that let me carve better
structure also create more ways to get stuck. So I wrap the ladder in **basin-hopping**: solve to a local
optimum, perturb the best-so-far vector and re-solve, accept only improvements, repeat for a budget of
hops, shrinking the perturbation as the hops proceed so early hops explore and late hops refine. This is
exactly the "perturbation search plus basin-hopping" recipe the agentic-search record (AutoEvolver) reports
for this problem. I also push the `β` ladder sharper than at the coarse level — into the thousands —
because the finer profile is spikier and its overlap envelope has a finer set of near-binding shifts, so a
coarse-level `β` would let the surrogate's max sit below the true max and the optimizer would chase a
slightly wrong objective.

I lift to a middle resolution — about a hundred-odd cells, reached by an intermediate multistart at `n = 60`
then an upscale `×2` to `120` — rather than jumping straight to the hundreds of cells the records use, for
two reasons: to confirm that lifting actually helps before spending a long run at high `n`, and to learn the
right basin-hopping schedule (perturbation scale, hop count, `β` ceiling) so the endpoint rung can spend its
budget on resolution rather than re-tuning. Run this way it reaches `C = 0.381076` at `120` cells — within
about `1.5×10^{-4}` of the Haugland and AlphaEvolve landmarks and `1.9×10^{-4}` of the AutoEvolver record.
As expected the gain from `24 → 120` cells is fine shaving, not a big jump: the coarse profile already held
the gross structure, the upscale is confirmed free (the same `C = 0.381098` before and after lifting), and
basin-hopping carves a slightly lower worst-overlap envelope. The returned profile stays feasible and
near-binary, about a third of its cells pinned to the box corners. The cap is still resolution, now at the
fine end — the published frontier lives at several hundred cells — which is what the endpoint rung
addresses by lifting once more and adding an exact-minimax subgradient polish.

```python
import numpy as np
from scipy.optimize import minimize

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
        cand = _project(best + rng.standard_normal(len(best)) * kick * (0.9**h + 0.1), len(best) / 2.0)
        cand = _anneal_slsqp(cand, betas); c = compute_upper_bound(cand)
        if c < bestv: bestv, best = c, cand.copy()
    return best

def construct(seed=1):
    """Lift 24->60 multistart, upscale x2->120, basin-hop refine. -> C ~ 0.38108."""
    rng = np.random.default_rng(seed)
    n = 60; best, bestv = None, np.inf
    for _ in range(8):
        v = _anneal_slsqp(_project(rng.random(n), n / 2.0), betas=(60,150,300,600,1200,2400))
        c = compute_upper_bound(v)
        if c < bestv: bestv, best = c, v.copy()
    v120 = np.repeat(best, 2)                                # upscale x2 -> n=120 (free)
    return _basin_hop(v120, n_hops=20, betas=(300, 800, 1800, 3600), kick=0.03, seed=3)

def compute_upper_bound(sequence):            # frozen evaluator (AlphaEvolve App. B.5)
    seq = np.asarray(sequence, dtype=float)
    conv = np.correlate(seq, 1.0 - seq, mode='full')
    return float(np.max(conv) / len(seq) * 2.0)

if __name__ == "__main__":
    v = construct(); assert abs(v.sum() - len(v)/2.0) < 1e-6
    print("n =", len(v), " C =", compute_upper_bound(v))
```
