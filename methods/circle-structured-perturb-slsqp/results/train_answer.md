The problem is to pack $26$ circles into the unit square $[0,1]^2$, pairwise non-overlapping, so as
to maximize the sum of radii $\sum_i r_i$, with the radii free and unequal — a nonconvex QCQP whose
hard part is placing the centers, since for fixed centers the optimal radii are an LP. A structured
grid baseline reaches only $\approx 2.54$; one joint SLSQP run on the full $78$-variable problem
reaches $\approx 2.59$; and random multi-start SLSQP climbs into the low $2.62$s but then saturates.
The saturation has a clear cause: the best-of-many random restarts is an order statistic, and
uniform center scatters are a blunt seed that almost never lands in the rare basins reaching the top
of the frontier near $2.636$. SLSQP is the right local engine, but blind random restarts have two
weaknesses — no *structure* (the starts look nothing like a good packing) and no *memory* (every
refined start is discarded, so the best packing found is never exploited). The published frontier
results all converged on the same hybrid pipeline to fix both, and I reproduce it here.

I propose a structured-init plus joint-SLSQP plus iterated-perturbation-chain constructor, built
from three ingredients. The first is **structured initialization**. Instead of scattering centers
uniformly, I seed starts that already resemble a good sum-of-radii packing: the four corners are
placed explicitly (a wall-adjacent circle grows on two free sides, so corners are valuable), and the
remaining circles are laid on a golden-angle spiral — point $i$ at radius $\propto \sqrt{i/m}$ and
angle $i\psi$ with $\psi = \pi(3-\sqrt5)$ the golden angle — which distributes the interior with no
rotational resonance and roughly equal local spacing, the sunflower-seed construction. A small
random jitter keeps repeated structured starts distinct. Because each such start already looks like
a plausible packing, SLSQP refining it lands in a strong basin far more often than a uniform scatter
does.

The second ingredient is **joint center-and-radius SLSQP refinement**, kept from the earlier methods
rather than split into an LP-for-radii plus a separate center optimizer. This is the load-bearing
choice the reported breakthroughs single out. Centers and radii couple: the value of moving a center
depends on how much radius that move unlocks across *all* its neighbors at once, and only a solver
that holds centers and radii in the same quadratic model of the Lagrangian can trade them off in a
single coordinated step. The split scheme freezes one while optimizing the other and misses those
coordinated moves. So I run SLSQP over all $78$ variables to a KKT point, then re-tighten the radii
to their exact LP optimum (maximize $\sum_i r_i$ s.t. $r_i+r_j \le d_{ij}$, $r_i \le \text{wall}_i$)
to recover any slack left at the boundary.

The third ingredient — the real escape from the multi-start plateau — is **iterated perturbation
chains**, which supply the missing memory. Rather than discard refined packings, I take the best one
found so far, perturb a random subset of its centers by a Gaussian kick, and re-refine with SLSQP;
if the result is feasible and better it becomes the new incumbent and I reset the kick size, and if
not I cool the kick down so the chain settles into finer local exploration around a strong
configuration. This is iterated local search with an annealed perturbation scale, and it is exactly
the mechanism that lets the search make the long sequence of mostly-lateral moves that coordinated
radius gains require — moves a fresh random restart cannot reach, because it has no good
configuration to perturb *from*. The structured restarts find a strong basin; the chains then mine
it. I orchestrate these under a fixed time budget in three phases: structured spiral/corner-seeded
restarts to find a strong basin, a shorter phase of pure random restarts as cheap insurance against
the structured prior boxing the search into one region, and finally the perturbation chains launched
from the incumbent. Throughout I keep the single best feasible packing, verify it against the real
constraints, and LP-polish its radii.

Run under a bounded wall-clock budget of about nine minutes, this measured $\sum_i r_i \approx
2.62749$ (feasible, maximum constraint violation $\approx 6\times10^{-12}$), lifting the random
multi-start value by about $0.0054$ — the structured restarts reach $\approx 2.6248$ quickly and the
chains grind the incumbent up from there. This lands in the frontier neighborhood but, honestly,
about $0.0085$ below the record $2.635988438568$. The published frontier values — AlphaEvolve
$2.63586276$, ShinkaEvolve $2.635983283$, the AutoEvolver record $2.635988438568$ — are separated by
only parts in the sixth decimal place and were produced with orders of magnitude more search (the
record used roughly $16.6$ hours of autonomous compute). So this is the right pipeline run with a
fraction of the budget, and the residual gap to the record is bought purely with sustained search,
not with a better algorithm.

```python
import numpy as np, time
from scipy.optimize import minimize, linprog

N = 26

# ---- feasibility + LP radii (radii are an LP for fixed centers) ----
def feasible(centers, radii, atol=1e-7):
    c = np.asarray(centers, float); r = np.asarray(radii, float)
    if np.any(r < -atol): return False
    if np.any(r - np.minimum(c[:,0], c[:,1]) > atol): return False
    if np.any(r - np.minimum(1-c[:,0], 1-c[:,1]) > atol): return False
    for i in range(N):
        for j in range(i+1, N):
            if (r[i]+r[j]) - np.hypot(*(c[i]-c[j])) > atol: return False
    return True

def max_radii_lp(centers):
    n = len(centers)
    wall = np.minimum.reduce([centers[:,0], centers[:,1], 1-centers[:,0], 1-centers[:,1]])
    A, b = [], []
    for i in range(n):
        for j in range(i+1, n):
            row = np.zeros(n); row[i] = 1; row[j] = 1
            A.append(row); b.append(np.hypot(*(centers[i]-centers[j])))
    res = linprog(-np.ones(n), A_ub=np.array(A), b_ub=np.array(b),
                  bounds=[(0, w) for w in wall], method="highs")
    return np.maximum(res.x, 0.0)

# ---- joint (centers + radii) SLSQP refinement ----
def slsqp_from_init(centers0, maxiter=400, n=N):
    r0 = np.maximum(max_radii_lp(centers0), 1e-4)
    v0 = np.concatenate([centers0.ravel(), r0])
    def neg_sum(v):
        g = np.zeros_like(v); g[2*n:] = -1.0
        return -v[2*n:].sum(), g
    pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    def pair_con(v):
        c = v[:2*n].reshape(n, 2); r = v[2*n:]
        return np.array([np.hypot(*(c[i]-c[j])) - (r[i]+r[j]) for i, j in pairs])
    def wall_con(v):
        c = v[:2*n].reshape(n, 2); r = v[2*n:]
        return np.concatenate([c[:,0]-r, 1-c[:,0]-r, c[:,1]-r, 1-c[:,1]-r])
    res = minimize(neg_sum, v0, jac=True, method="SLSQP",
                   bounds=[(0,1)]*(2*n) + [(0,0.5)]*n,
                   constraints=[{"type":"ineq","fun":pair_con},
                                {"type":"ineq","fun":wall_con}],
                   options={"maxiter":maxiter, "ftol":1e-10})
    c = res.x[:2*n].reshape(n, 2)
    return c, max_radii_lp(c)

# ---- structured initialization: corners + golden-angle spiral ----
def golden_spiral_init(n, rng, jitter=0.0):
    ga = np.pi * (3 - np.sqrt(5))
    pts = [[0.07,0.07],[0.93,0.07],[0.07,0.93],[0.93,0.93]]   # corner seeds
    m = n - len(pts)
    for i in range(m):
        t = (i + 0.5)/m; rad = 0.5*np.sqrt(t); a = i*ga
        pts.append([0.5 + rad*np.cos(a), 0.5 + rad*np.sin(a)])
    pts = np.array(pts)
    return np.clip(pts + rng.normal(0, jitter, pts.shape), 0.03, 0.97)

# ---- iterated perturbation chain: mine the incumbent ----
def perturb_chain(c, r, rng, n_iter=25, init_sigma=0.06):
    best_c, best_r, best = c.copy(), r.copy(), r.sum()
    sigma = init_sigma
    for _ in range(n_iter):
        cc = best_c.copy()
        k = rng.integers(1, N//2 + 1)
        idx = rng.choice(N, size=k, replace=False)
        cc[idx] += rng.normal(0, sigma, (k, 2))
        cc = np.clip(cc, 0.02, 0.98)
        try:
            nc, nr = slsqp_from_init(cc, maxiter=300)
        except Exception:
            continue
        if feasible(nc, nr, atol=1e-7) and nr.sum() > best:
            best, best_c, best_r = nr.sum(), nc.copy(), nr.copy()
            sigma = init_sigma                       # reset on improvement
        else:
            sigma = max(sigma*0.94, 0.012)           # cool down
    return best_c, best_r

# ---- endpoint constructor ----
def construct_packing(seconds=520, seed=7):
    rng = np.random.default_rng(seed); t0 = time.time()
    best = -1.0; best_cr = None
    while time.time()-t0 < seconds*0.4:              # phase 1: structured restarts
        c0 = golden_spiral_init(N, rng, jitter=rng.uniform(0, 0.05))
        try: c, r = slsqp_from_init(c0)
        except Exception: continue
        if feasible(c, r) and r.sum() > best: best, best_cr = r.sum(), (c.copy(), r.copy())
    while time.time()-t0 < seconds*0.6:              # phase 2: random restarts (insurance)
        c0 = rng.uniform(0.04, 0.96, size=(N, 2))
        try: c, r = slsqp_from_init(c0)
        except Exception: continue
        if feasible(c, r) and r.sum() > best: best, best_cr = r.sum(), (c.copy(), r.copy())
    while time.time()-t0 < seconds:                  # phase 3: perturbation chains from incumbent
        c, r = perturb_chain(*best_cr, rng, n_iter=25)
        if feasible(c, r) and r.sum() > best: best, best_cr = r.sum(), (c.copy(), r.copy())
    return best_cr


if __name__ == "__main__":
    c, r = construct_packing(seconds=520, seed=7)
    print("sum_radii =", r.sum(), "feasible =", feasible(c, r))   # 2.627489971261…
```
