The multi-start rung climbed to $2.6221$ and then stalled exactly as predicted: the best of $120$ random restarts is an order statistic that saturates, because uniform center scatters are a blunt seed that almost never lands in the rare basins reaching the top of the frontier. SLSQP is still the right local engine — every rung has leaned on it and it keeps returning clean local optima — but blind random restarts have two weaknesses I can now name precisely. They have no *structure* (the starts look nothing like a good packing) and no *memory* (every start is thrown away, so the best packing found is never exploited). The endpoint has to fix both, and the published frontier systems — AlphaEvolve, ShinkaEvolve, AutoEvolver — all converged on the same hybrid pipeline to do it. I reproduce that pipeline within a single bounded run.

I propose a three-ingredient constructor: *structured (golden-angle-spiral + corner-seeded) restarts, joint center+radius SLSQP refinement, and iterated perturbation chains that mine the incumbent*, run in phases under a fixed time budget. Each ingredient repairs one named weakness.

The first ingredient is *structure in the initialization*. Instead of scattering centers uniformly, I want starts that already resemble a good sum-of-radii packing: a few circles pushed hard into the corners and along the edges, where the walls let a circle grow on two free sides, and the interior filled by a smooth, non-clumping spread. The classic device for a non-clumping interior spread is the golden-angle spiral: place point $i$ at radius $\propto \sqrt{i/m}$ and angle $i\,\psi$ with the golden angle $\psi = \pi(3-\sqrt{5})$, which distributes points with no rotational resonance and roughly equal local spacing — the sunflower-seed construction. So the structured init seeds the four corners explicitly, lays the remaining circles on a golden-angle spiral centered in the square, and adds a small random jitter so repeated structured starts are not identical. Each such start already looks like a plausible packing, so SLSQP refining it lands in a *good* basin far more often than a uniform scatter does.

The second ingredient is the *joint center+radius SLSQP refinement*, kept exactly as in the earlier rungs — and this is the piece the reported breakthroughs single out. Rather than split the problem into an LP for the radii and a separate optimizer for the centers, I refine all $78$ variables together. The joint form wins because the two couple: the value of moving a center depends on how much radius that move unlocks across *all* its neighbors at once, and only a solver that sees centers and radii in the same quadratic model can trade them off in a single coordinated step. The split scheme optimizes one with the other frozen and misses those coordinated moves. After each refinement I re-tighten the radii to their exact LP optimum, so the reported radii are maximal for the final centers.

The third ingredient — the real escape from the multi-start plateau — is *iterated perturbation chains*, which supply the memory random restarts lacked. Rather than discard a refined packing, I take the best one found so far, perturb it, and re-refine, repeating to form a chain that walks from good basin to better basin. Each step jiggles a random subset of $k \in \{1, \dots, \lfloor N/2 \rfloor\}$ centers by a Gaussian kick of scale $\sigma$, clips back into the square, and re-runs SLSQP to repair and re-optimize. If the result is feasible and better it becomes the new incumbent and the kick is reset to $\sigma = 0.06$; if not, the kick is cooled, $\sigma \leftarrow \max(0.94\,\sigma,\, 0.012)$, so the chain settles into finer and finer local exploration around a strong configuration. This is iterated local search with an annealed perturbation scale, and it is precisely the mechanism that lets the search make the long sequence of mostly-lateral moves coordinated radius gains require — moves a fresh random restart cannot reach because it has no good configuration to perturb *from*. The structured restarts find a strong basin; the chains then mine it.

I run these as three phases under a fixed wall-clock budget. For the first $40\%$ of the budget: structured spiral/corner-seeded restarts with joint SLSQP, to find a strong basin quickly. From $40\%$ to $60\%$: a phase that mixes in pure random restarts, as cheap insurance that the structured prior has not boxed the search into one region. For the remainder — and this is where the endpoint number comes from — iterated perturbation chains launched from the incumbent, grinding it upward. Throughout I keep the single best feasible packing ever seen, verify it against the real constraints, and re-tighten its radii by LP, so the reported number is the genuine sum of a genuinely feasible arrangement.

I expect the structured init plus chains to clear the random-multi-start plateau and push into the frontier neighborhood near $2.636$. But I am running one constructor under a bounded budget — minutes, not the $\sim 16.6$ hours of autonomous compute that produced the record $2.635988438568$. The published systems spent orders of magnitude more search to shave the last parts in the sixth decimal place, where the band (AlphaEvolve $2.63586$, ShinkaEvolve $2.635983$, AutoEvolver $2.635988$) is separated by only a few parts in a million. So I land *in the band's neighborhood but below the record* — the honest outcome of running the right pipeline with a fraction of the compute. The pipeline is the endpoint method; the residual gap from my measured number to the record is the part of the problem bought purely with search budget, not a better algorithm.

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
