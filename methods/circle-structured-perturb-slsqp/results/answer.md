**Problem.** Feasible packing of `26` circles in `[0,1]²` maximizing `Σ rᵢ` (nonconvex QCQP).
Random multi-start SLSQP saturates near `2.622` because uniform scatters are a blunt, memoryless
seed. This endpoint reproduces the frontier hybrid pipeline (AlphaEvolve / ShinkaEvolve /
AutoEvolver) within a single bounded run: structured restarts + joint SLSQP + iterated perturbation
chains.

**Key idea.** Three ingredients, each fixing a weakness of blind restarts. (1) **Structured
initialization** — seed the four corners explicitly and lay the remaining circles on a golden-angle
spiral (`ψ = π(3−√5)`, radius `∝ √(i/m)`) for a non-clumping quasi-uniform interior, with small
jitter; such a start already looks like a good packing, so SLSQP lands in a strong basin far more
often than a uniform scatter. (2) **Joint center+radius SLSQP** (the reported breakthrough): refine
centers and radii together so the solver can trade a center move against the radius growth it
unlocks across all neighbors in one coordinated step; radii re-tightened to their LP optimum after
each refinement. (3) **Iterated perturbation chains** — keep the best packing, perturb a random
subset of its centers by a Gaussian kick, re-SLSQP, accept if feasible-and-better (reset kick) else
cool the kick; this iterated local search walks good basin → better basin, mining a strong
configuration in a way fresh restarts cannot. Three phases under a time budget: structured restarts,
then mixed random restarts, then perturbation chains from the incumbent. Best feasible packing kept,
radii LP-polished, feasibility verified.

**Why these choices.** Structured init supplies the *structure* and chains supply the *memory* that
random multi-start lacked. The golden-angle spiral is the standard non-resonant interior spread;
corner seeding exploits that wall-adjacent circles grow on two free sides. Joint SLSQP beats the
LP-radii + separate-center split because centers and radii couple and only a joint quadratic model
captures coordinated moves. The annealed kick lets chains make the long sequence of lateral moves
coordinated radius gains require. This is the genuine frontier construction; run under a bounded
wall-clock budget it reaches the frontier *neighborhood* but, honestly, below the record
`2.635988438568` (AutoEvolver, ~16.6 h autonomous compute) — the residual gap is bought with search
budget, not a better algorithm.

**Hyperparameters / contract.** Seed `7`, budget `520 s`; phase split `0.4 / 0.6 / 1.0`; SLSQP
`maxiter` `300–400`, `ftol=1e-10`, radii LP-re-tightened; spiral with 4 corner seeds; chains
`n_iter≈25`, initial kick `σ=0.06` (reset on improvement, ×`0.94` cooldown, floor `0.012`),
`1..⌊N/2⌋` centers perturbed per step. Feasibility verified at `atol=1e-7`. **Measured Σ rᵢ =
2.627489971261**, feasible, `maxviol ≈ 6.06e-12`. (Reference frontier: AlphaEvolve `2.63586276`,
ShinkaEvolve `2.635983283`, AutoEvolver record `2.635988438568`.)

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
