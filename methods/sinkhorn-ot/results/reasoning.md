OK, let me think about this from the pain. I have two histograms `r` and `c` on `d` bins, a ground cost `m_ij` for moving mass from bin `i` to bin `j`, and I want the cheapest plan to turn `r` into `c`. A plan is a matrix `P` with `p_ij` units sent from `i` to `j`; to be consistent with the histograms its row sums must be `r` and its column sums must be `c`. The cost is `⟨P, M⟩ = Σ_ij p_ij m_ij`. So the object I care about is

```
d_M(r,c) = min_{P ≥ 0, P1 = r, P^T1 = c} ⟨P, M⟩.
```

This is beautiful as a distance — it actually *uses* the geometry of the bins through `M`, unlike Hellinger or χ² or KL, which treat the `d` coordinates as unrelated. On bags of words it can tell that two synonyms are nearly the same feature; on images it knows two neighboring pixels are close. That's why it works so well in retrieval. But it's a linear program, and that's the whole problem. The feasible set `U(r,c)` is the transportation polytope, and minimizing a linear functional over a polytope lands me on a vertex. For a general ground metric the only solvers I have — network simplex, interior point, the EMD codes — cost at least `O(d^3 log d)`, and in practice it feels super-cubic. A single pair of histograms with a few hundred bins takes seconds. I cannot put that inside a learning loop over thousands of pairs. So the question is really: can I get a transport-flavored distance between histograms that's orders of magnitude cheaper, for *arbitrary* `M`, without giving up the metric structure?

Let me stare at *what kind of solution* the LP gives me, because maybe the cost and the answer's shape are the same disease. A vertex of `U(r,c)` is a sparse table: at most `2d−1` of the `d²` entries are nonzero. So the optimal plan is almost deterministic — for most source bins `i` there's essentially a single destination `j` that receives all of `r_i`. Think of `(X,Y)` jointly distributed as `P`: the LP wants the coupling that's as close to a deterministic map as it can be while respecting the marginals. Two things bother me about that. First, the search for *which* `2d−1` entries to light up is exactly the combinatorial cost. Second, this extreme plan is fragile: nudge `r` a hair and the optimal vertex can jump somewhere else, so the distance is piecewise-linear and non-differentiable, and the plan itself is a strange, spiky thing that no real noisy process would ever produce. If two histograms could plausibly be coupled in many low-cost ways, picking the single most extreme one feels like overfitting to the corners of the polytope.

So here's a thought: what if I don't *want* the vertex? What if I deliberately stay in the interior, and ask for a plan that's cheap *and* smooth? Smoothness of a joint distribution has a precise meter — its entropy `h(P) = −Σ p_ij log p_ij`. The vertex has tiny entropy (it's nearly deterministic). The smoothest possible coupling, with no regard to cost at all, is the independence table `rc^T`, where mass spreads as `p_ij = r_i c_j`. There's a clean inequality I can lean on: for *any* coupling with marginals `r,c`,

```
h(P) ≤ h(r) + h(c),
```

and it's tight exactly at `P = rc^T`, since `h(rc^T) = h(r)+h(c)`. Even nicer, the gap is something I recognize:

```
h(r) + h(c) − h(P) = KL(P || rc^T) = I(X;Y),
```

the mutual information of the coupled variables. So "high entropy" = "close to the independence table" = "low mutual information between source and destination." That gives me a knob. Tables with `KL(P||rc^T) ≤ α` — call that set `U_α(r,c)` — are the couplings with *sufficient* entropy, `h(P) ≥ h(r)+h(c)−α`. As `α` shrinks I'm forced toward the smooth independence table; as `α` grows I get the whole polytope back and recover the original LP. So I could define a distance by minimizing cost over this entropy-ball:

```
d_{M,α}(r,c) = min_{P ∈ U_α(r,c)} ⟨P, M⟩.
```

There's a principle underneath this that isn't just my taste: the maximum-entropy principle (Jaynes). When the data don't pin down a unique plan, the least committal, most plausible choice is the maximum-entropy one consistent with the constraints. For a *given* level of cost, pick the smoothest plan. And there's prior evidence this kind of move is sensible — Ferradans et al. (2013) regularize the transport LP in color-transfer to tame an over-irregular matching, using graph-based norms. They penalize for regularity. I want regularity too, but I want a penalty that *also* keeps me with a genuine distance and, crucially, buys me speed. The entropy/mutual-information ball is the natural candidate because entropy is exactly the smoothness meter here.

Before I worry about computing it, does this even still behave like a distance? Two sanity checks at the extremes. If `α` is large enough, the entropy constraint is vacuous. For any joint law, `h(X,Y) = h(X) + h(Y|X) ≥ h(X)` and also `h(X,Y) ≥ h(Y)`, so in particular `h(P) ≥ ½(h(r)+h(c))`. Therefore `KL(P||rc^T) = h(r)+h(c)-h(P) ≤ ½(h(r)+h(c))`; once `α` clears that upper bound, every coupling is allowed, `U_α = U`, and `d_{M,α} = d_M`, the ordinary transport distance. Good, it's a genuine generalization. At the other end, `α = 0` forces `KL(P||rc^T)=0`, so `P = rc^T`. Then `d_{M,0} = ⟨rc^T, M⟩ = r^T M c`, a closed form — mass moves as if source and destination were independent. If `M` is a squared Euclidean distance matrix, `m_ij = ‖φ_i − φ_j‖²`, then

```
r^T M c = Σ_ij r_i c_j (‖φ_i‖² + ‖φ_j‖² − 2⟨φ_i,φ_j⟩)
        = r^T u + c^T u − 2 r^T K c,
```

with `u_i = ‖φ_i‖²` and `K_ij = ⟨φ_i,φ_j⟩`. The first two terms are separate functions of `r` and `c`, and the last term is minus a positive-definite bilinear kernel, so `r^T M c` is negative definite on the simplex and `exp(−t r^T M c)` is positive definite for `t > 0`. So the two endpoints are sane.

Now the real worry: is the *interior* version still a metric for `α` in between? Symmetry is immediate from `M` symmetric. The triangle inequality is the thing. For the plain transport distance the proof uses the gluing lemma: take the optimal coupling `P` of `(x,y)` and the optimal coupling `Q` of `(y,z)`, glue them along `y` into a coupling of `(x,z)`, and bound its cost by the sum. Let me try to glue *and* carry the entropy constraint, because if the glued plan falls out of the entropy ball the whole argument collapses. Define

```
t_ijk = p_ij q_jk / y_j   (and 0 wherever y_j = 0),   S_ik = Σ_j t_ijk.
```

First, is `S` even in the right polytope? Column sums:

```
Σ_i S_ik = Σ_i Σ_j p_ij q_jk / y_j = Σ_j (q_jk / y_j) Σ_i p_ij = Σ_j (q_jk / y_j) y_j = Σ_j q_jk = z_k.
```

Row sums, symmetrically, `Σ_k S_ik = Σ_j (p_ij/y_j) Σ_k q_jk = Σ_j (p_ij/y_j) y_j = x_i`. So `S ∈ U(x,z)`. Now the entropy part. Read `t_ijk` as the joint law of a triple `(X,Y,Z)`. By construction

```
p(X,Y,Z) = p(X,Y) p(Y,Z) / p(Y) = p(X) p(Y|X) p(Z|Y),
```

so `Z` depends on `X` only through `Y` — it's a Markov chain `X → Y → Z`. The data-processing inequality then says `I(X;Y) ≥ I(X;Z)`. Rewrite both as entropies: `I(X;Z) = h(X)+h(Z)−h(X,Z)` and `I(X;Y)=h(X)+h(Y)−h(X,Y)`, and `S` is the `(X,Z)` marginal of `T`, so `h(S) = h(X,Z)`. Since `P ∈ U_α(x,y)` gives `I(X;Y) ≤ α`, I'd get `I(X;Z) ≤ α`, i.e. `h(S) ≥ h(x)+h(z)−α`, so `S ∈ U_α(x,z)`. If that holds, the entropy ball survives the gluing.

I'm leaning on DPI here, which I trust in the abstract, but the construction `t_ijk = p_ij q_jk / y_j` is fiddly enough — that divide by `y_j`, the claim that `S` lands in the right polytope and inside the ball — that I don't want to take it on faith. Let me build one concrete instance and just look. Take `x = (0.5, 0.5)`, `y = (0.3, 0.7)`, `z = (0.6, 0.4)`, and pick any two couplings with the right marginals,

```
P = [[0.2, 0.3],[0.1, 0.4]]   (rows 0.5,0.5 = x;  cols 0.3,0.7 = y),
Q = [[0.25, 0.05],[0.35, 0.35]] (rows 0.3,0.7 = y; cols 0.6,0.4 = z).
```

Gluing `S_ik = Σ_j p_ij q_jk / y_j` gives row sums `(0.5, 0.5)` and column sums `(0.6, 0.4)` — exactly `x` and `z`, so `S ∈ U(x,z)` as the algebra promised. For the entropy part I compute the two mutual informations directly: `I(X;Y) = Σ p_ij log(p_ij/(x_i y_j)) = 0.0242` nats, and for the glued table `I(X;Z) = 0.0023` nats. So `I(X;Z) < I(X;Y)` holds on the nose — DPI is doing what I claimed, and the gluing landed comfortably inside the smaller mutual-information ball, not just barely. That's the one step I was nervous about, and it checks out. Now the cost bound is the standard chain, using `m_ik ≤ m_ij + m_jk`:

```
d_{M,α}(x,z) ≤ ⟨S, M⟩ = Σ_ik m_ik Σ_j p_ij q_jk / y_j
             ≤ Σ_ijk (m_ij + m_jk) p_ij q_jk / y_j
             = Σ_ij m_ij p_ij Σ_k q_jk/y_j + Σ_jk m_jk q_jk Σ_i p_ij/y_j
             = Σ_ij m_ij p_ij + Σ_jk m_jk q_jk   (since Σ_k q_jk = y_j, Σ_i p_ij = y_j)
             = d_{M,α}(x,y) + d_{M,α}(y,z).
```

The triangle inequality holds for *every* `α ≥ 0`. (It can't satisfy the coincidence axiom for small `α`, because `d_{M,α}(r,r) > 0` when `h(r) > 0` — forcing entropy keeps you off the diagonal — but multiplying by the indicator `1_{r≠c}` fixes that if I need a true metric.) So this entropy-regularized transport is a legitimate distance. Good — now I've earned the right to make it fast.

How do I actually compute `min ⟨P,M⟩` subject to `KL(P||rc^T) ≤ α`? A hard ball constraint is awkward. Let me move the entropy into the objective with a multiplier — penalize plans of low entropy instead of constraining them:

```
P^λ = argmin_{P ∈ U(r,c)} ⟨P, M⟩ − (1/λ) h(P).
```

For a fixed pair `(r,c)`, the Lagrange multiplier that enforces a chosen active entropy level may depend on that pair, so I should not pretend one global `λ` represents one global `α` for all histograms. But the penalized program is the computational object I can tune directly. And look — `−h(P) = Σ p_ij log p_ij` is *strictly* convex. That changes the character of the problem. The plain LP had a flat linear objective over a polytope, so its solution sat at a corner and there could be ties; a strictly convex objective over a convex feasible set has at most one minimizer, so with positive marginals I'd expect a single plan, positive on every allowed row and column, and the corner-jumping non-differentiability to be gone. I'll come back and check that the thing I actually compute is this unique interior minimizer rather than just hoping so. The payoff I'm chasing is computational, though, so let me find the minimizer's form.

Constrained, strictly convex — write the Lagrangian for the two marginal equalities, with multiplier vectors `α` for the row constraint `P1 = r` and `β` for the column constraint `P^T1 = c`:

```
L(P, α, β) = Σ_ij [ (1/λ) p_ij log p_ij + p_ij m_ij ] + α^T(P1 − r) + β^T(P^T1 − c).
```

Stationarity in each `p_ij`:

```
∂L/∂p_ij = (1/λ)(log p_ij + 1) + m_ij + α_i + β_j = 0
  ⟹ log p_ij = −1 − λ m_ij − λ α_i − λ β_j
  ⟹ p_ij = e^{−1} · e^{−λ α_i} · e^{−λ m_ij} · e^{−λ β_j}.
```

Stop and look at the shape of that. The `(i,j)` dependence factors completely. The constant `e^{-1}` can be absorbed into either one-dimensional factor because `u` and `v` are only identified up to reciprocal rescaling. Define `u_i = e^{−λ α_i}`, absorb `e^{-1}` into `v_j`, and set the coupling factor to `K_ij = e^{−λ m_ij}`. Then

```
p_ij^λ = u_i K_ij v_j,   i.e.   P^λ = diag(u) · K · diag(v),   K = e^{−λ M}.
```

The optimal plan is a fixed nonnegative kernel `K = e^{−λM}`, **rescaled** by one positive vector across rows and one across columns. Whatever the right `u, v` are, `u → su`, `v → v/s` gives the same `P`, so the two vectors can only be pinned down up to a single multiplicative factor — but the plan `P^λ` itself I expect to be unique by the convexity argument above; I'll verify that on a concrete instance before trusting it.

Let me make this fully concrete on a case small enough to read, both to confirm the factored form really is the penalized minimizer and to see how the constants behave. Take `d = 2`, `r = (0.6, 0.4)`, `c = (0.5, 0.5)`, the cost `M = [[0,1],[1,0]]`, and `λ = 1`. Then `K = e^{−M} = [[1, 0.3679],[0.3679, 1]]`. The claim is that the minimizer of `⟨P,M⟩ − h(P)` over `U(r,c)` has the form `p_ij = u_i K_ij v_j`. There's a clean necessary-and-sufficient test for that: `log p_ij + λ m_ij` must be additively separable into `f_i + g_j`, since `log p_ij = log u_i + log K_ij + log v_j = (log u_i) − λ m_ij + (log v_j)`. So if I find the true minimizer some other way and form `A_ij = log p_ij + λ m_ij`, the cross-difference `A_11 − A_12 − A_21 + A_22` must be zero. I solve the `2×2` penalized program directly (it's four unknowns with two marginal equalities, a one-parameter convex line search), get the minimizer, and compute that cross-difference: it comes out `−2.2 × 10⁻¹⁶`, i.e. machine zero. So the optimum genuinely factors as `diag(u) K diag(v)` — this isn't a form I'm imposing, it's forced by stationarity, exactly as the Lagrangian said. And perturbing that plan along the only marginal-preserving direction `P + ε[[1,−1],[−1,1]]` raises the penalized objective for `ε = ±0.01, ±0.02` in every case, so it really is the minimizer, not a saddle. The "unique interior minimizer" I was hoping for is the one I'm computing.

That form — find positive diagonal scalings of a fixed nonnegative matrix so that the result has prescribed row and column sums — is matrix scaling, and it rings a bell: it's what Sinkhorn & Knopp (1967) studied. Their result is that for a nonnegative `K` with the right support (automatic here, since `K = e^{−λM} > 0` entrywise) the scaled matrix `diag(u) K diag(v)` with prescribed positive marginals is unique, and the factors are unique up to `u → su`, `v → v/s` — which matches the rescaling freedom I just saw. The economists have run the same iteration for decades under "gravity models" of origin–destination traffic, and it's iterative proportional fitting / RAS under other names. So I may not have to solve an LP at all: I just have to find the row/column scalings of `K` that produce marginals `r` and `c`. The combinatorial search for `2d−1` active cells has evaporated; what's left is rescaling a dense matrix — *if* the rescaling actually converges, which I should not assume from the existence theorem alone.

Let me write the scaling conditions explicitly and turn them into an iteration. The row-sum constraint:

```
diag(u) K diag(v) 1 = r  ⟹  u ⊙ (K v) = r  ⟹  u = r / (K v),
```

elementwise. The column-sum constraint:

```
diag(v) K^T diag(u) 1 = c  ⟹  v ⊙ (K^T u) = c  ⟹  v = c / (K^T u).
```

These two are coupled — `u` depends on `v` and vice versa — so I alternate: fix `v`, set `u = r/(Kv)`; fix that `u`, set `v = c/(K^T u)`; repeat. Each half-step is just a matrix–vector product `Kv` or `K^T u` plus an elementwise divide, so one iteration is `O(d²)`. No factorization, no pivoting, no combinatorial bookkeeping.

Why should this loop converge rather than oscillate? Each half-step is a projection. The set of matrices with the right row sums, `C₁ = {P : P1 = r}`, and the set with the right column sums, `C₂ = {P : P^T1 = c}`, are both *affine*, and `U(r,c) = C₁ ∩ C₂`. Setting `u = r/(Kv)` is precisely the KL (Bregman) projection of the current `diag(u)K diag(v)` onto `C₁` — it minimally adjusts the row scaling to fix the row marginal — and the `v` update is the KL projection onto `C₂`. Alternating Bregman projections onto two affine sets is a setting where convergence to the intersection is known, and there's a sharper story: the row-then-column map should be a contraction in Hilbert's projective metric (Birkhoff's theorem / nonlinear Perron–Frobenius, worked out for this iteration by Franklin & Lorenz), which would give a geometric rate.

I'd rather watch it than quote it. On the same `2×2` instance I run the alternation from `u = v = 1` and track the column-marginal error `‖(diag(u)K diag(v))^T 1 − c‖` after each full sweep:

```
6.5×10⁻², 1.34×10⁻², 2.71×10⁻³, 5.44×10⁻⁴, 1.09×10⁻⁴, 2.19×10⁻⁵, 4.40×10⁻⁶, 8.83×10⁻⁷, …
```

The error drops by a near-constant factor each sweep; the successive ratios are `0.205, 0.202, 0.201, 0.201, 0.201, …`, settling to a fixed `≈ 0.20 < 1`. That's exactly linear (geometric) convergence with a contraction ratio around `0.2` for this `K`, just as the Hilbert-metric picture predicts — and it's fast: eight cheap sweeps already reach `10⁻⁶`. So convergence isn't wishful; the iteration contracts, and at `O(d²)` per sweep this is cheap for any `M`. I'll note for later that the contraction ratio is a function of `K`: I should expect it to drift toward 1 as `K` gets more concentrated.

Let me also see what `λ` is really trading, and again I'll read it off the `2×2` case rather than argue it. `λ` is the inverse temperature `1/ε`. When `λ → 0`, `K = e^{−λM} → 11^T`, so every scaling should push toward `P = rc^T`, the independence table. Concretely the independence table for `r=(0.6,0.4)`, `c=(0.5,0.5)` is `(0.30, 0.30, 0.20, 0.20)`; at `λ = 0.001` the Sinkhorn plan comes out `(0.3001, 0.2999, 0.1999, 0.2001)` and its cost is `0.4998`, against the closed form `r^T M c = 0.5`. So the small-`λ` end really does degenerate to the smoothed independence distance — the numbers land where the algebra said. At the other end I want `d_M^λ → d_M`, the true EMD. Brute-forcing the `2×2` LP, the exact transport cost is `0.1` (move the excess `0.1` mass from bin 1 to bin 2). Running Sinkhorn at increasing `λ`: `λ=3` gives `0.1093`, `λ=8` gives `0.10000045`, `λ=20` gives `0.1000000`. So as `λ` grows the smoothed cost climbs down to the exact EMD, just slowly at first and then locking on. `λ` does slide continuously from "fully smoothed independence kernel" to "exact EMD," and I've now seen both endpoints hit their predicted values.

The cost of pushing `λ` up shows up in the same experiments. The contraction ratio I measured (`≈0.2` at `λ=1`) is a function of `K`, and as `K` becomes diagonally dominant at large `λ` it drifts toward 1 — which is why I needed thousands of sweeps to nail `λ=20` to the EMD where eight sufficed at `λ=1`. Worse, the entries `e^{−λ m_ij}` underflow to literal zeros in floating point, after which `Kv` can hit a zero and the divide blows up — exactly the failure the code guards against. So there's a real reason *not* to crank `λ`: moderate `λ` is both faster and numerically safer, and it's the regime where the distance is smooth and robust anyway. If I ever need very small `ε`, the fix is to carry `log u`, `log v` and do the products with a log-sum-exp, but for sensible `λ` the plain multiplicative loop is fine.

One more practical thing falls out of the form. Everything is matrix–vector products against the *same* `K`. If I want the distance from one `r` to a whole family `C = [c₁,…,c_N]`, I replace the vector `v` by a matrix and the column targets `c` by `C`, and the entire iteration runs as `O(d² N)` dense linear algebra — perfectly vectorized, ideal for a GPU. The LP had no such structure; here the speedup is not just better constants, but an embarrassingly parallel kernel.

Let me also lock down the relationship between the penalized object I'm computing, `d_M^λ`, and the constrained distance `d_{M,α}` whose triangle inequality I just proved. For a fixed pair `(r,c)`, when the entropy constraint is active, Lagrange duality gives a pair-dependent multiplier `λ` that lands on the same optimizer. The entropy of `P^λ` decreases monotonically as `λ` increases (more weight on cost means a sharper, lower-entropy plan), so to hit a target entropy `h(r)+h(c)−α` I can bisection-search `λ`. In practice the penalized `d_M^λ` is the cheaper object to compute and tune directly. I should keep the theorem straight, though: the metric statement belongs to the hard-constrained `d_{M,α}` (with the coincidence caveat fixed by `1_{r≠c}`), while the fixed-`λ` quantity is a fast smooth surrogate and is not proved to be a metric.

So the algorithm is: build `K = e^{−λM}`; initialize positive scaling vectors; alternate `v = c/(K^T u)` and `u = r/(Kv)` until the marginals are hit; the plan is `diag(u)K diag(v)` and the distance is `⟨diag(u)K diag(v), M⟩ = Σ u_i (K ⊙ M)_ij v_j`. In implementation notation I will pass `reg = ε = 1/λ`, so `K = exp(-M/reg)`. I also need to respect two details from the standard loop: if the source histogram has zero rows, remove those rows and put zeros back in the returned plan; and if `b` is a whole matrix of target histograms, compute the vector of costs directly instead of materializing a `d × d × N` stack of plans.

```python
import warnings
import numpy as np

def transport_plan(a, b, M, reg, num_iter=1000, stop_thr=1e-9, warn=True):
    # One target histogram. For many target columns, use transport_cost below.
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    M = np.asarray(M, dtype=float)
    if b.ndim != 1:
        raise ValueError("transport_plan expects one target histogram; use transport_cost for many targets.")

    active = a > 0
    if not np.any(active):
        raise ValueError("source histogram has no positive mass")
    if not np.all(active):
        P = np.zeros_like(M, dtype=float)
        P[active, :] = transport_plan(a[active], b, M[active, :], reg, num_iter, stop_thr, warn)
        return P

    # K = exp(-M/eps) = exp(-lambda M): the fixed Gibbs kernel the optimum rescales.
    K = np.exp(-M / reg)

    # u, v are the row/column scalings; the KKT form says P = diag(u) K diag(v).
    u = np.ones(M.shape[0]) / M.shape[0]
    v = np.ones(M.shape[1]) / M.shape[1]

    # Pre-divide K by a so the u-update is one matmul: u = 1 / (Kp v), Kp = diag(1/a) K.
    Kp = (1.0 / a)[:, None] * K

    for it in range(num_iter):
        uprev, vprev = u.copy(), v.copy()
        # column projection: v = b / (K^T u) enforces the column marginal b.
        KtU = K.T @ u
        v = b / KtU
        # row projection: u = a / (K v) enforces the row marginal a (folded into Kp).
        u = 1.0 / (Kp @ v)

        if KtU.min() == 0 or not np.all(np.isfinite(u)) or not np.all(np.isfinite(v)):
            if warn:
                warnings.warn(f"numerical errors at iteration {it}")
            u, v = uprev, vprev
            break

        if it % 10 == 0:
            # convergence = how far the current plan's column marginal is from b.
            marginal = v * (K.T @ u)          # = (diag(u) K diag(v))^T 1
            if np.linalg.norm(marginal - b) < stop_thr:
                break

    P = u[:, None] * K * v[None, :]            # the transport plan diag(u) K diag(v)
    return P

def transport_cost(a, b, M, reg, num_iter=1000, stop_thr=1e-9, warn=True):
    # Mirrors the vectorized cost path: if b has columns c_1,...,c_N, return N costs.
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    M = np.asarray(M, dtype=float)
    if b.ndim == 1:
        return float(np.sum(transport_plan(a, b, M, reg, num_iter, stop_thr, warn) * M))
    if b.ndim != 2:
        raise ValueError("target histogram must be a vector or a matrix with target columns")

    active = a > 0
    if not np.any(active):
        raise ValueError("source histogram has no positive mass")
    a = a[active]
    M = M[active, :]

    K = np.exp(-M / reg)
    u = np.ones((M.shape[0], b.shape[1])) / M.shape[0]
    v = np.ones((M.shape[1], b.shape[1])) / M.shape[1]
    Kp = (1.0 / a)[:, None] * K

    for it in range(num_iter):
        uprev, vprev = u.copy(), v.copy()
        KtU = K.T @ u
        v = b / KtU
        u = 1.0 / (Kp @ v)

        if np.any(KtU == 0) or not np.all(np.isfinite(u)) or not np.all(np.isfinite(v)):
            if warn:
                warnings.warn(f"numerical errors at iteration {it}")
            u, v = uprev, vprev
            break

        if it % 10 == 0:
            marginal = np.einsum("ik,ij,jk->jk", u, K, v)
            if np.linalg.norm(marginal - b) < stop_thr:
                break

    return np.einsum("ik,ij,jk,ij->k", u, K, v, M)

def stable_transport_plan(a, b, M, reg, num_iter=1000, stop_thr=1e-9):
    # Same fixed point in log-scalings; useful when exp(-M/reg) underflows.
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    M = np.asarray(M, dtype=float)
    if b.ndim != 1:
        raise ValueError("stable_transport_plan expects one target histogram")

    active = a > 0
    if not np.any(active):
        raise ValueError("source histogram has no positive mass")
    if not np.all(active):
        P = np.zeros_like(M, dtype=float)
        P[active, :] = stable_transport_plan(a[active], b, M[active, :], reg, num_iter, stop_thr)
        return P

    Mr = -M / reg
    loga = np.log(a)
    logb = np.full_like(b, -np.inf, dtype=float)
    logb[b > 0] = np.log(b[b > 0])
    u = np.zeros(M.shape[0])                    # log row scaling
    v = np.zeros(M.shape[1])                    # log column scaling

    def logsumexp(A, axis):
        m = A.max(axis=axis, keepdims=True)
        return (m + np.log(np.exp(A - m).sum(axis=axis, keepdims=True))).squeeze(axis)

    for it in range(num_iter):
        v = logb - logsumexp(Mr + u[:, None], axis=0)
        u = loga - logsumexp(Mr + v[None, :], axis=1)
        if it % 10 == 0:
            logP = Mr + u[:, None] + v[None, :]
            if np.linalg.norm(np.exp(logP).sum(axis=0) - b) < stop_thr:
                break

    return np.exp(Mr + u[:, None] + v[None, :])
```

The whole chain: the exact transport LP is a metric but lands on a brittle `≤2d−1`-nonzero vertex and costs `O(d^3 log d)`, non-differentiable and serial. Asking for the smoothest low-cost coupling by bounding mutual information gives a constrained family whose triangle inequality survives gluing because data processing keeps the glued coupling inside the same entropy ball. Moving that constraint into a negative-entropy penalty with weight `1/λ` makes the computation strictly convex and smooth, with the metric theorem now understood as belonging to the constrained form. Stationarity of the penalized objective forces the optimum to factor as `diag(u) K diag(v)` with `K = e^{−λM}`, which is exactly a Sinkhorn–Knopp matrix-scaling problem; the marginal conditions are alternating KL projections that contract in Hilbert's metric, so `O(d²)` matrix–vector products — vectorizable across many targets, GPU-friendly — replace the LP, with `λ` trading sharpness (closer to EMD, slower, prone to underflow) against smoothing and speed.
