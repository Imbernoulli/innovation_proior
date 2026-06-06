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

Before I worry about computing it, does this even still behave like a distance? Two sanity checks at the extremes. If `α` is large enough, the entropy constraint is vacuous — for any `P` in the polytope `h(P)` is bounded below by `½(h(r)+h(c))`, so once `α` clears that, `U_α = U` and `d_{M,α} = d_M`, the ordinary transport distance. Good, it's a genuine generalization. At the other end, `α = 0` forces `h(P) = h(r)+h(c)`, which by the equality case happens only at `P = rc^T`. So `d_{M,0} = ⟨rc^T, M⟩ = r^T M c`, a closed form — mass moves as if source and destination were independent. (And if `M` is a Euclidean distance matrix, `m_ij = ‖φ_i − φ_j‖²`, then `r^T M c = r^T u + c^T u − 2 r^T K c` with `u_i = ‖φ_i‖²`, `K_ij = ⟨φ_i,φ_j⟩`, which is a sum of negative-definite kernels, so `exp(−t r^T M c)` is a positive-definite kernel — a usable "independence kernel.") So the two endpoints are sane.

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

so `Z` depends on `X` only through `Y` — it's a Markov chain `X → Y → Z`. The data-processing inequality then says `I(X;Y) ≥ I(X;Z)`. Rewrite both as entropies: `I(X;Z) = h(X)+h(Z)−h(X,Z)` and `I(X;Y)=h(X)+h(Y)−h(X,Y)`, and `S` is the `(X,Z)` marginal of `T`, so `h(S) = h(X,Z)`. Since `P ∈ U_α(x,y)` gives `I(X;Y) ≤ α`, I get `I(X;Z) ≤ α`, i.e. `h(S) ≥ h(x)+h(z)−α`, so `S ∈ U_α(x,z)`. The entropy ball survives the gluing. Now the cost bound is the standard chain, using `m_ik ≤ m_ij + m_jk`:

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

By Lagrangian duality each `α` corresponds to some `λ ∈ [0,∞]`, so I can work with the penalized version and recover the constrained one later if I want. And look — `−h(P) = Σ p_ij log p_ij` is *strictly* convex. That instantly changes the character of the problem. The plain LP had a flat linear objective over a polytope, so its solution sat at a corner and there could be ties; the strictly convex penalty makes the objective curve into the interior, so the minimizer is **unique** and sits *inside* `U(r,c)`, not at a vertex. The non-differentiability that came from corner-jumping is gone too. So even before I count operations, the penalized problem is better-behaved: unique, smooth, interior. That's encouraging, but the payoff I'm chasing is computational, so let me find the actual minimizer.

Constrained, strictly convex — write the Lagrangian for the two marginal equalities, with multiplier vectors `α` for the row constraint `P1 = r` and `β` for the column constraint `P^T1 = c`:

```
L(P, α, β) = Σ_ij [ (1/λ) p_ij log p_ij + p_ij m_ij ] + α^T(P1 − r) + β^T(P^T1 − c).
```

Stationarity in each `p_ij`:

```
∂L/∂p_ij = (1/λ)(log p_ij + 1) + m_ij + α_i + β_j = 0
  ⟹ log p_ij = −1 − λ m_ij − λ α_i − λ β_j
  ⟹ p_ij = e^{−½ − λ α_i} · e^{−λ m_ij} · e^{−½ − λ β_j}.
```

Stop and look at the shape of that. The `(i,j)` dependence factors completely. Group the `i`-only factor into `u_i = e^{−½ − λ α_i}`, the `j`-only factor into `v_j = e^{−½ − λ β_j}`, and the coupling factor into `K_ij = e^{−λ m_ij}`. Then

```
p_ij^λ = u_i K_ij v_j,   i.e.   P^λ = diag(u) · K · diag(v),   K = e^{−λ M}.
```

The optimal plan is a fixed nonnegative kernel `K = e^{−λM}`, **rescaled** by one positive vector across rows and one across columns. The strict convexity guarantees a unique `P^λ`, and `u, v` are pinned down up to a single multiplicative factor (`u → su`, `v → v/s` gives the same `P`).

That form is *exactly* a matrix-scaling object, and I've seen this before. Sinkhorn & Knopp (1967): given a nonnegative matrix `K` with total support, there's a **unique** pair of positive diagonal scalings `diag(u) K diag(v)` achieving prescribed positive row and column sums, and you find it by alternately rescaling rows to hit the target row sums and columns to hit the target column sums — and that alternation **converges**. Here `K = e^{−λM} > 0` entrywise, so total support is free. The economists have run this for decades under "gravity models" of origin–destination traffic, and it's the same iteration as iterative proportional fitting / RAS. So I don't have to solve an LP at all: I just have to find the row/column scalings of `K` that produce marginals `r` and `c`. The combinatorial search for `2d−1` active cells has evaporated; what's left is rescaling a dense matrix.

Let me write the scaling conditions explicitly and turn them into an iteration. The row-sum constraint:

```
diag(u) K diag(v) 1 = r  ⟹  u ⊙ (K v) = r  ⟹  u = r / (K v),
```

elementwise. The column-sum constraint:

```
diag(v) K^T diag(u) 1 = c  ⟹  v ⊙ (K^T u) = c  ⟹  v = c / (K^T u).
```

These two are coupled — `u` depends on `v` and vice versa — so I alternate: fix `v`, set `u = r/(Kv)`; fix that `u`, set `v = c/(K^T u)`; repeat. Each half-step is just a matrix–vector product `Kv` or `K^T u` plus an elementwise divide, so one iteration is `O(d²)`. No factorization, no pivoting, no combinatorial bookkeeping.

Why should this loop converge rather than oscillate? Each half-step is a projection. The set of matrices with the right row sums, `C₁ = {P : P1 = r}`, and the set with the right column sums, `C₂ = {P : P^T1 = c}`, are both *affine*, and `U(r,c) = C₁ ∩ C₂`. Setting `u = r/(Kv)` is precisely the KL (Bregman) projection of the current `diag(u)K diag(v)` onto `C₁` — it minimally adjusts the row scaling to fix the row marginal — and the `v` update is the KL projection onto `C₂`. Alternating Bregman projections onto two affine sets converge to their intersection. And the rate is geometric: the row-then-column map is a contraction in Hilbert's projective metric (this is Birkhoff's theorem / nonlinear Perron–Frobenius, made precise for this iteration by Franklin & Lorenz), with a contraction ratio `< 1` set by `K`. So linear convergence, cheaply, for any `M`.

Let me also see what `λ` is really trading. `λ` is the inverse temperature `1/ε`. When `λ → 0`, `K = e^{−λM} → 11^T`, every scaling pushes toward `P = rc^T`, the independence table — maximal smoothing, the distance degenerates toward `r^T M c`. When `λ → ∞`, `K` concentrates on the smallest-cost entries, `P^λ` sharpens toward the LP vertex and `d_M^λ → d_M`. So `λ` slides continuously from "fully smoothed independence kernel" to "exact EMD." The cost of pushing `λ` up: as `K` becomes diagonally dominant the Hilbert-metric contraction ratio approaches 1, so it takes more iterations to converge; and worse, the entries `e^{−λ m_ij}` underflow to literal zeros in floating point, after which `Kv` can hit a zero and the divide blows up. So there's a real reason *not* to crank `λ` — moderate `λ` is both faster and numerically safer, and it's the regime where the distance is smooth and robust anyway. If I ever need very small `ε`, the fix is to carry `log u`, `log v` and do the products with a log-sum-exp, but for sensible `λ` the plain multiplicative loop is fine.

One more practical thing falls out of the form. Everything is matrix–vector products against the *same* `K`. If I want the distance from one `r` to a whole family `C = [c₁,…,c_N]`, I replace the vector `v` by a matrix and the column targets `c` by `C`, and the entire iteration runs as `O(d² N)` dense linear algebra — perfectly vectorized, ideal for a GPU. The LP had no such structure; here it's free. That's the "lightspeed" part: not just better constants, but an embarrassingly parallel kernel.

Let me also lock down the relationship between the penalized object I'm computing, `d_M^λ`, and the constrained distance `d_{M,α}` I proved is a metric. They're linked by duality — each `α` has a matching `λ`. The entropy of `P^λ` decreases monotonically as `λ` increases (more weight on cost ⇒ sharper ⇒ less entropy), so to hit a target entropy `h(r)+h(c)−α` I can just bisection-search `λ`. But honestly, the penalized `d_M^λ` is cheaper and already behaves well, so I'll use it directly as my distance — and note, strictly, that as a Lagrangian relaxation it isn't *provably* the metric `d_{M,α}` for a fixed `λ`, though it inherits all the good behavior.

So the algorithm is: build `K = e^{−λM}`; initialize a positive vector; alternate `u = r/(Kv)`, `v = c/(K^T u)` until the marginals are hit; the plan is `diag(u)K diag(v)` and the distance is `⟨diag(u)K diag(v), M⟩ = Σ u_i (K ⊙ M)_ij v_j`. Let me write it.

```python
import numpy as np

def sinkhorn(a, b, M, reg, num_iter=1000, stop_thr=1e-9):
    # a, b: source/target histograms in the simplex; M: ground cost; reg = eps = 1/lambda.
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    M = np.asarray(M, dtype=float)

    # K = exp(-M/eps) = exp(-lambda M): the fixed Gibbs kernel the optimum rescales.
    K = np.exp(-M / reg)

    # u, v are the row/column scalings; the KKT form says P = diag(u) K diag(v).
    u = np.ones(M.shape[0]) / M.shape[0]
    v = np.ones(M.shape[1]) / M.shape[1]

    # Pre-divide K by a so the u-update is one matmul: u = 1 / (Kp v), Kp = diag(1/a) K.
    Kp = (1.0 / a)[:, None] * K

    for it in range(num_iter):
        # column projection: v = b / (K^T u) enforces the column marginal b.
        KtU = K.T @ u
        v = b / KtU
        # row projection: u = a / (K v) enforces the row marginal a (folded into Kp).
        u = 1.0 / (Kp @ v)

        if it % 10 == 0:
            # convergence = how far the current plan's column marginal is from b.
            marginal = v * (K.T @ u)          # = (diag(u) K diag(v))^T 1
            if np.linalg.norm(marginal - b) < stop_thr:
                break

    P = u[:, None] * K * v[None, :]            # the transport plan diag(u) K diag(v)
    return P

def sinkhorn_distance(a, b, M, reg, **kw):
    P = sinkhorn(a, b, M, reg, **kw)
    return np.sum(P * M)                        # <P, M>, the entropic transport cost
```

The whole chain: the exact transport LP is a metric but lands on a brittle `≤2d−1`-nonzero vertex and costs `O(d^3 log d)`, non-differentiable and serial. Asking instead for the *smoothest* low-cost coupling — bounding mutual information / penalizing negative entropy with weight `1/λ` — makes the objective strictly convex, so the optimum becomes unique and interior, and the gluing lemma (now carrying the entropy constraint through the data-processing inequality) keeps it a genuine distance. Stationarity of the Lagrangian forces the optimum to factor as `diag(u) K diag(v)` with `K = e^{−λM}`, which is exactly a Sinkhorn–Knopp matrix-scaling problem; the marginal conditions `u = r/(Kv)`, `v = c/(K^T u)` are alternating KL projections that contract in Hilbert's metric, so a handful of `O(d²)` matrix–vector products — vectorizable across many targets, GPU-friendly — replaces the LP, with `λ` trading sharpness (closer to EMD, slower, prone to underflow) against smoothing and speed.
