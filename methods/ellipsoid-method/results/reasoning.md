Let me start from the thing that actually bothers me, which is that we have no idea whether linear programming is hard. I can write down a system of linear inequalities `A x ≤ b` with integer coefficients and ask the flattest possible question — does a solution exist? — and I cannot answer it in time guaranteed to be polynomial in the size of what I wrote down. Simplex walks the vertices of the polyhedron and in practice it's wonderful, but Klee and Minty built those perturbed cubes where a natural pivot rule marches through all `2ⁿ` vertices, so as a worst-case bound simplex gives me `2ⁿ`, which is useless as a complexity statement. And the maddening part is that feasibility is in `NP ∩ co-NP`: if the system is feasible there's a solution I can exhibit and check, and if it's infeasible then by Farkas there's a nonnegative combination `yᵀA = 0`, `yᵀb < 0` I can exhibit and check. Both answers have short certificates. Everything in my experience says a problem with certificates on both sides should be in `P`. But nobody has shown it. And Karp's people have been asking, pointedly, whether this very problem is NP-complete — which would mean if I *did* put it in `P` I'd have collapsed `P = NP`. So either I find a polynomial algorithm and prove the problem isn't complete, or I'm staring at `P = NP`. That's the size of the prize. I want a `poly(L)` algorithm, `L` being the bit-length of the integer data.

So let me not think about vertices at all. Vertices are a combinatorial object and the combinatorics is exactly what blows up. Let me think about the feasible set `P = { x : A x ≤ b }` as a geometric body and ask: can I *trap* it? If I could keep a region that's guaranteed to contain `P` (when `P` is nonempty) and keep shrinking that region by a definite factor each step, then after enough steps the trapping region is so small that either I've stumbled into `P` or `P` had to be empty. That's a continuous, geometric search, not a walk on a discrete skeleton. The whole game becomes: how fast can I shrink the trap, and how do I know when to stop?

Two facts make this even conceivable, and they come straight from the integrality of the data. First, I don't have to trap all of `ℝⁿ`. If `P` is nonempty it contains a point inside the ball `S = { ‖x‖ ≤ 2^L }` — the determinant bounds behind linear-program certificates keep some feasible rational point that small, even when the feasible set is not a pointed polytope with convenient vertices. So my initial trap can just be that ball. Second — and this is the one that makes *stopping* possible — let me measure infeasibility by the residual `σ(x) = max_i (A_i x − b_i)`. Then `x` solves the system exactly when `σ(x) ≤ 0`. If the system is **infeasible**, then for *every* `x`, `σ(x) ≥ 2^{−L}`. There is no ambiguous strip near zero: feasible means the minimum residual over `S` is at most `0`, while infeasible means even the global minimum residual is at least `2^{−L}`. So I can choose an acceptance tolerance `ε < 2^{−L}` and know that a point with `σ(x) ≤ ε` cannot exist in the infeasible case. The integer data gives me both a bounded search region and a resolution at which the answer is forced.

Now, the trap. What shape should it be? The obvious thing is to do what the convex-optimization people do with cutting planes: keep a polyhedron containing the solution set, query a point, get a violated inequality, slice off the bad half-space, recurse. That's the Newman and Levin style of localization — maintain a polyhedral localization set, cut it with each new subgradient hyperplane. And it does shrink the set. But I keep hitting the same wall when I try to bound it: the description of the polyhedron *grows*. Every cut adds another facet. After `k` steps my localization set has `k` more inequalities, so the work to even find a query point inside it climbs as the algorithm runs, and I get no clean dimension-only handle on how fast the *volume* goes down. The complexity of the localization set is itself unbounded. That's no good — I need the trap to have a **fixed-size description** so the per-step cost stays flat, and I need a guaranteed shrink rate that depends only on the dimension `n`, not on how long I've been running.

So: not a polyhedron. I want a body that (a) is described by a constant number of parameters and (b) has a tidy notion of volume I can drive down. An **ellipsoid** is exactly that. An ellipsoid is the affine image of a ball, `E = { x + Q z : ‖z‖ ≤ 1 }`, carried by a center `x` (that's `n` numbers) and a matrix `Q` (that's `n²` numbers) — fixed size, forever, no matter how many cuts I make. Equivalently `E(A,a) = { x : (x−a)ᵀ A⁻¹ (x−a) ≤ 1 }` with `A = Q Qᵀ` positive definite. And its volume is `vol(E) = |det Q| · V_n = √(det A) · V_n` where `V_n` is the unit-ball volume — so volume is just a determinant. That's the object. This is, in fact, the geometry Shor was already using for nonsmooth convex minimization: his subgradient method with *space dilation* rescales coordinates in the gradient direction between steps, which is precisely carrying a variable metric — a matrix — and updating it by a low-rank change rather than appending constraints. The localization set in disguise is an ellipsoid. Yudin and Nemirovski, looking at the informational complexity of convex optimization, made that explicit as a method of central sections. So the trap-shape question already has an answer in that literature; what's *not* there is the thing I actually need — an integer-data, bit-length, exact-decision analysis. Let me build the geometric step first and then go get the polynomial bound.

One step of the trap, then. I have an ellipsoid `E` known to contain `P`, with center `x`. I evaluate the system at `x`. If `σ(x) ≤ 0`, I'm done — `x` is a solution. Otherwise some specific row is violated: there's an index `ℓ` with `A_ℓ x − b_ℓ = σ(x) > 0`. Every point of `P` satisfies `A_ℓ y ≤ b_ℓ`, i.e. `A_ℓ (y − x) ≤ b_ℓ − A_ℓ x < 0`. So `P` lies entirely in the half-space `{ y : A_ℓ (y − x) ≤ 0 }` — the half on the *other* side of the hyperplane through the center with normal `A_ℓ`. That means `P ⊆ E ∩ { A_ℓ(y−x) ≤ 0 }`, a **half-ellipsoid**: I've cut `E` through its own center and `P` is on one side. The cut passes through the center deliberately — a central cut throws away a full half of the ellipsoid's bulk, and (I'll see) the re-enclosure I get is then independent of how the body sits.

The half-ellipsoid is no longer in the family I can carry. If I keep the exact intersection, I am back to a localization set whose description grows after every cut. So I replace the half-ellipsoid by one ellipsoid again. It has to contain the whole half, because otherwise I might lose `P`; and it should have the least possible volume, because every bit of unnecessary enlargement slows the only progress measure I have. If that least-volume re-enclosure is always smaller than the ellipsoid I started with by a dimension-only factor, the determinant marches geometrically downward.

Let me actually find the minimum enclosing ellipsoid of a half-ball, because by affine invariance the half-ellipsoid case reduces to it. Take `E` to be the unit ball and, by rotating, take the cut to be `x₁ ≥ 0`, so I'm enclosing `H = { ‖x‖ ≤ 1, x₁ ≥ 0 }`. By symmetry the enclosing ellipsoid should be a ball squashed along the `x₁` axis and shifted in the `+x₁` direction: centered at `(t, 0, …, 0)` for some `t > 0`, stretched differently along `x₁` than along the rest. Write the candidate as

```
E⁺ = { x : (1/α²)(x₁ − t)² + (1/β²) Σ_{i≥2} x_i² ≤ 1 }.
```

I need `E⁺ ⊇ H`. For a fixed shift `t`, the north pole `x₁ = 1` forces `α ≥ 1 − t`; at the optimum I cannot leave slack there, because reducing `α` would reduce volume while preserving symmetry, so `α = 1 − t`. The equator rim `x₁ = 0`, `Σ_{i≥2} x_i² = 1`, forces

```
t²/α² + 1/β² ≤ 1,
```

so `β² ≥ 1/(1 − t²/α²)`, and again the optimum uses equality. Substituting `α = 1 − t` gives

```
β² = (1 − t)²/(1 − 2t),        0 < t < 1/2.
```

The volume, up to the unit-ball constant, is

```
α β^{n−1} = (1 − t)^n (1 − 2t)^{−(n−1)/2}.
```

Now minimize the log:

```
d/dt [ n ln(1 − t) − ((n−1)/2) ln(1 − 2t) ]
 = −n/(1 − t) + (n−1)/(1 − 2t).
```

Setting this to zero gives `(n−1)(1−t) = n(1−2t)`, hence `(n+1)t = 1` and

```
t = 1/(n+1),        α = n/(n+1),        β = n/√(n²−1).
```

So the only candidate that can be volume-minimal is

```
E⁺ = { x : ((n+1)/n)² (x₁ − 1/(n+1))² + ((n²−1)/n²) Σ_{i≥2} x_i² ≤ 1 }.
```

I still have to make sure no intermediate cross-section `0 < x₁ < 1` sticks out. Take any `x` with `x₁ ≥ 0` and `‖x‖² = Σ_i x_i² ≤ 1`, and expand the left side:

```
((n+1)/n)²(x₁ − 1/(n+1))² + ((n²−1)/n²) Σ_{i≥2} x_i²
 = (n²+2n+1)/n² · x₁² − ((n+1)/n)² · 2x₁/(n+1) + 1/n² + (n²−1)/n² · Σ_{i≥2} x_i²
 = (2n+2)/n² · x₁² − (2n+2)/n² · x₁ + 1/n² + (n²−1)/n² · Σ_{i≥1} x_i²
 = (2n+2)/n² · x₁(x₁ − 1) + 1/n² + (n²−1)/n² · ‖x‖².
```

I fold the `x₁²` term back into the full sum `Σ_{i≥1}` to get `‖x‖²`. Now `x₁ ∈ [0,1]` because `x₁ ≥ 0` and `x₁² ≤ ‖x‖² ≤ 1`, so `x₁(x₁−1) ≤ 0`, killing the first term; and `‖x‖² ≤ 1`, so the whole thing is `≤ 1/n² + (n²−1)/n² = 1`. So every point of the half-ball lies in `E⁺`, and the constants are not arbitrary: the minimum-volume calculation forced `t = 1/(n+1)`, `α = n/(n+1)`, and `β = n/√(n²−1)`.

Now the payoff, the volume ratio. `vol(E⁺)/vol(E) = α · β^{n−1}` (the unit ball had all semi-axes 1):

```
vol(E⁺)/vol(E) = (n/(n+1)) · (n²/(n²−1))^{(n−1)/2}.
```

Is this less than 1, and by how much? I can do it without losing a sign by using only `1+u ≤ e^u`. The first factor is

```
n/(n+1) = 1 − 1/(n+1) = 1 + (−1/(n+1)) ≤ e^{−1/(n+1)}.
```

The second factor has a positive increment:

```
n²/(n²−1) = 1 + 1/(n²−1) ≤ e^{1/(n²−1)}.
```

Raising it to `(n−1)/2` gives

```
(n²/(n²−1))^{(n−1)/2}
 ≤ e^{(n−1)/(2(n²−1))}
 = e^{1/(2(n+1))},
```

because `n²−1 = (n−1)(n+1)`. Multiplying the two bounds, or equivalently adding the two log bounds, gives

```
vol(E⁺)/vol(E) ≤ e^{−1/(n+1)} · e^{1/(2(n+1))} = e^{−1/(2(n+1))}.
```

There it is — `e^{−1/(2(n+1))} < 1`, a fixed shrink factor depending **only on the dimension `n`**, not on the data, not on the iteration count. Exactly the thing the polyhedral cutting-plane method couldn't promise. Every central cut and re-enclosure multiplies the volume by at most `e^{−1/(2(n+1))}`. The trap shrinks geometrically.

Now I need the actual update — `E⁺`'s center and matrix in the general case, not just the unit ball. Let the current ellipsoid be

```
E(A,x) = { y : (y−x)ᵀ A⁻¹ (y−x) ≤ 1 },
```

and choose a square root `S` with `A = S Sᵀ`, so `y = x + S u` maps the unit ball in `u`-space onto `E(A,x)`. The violated row gives a normal `a` with `P ⊆ { y : aᵀ(y−x) ≤ 0 }`. In `u`-coordinates the cut is `(Sᵀa)ᵀu ≤ 0`; after normalizing,

```
v = Sᵀa / √(aᵀ A a),
```

this is the canonical half-ball with normal `v`. For the side `vᵀu ≤ 0`, the half-ball center is shifted by `−v/(n+1)`, and the shape matrix in `u`-coordinates is

```
β² ( I − (2/(n+1)) v vᵀ ),        β² = n²/(n²−1).
```

Pushing this back through `S`, the center displacement becomes

```
S v = S Sᵀa / √(aᵀ A a) = A a / √(aᵀ A a).
```

So if I define

```
b := A a / √(aᵀ A a),
```

the center moves

```
x⁺ = x − (1/(n+1)) · b,
```

a step of fraction `1/(n+1)` of the way to the boundary on the feasible side of the central cut. The pushed-forward shape matrix is

```
A⁺ = (n²/(n²−1)) ( A − (2/(n+1)) · b bᵀ ).
```

The `(n²/(n²−1))` is the transverse factor `β²`; the `−(2/(n+1)) b bᵀ` term is the extra rank-one deflation in the cut direction, because `α²/β² = (n−1)/(n+1) = 1 − 2/(n+1)`. The determinant, the product of the squared semi-axes, is exactly `α² β^{2(n−1)}` times the old determinant, so the volume ratio is `α β^{n−1}` as before. It's a rank-one update of a positive-definite matrix — the same shape as a variable-metric quasi-Newton step, which is the family Shor's dilation method lives in. Good; it's `O(n²)` work per step (a matrix-vector product and an outer-product update), and the description never grows.

So the geometric engine is: start with `E₀ =` the ball of radius `2^L` centered at the origin (Lemma-1 guarantees any solution is inside it); at the center, either accept or read off a violated row `A_ℓ`; set `b = A_ℓ`-direction in the `A`-metric; update center and matrix by the formulas above; repeat. Volume falls by `e^{−1/(2(n+1))}` each time.

Now the question that turns this from "a method that converges" into "a polynomial-time decision": **when do I stop, and what do I conclude?** Volume is the bridge, but I have to be careful about which set has volume. The original feasible set can be flat: a single point or a lower-dimensional face has zero `n`-dimensional volume, so a volume floor for `P` itself would be false. The residual gap tells me how to repair that before I use volume. I choose, say,

```
ε = 2^{−(L+1)}
```

and search the relaxed body

```
P_ε = { y : A_i y ≤ b_i + ε for every i }.
```

If the original system is infeasible, Lemma 2 gives `σ(y) ≥ 2^{−L}` for every `y`, so `P_ε` is still empty. If the original system is feasible, a bounded feasible point exists, and relaxing every right-hand side by `ε` gives a full-dimensional cushion around it; because the row norms and the relevant determinants are bounded by `2^{O(L)}`, that cushion has volume at least `2^{−O(nL)}` inside the bounded search region. This is the body whose volume floor I can safely use. When the center `x` has `σ(x) > ε`, a row `ℓ` satisfies `A_ℓ x − b_ℓ > ε`, while every `y ∈ P_ε` satisfies `A_ℓ y ≤ b_ℓ + ε`, hence `A_ℓ(y−x) < 0`. So the same central cut is valid, now for `P_ε`.

Start: `E₀` is a ball of radius `2^L`, so `vol(E₀) ≤ (2^L)ⁿ V_n`, and `log vol(E₀) = O(nL)`. The relaxed feasible body, if it exists, has `log vol(P_ε) = −O(nL)`. Each step multiplies `vol` by `≤ e^{−1/(2(n+1))}`, i.e. subtracts `1/(2(n+1))` from `log vol`. To go from `log vol(E₀) = O(nL)` down past the floor `log vol(P_ε) = −O(nL)`, I need

```
iterations ≤ 2(n+1) · [ log vol(E₀) − log vol(P_ε) ] = 2(n+1) · O(nL) = O(n²L)
```

steps. Concretely I can fix a horizon like `M = 16 n² L` iterations. The logic of the decision is a contradiction argument. If the original system is feasible, then `P_ε` is nonempty with that volume floor, and every valid cut preserves `P_ε ⊆ E_k`; therefore `vol(E_k) ≥ vol(P_ε)` forever. But if I make `M` consecutive cuts without ever accepting a center with `σ(x_k) ≤ ε`, the shrink bound gives `vol(E_M) < vol(P_ε)`. Those two statements cannot both hold. So I run `M` center queries. If some center has `σ(x_k) ≤ ε`, I declare **feasible**, because an infeasible integer system cannot have such a point. If no center does, I declare **infeasible**, because a feasible system would have made the relaxed body too large to keep cutting past the computed floor. The `2^{−L}` infeasibility gap is what makes the rounded, finite-precision version exact rather than merely approximate.

Now I hit the wall that actually almost sinks the whole thing, and it's not geometry — it's arithmetic. The update has a `√(aᵀ A a)` in it. Square roots are irrational. If I insist on exact arithmetic, the entries of `A_k` and `x_k` become algebraic numbers whose bit-length can *grow* from step to step, and after `O(n²L)` steps the numbers themselves could be exponentially long — which would destroy the polynomial bound just as surely as the polyhedral method's growing facet count did. So I cannot run this exactly. I have to **round** every entry to some fixed precision `p` bits after the point. But the moment I round, I've broken the one thing the whole method rests on: the rounded ellipsoid `Ẽ⁺` is no longer *guaranteed* to contain the half-ellipsoid. A rounded-in ellipsoid that's a hair too small might exclude part of `P`, and then I could throw away the actual solution and wrongly declare infeasible. The rounding both saves me and threatens to make me unsound.

The patch is to deliberately make the new ellipsoid a touch *bigger* than the minimal one, by exactly enough slack to swallow the rounding error, while still keeping the per-step volume ratio strictly below 1. In the `Q`-matrix form, I can compute the exact central-cut axes `Δ`, multiply all semi-axes by a small cushion such as `2^{1/(2n²)}`, and then round the entries to a grid of accuracy about `δ = 2^{−8nL}`. In the `A = Q Qᵀ` form, the same cushion is a scalar outward factor on the shape matrix after the rank-one update. The minimum enclosing ellipsoid was tight; I don't need tight, I need *containing* and *shrinking*. The cushion is chosen larger than the possible inward motion from rounding the center and matrix entries, so the rounded ellipsoid still contains the exact half-ellipsoid. The exact shrink has enough margin that multiplying by this cushion leaves a volume ratio below one. So I round to control bit-length and I inflate to restore correctness; the two together keep the method both polynomial *and* sound. Choosing `p = O(nL)` bits of precision and matching the inflation to that grid, every entry of `x_k`, `A_k`, and the residual stays bounded — `‖x_k‖ ≤ 2^{O(L)}`, `‖A_k‖` controlled, `det A_k` bounded away from `0` so the ellipsoid never degenerates — and each step is `O(n²)` operations carried to `O(nL)`-bit precision. Across `M = O(n²L)` steps that's `O(n²(n²+m)L)` operations total, with `O(nm + n²)` numbers in memory each `O(nL)` bits. Polynomial in `L`. The decision procedure is built.

Step back and look at what the loop actually needed from the constraints. At no point did I use the *list* `A x ≤ b` as a whole, except to find one row violated by the current center. The same loop works if I only have a subroutine that, at the current center `x`, either certifies membership in the target body or returns a normal `a` such that every admissible `y` satisfies `aᵀ(y−x) ≤ 0`. That's the separation oracle in the form the ellipsoid update needs. For an explicit linear system the oracle is just "scan all rows and return the maximum residual"; for an implicit system the arithmetic of the localization loop is unchanged, and the total cost is the number of oracle calls plus the `O(n²)` matrix work per call.

And then optimization itself falls out of feasibility by binary search. To maximize `cᵀx` over `P`, I check nonemptiness of `P ∩ { cᵀx ≥ d }` and move `d` upward or downward. The separation oracle for this intersection is still simple: first ask whether `x` violates `P`; if it does, return that cut, and if it does not but `cᵀx < d`, return the violated objective cut with normal `−c`. Lemma 1 bounds every relevant point for a finite integer optimum by `‖x‖ ≤ 2^L`, so the objective lies in `[-2^L‖c‖, 2^L‖c‖]`. The same determinant bounds that gave the feasibility gap also give a rational separation between distinct possible vertex objective values, so enough feasibility decisions at that bit precision recover the optimum exactly. The `e^{−1/(2(n+1))}` shrink is slow and structure-blind, so I should not expect this to compete with boundary-walking in ordinary computations; its force is the worst-case guarantee.

I can now write the compact engine. The NumPy code mirrors the exact central-cut rank-one update; a production bounded-precision version wraps the update in the outward rounding and inflation I just worked through. The loop length is the `O(n²L)` horizon, and the only thing the loop ever asks of the problem is the separation oracle. In the explicit integer wrapper I accept a residual tolerance below `2^{−L}`; that is the relaxed body `P_ε`, and the infeasibility gap keeps it from accepting an empty original system.

```python
import math
import numpy as np

def as_integer_system(Arows, brhs):
    """Validate integer data for Arows x <= brhs."""
    Arows = np.asarray(Arows, dtype=float)
    brhs = np.asarray(brhs, dtype=float).reshape(-1)
    if Arows.ndim != 2 or brhs.shape[0] != Arows.shape[0]:
        raise ValueError("expected Arows with shape (m, n) and brhs with shape (m,)")
    if Arows.shape[0] == 0:
        raise ValueError("expected at least one inequality")
    if not np.all(np.isfinite(Arows)) or not np.all(np.isfinite(brhs)):
        raise ValueError("all coefficients must be finite")
    Aint = np.rint(Arows)
    bint = np.rint(brhs)
    if not np.array_equal(Arows, Aint) or not np.array_equal(brhs, bint):
        raise ValueError("this decision wrapper expects integer coefficients")
    return Aint.astype(float), bint.astype(float)

def encoding_length(Arows, brhs):
    """Binary encoding length for an integer system Arows x <= brhs."""
    Arows, brhs = as_integer_system(Arows, brhs)
    s = sum(math.log2(abs(int(a)) + 1) for a in np.ravel(Arows))
    s += sum(math.log2(abs(int(bi)) + 1) for bi in np.ravel(brhs))
    s += math.log2(max(1, Arows.shape[0] * Arows.shape[1]))
    return int(math.ceil(s)) + 1

def residual(Arows, brhs, x):
    """sigma(x) = max_i (A_i @ x - b_i)."""
    return float(np.max(np.asarray(Arows, dtype=float) @ np.asarray(x, dtype=float) - brhs))

def separation_for_system(Arows, brhs, tol=0.0):
    """Return a center-valid cut normal, or None inside the residual tolerance."""
    Arows, brhs = as_integer_system(Arows, brhs)

    def separate(x):
        x = np.asarray(x, dtype=float)
        slack = Arows @ x - brhs
        row = int(np.argmax(slack))
        if slack[row] <= tol:
            return None
        return Arows[row].copy()

    return separate

def decide_feasibility(separate, n, R, max_iters, inflate=1.0):
    """Central-cut ellipsoid feasibility from a separation oracle.

    The oracle returns None when the center is accepted; otherwise it returns a
    normal a such that every feasible y satisfies a @ (y - x) <= 0.
    """
    if n < 2:
        raise ValueError("the central-cut update below assumes n >= 2")
    x = np.zeros(n, dtype=float)
    A = (float(R) ** 2) * np.eye(n)
    scale = n**2 / (n**2 - 1)

    for _ in range(max_iters):
        a = separate(x)
        if a is None:
            return x

        a = np.asarray(a, dtype=float)
        Aa = A @ a
        q = float(a @ Aa)
        if q <= 0:
            return None

        b = Aa / np.sqrt(q)
        x = x - b / (n + 1)
        A = inflate * scale * (A - (2.0 / (n + 1)) * np.outer(b, b))
        A = 0.5 * (A + A.T)

    return None

def decide_linear_inequalities(Arows, brhs, max_iters=None, tol=None):
    """Decide integer LP feasibility for Arows x <= brhs.

    With the default tolerance, a returned center may be epsilon-feasible
    rather than exactly feasible; the integer residual gap makes the status
    exact in the bounded-precision decision model.
    """
    Arows, brhs = as_integer_system(Arows, brhs)
    n = Arows.shape[1]
    L = encoding_length(Arows, brhs)
    R = 2.0**L
    max_iters = 16 * n * n * L if max_iters is None else max_iters
    tol = 0.5 * 2.0 ** (-L) if tol is None else tol
    sep = separation_for_system(Arows, brhs, tol=tol)
    x = decide_feasibility(sep, n, R, max_iters)
    return ("feasible", x) if x is not None else ("infeasible", None)

def maximize_linear(base_separate, c, n, L, tol=None):
    """Maximize c.x over P by bisection, using only centered separation for P."""
    c = np.asarray(c, dtype=float)
    R = 2.0**L
    max_iters = 16 * n * n * L
    tol = 2.0 ** (-L) if tol is None else tol
    bound = R * float(np.linalg.norm(c))
    if bound == 0.0:
        return decide_feasibility(base_separate, n, R, max_iters), 0.0
    lo, hi = -bound, bound
    best = None

    def with_objective_floor(d):
        def separate(x):
            cut = base_separate(x)
            if cut is not None:
                return cut
            if float(c @ x) < d:
                return -c
            return None

        return separate

    while hi - lo > tol:
        d = 0.5 * (lo + hi)
        x = decide_feasibility(with_objective_floor(d), n, R, max_iters)
        if x is None:
            hi = d
        else:
            lo, best = d, x
    return best, lo
```

The chain, end to end: integer data hands me a bounded search ball (any solution sits within radius `2^L`) and a residual gap (infeasible systems are violated by at least `2^{−L}`); I search a relaxed body with `ε < 2^{−L}` so feasible systems have positive volume while infeasible systems remain empty; I trap that target in an ellipsoid because, unlike a polyhedral localization set, an ellipsoid keeps a fixed-size description and a dimension-only shrink rate; I query the center, and a violated constraint slices the ellipsoid through its center, putting the target body in a half-ellipsoid; I re-enclose that half in the minimum-volume ellipsoid, whose center-and-matrix I get by transporting the canonical half-ball formula through the affine map, yielding the rank-one PSD update and a volume drop of `e^{−1/(2(n+1))}` every step; the bounded start volume `2^{O(nL)}` and the relaxed-body volume floor `2^{−O(nL)}` cap the run at `O(n²L)` steps; rounding to `O(nL)` bits keeps the numbers short and a deliberate inflation of the update restores the containment the rounding would break — leaving a polynomial-time, exact feasibility decision, and LP optimization follows by feasibility bisection on `cᵀx ≥ d`.
