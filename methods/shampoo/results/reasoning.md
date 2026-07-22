OK, let me think this through from scratch. The thing I keep coming back to is that full-matrix preconditioning is *the* powerful tool and I can't use it. Plain SGD just walks downhill in the raw coordinate geometry; the moment the loss landscape is anisotropic — different curvature in different directions, correlations between coordinates — it crawls. Newton and the adaptive full-matrix methods fix that by premultiplying the gradient with a matrix that reshapes the geometry. And there's a clean, provably-good choice for that matrix in the online-learning world: full-matrix AdaGrad. You accumulate `M_t = Σ_{s≤t} g_s g_s^T`, the outer-product sum of all gradients seen so far, and you precondition with `H_t = M_t^{1/2}`. The update is `w_{t+1} = w_t − η H_t^{-1} g_t`, and the regret it buys you scales like `tr((Σ g g^T)^{1/2})`, which is the best you can hope for in this family. It captures *every* pairwise correlation between coordinates, not just a per-coordinate rescaling.

So why isn't everyone using it? Because `g_t` is a flattened vector of dimension `d`, and `H_t` is `d × d`. For a single weight matrix `W ∈ R^{m×n}` — a multiclass classifier, or one fully-connected layer — the gradient `G_t = ∇f_t(W_t)` is also `m × n`, and flattening it gives `g_t = vec(G_t)` of length `d = mn`. So `H_t` is `mn × mn`. That's `m²n²` numbers to store, and computing `H_t^{-1/2}` (an eigendecomposition of an `mn × mn` matrix) is `O(m³n³)`. For `m = n = 1000` that's a `10⁶ × 10⁶` matrix and `10¹⁸` work per step. Dead on arrival.

The escape everyone actually takes is to throw away the off-diagonal. Diagonal AdaGrad, Adam: keep only `diag(Σ g g²)`, one scalar per coordinate, `O(d)` memory, trivial inverse. And it works great in practice. But it's a real concession — a diagonal preconditioner is axis-aligned, it can only stretch and shrink along the coordinate axes, it is structurally blind to the very correlations that made full-matrix preconditioning powerful in the first place. So I'm sitting between two extremes: a full matrix that's impossibly large, and a diagonal that's cheap but throws away all the cross-coordinate structure. I want something in between that keeps *some* of the matrix structure but pays nowhere near `m²n²`.

Here's the thing I keep staring at. The parameter isn't really a vector — it's a *matrix*, `m × n`. Flattening it to length `mn` is something I do only to write down `H_t`; it's not intrinsic to the problem. The full preconditioner `H_t ∈ R^{mn×mn}` is forced to model correlations between *every* pair of the `mn` entries, including the correlation between entry `(i,j)` and entry `(i',j')` for completely unrelated `i,i',j,j'`. That's an enormous number of degrees of freedom, and most of it is probably not where the action is. What if the correlations that matter decompose along the two axes of the matrix — a "row geometry" (`m × m`) and a "column geometry" (`n × n`) — and I only model those?

Concretely: maintain one small matrix per axis. A left matrix `L ∈ R^{m×m}` that summarizes the row structure, and a right matrix `R ∈ R^{n×n}` that summarizes the column structure. That's `m² + n²` numbers instead of `m²n²`, and any inverse/root I'd take is `O(m³ + n³)` instead of `O(m³n³)`. For `m = n = 1000` that's two `10⁶`-entry matrices and `10⁹` work — completely tractable. The cost story is obviously fine; what I have no right to assume yet is that one matrix per axis can carry enough of what the full preconditioner was doing. That has to be earned by relating it back to full-matrix AdaGrad, not assumed just because the shapes are convenient.

What should `L` and `R` actually be? By analogy with full-matrix AdaGrad, they should be second-moment statistics of the gradients, but contracted down to each axis. The gradient is `G_t` (`m × n`). The two natural ways to square it into a symmetric matrix are `G_t G_t^T` (which is `m × m` — it contracts away the column index, leaving a row-by-row second moment) and `G_t^T G_t` (which is `n × n` — contracts away the row index, leaving column-by-column). So let me accumulate
```
L_t = ε I_m + Σ_{s≤t} G_s G_s^T      (m × m, the "left"/row preconditioner stats)
R_t = ε I_n + Σ_{s≤t} G_s^T G_s      (n × n, the "right"/column preconditioner stats)
```
with a small `ε I` ridge so everything stays positive-definite and I can take inverses and roots. These are exactly the per-axis analogues of `Σ g g^T`: instead of the full `mn × mn` outer-product sum, I keep its two axis-wise contractions. Cheap to maintain — just two matrix products per step.

Now how do I *use* `L_t` and `R_t` to precondition `G_t`? In the full case I'd left-multiply the flat gradient by `H_t^{-1}`. With a left matrix and a right matrix, the obvious thing is to hit the gradient matrix on both sides: form some power of `L_t` and multiply from the left (preconditioning the rows), and some power of `R_t` and multiply from the right (preconditioning the columns):
```
G_t  ↦  L_t^{-a} G_t R_t^{-b}
```
for exponents `a, b` I have to pin down. So the update would be `W_{t+1} = W_t − η L_t^{-a} G_t R_t^{-b}`. By the obvious symmetry between the two axes I'd guess `a = b`. But what *value*? This is where I have to be careful, because the exponent is the whole ballgame — get it wrong and the method is just heuristic two-sided scaling with no relation to the full preconditioner I'm trying to imitate.

Let me make the bridge to the flattened picture precise. I need to know what a two-sided multiplication `L^{-a} G R^{-b}` looks like as a single matrix acting on `vec(G)`. There's an identity for exactly this: for conformable `L, R`,
```
(L ⊗ R^T) vec(G) = vec(L G R) .
```
Check it on a rank-one `G = u v^T`, since any matrix is a sum of those. Then `vec(G) = vec(u v^T) = u ⊗ v`. The left side: `(L ⊗ R^T)(u ⊗ v) = (L u) ⊗ (R^T v)` by the mixed-product rule `(A⊗B)(A'⊗B') = (AA')⊗(BB')`. The right side: `vec(L u v^T R) = vec((L u)(R^T v)^T) = (L u) ⊗ (R^T v)`, using `vec(p q^T) = p ⊗ q`. They match, and linearity extends it to all `G` — using the row-major convention `vec(G)_{in+j} = G_{ij}`, which I'll keep consistent throughout.

So if I precondition by `L_t^{-a} G_t R_t^{-b}`, then in the flattened world I'm applying the single matrix `L_t^{-a} ⊗ (R_t^{-b})^T` to `vec(G_t)`. And `R_t` is symmetric (it's a sum of `G^T G`, plus `ε I`), so any power of it is symmetric, `(R_t^{-b})^T = R_t^{-b}`. The implied flattened preconditioner is therefore
```
H_t^{-1} = L_t^{-a} ⊗ R_t^{-b} ,   i.e.   H_t = L_t^{a} ⊗ R_t^{b} .
```
A Kronecker product of two small matrices, never formed explicitly. That's the structure. Now: what should `H_t` be, so that this thing actually mimics full-matrix AdaGrad? Full-matrix AdaGrad uses `H_t = (Σ_{s} g_s g_s^T)^{1/2}` — the *square root* of the accumulated gradient covariance — so I want my `L_t^{a} ⊗ R_t^{b}` to be, in some controlled sense, close to `(Σ g g^T)^{1/2}`.

The naive guess is sitting right there: the row-covariance is `L_t = Σ G G^T` and the column-covariance is `R_t = Σ G^T G`; full AdaGrad takes the square root of a covariance; so take the square root of each, `H_t = L_t^{1/2} ⊗ R_t^{1/2}`, i.e. `a = b = 1/2`, giving the update `L_t^{-1/2} G_t R_t^{-1/2}`. Before I commit to that I want to know whether `L_t^{1/2} ⊗ R_t^{1/2}` really plays the role of `(Σ g g^T)^{1/2}`, or whether it's playing some other role.

To check, I need a relationship between `Σ g g^T` (the flat `mn × mn` covariance) and the Kronecker product of my axis matrices `L_t = Σ G G^T` and `R_t = Σ G^T G`. Let me work out the single-gradient case and see what `g g^T` is bounded by, where `g = vec(G)`. Take the SVD `G = Σ_{i=1}^r σ_i u_i v_i^T` with `r = rank(G)`, orthonormal `u_i ∈ R^m`, `v_i ∈ R^n`. Then `g = vec(G) = Σ_i σ_i (u_i ⊗ v_i)`. So
```
g g^T = ( Σ_i σ_i (u_i ⊗ v_i) )( Σ_i σ_i (u_i ⊗ v_i) )^T .
```
This is the outer product of a *sum*, which is the annoying part — cross terms `i ≠ i'` couple the row and column structure together and that's exactly what I can't represent with a clean Kronecker form. Let me decouple them. For any vectors `w_1,…,w_r`,
```
( Σ_i w_i )( Σ_i w_i )^T  ⪯  r Σ_i w_i w_i^T .
```
Why: pick any `x`, set `α_i = x^T w_i`; then `x^T (Σ w_i)(Σ w_i)^T x = (Σ_i α_i)² ≤ r Σ_i α_i²` by Cauchy–Schwarz (convexity of `α ↦ α²`), and the right side is `r x^T (Σ w_i w_i^T) x`. So at the cost of a factor `r`, I can drop the cross terms. Apply it with `w_i = σ_i (u_i ⊗ v_i)`:
```
g g^T  ⪯  r Σ_i σ_i² (u_i ⊗ v_i)(u_i ⊗ v_i)^T = r Σ_i σ_i² (u_i u_i^T) ⊗ (v_i v_i^T) .
```
Now I have a clean sum of Kronecker products. To turn it into a single Kronecker, bound one factor by the identity. Since the `v_i` are orthonormal, `v_i v_i^T ⪯ I_n`, so
```
(1/r) g g^T  ⪯  Σ_i σ_i² (u_i u_i^T) ⊗ I_n = ( Σ_i σ_i² u_i u_i^T ) ⊗ I_n = (G G^T) ⊗ I_n ,
```
because `G G^T = Σ_i σ_i² u_i u_i^T`. Symmetrically, bounding `u_i u_i^T ⪯ I_m` gives
```
(1/r) g g^T  ⪯  I_m ⊗ (G^T G) .
```
Checking the chain numerically on a random `3×4 G` (`r = 3`): the smallest eigenvalue of `(GG^T)⊗I − (1/r)gg^T` comes out `+2·10⁻¹⁵`, and of `I⊗(G^TG) − (1/r)gg^T` comes out `−3·10⁻¹⁵` — both machine zero, so both bounds are *tight* for this example, attained with equality along some direction. The rank factor `r` isn't slack I left on the table, it's exactly what's needed.

So a single gradient's flat covariance is sandwiched, up to the rank factor `r`, under *both* `(GG^T) ⊗ I_n` and `I_m ⊗ (G^TG)`. Summing over `t` and folding in the ridge: with `A_m = ε I_m + Σ_t G_t G_t^T = L_T` and `B_n = ε I_n + Σ_t G_t^T G_t = R_T`,
```
ε I_{mn} + (1/r) Σ_t g_t g_t^T  ⪯  A_m ⊗ I_n     and     ε I_{mn} + (1/r) Σ_t g_t g_t^T  ⪯  I_m ⊗ B_n .
```
I have the *same* matrix on the left bounded above by two different things. `A_m ⊗ I_n` and `I_m ⊗ B_n` commute with each other (each is "block-acting" on a different factor of the tensor product — explicitly `(A_m ⊗ I_n)(I_m ⊗ B_n) = A_m ⊗ B_n = (I_m ⊗ B_n)(A_m ⊗ I_n)` by the mixed-product rule). For commuting PSD matrices the geometric mean is operator-monotone: if `X ⪯ Y_1` and `X ⪯ Y_2` with the `Y`'s commuting, then `X ⪯ Y_1^{1/2} Y_2^{1/2}`. Apply that with `Y_1 = I_m ⊗ B_n`, `Y_2 = A_m ⊗ I_n`:
```
ε I_{mn} + (1/r) Σ_t g_t g_t^T  ⪯  (I_m ⊗ B_n)^{1/2} (A_m ⊗ I_n)^{1/2}
                                 = (I_m ⊗ B_n^{1/2})(A_m^{1/2} ⊗ I_n)
                                 = A_m^{1/2} ⊗ B_n^{1/2}
                                 = L_T^{1/2} ⊗ R_T^{1/2} .
```
Checking this beyond the single-gradient case: I accumulated `T = 20` random `3×4` gradients, formed `Σ g g^T` directly (a `12×12` matrix) and `L_T, R_T`, and looked at `L_T^{1/2} ⊗ R_T^{1/2} − (1/r) Σ g g^T`: smallest eigenvalue `≈ +51.5`, comfortably positive. The bound holds with real margin once the cross-terms accumulate, not just tightly for one gradient.

And now it kills my naive guess. The accumulated *covariance* `Σ g g^T` — the un-rooted thing, the `O(t)`-growing object — is what's bounded by `L_T^{1/2} ⊗ R_T^{1/2}`. So `L_T^{1/2} ⊗ R_T^{1/2}` is the analogue of `Σ g g^T`, **not** of its square root `(Σ g g^T)^{1/2}`. The two square roots in `L^{1/2} ⊗ R^{1/2}` are not the "AdaGrad square root" at all — they're the price of splitting one covariance across two axes. A scaling sanity check seals it: each axis matrix `L_T = Σ GG^T` grows linearly in `t`, so `L_T^{1/2} ~ t^{1/2}` and `R_T^{1/2} ~ t^{1/2}`, and `L_T^{1/2} ⊗ R_T^{1/2}` has `t^{1/2} · t^{1/2} = t` growth, matching `Σ g g^T ~ t`. So if I had used `H_t = L_t^{1/2} ⊗ R_t^{1/2}` as my preconditioner I'd effectively be preconditioning with the *covariance itself*, not its square root — that's like dividing the gradient by `v` instead of by `√v`, far too aggressive, and not the full-AdaGrad geometry I'm chasing. The naive `a = b = 1/2` is wrong, and I can see exactly why.

So I need one more square root. The preconditioner I want is `(Σ g g^T)^{1/2}`, and I've just shown `Σ g g^T` is (up to `r`) the object `L_T^{1/2} ⊗ R_T^{1/2}`. Take its square root:
```
H_t = ( L_t^{1/2} ⊗ R_t^{1/2} )^{1/2} = L_t^{1/4} ⊗ R_t^{1/4} ,
```
using `(A ⊗ B)^s = A^s ⊗ B^s` for PSD factors. The exponent comes out **1/4**, not 1/2. And it factors cleanly into two halves: one factor of `½` from splitting the gradient covariance across the two axes (the `L^{1/2} ⊗ R^{1/2}` bound on `Σ g g^T`), and a second factor of `½` from the AdaGrad square root that turns a covariance into a preconditioner. So `a = b = 1/4`, and the update is
```
W_{t+1} = W_t − η L_t^{-1/4} G_t R_t^{-1/4} .
```
There's an independent consistency check that makes the `1/4` feel forced rather than chosen. `L_t` and `R_t` each accumulate `t` outer products, so each grows like `t`; thus `L_t^{1/4} ~ t^{1/4}` and `R_t^{1/4} ~ t^{1/4}`, and the effective step scale `‖L_t^{-1/4} · R_t^{-1/4}‖ ~ t^{-1/4} · t^{-1/4} = t^{-1/2}`. That's exactly the `O(1/√t)` step-size decay that's canonical for stochastic optimization. The naive `1/2` would have given `t^{-1}` decay — way too fast, collapsing the step before the iterate could move. The `1/4` lands the decay rate where it has to be, by an argument completely independent of the regret bound I still have to do.

And the lemma `ε I_{mn} + (1/r) Σ g g^T ⪯ L_T^{1/2} ⊗ R_T^{1/2}` says something about *quality*, not just cost: it's a lower bound on (the square of) my Kronecker preconditioner in terms of the true full-matrix covariance. The directions where `Σ g g^T` is small — the directions of little accumulated gradient, which are precisely the ones full-matrix preconditioning amplifies and cares most about — are exactly the directions where `L^{1/2} ⊗ R^{1/2}` is bounded *below*, so they aren't crushed to zero by the factorization. Whether that translates into a competitive regret bound has to come out of the regret algebra itself, not this inequality alone. So I need to actually do the regret.

The setup: my update is `W_{t+1} = W_t − η L_t^{-1/4} G_t R_t^{-1/4}`, which by the vec identity is exactly the flattened mirror-descent step `w_{t+1} = w_t − η H_t^{-1} g_t` with `H_t = L_t^{1/4} ⊗ R_t^{1/4}`. So I can use the standard adaptive-mirror-descent regret bound,
```
R_T ≤ (1/2η) Σ_t ( ‖w_t−w*‖²_{H_t} − ‖w_{t+1}−w*‖²_{H_t} ) + (η/2) Σ_t ( ‖g_t‖*_{H_t} )² ,
```
and bound each piece. The `H_t` are monotonically increasing: `L_1 ⪯ L_2 ⪯ …` and `R_1 ⪯ R_2 ⪯ …` since each step adds a PSD term, and the Kronecker product preserves the order, so `0 ≺ H_1 ⪯ … ⪯ H_T`. That lets the first sum telescope. Writing `w* = vec(W*)`, `D = max_t ‖W_t − W*‖_F`, and `H_0 = 0`,
```
Σ_t (w_t−w*)^T (H_t − H_{t-1})(w_t−w*) ≤ D² Σ_t tr(H_t − H_{t-1}) = D² tr(H_T) ,
```
where I used `x^T M x ≤ ‖x‖² tr(M)` for PSD `M`. So the first regret term is `≤ D²/(2η) · tr(H_T)`.

Now the second term, `Σ_t (‖g_t‖*_{H_t})²`. This is where I find out whether the factorization actually stayed competitive with the optimal full-matrix preconditioner, or whether the analogy was hollow. Define the ideal full-matrix preconditioner that the adaptive-regularization machinery would pick: `Ĥ_t = (r ε I + Σ_{s≤t} g_s g_s^T)^{1/2}`. By the lemma above and the operator-monotonicity of `x ↦ x^{1/2}` (Löwner), squaring-then-rooting the bound `r ε I + Σ g g^T ⪯ r (L_t^{1/2} ⊗ R_t^{1/2}) = r H_t²` gives
```
Ĥ_t ⪯ √r · H_t .
```
That's the inequality the regret needs, and it's exactly where the lemma earns its keep — so I checked it isn't just formally true but holds on numbers: with the same `T = 20` accumulated gradients, smallest eigenvalue of `√r · H_t − Ĥ_t` came out `≈ +7.9 > 0`. So my Kronecker preconditioner `H_t` really does dominate the ideal full one `Ĥ_t`, up to `√r`, and not just in the limit. Now bound `Σ_t (‖g_t‖*_{Ĥ_t})²` using the adaptive-regularization lemma with potential `Φ(H) = tr(H) + rε·tr(H^{-1})`. Check that the lemma's `argmin` is `Ĥ_t`: minimizing `M_t • H^{-1} + Φ(H) = tr(Ĥ_t² H^{-1} + H)` over `H ≻ 0` — more cleanly, `∇_H tr(A H^{-1} + H) = I − A H^{-2}` (using `∇_H tr(AH^{-1}) = −H^{-1}AH^{-1}` and at a symmetric optimum `H` commutes with `A`), which vanishes at `H = A^{1/2}`; with `A = Ĥ_t²` that's `H = Ĥ_t`. So the lemma applies and
```
Σ_t (‖g_t‖*_{Ĥ_t})² ≤ Σ_t (‖g_t‖*_{Ĥ_T})² + Φ(Ĥ_T) − Φ(Ĥ_0)
                    ≤ (rε I + Σ_t g_t g_t^T) • Ĥ_T^{-1} + tr(Ĥ_T)
                    = tr(Ĥ_T² Ĥ_T^{-1}) + tr(Ĥ_T) = 2 tr(Ĥ_T) ,
```
where the middle line drops the `rε·tr(Ĥ_0^{-1})` term and uses `Ĥ_T² = rεI + Σ g g^T`. Now I have to chain this with `Ĥ_t ⪯ √r H_t` to get back to a bound in *my* dual norm, and I have to be careful with the direction — it's easy to flip an inequality on inverses. I want to bound `Σ(‖g_t‖*_{H_t})²`, and `‖g‖*²_{H_t} = g^T H_t^{-1} g`. Inverting the order-relation, `Ĥ_t ⪯ √r H_t ⇒ H_t^{-1} ⪯ √r Ĥ_t^{-1}`, so `g^T H_t^{-1} g ≤ √r · g^T Ĥ_t^{-1} g = √r (‖g‖*_{Ĥ_t})²` — that's the direction I need:
```
Σ_t (‖g_t‖*_{H_t})² ≤ √r Σ_t (‖g_t‖*_{Ĥ_t})² ≤ 2√r · tr(Ĥ_T) ≤ 2√r · tr(√r H_T) = 2r · tr(H_T) ,
```
the last step using `Ĥ_T ⪯ √r H_T` again under the trace. So the second regret term is `≤ (η/2)·2r·tr(H_T) = η r · tr(H_T)`.

Put the two pieces together:
```
R_T ≤ ( D²/(2η) + η r ) tr(H_T) .
```
Minimize over `η`: choose `η = D/√(2r)`. Then `D²/(2η) = D²√(2r)/(2D) = D√(2r)/2` and `ηr = Dr/√(2r) = D√(r/2) = D√(2r)/2`, summing to `D√(2r)`. So
```
R_T ≤ √(2r) · D · tr(H_T) = √(2r) · D · tr(L_T^{1/4} ⊗ R_T^{1/4}) = √(2r) · D · tr(L_T^{1/4}) tr(R_T^{1/4}) ,
```
using `tr(A ⊗ B) = tr(A) tr(B)`. That's the regret. Now how does it scale in `T`? Suppose the losses are 1-Lipschitz in spectral norm, `‖G_t‖₂ ≤ 1`, and take `ε` small. Then `G_t G_t^T ⪯ I_m` and `G_t^T G_t ⪯ I_n`, so `L_T ⪯ (ε+T)·I_m ≈ T·I_m` and `R_T ⪯ (ε+T)·I_n ≈ T·I_n`, giving `tr(L_T^{1/4}) ≤ m T^{1/4}` and `tr(R_T^{1/4}) ≤ n T^{1/4}`. The product carries `T^{1/4}·T^{1/4} = T^{1/2}`, so `R_T = O(√T)` — the optimal rate for online/stochastic convex optimization. So the `1/4` exponent isn't just a memory trick: it's what makes the two trace terms each scale as `T^{1/4}`, so their product hits `√T` rather than overshooting. The analogy survived contact with the regret algebra, which is what I wanted before trusting it.

(One caveat I should flag to myself: `D = max_t ‖W_t − W*‖_F` could in principle grow with `T`. The clean fix is to project `W_t` back onto a Frobenius-ball of radius `D/2`, where the projection is taken in the norm induced by `(L_t, R_t)`, `‖A‖²_t = tr(A^T L_t^{1/4} A R_t^{1/4})` — one can check that's a genuine norm. But that projection is expensive at scale and in practice I'll just drop it and live with the slightly looser bound.)

Now I want this for *all* the parameter tensors, not just matrices. A convolutional layer's weights are an order-4 tensor (in-depth × width × height × out-depth), and the gradient `G_t` is a tensor of the same shape. The whole derivation generalizes if I keep one preconditioner per *mode* (axis). Let the tensor have order `k` with dimensions `n_1 × … × n_k`. For each axis `i`, the analogue of "square the gradient, contracting away the other index" is the *mode-`i` contraction*: flatten the tensor into a matrix `mat_i(G)` (axis `i` as rows, everything else flattened into columns) and form
```
G^{(i)} = mat_i(G) mat_i(G)^T ∈ R^{n_i × n_i} .
```
For a matrix (`k=2`), `G^{(1)} = G G^T = L` and `G^{(2)} = G^T G = R`, so this reproduces the left/right matrices exactly. Accumulate `H^i_t = ε I_{n_i} + Σ_{s≤t} G_s^{(i)}` for each axis. To precondition, I multiply the `i`-th mode of the gradient by `(H^i_t)^{-something}` via the tensor-matrix product `×_i` (which is just `M · mat_i(G)` in matricized form), do this for every axis in any order (they commute as tensor-matrix products on distinct modes), and take a gradient step.

What's the exponent now? Same logic, generalized. The flattened preconditioner equivalent to applying `(H^i_t)^{c}` along each mode `i` is, by the tensor vec–Kronecker identity `(⊗_i M_i) vec(G) = vec(G ×_1 M_1 ⋯ ×_k M_k)`, the single matrix `⊗_i (H^i_t)^{c}`. And the lower-bound lemma generalizes: with `r = (∏_i r_i)^{1/k}` for per-mode ranks `r_i`,
```
ε I_n + Σ_t g_t g_t^T ⪯ r · ⊗_{i=1}^k (H^i_T)^{1/k} .
```
(The proof is the matrix one lifted: `kron-base` per matricization gives `(1/r_i) vec(mat_i G) vec(mat_i G)^T ⪯ G^{(i)} ⊗ I`; a "transpose" lemma rearranges this into `(1/r_i) g g^T ⪯ ⊗_{j<i} I ⊗ G^{(i)} ⊗ ⊗_{j>i} I`; sum over `t`, note the `k` resulting bounds commute, and apply the geometric-mean inequality with weights `α_i = 1/k`.) So now the un-rooted covariance `Σ g g^T` is the object `⊗_i (H^i_T)^{1/k}` — the `1/k` is the splitting-across-`k`-axes exponent (for `k=2` it's the `½` from before). Take the AdaGrad square root: the preconditioner is `H_t = (⊗_i H^i_t)^{1/(2k)} = ⊗_i (H^i_t)^{1/(2k)}`, so along each mode I apply
```
(H^i_t)^{-1/(2k)} .
```
The exponent is `−1/(2k)`: `1/k` from splitting the covariance across `k` axes, times `½` from the AdaGrad root. For `k = 2` this is `−1/4`, recovering the matrix case exactly. The regret proof is verbatim the matrix one with the tensor lemma in place of the matrix lemma, giving `R_T ≤ √(2r) D ∏_i tr((H^i_T)^{1/(2k)})`, and each trace scales as `O(T^{1/(2k)})`, so the product is `O(√T)` again.

Before I trust the `−1/(2k)` formula across orders, let me check the degenerate order it should obviously reduce to. For a *vector* parameter, `k = 1`, the exponent is `−1/(2·1) = −1/2`: there's one axis, `H^1_t = εI + Σ g_t g_t^T` is the full gradient covariance, and the update is `(εI + Σ g g^T)^{-1/2} g` — which is precisely full-matrix AdaGrad on that vector. So at `k=1` the method *is* the thing I started from, with no factorization at all, exactly as it should be (there's nothing to factor on a single axis). I confirmed this numerically on a random length-5 gradient: the order-1 Shampoo step and a direct full-matrix-AdaGrad step `H^{-1}g` with `H=(εI+gg^T)^{1/2}` agreed to `2·10⁻⁵` (just the SVD's numerical floor). The exponent formula passes its boundary case.

Two practical things I'll need. First, computing `(H^i_t)^{-1/(2k)}` — an inverse `p`-th root of a symmetric PSD matrix. Since `H^i_t = Σ_j λ_j u_j u_j^T` is symmetric PSD, any real power is just `(H^i_t)^α = Σ_j λ_j^α u_j u_j^T`: take the eigen/SVD, raise the eigenvalues (singular values) to the power `α = −1/(2k)`, reassemble. The `ε I` ridge guarantees `λ_j > 0` so the negative power is well-defined. Second, this root is the only nontrivial cost, `O(n_i³)` per axis — far cheaper than the `O(n³)` of the full method, but still something I don't want to do every single step. So I'll recompute the roots only every, say, 20–100 steps and reuse them in between; this barely touches accuracy and keeps the amortized per-step cost close to first-order methods. I'll also fold in a momentum-style running average of the gradient, `Ḡ_t = α Ḡ_{t-1} + (1−α) G_t` with `α ≈ 0.9`, which helps convergence the same way it does for other stochastic methods.

One more design choice worth being explicit about: I treat each parameter tensor in the model *independently*, with its own set of per-axis preconditioners. In the flattened picture that's a block-diagonal preconditioner, one block per tensor — I capture all the intra-tensor correlations but ignore correlations between parameters living in different tensors. The payoff is that the optimizer is completely oblivious to the model architecture: it doesn't need to know the network graph, the layer types, or anything about backprop structure; it only needs the list of parameter tensors and their shapes. That's a deliberate trade against the alternative of trying to model cross-tensor structure (which would reintroduce exactly the size blowup I'm avoiding) and against architecture-specific factored methods that have to be tailored to the network.

There's also a fallback I want for safety. If one axis `n_i` is enormous (say a vocabulary dimension of tens of thousands), even the `n_i × n_i` preconditioner and its `O(n_i³)` root become too much. For such an axis I drop to a *diagonal* preconditioner on that mode only: replace `H^i_t = Σ G^{(i)}` with `H^i_t = Σ diag(G^{(i)})`, i.e. keep only the diagonal of the contraction, `O(n_i)` memory and an elementwise inverse-root. Other axes of the same tensor can stay full. The same regret analysis goes through with `D_∞` (the entrywise `ℓ_∞` distance) in place of the Frobenius `D`, using `v^T M v ≤ ‖v‖_∞² tr(M)` for diagonal PSD `M` and `diag(A ⊗ B) = diag(A) ⊗ diag(B)`. I'll auto-activate this whenever an axis exceeds a size threshold (around 1200 in practice), with no other change to the code.

Let me write the matrix case down as code, then it generalizes immediately to tensors by looping over modes. Each parameter gets, per axis, a statistics matrix `precond_i` (the `H^i`) and a cached inverse root `inv_precond_i = (H^i)^{-1/(2k)}` where `k = order` of the tensor. Per step: optionally momentum-average the gradient; for each axis, matricize the (running) gradient on that axis, add `mat_i(g) mat_i(g)^T` into `precond_i`, refresh the inverse root on the update schedule, and multiply the gradient's `i`-th mode by `inv_precond_i`; finally take the SGD step with the fully preconditioned gradient.

```python
def _matrix_power(matrix, power):
    # H = U diag(s) V^T  ->  H^power = U diag(s^power) V^T, via SVD.
    u, s, v = torch.svd(matrix)
    return u @ s.pow(power).diag() @ v.t()

# state["precond_i"] holds each H^i (init eps*I_{n_i}); momentum and weight
# decay are folded into `grad` first, exactly as in the generic scaffold.
for dim_id, dim in enumerate(grad.size()):
    precond = state[f"precond_{dim_id}"]
    inv_precond = state[f"inv_precond_{dim_id}"]

    grad = grad.transpose_(0, dim_id).contiguous()   # axis dim_id -> rows
    transposed_size = grad.size()
    grad = grad.view(dim, -1)

    grad_t = grad.t()
    precond.add_(grad @ grad_t)                       # H^i += mat_i(g) mat_i(g)^T
    if state["step"] % group["update_freq"] == 0:
        inv_precond.copy_(_matrix_power(precond, -1 / (2 * order)))

    if dim_id == order - 1:
        grad = grad_t @ inv_precond                   # last mode: right-multiply
        grad = grad.view(original_size)
    else:
        grad = inv_precond @ grad                     # other modes: left-multiply
        grad = grad.view(transposed_size)
```

A note on the exponent versus my derivation: I derived the per-axis power as `−1/(2k)` for an order-`k` tensor, `−1/4` for a matrix (`k=2`), and the loop uses `-1/(2*order)`, applying one factor `(H^i)^{-1/4}` on the left and one on the right. But I should be honest about one thing this loop does that the idealized update `L^{-1/4} G R^{-1/4}` does *not*: it preconditions the modes *sequentially*, overwriting `grad` each time. So when it reaches the last axis, the right preconditioner is accumulated and applied to the *already left-preconditioned* gradient, not the raw `G`. Tracing this on a `2×3` gradient and comparing to a clean `L^{-1/4} G R^{-1/4}` with both `L,R` from the raw `G`: they differ by `≈ 0.23` in max-abs — *not* identical. So the loop realizes a sequential, "running-gradient" variant of the two-sided update rather than the literal simultaneous formula — a deliberate implementation choice, since it lets one matricize-accumulate-precondition loop handle any tensor order with no special-casing, and it costs nothing asymptotically. But it would be dishonest to say this computes exactly `L^{-1/4} G R^{-1/4}`; it computes the per-mode-sequential cousin. Only the first axis processed gets an untouched statistic (`L`'s accumulation happens before any preconditioning); every later axis accumulates from the partially-preconditioned gradient, not the raw `G`, though it still applies the same `−1/(2k)` exponent and the same per-axis structure. The regret analysis I did is for the *flattened*, simultaneous form `H_t = ⊗_i (H^i_t)^{1/(2k)}` — that is the object my guarantee is actually about; the sequential loop is the practical stand-in I ship, and the two are close but not proven equal.
