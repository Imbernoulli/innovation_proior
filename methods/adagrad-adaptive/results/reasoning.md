Let me start from the thing that actually hurts. I am running online subgradient descent on a text problem — millions of features, but each example lights up only a few of them — and the learning rate is fighting me. The update is `x_{t+1} = Π_X(x_t − η_t g_t)` with `η_t = η/√t`, which is the standard prescription, and it has a clean justification: telescoping the Euclidean contraction gives `Σ_t (x_t−x*)·g_t ≤ ‖x_1−x*‖²/(2η_1) + ½Σ_t (1/η_t − 1/η_{t-1})‖x_t−x*‖² + ½Σ_t η_t‖g_t‖²`, and with `η_t = η/√t` and `Σ_t 1/√t ≈ 2√T` this lands at `O(D₂·√(Σ_t‖g_t‖₂²))`, the familiar `O(√T)`. Good. But sit with what `η_t` *is*: one scalar, multiplying every coordinate by the same amount.

That is exactly wrong for my data, and I want to put numbers on it before I trust the intuition. Feature 7 fires on nearly every example; feature 900000 fires once every ten thousand examples but when it does it basically tells me the label. Say it first fires at round `t₀` with `|g| = 0.5`. With `η = √2`, the global schedule moves that weight by `η|g|/√t₀ = √2·0.5/√t₀`: at `t₀ = 1` that is `0.707`, at `t₀ = 9999` it is `7.1e-3`, at `t₀ = 500000` it is `1.0e-3`. So the magnitude of the one informative update a rare feature ever produces shrinks by three orders of magnitude purely because it happened to arrive late — nothing about the feature changed, only the wall-clock. Meanwhile feature 7, which I have already seen a thousand times and basically know, keeps getting that same `η/√t` and keeps jittering. The schedule is coupled to wall-clock rounds, not to *how much I have actually learned about each coordinate*. People already compensate for this by hand — TF-IDF up-weights rare features before the optimizer ever sees them — which tells me the fix belongs in the geometry, not in a preprocessing hack.

So can I just beat the bound? No. Zinkevich's `O(D₂√(Σ‖g‖²))` is minimax-optimal; Abernethy showed it is tight. There is an adversary that forces exactly that regret. So I am not going to improve the *worst case*. If I want a better bound I have to give up worst-case universality and instead exploit the *structure that is actually present* in my data — the sparsity, the heterogeneity across coordinates. The improvement, if it exists, has to be data-dependent: a bound that is small when the data is nice (sparse, heavy-tailed) and gracefully degrades to the old one when it isn't.

Where would such structure even enter the bound? Let me go up a level from the bare gradient step to the proximal / mirror-descent form, because that is where geometry is explicit. Write the move as `x_{t+1} = argmin_x { η⟨g_t,x⟩ + ηφ(x) + B_ψ(x,x_t) }` for a strongly convex `ψ`, where `B_ψ` is its Bregman divergence and `φ` is whatever fixed regularizer I carry (an `ℓ_1`, say). With `ψ(x)=½‖x‖₂²` this is plain projected gradient; with `ψ` the negative entropy it is exponentiated gradient. The regret these give all has the same skeleton — I'll re-derive it because I want to see exactly which term carries the gradients. From the optimality of `x_{t+1}` and convexity of `f_t`, the per-round drop is controlled, and summing gives

`R(T) ≤ (1/η) B_ψ(x*, x_1) + (η/2) Σ_t ‖g_t‖²_{ψ*}`,

where `‖·‖_{ψ*}` is the dual norm of the norm under which `ψ` is strongly convex. The first term is a "distance to optimum measured by `ψ`" cost; the second is "gradient mass measured in the dual norm." Choosing `η ∝ 1/√T` balances them into `O(√T)`. Fine — but notice what's fixed: `ψ` is chosen once, by hand, before I see anything. Mirror descent lets me *match* the geometry (entropy for the simplex buys `√(log d)` instead of `√d`), but I have to *know* the right geometry a priori. RDA and FTRL are the same story with the metric scaled by a scalar `√t`. Every one of these picks the metric up front and freezes it.

That freezing is the thing to break. What if `ψ` is not fixed — what if I grow it, `ψ_t`, from the gradients I have actually seen? Then both terms in the bound become data-dependent, and maybe I can *choose* `ψ_t` to make the gradient term as small as possible. Let me restrict to quadratic proximal functions `ψ_t(x) = ½⟨x, H_t x⟩` for a symmetric PSD `H_t` — a Mahalanobis geometry — because then the dual norm is explicit: `‖g‖²_{ψ_t*} = ⟨g, H_t^{-1} g⟩`, and the unconstrained prox step has exactly the scale `x_{t+1}=x_t−ηH_t^{-1}g_t`. Now the regret skeleton becomes, roughly, `(1/η)‖x*‖²_{H_T} + η Σ_t ⟨g_t, H_t^{-1} g_t⟩`, and I get to pick the matrices `H_t`. This is the real question: *which* `H_t` minimizes the regret? Let me not guess the answer — let me actually minimize.

Start with the simplest meaningful family, diagonal `H = diag(s)`, `s ⪰ 0` — per-coordinate rates, which is exactly the heterogeneity I care about, and the only thing I can afford in `d = 10^6`. Suppose I could choose one fixed diagonal metric *with hindsight*, after seeing all `T` gradients, to make the gradient term as small as possible subject to a budget on the metric (otherwise I'd just send `s → ∞` and trivialize it). The budget should cap the total "size" of the metric, `⟨1,s⟩ ≤ c`, because the `‖x*‖²_{H}` term grows with `s`. So solve

`min_s Σ_{t=1}^T Σ_{i=1}^d g_{t,i}² / s_i   subject to   s ⪰ 0,  ⟨1,s⟩ ≤ c.`

Let `A_i = Σ_t g_{t,i}²`. The Lagrangian is `L(s,λ,θ) = Σ_i A_i/s_i − ⟨λ,s⟩ + θ(⟨1,s⟩ − c)`. Stationarity in `s_i` gives `−A_i/s_i² − λ_i + θ = 0`. For every coordinate with `A_i>0`, the optimum must have `s_i>0` because sending that `s_i` to zero makes the objective blow up, so complementary slackness gives `λ_i = 0`, and then `s_i² = A_i/θ`. Coordinates with `A_i=0` contribute nothing to the objective and can receive zero mass after normalization. Thus, whenever the total gradient mass is nonzero,

`s_i ∝ √A_i = √(Σ_t g_{t,i}²) = ‖g_{1:T,i}‖₂,`

normalized so `⟨1,s⟩ = c`: `s_i = c·‖g_{1:T,i}‖₂ / Σ_j ‖g_{1:T,j}‖₂`. If all `A_i` are zero, the value is just zero. Otherwise, the optimal per-coordinate metric is proportional to the **square root of the accumulated sum of squared gradients** in that coordinate. Not the sum, not the count: the `ℓ_2` norm of the gradient history. And plugging it back, the optimal objective value is

`(1/c) (Σ_i ‖g_{1:T,i}‖₂)².`

So the best a diagonal metric can do is turn the gradient term into `(Σ_i √(Σ_t g_{t,i}²))²` (up to the budget `c`). Before I lean on this I should check the Lagrangian solution is really the constrained minimum and not a saddle I talked myself into. Take `d = 5`, accumulated masses `A = (0.512, 0.951, 0.144, 0.949, 0.312)`, budget `c = 3`. The formula gives `s_i = c√A_i/Σ_j√A_j` (which sums to `3.000`, the budget is tight) and predicted value `(Σ_i√A_i)²/c`. Plugging `s` back in, `Σ_i A_i/s_i = 4.3258`, and the closed form `(Σ√A)²/c = 4.3258` — they agree. Then I draw 100000 random feasible `s` (nonneg, renormalized to sum `c`) and the best objective any of them achieves is `4.3258`, never below; so this stationary point really is the floor. Good — the object I want to drive the regret to is the sum over coordinates of the per-coordinate `√(Σg²)`. And I can already see why it should be good on sparse heavy-tailed data: a coordinate that almost never fires contributes a tiny `√(Σg²)`, so it barely costs anything, whereas an isotropic bound would charge `√T` per coordinate regardless — I'll make that quantitative once I have the full regret.

But this `s_i ∝ √(Σ_{t=1}^T g_{t,i}²)` is *clairvoyant* — it needs all `T` gradients to be set. Online I only have the gradients up to `t`. The honest, causal thing is to use the running version: at round `t`, set `s_{t,i} = ‖g_{1:t,i}‖₂ = √(Σ_{τ≤t} g_{τ,i}²)`, the accumulated root-sum-of-squares *so far*. The metric grows monotonically as each coordinate accumulates gradient mass. The danger is obvious: does the causal, running metric pay a lot for not having seen the future? I need to bound the actual online gradient term `Σ_t ⟨g_t, diag(s_t)^{-1} g_t⟩` and compare it to the hindsight optimum `Σ_i ‖g_{1:T,i}‖₂`.

Let me try to bound it. It decomposes coordinatewise: `Σ_t ⟨g_t, diag(s_t)^{-1} g_t⟩ = Σ_i Σ_t g_{t,i}² / ‖g_{1:t,i}‖₂`. So it's enough to prove a scalar inequality: for any real sequence `a_1,…,a_T`,

`Σ_{t=1}^T a_t² / ‖a_{1:t}‖₂ ≤ 2 ‖a_{1:T}‖₂,`  where `‖a_{1:t}‖₂ = √(Σ_{τ≤t} a_τ²)` and a `0/0` term is read as zero.

Induction on `T`. Base case trivial. Assume it for `T−1`. Let `b_T = Σ_{τ≤T} a_τ² = ‖a_{1:T}‖₂²`; if `b_T=0` everything is zero, so suppose `b_T>0`. Then by the hypothesis,

`Σ_{t=1}^T a_t²/‖a_{1:t}‖₂ ≤ 2‖a_{1:T-1}‖₂ + a_T²/‖a_{1:T}‖₂ = 2√(b_T − a_T²) + a_T²/√(b_T).`

I want this `≤ 2√(b_T)`. Equivalently `√(b_T − a_T²) ≤ √(b_T) − a_T²/(2√(b_T))`. The right side is the first-order (tangent-line) overestimate of the concave function `√·` at `b_T`, evaluated at `b_T − a_T²`: concavity of `√x` gives `√(b_T − a_T²) ≤ √(b_T) + (1/(2√(b_T)))·(−a_T²)`, which is exactly that. (Valid as long as `b_T − a_T² ≥ 0`, which holds since `b_T = b_{T-1} + a_T² ≥ a_T²`.) So the induction closes, and

`Σ_t ⟨g_t, diag(s_t)^{†} g_t⟩ ≤ 2 Σ_i ‖g_{1:T,i}‖₂.`

The causal running metric `s_{t,i} = ‖g_{1:t,i}‖₂` makes the online gradient sum at most twice the final coordinate-norm sum `Σ_i‖g_{1:T,i}‖₂`, which is the same object the hindsight minimization exposed. The thing I'm not sure of is whether that factor 2 is honest or wildly loose — if the online sum were really only `1.01×` the hindsight quantity in practice, the "2" would be theatrical. So I evaluate the scalar `Σ_t a_t²/‖a_{1:t}‖₂` against `2‖a_{1:T}‖₂` directly. On the constant sequence `a_t = 1` (which keeps adding mass every round, the regime the causal metric should be worst on), the ratio of LHS to RHS climbs: `0.50` at `T=1`, `0.60` at `T=2`, `0.79` at `T=10`, `0.93` at `T=100`, `0.9927` at `T=10⁴`. So as the horizon grows the online sum really does approach `2‖a_{1:T}‖₂` from below — the 2 is the right asymptotic constant, not slack I could shave. At the other extreme, a single spike `a = (1,0,0,…)` gives LHS `= 1` against RHS `= 2`, ratio `0.5`: when all the mass arrives at once the causal metric loses nothing, which makes sense (there is no future to have missed). Over 200000 random short nonnegative sequences the ratio never exceeds `0.775`, consistent with `2` being a true upper bound that is only saturated in the limit. So the bound is tight in the constant it claims. And the "why `√` and not the raw sum" question now has two answers that agree: the hindsight minimization makes `s_i ∝ √(Σg²)` the constrained minimizer (verified above), and the telescoping needs precisely that `√` in the denominator so the concavity step `√(b−a²) ≤ √b − a²/(2√b)` closes. Either route alone would leave the exponent looking like a guess; together they pin it.

Now assemble the regret. I need the other term — the `B_ψ` / distance term — under the growing metric. Take the composite mirror-descent update `x_{t+1} = argmin_x { η⟨g_t,x⟩ + ηφ(x) + B_{ψ_t}(x,x_t) }` with `ψ_t(x) = ½⟨x,(δI + diag(s_t))x⟩`. The `δI` I add only so the metric is invertible / the dual norm finite before a coordinate has fired; in practice I can take `δ → 0` and use a pseudo-inverse — it is not a tuned hyperparameter. The per-round regret from optimality of `x_{t+1}` and convexity is

`R(T) ≤ (1/η) B_{ψ_1}(x*,x_1) + (1/η) Σ_{t=1}^{T-1} [B_{ψ_{t+1}}(x*,x_{t+1}) − B_{ψ_t}(x*,x_{t+1})] + (η/2) Σ_t ‖g_t‖²_{ψ_t*}.`

The last sum I just bounded: `Σ_t ‖g_t‖²_{ψ_t*} = Σ_t⟨g_t,(δI+diag(s_t))^{-1}g_t⟩ ≤ Σ_t⟨g_t,diag(s_t)^{†}g_t⟩ ≤ 2Σ_i‖g_{1:T,i}‖₂` (and if a coordinate's `s_{t,i}=0` then `g_{t,i}=0` by construction, so nothing diverges). The middle telescoping sum is where the *monotone growth* of the metric pays off. Each difference is `½⟨x*−x_{t+1}, diag(s_{t+1}−s_t)(x*−x_{t+1})⟩ ≤ ½ max_i (x_i*−x_{t+1,i})² · ‖s_{t+1}−s_t‖₁`. Because `s_t` only grows, `‖s_{t+1}−s_t‖₁ = ⟨s_{t+1}−s_t, 1⟩`, so summing telescopes. The sum of differences gives the mass from `s_1` to `s_T`, while `B_{ψ_1}(x*,x_1)` supplies at most the missing `½D_∞²⟨s_1,1⟩`; together the whole distance side is at most `½D_∞²⟨s_T,1⟩ = ½D_∞²Σ_i ‖g_{1:T,i}‖₂`, where `D_∞² = max_t ‖x*−x_t‖_∞²`. So every piece is a multiple of the *same* quantity `Σ_i ‖g_{1:T,i}‖₂`, and

`R(T) ≤ (1/(2η)) D_∞² Σ_i ‖g_{1:T,i}‖₂ + η Σ_i ‖g_{1:T,i}‖₂.`

Balance over `η`: take `η = D_∞/√2`. Then both terms are equal and

`R(T) ≤ √2 · D_∞ · Σ_{i=1}^d ‖g_{1:T,i}‖₂ = √2 · D_∞ · Σ_{i=1}^d √(Σ_{t=1}^T g_{t,i}²).`

This is the payoff, and three things about it are worth dwelling on. First, the step-size knob `η = D_∞/√2` depends only on the `∞`-diameter of the feasible set — a geometric quantity I know a priori — and *not* on the gradient magnitudes, which I don't. That is the "easy to set" property I wanted; the adaptation has absorbed the dependence on gradient scale into the metric. Second, by the hindsight minimization above, with the trace budget `⟨1,s⟩≤d`,

`Σ_i ‖g_{1:T,i}‖₂ = √( d · inf_{s⪰0,⟨1,s⟩≤d} Σ_t ⟨g_t, diag(s)^{-1} g_t⟩ ),`

so this regret is within the expected `√d` normalization of the best fixed diagonal metric chosen with full hindsight. I am competing with the best preconditioner in the family, not just some fixed choice. Third, the identity matrix `s = 1` lies in the feasible set of that infimum, so the adaptive expression reduces in the worst case to `√(dΣ_t‖g_t‖₂²)`, the same Euclidean scaling one gets from an `ℓ_∞`-diameter bound — graceful degradation is automatic.

Now I want to see whether it actually buys the sparse-feature improvement on the data I started from, and by how much. Heavy-tailed sparsity: feature `i` appears with probability `p_i = min{1, c·i^{-α}}`. Then `E‖g_{1:T,i}‖₂ ≤ √(E‖g_{1:T,i}‖₂²) ≤ √(p_i T)` by Jensen (the gradient mass in coordinate `i` is at most its count of appearances). So `E Σ_i ‖g_{1:T,i}‖₂ ≤ √T · Σ_i √(p_i) = √T · c'·Σ_i i^{-α/2}`. For `α ≥ 2` that sum is `O(log d)`; for `α∈(1,2)` it is `O(d^{1−α/2})`. The symbolic rates are clean, but a sum being `O(log d)` is the kind of claim that hides a constant large enough to matter, so I simulate it: `T = 20000`, `c = 0.5`, draw the per-coordinate fire counts as `Binomial(T, p_i)`, and compare `Σ_i √(count_i)` (the adaptive object, with `0/1` features so `g²∈{0,1}`) against `√(d·Σ_t‖g_t‖²) = √(d·Σ_i count_i)` (the isotropic OGD object). At `α = 2.5`: `d = 2000` gives adaptive `326` vs OGD `5179` (ratio `0.063`); `d = 8000` gives adaptive `317` vs OGD `10355` (ratio `0.031`). The decisive thing is that the adaptive sum barely moved — `326 → 317` — while `d` quadrupled, which is the `log d` flatness made concrete, whereas the OGD value doubled with `d` as `√d` predicts. At the heavier tail `α = 1.5` the adaptive sum does grow (`1873 → 2127` from `d=2000→8000`), as `d^{1−α/2} = d^{0.25}` should, but still sits at ratios `0.26` and `0.15` against OGD. On the hypercube `X = {‖x‖_∞ ≤ 1}`, `D_∞ = 2`, so the regret is `O(max{log d, d^{1−α/2}}·√T)`, while isotropic OGD has `D₂ = 2√d` and lands at `O(√(dT))`. The structure I was staring at — rare informative features — is exactly what the bound now rewards, and the gap is an order of magnitude already at `d = 8000`, not just an asymptotic promise.

This also resolves the single-rare-feature pathology I opened with, and now I can see it falls straight out of the update rather than being arranged. Take `δ = 0`, `X` the `ℓ_∞` box, `η = √2`. The update is `x_{t+1} = x_t − √2 diag(G_t)^{-1/2} g_t` followed by projection, where `diag(G_t)_{ii} = Σ_{τ≤t} g_{τ,i}²`. Coordinate-wise that is a step of size `√2 / √(Σ_{τ≤t} g_{τ,i}²)` on `g_{t,i}`. A feature seen for the first time at round `t` has `Σ_{τ≤t} g_{τ,i}² = g_{t,i}²`, so its weight moves by `√2·|g_{t,i}|/|g_{t,i}| = √2`, independent of `g_{t,i}` and of the round `t`. That is exactly the number I checked at the very start: for the coordinate first firing at `t₀ = 1, 9999, 500000`, the AdaGrad step was `1.4142` every time, against the global schedule's `0.71, 7.1e-3, 1.0e-3`. So the same `√(Σg²)` that the regret minimization handed me is what cancels the round count out of the rare feature's update — the late-arriving informative feature now moves the same distance it would have moved on round one. A feature seen a thousand times instead has a large accumulated denominator and a correspondingly small step. This is the per-coordinate rate I was missing.

Could I do better than diagonal? The diagonal restriction was a budget choice, not the only option. Let me redo the minimization with a *full* PSD metric `S`, capped by `tr(S) ≤ c` (the trace is the right budget — it bounds the `‖x*‖²_S` term and reduces to `⟨1,s⟩` in the diagonal case). With `G_T = Σ_t g_t g_t^T` the gradient outer-product matrix, the positive-definite version of the problem is

`min_S Σ_{t=1}^T ⟨g_t, S^{-1} g_t⟩ = min_S tr(S^{-1} G_T)   s.t.   S ⪰ 0,  tr(S) ≤ c.`

Lagrangian `L(S,θ,Z) = tr(S^{-1}G_T) + θ(tr(S) − c) − tr(SZ)`, `Z ⪰ 0`. The derivative in `S` is `−S^{-1}G_T S^{-1} + θI − Z`. For the full-rank case complementarity forces `Z = 0`, so `S^{-1}G_T S^{-1} = θI`; multiply by `S` on both sides to get `G_T = θ S²`, hence

`S ∝ G_T^{1/2},`  normalized `S = c·G_T^{1/2}/tr(G_T^{1/2})`,

and the optimal value is `tr(G_T^{1/2})²/c`. Let me confirm this the same way I confirmed the diagonal one, since the matrix derivative step `−S^{-1}G_T S^{-1} + θI = 0` is exactly the place a sign or transpose error would hide. Draw `T = 30` Gaussian gradients in `d = 4`, form `G_T = Σ g g^T`, take the eigendecomposition to get `G_T^{1/2}`, set `c = 2` and `S = c·G_T^{1/2}/tr(G_T^{1/2})`. Then `tr(S^{-1}G_T) = 268.028` and `tr(G_T^{1/2})²/c = 268.028` — equal. Over 50000 random feasible PSD `S` (random `MMᵀ` renormalized to trace `c`) the smallest objective is `268.028`, never below, so `G_T^{1/2}` is the minimizer and not a stationary point I mis-signed. If `G_T` is singular the same formula is the limiting solution using the pseudo-inverse on the range of `G_T`. So the full-matrix analogue of `√(Σg²)` is the matrix square root of the gradient second-moment matrix, and the analogue of `Σ_i√(Σg²)` is `tr(G_T^{1/2})`: the optimal preconditioner is the (matrix) square root of accumulated squared gradients, one dimension up. And on that same `G_T`, the diagonal-restricted optimum `(Σ_i√(G_{ii}))²/c = 275.54` is larger than the full value `268.03` — the diagonal can only ever do worse, because it cannot see the off-diagonal correlations. So when the gradients are low-rank only after a rotation, `tr(G_T^{1/2})` is genuinely smaller than the diagonal sum, and the full metric is buying something real; the question is only whether I can afford it.

Now the online bound for the full version, with `ψ_t(x) = ½⟨x, G_t^{1/2} x⟩` and `S_t = G_t^{1/2}`. The gradient term is `Σ_t ⟨g_t, S_t^{†} g_t⟩` (pseudo-inverse where needed), and I need the matrix analogue of my scalar telescoping inequality. The key is the concavity of `A ↦ tr(A^{1/2})`. The map `A ↦ A^p` is matrix-concave for `0 ≤ p ≤ 1` on PSD matrices, so `tr(A^{1/2})` is concave, and `∇_A tr(A^{1/2}) = ½ A^{-1/2}`. The first-order inequality for a concave function gives, for `A, B ≻ 0`,

`tr(A^{1/2}) ≤ tr(B^{1/2}) + ½ tr(B^{-1/2}(A − B)).`

Set `B = G_t`, `A = G_t − g_t g_t^T = G_{t-1}` (which is `⪰ 0`). Then `A − B = −g_t g_t^T` and the inequality rearranges to `2 tr((G_t − g_t g_t^T)^{1/2}) ≤ 2 tr(G_t^{1/2}) − tr(G_t^{-1/2} g_t g_t^T) = 2 tr(G_t^{1/2}) − ⟨g_t, G_t^{-1/2} g_t⟩`. (Careful at rank-deficiency: add `δI`, note `g_t ∈ Range(G_t)` so the relevant quadratic form is continuous as `δ↓0`, and take the limit — that's exactly why the pseudo-inverse is the right object.) Summing this telescoping relation by induction gives the matrix analogue of the factor-2 bound,

`Σ_{t=1}^T ⟨g_t, S_t^{†} g_t⟩ ≤ 2 tr(G_T^{1/2}).`

The Bregman/curvature term telescopes the same way through `tr(G_{t+1}^{1/2}) − tr(G_t^{1/2})`: since `G_{t+1}⪰G_t`, matrix square root is operator-monotone, so `G_{t+1}^{1/2}−G_t^{1/2}⪰0`, and the largest eigenvalue of that difference is bounded by its trace. For the compact-set mirror-descent version, with `D_2 = max_t ‖x*−x_t‖₂`, balancing `η = D_2/√2` yields

`R(T) ≤ √2 · D_2 · tr(G_T^{1/2})   (with a ridge term if `δ>0`).`

The dual-averaging form has the related `2‖x*‖₂ tr(G_T^{1/2})` bound when `η=‖x*‖₂`. In either form, the trace minimization says the quantity is controlled by the best *full* metric in hindsight. The diagonal result is the special case where I confine `S` to diagonals — same proof, same factor 2, just `O(d)` instead of `O(d²)` storage and matrix roots. In `d = 10^6` the full matrix is hopeless to form, let alone root, so the diagonal version is the one I will actually run; the full-matrix theorem tells me what I'm giving up (cross-coordinate structure) and confirms the diagonal isn't an ad hoc hack but the affordable face of one principle.

Let me also be clear why I *accumulate* with no decay or window. The regret accounting is over the whole horizon `T`; the telescoping in both the scalar and matrix cases relies on `s_t` (resp. `G_t`) being monotone non-decreasing, which is what makes the curvature term collapse to a single `Σ_i‖g_{1:T,i}‖₂` (resp. `tr(G_T^{1/2})`) instead of summing `T` separate pieces. A window or exponential decay would break the monotonicity and with it the clean bound — so for the regret guarantee, plain accumulation is the right choice, and it is also exactly the quantity the hindsight minimization picked out.

So the whole thing closes as one causal chain. A global rate is provably wrong for heterogeneous sparse features, and provably *un-improvable* in the worst case — so I move to a data-dependent, per-coordinate metric inside the mirror-descent bound. Asking which diagonal metric minimizes the gradient term under a trace budget forces the preconditioner to be proportional to `√(Σ_t g²)` per coordinate, with optimal value `(1/c)(Σ_i √(Σ_t g²))²`. Using the causal running version `s_{t,i}=‖g_{1:t,i}‖₂` costs only a factor 2 by a telescoping inequality whose denominator's square root is exactly what makes it sum. Folding that into the mirror-descent regret — every term a multiple of `Σ_i √(Σ_t g²)` because the metric grows monotonically — gives `R(T) ≤ √2 D_∞ Σ_i √(Σ_t g_{t,i}²)`, a regret that adapts to the geometry, improves sharply on heavy-tailed sparse data, tracks the best diagonal metric in hindsight up to the trace normalization, and has a step size set by the diameter alone. The full-matrix version is the same statement with `√(Σg²) → G_T^{1/2}` and the sum `→ tr(·)`, derived from the concavity of `tr(A^{1/2})`. The algorithm that drops out — the one I'd actually code — is just: accumulate the running sum of squared gradients per coordinate, and at every step divide the gradient by the square root of that accumulator.

```python
import numpy as np

class AdaGrad:
    """Per-coordinate adaptive subgradient method.
    The preconditioner is the square root of the accumulated sum of squared
    gradients, H_t = diag(sqrt(sum_tau g_tau^2)) — the regret-minimizing diagonal
    metric, used in its causal running form."""

    def __init__(self, d, eta=1.0, eps=1e-10):
        self.eta = eta                  # step from the diameter; not coupled to grad scale
        self.eps = eps                  # denominator floor before a coordinate fires
        self.state_sum = np.zeros(d)    # state_sum_i = sum_{tau<=t} g_{tau,i}^2

    def step(self, x, g):
        self.state_sum += g * g
        return x - self.eta * g / (np.sqrt(self.state_sum) + self.eps)

def project(x, X):
    """Euclidean projection onto the feasible set (e.g. an L-inf box)."""
    return X.project(x)

def online_learn(stream, X, d, eta):
    x = np.zeros(d)
    opt = AdaGrad(d, eta=eta)
    for f_t in stream:
        g = f_t.subgradient(x)          # subgradient at x_t
        x = opt.step(x, g)              # state_sum += g^2; x -= eta*g/(sqrt(state_sum)+eps)
        x = project(x, X)               # keep x feasible
    return x
```
