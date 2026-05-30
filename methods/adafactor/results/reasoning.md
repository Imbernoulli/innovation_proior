OK. Let me think about where the memory actually goes when I train a big model with Adam, because that's the thing that's biting me. I have the parameters themselves — one big array. I have the gradient — another array the same size. Fine, those are unavoidable. Then Adam adds its state: the first moment `m`, an exponential moving average of the gradient, same shape as the parameters; and the second moment `v`, an exponential moving average of the *squared* gradient, again same shape as the parameters. So the optimizer alone is carrying two more full copies of the model. Parameters plus gradient plus `m` plus `v` is four model-sized arrays live at once.

And the place I actually want to grow is exactly the place this hurts most. On translation and language data, quality keeps climbing as the model gets bigger — billions of parameters, huge embedding tables, wide weight matrices. Compute has gotten cheap relative to memory; what stops me from fitting a bigger model isn't flops, it's that the parameters and all this auxiliary state don't fit. Two full extra copies for the optimizer is the thing standing between me and a model twice the size. So the question is sharp: can I keep what makes Adam good — the per-coordinate rescaling, dividing each parameter's step by the running magnitude of its own gradient — while paying a lot less than a full model-sized buffer for it?

Let me be precise about *why* that buffer is the size it is, because that's where any saving has to come from. The whole point of `v` is that it's per-coordinate. Adam steps `x ← x − α·m̂/(√v̂ + ε)`, and the `√v̂` in the denominator is a separate number for every single parameter — that's the adaptivity. Parameter `i` gets divided by the recent RMS of *its* gradient, parameter `j` by the recent RMS of *its* gradient. That's exactly what lets one learning rate work across coordinates whose gradients differ by orders of magnitude. So `v` genuinely has one number per parameter because the rescaling genuinely is per parameter. I can't just throw entries away; if I drop the resolution I lose the adaptivity that's the reason I'm using this method at all.

So I can't reduce the *granularity* of the rescaling. What I might be able to reduce is the *storage I need to reconstruct* that granularity. Those are different. If the per-coordinate denominators, laid out as a matrix, have structure — if they're not arbitrary — then I might be able to store something small and rebuild the full per-coordinate denominator on the fly each step.

Where would structure come from? Most of the parameters in these models are in two-dimensional weight matrices — the linear maps, the embedding tables. Take one such matrix `W ∈ ℝ^{n×m}`. Under Adam I'm keeping `V ∈ ℝ^{n×m}`, the moving average of `(∇_W f)²` entrywise, same `n×m` shape. That's `nm` numbers, and `nm` is the whole problem. But a matrix isn't a featureless bag of `nm` scalars — it has rows and columns. Maybe the right thing to store isn't every entry of `V` but some per-row and per-column summary, `n + m` numbers instead of `nm`, and then reconstruct an estimate `V̂` of the full `V` from those at each step. For a big square matrix that's roughly `2n` instead of `n²` — the saving I need. The question becomes: store small factors, reconstruct `V̂ ≈ V`, divide by `√V̂`. What's the right factorization?

The cleanest version of "per-row and per-column summary" is a rank-1 outer product. Write `V̂ = R Sᵀ` with `R ∈ ℝⁿ`, `S ∈ ℝᵐ`, so `V̂_{ij} = R_i S_j`. That's exactly `n + m` numbers. More generally I could go rank-`k`, `R ∈ ℝ^{n×k}`, `S ∈ ℝ^{k×m}`, `V ≈ RS`, but rank-1 is the extreme that gives the most saving, so let me see how far rank-1 gets me and whether it's good enough. Now — what's the *best* rank-1 (and the factors I should keep)?

My first instinct is the textbook answer: the best low-rank approximation of a matrix is the truncated SVD. Keep the top singular value and its left/right singular vectors; Eckart–Young says that's the optimal rank-1 approximation in Frobenius norm. Done?

No — stare at this, it doesn't fit the problem at all. Two things break.

First, and this is the one that actually kills it: I am not approximating a *fixed* matrix once. `V` is a *moving average*, updated every single step as `V_t = β₂ V_{t-1} + (1−β₂) G_t²`. If I'm only storing small factors of `V`, then to do that update I'd need the factors of `V_t` to be computable from the factors of `V_{t-1}` and the new squared gradient. But the SVD doesn't behave that way under addition — the singular vectors of `β₂ V_{t-1} + (1−β₂) G_t²` are not any simple function of the singular vectors of `V_{t-1}` and of `G_t²`. The factorization doesn't commute with the exponential smoothing. The moving average of the factors is *not* the factors of the moving average. So I'd have to reconstruct full `V`, add, re-factor — which defeats the entire purpose, since reconstructing full `V` is the `nm` memory I'm trying to avoid. And re-running an SVD every step would be a compute bottleneck on top of it.

Second: SVD factors can be negative. The singular vectors have signed entries, so `V̂ = σ u vᵀ` can have negative entries. But `V` is a second moment — it's a moving average of squares, it's nonnegative — and I need to take `1/√V̂` of the reconstruction. A negative entry there is a square root of a negative number. Nonsensical. So even setting aside the smoothing problem, Frobenius-optimal SVD violates the one hard constraint I have on the output.

So I need a factorization with two properties the SVD lacks: the factors I store must be **nonnegative** (so the reconstruction stays nonnegative and square-rootable), and — the real requirement — the thing I store must be **linear in `V`**, so that smoothing and factoring commute and I can maintain the small factors directly as moving averages without ever materializing full `V`. Linearity is the property that makes the whole scheme possible; let me hold onto it.

Nonnegative factors of a nonnegative matrix — that's nonnegative matrix factorization. NMF approximates a nonnegative `V` by `RS` with `R, S ≥ 0`, under a cost function chosen to play nicely with nonnegativity. The Frobenius norm is one option, but there's another standard NMF cost that I have a feeling will behave better here: the generalized Kullback–Leibler divergence, the I-divergence. For nonnegative scalars,

  `d(p, q) = p log(p/q) − p + q`,

with the conventions `0/0 = 0`, `0 log 0 = 0`, and `p/0 = ∞` for `p > 0`. Quick sanity check that this is a sensible discrepancy: is `d(p,q) ≥ 0`, zero only at `p = q`? Set `x = p/q`. Then `d = q(x log x − x + 1)`. The function `x log x ≥ x − 1` for `x > 0` (it's the standard convexity inequality for `x log x`, with equality only at `x = 1`), so `x log x − x + 1 ≥ 0`, zero exactly at `x = 1`, i.e. `p = q`. Good, it's a genuine divergence.

Now minimize the total elementwise divergence subject to nonnegativity:

  minimize over `R ≥ 0, S ≥ 0`:  `Σ_{i=1}^n Σ_{j=1}^m d(V_{ij}, [RS]_{ij})`.

For general rank `k` this is hard — no closed form, you alternate between solving for `R` with `S` fixed and vice versa. But I only want rank 1. Let me just try to solve the rank-1 case directly and see if it closes.

Rank 1 means `R ∈ ℝⁿ`, `S ∈ ℝᵐ`, and `[RS]_{ij} = R_i S_j`. Substitute the divergence and expand:

  `Σ_{i,j} d(V_{ij}, R_i S_j) = Σ_{i,j} ( V_{ij} log(V_{ij}/(R_i S_j)) − V_{ij} + R_i S_j )`.

Split the log: `log(V_{ij}/(R_i S_j)) = log V_{ij} − log R_i − log S_j`. So the objective is

  `Σ_{i,j} V_{ij} log V_{ij}  −  Σ_{i,j} V_{ij} log R_i  −  Σ_{i,j} V_{ij} log S_j  −  Σ_{i,j} V_{ij}  +  Σ_{i,j} R_i S_j`.

The first and fourth terms don't involve `R` or `S` at all — constants for the optimization. So I differentiate the rest. Take `∂/∂R_i`. The term `−Σ_{i',j} V_{i'j} log R_{i'}` contributes, for the specific `i`, `−(Σ_j V_{ij})·(1/R_i)`. The term `Σ_{i',j} R_{i'} S_j` contributes `Σ_j S_j`. Setting the derivative to zero:

  `−(Σ_j V_{ij})/R_i + Σ_j S_j = 0  ⟹  R_i = (Σ_j V_{ij}) / (Σ_j S_j)`.

By the same computation in the other variable, `∂/∂S_j = 0` gives

  `−(Σ_i V_{ij})/S_j + Σ_i R_i = 0  ⟹  S_j = (Σ_i V_{ij}) / (Σ_i R_i)`.

Look at what these say. `R_i` is proportional to the `i`-th *row sum* of `V`, `Σ_j V_{ij}`. `S_j` is proportional to the `j`-th *column sum* of `V`, `Σ_i V_{ij}`. The proportionality constants `Σ_j S_j` and `Σ_i R_i` are just scalars that couple the two equations. I took derivatives as if the relevant factors were positive, so I need to check the boundary cases before trusting the formula. If the entire matrix is zero, `d(0,q)=q`, so the minimum is reached by any factors whose product is the zero matrix; the grand-total denominator is just a degenerate zero case there. If only row `i` has zero sum, then every `V_{ij}` in that row is zero, and the row's contribution is `Σ_j R_i S_j`; with any positive column mass the minimum sets `R_i=0`, exactly the row-sum formula. The same argument sets `S_j=0` for a zero column. So the stationary equations and the boundary cases agree: zero rows and columns get zero factors, and the positive submatrix follows the derivative equations.

There's a scale ambiguity I should pin down, because the loss only sees the product `R S`: for any `α > 0`, `(αR, S/α)` gives the same `RS` and hence the same loss. So the minimizer isn't a single `(R,S)` — it's a one-parameter family, and I can fix the gauge however is convenient. Fix `Σ_i R_i = Σ_{i,j} V_{ij}`, the grand total of all entries. Then from `R_i = (Σ_j V_{ij})/(Σ_j S_j)`: summing both sides over `i` gives `Σ_i R_i = (Σ_{i,j} V_{ij})/(Σ_j S_j)`, and since I set the left side equal to `Σ_{i,j} V_{ij}`, I get `Σ_j S_j = 1`. That collapses the relations to a clean canonical solution:

  `R_i = Σ_j V_{ij}`  (the row sums),  `S_j = (Σ_i V_{ij}) / (Σ_{i,j} V_{ij})`  (the column sums, normalized by the grand total).

In vector form, with `1_ℓ` the all-ones column vector in `ℝ^ℓ`:

  `R = V 1_m`,  `S = (1_nᵀ V) / (1_nᵀ V 1_m)`,

and the reconstruction

  `V̂ = R S = (V 1_m)(1_nᵀ V) / (1_nᵀ V 1_m)`.

Entrywise, if I call the raw column sums `C = 1_nᵀ V` (so `C_j = Σ_i V_{ij}`) and keep `R_i = Σ_j V_{ij}` as the raw row sums, then

  `V̂_{ij} = R_i C_j / (Σ_k R_k)`,

since `Σ_k R_k = Σ_{i,j} V_{ij} = 1_nᵀ V 1_m` is the grand total, which is the same whether I sum the row sums or the column sums. The whole family of minimizers is exactly the pairs whose product equals this `V̂` — the gauge just slides the split between `R` and `S`.

Now let me check this is what I actually want, not just what the math handed me.

Is it nonnegative? `V ≥ 0` entrywise, so all row sums and column sums are `≥ 0`, and `V̂_{ij} = R_i C_j / Σ_k R_k ≥ 0`. Good — square-rootable, the hard constraint is satisfied by construction.

Does it recover `V` when it should? If `V` is genuinely rank 1, say `V = a bᵀ`, then row sums `R = a (1ᵀb)`, column sums `C = b (1ᵀa)`, grand total `Σ R = (1ᵀa)(1ᵀb)`, and `V̂_{ij} = a_i(1ᵀb)·b_j(1ᵀa)/((1ᵀa)(1ᵀb)) = a_i b_j = V_{ij}`. Exact recovery. So I'm only ever losing information to the extent `V` is *not* rank 1, which is the price I'm knowingly paying.

And — this is the property the whole scheme hinges on — `V̂` depends on `V` only through the row sums `V 1_m` and the column sums `1_nᵀ V`, and *those are linear functions of `V`*. This is the thing the SVD couldn't give me. Because the row sum is linear, the row sum of a moving average equals the moving average of the row sums:

  row-sum of `(β₂ V_{t-1} + (1−β₂) G_t²)` = `β₂·(row-sum of V_{t-1}) + (1−β₂)·(row-sum of G_t²)`,

and identically for columns. So I never need to form `V` at all. I keep two small running averages — `R_t ∈ ℝⁿ` of the row sums of the squared gradients, `C_t ∈ ℝᵐ` of the column sums — update each by exponential smoothing against the new squared gradient's row/column sums, and reconstruct `V̂_t = R_t C_t / (1_nᵀ R_t)` on the fly when I need the denominator. Storage is `n + m`, not `nm`. The factoring and the smoothing commute exactly, which is precisely what made SVD unusable and makes this work. And note `1_nᵀ R_t = C_t 1_m` — both equal the grand total — so even though the formula looks like it singles out the row sums for the normalizer, rows and columns are treated symmetrically; I could normalize by either.

So the matrix update, with Adam's machinery otherwise intact and momentum set aside for the moment (`β₁ = 0`):

  `R_t = β₂ R_{t-1} + (1−β₂) (G_t²) 1_m`
  `C_t = β₂ C_{t-1} + (1−β₂) 1_nᵀ (G_t²)`
  `V̂_t = (R_t C_t / 1_nᵀ R_t) / (1 − β₂ᵗ)`
  `X_t = X_{t-1} − α_t G_t / (√V̂_t + ε)`

with the same bias-correction `1/(1−β₂ᵗ)` as Adam since `R` and `C` are zero-initialized EMAs. For a parameter that isn't a matrix — a vector or scalar — there's nothing to factor, so I just keep the ordinary full `v` there; those are small anyway.

One worry before I trust this: by collapsing `V` to row and column sums, am I throwing away something the model needs? Two crude alternatives bound the question — instead of the outer product, just use the row means alone (one denominator per row, broadcast across columns) or the column means alone. If I imagine the shared input-embedding / output-softmax matrix where each row is a vocabulary token: frequent tokens get large-magnitude gradients, rare tokens tiny ones, so the variation that matters runs *down the rows*. A row-only summary keeps that and should be about as good as the full thing; a column-only summary averages frequent and rare tokens together within each column and destroys exactly the structure that matters — it should be much worse. The rank-1 outer product keeps both the row and the column profile, so it should be safe where either crude version is, and the column-only failure mode is the one to watch. That matches the intuition that the row/column structure is real signal, not noise I'm safe to average over.

Now, I came here to save memory, and I've cut the *second*-moment buffer for matrices from `nm` to `n+m`. But Adam has *two* buffers. The first moment `m` is still a full model-sized array. If I'm serious about memory, I should ask whether I can drop momentum entirely — set `β₁ = 0`, no first-moment accumulator at all. For vectors and scalars that takes their extra memory to zero; for matrices it leaves only the `O(n+m)` factored second moment. That's the dream: sublinear extra memory everywhere.

Does dropping momentum cost me anything? I try it. With the usual schedule that includes a learning-rate warmup, the run with momentum and the run without it land in essentially the same place. But take the warmup *away*, and without momentum training falls apart — it goes unstable and the model never gets off the ground, while the same run *with* momentum is fine. So momentum was quietly stabilizing something, and the warmup was independently papering over the same crack. Pull both away and the underlying problem shows. I'd rather understand and fix that problem directly than carry a full momentum buffer just to mask it — so what is the instability?

The decay rate has its own tension. Reddi et al. (2018) show that *fast* decay of the second moment — small `β₂`, the average forgets quickly — can cause Adam to fail to converge. The standard fix is to push `β₂` up, decay slowly, remember longer. But slow decay has its own failure mode, and I think that's exactly what's biting me without warmup. A slowly-decaying `v̂` is built from gradients far in the past. If the model is moving fast, that stale estimate no longer matches the current gradient scale, and the step `g/√v̂` can come out much larger than intended — and a few oversized steps are enough to destabilize training.

Let me get a handle on "much larger than intended" so it's measurable, not vibes. Look at the *unscaled* update on a whole matrix `X`, `u_x = −g_x/√v̂_x` for each entry `x`, and take its root-mean-square over the matrix:

  `RMS(U_t) = √( mean_{x∈X} ( g_x² / v̂_x ) )`.

If `v̂` is doing its job — tracking `g²` — then for each entry `g_x²/v̂_x ≈ 1`, so the mean is about 1 and `RMS(U_t) ≈ 1`. So `RMS(U_t)` near 1 is the signature of a healthy estimator, and `RMS(U_t)` drifting well above 1 is a direct readout that `v̂` is stale and the steps are too big. I watch this on a real weight matrix: with fast decay (`β₂ = 0.9`) it sits near 1, just as the picture predicts; with slow decay (`β₂ = 0.999`) it fluctuates wildly, spiking well above 1. And those spikes line up with the instability. So I have a candidate cause — out-of-date second moments producing larger-than-desired updates — and a number that detects it. (It's a *hypothesis*, not proof: the large updates could in principle be a symptom rather than the cause. But I can check those `RMS(U_t)` traces come from runs *with* warmup that didn't go unstable, so the large updates appear even without any instability to have caused them — the arrow points from stale estimate to big step, not the other way. And if I'm right, directly suppressing the big steps should cure the instability, which is a test I can run.)

So: directly suppress the larger-than-desired updates. I have `RMS(U_t)`, the actual RMS of the unscaled step, and I know I *want* it near some target `d` (think `d = 1`). Whenever it exceeds `d`, scale the whole update down so its RMS is exactly `d`; when it's already below, leave it alone:

  `Û_t = U_t / max(1, RMS(U_t)/d)`.

If `RMS(U_t) ≤ d` the denominator is 1 and nothing changes; if `RMS(U_t) > d` it divides by `RMS(U_t)/d`, bringing the RMS down to exactly `d`. Then the real step is `α_t Û_t`.

I should make sure this isn't just ordinary gradient clipping renamed. Gradient clipping rescales the *gradient* when its norm is too big. For plain SGD the step direction is the gradient, so capping the gradient caps the step — fine. But here the step is the gradient *after* per-coordinate division by `√V̂`, and that division can blow a perfectly modest gradient up into a huge update when `V̂` is stale and small. Capping the gradient norm says nothing about the size of `g/√V̂`. The whole problem lives *after* the adaptive rescaling, so I have to clip *there* — on `U = G/√V̂`, the actual update direction — not on `G`. That's the distinction: cap the real update, not the raw gradient. The threshold has to be tight enough to catch the spikes; `d = 1` matches the healthy scale `RMS(U_t) ≈ 1`, while a much looser cap can let the same oversized updates through.

Clipping treats the oversized update after it appears; I also want the estimator to be less stale in the first place. Fast decay fails to converge; slow decay goes stale early. What I actually want is to decay fast *early* — when the model is moving fastest and a stale estimate is most dangerous — and slow *later*, once things settle and I'd like the longer memory. That's a `β₂` that *increases* over time, which is also the shape Reddi et al. (2018) advocate for the convergence fix.

Adam's bias correction is already hiding that shape. Adam keeps `v_t = β₂ v_{t-1} + (1−β₂) g_t²` and reports `v̂_t = v_t/(1−β₂ᵗ)`. Let me rewrite the recursion directly in terms of the corrected `v̂` and see what effective decay it implies:

  `v̂_t = v_t/(1−β₂ᵗ) = [β₂ v_{t-1} + (1−β₂) g_t²]/(1−β₂ᵗ)`.

Substitute `v_{t-1} = (1−β₂^{t-1}) v̂_{t-1}`:

  `v̂_t = [β₂(1−β₂^{t-1})/(1−β₂ᵗ)]·v̂_{t-1} + [(1−β₂)/(1−β₂ᵗ)]·g_t²`.

Name the first coefficient `β̂₂_t = β₂(1−β₂^{t-1})/(1−β₂ᵗ)`. Check the second coefficient is `1 − β̂₂_t`:

  `1 − β̂₂_t = [(1−β₂ᵗ) − β₂(1−β₂^{t-1})]/(1−β₂ᵗ) = [(1−β₂ᵗ) − (β₂−β₂ᵗ)]/(1−β₂ᵗ) = (1−β₂)/(1−β₂ᵗ)`.

Exactly. So Adam's bias-corrected recursion is identical to a *plain* EMA with no separate correction, run with a time-varying decay

  `v̂_t = β̂₂_t v̂_{t-1} + (1 − β̂₂_t) g_t²`,  `β̂₂_t = β₂(1−β₂^{t-1})/(1−β₂ᵗ)`.

At `t = 1`, `β̂₂_1 = β₂(1−1)/(1−β₂) = 0`, so `v̂_1 = g_1²` — no stale prior, the estimate is just the first squared gradient. As `t → ∞`, `β̂₂_t → β₂`. So the bias correction *is* an increasing decay rate that starts at 0 and rises to `β₂`. That reframing is the key: I don't need to bolt a schedule onto Adam, I need to *choose* the increasing schedule `β̂₂_t` directly.

So propose the family

  `β̂₂_t = 1 − t^{-c}`,  `t ≥ 1`,  `c > 0`.

At `t = 1` it's `1 − 1 = 0` (no prior on the first step, just like the corrected Adam), and it rises toward 1 as `t → ∞`. The exponent `c` controls how fast. This is the "fast early, slow late" shape I wanted.

Bias correction exists to fix the zero initialization: a fresh zero-initialized EMA underestimates the true second moment early on, and the `1/(1−β₂ᵗ)` factor rescales it back. If my schedule already produces an unbiased estimate, I don't need that factor at all. Let me check.

Unroll the recursion `v_t = β̂₂_t v_{t-1} + (1−β̂₂_t) g_t²` from `v_0 = 0`:

  `v_t = Σ_{i=1}^t (1 − β̂₂_i) ( Π_{j=i+1}^t β̂₂_j ) g_i²`,

each past squared gradient `g_i²` weighted by `(1−β̂₂_i)` for when it entered times the product of all the decays since. Take the expectation, pulling it inside the (deterministic) weights:

  `E[v_t] = Σ_{i=1}^t (1 − β̂₂_i)( Π_{j=i+1}^t β̂₂_j ) E[g_i²]`.

I want `E[v_t] ≈ E[g_t²]`, the current second moment. Split `E[g_i²] = E[g_t²] + (E[g_i²] − E[g_t²])`:

  `E[v_t] = ( Σ_{i=1}^t (1−β̂₂_i) Π_{j=i+1}^t β̂₂_j ) E[g_t²]  +  Σ_{i=1}^t (1−β̂₂_i) Π_{j=i+1}^t β̂₂_j (E[g_i²] − E[g_t²])`.

In the stationary case the second sum vanishes (all `E[g_i²]` equal), and even nonstationary it's the tracking error from drift. So the zero-initialization bias is gone, and in the stationary case `E[v_t] = E[g_t²]` exactly, *provided the weights sum to 1*:

  `Σ_{i=1}^t (1 − β̂₂_i) Π_{j=i+1}^t β̂₂_j = 1`.

Prove that holds, by induction on `t`. At `t = 1` the sum is just `1 − β̂₂_1`, and I need that to be 1, i.e. `β̂₂_1 = 0` — which my schedule satisfies, `1 − 1^{-c} = 0`. Inductive step: assume the sum equals 1 at `t−1`. At `t`, peel off the `i = t` term, `(1 − β̂₂_t)`, and notice every `i < t` term picks up exactly one extra factor `β̂₂_t` from `Π_{j=i+1}^t` versus `Π_{j=i+1}^{t-1}`:

  `Σ_{i=1}^t (1−β̂₂_i) Π_{j=i+1}^t β̂₂_j = β̂₂_t · Σ_{i=1}^{t-1}(1−β̂₂_i)Π_{j=i+1}^{t-1} β̂₂_j + (1−β̂₂_t) = β̂₂_t·1 + (1−β̂₂_t) = 1`.

Done. So *any* schedule with `β̂₂_1 = 0` makes the EMA weights sum to one; under the same stationarity approximation that justifies Adam's bias correction, that means no separate correction term is needed. Bias correction falls out for free.

There's a second condition I should impose, separate from unbiasedness: I don't want gradients from the distant past to keep nontrivial weight forever, or the estimate never really tracks the present. So I want, for every fixed `i`,

  `lim_{t→∞} (1 − β̂₂_i) Π_{j=i+1}^t β̂₂_j = 0`.

With `β̂₂_j = 1 − j^{-c}`, the `i = i` factor `1−β̂₂_i = i^{-c}` is just a constant in `t`, so this is asking whether `Π_{j=i+1}^t (1 − j^{-c}) → 0`. Pull out the finite head and look at the tail `Π_{j≥2}(1 − j^{-c})`. The standard fact: for `0 ≤ a_j < 1`, the infinite product `Π(1−a_j)` converges to a *nonzero* limit iff `Σ a_j` converges. Here `a_j = j^{-c}`, and `Σ_j j^{-c}` diverges exactly when `c ≤ 1`. So the product goes to *zero* — which is what I want — iff `c ≤ 1`. If `c > 1` the decay rises too fast: `β̂₂_t → 1` so quickly that ancient gradients keep a weight bounded away from zero forever, and the estimator freezes around its early history. So the schedule must satisfy `0 < c ≤ 1`. The boundary `c = 1` is a tidy special case: `β̂₂_t = 1 − 1/t`, and unrolling gives `v_t = (1/t) Σ_{i=1}^t g_i²` — a plain running arithmetic mean of all squared gradients. A middle value like `c = 0.8` sits between "forget fast" and "remember everything," which is the regime I want, and it pairs well with update clipping.

One more piece, and it's about what the learning rate even *means*. Adam's `α` is an *absolute* target step size — it says "move each parameter about `α` in the rescaled metric." But there's an observation worth taking seriously: training tends to behave well when each parameter's update is roughly `10⁻²` to `10⁻³` *times the magnitude of the parameter itself*. That's a *relative* statement — the right step for a parameter depends on how big that parameter currently is. An absolute `α` ignores this, which is why getting one `α` to work across, say, an embedding matrix and a deep weight matrix usually forces you to hand-engineer the initialization scales — initialize embeddings small, multiply them up by `√d_model` in the forward pass — purely so a single absolute step is appropriate for both. If I instead scale the step by the parameter's own scale, that fragile coupling goes away and the optimizer adapts to whatever scale the parameters happen to be at.

So define the parameter scale as the RMS of its entries, and lower-bound it by a small `ε₂`:

  `α_t = max(ε₂, RMS(X_{t-1})) · ρ_t`,

where `ρ_t` is now a *relative* step schedule. The `max(ε₂, ·)` matters specifically for parameters initialized at or near zero: their RMS is ~0, so without the floor the relative step would be ~0 and they could never move off zero. The floor `ε₂` gives them a minimum scale to escape the origin. Pick `ε₂ = 10⁻³`.

Now assemble everything into one rule. For a matrix parameter `X ∈ ℝ^{n×m}`:

  `α_t = max(ε₂, RMS(X_{t-1})) ρ_t`
  `G_t = ∇f_t(X_{t-1})`
  `R_t = β̂₂_t R_{t-1} + (1−β̂₂_t)(G_t² + ε₁ 1_n 1_mᵀ) 1_m`
  `C_t = β̂₂_t C_{t-1} + (1−β̂₂_t) 1_nᵀ(G_t² + ε₁ 1_n 1_mᵀ)`
  `V̂_t = R_t C_t / 1_nᵀ R_t`
  `U_t = G_t / √V̂_t`
  `Û_t = U_t / max(1, RMS(U_t)/d)`
  `X_t = X_{t-1} − α_t Û_t`.

A couple of details I want to be deliberate about. The `ε₁` (take it tiny, `10⁻³⁰`) is added *inside* the accumulator, to the squared gradient, purely so `V̂` can't be exactly zero and blow up the division — it's a floor on the second moment, not the additive-`ε` outside a square root that Adam uses. There's no `1/(1−β̂₂ᵗ)` bias-correction factor anywhere, because the `β̂₂_1 = 0` schedule already made the estimate unbiased — that's the payoff from the increasing-decay derivation. And `β̂₂_t = 1 − t^{-c}` with `c = 0.8`.

For a parameter that's a vector or scalar there's nothing to factor — keep the ordinary unfactored second moment, but everything else (relative step, increasing decay, update clipping) carries over:

  `α_t = max(ε₂, RMS(X_{t-1})) ρ_t`
  `V̂_t = β̂₂_t V̂_{t-1} + (1−β̂₂_t)(G_t² + ε₁ 1_n)`
  `U_t = G_t / √V̂_t`
  `Û_t = U_t / max(1, RMS(U_t)/d)`
  `X_t = X_{t-1} − α_t Û_t`.

And momentum stays off, `β₁ = 0`, so there's no first-moment buffer at all; if I ever wanted it back I could reinstate the first-moment EMA on the clipped update exactly as Adam does, at the cost of the extra buffer. The recommended knobs: `ε₁ = 10⁻³⁰`, `ε₂ = 10⁻³`, `d = 1`, `ρ_t = min(10⁻², 1/√t)`, `β̂₂_t = 1 − t^{-0.8}`.

Let me write it as code I'd actually run. The thing to get right when implementing: rather than literally storing the row/column *sums* `R` and `C` and the grand-total normalizer separately, it's cleaner and numerically nicer to keep the row and column *means* of the squared gradient. This is not an approximation beyond the rank-1 fit. If `r_i = (1/m)Σ_j V_{ij}` and `c_j = (1/n)Σ_i V_{ij}`, then `V̂_{ij} = R_i C_j / Σ_k R_k = r_i c_j / mean_i(r)`, so `1/√V̂_{ij} = √(mean_i(r)/r_i) · √(1/c_j)`. That is exactly `rsqrt(r_i / mean(r))` along the row axis times `rsqrt(c_j)` along the column axis, broadcast into the full matrix. A parameter counts as a matrix to factor when it has at least two axes; the row buffer has the shape of all-but-the-last axis, the column buffer all-but-the-second-last.

```python
import math
import torch
from torch.optim import Optimizer


class Adafactor(Optimizer):
    """Adaptive optimizer with sublinear extra memory.

    Matrix params: store only row/col means of the squared gradient and
    reconstruct the per-coordinate denominator as their rank-1 outer product
    (the I-divergence-optimal rank-1 fit to the second-moment matrix).
    Plus: no first moment by default, an increasing second-moment decay that
    removes bias correction, update clipping by RMS, and a relative step size.
    """

    def __init__(self, params, lr=None, eps=(1e-30, 1e-3), clip_threshold=1.0,
                 decay_rate=-0.8, beta1=None, scale_parameter=True,
                 relative_step=True, warmup_init=False):
        if lr is not None and relative_step:
            raise ValueError("Cannot combine manual `lr` with relative_step=True")
        if warmup_init and not relative_step:
            raise ValueError("warmup_init=True requires relative_step=True")
        # eps = (eps1, eps2): eps1 floors the second moment, eps2 floors the
        # parameter scale so zero-initialized params can escape 0.
        defaults = dict(lr=lr, eps=eps, clip_threshold=clip_threshold,
                        decay_rate=decay_rate, beta1=beta1,
                        scale_parameter=scale_parameter,
                        relative_step=relative_step, warmup_init=warmup_init)
        super().__init__(params, defaults)

    @staticmethod
    def _rms(t):                                   # RMS over all entries
        return t.norm(2) / (t.numel() ** 0.5)

    @staticmethod
    def _get_lr(group, state):
        rel = group["lr"]
        if group["relative_step"]:                 # rho_t = min(1e-2, 1/sqrt(t))
            min_step = 1e-6 * state["step"] if group["warmup_init"] else 1e-2
            rel = min(min_step, 1.0 / math.sqrt(state["step"]))
        scale = 1.0
        if group["scale_parameter"]:               # relative step: max(eps2, RMS(X)) * rho
            scale = max(group["eps"][1], state["RMS"])
        return scale * rel

    @staticmethod
    def _approx_sq_grad(row_mean, col_mean):
        # 1/sqrt(V_hat) = rsqrt(r_i / mean(r)) * rsqrt(c_j), the rank-1
        # reconstruction from row/col sums in mean form (outer product).
        r = (row_mean / row_mean.mean(dim=-1, keepdim=True)).rsqrt_().unsqueeze(-1)
        c = col_mean.unsqueeze(-2).rsqrt()
        return torch.mul(r, c)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("Adafactor does not support sparse gradients")
                if grad.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.float()
                state = self.state[p]
                factored = grad.dim() >= 2            # factor over the final two axes
                use_first_moment = group["beta1"] is not None

                if len(state) == 0:                  # lazy state init
                    state["step"] = 0
                    if use_first_moment:
                        state["exp_avg"] = torch.zeros_like(grad)
                    if factored:                     # O(n)+O(m), not O(nm)
                        state["row"] = torch.zeros(grad.shape[:-1]).to(grad)
                        state["col"] = torch.zeros(grad.shape[:-2] + grad.shape[-1:]).to(grad)
                    else:
                        state["v"] = torch.zeros_like(grad)
                    state["RMS"] = 0
                else:
                    if use_first_moment:
                        state["exp_avg"] = state["exp_avg"].to(grad)
                    if factored:
                        state["row"] = state["row"].to(grad)
                        state["col"] = state["col"].to(grad)
                    else:
                        state["v"] = state["v"].to(grad)

                p_data_fp32 = p
                if p.dtype in {torch.float16, torch.bfloat16}:
                    p_data_fp32 = p_data_fp32.float()

                state["step"] += 1
                state["RMS"] = self._rms(p_data_fp32) # parameter scale, for relative step
                lr = self._get_lr(group, state)

                beta2t = 1.0 - state["step"] ** group["decay_rate"]   # beta2_hat_t = 1 - t^{-0.8}
                sq = grad ** 2 + group["eps"][0]                       # g^2 + eps1, floors V_hat

                if factored:
                    row, col = state["row"], state["col"]             # running row/col MEANS of g^2
                    row.mul_(beta2t).add_(sq.mean(dim=-1), alpha=1.0 - beta2t)
                    col.mul_(beta2t).add_(sq.mean(dim=-2), alpha=1.0 - beta2t)
                    update = self._approx_sq_grad(row, col)           # 1/sqrt(V_hat), rank-1
                    update.mul_(grad)                                 # U = G / sqrt(V_hat)
                else:
                    v = state["v"]
                    v.mul_(beta2t).add_(sq, alpha=1.0 - beta2t)
                    update = v.rsqrt().mul_(grad)

                # update clipping: cap RMS(U) at d, leave it alone if already below
                update.div_((self._rms(update) / group["clip_threshold"]).clamp_(min=1.0))
                update.mul_(lr)                                       # alpha_t * U_hat

                if use_first_moment:                                 # optional momentum on the clipped update
                    exp_avg = state["exp_avg"]
                    exp_avg.mul_(group["beta1"]).add_(update, alpha=1 - group["beta1"])
                    update = exp_avg

                p_data_fp32.add_(update, alpha=-1.0)                  # X <- X - alpha_t * U_hat
                if p.dtype in {torch.float16, torch.bfloat16}:
                    p.copy_(p_data_fp32)
        return loss
```

So the chain, start to finish: Adam's second-moment buffer is a full model-sized copy and that's what's capping model size, but the per-coordinate denominator it stores has row/column structure on a weight matrix, so I try to keep only a rank-1 summary and reconstruct the rest. The textbook rank-1 (truncated SVD) is wrong twice — it doesn't commute with the exponential smoothing of a running accumulator and it isn't nonnegative — so I switch to a nonnegative factorization under the I-divergence, and minimizing it at rank 1 hands back exactly the row sums and column sums of `V`, with reconstruction `V̂_{ij} = R_i C_j / Σ_k R_k`; because row/column sums are linear in `V`, smoothing and factoring commute and I can maintain just `R ∈ ℝⁿ` and `C ∈ ℝᵐ`, dropping the buffer from `O(nm)` to `O(n+m)`. Then I drop the *first*-moment buffer too (`β₁ = 0`), which exposes an instability — stale slow-decaying second moments produce larger-than-desired steps, visible as `RMS(U_t)` straying from 1 — and I fix it directly by clipping the actual update `U = G/√V̂` by its RMS (not the gradient, which adaptive scaling renders useless to clip). I attack the same stale-estimate problem at its root with an increasing decay `β̂₂_t = 1 − t^{-c}`, which decays fast early and slow late, removes bias correction for free as long as `β̂₂_1 = 0`, and needs `c ≤ 1` so old gradients eventually fade. Finally I make the step *relative* — scale it by `max(ε₂, RMS(X))` so each parameter moves a fixed fraction of its own size — and the resulting optimizer keeps Adam-style per-coordinate adaptivity while its extra memory is sublinear in the parameters.
