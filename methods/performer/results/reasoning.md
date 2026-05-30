Let me start from the thing that actually hurts. I have a Transformer and I want to run it on long sequences — protein chains of eight thousand residues, images flattened into twelve thousand pixels, whole books. And it falls over, not because the model is bad but because of one operation. Attention takes queries, keys, and values `Q, K, V ∈ R^{L×d}`, forms the score matrix `A = exp(QKᵀ/√d)`, normalizes each row by `D = diag(A 1_L)`, and returns `D⁻¹ A V`. The cost is entirely in `A`: it is `L × L`, so I spend `O(L²d)` time to build it and `O(L²)` memory to hold it. For `L` in the thousands that quadratic term is the whole story. Double the sequence, quadruple the bill. I want this linear in `L`.

So where does the `L²` actually come from? Not from anything fundamental about what attention *means* — it comes from the *order of operations*. I build the full matrix `A` and only then multiply by `V`. If I never had to instantiate `A`, the `L²` object would never exist. Let me stare at one row of the unnormalized output, `(A V)_i = Σ_j A(i,j) v_j = Σ_j exp(q_iᵀk_j) v_j`. The reason I can't factor anything out is that `exp(q_iᵀk_j)` is a function of `q_i` and `k_j` *jointly* — it's not a product of something-in-`i` times something-in-`j`. If instead the score were a plain dot product of per-token vectors, `A(i,j) = φ(q_i)ᵀφ(k_j)` for some map `φ : R^d → R^r`, then watch what happens:

    (A V)_i = Σ_j (φ(q_i)ᵀφ(k_j)) v_j = φ(q_i)ᵀ (Σ_j φ(k_j) v_jᵀ).

The inner sum `Σ_j φ(k_j) v_jᵀ` is an `r × d` matrix that *does not depend on `i`* — I compute it once, in `O(Lrd)`, and then every query just multiplies into it, another `O(Lrd)`. The `L × L` matrix is gone; I never form it. The same trick handles the normalizer: `D(i) = Σ_j A(i,j) = φ(q_i)ᵀ(Σ_j φ(k_j))`. In matrix form, if `Q'` and `K'` stack the feature rows `φ(q_i)ᵀ` and `φ(k_j)ᵀ`, the whole thing is

    Att̂ = D̂⁻¹ (Q'((K')ᵀ V)),    D̂ = diag(Q'((K')ᵀ 1_L)),

and the parentheses are the entire point: `(K')ᵀV` first (an `r × d` matrix), then `Q'` times that. Time `O(Lrd)`, memory `O(Lr + Ld + rd)`. Linear in `L`. So the question is not "how do I speed up attention" — it's "can I write `exp(q_iᵀk_j)` as a dot product `φ(q_i)ᵀφ(k_j)`?" If I can, I'm done with the complexity problem in one line.

Can I? `exp(xᵀy)` is not literally a finite dot product — it's a positive-definite kernel, and a kernel is an inner product in some feature space, but that space is infinite-dimensional. The Gaussian kernel `exp(-‖x-y‖²/2)` has the same issue. But there's a standard escape hatch: don't represent the kernel exactly, represent it *in expectation*. Rahimi and Recht's random features: a shift-invariant kernel is the Fourier transform of a probability density, so `K(x,y) = E_{ω∼p}[ζ_ω(x) \overline{ζ_ω(y)}]`. Draw `m` random frequencies `ω_1,…,ω_m`, form a finite feature map, and the dot product of those maps is an *unbiased* estimate of the kernel with variance falling like `1/m`. That's exactly the form I want — a dot product `φ(x)ᵀφ(y)` whose *mean* is the kernel. If I plug a random-feature `φ` into the reassociation above, I get an *unbiased estimator of attention* at linear cost. The bias question and the cost question separate cleanly: cost is handled by the dot-product structure, accuracy is handled by `m`.

Let me make the kernel connection concrete first. My score, dropping the `√d` by folding it into `q,k`, is the softmax kernel `SM(x,y) = exp(xᵀy)`. Pull the norms out:

    exp(xᵀy) = exp(-‖x‖²/2) · exp(‖x+y‖²/2) · exp(-‖y‖²/2),

since `‖x+y‖²/2 = (‖x‖² + 2xᵀy + ‖y‖²)/2`, so the middle factor carries `exp(xᵀy)` times `exp(‖x‖²/2)exp(‖y‖²/2)`, which the outer two factors cancel. Equivalently `SM(x,y) = exp(‖x‖²/2) K_gauss(x,y) exp(‖y‖²/2)` where `K_gauss` is the Gaussian kernel. So the softmax kernel is the Gaussian kernel reweighted by per-vector factors, and the Gaussian kernel is shift-invariant — random features apply directly.

The textbook random feature for the Gaussian kernel is trigonometric. The general template is

    φ(x) = (h(x)/√m)(f_1(ω_1ᵀx),…,f_1(ω_mᵀx),…,f_l(ω_1ᵀx),…,f_l(ω_mᵀx)),

with `ω_i ∼ N(0,I_d)`. Taking `l=2`, `f_1=sin`, `f_2=cos`, `h=1` gives the Gaussian kernel because `E_ω[cos(ωᵀ(x-y))] = exp(-‖x-y‖²/2)` (the real part of the characteristic function of a standard Gaussian). To get the softmax kernel I just absorb the norm reweighting into `h`: set `h(x) = exp(‖x‖²/2)`, keep `sin/cos`, and I have an unbiased estimator of `SM(x,y)`. Call it `SM̂_trig`. Drop it into the reassociation and I have linear attention that, in expectation, *is* softmax attention.

Let me actually try to run this in my head, because something feels off. The output row is `D̂⁻¹ (A V)_i`, and `D̂(i) = φ(q_i)ᵀ(Σ_j φ(k_j))`. The exact `D(i) = Σ_j exp(q_iᵀk_j)` is a sum of positive numbers — it's a positive normalizer, and `D⁻¹` makes each attention row a probability distribution over the value vectors. That positivity is not incidental; it's what "attention" *is*: a convex combination of values. Now my estimate `φ(q_i)ᵀφ(k_j)` uses `sin` and `cos`, which take values in `[-1,1]`. Individual estimated scores can be **negative**. And `D̂(i)`, a sum of these, can be negative or near zero. Then `D̂⁻¹` flips sign or explodes, and the "convex combination" is garbage. I'd expect this to show up as wildly unstable training — losses diverging, `NaN`s — and that is exactly the failure mode I'd predict for a feature map that doesn't respect the non-negativity of the scores.

And it's worse than just "sometimes negative." Where does it hurt most? The attention matrix has many small entries — most token pairs are low-relevance, `SM(x,y)` close to zero. An estimator with high *relative* variance near zero will turn those small positive numbers into noisy values straddling zero. Let me quantify the variance to see how bad. For the trigonometric estimator, with `z = x+y`, `Δ = x-y`, and using that the variance of an average of `m` i.i.d. terms is `1/m` times the single-sample variance,

    MSE(SM̂_trig) = (1/m) exp(‖x‖² + ‖y‖²) · Var(cos(ωᵀΔ)).

I need `Var(cos(ωᵀΔ))` for `ω ∼ N(0,I)`. Using `cos²= (1+cos2)/2` and `E[cos(ωᵀΔ)] = exp(-‖Δ‖²/2)`: `E[cos²(ωᵀΔ)] = ½(1 + E[cos(ωᵀ(2Δ))]) = ½(1 + exp(-2‖Δ‖²))`, so `Var = ½(1+exp(-2‖Δ‖²)) - exp(-‖Δ‖²) = ½(1 - exp(-‖Δ‖²))²`. Therefore

    MSE(SM̂_trig) = (1/2m) exp(‖x‖²+‖y‖²)(1 - exp(-‖Δ‖²))² = (1/2m) exp(‖z‖²) SM(x,y)⁻² (1 - exp(-‖Δ‖²))²,

where the last step uses `exp(‖x‖²+‖y‖²) = exp(‖z‖²)·exp(-2xᵀy) = exp(‖z‖²) SM(x,y)⁻²`. There it is: the `SM⁻²` factor. As `SM(x,y) → 0`, the MSE goes to **infinity**. The estimator is *least* accurate precisely on the small entries, which are the majority, and small errors there get amplified by `D̂⁻¹`. So trigonometric features aren't just occasionally negative — they have provably exploding relative error in the regime that dominates the matrix. This is the wall. The reassociation trick is correct and the cost is linear, but `sin/cos` features make the thing untrainable.

I need a feature map that is (a) unbiased for the softmax kernel, (b) **non-negative**, so `D̂` stays a positive normalizer, and (c) low-variance *as the kernel goes to zero*, the opposite of trig. Non-negativity says: no `sin`, no `cos`. I want something like `exp`, which is always positive. Can I write `exp(xᵀy)` as an expectation of a product of `exp`s? Let me go back to completing the square, but this time aim for the exponential form directly.

I have `exp(xᵀy) = exp(-‖x‖²/2)·exp(‖x+y‖²/2)·exp(-‖y‖²/2)`. The awkward factor is `exp(‖x+y‖²/2)`. I want to turn it into an expectation over a Gaussian `ω`. The trick is the Gaussian normalization identity: for any fixed `c ∈ R^d`,

    (2π)^{-d/2} ∫ exp(-‖ω - c‖²/2) dω = 1,

because it's just a shifted unit Gaussian integrating to one. So `exp(‖x+y‖²/2)` equals itself times that integral, with `c = x+y`:

    exp(‖x+y‖²/2) = (2π)^{-d/2} ∫ exp(-‖ω-(x+y)‖²/2) · exp(‖x+y‖²/2) dω.

Expand the exponent: `-‖ω-(x+y)‖²/2 + ‖x+y‖²/2 = -‖ω‖²/2 + ωᵀ(x+y) - ‖x+y‖²/2 + ‖x+y‖²/2 = -‖ω‖²/2 + ωᵀ(x+y)`. The `‖x+y‖²` terms cancel exactly. So

    exp(‖x+y‖²/2) = (2π)^{-d/2} ∫ exp(-‖ω‖²/2) exp(ωᵀx) exp(ωᵀy) dω = E_{ω∼N(0,I)}[exp(ωᵀx) exp(ωᵀy)].

That's clean. Substitute back:

    SM(x,y) = exp(xᵀy) = E_{ω∼N(0,I)}[ exp(ωᵀx - ‖x‖²/2) · exp(ωᵀy - ‖y‖²/2) ].

This is exactly the unbiased-dot-product form I wanted, and crucially the two factors are *separable* — one depends only on `x`, the other only on `y` — and each is an exponential, hence strictly positive. So the feature map is

    φ⁺(u) = (exp(-‖u‖²/2)/√m)(exp(ω_1ᵀu), …, exp(ω_mᵀu)),    ω_i ∼ N(0,I),

and `φ⁺(x)ᵀφ⁺(y) = (1/m)Σ_i exp(ω_iᵀx - ‖x‖²/2)exp(ω_iᵀy - ‖y‖²/2)` is an unbiased, non-negative estimator of `SM(x,y)`. Call it `SM̂⁺`. Every entry it produces is positive, so `D̂` is a genuine positive normalizer and `D̂⁻¹` never blows up. Property (a) and (b) are satisfied by construction. Now I need to check (c), the variance near zero — the place trig features died.

Same computation as before. `MSE(SM̂⁺) = (1/m) exp(-(‖x‖²+‖y‖²)) Var(exp(ωᵀz))`, `z = x+y`, because the feature is `exp(-‖x‖²/2)exp(ωᵀx)` and likewise for `y`, so the product's randomness is `exp(ωᵀz)` scaled by `exp(-(‖x‖²+‖y‖²)/2)`, squared in the variance. I need `Var(exp(ωᵀz)) = E[exp(2ωᵀz)] - E[exp(ωᵀz)]²`. The moment generating function of `ωᵀz ∼ N(0,‖z‖²)` is `E[exp(t ωᵀz)] = exp(t²‖z‖²/2)`, so `E[exp(ωᵀz)] = exp(‖z‖²/2)` and `E[exp(2ωᵀz)] = exp(2‖z‖²)`. Thus `Var = exp(2‖z‖²) - exp(‖z‖²) = exp(‖z‖²)(exp(‖z‖²)-1)`. Putting it together,

    MSE(SM̂⁺) = (1/m) exp(-(‖x‖²+‖y‖²)) exp(‖z‖²)(exp(‖z‖²) - 1).

Now `exp(-(‖x‖²+‖y‖²))exp(‖z‖²) = exp(2xᵀy) = SM(x,y)²`, so

    MSE(SM̂⁺) = (1/m) exp(‖z‖²) SM(x,y)² (1 - exp(-‖z‖²)).

Compare to the trig MSE, which had `SM⁻²` out front. This one has `SM²` out front. As `SM(x,y) → 0`, `MSE(SM̂⁺) → 0`. The exponential estimator is *most* accurate exactly where the trigonometric one was *least* accurate — on the small entries that dominate the matrix and that get amplified by the normalizer. That asymmetry is the whole ballgame. Trig MSE blows up as `SM→0`; positive MSE vanishes as `SM→0`. So positive features aren't a cosmetic fix for the sign — they fix the variance in the regime that was killing training.

Can I push the variance down further for free? The estimator uses `exp(ωᵀz)`. Since `ω` and `-ω` have the same distribution (Gaussian is symmetric), I could symmetrize: use both `exp(ωᵀz)` and `exp(-ωᵀz)`, i.e. set `l=2` with `f_1=exp(u)`, `f_2=exp(-u)`, and `h(u) = exp(-‖u‖²/2)/√2`. The estimator becomes proportional to `cosh(ωᵀz)`, still positive, still unbiased (the cross terms in expectation give the same `SM`). Its MSE: averaging `½(e^{ωᵀz} + e^{-ωᵀz})` and using that the two have equal variance and a *negative* covariance, the variance of the average is below the variance of either alone. Working it out, `MSE(SM̂^{hyp+}) = ½(1 - exp(-‖z‖²)) MSE(SM̂⁺)`. Since `½(1-exp(-‖z‖²)) ≤ ½`, this is strictly better than the plain positive estimator — in fact better than `SM̂⁺` with twice as many features. A free factor from a symmetry I already had.

Now the second lever. I'm drawing `m` independent Gaussian directions `ω_i`. Independence is wasteful — independent random directions in high dimensions cluster and waste samples. If I instead force `ω_1,…,ω_m` to be *exactly orthogonal*, I cover the sphere more evenly. And I can do this without breaking unbiasedness: for an isotropic distribution like `N(0,I)`, I can sample a Gaussian block and run Gram-Schmidt, which keeps each `ω_i`'s marginal distribution the same (still `N(0,I)` directionally with chi-distributed length) while making them pairwise orthogonal. The marginals are unchanged, so each individual feature is still unbiased, so the average is still unbiased. The only constraint is `m ≤ d` (can't have more than `d` orthogonal vectors in `R^d`). Does it actually reduce variance, and by how much?

Let me set up the general object, because the same argument will cover softmax, its symmetrized version, and the Gaussian kernel at once. All of these have the form

    F(z) = E_{ω∼Ω}[g(ωᵀz)],

where `Ω` is isotropic and `g` is an entire function with **non-negative** power-series coefficients — for softmax, `Ω = N(0,I)`, `g = exp`, `g(u) = Σ_s a_s u^s` with `a_s = 1/s! ≥ 0`, and `SM(x,y) = exp(-(‖x‖²+‖y‖²)/2) F(x+y)`. The non-negativity of the `a_s` is going to matter; it's the algebraic fingerprint of positivity. Call such an `F` a well-behaved (entire, non-negative-coefficient) kernel function. The independent estimator is `F̂^iid = (1/m)Σ_i g(ω_i^{iid ᵀ}z)`, the orthogonal one `F̂^ort = (1/m)Σ_i g(ω_i^{ort ᵀ}z)`.

Both are unbiased, so MSE equals variance, and the difference is entirely in the cross terms:

    MSE(F̂^iid) - MSE(F̂^ort) = (1 - 1/m)(E[X_1^{iid}X_2^{iid}] - E[X_1^{ort}X_2^{ort}]),

where `X_i = g(ω_iᵀz)`, because the diagonal terms `E[X_i²]` are equal (same marginals) and there are `m(m-1)` ordered cross pairs, each scaled by `1/m²`, giving the `(1-1/m)` prefactor. So I need the sign and size of `E[X_1^{iid}X_2^{iid}] - E[X_1^{ort}X_2^{ort}]`. Expand each `X_i` in the power series: `X_i = Σ_s a_s (ω_iᵀz)^s`. The cross expectation is a sum over monomials `E[(ω_1ᵀz)^{d_1}(ω_2ᵀz)^{d_2}]` with non-negative coefficients `a_{d_1} a_{d_2}`. So it suffices to compare these mixed moments under independence versus orthogonality.

Here's the geometric fact that does the work. Write `ω_i = ‖ω_i‖ · û_i`, where `û_i` is the direction. The lengths `‖ω_i‖` are drawn the same way in both schemes (orthogonalization only touches directions), and they're independent of the directions. For the directions: a set of `m` orthonormal directions is a random rotation of the canonical basis `e_1,…,e_m`. Rotating the basis and then projecting `z` onto it is the same in distribution as fixing the basis and randomly rotating `z` — and a randomly rotated `z` is distributed as `‖z‖ · g/‖g‖` for a fresh Gaussian `g`. So

    E[(ω_1^{ort ᵀ}z)^{d_1}⋯(ω_m^{ort ᵀ}z)^{d_m}] = (∏_i E[‖ω_i‖^{d_i}]) ‖z‖^{Σd_i} E[ g_1^{d_1}⋯g_m^{d_m} / ‖g‖^{Σd_i} ],

while for the independent case the directions are independent, so the last expectation *factorizes* into `∏_i E[g_i^{d_i}/‖g‖^{d_i}]`. The difference between the joint moment and the product of marginal moments is exactly an orthogonality effect. Using a normalization lemma — `E[g_1^{k_1}⋯g_s^{k_s}/‖g‖^{k_1+…+k_s}] = ∏_i E[g_i^{k_i}] / E[‖g‖^{k_1+…+k_s}]` (proved by introducing an independent Gaussian length and using that `g/‖g‖·‖g̃‖ ∼ g`) — both sides reduce to ratios of `χ`-distribution moments, and the orthogonal-minus-independent comparison comes down to a single ratio

    τ(d_1,…,d_m) = ∏_i μ_d(d_i) / μ_d(Σ_i d_i),    μ_d(j) = j-th moment of the χ_d distribution.

Two facts pin it down. First, any monomial with an *odd* `d_i` contributes zero (odd Gaussian moments vanish), and any monomial supported on a *single* index contributes zero (then it's the same in both schemes). So only monomials with at least two even, positive exponents survive. Second, for those, with `μ_d(j) = 2^{j/2} Γ((d+j)/2)/Γ(d/2)`, a short calculation shows `τ` is maximized at the lowest nontrivial monomial `d_i = d_j = 2`, all others zero, where it equals exactly `d/(d+2) < 1`. So `τ ≤ d/(d+2)` on every surviving monomial, meaning `1 - τ ≥ 2/(d+2) > 0`. Because all the coefficients `a_s a_t` are non-negative (this is where positivity is load-bearing — if coefficients could be negative the surviving terms could cancel or flip sign), every surviving term in the difference is `≥ 0`, so the orthogonal estimator has *smaller or equal* second moment, hence smaller MSE. Keeping the leading `(d_i=d_j=2)` term as a lower bound gives the explicit gap:

    MSE(F̂^ort) ≤ MSE(F̂^iid) - (1 - 1/m)(2/(d+2))(F(z) - a_0)²,

where `a_0 = g(0)` is the constant term (for `exp`, `a_0 = 1`). Translating back to the softmax kernel via `F(z) = SM(x,y)exp((‖x‖²+‖y‖²)/2)` and the `exp(-(‖x‖²+‖y‖²))` prefactor, the gap is `(2(m-1)/(m(d+2)))(SM(x,y) - exp(-(‖x‖²+‖y‖²)/2))²`. And note: this holds for **every** `d > 0`, not just asymptotically — orthogonality helps even in low dimension, which earlier orthogonal-feature analyses couldn't claim. The same power-series machinery, applied to the moment generating function rather than the second moment, gives exponentially small tail bounds for the orthogonal estimator that are strictly tighter than the independent ones — same mechanism, the `θ⁴ m(m-1)/(4(d+2))` correction subtracted off the iid MGF bound.

So the recipe is set: positive (`R+`) features for non-negativity and vanishing variance near zero, orthogonal (`O`) sampling for a further provable variance cut. Together — fast attention via positive orthogonal random features. Stack them into the reassociation and I have an unbiased, stable, linear-time approximation of full softmax attention with `m ≤ d` random features.

One more variant worth deriving, because the unbounded `exp` makes me slightly nervous about numerical range. What if I sample `ω` not from `N(0,I)` but uniformly from the sphere of radius `√d` — i.e. replace `ω` by `√d · ω/‖ω‖`? Call the resulting kernel `SMREG`, the regularized softmax kernel. It's the same construction with a different isotropic `Ω`, so all the orthogonality results carry over. How close is `SMREG` to `SM`? Expand `SM` and `SMREG` in the Taylor series of the exponential. For `SM`, `F(z) = Σ_k (1/(2k)!)‖z‖^{2k} d^k E[(ω̂ᵀe_1)^{2k}]` after using isotropy to kill odd terms and writing `ω = ‖ω‖ω̂`. The angular moment `A(2k,d) = E[(ω̂ᵀe_1)^{2k}] = (2k-1)!! / [(d+2k-2)(d+2k-4)⋯d]`, which I can get by integrating `sin^{d-2}θ` against `cos^{2k}θ` and a partial-integration recursion `F(k,d) = (k-1)/(d+1) F(k-2,d+2)`. The ratio of the `k`-th terms is `f(k,d) = d^k/[(d+2k-2)⋯d] ≤ 1` for `k ≥ 1`, with equality at `k=0`. So `SMREG ≤ SM` term by term — `SMREG` is a universal lower bound on the softmax kernel. And bounding the tail with a Poisson concentration argument (the partial sums of `w^k/k!` are a Poisson CDF, `w = ‖z‖²/2`), the ratio satisfies

    SMREG(x,y)/SM(x,y) ≥ 1 - 2/d^{1/3} + o(1/d^{1/3}),

so for the dimensions I actually use, `SMREG` tracks `SM` to within a vanishing relative gap while being bounded and a clean lower bound. A safe drop-in proxy.

Now the causal case, because half of what I care about is autoregressive. There, `A` is masked to its lower triangle: token `i` attends only to `j ≤ i`. The reassociation I love relies on summing `Σ_j φ(k_j)v_jᵀ` over *all* `j`; with masking I instead need, for each `i`, the *prefix* sum over `j ≤ i`. So define the running outer-product accumulator `G_i = Σ_{j≤i} φ(k_j) (v_j, 1)ᵀ`, an `r × (d+1)` matrix (I append a `1` so the same machinery computes the normalizer `D̂` alongside `AV`). Then the `i`-th output row is `φ(q_i)ᵀ G_i`. Computing `G_i` from `G_{i-1}` is a single addition of one outer product, so the whole pass is a prefix-sum (cumulative sum) over the sequence: `O(L)` total work, `O(d r)` state, and `O(log L)` depth if I parallelize the scan. I never form the `L × L` mask; causality is just "use the running sum so far." Linear, and exactly the same features.

A last generalization falls out for free. Nothing in the reassociation needed the feature map to come from the softmax kernel specifically — it only needed `φ(x) ≥ 0` so the normalizer stays positive, and some `φ` that produces a sensible similarity. So I can take `φ(x) = f(ω_iᵀx) + ε` for any non-negative `f` and treat the choice of `f` as a hyperparameter — a *generalized* kernelizable attention. `f = ReLU` is a natural one: cheap, non-negative, and empirically the best-behaved on long sequences (no normalization fuss, no risk of `exp` overflow). The softmax estimator is then one option (`f = exp` with the norm-correction) and ReLU another, both at linear cost.

Let me write the code, grounded in how this is actually implemented. The pieces are: drawing an orthogonal Gaussian projection block; the positive softmax feature map (with a log-sum-exp-style max subtraction for numerical safety, and a small `ε`); the ReLU generalized feature map; and the reassociated linear attention, non-causal as a single matrix product and causal as a chunked prefix-sum.

```python
import math
from functools import partial
import torch
import torch.nn as nn
from einops import repeat

# --- Orthogonal random projection (the "O" of FAVOR+) ---------------------
# Build an m x d matrix whose rows are exactly orthogonal within each d x d
# block (QR of a Gaussian block: marginals stay Gaussian, rows orthogonal),
# then rescale each row's length by an independent chi-distributed norm so the
# marginal of every row is exactly N(0, I_d). Unbiasedness is preserved.
def orthogonal_block(d, device=None):
    g = torch.randn((d, d), device=device)
    q, _ = torch.linalg.qr(g)            # columns orthonormal
    return q.t()                         # rows orthonormal

def gaussian_orthogonal_matrix(m, d, scaling=0, device=None):
    blocks = [orthogonal_block(d, device) for _ in range(m // d)]
    rem = m - (m // d) * d
    if rem > 0:
        blocks.append(orthogonal_block(d, device)[:rem])
    W = torch.cat(blocks)                # [m, d], orthogonal within blocks
    if scaling == 0:                     # row norms ~ chi_d, i.e. ||N(0,I_d)||
        mult = torch.randn((m, d), device=device).norm(dim=1)
    else:                                # all rows length sqrt(d): the SMREG sphere
        mult = math.sqrt(d) * torch.ones((m,), device=device)
    return torch.diag(mult) @ W

# --- Positive softmax features (the "R+"): unbiased, non-negative ----------
# phi(u) = exp(-||u||^2/2)/sqrt(m) * exp(W u), with W u = data_dash.
# d^{-1/4} per-vector scaling folds the 1/sqrt(d) of softmax into q,k.
# Subtracting the running max keeps exp() in range (it cancels in D^{-1} A V).
def softmax_kernel(data, projection, is_query, eps=1e-4):
    b, h, *_ = data.shape
    norm = data.shape[-1] ** -0.25
    ratio = projection.shape[0] ** -0.5
    W = repeat(projection, 'm d -> b h m d', b=b, h=h).type_as(data)
    data_dash = torch.einsum('...id,...md->...im', norm * data, W)   # W u
    diag = (data ** 2).sum(dim=-1, keepdim=True) * (norm ** 2) / 2.0  # ||u||^2/2
    if is_query:                          # stabilize per-query
        data_dash = ratio * (torch.exp(data_dash - diag
                     - data_dash.amax(dim=-1, keepdim=True).detach()) + eps)
    else:                                 # stabilize over all keys
        data_dash = ratio * (torch.exp(data_dash - diag
                     - data_dash.amax(dim=(-1, -2), keepdim=True).detach()) + eps)
    return data_dash.type_as(data)

# --- Generalized features: phi(u) = f(W u) + eps, e.g. f = ReLU -----------
def generalized_kernel(data, projection, kernel_fn=nn.ReLU(), eps=1e-3):
    b, h, *_ = data.shape
    norm = data.shape[-1] ** -0.25
    W = repeat(projection, 'm d -> b h m d', b=b, h=h).type_as(data)
    data_dash = torch.einsum('...id,...md->...im', norm * data, W)
    return (kernel_fn(data_dash) + eps).type_as(data)

# --- The reassociation: never build the L x L matrix ----------------------
# Non-causal: (K'^T V) first (r x d), then Q' times it.  D^{-1} from K'^T 1_L.
def linear_attention(q, k, v):
    k_sum = k.sum(dim=-2)                                   # K'^T 1_L
    D_inv = 1.0 / torch.einsum('...id,...d->...i', q, k_sum)
    context = torch.einsum('...id,...ie->...de', k, v)      # K'^T V  (r x d)
    return torch.einsum('...de,...id,...i->...ie', context, q, D_inv)

# Causal: prefix-sum over the sequence so token i sees only j <= i.
def causal_linear_attention(q, k, v, chunk=128, eps=1e-6):
    last_k, last_ctx, outs = 0, 0, []
    for q_, k_, v_ in zip(*(t.chunk(chunk, dim=-2) for t in (q, k, v))):
        k_cumsum = last_k + k_.cumsum(dim=-2)               # running sum of phi(k)
        D_inv = 1.0 / torch.einsum('...id,...id->...i', q_, k_cumsum + eps)
        ctx = torch.einsum('...id,...ie->...ide', k_, v_)   # outer products
        ctx_cumsum = last_ctx + ctx.cumsum(dim=-3)          # running sum G_i
        outs.append(torch.einsum('...ide,...id,...i->...ie', ctx_cumsum, q_, D_inv))
        last_k, last_ctx = k_cumsum[..., -1:, :], ctx_cumsum[..., -1:, :, :]
    return torch.cat(outs, dim=-2)

class FastAttention(nn.Module):
    def __init__(self, dim_heads, nb_features=None, causal=False,
                 generalized=False, kernel_fn=nn.ReLU(), ortho_scaling=0):
        super().__init__()
        nb_features = nb_features or int(dim_heads * math.log(dim_heads))  # m = Theta(d log d)
        self.create_proj = partial(gaussian_orthogonal_matrix,
                                   m=nb_features, d=dim_heads, scaling=ortho_scaling)
        self.register_buffer('projection', self.create_proj())
        self.causal, self.generalized, self.kernel_fn = causal, generalized, kernel_fn

    @torch.no_grad()
    def redraw(self, device):            # periodically resample to avoid unlucky W
        self.projection.copy_(self.create_proj(device=device))

    def forward(self, q, k, v):
        if self.generalized:
            feat = partial(generalized_kernel, projection=self.projection,
                           kernel_fn=self.kernel_fn)
            q, k = feat(q), feat(k)
        else:
            feat = partial(softmax_kernel, projection=self.projection)
            q, k = feat(q, is_query=True), feat(k, is_query=False)
        attn = causal_linear_attention if self.causal else linear_attention
        return attn(q, k, v)
```

The causal chain in one breath: softmax attention is quadratic only because I build `A` before multiplying by `V`; the fix is to write `exp(q_iᵀk_j) = φ(q_i)ᵀφ(k_j)` so I can reassociate and never form `A`; an unbiased such `φ` comes from random features, but the textbook trigonometric map can go negative and has variance exploding as the kernel goes to zero, which destroys the normalizer and training; completing the square in the Gaussian integral turns `exp(xᵀy)` into an expectation of a product of `exp`s, giving a strictly positive feature map whose variance instead *vanishes* as the kernel goes to zero; sampling those random directions orthogonally cuts the variance further by a provable `2/(d+2)` gap that holds at every dimension precisely because the feature coefficients are non-negative; and the same reassociation, run as a prefix-sum, handles the causal mask in linear time — so full softmax attention becomes an unbiased, stable, `O(Lmd)` computation.
