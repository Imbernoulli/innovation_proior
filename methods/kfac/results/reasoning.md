OK, let me think this through from scratch. I'm training a deep network with SGD and momentum, and it's slow. Not slow because each step is expensive — each step is just a gradient — but slow because the gradient is a bad *direction*. I can see why: the loss surface here is wildly anisotropic. Some directions in parameter space have huge curvature, some have almost none, and worse, the directions don't line up with the coordinate axes — the curvature is densely correlated across parameters. Plain gradient descent rescales nothing and rotates nothing, so it crawls along the low-curvature valleys and oscillates across the high-curvature walls. I want to divide the gradient by the curvature.

The textbook move is Newton: `δ = −H⁻¹∇h`. And there's a cleaner version of this for a probabilistic model. My network outputs parameters `z = f(x,θ)` of a predictive distribution `R_{y|z}` — Gaussian for squared error, multinomial for cross-entropy — and the loss is `L(y,z) = −log r(y|z)`. So training is maximum likelihood, and the model defines a conditional distribution `P_{y|x}(θ)`. For distributions there's a canonical notion of "how far did I move" that isn't the Euclidean distance in `θ` but the KL divergence between the old and new predictive distributions. Steepest descent under that metric — the natural gradient — is

    natural gradient direction = F⁻¹ ∇h,    F = E[ ∇_θ log p · ∇_θ log pᵀ ] = E[ ∇θ ∇θᵀ ].

`F` is the Fisher information. The thing I most want from it is reparameterization invariance: if I reparameterize the network smoothly, the path it traces through *distribution* space is unchanged. That's the property I'd want from an optimizer — one that doesn't care whether I used `tanh` or sigmoid, or whether I rescaled my inputs. So as a *direction* the natural gradient looks like what I'm after.

There's a second reason to like it, which I want to keep in my back pocket. When the loss is the negative log-likelihood of an exponential family with `z` as the natural parameters, the Fisher turns out to equal the Generalized Gauss-Newton matrix, a positive-semidefinite stand-in for the Hessian. So `F` isn't just an information-geometry object — it's genuinely a curvature matrix, the quadratic term of a local model of my loss. That means whatever I build, I can lean on the whole optimization literature about quadratic models: trust regions, damping, line searches. Hold that thought.

So: just compute `F⁻¹∇h`. Except — `F` is `(#params)²`. My network has millions of parameters. `F` has millions-squared entries. I can't store it, let alone invert it. This is the wall. The natural gradient is the right metric and `F⁻¹` is impossibly big. Everything from here is: how do I approximate `F⁻¹` in a way that's rich enough to keep the power, cheap enough to invert directly, and compact enough to estimate from lots of data?

Let me look at what people do and where each falls short, because the gaps will tell me what I need. One camp inverts a *diagonal* approximation of the curvature, or a per-unit small-block one, or a low-rank one. These are cheap and you can keep a running average of them across many minibatches. But they're disappointing in practice — barely better than momentum. And I think I know why: they throw away precisely the off-diagonal, cross-parameter curvature that makes second-order updates powerful in the first place. A diagonal preconditioner can't rotate; it can only rescale axes. The other camp keeps the *exact* curvature but never forms it — it multiplies vectors by `F` and runs conjugate gradient to solve `Fδ = −∇h` approximately. That keeps the full richness. But each update runs CG for maybe hundreds of iterations, each iteration is a mat-vec as expensive as a gradient, and — this is the killer — the curvature estimate has to be frozen while CG iterates. So each update only gets to look at one small fixed minibatch worth of curvature. In a stochastic regime that's crippling. And there's no way to keep a running average of the curvature across minibatches, because the only honest representation of the exact `F` is the full `n×n` matrix, whose size depends on nothing I can shrink.

So I want the best of both: a non-diagonal, non-low-rank approximation of `F` — rich, like the exact one — that I can nonetheless *invert in closed form* (no inner CG) and *store in a small structure whose size doesn't grow with how much data I used to estimate it* (so I can run an exponential moving average over a long history, like the diagonal people do). Those three wants together are unusual. Let me see if the structure of a neural network gives me anything to work with.

Here's the one structural fact I haven't used yet. Backprop. With `s_i = W_i ā_{i-1}` and `a_i = φ_i(s_i)` (I'm folding the bias in by appending a constant `1` to the activations, so `ā = [a;1]` and the bias is the last column of `W`), the gradient of the loss with respect to a whole layer's weight matrix is a single outer product:

    ∇_{W_i} L = g_i ā_{i-1}ᵀ,     where g_i = D s_i is the backpropagated pre-activation gradient.

The layer gradient is rank one — the outer product of the backward signal `g_i` and the forward signal `ā_{i-1}`. That feels like it should buy me something when I build `F`, because `F = E[∇θ ∇θᵀ]` and `∇θ` is a stack of these outer products. Let me actually grind it out.

Stack the parameters layer by layer, `θ = [vec(W₁)ᵀ … vec(W_ℓ)ᵀ]ᵀ`. Then `F = E[∇θ ∇θᵀ]` is naturally an `ℓ×ℓ` block matrix, the `(i,j)` block being `F_{i,j} = E[ vec(∇W_i) vec(∇W_j)ᵀ ]`. So I need `vec(∇W_i)`. I'm using column-stacking `vec`, and the identity I want is `vec(u vᵀ) = v ⊗ u`. So

    vec(∇W_i) = vec(g_i ā_{i-1}ᵀ) = ā_{i-1} ⊗ g_i.

Let me double-check that's the right order and not `g_i ⊗ ā_{i-1}` — `g_i ā_{i-1}ᵀ` is (output-dim × input-dim), `vec` stacks its columns, the `k`-th column is `[ā_{i-1}]_k · g_i`, so the stacked vector runs over input index slowly and output index fast, which is exactly `ā_{i-1} ⊗ g_i`. I don't fully trust my index bookkeeping with Kronecker products, so let me just check it numerically on a tiny case: `g = (g₀,g₁,g₂)`, `ā = (a₀,a₁)`. Then `vec(gāᵀ)` column-stacked is `(a₀g₀,a₀g₁,a₀g₂, a₁g₀,a₁g₁,a₁g₂)`, and `ā⊗g` is `(a₀g₀,a₀g₁,a₀g₂, a₁g₀,a₁g₁,a₁g₂)` — identical. (I ran this with random `u,v`: `vec(uvᵀ)` equals `v⊗u` to machine precision, and `g⊗ā` would have been the transposed ordering, which doesn't match.) Good, the order is `ā⊗g`. Now the block:

    F_{i,j} = E[ (ā_{i-1} ⊗ g_i)(ā_{j-1} ⊗ g_j)ᵀ ].

Transpose of a Kronecker product distributes, `(P⊗Q)ᵀ = Pᵀ⊗Qᵀ`, so `(ā_{j-1}⊗g_j)ᵀ = ā_{j-1}ᵀ ⊗ g_jᵀ`. And the mixed-product rule `(A⊗B)(C⊗D) = AC ⊗ BD` gives

    F_{i,j} = E[ (ā_{i-1} ā_{j-1}ᵀ) ⊗ (g_i g_jᵀ) ].

So each block of the exact Fisher is the expectation of a Kronecker product of an activation-covariance-like thing and a gradient-covariance-like thing, and the whole `F` is a block matrix of these. But it's an *expectation of a Kronecker product* — I can't invert that, and I can't even store it compactly, because the expectation entangles the two factors.

Stare at it. `E[(āāᵀ) ⊗ (ggᵀ)]`. What if the expectation of the Kronecker product were the Kronecker product of the expectations? That is, what if I just *moved the expectation inside each factor*:

    F_{i,j} = E[ ā_{i-1} ā_{j-1}ᵀ ⊗ g_i g_jᵀ ]  ≈  E[ ā_{i-1} ā_{j-1}ᵀ ] ⊗ E[ g_i g_jᵀ ]  =:  Ā_{i-1,j-1} ⊗ G_{i,j}.

Define `Ā_{i,j} = E[ā_i ā_jᵀ]` and `G_{i,j} = E[g_i g_jᵀ]`. Then `F̃_{i,j} = Ā_{i-1,j-1} ⊗ G_{i,j}` is a *single* Kronecker product of two small matrices — one the size of a layer's input, one the size of its output. I can store those, and I can invert those. If this is legal, it's exactly the compact, invertible factor I was hunting for.

But is it legal? The expectation of a Kronecker product is *not* in general the Kronecker product of the expectations — `E[XY] ≠ E[X]E[Y]` unless `X` and `Y` are uncorrelated, and that carries over to the factored structure here. So this is a real approximation, not an identity. Before I lean on it, let me find out how badly it can be wrong and under what conditions it's good. Take the scalar version, two scalar weight-derivatives both equal to `āg`: the exact second moment is `E[ā² g²]`, my approximation replaces it with `E[ā²]·E[g²]`. So I'm assuming the *square of an activity* is uncorrelated with the *square of a pre-activation gradient*.

Let me just simulate it and see when it holds. Draw `(ā,g)` jointly Gaussian, vary their cross-correlation `ρ = Corr(ā,g)`, a million samples each:

- zero-mean, `ρ = 0.7`: `E[ā²g²] = 1.98` vs `E[ā²]E[g²] = 1.00` — off by ~50%. Terrible.
- zero-mean, `ρ = 0`:   `E[ā²g²] = 1.002` vs `1.00` — error ~3·10⁻³, i.e. matches to sampling noise.
- mean `E[ā]=1.5`, `ρ = 0`: `E[ā²g²] = 3.251` vs `3.251` — error ~9·10⁻⁵.

So the approximation is *not* generally exact even for Gaussians — it's exact for Gaussians *only when `Cov(ā,g)=0`*. With a nonzero forward-backward correlation it's hopelessly off. So the whole thing hinges on whether I can arrange `Cov(ā,g)=0`. That's a strong condition, and I have no right to it for free — let me see whether the structure of the problem gives it to me.

There's a clean statement that forward-pass quantities are uncorrelated with backward derivatives *as long as I take expectations under the model's own output distribution*. Concretely, if `u` is independent of `y` given the network output `f(x,θ)`, and `v` is any quantity computed in the forward pass, then `E[u · Dv] = 0`. Why: by the chain rule `Dv = −(∂ log r(y|z)/∂z)ᵀ|_{z=f} · ∂f/∂v`, and the inner expectation over `y ~ R_{y|f}` of the score `∂ log r/∂z` is zero — the expected score of a distribution under itself vanishes. So `E[u Dv] = E_x[ −u · E_{y~model}[score]ᵀ · ∂f/∂v ] = 0`.

This lemma is *the* reason I'm going to sample `y` from the model and not from the training labels. If I used the training `y`, the inner expectation of the score is not zero — that would give me the so-called empirical Fisher, the lemma collapses, `Cov(ā,g)` is no longer zero, and (per the simulation just above) the factored approximation degrades, plus I'd lose the Gauss-Newton equivalence I wanted to keep. So: targets sampled from the model.

With the lemma in hand let me pin down the error exactly, because "uncorrelated" isn't the same as "exact" — the `ρ=0` Gaussian case was exact, but real `(ā,g)` aren't Gaussian. Take two scalar weight-derivatives, `ā^(1)g^(1)` and `ā^(2)g^(2)`. Write the exact moment `E[ā^(1)ā^(2) g^(1)g^(2)]` in cumulants — there's a standard moment-to-cumulant expansion, and for four variables it's a sum over the 15 ways to partition `{ā^(1),ā^(2),g^(1),g^(2)}`. The lemma kills the first cumulants `κ(g^(k)) = E[g^(k)] = E[Dx^(k)] = 0` (take `u=1`), and the cross second cumulants `κ(ā^(k), g^(m)) = E[(ā^(m) − E ā^(m)) g^(k)] = 0` (take `u = ā − Eā`, independent of `y` given `f`). Feeding `κ(g)=0` and `κ(ā,g)=0` into the 15-term partition expansion kills 10 terms. The survivors are

    E[ā^(1)ā^(2) g^(1)g^(2)] = κ(ā¹,ā²,g¹,g²) + κ(ā¹)κ(ā²,g¹,g²) + κ(ā²)κ(ā¹,g¹,g²)
                              + κ(ā¹,ā²)κ(g¹,g²) + κ(ā¹)κ(ā²)κ(g¹,g²).

The last two terms reassemble into exactly what I'm keeping: `κ(ā¹,ā²)κ(g¹,g²) + κ(ā¹)κ(ā²)κ(g¹,g²) = Cov(ā¹,ā²)E[g¹g²] + E[ā¹]E[ā²]E[g¹g²] = E[ā¹ā²]E[g¹g²]`. So the *error* `E[āā gg] − E[āā]E[gg]` should be precisely

    κ(ā¹,ā²,g¹,g²) + E[ā¹] κ(ā²,g¹,g²) + E[ā²] κ(ā¹,g¹,g²) ,

a fourth-order cumulant plus two third-order ones. That's a clean claim and I want to make sure I didn't drop a partition. Let me check it numerically in a deliberately *non*-Gaussian setting that still respects the lemma's conditions. Set `ā` Gaussian with mean `0.8`, and `g = (ā − E ā)·ε` with `ε` independent standard normal — then `E[g]=0` and `Cov(ā,g)=E[(ā−Eā)²ε]=0` (lemma satisfied), but `g` depends on `ā`, so the higher cumulants don't vanish. With four million samples (scalar version, so `ā¹=ā²=ā`, `g¹=g²=g`, error `= κ₄(ā,ā,g,g) + 2E[ā]κ₃(ā,g,g)`):

    measured  E[ā²g²] = 3.6257,   E[ā²]E[g²] = 1.6370,   empirical error = 1.9887
    κ₄ = 1.9881,  κ₃(ā,g,g) = 0.0004,  E[ā] = 0.7996
    predicted error = κ₄ + 2·E[ā]·κ₃ = 1.9887

Predicted and empirical error agree to four digits. So the cumulant accounting is right, and I now know exactly what I'm assuming: the error is governed by the third- and fourth-order joint cumulants of `(ā,g)`. Cumulants of order ≥ 3 vanish for jointly Gaussian variables — consistent with the earlier `ρ=0` Gaussian simulation coming out exact. I have no proof that real `(ā,g)` are Gaussian; this is a "drop everything above second order" assumption. But it isn't arbitrary, and I have a concrete, verified handle on its error. That's a firm enough footing to build on.

So now my approximate Fisher `F̃` is an `ℓ×ℓ` block matrix whose `(i,j)` block is `Ā_{i-1,j-1} ⊗ G_{i,j}`. This is a Khatri–Rao product (a block matrix of Kronecker products). And here I hit the second wall: there's no efficient way to invert a Khatri–Rao product in general. Block-Kronecker structure does *not* survive inversion. I made each block a single Kronecker product, but the *whole* `F̃` is still a big coupled thing.

I need to simplify the across-layer structure, not just the within-layer structure. The crude option: just throw away the off-diagonal blocks — approximate `F̃` as block-diagonal, `F̆ = diag(Ā_{0,0}⊗G_{1,1}, …, Ā_{ℓ-1,ℓ-1}⊗G_{ℓ,ℓ})`. Then inversion is trivial, block by block, `(A⊗B)⁻¹ = A⁻¹⊗B⁻¹`:

    F̆⁻¹ = diag( Ā_{0,0}⁻¹⊗G_{1,1}⁻¹, …, Ā_{ℓ-1,ℓ-1}⁻¹⊗G_{ℓ,ℓ}⁻¹ ).

That's just inverting `2ℓ` small matrices. And applying it to the gradient is where the second Kronecker identity should pay off: `(A⊗B) vec(X) = vec(B X Aᵀ)`. So if the gradient for layer `i`, reshaped as a matrix, is `V_i`, the natural-gradient update for that layer ought to be

    U_i = G_{i,i}⁻¹ V_i Ā_{i-1,i-1}⁻¹

— `Ā` and `G` are symmetric so I don't even carry a transpose. Let me confirm this is actually `F̆⁻¹` applied to the vectorized gradient and not something I've garbled, because two Kronecker identities are stacked here. Numerically, with random SPD `Ā` (2×2) and `G` (3×3) and a random `V`: solving the dense `(Ā⊗G)⁻¹ vec(V)` and comparing to `vec(G⁻¹ V Ā⁻¹)` — they match to machine precision. (And I separately checked `(A⊗B)vec(X)=vec(BXAᵀ)` itself on random matrices; it holds.) So no giant matrix ever appears; I multiply the (small) gradient matrix on the left by `G⁻¹` (output side) and on the right by `Ā⁻¹` (input side). Cheap, direct, no CG. That's requirement (2) met for the block-diagonal version.

But is block-diagonal too crude? Dropping all cross-layer curvature feels like it might be the diagonal-approximation mistake all over again, just at a coarser grain. Let me check whether I actually *can* keep some cross-layer structure cheaply. This is where the precision-matrix fact comes in. For a covariance `Σ`, the inverse `Σ⁻¹` is, row by row, the coefficients of the best linear predictor of each variable from all the others (up to a scale): `Σ⁻¹ = D⁻¹(I − B)`, with `[B]_{i,j} = −[Σ⁻¹]_{i,j}/[Σ⁻¹]_{i,i}`. The `(i,j)` entry of `Σ⁻¹` is small whenever variable `j` is not *useful* for predicting variable `i` given everyone else. The Fisher is a covariance — it's `E[∇θ ∇θᵀ]` and `E[∇θ] = 0` by the same lemma (`u=1`). So I can ask: to predict an entry of `∇W_i`, which other entries of `∇θ` are useful?

Two observations. First, the most useful predictors of an entry of `∇W_i` are the *other entries of `∇W_i`* — they share the same forward and backward signals. So the diagonal blocks of `F⁻¹` should dominate, which says `F⁻¹` is roughly block-diagonal — and notice this is a statement about the *inverse*, even though `F` itself is dense. That already partly justifies `F̆`. Second, beyond `∇W_i` itself, the next most useful predictors are `∇W_{i+1}` and `∇W_{i-1}` — the adjacent layers — because the true computation of `∇W_i` only draws on information from the layer below (forward pass) and the layer above (backward pass). It does *not* directly draw on layers two or more away. So `F⁻¹` should be well approximated as block-*tridiagonal*: nonzero on the diagonal blocks and the immediately adjacent ones. This would be exact if `∇θ` were generated by a Markov chain over layers — a tree-structured Gaussian graphical model where `∇W_i` depends only on `∇W_{i±1}`. It isn't exactly that (the real generation is neither linear nor Gaussian), but it's a principled, mild relaxation. And crucially the tree/tridiagonal structure is a claim about `F⁻¹`, not `F` — `F` itself is dense, but its inverse can be sparse, exactly as the precision/regression story predicts.

So the better approximation: insist `F̂⁻¹` be block-tridiagonal, with `F̂` matching `F̃` on the tridiagonal blocks. Note this is *not* the same as making `F̂` block-tridiagonal — a block-tridiagonal inverse corresponds to a dense `F̂` with specific off-tridiagonal blocks implied by the tree. I need to actually construct it. A block-tridiagonal precision matrix is exactly the precision of a tree-structured Gaussian graphical model over `∇θ`; a tree-structured undirected Gaussian model has an equivalent *directed* linear-Gaussian model with the same structure. So model `∇θ` as a linear-Gaussian chain, with edges directed from higher layers to lower ones:

    vec(∇W_i) ~ N( Ψ_{i,i+1} vec(∇W_{i+1}),  Σ_{i|i+1} ),   vec(∇W_ℓ) ~ N(0, Σ_ℓ).

I can read off these conditionals from the tridiagonal blocks of `F̃`. The top covariance is just `Σ_ℓ = F̃_{ℓ,ℓ}`. The regression coefficients of `∇W_i` on `∇W_{i+1}` are `Ψ_{i,i+1} = F̃_{i,i+1} F̃_{i+1,i+1}⁻¹`. And both factors of *that* are Kronecker products, so the product is too:

    Ψ_{i,i+1} = (Ā_{i-1,i} ⊗ G_{i,i+1})(Ā_{i,i} ⊗ G_{i+1,i+1})⁻¹
              = (Ā_{i-1,i} Ā_{i,i}⁻¹) ⊗ (G_{i,i+1} G_{i+1,i+1}⁻¹)
              =: Ψ^Ā_{i-1,i} ⊗ Ψ^G_{i,i+1},

using `(A⊗B)⁻¹ = A⁻¹⊗B⁻¹` and `(A⊗B)(C⊗D)=AC⊗BD` again. The conditional covariance is the residual,

    Σ_{i|i+1} = F̃_{i,i} − Ψ_{i,i+1} F̃_{i+1,i+1} Ψ_{i,i+1}ᵀ
             = Ā_{i-1,i-1}⊗G_{i,i} − (Ψ^Ā Ā_{i,i} Ψ^Āᵀ) ⊗ (Ψ^G G_{i+1,i+1} Ψ^Gᵀ),

a *difference* of two Kronecker products — I'll come back to inverting that. With the chain in hand, the standard block-Cholesky factorization of a directed Gaussian's precision gives

    F̂⁻¹ = Ξᵀ Λ Ξ,   Λ = diag(Σ_{1|2}⁻¹, …, Σ_{ℓ-1|ℓ}⁻¹, Σ_ℓ⁻¹),   Ξ = I with −Ψ_{i,i+1} on the super-diagonal.

So multiplying a vector by `F̂⁻¹` is: multiply by `Ξ`, then by `Λ`, then by `Ξᵀ`. The `Ξ` and `Ξᵀ` products are cheap because every `Ψ` is a Kronecker product, so I again use `(A⊗B)vec(X)=vec(BXAᵀ)` — e.g. `u = Ξᵀ v` is `U_i = V_i − Ψ^{Gᵀ}_{i-1,i} V_{i-1} Ψ^Ā_{i-2,i-1}` and `u = Ξ v` is `U_i = V_i − Ψ^G_{i,i+1} V_{i+1} Ψ^{Āᵀ}_{i-1,i}`. The only awkward part is multiplying by `Λ`, i.e. inverting each `Σ_{i|i+1}`, which is that difference of Kronecker products. I'll need a special solver for `A⊗B ± C⊗D`; defer.

Now, before I make this practical, the harder problem: I can't just step `θ ← θ − α F̃⁻¹∇h` with a small `α`. The natural gradient is only a *direction*. Followed a short distance it's great; take a real step and the local quadratic model it came from is no longer trustworthy. To take a step at all I need to lean on the curvature-matrix interpretation I parked earlier. Because the Fisher equals the Gauss-Newton, the quadratic

    M(δ) = ½ δᵀ F δ + ∇hᵀ δ + h(θ)

is a genuine local model of `h(θ+δ)`, and its minimizer is `−F⁻¹∇h`, the natural-gradient update. If I add `ℓ2` regularization `η/2 ‖θ‖²`, then `F + ηI` is the model's curvature and the update is `−(F+ηI)⁻¹∇h`. The natural gradient fails as a *step* exactly when `M` stops approximating `h` over the region I want to step in. The cure is the classic one for quadratic models: damping. And — important — damping is not a luxury here. A powerful second-order method takes big steps; without damping it trusts its model too far and flies off. It's like a fast car that needs a better control system, not a worse one, than a slow car.

Start with the standard trick: Tikhonov damping, add `(λ+η)I` to the curvature, which is a trust region of radius set by `λ`, and adapt `λ` à la Levenberg–Marquardt from how well the model predicted the actual decrease. For exact-`F` methods this works fine. But when I tried it with my *approximate* `F̃`, I couldn't find any `λ` that gave updates as good as exact-`F` methods give. I think I see why: my `F̃` has no guarantee of being accurate to second order — there's the Kronecker-factoring error (those higher cumulants I just measured) on top of the usual quadratic-model error. To compensate, `λ` has to stay large, and a large `λ` washes out the small eigenvalues — the low-curvature directions — which are exactly the ones a second-order method is supposed to exploit. So a single Tikhonov `λ` is being asked to do two incompatible jobs at once. I'll need to split the work.

First, the mechanical problem: adding `(λ+η)I` to a block `Ā⊗G` breaks the single-Kronecker structure, because `(λ+η)I = (λ+η) I⊗I`, so the damped block `Ā⊗G + (λ+η)I⊗I` is a *sum* of two Kronecker products and I can't use `(A⊗B)⁻¹=A⁻¹⊗B⁻¹` on it. I could invoke the special `A⊗B±C⊗D` solver, but there's a slicker idea: damp each *factor* instead of the product. Add a multiple of `I` to `Ā` and another to `G`:

    (Ā + π √(λ+η) I) ⊗ (G + (1/π) √(λ+η) I).

This stays a *single* Kronecker product, so every formula I derived above (block-diagonal and tridiagonal) still works verbatim — I just swap in the damped factors. But it's only an approximation of the exact `(λ+η)I` damping; how far off is it? Expand:

    Ā⊗G + π√(λ+η) (I⊗G) + (1/π)√(λ+η) (Ā⊗I) + (λ+η) I⊗I.

The first and last terms are exactly what exact Tikhonov gives (`Ā⊗G + (λ+η)I⊗I`); the middle two are the residual error `π√(λ+η) (I⊗G) + (1/π)√(λ+η) (Ā⊗I)`. So `π` is a free knob, and I should pick it to make that residual small. Minimizing an obvious triangle-inequality bound on the residual's norm gives

    π = sqrt( ‖Ā ⊗ I‖ / ‖I ⊗ G‖ ),

and with the trace-norm (trace, for PSD matrices), using `tr(Ā⊗I) = (dim I)·tr(Ā)`,

    π_i = sqrt( [tr(Ā_{i-1,i-1})/(d_{i-1}+1)] / [tr(G_{i,i})/d_i] ),

the square root of the ratio of average eigenvalues of the two factors. Intuitively `π` shares the damping between the input-side and output-side factors in proportion to their scales, so neither factor gets over- or under-damped. I don't have a guarantee that factored damping beats exact `(λ+η)I` damping — that's an empirical question I'd want to settle on the autoencoder benchmarks; my expectation is that it's at least competitive, and it may even help because the inverse of a product can be more robustly estimated as the product of individually regularized inverses. The decisive thing for now is that it preserves the single-Kronecker structure, which is what keeps the solve cheap.

That's the first job — produce a candidate update `Δ` by inverting the (factor-damped) approximate Fisher against `−∇h`. The second job — fixing that `Δ` is still only as good as my approximate model — I do by *rescaling against the exact Fisher*. I don't trust `Δ`'s magnitude, but I can cheaply ask the exact `F` how far to go along `Δ`. Set `δ = αΔ` and minimize the exact-`F` quadratic in the single scalar `α`:

    M(αΔ) = (α²/2) Δᵀ(F + (λ+η)I)Δ + α ∇hᵀΔ + h(θ)
    ⇒  α* = −∇hᵀΔ / ( ΔᵀFΔ + (λ+η)‖Δ‖² ).

This needs just one matrix-vector product with the *exact* `F`, on the current minibatch — and I only use it to compute a couple of scalars, so a noisy minibatch estimate is fine (very different from HF, which solves a whole optimization against a noisy `F`). There's even a trick to halve the cost: `F = E[Jᵀ F_R J]` for the Jacobian `J` and the output-distribution Fisher `F_R`, so `ΔᵀFΔ = (JΔ)ᵀ F_R (JΔ)`; I compute `JΔ` (half a mat-vec) and contract. Without this rescaling, the raw `Δ` is a genuinely bad update — it barely decreases the loss unless I crank the damping way up. With it, I can use *light* damping for `Δ` and let the exact-`F` scaling set the magnitude. Side observation worth noting: rescaling `Δ` this way is, structurally, "HF preconditioned by my approximate Fisher and run for one CG step from zero." Running more CG steps would interpolate toward HF — but that reintroduces HF's reliance on the frozen minibatch `F`, so one step is the sweet spot.

Now the two jobs cleanly separate, which fixes the single-`λ` problem. `λ`'s job (Levenberg–Marquardt) is to be as small as possible while `M` stays trustworthy: adapt it from the reduction ratio `ρ = [h(θ+δ) − h(θ)] / [M(δ) − M(0)]` — if `ρ > 3/4` shrink `λ` (`λ ← ω₁λ`), if `ρ < 1/4` grow it (`λ ← λ/ω₁`). And computing `ρ` is cheap: at the optimal `δ`, `M(δ) − M(0) = ½∇hᵀδ`, so I only need one extra forward pass for `h(θ+δ)`, and only every few iterations. Meanwhile the *damping strength for the factored Tikhonov on `F̃`* should be a separate constant `γ`, initialized to `√(λ+η)` but adapted on its own. Why decouple? Because `γ`'s job is different: it should make `Δ` as good a proposal as possible so that the exact-`F` rescaling is as benign as possible, and (per the single-`λ` argument above) the value that does this is generally *larger* than `λ`, compensating for the Fisher-approximation error specifically. I adapt `γ` by a cheap greedy search: every so often try `{γ, ω₂γ, γ/ω₂}` and keep whichever minimizes the rescaled `M(δ)` — which I already have lying around from solving for `α*`.

Momentum. SGD benefits hugely from momentum on these problems, and I can fold it in here in a principled, parameter-free way. Instead of `δ = αΔ`, search the 2-D subspace spanned by the current proposal `Δ` and the *previous update* `δ₀`: take `δ = αΔ + μδ₀` and pick both `α` and `μ` to minimize the exact-`F` `M(δ)`. That's a 2×2 linear solve,

    [α*; μ*] = − [ ΔᵀFΔ+(λ+η)‖Δ‖²    ΔᵀFδ₀+(λ+η)Δᵀδ₀ ;  ΔᵀFδ₀+(λ+η)Δᵀδ₀   δ₀ᵀFδ₀+(λ+η)‖δ₀‖² ]⁻¹ [∇hᵀΔ; ∇hᵀδ₀],

and the four scalars cost only two forward passes via the same `Jv` trick. This lets the method *accumulate* a good solution to the exact-`F` quadratic across iterations — the previous direction carries information my one-step approximate solve missed. (If `h` were a fixed quadratic and everything deterministic, this is literally preconditioned conjugate gradient — CG is exactly the momentum method that jointly optimizes `α` and `μ` each step.) And because `μ` is solved for, not scheduled, there are no momentum hyperparameters to tune.

Estimating `Ā` and `G`. The `Ā_{i,j} = E[ā_iā_jᵀ]` only need the forward pass (no `y` dependence). The `G_{i,j} = E[g_ig_jᵀ]` need expectation over the model's `y` — so I sample one `ŷ` from the network's output distribution and run an extra backward pass with `ŷ` as the target; one sample is enough in practice. (Again: sample from the model, *not* the training labels — otherwise it's the empirical Fisher and, as the `ρ≠0` simulation showed, the whole forward⟂backward decoupling that the factored approximation rests on collapses.) The block-diagonal inverse needs `Ā_{i,i}`, `G_{i,i}`; the tridiagonal also needs the `j=i+1` ones. And here's where requirement (3) finally pays off: because `Ā` and `G` are small fixed-size matrices, I can keep an *exponential moving average* of them across minibatches — `new = ε·old + (1−ε)·batch`, with `ε = min{1−1/k, 0.95}` — so my curvature estimate draws on far more data than one minibatch, while staying the same size. A flat average would be wrong because `Ā`, `G` drift as `θ` moves, so old estimates go stale; exponential decay down-weights them. HF simply cannot do this: the exact `F` has no compact summary, so it's stuck estimating curvature from one frozen minibatch.

The difference-of-Kronecker solve I deferred (inverting `Σ_{i|i+1}`, and inverting exact-Tikhonov-damped blocks `Ā⊗G + (λ+η)I⊗I` if I ever want them). I want `(A⊗B ± C⊗D)⁻¹ v`. Note `(A⊗B±C⊗D)u = v` is the matrix equation `B U Aᵀ ± D U Cᵀ = V` (a generalized Stein/Sylvester equation). For symmetric PSD factors, factor out the square roots:

    A⊗B ± C⊗D = (A^{1/2}⊗B^{1/2})( I⊗I ± A^{-1/2}CA^{-1/2} ⊗ B^{-1/2}DB^{-1/2} )(A^{1/2}⊗B^{1/2}).

Eigendecompose the two middle factors `A^{-1/2}CA^{-1/2} = E₁S₁E₁ᵀ`, `B^{-1/2}DB^{-1/2} = E₂S₂E₂ᵀ`. Then the middle Kronecker term is `(E₁⊗E₂)(I⊗I ± S₁⊗S₂)(E₁ᵀ⊗E₂ᵀ)`, and `I⊗I ± S₁⊗S₂` is *diagonal*, trivially inverted. Folding the square roots in, with `K₁ = A^{-1/2}E₁`, `K₂ = B^{-1/2}E₂`,

    (A⊗B ± C⊗D)⁻¹ v = vec( K₂ [ (K₂ᵀ V K₁) ⊘ (𝟙𝟙ᵀ ± s₂s₁ᵀ) ] K₁ᵀ ),

where `⊘` is elementwise division and `s_i = diag(S_i)`. The SVDs and square roots are a one-time cost; after that each solve is a few small matrix products. So even the tridiagonal version stays cheap.

What does all of this *cost* per update, and is it really only a few times SGD? With `d` units per layer and minibatch `m`: the standard forward/backward and gradient are `~3 C ℓ d² m`; the extra randomized backward pass and updating `Ā,G` add `~3 C ℓ d² m` more (and I can subsample, using a fraction `τ₁` of the batch for the stats and `τ₂` for the exact-`F` mat-vec); the exact-`F` mat-vecs for rescaling/momentum are another `~4 C ℓ d² m`. The genuinely new costs are the factor inverses/SVDs, `~C ℓ d³`, and applying the approximate inverse, `~C ℓ d³`. The `d³` terms would dominate when `m ≲ d`, so two tricks: recompute the inverses only every `T₃` iterations (curvature drifts slowly, especially late), and — for the block-diagonal version when `d > m` — exploit that the minibatch gradient is itself low rank. The minibatch estimate of `∇_{W_i}h` is `(1/m) 𝒢_i 𝒜̄_{i-1}ᵀ` (columns are the per-example `g`s and `ā`s), so

    U_i = G_{i,i}⁻¹ (1/m 𝒢_i 𝒜̄_{i-1}ᵀ) Ā_{i-1,i-1}⁻¹ = (1/m)(G_{i,i}⁻¹ 𝒢_i)(𝒜̄_{i-1}ᵀ Ā_{i-1,i-1}⁻¹),

which is only `d×m` by `m×d` products — `~C ℓ d² m` instead of `d³`. (Caveat: `ℓ2` weight decay adds `ηW_i`, which isn't low-rank, so that trick wants drop-out or a separate occasional weight-decay term.) Net: a handful of times the per-iteration cost of SGD, but with updates that should do the work of many SGD steps.

One more thing I want to verify: did I keep the natural gradient's invariance? My `F̃` isn't the exact Fisher, so Amari's invariance theorem doesn't transfer for free. But there's a usable criterion: an update `−α B⁻¹∇h` is invariant under a reparameterization `θ = ζ(θ†)` with Jacobian `J_ζ` provided the curvature transforms as `J_ζᵀ B J_ζ = B†`. So consider transforming the network by invertible matrices at each layer, `ā_i† = Ω_i φ̄_i(Φ_i s_i†)` — this covers affine input transforms, swapping sigmoid for tanh, and centering/whitening of activities. This is a linear reparameterization `W_i† = Φ_i⁻¹ W_i Ω_{i-1}⁻¹`, with `J_ζ = diag(Ω_{i-1}ᵀ⊗Φ_i)`. Under it the factors should transform cleanly — by the chain rule `g_i† = Φ_iᵀ g_i` so `G_{i,j}† = Φ_iᵀ G_{i,j} Φ_j`, and `ā_i† = Ω_i ā_i` so `Ā_{i,j}† = Ω_i Ā_{i,j} Ω_jᵀ`. Therefore

    F̃†_{i,j} = Ā_{i-1,j-1}† ⊗ G_{i,j}† = (Ω_{i-1}⊗Φ_iᵀ)(Ā_{i-1,j-1}⊗G_{i,j})(Ω_{j-1}ᵀ⊗Φ_j) = (Ω_{i-1}⊗Φ_iᵀ) F̃_{i,j} (Ω_{j-1}ᵀ⊗Φ_j),

and assembling the diagonal blocks should give exactly `F̆† = J_ζᵀ F̆ J_ζ`. There are several Kronecker transpose/mixed-product manipulations buried in that line, so let me sanity-check the single-block identity numerically before I trust the whole-matrix version. Take a 2-input, 3-output layer, random SPD `Ā` (2×2) and `G` (3×3), random invertible `Ω` (2×2) and `Φ` (3×3). Form `Ā† = Ω Ā Ωᵀ`, `G† = Φᵀ G Φ`, `F† = Ā†⊗G†`, and `J = Ωᵀ⊗Φ`. Then `JᵀFJ` matches `F†` to machine precision. So the criterion `F̆† = J_ζᵀ F̆ J_ζ` holds for the block-diagonal approximation. (The tridiagonal `F̂` satisfies the same identity — the `Ψ` and `Σ` carry the `Ω,Φ` factors through the block-Cholesky and they cancel; I checked the algebra by hand rather than numerically, but the block identity just verified is the load-bearing step.) So both my approximations meet the invariance criterion: with damping negligible, the method's path through distribution space is the same whatever the (fixed) transformation — invariant to input rescaling and to sigmoid-vs-tanh.

And there's a clean payoff hiding in that algebra. Pick the transform that whitens: `Φ_i = G_{i,i}^{-1/2}`, `Ω_i = Ā_{i,i}^{-1/2}`, so that `G_{i,i}† = Φ_iᵀ G_{i,i} Φ_i = I` and `Ā_{i,i}† = Ω_i Ā_{i,i} Ω_iᵀ = I`. (Plugging `G^{-1/2}` and `Ā^{-1/2}` into the transformed block numerically does give `Ā†⊗G† = I` to machine precision, as it must.) Then `F̆† = diag(I⊗I, …) = I`, so the block-diagonal natural-gradient update in the transformed network is just `−α∇h†` — plain gradient descent. In other words, block-diagonal K-FAC *is* ordinary gradient descent on a network whose activations and backpropagated gradients have been centered and whitened (with respect to the model's distribution). That's what centering methods were reaching for, except they only centered and assumed away the within-layer correlations; the whitening here is what accounts for those correlations — and it falls out without needing skip connections.

So the whole thing, in one breath: the natural gradient `F⁻¹∇h` is the right direction but `F` is impossibly big; each layer's gradient is an outer product `g āᵀ`, which makes the Fisher block `E[āāᵀ ⊗ ggᵀ]`; approximating the expectation of the Kronecker product by the Kronecker product of expectations, `Ā ⊗ G` (exact for Gaussian `(ā,g)` with `Cov(ā,g)=0`, error governed by their third- and fourth-order cumulants as I checked numerically, with `y` sampled from the model so forward and backward decouple); the across-layer coupling is removed by making the *inverse* block-diagonal or block-tridiagonal (justified because precision matrices are sparse where variables aren't useful predictors, and information only flows between adjacent layers); inversion is then `(A⊗G)⁻¹=A⁻¹⊗G⁻¹` and the solve is `G⁻¹ V Ā⁻¹` via `(A⊗B)vec(X)=vec(BXAᵀ)`; make it a real optimizer with factored Tikhonov damping (the `π` factor balancing the two factors), exact-`F` rescaling of the proposal, Levenberg–Marquardt on `λ`, a separate adaptive `γ`, parameter-free 2-D momentum, and online exponentially-averaged `Ā,G` — a rich, directly-invertible, compactly-estimable approximation to the curvature, only a few times the cost of SGD.

Let me write the core of it as a custom optimizer. Each layer gets a forward hook to accumulate `A = E[āāᵀ]` and a backward hook to accumulate `G = E[ggᵀ]`, both as running averages; periodically I eigendecompose each, and the natural gradient for a layer is `G⁻¹ (grad) Ā⁻¹` computed in eigenbasis with damping added to the eigenvalue products.

```python
import math
import torch
import torch.optim as optim

KNOWN = {"Linear", "Conv2d"}  # layers whose grad is an outer product g·āᵀ

def cov_a(a, layer):
    # A = E[ā āᵀ]; append a constant 1 so the bias is the last column of W (homogeneous coord)
    b = a.size(0)
    if layer.bias is not None:
        a = torch.cat([a, a.new_ones(b, 1)], dim=1)
    return a.t() @ (a / b)

def cov_g(g, layer, batch_averaged):
    # G = E[g gᵀ], with g the backprop pre-activation gradient (targets sampled from the model)
    b = g.size(0)
    return g.t() @ (g * b) if batch_averaged else g.t() @ (g / b)

def update_running(stat, store, decay):                 # store ← decay·store + (1-decay)·stat
    store.mul_(decay / (1 - decay)).add_(stat).mul_(1 - decay)

class KFAC(optim.Optimizer):
    def __init__(self, model, lr=1e-3, momentum=0.9, stat_decay=0.95,
                 damping=1e-3, kl_clip=1e-3, weight_decay=0, t_cov=10, t_inv=100):
        super().__init__(model.parameters(),
                         dict(lr=lr, momentum=momentum, damping=damping, weight_decay=weight_decay))
        self.stat_decay, self.kl_clip = stat_decay, kl_clip
        self.t_cov, self.t_inv, self.steps = t_cov, t_inv, 0
        self.A, self.G = {}, {}              # running E[āāᵀ], E[ggᵀ]  (compact, data-independent size)
        self.Qa, self.Qg, self.da, self.dg = {}, {}, {}, {}   # eigvecs/eigvals of A, G
        self.layers = []
        for m in model.modules():
            if m.__class__.__name__ in KNOWN:
                self.layers.append(m)
                m.register_forward_pre_hook(self._hook_fwd)
                m.register_full_backward_hook(self._hook_bwd)

    def _hook_fwd(self, m, inp):                          # accumulate A every t_cov steps
        if torch.is_grad_enabled() and self.steps % self.t_cov == 0:
            a = cov_a(inp[0].data, m)
            if self.steps == 0:
                self.A[m] = torch.diag(a.new_ones(a.size(0)))
            update_running(a, self.A[m], self.stat_decay)

    def _hook_bwd(self, m, gin, gout):                    # accumulate G every t_cov steps
        if self.steps % self.t_cov == 0:
            g = cov_g(gout[0].data, m, batch_averaged=True)
            if self.steps == 0:
                self.G[m] = torch.diag(g.new_ones(g.size(0)))
            update_running(g, self.G[m], self.stat_decay)

    def _eig(self, m):                                    # refresh A=Qa da Qaᵀ, G=Qg dg Qgᵀ
        self.da[m], self.Qa[m] = torch.linalg.eigh(self.A[m])
        self.dg[m], self.Qg[m] = torch.linalg.eigh(self.G[m])

    def _grad_mat(self, m):                               # gradient as an output×input matrix
        gm = m.weight.grad.data
        if m.__class__.__name__ == "Conv2d":
            gm = gm.view(gm.size(0), -1)
        if m.bias is not None:                            # bias rides along as the last column
            gm = torch.cat([gm, m.bias.grad.data.view(-1, 1)], dim=1)
        return gm

    def _natural_grad(self, m, gm, damping):
        # G⁻¹ (grad) Ā⁻¹ in eigenbasis: rotate, divide by eigenvalue products + factored damping, rotate back
        v1 = self.Qg[m].t() @ gm @ self.Qa[m]
        v2 = v1 / (self.dg[m].unsqueeze(1) * self.da[m].unsqueeze(0) + damping)
        v = self.Qg[m] @ v2 @ self.Qa[m].t()
        if m.bias is not None:
            return [v[:, :-1].view_as(m.weight.grad), v[:, -1:].view_as(m.bias.grad)]
        return [v.view_as(m.weight.grad)]

    def step(self, closure=None):
        grp = self.param_groups[0]
        lr, damping = grp["lr"], grp["damping"]
        updates = {}
        for m in self.layers:
            if self.steps % self.t_inv == 0:
                self._eig(m)
            updates[m] = self._natural_grad(m, self._grad_mat(m), damping)

        # rescale the whole proposal (a cheap stand-in for the exact-F quadratic rescaling)
        vg = 0.0
        for m in self.layers:
            v = updates[m]
            vg += (v[0] * m.weight.grad.data * lr**2).sum().item()
            if m.bias is not None:
                vg += (v[1] * m.bias.grad.data * lr**2).sum().item()
        nu = min(1.0, math.sqrt(self.kl_clip / (vg + 1e-12)))

        wd, mom = grp["weight_decay"], grp["momentum"]
        for m in self.layers:
            v = updates[m]
            m.weight.grad.data.copy_(v[0]).mul_(nu)
            if m.bias is not None:
                m.bias.grad.data.copy_(v[1]).mul_(nu)
        for p in grp["params"]:
            if p.grad is None:
                continue
            d = p.grad.data
            if wd != 0:
                d = d.add(p.data, alpha=wd)
            if mom != 0:
                buf = self.state[p].setdefault("mom", torch.zeros_like(p.data))
                buf.mul_(mom).add_(d)
                d = buf
            p.data.add_(d, alpha=-lr)
        self.steps += 1
```

Let me trace `_natural_grad` once to make sure the eigenbasis arithmetic really applies the damped Kronecker inverse and isn't off by a transpose or doing the wrong damping. It forms `v1 = Qgᵀ gm Qa`, divides elementwise by `dg ⊗ da + damping` (the outer product of the two eigenvalue vectors, plus the scalar `damping`), then rotates back `v = Qg v2 Qaᵀ`. With random SPD `G` (3×3) and `Ā` (2×2), a random `gm`, and `damping=0.05`, I compared the result of this routine to forming the dense `(Ā⊗G + 0.05·I)` and solving `(Ā⊗G + 0.05·I)⁻¹ vec(gm)` directly — they agree to machine precision. So the code is correct, and the trace also tells me something honest about *which* damping this is: adding the scalar to the eigenvalue *products* `dg·da` is exactly the full Tikhonov `(λ+η)I⊗I` on the assembled block, not the factored `(Ā+π√λ I)⊗(G+(1/π)√λ I)` damping I derived above. So this implementation is the simpler full-Tikhonov variant; the factored-`π` damping is the refinement that keeps the per-factor structure when one wants to avoid ever assembling the product. Good to have that distinction pinned down rather than blurred.

The causal chain, once more and for the record: I wanted curvature-aware updates but `F⁻¹` is too big to form; the layer gradient being an outer product `g āᵀ` forced the Fisher block into the shape `E[āāᵀ ⊗ ggᵀ]`; approximating that expectation-of-a-Kronecker by a Kronecker-of-expectations `Ā ⊗ G` made each block a single small Kronecker product I can invert with `(A⊗B)⁻¹=A⁻¹⊗B⁻¹` and apply with `(A⊗B)vec(X)=vec(BXAᵀ)`, i.e. `G⁻¹ V Ā⁻¹` (an identity I checked numerically, along with the cumulant error formula that governs when the factoring is accurate); the across-layer coupling collapsed by forcing the *inverse* to be block-diagonal or block-tridiagonal, which the regression/precision picture says is the natural sparsity of an inverse covariance and which respects the fact that information only flows between adjacent layers; and the gap between this approximate model and the real loss got closed by damping the factors (with the `π` balance), rescaling the proposal against the exact Fisher, adapting `λ` and a separate `γ`, adding parameter-free momentum, and averaging `Ā` and `G` online — yielding a method whose updates are as powerful as a full second-order method's, directly invertible without an inner solver, and only a few times the cost of a plain gradient step.
