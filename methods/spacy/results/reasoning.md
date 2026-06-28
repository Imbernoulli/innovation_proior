OK, let me think this through from scratch. I have gridded time series — climate, say. At every point on a grid I get a time series, and the grid is huge: a hundred-by-a-hundred grid is ten thousand "variables", a global temperature map is tens of thousands per variate. I want the causal story: which driving processes push on which, with what time lags, possibly with same-time coupling. The thing scientists actually care about — teleconnections, an oscillation over the Pacific nudging rainfall in Southeast Asia a month later — is a causal statement, and I want to recover it from the raw grid, unsupervised.

So the naive move is: treat the ten thousand grid points as ten thousand variables and run a temporal causal discovery algorithm on them. Let me see why that dies, because the way it dies tells me what to build.

First, scale. The mature constraint-based methods — the PC-algorithm family extended to time series — work by conditional-independence tests: is X independent of Y given some conditioning set S? The number of such tests, and the size of the conditioning sets, blows up with the number of variables. Ten thousand variables is a non-starter. So whatever I do, I can't reason about causality directly among grid points; I need a small number of variables.

Second, and this one is subtler and more damning, spatial redundancy. Two grid points a few kilometres apart have nearly identical series — they're not two independent variables, they're one signal sampled twice. Now picture a real long-range causal link: a mode over the mid-Pacific drives rainfall in Southeast Asia. A CI test trying to detect that link conditions on a set of "other" variables. But the obvious other variables include the dozens of grid points right next to the Pacific mode, which are near-copies of it. Conditioning on a near-copy of the cause soaks up exactly the statistical signal I was trying to attribute to the distant link. The test loses power; the long-range edge vanishes or gets mangled. So even if scale weren't a problem, spatial correlation actively sabotages the discovery.

Both pathologies point the same way: don't do causal discovery on the grid. Do it on a small set of latent driving factors, `D` of them, with `D << L`. Each factor is a single time series; the causal graph lives among the `D` factors. That instantly fixes scale (a graph over thirty nodes, not ten thousand) and it sidesteps redundancy, because a factor *is* the aggregate of a correlated region rather than one of its redundant members.

Fine — extract latents, then do causal discovery on them. But here's where I have to be careful, because there's an obvious version of this that's quietly wrong. The obvious version is two-stage: run a dimensionality reduction (PCA, or Varimax-rotated PCA) to get `D` components, *then* run a causal discovery method on the components. People do exactly this in climate. The trouble is that the reduction step has no idea about causality. PCA picks directions of maximal variance; Varimax rotates them to be "simple"; neither knows or cares whether the resulting components correspond to causally coherent entities. You can easily get components that smear two distinct causal drivers together, or split one across several, and then no causal discovery on top can recover what was destroyed upstream. The reduction and the discovery are optimizing different, unaligned objectives.

And there's a concrete symptom of the mismatch I should keep in mind. PCA/Varimax modes have weight vectors that are nonzero essentially everywhere on the grid — a "mode" is a global pattern with support over the whole map. When you try to read it as "a region", you can't: it's diffuse, it overlaps everything, and projecting its causal effect back to the grid wires up half the planet. So the two-stage approach gives me factors that are both causally unaligned and spatially uninterpretable.

So the requirement I keep arriving at is that the latent extraction and the causal discovery cannot be two separate stages with two separate objectives. If the reduction is fixed before discovery starts, the discovery can only work with whatever the reduction left it, and the reduction had no causal objective. The way out is to make them *the same optimization* — infer the factors and the graph jointly, end-to-end, so the factors are shaped by the pressure to be causally coherent rather than handed down by an unrelated variance criterion.

Let me set up the generative picture, because if I write down how the data is *made*, inference falls out as inverting it. I'll posit that the observed grid is produced by a small number of latent series `Z ∈ R^{D×T}` evolving under some causal dynamics, and that each latent gets spread over the grid by some spatial pattern, and then I observe the superposition (plus noise). Write `F ∈ R^{L×D}` for the spatial patterns — column `d` says how strongly latent `d` shows up at each grid point. The cleanest version is `X = F Z` (plus noise): each grid point's series is a linear combination, with weights `F_ℓ`, of the latent series. Allow a per-grid-point nonlinearity on top, because real sensors and physics aren't linear: `X_ℓ^t = g_ℓ([FZ]_ℓ^t) + ε_ℓ^t`.

Now the crux question is: what is `F`? This is where I have to think, not just code.

Option one: let `F` be a free `L×D` matrix and learn every entry. That's `L·D` parameters — for a 100×100 grid and 30 factors that's 300,000 free numbers — and, worse, it has no notion of space at all. Nothing stops a learned column of `F` from being a scattered, diffuse, everywhere-nonzero pattern, which is exactly the uninterpretable mess I'm trying to escape, and exactly what unconstrained reduction gives. A free matrix throws away the one structural fact I have: nearby grid points belong together.

So I want to *bake in* the spatial structure. What do I actually believe about a factor's footprint? I believe a driving mode occupies a coherent *region* — a blob — and that the influence falls off smoothly as you move away from the centre of the region. That's a locality-plus-smoothness prior. The mathematical object that expresses "smooth bump centred somewhere, decaying with distance" is a radial basis function: `F_ℓd = exp(−‖x_ℓ − ρ_d‖² / scale_d)`, where `x_ℓ` is the coordinate of grid point `ℓ`, `ρ_d` is the centre of factor `d`, and `scale_d` controls how wide the blob is. Now each factor is described by a handful of numbers — a centre and a scale — instead of `L` free weights. That's enormously parameter-efficient, it *forces* locality (the kernel decays, so the factor is a localized region, not a global smear), and it's smooth by construction. The centre and scale are interpretable: I can point at the map and say "this factor is here, this big".

That dissolves the diffuseness symptom, and the parameter count is a clear win. But I'm uneasy calling it settled on those grounds alone — "it's smooth and interpretable" is the kind of justification that sounds good and commits me to nothing. A kernel is a strong assumption about the data; before I lean on it I'd want to know whether it earns its place beyond aesthetics. I'll proceed with factors-as-kernels for now and force myself to come back to that question once the rest of the model is on the table.

One thing I should refuse to give up while I'm here. There's a tempting sparsity assumption floating around in this space: insist that each *observed* grid point is driven by exactly *one* latent — single-parent decoding. That assumption is seductive because it makes identifiability easy. But it's physically wrong for my data: a grid point over the tropical Pacific is jointly pushed by atmospheric and oceanic modes that overlap there. If I forbid overlap, I either collapse those modes into one or scatter them, and I lose exactly the multi-driver structure I care about. So I will allow factors to overlap — a grid point can be a blend of several kernels. I'm choosing the harder modelling problem on purpose, and I'll have to earn identifiability some other way.

Now the latent dynamics. The `Z` series evolve causally and I need a model `p(Z | G)` for "how likely is this set of latent series under causal graph `G`". What do I want from it? I want it to (a) be differentiable, so I can learn `G` by gradient; (b) handle *instantaneous* edges, not just lagged ones — same-month couplings are real in climate; (c) handle nonlinear functional relationships; and (d) cope with noise whose character depends on history, because variance in these systems isn't constant. Granger-style models give me none of (b)–(d): Granger is purely about past-predicts-future, no instantaneous effects, no confounders, simple noise. Linear SCMs give me (a) but not (c). So I want a differentiable nonlinear SCM with instantaneous edges and history-dependent noise. That's exactly an additive-noise temporal SCM of the form `Z_d^t = f_d(Pa(<t), Pa(t)) + (noise that can depend on history)`, where `Pa(<t)` are lagged parents and `Pa(t)` are instantaneous parents.

How do I make `f_d` differentiable in the graph and shared across nodes? I keep a trainable embedding per node (per lag), and I form `f_d` as a *sum over potential parents*, each parent's contribution gated by the edge variable `G^{(k)}_{j,d}` (lag `k`, source `j`, target `d`) and passed through shared MLPs. Something like `f_d = ξ( Σ_k Σ_j G^{(k)}_{j,d} · λ([Z_j^{t−k}, e_j^k], e_d^0) )`: the inner network `λ` encodes each parent-plus-its-embedding, the edge variable switches it on or off, we sum the live contributions, and the outer network `ξ` maps the aggregate to the prediction. Because the `G` factors multiply the contributions, the gradient flows into `G`, and the same `ξ, λ` serve every node — so adding nodes doesn't add functions. For the noise, instead of a fixed Gaussian I push the residual through a conditional spline normalizing flow whose parameters are predicted from history; that lets the noise distribution bend with the past. This is the differentiable temporal SCM I'll drop into the latent space. (And a linear special case is handy for sanity: `f_d = Σ_k Σ_{d'} (G ∘ W)^k_{d',d} Z_{d'}^{t−k}` with isotropic Gaussian noise — same skeleton, weights instead of MLPs.)

Now I have a full generative model: `G` makes `Z`; `F` (kernels) and `g_ℓ` map `Z` to `X`. I want to fit it to data, i.e. maximize `log p(X)`. But `p(X) = ∫ p(X | Z, F) p(Z | G) p(F) p(G) dZ dG dF` integrates out three different latent objects — the latent series, the graph, the spatial factors. That integral is intractable. Standard move: variational inference. Introduce a posterior `q_φ(Z|X) q_φ(G) q_φ(F)`, and lower-bound the evidence.

Let me actually do the bound rather than wave at it. Start from the log-likelihood, insert the variational distribution top and bottom, and use Jensen:

`log p_θ(X) = log ∫ q(Z|X)q(G)q(F) · [ p_θ(X,Z,G,F) / (q(Z|X)q(G)q(F)) ] dZ dG dF`
`           ≥ E_{q(Z|X)q(G)q(F)} [ log ( p_θ(X,Z,G,F) / (q(Z|X)q(G)q(F)) ) ].`

Then factor the joint by the generative assumptions: `p_θ(X,Z,G,F) = p_θ(X|Z,F) p_θ(Z|G) p(G) p(F)`. Across the `N` i.i.d. samples, observations factorize given `(Z,F)`, and latent series factorize given `G`, so `p_θ(X|Z,F)=∏_n p_θ(X^n|Z^n,F)` and `p_θ(Z|G)=∏_n p_θ(Z^n|G)`. Substituting and grouping the log-ratio by which variable each piece belongs to:

`log p_θ(X) ≥ Σ_n E_{q(Z^n|X^n)q(G)q(F)} [ log p_θ(X^n | Z^n, F) + ( log p_θ(Z^n | G) − log q(Z^n | X^n) ) ] − KL(q(G) ‖ p(G)) − KL(q(F) ‖ p(F)).`

Good — that's the objective, and every term has a clean job. The first is reconstruction: how well do the decoded factors-times-latents match the grid; with Gaussian observation noise, the log-likelihood is a negative squared error. The middle bracket is the latent-causal fit plus the variational entropy: `log p(Z|G)` scores the latent series under the SCM (the spline-flow log-likelihood of the residual `Z^t − f(Pa)`, or for the linear variant the Gaussian residual score), and `−log q(Z|X)` is the Gaussian encoder entropy. When I minimize the negative ELBO in code this becomes a latent negative log-likelihood plus `E_q log q`, i.e. a negative entropy term, with constants that do not affect gradients. The `KL(q(G)‖p(G))` and `KL(q(F)‖p(F))` keep the graph and factor posteriors near their priors. That's it — minimize the negative of this.

Now I have to design the three `q`'s, and each forces a small decision.

The latent encoder `q(Z|X)`: amortize it. A neural network reads the grid window and outputs a Gaussian mean and log-variance for `Z`; sample with the reparameterization trick `Z = μ + σ⊙η`, `η∼N(0,I)`, so gradients flow. Standard, nothing exotic.

The spatial-factor posterior `q(F)`: the factor is determined by its centre `ρ_d` and scale `γ_d`, so I put Gaussians on those and reparameterize them. Two wrinkles. The centre must live on the grid, in `[0,1]^K` — so I sample `ρ_d` unconstrained and squash with a sigmoid. And a single scalar scale only gives circular blobs; real modes are elongated. So I let the scale parameters build a small positive-definite matrix `P = A Aᵀ + diag(exp(B))`. Mathematically one can call the inverse of a covariance a precision; the implementation uses this sampled positive-definite matrix directly in the exponent, so the actual bump is `exp(−½ (x−ρ)ᵀ P (x−ρ))`. That still gives anisotropic ellipses, just with `P` acting as the learned precision. The KL of these Gaussians against `N(0,I)` priors is closed-form.

The graph posterior `q(G)`: each possible edge is present or absent, so model `G` as independent Bernoullis — one probability per (lag, source, target). But two problems. (1) I need to backprop through a *discrete* graph sample to get the ELBO gradient. Reach for the Gumbel-softmax with hard samples and a straight-through estimator: I get a genuine binary `G` forward, and a usable gradient backward. REINFORCE would also give an unbiased gradient but with brutal variance, so Gumbel it is. (2) The instantaneous slice must be a DAG, and naively sampling each ordered pair `(i,j)` and `(j,i)` independently invites immediate two-cycles `i→j→i`. Cleaner: for each unordered pair sample a single three-way categorical — `i→j`, or `j→i`, or no edge — so I never instantiate both directions at once. (Lagged edges have no such constraint — they point forward in time and can even be self-loops — so plain Bernoullis there.)

That three-way trick removes two-cycles but not longer instantaneous cycles, so I still need to *enforce acyclicity* on `G^0`. I don't want a combinatorial DAG search inside a gradient loop. The continuous option I know of is the trace-of-matrix-exponential penalty `h(G^0) = tr(e^{G^0}) − D`. I want to convince myself it actually does what I need before I build a whole augmented-Lagrangian schedule around it, so let me evaluate it on a few tiny `3×3` slices by hand. `tr(e^{G}) = Σ_k tr(G^k)/k!`, and `tr(G^k)` counts closed walks of length `k`; the `−D` kills the `k=0` term, so `h` is built entirely out of cycles. For the strictly-triangular chain `1→2→3` (the adjacency has ones only above the diagonal), `G` is nilpotent — `G^3 = 0`, every `tr(G^k)=0` for `k≥1` — so `h = 0`. Compute it: I get `tr(e^G) − 3 = 3 − 3 = 0`. For a 2-cycle `1↔2`, `G^2` has the loop, `tr(G^2)=2`, and `e^G` gives `tr(e^{G}) − 3 ≈ 4.086 − 3 = 1.086 > 0`. For a 3-cycle `1→2→3→1`, `tr(G^3)=3` is the first nonzero trace and `h ≈ 3.504 − 3 = 0.504 > 0`. So it reads zero exactly on the DAG and strictly positive on each cyclic slice, and it's smooth in the entries — which is precisely the property that lets gradient descent push it down. Good; the penalty behaves. I fold it (with a sparsity term, because causal graphs should be sparse — penalize `‖G‖²`) into the graph prior `p(G) ∝ exp(−α‖G‖² − σ·h(G^0))`, and rather than a fixed weight `σ` I use an augmented-Lagrangian schedule so the acyclicity term ramps up until `h(G^0)→0`.

A couple of training pragmatics I can already feel I'll need. If I let the SCM and graph learn from epoch zero, the half-formed, wrong early graph corrupts the factors and latents, and everything chases its own tail. So freeze the latent-SCM and graph parameters for the first stretch of training — let the spatial factors and the encoder settle into a sensible representation first — then unfreeze and learn the causal structure on top of stable latents. And the latent-prior term in the ELBO can overwhelm or be overwhelmed by reconstruction; a β-weighting on that KL-like term (β-VAE style, scaling with `D`) keeps the two in balance.

Now back to the question I deferred — whether the kernels earn their place. The reason it can't wait any longer is that I've made a choice that should hurt me. I allowed factors to overlap and I dropped the single-parent assumption, and overlapping factors under a nonlinear mixing are exactly the setting where the latents I recover could be some entangled rotation of the true ones rather than the true ones themselves. If they're not identifiable, the whole "the factors are causally relevant entities" promise is hollow — I could be reading the causal graph off a scrambled basis. So I have to ask whether anything in my construction pins the latents down.

The standard latent-identifiability results in temporal settings give recovery up to permutation and scaling, but they buy it with assumptions I explicitly don't want: no instantaneous effects, or a sparse latent graph, or "sufficient variability" of the latent process across regimes. All three are hard to verify and I'd rather not assume any of them. So I'd need a *different* source of identifiability, and the only structural fact I have that those settings lack is *space*: `L >> D`, thousands of grid equations constraining a few dozen latents. An overdetermined system *can* pin down its unknowns, but "wildly overdetermined" is not by itself a proof — overdetermined systems are also routinely inconsistent or degenerate. I should test whether the overdetermination actually buys recovery before I trust it, starting with the simplest case I can compute.

To reason cleanly, push the grid to its limit: instead of `L` discrete points, imagine a *continuous* spatial domain `(0,1)^K` with a series at every point. Then a factor isn't a column of numbers, it's a *function* `F_{ψ}(ℓ)` of the location `ℓ`, and the model at each point is `X^t(ℓ) = g_ℓ( F_ℓᵀ Z^t ) + ε`, with `F_ℓ = [F_{ψ_1}(ℓ), …, F_{ψ_D}(ℓ)]` evaluated from a family of functions. Call this object a spatial factor process. Now I can use real analysis on the functions, and the kernel choice stops being cosmetic: I'm going to need the factor functions to be (i) *linearly independent* as a family and (ii) *real analytic*. RBFs are both. Hold that thought.

What does identifiability even mean here? Two models — the true `(Z, F, g, p_ε)` and a learned `(Ẑ, F̂, ĝ, p_ε)` — that produce *the same observational distribution at every location and time* should force `Ẑ = P S Z` (permutation `P`, scaling `S`) and the same factor family. So suppose `p(X^t(ℓ)|Z; F_ℓ) = p(X̂^t(ℓ)|Ẑ; F̂_ℓ)` for all `ℓ, t`. I need to crank that down to a statement about the noiseless signals.

The noise is additive, and additive noise convolves densities. Concretely `p(X^t(ℓ)=x | Z) = ∫ δ_{x̄}(z) p_ε(x−z) dz = (δ_{x̄} * p_ε)(x)` where `x̄ = g_ℓ(F_ℓᵀ Z)` is the clean signal and `δ_{x̄}` is a point mass there; likewise `(δ_{x̃} * p_ε)(x)` with `x̃ = ĝ_ℓ(F̂_ℓᵀ Ẑ)`. Equal observational densities give `δ_{x̄} * p_ε = δ_{x̃} * p_ε`. Convolutions are products under Fourier transform, so transforming both sides: `e^{is x̄} φ_ε(s) = e^{is x̃} φ_ε(s)`, where `φ_ε` is the characteristic function of the noise. The one assumption I need on the noise is that `φ_ε` is nonzero almost everywhere (its zero set has measure zero); then I can cancel it and conclude `e^{is x̄} = e^{is x̃}` for almost all `s`, hence `x̄ = x̃`.

Let me make sure that chain isn't sleight of hand, because "equal noisy densities ⟹ equal clean signals" is exactly the kind of step that's easy to assert and easy to get wrong. Take Gaussian noise `σ=0.5`, and two candidate clean signals `x̄=0` and `x̃=0.3`. The two observed densities are then Gaussians centred at `0` and `0.3`. If the lemma is real, these must be *distinguishable* — equal observed densities should not be possible unless the centres coincide. Numerically integrating `|p_1 − p_2|` over the line gives an `L¹` distance of about `0.47` for the `0`-vs-`0.3` pair, and `0` (to machine precision) for the `0`-vs-`0` pair. So shifting the clean signal does move the observed density; you cannot get equal noisy densities from unequal clean signals. And the cancellation step is legitimate here because the Gaussian characteristic function `φ_ε(s)=e^{−σ²s²/2}` never hits zero — its minimum over `s∈[−30,30]` is about `1.4×10^{−49}`, small but strictly positive, so dividing it out is allowed. Good: the denoising step holds, and `g_ℓ(F_ℓᵀ Z) = ĝ_ℓ(F̂_ℓᵀ Ẑ)` for all `ℓ, t` — the noise is gone. This "denoising via characteristic functions" is a known device from the variational-autoencoder/nonlinear-ICA identifiability literature; on its own it only gets me to equality of the clean signals, and everything that makes the latents identifiable has to come from what I now do with *space*.

Take the easy case first to feel the mechanism: `g = ĝ = Id`, the linear map. Then denoising says `F_ℓᵀ Z = F̂_ℓᵀ Ẑ` at every location, i.e. `Σ_j F_{ψ_j}(ℓ) Z_j − Σ_j F_{ψ̂_j}(ℓ) Ẑ_j = 0` for all `ℓ` and all `t`. This is a linear combination of the factor functions that vanishes *identically over the whole domain*. The claim I want to extract is that linear independence of the factor functions forces the coefficients to match — but let me actually run this on a toy before believing it, because it's the load-bearing step and the place the whole argument could be hollow.

Put `D=2` RBF factors on a fine 1-D grid, centres at `0.3` and `0.7`, width `0.02`, and a true latent vector `Z=(1.3, −0.8)` at one timestep. The clean field is `X(ℓ)=F_ℓᵀ Z`. Now I play the adversary: using the *same* kernel family, can I find any other coefficient vector `Ẑ` that reproduces `X(ℓ)` at every grid point? Least-squares against the field gives `Ẑ=(1.3, −0.8)` back, with a max reconstruction error around `9×10^{−16}` — exact to machine precision, no second solution. And the reason is visible in the Gram matrix `FᵀF`: its eigenvalues come out `≈69.4` and `≈72.0`, both strictly positive, so the two factor functions are linearly independent on the grid and the only coefficient vector producing the field is the true one. That's the mechanism, made concrete: independence ⟹ unique coefficients ⟹ recovered latents.

The check also tells me exactly when it would fail, which is reassuring rather than alarming. Collapse the two centres onto the same point `0.5` so the factors become identical: the Gram matrix's eigenvalues become `≈0` and `≈141`. The zero eigenvalue is the linear dependence, and with it `Ẑ` is no longer unique — any shift along the null direction reproduces the same field. So the recovery lives or dies on the factor functions being a linearly independent family; collided kernels are the failure mode, and distinct-centre RBFs avoid it.

With that confidence, the general argument is clean. Suppose the true and learned factor index-sets were completely disjoint — no shared functions. Then all `2D` functions are distinct members of one linearly independent family, and a vanishing combination forces every coefficient to be zero: `Z_j = 0` and `Ẑ_j = 0` for all `j, t`. But that contradicts non-degeneracy — none of my latent series is identically zero. So the index sets must overlap. Sharpen it: let `V` be the set of matched pairs `(i,j)` with `ψ_i = ψ̂_j`, let `I` be the matched true-indices. Regroup the vanishing sum into the unmatched-true terms, the unmatched-learned terms, and the matched terms `Σ_{j∈I} F_{ψ_j}(ℓ)(Z_j − Ẑ_{V(j)})`. If any true index were unmatched, linear independence would force its `Z_j = 0` — again contradicting non-degeneracy. So every index matches: the families are equal, and on the matched indices `Z_j = Ẑ_{V(j)}`, i.e. `Z` and `Ẑ` agree up to a permutation. Linear identifiability, no instantaneous/sparsity/variability assumption anywhere — paid for entirely by `L>>D` linear independence, exactly as the toy showed.

Now the real prize: nonlinear, invertible `g_ℓ`. Denoising gives `g_ℓ(F_ℓᵀ Z) = ĝ_ℓ(F̂_ℓᵀ Ẑ)`, and since `g_ℓ` is a diffeomorphism I can invert it: `F_ℓᵀ Z = (g_ℓ^{-1} ∘ ĝ_ℓ)(F̂_ℓᵀ Ẑ) =: h_ℓ(F̂_ℓᵀ Ẑ)` for all `ℓ, t`. So a *linear* readout of `Z` equals a *nonlinear* `h_ℓ` of a linear readout of `Ẑ`, simultaneously at every point. That over-constraint is what overdetermination gives me, and I'm going to squeeze it until `h_ℓ` is forced to be affine and the linear map between `Z` and `Ẑ` is forced to be permutation-scaling.

First I need enough good evaluation points. Build the `D×D` matrix `M_F(ℓ_1,…,ℓ_D)` whose columns are `F_ℓ` at `D` chosen locations. Its determinant is a polynomial in the factor functions, hence real analytic; and because the family is linearly independent it isn't identically zero (you can always find points making it full rank). A nonzero real-analytic function has a zero set of measure zero — so the set `Φ_F` of `D`-tuples where `M_F` is full rank has full measure, and (being the complement of a closed zero-set) is open. Same for `Φ_F̂`. Their intersection therefore has full measure, is nonempty, and is open. So I can pick `D` locations where *both* `M_F` and `M_F̂` are invertible, and I can wiggle them within an open ball.

Pick such a tuple. Stacking `F_ℓᵀ Z = h_ℓ(F̂_ℓᵀ Ẑ)` over the `D` points gives `M_Fᵀ Z = 𝓗(M_F̂ᵀ Ẑ)` where `𝓗` applies the `h`'s componentwise. Invert: `Z = (M_Fᵀ)^{-1} 𝓗(M_F̂ᵀ Ẑ) =: Θ_ℓ(Ẑ)`. Since the `h`'s are invertible and the matrices invertible, `Θ_ℓ` is an invertible map between `Z` and `Ẑ`. Crucially, I could have used any other valid tuple `ℓ'` in the intersection and gotten `Z = Θ_{ℓ'}(Ẑ)` — *the same* `Z`. So `Θ_ℓ = Θ_{ℓ'}` as maps on the open set. That equality across nearby evaluation points is the lever.

Look at the Jacobian. By the chain rule, `J_{Θ_ℓ}(Ẑ) = (M_Fᵀ)^{-1} 𝓗'(M_F̂ᵀ Ẑ) M_F̂ᵀ`, where `𝓗'` is the diagonal of `h'_{ℓ_k}`. Take `log|det|`:

`log|det J_{Θ_ℓ}| = log|det (M_Fᵀ)^{-1}| + log|det M_F̂ᵀ| + Σ_k log|h'_{ℓ_k}(F̂_{ℓ_k}ᵀ Ẑ)|.`

Differentiate with respect to `Ẑ_i`. The two `det` terms don't depend on `Ẑ`, so they drop, and the chain rule on the sum gives

`∂/∂Ẑ_i log|det J_{Θ_ℓ}| = Σ_k [ h''_{ℓ_k}/h'_{ℓ_k} ](F̂_{ℓ_k}ᵀ Ẑ) · F_{ψ̂_i}(ℓ_k).`

Now use that the map — and hence its Jacobian and this derivative — is the same if I perturb one evaluation point. Move `ℓ_1` to `ℓ_1 + δu` for a small `δ` and any unit direction `u`, keeping the others fixed; both tuples lie in the open intersection, so the derivative is unchanged. Subtracting the two expressions, every term `k ≥ 2` cancels and I'm left with: the single quantity

`Γ_i(Ẑ) := [ h''_ℓ / h'_ℓ ](F̂_ℓᵀ Ẑ) · F_{ψ̂_i}(ℓ)`

takes the same value at `ℓ_1` and at `ℓ_1 + δu`. Since `δ` and `u` were arbitrary in the ball, `Γ_i` is *constant in `ℓ`* over an open neighbourhood. Do the same differentiating in `Ẑ_j` for another index `j`: `Γ_j` is also constant in `ℓ`. Two functions that are each constant in `ℓ` satisfy `F_{ψ̂_i}(ℓ) Γ_j = F_{ψ̂_j}(ℓ) Γ_i` across the ball. But `F_{ψ̂_i}` and `F_{ψ̂_j}` are linearly independent — and they stay linearly independent restricted to any open set, because an analytic function vanishing on an open set vanishes everywhere (the identity theorem), so a nontrivial linear dependence on the ball would propagate to a global one, contradiction. Linear independence of the two factor functions forces the constants to vanish: `Γ_i = Γ_j = 0`. Repeating over all index pairs, `Γ_i ≡ 0` for every `i`.

`Γ_i ≡ 0` means `[h''_ℓ/h'_ℓ](F̂_ℓᵀ Ẑ) · F_{ψ̂_i}(ℓ) = 0` for all `i` and all `ℓ` in the ball. If `h''_{ℓ_0}` were nonzero at some point, then every `F_{ψ̂_i}(ℓ_0)` would have to be zero. But replacing the first evaluation point in the valid tuple by that nearby `ℓ_0` would give `M_F̂` a zero column, so the tuple could not remain in the full-rank open set. Contradiction. Therefore `h''_ℓ ≡ 0` on the reachable scalar arguments: each `h_ℓ` is affine in its input. The nonlinear ambiguity has been squeezed down to an affine one.

I have to be precise about what remains. From `h''=0` alone I get `h_ℓ(y)=a(ℓ)y+b(ℓ)`, and a location-dependent slope `a(ℓ)` would be a real extra gauge: it could be absorbed into the spatial factors and then undone by the pointwise decoder. The permutation-scaling conclusion needs that gauge ruled out; otherwise the argument only reaches "affine ambiguity" and the factor family could still move under a common spatial multiplier. So the admissible kernel family has to include the no-gauge condition I actually need: no nonconstant common spatial multiplier may take a full-rank `D`-tuple of learned kernel factors back into the same admissible factor family on an open set. With that condition, the affine slope is spatially constant on the open set; any per-coordinate constants left over are exactly the latent scalings.

Now the last step is clean. Differentiate the constraint `F_ℓᵀ Z = h_ℓ(F̂_ℓᵀ Ẑ)` with respect to `Ẑ_i`: `Σ_k F_{ψ_k}(ℓ) ∂Z_k/∂Ẑ_i = c_i F_{ψ̂_i}(ℓ)` on the ball, with `c_i` absorbing the allowed constant scaling for coordinate `i`. The left side is a linear combination of the *true* factor functions `{F_{ψ_k}}`, which are linearly independent on the ball. So the equation can hold only if `F_{ψ̂_i}` equals one of the `F_{ψ_k}` — call it `k(i)` — and then `∂Z_k/∂Ẑ_i` is nonzero for exactly that single `k = k(i)` and zero for all others. If `F_{ψ̂_i}` matched none of the `F_{ψ_k}`, linear independence would force every `∂Z_k/∂Ẑ_i = 0`, making that Jacobian column zero and contradicting invertibility of `Θ`. So each column of the Jacobian has exactly one nonzero entry: the Jacobian relating `Z` and `Ẑ` is a permutation times a diagonal scaling. Hence `Z = P S Ẑ` and the factor families coincide. Identifiability, up to the unavoidable permutation and scaling, for nonlinear invertible mixing — provided the analytic kernel family does not admit the extra common-multiplier gauge. No "no instantaneous effects", no sparsity, no sufficient variability. The overdetermination `L>>D` did the work.

That settles the question I deferred. The two properties the argument actually leaned on — linear independence of the factor family (it carried the linear case and reappeared at every step) and real-analyticity (it gave me the full-measure set of good evaluation tuples and the identity-theorem propagation) — are exactly the two properties RBFs have, on top of the locality and cheap parameters I originally wanted them for. So the kernel choice wasn't only cosmetic; the same structural facts that make a kernel interpretable are the ones the identifiability rests on. I'd expect, then, that the result isn't special to the Gaussian bump — any linearly-independent analytic family, a Matérn say, ought to work the same way, and that's a prediction I could test the same way I tested the linear case, by checking the Gram matrix of a candidate family stays full-rank. Of course the argument lives in the continuous-domain idealization; on a finite grid the measure-zero pathologies don't literally vanish, but with `L>>D` the system is overdetermined and those pathologies are probabilistically negligible — which is the regime I'm in, and which the `D=2` toy already showed concretely (a full-rank Gram on a finite grid of 400 points).

The theorem identifies the *latents and factors*. Recovering the *causal graph* among the latents is then a separate guarantee inherited from the temporal SCM I plugged in — its own identifiability holds under causal stationarity, the Markov property, minimality, sufficiency, a well-defined density, and the smoothness/non-invertibility conditions on `f` and `g`. Identify the latents first, then the SCM identifies the graph on them.

Let me now put the pieces into code, grounded in how the model actually runs. The spatial factors as a variational module — Gaussians on centre and scale, sigmoid the centre into the grid, build the anisotropic precision-like matrix, exponentiate the Mahalanobis form:

```python
class SpatialFactors(nn.Module):
    # q(F): each factor = an RBF kernel with a learned center rho and
    # an anisotropic precision-like matrix P = A A^T + diag(exp(B)).
    def __init__(self, num_variates, num_nodes, nx, ny, spherical=False, simple=False):
        self.spherical, self.simple = spherical, simple
        self.rho_mu     = nn.Parameter(torch.zeros(num_variates, num_nodes, 1, 2))
        self.rho_logvar = nn.Parameter(torch.zeros(num_variates, num_nodes, 1, 2))
        gamma_dim = 1 if (spherical or simple) else 6                 # 6 = 4 for A, 2 for B
        self.gamma_mu     = nn.Parameter(torch.zeros(num_variates, num_nodes, 1, gamma_dim))
        self.gamma_logvar = nn.Parameter(torch.zeros(num_variates, num_nodes, 1, gamma_dim))
        self.grid_coords = create_grid(nx, ny)[None, None].expand(num_variates, num_nodes, -1, -1)
        self.sigmoid = nn.Sigmoid()

    def get_spatial_factors(self):
        centers = self.sigmoid(reparameterize(self.rho_mu, self.rho_logvar))  # center in [0,1]^2
        scale   = reparameterize(self.gamma_mu, self.gamma_logvar)
        if self.spherical or self.simple:
            scale = torch.exp(scale)
        distance_mode = 'Haversine' if self.spherical else 'Euclidean'
        grid_dist = calculate_distance(self.grid_coords, centers, distance_mode=distance_mode)
        if self.simple:
            exponent = torch.sum(-torch.square(self.grid_coords - centers) / scale.expand(-1, -1, -1, 2), dim=-1)
            return torch.exp(exponent)
        if self.spherical:
            P = scale                                                       # scalar Haversine precision
        else:
            A = scale[..., :4].view(*scale.shape[:-1], 2, 2)
            B = scale[..., 4:]
            P = A @ A.transpose(-1, -2) + torch.diag_embed(torch.exp(B))   # PD, anisotropic
        exponent = -0.5 * torch.einsum('...ik,...kl,...il->...i', grid_dist, P, grid_dist)
        return torch.exp(exponent)               # F_{l,d}, shape (V, D, nx*ny)

    def calculate_entropy(self):
        # closed-form Gaussian entropies of the rho and gamma posteriors
        ...
```

The graph posterior — three-way categorical for the instantaneous slice so I never sample both directions, Bernoulli for lagged slices, hard Gumbel samples, sparsity = sum of edges, acyclicity = trace-exp penalty:

```python
class TemporalAdjacencyMatrix(ThreeWayGraphDist):
    def __init__(self, input_dim, lag):
        super().__init__(input_dim)                       # 3-way logits for instantaneous
        self.logits_lag = nn.Parameter(torch.zeros(2, lag, input_dim, input_dim))

    def sample_graph(self):                               # Gumbel-softmax, hard
        adj = torch.zeros(self.lag + 1, self.input_dim, self.input_dim)
        adj[0]  = self._triangular_vec_to_matrix(
                     F.gumbel_softmax(self.logits, tau=1.0, hard=True, dim=0))   # i->j / j->i / none
        adj[1:] = F.gumbel_softmax(self.logits_lag, tau=1.0, hard=True, dim=0)[1]
        return adj

    def calculate_sparsity(self, G):       return torch.sum(G)              # sparsity prior
    def calculate_dagness_penalty(self, G0):                                # NOTEARS h(G0)
        return torch.trace(torch.matrix_exp(G0)) - G0.shape[-1]
    def entropy(self):                     ...                              # categorical + Bernoulli
```

The latent SCM — per-node embeddings, encode each (history) node, aggregate by the lag-flipped adjacency, run through the outer net; with a conditional-spline-flow likelihood scoring the residual `Z^t − f(Pa)`:

```python
class RhinoSCM(nn.Module):
    def __init__(self, embedding_dim, lag, num_nodes):
        self.embeddings = nn.Parameter(torch.randn(lag+1, num_nodes, embedding_dim)*0.01)
        self.f = MLP(2*embedding_dim, 1, embedding_dim, layers=2)
        self.g = MLP(embedding_dim+1, embedding_dim, embedding_dim, layers=2)

    def forward(self, Z, A):                              # Z: (batch, lag+1, num_nodes)
        E = self.embeddings.expand(Z.shape[0], -1, -1, -1)
        X_enc = self.g(torch.cat((Z.unsqueeze(-1), E), dim=-1))      # encode each parent
        X_sum = torch.einsum("lij,blio->bjo", A.flip([0]), X_enc)    # edge-gated aggregate
        X_sum = torch.cat([X_sum, E[:, 0]], dim=-1)
        return self.f(X_sum).squeeze(-1)                             # predicted Z^t  (= f(Pa))
# likelihood: -log p_spline-flow( Z^t - f(Pa) | history )  -> the log p(Z|G) term
```

The decoder `g_ℓ` — multiply factors by latents, then a shared MLP with a per-grid-point embedding so the pointwise nonlinearity can vary across the grid:

```python
class SpatialDecoderNN(nn.Module):
    def __init__(self, nx, ny, num_variates, embedding_dim, lag, num_nodes):
        self.embeddings = nn.Parameter(torch.randn(num_variates, nx*ny, embedding_dim)*1e-1)
        self.g = MLP(embedding_dim + 1, 1, hidden_dim=64, num_layers=2)
    def forward(self, Z, F):
        X_hat = torch.einsum("bd,vdl->bvl", Z, F)        # [F Z]_l
        X_in  = torch.cat((X_hat.unsqueeze(-1),
                           self.embeddings.expand(X_hat.shape[0], -1, -1, -1)), dim=-1)
        return self.g(X_in).squeeze(-1)                  # g_l([F Z]_l)
```

And the assembly — encode `X→Z`, sample `G`, run the SCM, decode through `F`, and form the minimized negative-ELBO terms: reconstruction MSE, the β-weighted latent negative log-likelihood plus encoder `E_q log q`, the graph sparsity/entropy/acyclicity, and the factor entropy:

```python
class SPACY(nn.Module):
    def forward(self, X):
        Zp = self.f_tilde(time_lag(X, self.lag))         # encoder q(Z|X): mean, logvar
        Z_mean, Z_logvar = Zp[..., :D], Zp[..., D:]
        Z = reparameterize(Z_mean, Z_logvar)
        G = self.temporal_graph_dist.sample_graph()      # q(G), Gumbel-softmax
        Z_hat = self.scm_model(Z, G)                     # f(Pa) under sampled G
        F = self.spatial_factors.get_spatial_factors()   # q(F)
        X_hat = self.spatial_decoder(Z[:, -1], F)        # g_l(F Z)
        return X_lag, X_hat, Z_mean, Z_logvar, Z_hat, Z, G, F

    def compute_loss_terms(self, ...):
        recon = torch.sum((X_hat - X_true)**2) / batch                       # -log p(X|Z,F)
        cd    = self.scm_model.calculate_likelihood(Z[:,-1], Z_hat, Z[:,:-1], G)  # -log p(Z|G)
        z_entropy = -gaussian_entropy(Z_logvar)                              # E_q log q(Z|X), constants aside
        kl_term = (D/4) * (cd + z_entropy)                                   # beta-weighted negative-ELBO term
        sparsity = self.graph_sparsity_factor * self.temporal_graph_dist.calculate_sparsity(G)
        dag      = self.temporal_graph_dist.calculate_dagness_penalty(G[0]) / total_fragments
        g_ent    = -self.temporal_graph_dist.entropy() / total_fragments
        f_ent    = self.spatial_factors.calculate_entropy() / total_fragments
        return recon + sparsity/total_fragments + kl_term + g_ent + f_ent, dag
# total loss = ELBO terms, with dag fed to an augmented-Lagrangian wrapper that ramps it to 0;
# train: freeze SCM + graph for the first ~200 epochs (let F and the encoder settle), then unfreeze.
```

The causal chain, end to end: the grid is too big and too spatially redundant to do causal discovery on directly, so I discover causality among a few latents instead; to keep the latents causally meaningful I infer them *jointly* with the graph rather than reducing first; to aggregate correlated grid points into coherent, interpretable, parameter-cheap regions I parametrize the latent-to-grid map as RBF kernels and let them overlap; I score the latents' dynamics with a differentiable nonlinear temporal SCM that admits instantaneous edges and history-dependent noise; I tie it all together with a variational lower bound and learn the graph with Gumbel-softmax samples under a trace-exponential acyclicity constraint driven by an augmented Lagrangian; and the reason any of this recovers the *true* latents — without the usual no-instantaneous / sparsity / sufficient-variability crutches — is that pushing the grid to a continuous domain turns `L>>D` into an overdetermined system whose linearly-independent, real-analytic factor functions, after denoising the additive noise via characteristic functions, force the map between any two distribution-matching models to be affine and then permutation-scaling.
