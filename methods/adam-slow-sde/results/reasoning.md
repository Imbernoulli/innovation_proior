OK, let me think this through from scratch. The thing that actually bugs me is a mismatch between theory and practice. Everything I understand about implicit bias near a manifold of minimizers is about SGD, but nobody trains anything with plain SGD anymore — it's Adam, or RMSProp, or one of the block/layer/Kronecker variants. So whatever story I have about which flat solution gets selected is a story about an optimizer that isn't used. I want the story for the adaptive ones. Let me first get very concrete about what the SGD story even is, because I'll need every piece of its machinery, and then I'll find out exactly where it breaks when I bolt on a preconditioner.

The setup I trust is this. The loss `L` is over-parameterized, so its global minimizers aren't a point but a connected set. I'll model that set as a smooth compact submanifold `Γ` of dimension `d−m`, with `∇²L` having rank exactly `m` on `Γ` — full rank only in the `m` normal directions, zero in the `d−m` tangent directions. That's not a wild assumption: people see mode connectivity, and language-model training looks like it lives in a long flat river valley whose floor is a continuum of near-minima. Once the learning rate `η` is small enough, a converged run is pinned near some such `Γ`.

Now the two-phase picture. From a generic start, SGD spends roughly `Õ(η⁻¹)` steps barreling down onto `Γ`, and then it doesn't stop — it can't, because there's gradient noise and the gradient itself has essentially vanished, so it just jitters. Over a much longer stretch, `O(η⁻²)` steps, that jitter has a slow, systematic component, and *that* is the implicit bias. The trouble with the usual SDE `dθ = −∇L dt + √η Σ^{1/2} dW` is that it's a faithful picture of the *descent*, the first phase. Try to push it into the second phase and the error you accumulate is unbounded, because the interesting motion is `O(η²)`-small per step and the SDE's own discretization error swamps it. So the whole game in the manifold phase is to stop tracking the raw iterate and instead track its *projection* onto `Γ`.

The projection is the clean idea. Define `Φ(x)` = where the deterministic gradient flow `ẋ = −∇L(x)` starting at `x` ends up; if that limit lands on `Γ`, that's `Φ(x)`. Two facts make this the right object. First, `Φ` is constant along a gradient-flow trajectory, so `∂Φ(x)·∇L(x) = 0` — differentiating `Φ(x(t)) = const` once gives exactly that. Second, restricted to `Γ`, `∂Φ(ζ)` is the orthogonal projection onto the tangent space `T_ζΓ`. (Take a curve inside `Γ` through `ζ` with velocity `w ∈ T_ζΓ`; since `Φ` is identity on `Γ`, `∂Φ(ζ)w = w`. And for a normal direction, follow it down the flow — it gets killed.) So if I look at `φ_k := Φ(θ_k)` instead of `θ_k`, the fast normal-direction relaxation is divided out and only the slow tangential wandering survives. That's the device that buys the long horizon.

What's the slow drift of `φ_k` for SGD? This is where I have to actually do the giant-step calculation, because I'll be reusing its skeleton later. Group the steps into "rounds" of `H = 1/η` steps and bundle rounds into a "giant step" of `R_grp = η^{−β}` rounds for some `β∈(0,½)`, so a giant step is `η^{−1−β}` steps. Over a giant step I want the first and second moments of `Δφ`. Write `θ_k = φ + x_k` with `x_k` the small normal offset, `O(√(η log(1/η)))`. Expand `Φ(θ_{k+1}) − Φ(θ_k)` to second order in the step `θ_{k+1}−θ_k = −η g_k`. The first-order piece `∂Φ·(−η g_k)`: its expectation involves `∂Φ·∇L`, which is zero, so the mean gradient contribution to the drift cancels — the only thing that survives at first order is the noise `∂Φ·(−η z_k)`, which is mean-zero and gives the *diffusion*. The second-order piece `½∂²Φ[η g_k, η g_k]` is where the drift comes from: in expectation `η² g_k g_k^T → η²(∇L∇L^T + Σ)`, and the gradient-outer-product part again gets folded away by `∂Φ·∇L = 0`, leaving `½η²∂²Φ[Σ]`. Sum over `H = 1/η` steps per round and the per-round drift is `~½η·∂²Φ[Σ]`; sum the rounds and over the giant step the mean change of `φ` is `½η^{1−β}·∂²Φ[Σ]` plus lower order, and the second moment is `η^{1−β}·Σ_∥` with `Σ_∥ = ∂Φ Σ ∂Φ` the tangent-projected covariance. Reading these as a drift and a diffusion with effective time `η^{1−β}` gives the slow SDE

`dζ = ∂Φ(ζ) Σ^{1/2}(ζ) dW + ½ ∂²Φ(ζ)[Σ(ζ)] dt`.

Now I want the drift in a form I can interpret. There's an identity for `∂²Φ` — differentiate `∂Φ·∇L = 0` again and use `∇²L` — that turns `½∂²Φ[Σ]` into `−½ ∇³L[Σ̂_◇]` projected to the tangent space, where `Σ̂_◇` is the *normal*-space part of the covariance reweighted by `1/(λ_i+λ_j)` over Hessian eigenpairs (the Lyapunov-type weighting that comes from solving the fast OU relaxation in the normal directions). So the SGD slow SDE is

`dζ = P_ζ( Σ_∥^{1/2} dW − ½ ∇³L(ζ)[Σ̂_◇(ζ)] dt )`,

with `P_ζ` the operator that keeps the projected step on `Γ`. And the drift is the negative *semi*-gradient of `μ(ζ) = ⟨∇²L(ζ), Σ̂_◇(ζ)⟩` — "semi" because if you write `μ(ζ₁,ζ₂) = ⟨∇²L(ζ₁), Σ̂_◇(ζ₂)⟩`, the drift is `−½∇_{ζ₁}μ(ζ,ζ)`, holding the covariance argument frozen. So SGD wanders on `Γ` doing semi-gradient descent on a sharpness measure made from the Hessian and the noise. Good — that's the whole SGD story, and I had to relive it because I'm about to reuse the giant-step moment lemmas and the `∂²Φ` identity verbatim.

The special clean case is label noise. If at every step you add fresh `±δ` to the label of an over-parameterized `ℓ₂` regression, then on the zero-loss manifold the noise covariance is exactly proportional to the Hessian: `Σ(ζ) = E[ζ_t² ∇h∇h^T] = δ²∇²L(ζ) = α∇²L(ζ)`. Plug that into the slow SDE and the diffusion vanishes — because `Σ^{1/2}x` then points purely along normal directions, and the projection `∂Φ` annihilates normal directions — so the stochastic process collapses to a deterministic flow whose fixed points satisfy `∇_Γ tr(∇²L) = 0`. SGD under label noise minimizes `tr(H)` along `Γ`. That's the benchmark I want to beat or at least separate from.

Now the actual question: what changes when I replace SGD by an adaptive method? Let me write the update I care about as generically as I can so I do this once for the whole family. Momentum:

`m_{k+1} = β₁ m_k + (1−β₁) g_k`,

a second-moment state built from the gradient outer product through some *linear* map `V` (for Adam, `V(g g^T) = diag(g g^T) = g⊙²`; for block/layer variants, `V` averages diagonal entries inside a block; for the Kronecker method, `V` returns the two factor matrices):

`v_{k+1} = β₂ v_k + (1−β₂) V(g_k g_k^T)`,

and a preconditioned step where a smooth map `S` turns the second-moment state into a positive-definite matrix:

`θ_{k+1} = θ_k − η S(v_{k+1}) m_{k+1}`.

For Adam, `S(v) = Diag(1/(√v + ε))`. RMSProp is just `β₁ = 0`. I'll require `S` to be `ρ_s`-smooth, to map nonnegative inputs to positive-definite matrices, and to be uniformly bounded below, `S(v) ⪰ I/R₀`, which I'll need for descent. This single template covers Adam, RMSProp, the block and layer variants, and the Kronecker method. If I can run the manifold analysis on this template I get all of them at once.

So where does the SGD machinery break? Two places, and I need to be honest about both before I can fix anything.

First wall. The SGD slow-SDE derivation quietly used rotational equivariance. Plain SGD treats all directions identically, so I could pretend the Hessian is diagonal and `Γ` is a coordinate subspace, which is what made the giant-step moment formulas tractable. The preconditioner `S(v)` destroys that symmetry — it singles out coordinates — so I cannot diagonalize my way to the clean formulas. The projection `Φ` itself feels wrong now: the optimizer's "clean" converging direction isn't `−∇L`, it's `−S∇L`. Following gradient flow to define the projection would project along the wrong direction.

Let me chase that. Suppose I naively kept the gradient-flow projection `Φ`. Then on `Γ`, `∂Φ` projects orthogonally, but the optimizer's drift lives in `S`-skewed directions, and the cancellation `∂Φ·(drift) = 0` that I leaned on — the thing that killed the mean-gradient term — no longer holds, because the analogous identity would need `∂Φ·S∇L = 0`, not `∂Φ·∇L = 0`. So the divergent `η⁻¹` term doesn't cancel and the whole long-horizon trick falls apart. The projection has to follow the preconditioned flow, not the gradient flow.

So redefine it: `Φ_S(x)` = limit of the *preconditioned* flow `ẋ = −S∇L(x)`. Differentiating `Φ_S(x(t)) = const` along this flow gives the identity I actually need:

`∂Φ_S(x)·S∇L(x) = 0`,

and differentiating once more,

`∂²Φ_S(x)[S∇L, S∇L] = −∂Φ_S(x) S∇²L(x) S∇L(x)`.

Good, those are the analogues of the SGD identities, with `S` inserted exactly where it should be. And I can also work out how `∂Φ_S` acts on tangent and normal vectors at `ζ∈Γ`: for `w∈T_ζΓ`, since `Φ_S` is identity on `Γ`, `∂Φ_S(ζ)w = w`; for a normal vector `u`, I push `ζ` infinitesimally along `∇²L(ζ)^† u` so that `∇L` picks up exactly `tu + o(t)`, apply the first identity, divide by `t`, and get `∂Φ_S(ζ) S u = 0`. Combining, `∂Φ_S(ζ) S ∇²L(ζ) = 0`: the preconditioned-projection Jacobian kills `S` times the Hessian. These three little lemmas are the load-bearing facts for everything below.

But I still have the rotational-equivariance problem for the *moment calculation* — the actual giant-step bookkeeping. Fix the preconditioner at the start of a giant step, `S₀ = S(v₀)`, and factor `S₀ = P P` with `P = S₀^{1/2}`. Reparameterize `x' = P^{−1}x`, `L'(x') = L(Px')`. Then `∇L'(x') = P∇L(Px')`, and the frozen preconditioned flow becomes

`ẋ' = P^{−1}ẋ = −P^{−1}S₀∇L(Px') = −P∇L(Px') = −∇L'(x')`.

So in the primed coordinates the preconditioner is gone. The projections are tied by `Φ_S(Px') = PΦ'(x')`: flow in `x'`, map back by `P`, and I land at the same point on `Γ` as the original preconditioned flow. This is the precise way to reuse the SGD giant-step lemmas; the `P` and `P^{-1}` factors have to be transported through the projection relation, not moved by inspection.

Second wall. The preconditioner isn't a constant — `v` keeps moving, so `S(v_k)` drifts, and I just pretended it was frozen at `S₀` over a giant step. Is that legitimate? It depends entirely on how fast `v` moves, i.e. on `β₂`. Let me feel out the two extremes. If `1−β₂` is `Θ(1)` (β₂ far from 1), then `v` re-equilibrates in `O(1)` steps; over a giant step the preconditioner changes by `Θ(1)` and there's no hope of treating it as approximately constant — the moments become intractable. Worse, when `β₂` is so small that `β₂ < β₁²`, these methods are known to outright fail to converge, so that regime isn't even meaningful. The other extreme: `1−β₂` exponentially tiny, `v` essentially frozen for the whole `O(η⁻²)` horizon — then `S` never adapts and the method is just preconditioned-SGD with a fixed matrix, which is boring and not what real Adam does. So I need the in-between rate where `v` moves on *exactly* the same `O(η⁻²)` clock as the manifold wandering: slow enough per step that `S₀` is a good freeze within a single giant step, but accumulating a `Θ(1)` change over the full horizon so that adaptiveness genuinely shapes the bias.

Let me solve for that rate. The slow SDE lives on timescale `t = k η²`. One step changes `v` by `(1−β₂)(V(g g^T) − v)`, an `O(1−β₂)` quantity. Over `O(η⁻²)` steps the accumulated change is `O((1−β₂)·η⁻²)`, and I want that to be `Θ(1)`. So `1−β₂ = Θ(η²)`. Call it the "2-scheme," and set the constant `c := (1−β₂)/η²`. Then within one giant step of `η^{−1−β}` steps, `v` changes by `O((1−β₂)·η^{−1−β}) = O(η^{1−β})`, which is `o(1)` — small enough that freezing `S₀` only costs me lower-order error — yet over the full run it sweeps a `Θ(1)` arc. That single scaling choice resolves the second wall, and it's forced, not chosen: it's the unique rate that matches the preconditioner's clock to the manifold's clock. I'll need to add `v` as a genuine state variable of the SDE.

Before I can trust any of this I have to make sure the method even gets to the manifold and *stays* there with high probability over the whole `O(η⁻²)` horizon — otherwise the projection is undefined and the moment calculus is built on sand. The SGD papers could lean on existing convergence results; for this general adaptive template I can't, because the existing Adam convergence bounds are the wrong shape. They assume convexity, or use `1/√t` step sizes, or bound only the *average* gradient norm, or hold only *in expectation*, or don't go to zero as `η→0`. I need a *high-probability* bound on the *last-iterate* optimality gap `L(θ_K) − L*` that vanishes with `η`, for the whole family, under the 2-scheme. So I'll prove my own.

Let me build it. The natural condition near `Γ` is Polyak–Łojasiewicz: `2μ(L(θ)−L*) ≤ ‖∇L(θ)‖²`. Start with a descent lemma. By smoothness, `L(θ_k) − L(θ_{k−1}) ≤ −η⟨∇L(θ_{k−1}), u_k⟩ + ½ρη²‖u_k‖²` with `u_k = S(v_k)m_k`. The momentum makes `u_k` a geometric blend of past preconditioned gradients; unrolling the recursion `u_k = S(v_k)(β₁ m_{k−1} + (1−β₁)g_{k−1})` and bounding the cross terms by the smoothness of `∇L` and `S` (each one-step change of `θ` is `O(η)`, of `v` is `O(η²)`), the momentum telescopes into

`L(θ_k) − L(θ_{k−1}) ≤ C₂η² − η(1−β₁)Σ_{i=1}^{k} β₁^{k−i}⟨∇L(θ_{i−1}), U_i⟩`,

where `U_i = S(v_{i−1})g_{i−1}` is the *current*-gradient preconditioned step. Now peel each inner product into a deterministic part and a martingale part: with `ṽ_i := β₂ v_{i−1} + (1−β₂)E_{i−1}[V(g g^T)]`,

`⟨∇L(θ_{i−1}), U_i⟩ = ∇L(θ_{i−1})^T S(ṽ_i) ∇L(θ_{i−1}) − Y_i − X_i`,

where `Y_i` is a tiny `O(η²‖∇L‖)` term from `S(v_i)−S(ṽ_i)` (Lipschitz `S` times the `O(η²)` move of `v`), and `X_i = ⟨z_{i−1}, S(ṽ_i)^T∇L(θ_{i−1})⟩` is mean-zero given the past with `|X_i| ≤ C‖∇L(θ_{i−1})‖`. The quadratic form is the good term: since `S(ṽ_i) ⪰ I/R₀`, it's `≥ (1/R₀)‖∇L(θ_{i−1})‖²`. So, dropping the negligible `Y` into the `η²` bucket,

`L(θ_k) − L(θ_{k−1}) ≤ C̃₃η² − η(1−β₁)Σ β₁^{k−i}((1/R₀)‖∇L(θ_{i−1})‖² − X_i)`.

Apply PL to the squared-gradient term, set `γ := 1 − 2ημ(1−β₁)/R₀`, and unroll:

`L(θ_k) − L* ≤ γ^k(L(θ₀)−L*) + η(1−β₁)Σ_i X_i Σ_{j≥i} γ^{k−j}β₁^{j−i} + C₃η`.

The middle term is a weighted martingale sum. Bound it with Azuma–Hoeffding, but carefully — `|X_i|` is controlled by `‖∇L(θ_{i−1})‖`, which by smoothness is `≤ √(2ρ(L(θ_{i−1})−L*))`, i.e. controlled by the very quantity I'm bounding. So I do an induction: assume `L(θ_{i−1})−L* ≤ ψ(i,δ)` for all earlier `i`, mask the martingale wherever that fails (it fails only with the small probability I'm carrying), apply Azuma to the masked sequence, take a union bound, and close the loop. The coefficient sums collapse because `β₁/γ ≤ 0.95` for small `η` (which holds since `β₁ ≤ 0.9` — that's all I need `β₁ ≤ 0.9` for; any constant below 1 works, the constants just change). The geometric sums give `Σγ^{2k−2i} = O(1/(1−γ)) = O(R₀/(μ(1−β₁)η))`, and the `√` of an `η·(1/η)` cancels one power of `η`, leaving

`L(θ_k) − L* ≤ C_{5a}γ^k(L(θ₀)−L*) + C_{5b}η log(K/δ)`

with probability `1−δ`, for all `k ≤ K = O(poly(1/η))`. Set `K = O((1/η)log(1/η))` so `γ^K = O(η)` and the last-iterate gap is `Õ(η)`. That's the convergence theorem — last-iterate, high-probability, vanishing with `η`, for the whole AGM family. (As a side effect, plugging `K = O(η⁻²)` shows the iterate stays within an `Õ(√η)` tube of `Γ` for the *entire* horizon, which is exactly what the manifold analysis needs.)

There's a gap I glossed: PL only holds *near* `Γ`, not globally, and if the iterate ever wandered out I couldn't characterize it. I need to bound the probability of escape. Trick: build a proxy loss `L̃` that equals `L` inside the `ε₁`-tube but adds a quadratic wall `½C(dist(θ,Γ)−ε₁)²` further out. To make this rigorous I need the distance-to-`Γ` and the unit normal to be smooth, which the tubular neighborhood theorem gives on a tube of radius `τ_Γ` (and `∇r = n`, the unit normal). I also need the loss to actually rise as you leave `Γ`: a third-order Taylor expansion of `∇L` at the foot point `φ = P(θ)`, using `∇L(φ)=0` and `∇²L(φ)` positive on the normal space with a uniform eigenvalue floor `m>0` (compactness), gives `⟨∇L(θ), n⟩ ≥ m·r − C r²`, so for `r` below `m/C^{(3)}` the angle between `∇L` and the outward normal is non-obtuse — `Γ^τ` is genuinely a valley with `Γ` at the floor. With that, `L̃` is globally `(μ, L̄)`-PL: in the wall region `‖∇L̃‖² = ‖∇L‖² + 2C(r−ε₁)⟨∇L,n⟩ + C²(r−ε₁)² ≥ ‖∇L‖² + C²(r−ε₁)²` (the cross term is `≥0` by non-obtuseness), and combining with PL of `L` inside gives the PL of `L̃`. On the sublevel set `{L̃ < L_m}` we have `L̃ = L` and `θ ∈ Γ^{ε₁}`. Also, wherever `‖∇L̃‖` is large the loss strictly decreases each step (a no-PL corollary of the descent lemma), and a single step moves `θ` by at most `ηR·R₂`, so for small `η` the iterate can't jump across the safety band. Induct: starting near enough to `Γ`, the run stays in `Γ^{ε₁}` almost surely and the convergence bound applies with `L = L̃ = L`. Good — the foundation is solid; now back to the actual implicit bias.

I redo the giant-step moments, but now in the reparameterized space and with `v` tracked. Within a giant step freeze `S₀ = S(v₀) = PP`. The single-step update, after I show momentum is harmless, is effectively `θ_{k+1} ≈ θ_k − η S₀ g_k + (small)`. Why is momentum harmless? Because after convergence `∇L` crawls — successive `E[∇L(θ_{k+1})] − E[∇L(θ_k)]` is `O(η^{1.5})` (it's `η∇²L·S·m` in expectation, and the conditional mean of `m` is `O(η^{0.5})` since `∇L` is `O(η^{0.5})` post-convergence) — and the momentum only averages the last `O(log(1/η))` gradients, so `E[m_k]` and `E[g_k]` differ by `O(η^{1.5}log(1/η))`, utterly negligible against the `η²` drift. So I can replace `m` by `g` in every moment. (This is the same conclusion as the momentum-is-marginal result for SGD, recovered here by direct moment bounds.) Likewise the `(S(v_k)−S₀)g_k` term is `O(η^{1.5−β})` in expectation because `v` only crept `O(η²·#steps)`. So the leading update really is the fixed-preconditioner one.

Now reparameterize: `θ' = P^{−1}θ`, `L'(x')=L(Px')`. The update of `Pθ'` is `Pθ'_{t+1} = Pθ'_t − ηS₀∇L(Pθ'_t) + (small)`, i.e. `θ'_{t+1} = θ'_t − η∇L'(θ'_t) + (small)` — a plain SGD step on `L'`. So I apply the SGD giant-step moment lemmas to `θ'` with projection `Φ'` and noise covariance `Σ' = PΣP`. The first-moment lemma gives `E[Δφ'] = −(Hη²/2)·∂Φ'·∂²(∇L')[𝒱_{∇²L'}(Σ')] + …` over a round, where `𝒱_H(·) = Σ_{i,j} (λ_i+λ_j)^{−1}⟨·, v_iv_j^T⟩v_iv_j^T` is the Lyapunov reweighting (the closed-form solution of the fast normal OU relaxation). Mapping the projected change back with `Φ_S(Px') = PΦ'(x')`, and using the chain rules for `∇²L'` and `∂²(∇L')`, gives the original-space round drift

`E[Δφ] = (Hη²/2)·S₀·∂²Φ_S(φ)[S₀ Σ S₀] + …`,

and the round second moment

`E[ΔφΔφ^T] = Hη²·S₀ ∂Φ_S(φ) S₀ Σ S₀ ∂Φ_S(φ) S₀ + …`.

Summing `R_grp` rounds over a giant step (and discarding an initial `R₀ = O(log(1/η))` rounds where the OU hasn't mixed, whose contribution is `Õ(η)`), with effective time `η^{1−β}`,

`E[Δφ]^{giant} = (η^{1−β}/2)·S₀ ∂²Φ_S(φ)[S₀ Σ S₀] + …`,
`E[ΔφΔφ^T]^{giant} = η^{1−β}·S₀ Σ_∥ S₀ + …`, with `Σ_∥ = ∂Φ_S S₀ Σ S₀ ∂Φ_S`.

And the `v`-moments. One round changes `v` by `(β₂^H − 1)v + (1−β₂)Σ β₂^{H−i}V(g_ig_i^T)`, and `E[V(g g^T)] = V(Σ(φ)) + O(η^{0.5−0.5β})` since `θ` is within `O(η^{0.5−0.5β})` of `φ`. Telescoping rounds across the giant step,

`E[Δv]^{giant} = cη^{1−β}(V(Σ(φ)) − v) + O(η^{1.5−1.5β})`,

because `1 − β₂^{R_grp H} = cη^{1−β} + O(η^{2−2β})`, not `cη`: the giant step contains `η^{−1−β}` ordinary steps and each contributes a `cη²` relaxation rate. The second moment is `O(η^{2−β})`, negligible — so `v` evolves by a *deterministic* drift, no diffusion. Reading drift and diffusion at effective time `η^{1−β}` off these moments gives the coupled slow dynamics

`dζ = S∂Φ_S(ζ) S Σ^{1/2}(ζ) dW + ½ S ∂²Φ_S(ζ)[S Σ S] dt`,
`dv = c(V(Σ(ζ)) − v) dt`.

Rewriting the `∂²Φ_S` drift with the second `Φ_S` identity into the `∇³L` form and folding the projection into the `P_{ζ,S}` operator, I land on the clean statement:

`dζ = P_{ζ,S(t)}( Σ_∥^{1/2}(ζ;S) dW − ½ S(t) ∇³L(ζ)[Σ_◇(ζ;S)] dt )`,
`dv = c(V(Σ(ζ)) − v) dt`, with `c = (1−β₂)/η²`,
`Σ_◇(ζ;S) = SΣS − Σ_∥`, `Σ_∥(ζ;S) = ∂Φ_S SΣS ∂Φ_S`.

Let me stare at this. It's the SGD slow SDE with three changes, each tracing to one decision I made. The projection is `P_{ζ,S(t)}` — state-dependent, because the converging direction is `−S∇L` — whereas SGD's was fixed. The covariance enters as `SΣS`, filtered through the preconditioner on both sides, instead of bare `Σ`. And there's a brand-new line: `v` is a live state relaxing toward `V(Σ(ζ))` on the same slow clock, which is the whole point of the 2-scheme. The drift is again a negative *semi-gradient*, now of `μ(ζ,v) = ⟨∇²L(ζ), Σ_◇(ζ;S)⟩`, preconditioned by `S(t)` — "adaptive semi-gradient descent on a sharpness measure." And `β₁` is nowhere in it: momentum doesn't touch the bias, exactly as the moment bounds said it wouldn't.

I should make sure the SDE keeps `ζ` on `Γ`. The viability (Nagumo) condition is that the drift minus the Itô correction `½Σ_j D[A_j]A_j` lies in the tangent space, equivalently that the normal projection `P_⊥ = I − ∂Φ_S` annihilates it. Expanding `P_⊥ Σ_j D[A_j]A_j` with `A_j = S∂Φ_S S Σ^{1/2}_j` and using the second-derivative identity for `∂²Φ_S` reduces it to `−P_⊥ ∇²L^† ∂²(∇L)[S Σ_∥]`, which matches `P_⊥` of the drift term term-for-term. So they cancel and `ζ(t)` stays on `Γ`. Good.

Now the payoff is the label-noise reduction, where I can finally see the difference from SGD concretely. With `Σ = αH` (here `H := ∇²L(ζ)`), look at the diffusion: it carries `S∂Φ_S S Σ^{1/2}`. But `Σ^{1/2}x = √α H^{1/2}x` lies in the normal space, and `∂Φ_S S` kills the normal space (that's the lemma `∂Φ_S(ζ)S∇²L(ζ)=0` applied through `H^{1/2}`). So the diffusion vanishes and the slow SDE collapses to an ODE:

`dζ = −(α/2) S_t ∂Φ_{S_t}(ζ) S_t ∇³L(ζ)[S_t] dt`,  `dv = c(V(Σ(ζ)) − v)dt`.

At a fixed point, `v = V(Σ(ζ)) = V(αH) = α·diag(H)` for the Adam choice `V = diag`, and `S = Diag(1/√v) = Diag(1/√(α diag H))` (take `ε=0` first). The fixed-point condition is `S P_∥ ∇³L[S] = 0` with `P_∥ = ∂Φ_S S`. The thing to evaluate is `∇³L(ζ)[S]`. Since `S` is diagonal I only sum diagonal entries:

`∇³L(ζ)[S] = Σ_j (α H_{jj})^{−1/2} ∇H_{jj}`.

Now `(H_{jj})^{−1/2}∇H_{jj} = 2∇(H_{jj}^{1/2})`, so

`∇³L(ζ)[S] = (2/√α) Σ_j ∇(H_{jj}^{1/2}) = (2/√α) ∇ tr(Diag(H)^{1/2})`.

There it is. The Adam fixed-point condition becomes `S P_∥ ∇ tr(Diag(H)^{1/2}) = 0`. The only part of this vector equation that can constrain motion along `Γ` is its tangent component: `P_∥ = ∂Φ_S S` annihilates normal directions and is invertible as a map on the tangent bundle, while the outer `S` is invertible for `ε>0` and is read as the `ε↓0` active-coordinate limit in the clean formula. So the fixed-point condition strips down to

`∇_Γ tr(Diag(H)^{1/2}) = 0`.

So Adam under label noise minimizes `tr(Diag(H)^{1/2})` along the manifold — *not* SGD's `tr(H)`. The square root is the entire difference, and it comes from the `1/√v` preconditioner: where SGD weights each diagonal Hessian entry linearly, Adam divides by `√(diag H)`, so its drift on `H_{jj}` is `H_{jj}^{−1/2}` — which integrates to `H_{jj}^{1/2}` instead of `H_{jj}`. The adaptive denominator is literally the reason for the exponent.

This also hands me a free knob. Nothing forced the exponent to be `½` — that was just Adam's `√v`. Replace `S = Diag(1/(v^λ + ε))` for a tunable `λ ∈ [0,1)` (call it AdamE-λ; `λ=½` is Adam, `λ=0` strips the second moment and is plain momentum-SGD). Rerun the same line: `S = Diag((αH_{jj})^{−λ})`, so `∇³L[S] = Σ_j(αH_{jj})^{−λ}∇H_{jj}`, and `H_{jj}^{−λ}∇H_{jj} = (1/(1−λ))∇(H_{jj}^{1−λ})`, giving `∇³L[S] = (1/((1−λ)α^λ))∇tr(Diag(H)^{1−λ})` and the fixed point `∇_Γ tr(Diag(H)^{1−λ}) = 0`. Tuning the second-moment exponent *exactly* tunes the exponent of the implicit regularizer, continuously interpolating from SGD's `tr(H)` at `λ=0` to Adam's `tr(Diag(H)^{1/2})` at `λ=½`. The structure couldn't be cleaner.

I should check the `ε>0` case, since real Adam keeps `ε` around `10⁻⁸`. Now `S = Diag(1/(√(α H_{jj}) + ε))`. I need an antiderivative of `1/(√(αx)+ε)`: define `ψ(x) = √(αx) − ε ln(√(αx)+ε)`, then `ψ'(x) = α/(2(√(αx)+ε))`, so `1/(√(αx)+ε) = (2/α)ψ'(x)`. Hence `∇³L[S] = (2/α)∇ tr(ψ(Diag(H)))`, and the fixed point is `∇_Γ tr(Diag(H)^{1/2} − (ε/√α) ln((√α/ε)Diag(H)^{1/2} + I)) = 0`, after dropping the additive constant inside the logarithm. The scalar term `√x − (ε/√α)ln(1 + (√α/ε)√x)` is non-negative because `ln(1+y)≤y`; on bounded Hessian diagonals the correction is `O(ε log(1/ε))`, so for the realistic tiny `ε` the regularizer is `tr(Diag(H)^{1/2})` up to a negligible logarithmic shave. Good — the clean result is the right one to carry.

What does `tr(Diag(H)^{1/2})` actually *do*? I need a setting where I can read off the Hessian and connect the exponent to generalization. The diagonal linear network is perfect: `θ = (u,v)`, estimate `ŵ = u⊙² − v⊙²`, loss `½(⟨z_i, ŵ⟩ − y_i)²`, ground truth `κ`-sparse, `d ≫ n`. Compute the Hessian on `Γ`: differentiating twice, the term proportional to the residual `⟨z_i,ŵ⟩−y_i` vanishes on the zero-loss manifold, leaving `∇²L(θ) = (4/n)Σ_i (z_i⊙u, −z_i⊙v)(...)^T`, whose diagonal is `diag(∇²L) = 4θ⊙²`, i.e. `4(u_i², v_i²)`. So `tr(Diag(H)^{e}) ∝ Σ_i(|u_i|^{2e} + |v_i|^{2e})`. The loss only sees `u_i² − v_i²`, so if both `u_i, v_i` are nonzero at an optimum I can shrink them keeping `u_i²−v_i²` fixed and strictly lower `Σ(|u_i|^{2e}+|v_i|^{2e})`, which means it wasn't a minimizer of the regularizer after all. So at the optimum `u_i = 0` or `v_i = 0` for every `i`, and then `Σ(|u_i|^{2e}+|v_i|^{2e}) = Σ|u_i²−v_i²|^{e} = ‖ŵ‖_e^e`. Minimizing `tr(Diag(H)^e)` on `Γ` is exactly minimizing the `ℓ_e` quasi-norm through its monotone `e`-th power. So: SGD (`e=1`) recovers the min-`ℓ₁` interpolator; Adam (`e=½`) recovers the min-`ℓ_{0.5}` interpolator; AdamE-λ gives `ℓ_{1−λ}`. Just as lasso beats ridge for sparse recovery, the sub-`ℓ₁` penalty `ℓ_{0.5}` is even more sparsity-promoting — so I'd expect Adam, and even AdamE with a small positive `λ`, to recover a sparse ground truth from *fewer* samples than SGD. That's the prediction I'd want to validate: sweep `n_train`, watch the test loss collapse earlier for Adam than for SGD.

But I should be honest about when the `tr(Diag(H)^{1/2})` story is even the *right* story, because the diagonal of `H` is only meaningful as "the spectrum" when `H` is (close to) diagonal — which the diagonal net arranges in expectation. If `H` is far from diagonal, `tr(Diag(H)^{1/2})` is not `‖eigenvalues‖_{0.5}`, and there's no reason the bias should help. Deep matrix factorization is the stress test: there, SGD's `tr(H)` is known to track the nuclear norm of the product matrix, which favors low rank and generalizes well on a low-rank ground truth. Adam's `tr(Diag(H)^{1/2})` does *not* reduce to the nuclear norm, so I'd predict Adam drives `tr(Diag(H)^{1/2})` down while leaving `tr(H)` high (even non-monotone), converging to a higher-`tr(H)`, worse-generalizing solution than SGD. A separation in the opposite direction — same theory, but now the unique bias *hurts*. That asymmetry is exactly what makes "Adam reduces a *unique* form of sharpness" the right framing rather than "Adam is just better."

One more family member worth chasing, because it tells me the limits of "explicit regularizer." For the Kronecker-factored method, `S` is `((V_R+εI)⊗(V_L+εI))^{−1/2}`, not diagonal. The question of whether *some* scalar regularizer `ψ` exists with the bias `∇_Γ ψ = 0` is the question of whether the vector field `A(ζ) = ∇³L(ζ)[S(V(Σ(ζ)))]` is conservative — has a potential. For Adam it did (that's how `tr(Diag(H)^{1/2})` popped out). For the Kronecker method, even with diagonal `H`, I can write `S` explicitly via the Kronecker structure and check the curl of `A`: it's nonzero. By Stokes–Cartan a nonzero curl means no potential exists. So the manifold drift is well-defined, but it is not gradient descent on any single scalar regularizer — a qualitatively different, "non-conservative" implicit bias. Worth flagging that the explicit-regularizer picture is a feature of the diagonal preconditioners, not a universal.

Stepping back to write the actual approximation guarantee I've been implicitly claiming. I have the giant-step first/second/sixth moments of the projected state `X̄ = (Φ_S(θ), v)` matching a drift `b` and diffusion `σ` to high order; the standard weak-approximation machinery (one-step moment matching to `O(η_e²)` for the first two moments and a controlled sixth-moment / third-derivative remainder, summed over `T/η_e` giant steps via a telescoping `u_{l,n}` argument with `u` the solution of the backward equation) then gives, for any `C³` test function `g`, `max_k |E[g(X̄_k)] − E[g(X(kη²))]| = Õ(η^{0.25})` over `K = ⌊Tη⁻²⌋` steps — after the `O((1/η)log(1/η))` convergence steps to get onto the manifold. The `0.25` comes out of optimizing the giant-step exponent: with `β` the giant-step parameter, the two surviving one-giant-step error exponents in the effective step size `η_e=η^{1−β}` are `a₁ = (1.5−2β)/(1−β)` and `a₂ = 1/(1−β)`. After summing over `T/η_e` giant steps the exponents drop to `a₁−1=(0.5−β)/(1−β)` and `a₂−1=β/(1−β)`; balancing them gives `β=0.25`, and then `η_e^{1/3}=η^{0.25}`. The `C⁵` smoothness of `L` (and `Σ^{1/2}`) and `C⁴` of `S` are exactly what's needed: the drift/diffusion must be `C⁴` to push `C³` test functions through, and `∂²Φ_S` (which the drift uses) needs `Φ_S ∈ C⁴`, hence `∇L ∈ C⁴`, hence `L ∈ C⁵`.

Now let me write it as code, grounded in the standard Adam update generalized to the `(V,S)` template, plus the diagonal-net experiment exactly as set up. The optimizer's per-step math is just the three lines; for the diagonal instances `V(g g^T) = g⊙²` and `S(v)m = m/(v^λ+ε)`, so no matrix algebra is needed and `λ` is the dial that moves the implicit regularizer's exponent.

```python
import torch

class AGM(torch.optim.Optimizer):
    # m <- b1 m + (1-b1) g                 # momentum: provably doesn't change the bias
    # v <- b2 v + (1-b2) V(g gᵀ)=g⊙²       # second moment; 1-b2 = Θ(η²) is the "2-scheme"
    # θ <- θ - η S(v) m,  S(v)=Diag(1/(v^λ+ε))   # λ=1/2 is Adam; λ tunes the exponent
    def __init__(self, params, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8, lam=0.5):
        super().__init__(params, dict(lr=lr, b1=b1, b2=b2, eps=eps, lam=lam))

    @torch.no_grad()
    def step(self):
        for grp in self.param_groups:
            b1, b2, eps, lam, lr = (grp[k] for k in ("b1", "b2", "eps", "lam", "lr"))
            for p in grp["params"]:
                if p.grad is None:
                    continue
                g, st = p.grad, self.state[p]
                if not st:
                    st["m"], st["v"] = torch.zeros_like(p), torch.zeros_like(p)
                m, v = st["m"], st["v"]
                m.mul_(b1).add_(g, alpha=1 - b1)          # first moment
                v.mul_(b2).addcmul_(g, g, value=1 - b2)   # second moment, V=diag
                p.addcdiv_(m, v.pow(lam).add_(eps), value=-lr)   # θ -= η S(v) m
```

And the diagonal-net diagnostic that the label-noise reduction suggests — SGD selects `ℓ₁`, Adam selects `ℓ_{0.5}`, so I expect Adam to recover the sparse `w*` from fewer samples:

```python
def run_diagnet(opt_name, n_train, d=10000, kappa=50, delta=0.1,
                steps=20000, lr=1e-2, lam=0.5, seed=0):
    g = torch.Generator().manual_seed(seed)
    w_star = torch.zeros(d)
    w_star[torch.randperm(d, generator=g)[:kappa]] = torch.randn(kappa, generator=g)
    Z = (torch.randint(0, 2, (n_train, d), generator=g) * 2 - 1).float()  # z ∈ {±1}^d
    y = Z @ w_star                                                        # clean labels
    Zt = (torch.randint(0, 2, (2000, d), generator=g) * 2 - 1).float()
    yt = Zt @ w_star

    u = torch.full((d,), 0.1, requires_grad=True)   # diagonal-net params: ŵ = u² - v²
    v = torch.full((d,), 0.1, requires_grad=True)
    opt = (torch.optim.SGD([u, v], lr=lr) if opt_name == "sgd"
           else AGM([u, v], lr=lr, lam=lam))        # lam=0.5 → Adam; 0<lam<0.5 → AdamE

    for _ in range(steps):
        i = torch.randint(0, n_train, (1,), generator=g).item()
        noisy = y[i] + delta * (2 * torch.randint(0, 2, (1,), generator=g).item() - 1)
        loss = 0.5 * ((Z[i] * (u**2 - v**2)).sum() - noisy) ** 2   # fresh label noise
        opt.zero_grad(); loss.backward(); opt.step()

    with torch.no_grad():
        test_loss = 0.5 * ((Zt @ (u**2 - v**2) - yt) ** 2).mean()  # recovered if < 1
    return test_loss.item()
```

So the causal chain is: SGD near a manifold does semi-gradient descent on a Hessian-noise sharpness measure, captured by a long-horizon SDE that tracks the *projection* onto the manifold; to carry that to adaptive methods I had to swap the gradient-flow projection for a preconditioned-flow projection (because the converging direction is `−S∇L`, and a reparameterization `x'=P^{−1}x` turns the preconditioned flow back into a plain one so the old moment calculus transfers), and I had to scale `1−β₂ = Θ(η²)` so the preconditioner moves on the manifold's own clock and enters the SDE as a live OU-like state; the resulting drift is adaptive semi-gradient descent on `⟨∇²L, SΣS − Σ_∥⟩`, with momentum provably irrelevant; under label noise the diffusion dies and the `1/√v` denominator turns SGD's linear `H_{jj}` weighting into `H_{jj}^{−1/2}`, integrating to the regularizer `tr(Diag(H)^{1/2})` (and `tr(Diag(H)^{1−λ})` for exponent `λ`); on a diagonal net that's an `ℓ_{0.5}` (vs SGD's `ℓ₁`) penalty on the recovered vector, predicting sparser, more sample-efficient recovery — while on deep matrix factorization, where the useful bias is toward low rank, the same unique sharpness reduction is the wrong one and should hurt.
