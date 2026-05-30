OK, let me think this through from scratch. The thing I actually care about is the bill for a pre-training run. That bill is, near enough, the number of optimizer steps times the cost of one step. Adam has been the default for years and nobody has reliably beaten it on wall-clock for these decoder-only language models. So the prize is concrete: cut the steps-to-target-loss in half without making each step meaningfully more expensive. If I can do that, I halve the compute, the wall-clock, the dollars. And I should be honest about the right yardstick from the start: it's run-time (or compute, or steps at matched per-step cost) to reach a fixed loss. Comparing loss at a fixed step number is a trap, because a method that does twice the work per step could "win" at fixed steps and lose on the clock.

So why is Adam not already optimal? Let me strip it down. m is an EMA of the gradient, v an EMA of the squared gradient, and the update is η·m̂/(√v̂+ε). If I throw away the moving averages to see the skeleton, m→g and v→g², so the update collapses to η·g/|g| = η·sign(g), coordinatewise. That's SignGD — the same object RProp and RMSProp were circling around. So Adam is, in spirit, taking a fixed-magnitude step of size η in every single coordinate, modulated by some smoothing.

Now, is that the right thing to do? It depends entirely on the landscape, so let me look at the landscape. The Hessian spectra people have measured on deep nets, and on Transformers specifically, are spread over many orders of magnitude. If I just take a language model and look at the positive diagonal entries of its Hessian, the histogram is wide — some coordinates are extremely sharp, some extremely flat. That's not noise; that's the structure of the problem. So let me set up the smallest example that has this property and watch the optimizers fail on it.

Take a separable two-coordinate loss, L(θ₁,θ₂) = L₁(θ₁) + L₂(θ₂), with L₁ sharp (curvature h₁ at its min) and L₂ flat (curvature h₂ at its min), and h₁ ≫ h₂. Plain GD does θᵢ ← θᵢ − η·L′ᵢ. For a quadratic with curvature h, the iteration θ ← θ − η h θ converges only if η < 2/h, and it's fastest near η ≈ 1/h. So the *ideal* learning rate is ≈1/h₁ for the sharp coordinate and ≈1/h₂ for the flat one. But GD has one shared η. If I pick η bigger than ≈1/h₁, the sharp coordinate diverges or oscillates. So I'm capped at η ≈ 1/h₁, which is tiny for the flat coordinate, and θ₂ crawls. The convergence in the flat direction is throttled by the ratio h₁/h₂ — the condition number. That's the disease.

Does SignGD (i.e. Adam) cure it? No, and it's worth seeing exactly why, because the failure is different. SignGD moves every coordinate by exactly η. Same step size everywhere. But "same step size" is not "same progress." For a quadratic of curvature h, a step of size η decreases the loss by about h·η·|θ−θ*| minus an η²h/2 overshoot term — the key point is the loss decrease per unit step is governed by h. In the sharp coordinate, a step of η races to the valley in a couple of steps and then bounces back and forth across it forever (it can't stop without shrinking η). In the flat coordinate, the same η barely dents the loss, because the surface is nearly flat — you'd need to travel a long way to make progress, and you're only moving η per step. To actually settle the sharp coordinate I'd have to decay η to zero, and that makes the flat coordinate even more hopeless. So Adam, like SignGD, makes uniform-sized moves and gets non-uniform *progress*: it equalizes the wrong thing. What I want equalized is the loss decrease across coordinates, and a uniform-step method can't deliver that on heterogeneous curvature.

So state the wish precisely: I want the sharp coordinate to take a *relatively smaller* step and the flat coordinate a *relatively larger* one, calibrated so that each coordinate makes comparable progress in loss. The quantity that tells me "how much loss do I get per unit of step here" is exactly the curvature h. If I scale each coordinate's step by 1/h, then a coordinate's update becomes g/h and the loss it sheds per step is about g²/(2h)·... — let me just do it cleanly. Local quadratic in one coordinate, q(θ) = ½h(θ−θ*)² , gradient g = h(θ−θ*). The step that lands exactly on θ* is Δ = −g/h, and it removes the entire local loss ½g²/h. So if every coordinate uses g/h, every coordinate empties its local quadratic in one step — progress is equalized automatically, and the rate stops caring about h₁/h₂. That's Newton's method, θ ← θ − H⁻¹∇L, restricted to the diagonal here because the problem is separable. So the curvature is not just helpful, it's the precise object that fixes the heterogeneity.

Except Newton has known failure modes, and I should walk into each one rather than pretend it's fine.

Size. H is d×d. For a model with billions of parameters there is no forming it, no storing it, no inverting it. Even the structured approximations — Kronecker-factored blocks, full-matrix preconditioners over reshaped gradients — are heavy, and worse, they tend to be recomputed often. Remember my constraint: any curvature cost must be amortizable to a few percent per step. A preconditioner I refresh every step that costs more than a gradient has already lost the wall-clock game before it starts. So whatever curvature I use has to be (a) small — diagonal, one vector the size of the parameters — and (b) cheap to refresh, and ideally refreshed *rarely*.

Indefiniteness. Away from a minimum, the loss is non-convex, so H has negative eigenvalues. Watch what Newton does in a negative-curvature coordinate: the local model is a downward parabola, q(θ)=½h(θ−c)² with h<0, gradient g=h(θ−c), and the Newton step −g/h = −(θ−c) walks me to c — which is the *maximum*. The step points uphill. On my toy, if I make L₁ genuinely non-convex (a sharp well with a hump beside it), vanilla Newton slides straight to the saddle/maximum and parks there. That's catastrophic: the very correction that fixed heterogeneity will, in the wrong region, climb the loss.

And the Hessian isn't stationary. It changes fast along the trajectory. So even where it's positive, a step extrapolated from a stale or purely-local quadratic can be far too big — the quadratic model was only good in a small neighborhood and I've stepped out of it.

The classical fixes for the second and third mines exist — trust regions, backtracking line search, cubic regularization — they all clamp how far you trust the quadratic. But they add machinery and per-step cost, and the whole point is to stay cheap. Let me see if there's something blunter.

Here's the structure of what I have. Most coordinates, most of the time, have benign positive curvature and the Newton step g/h is exactly what I want. A minority of coordinates — negative, near-zero, or rapidly-changing curvature — produce a Newton step that is garbage and possibly enormous. So the failure mode is not "Newton is wrong everywhere," it's "Newton is occasionally and unboundedly wrong." If the damage from a bad coordinate were *bounded*, I could tolerate a lot of inaccuracy and even a lot of staleness. What bounds the damage of a single coordinate's update? Capping its magnitude. Clip the per-coordinate update at some threshold ρ: clip(g/h, ρ) = max(min(g/h, ρ), −ρ). Now think about what that does in each regime. Where curvature is healthy and the Newton step is modest (smaller than ρ), the clip is inactive and I get the full curvature-aware step. Where curvature is tiny or negative, g/h is huge (or huge with the wrong sign), and the clip pins it to ±ρ — a bounded, sign-following step. A bounded step in the gradient's own sign direction is just a SignGD step. So clipping makes the optimizer *default to SignGD exactly in the coordinates where the second-order information is untrustworthy*, and use real Newton where it's trustworthy. That's the safeguard I wanted, and it costs one elementwise min/max.

Let me make sure the negative-curvature case really degrades gracefully and doesn't, say, keep the bad sign in some perverse way. If h<0, the Newton step g/h has the *opposite* sign of g — it would move uphill. But I'm going to enforce positivity: I'll only ever divide by max(γh, ε) with a small positive floor ε, and I'll prefer estimators that can't produce negative curvature in the first place. So a negative or near-zero h gets floored to ε, the ratio g/ε is enormous with the sign of g, and the clip pins it to ρ·sign(g) — a downhill SignGD step. Exactly right: in the region where Newton would have climbed to the saddle, the clip makes me descend instead, and only once I'm in a convex valley does the unclipped Newton behavior take over and snap me to the minimum. On the toy, that's a trajectory that starts off SignGD-like in the non-convex part and then converges in a handful of steps in the valley — beating both Adam (which bounces) and Newton (which finds the saddle).

Now I need the cheap, ideally-positive, diagonal curvature estimate. Two routes.

Route one: estimate diag(H) directly, assuming nothing about the loss. The classic trick is Hutchinson's. Draw a random vector u with E[uuᵀ] = I — a spherical Gaussian u ~ 𝒩(0, I_d) works. Consider û = u ⊙ (H u). Take the i-th coordinate of its expectation: E[uᵢ (Hu)ᵢ] = E[uᵢ Σⱼ Hᵢⱼ uⱼ] = Σⱼ Hᵢⱼ E[uᵢ uⱼ] = Σⱼ Hᵢⱼ δᵢⱼ = Hᵢᵢ. So E[û] = diag(H), unbiased, exactly. And I never form H: the only thing I need is the Hessian-vector product Hu, and Hu = ∇_θ ⟨∇_θ L, u⟩ — differentiate the scalar ⟨∇L, u⟩ once more. That's a double backward, cost a small constant times a gradient. So the whole estimator is: pick u, compute the gradient, dot it with u, backprop that scalar, multiply elementwise by u. One subtlety: this û is *not* guaranteed non-negative coordinatewise — Hu can have any sign — so individual estimates can be negative. That's fine, because the max(γh, ε) floor and the clip already handle negatives by defaulting to SignGD there. So Hutchinson gives me an unbiased, structure-agnostic diagonal at the price of a Hessian-vector product.

Route two: exploit the structure of the loss to get something that's *always positive* and needs only an ordinary gradient — no Hessian-vector product at all. The loss in language modeling is ℓ(θ) = ce(f(θ,x), y), cross-entropy of the logits f ∈ ℝ^V against the label y, V the vocabulary size. Differentiate twice through the composition. By the chain rule the Hessian splits:

  ∇²_θ ℓ = J_θf · S · J_θfᵀ + J_θθf[q],

where J_θf ∈ ℝ^{d×V} is the Jacobian of the logits w.r.t. parameters, S = ∂²ce/∂t²|_{t=f} ∈ ℝ^{V×V} is the Hessian of the loss in *logit* space, q = ∂ce/∂t is the gradient in logit space, and J_θθf[q] is the second derivative of the (vector-valued) logit map contracted against q. The first term is the Gauss-Newton matrix; the second is the part that comes from the curvature of the network map itself. For neural nets the second term is empirically small relative to the first, so I'll drop it and estimate just the Gauss-Newton term. And here's the gift: S is the Hessian of a convex loss (cross-entropy is convex in the logits), so S ⪰ 0, hence J S Jᵀ ⪰ 0. The Gauss-Newton matrix is positive semidefinite *by construction*. A PSD preconditioner means the preconditioned step is always a descent direction. So route two automatically gives me the "positive curvature only" property I was bolting on by hand.

But J S Jᵀ is still a d×d object I can't form. I need its diagonal, cheaply. Let me massage S. For softmax + cross-entropy, write p = softmax(f). A direct computation gives S = diag(p) − p pᵀ. Notice what's *not* there: y. S depends only on the logits f, not on the label. (That's actually a general property of exponential families — the Hessian of the negative log-likelihood depends only on the natural parameters, not on the realized example.) So I'm free to write S as an expectation over *any* label distribution I like; choose the model's own: S = E_{ŷ ~ Cat(f)}[ ∂²ce(f, ŷ)/∂t² ], which is trivially true since the integrand doesn't depend on ŷ.

Now apply the second Bartlett identity. For the negative log-likelihood of a probabilistic model — and ce against a label sampled from the model *is* exactly the NLL of the categorical Cat(f) — the expected Hessian equals the expected outer product of the score (this is just the Fisher = expected Hessian-of-NLL fact):

  E_{ŷ ~ Cat(f)}[ ∂²ce/∂t² ] = E_{ŷ ~ Cat(f)}[ (∂ce/∂t)(∂ce/∂t)ᵀ ].

So S = E_{ŷ}[ (∂ce/∂t)(∂ce/∂t)ᵀ ]. Sandwich it with the Jacobian and use the chain rule J_θf · (∂ce/∂t) = ∇_θ ce(f, ŷ):

  J_θf · S · J_θfᵀ = E_{ŷ ~ Cat(f)}[ J_θf (∂ce/∂t)(∂ce/∂t)ᵀ J_θfᵀ ] = E_{ŷ ~ Cat(f)}[ ∇_θ ce(f, ŷ) · ∇_θ ce(f, ŷ)ᵀ ].

Take the diagonal:

  diag( J S Jᵀ ) = E_{ŷ ~ Cat(f)}[ ∇_θ ce(f, ŷ) ⊙ ∇_θ ce(f, ŷ) ].

So the per-coordinate Gauss-Newton curvature is the expected elementwise square of the gradient of the loss computed against a label *sampled from the model* — not the real label. That's striking: just resample the label from the model's own softmax and square the resulting gradient. It's manifestly non-negative, and it costs one ordinary backward.

There's one practical snag. The clean estimator is per-example, (1/B) Σ_b ∇ce_b ⊙ ∇ce_b for a minibatch of B, but autodiff hands me only the *averaged* gradient over the batch, not the individual per-example gradients, and the average squared is not the squared-then-averaged. Here the *first* Bartlett identity rescues me: under labels sampled from the model, the score has zero mean, E_{ŷ_b}[∇ce(f(x_b), ŷ_b)] = 0. Let L̂(θ) = (1/B) Σ_b ce(f(θ,x_b), ŷ_b) be the minibatch loss on sampled labels. Then

  E[ B · ∇L̂ ⊙ ∇L̂ ] = E[ (1/B) (Σ_b ∇ce_b) ⊙ (Σ_b ∇ce_b) ].

Expand the outer product into diagonal (b=b') and cross (b≠b') terms. The sampled labels are independent across examples and each score is zero-mean, so every cross term factorizes into E[∇ce_b] ⊙ E[∇ce_{b'}] = 0. Only the diagonal survives:

  = E[ (1/B) Σ_b ∇ce_b ⊙ ∇ce_b ],

which is exactly the per-example estimator I wanted, hence equals diag of the minibatch Gauss-Newton. So the implementable estimator is B · ∇L̂ ⊙ ∇L̂ — sample labels from the model, take one ordinary minibatch gradient, square it elementwise, scale by the batch size B. Because everything went through the two Bartlett identities, I'll think of it as the Gauss-Newton-Bartlett estimate. (For a regression head with squared loss the same machinery gives S = I and the estimate J Jᵀ; consistent with assuming y ~ 𝒩(f, σ²).)

So now I have two estimators and they trade off cleanly. Hutchinson assumes nothing about the loss but needs a Hessian-vector product and can return negative entries. Gauss-Newton-Bartlett assumes the loss is a negative log-likelihood (true for the LM cross-entropy) but needs only a gradient on resampled labels and is always non-negative — so it never even tempts the negative-curvature failure, and it guarantees a descent direction. Both cost on the order of one gradient.

Now the rest of the algorithm almost writes itself, but each knob deserves its reason.

The estimate from a single minibatch is as noisy as a minibatch gradient. Adam denoises the gradient's second moment with an EMA; I'll do the same to the curvature: h_t = β₂ h_{t−k} + (1−β₂) ĥ_t. The numerator gets the usual gradient EMA, m_t = β₁ m_{t−1} + (1−β₁) g_t.

How often do I refresh the curvature? This is where the clip pays off. Because a stale or noisy h can only ever cost me a bounded, SignGD-sized step (the clip guarantees it), I don't need to refresh every step — which is precisely the overhead that sank earlier diagonal-Hessian methods. So refresh only every k steps; in practice k around 10 makes the curvature cost ~5% amortized, turning the step-count win into a wall-clock win. With k=1 I pay roughly double per step for no real benefit; with k far larger the estimate goes too stale. The clip is what *licenses* the infrequency.

Put the update together. With a small positive floor ε to avoid dividing by zero,

  θ_{t+1} ← θ_t − η_t · clip( m_t / max(γ h_t, ε), 1 ),

and a decoupled weight-decay step θ ← (1−η_t λ) θ beforehand, AdamW-style.

Writing the clip threshold as 1 with the γ inside the denominator is just a reparameterization, but a convenient one. Algebraically,

  η · clip( m / max(γ h, ε), 1 ) = (η/γ) · clip( m / max(h, ε/γ), γ ).

Read the right-hand side: I'm clipping the *raw* Newton-ish ratio m/h at γ, then rescaling by η/γ. So γ is really the clip threshold on m/h, and η/γ is the overall step scale. Why factor it this way instead of clipping m/h directly at some ρ? Because then the typical update magnitude would swing wildly with the threshold. With this form, every *clipped* coordinate ends up contributing exactly η to the update (the (η/γ)·γ from the saturated clip), independent of γ. So γ no longer controls the size of the update; it controls the *fraction* of coordinates that get clipped. That decouples the two things I tune: η sets the scale, γ sets how aggressive the second-order behavior is. In the extreme γ→0 every coordinate clips and the update is η·sign(m) everywhere — pure momentum SignSGD — which is the safe limit. In practice I'll tune γ to keep a healthy fraction of coordinates clipped (most of them — somewhere in the 50–90% range), i.e. tune to the observed clip rate rather than to an abstract number.

And the negative/tiny-h case, once more, concretely with this update: if h_i < 0 then γh_i is negative, max(γh_i, ε) = ε, the ratio m_i/ε is enormous with sign(m_i), and clip(·, 1) = sign(m_i). The coordinate's update is η·sign(m_i) — momentum SignSGD. So the optimizer uses SignSGD as an automatic backup wherever the curvature is negative or mistakenly small, and only takes genuine curvature-scaled steps where the estimate is trustworthy. The clip also caps the worst-case update at η in every coordinate (η·ρ in general, ρ=1 here), which is the stability property that frequently bites second-order methods. And because most coordinates aren't clipped and self-adjust, I can afford the per-coordinate cap to correspond to a larger overall step than plain SignSGD would survive.

Let me now make sure the "uniform progress, condition-number-free" intuition is real and not wishful, by actually doing the convergence argument on a clean convex model — because if the clip didn't preserve descent, the whole edifice is decoration. Analyze the deterministic clipped-Newton iterate (drop the EMAs and the diagonal-only restriction; do the clipping in the Hessian's eigenbasis so it's coordinate-wise there). Write the eigendecomposition ∇²L(θ) = Vᵀ Σ V, Σ = diag(σ₁,…,σ_d), v_i the i-th row of V. The update is

  θ_+ = θ − η Vᵀ clip( V (∇²L)⁻¹ ∇L, ρ ).

Assume L is strictly convex with minimizer θ*, μ = λ_min(∇²L(θ*)), and a mild multiplicative-Lipschitz condition on the Hessian: within radius R, ‖∇²L(θ′)⁻¹ ∇²L(θ)‖ ≤ 2, i.e. the Hessian only changes by a constant factor over a ball of radius R. (I deliberately don't assume a global smoothness bound — that's the whole point, I want the rate free of the largest eigenvalue.)

Descent lemma. Take ηρ ≤ R/√d so the step stays inside the ball. Let f(t) = L(tθ_+ + (1−t)θ) and u = clip(Σ⁻¹ V∇L, ρ). The multiplicative-Lipschitz condition gives f″(t) ≤ 2f″(0) on [0,1], so a Taylor expansion bounds f(1) ≤ f(0) + f′(0) + f″(0). Compute the two terms. First derivative:

  f′(0) = ⟨∇L, −η Vᵀ u⟩ = −η ⟨V∇L, u⟩ = −η ⟨V∇L, clip(Σ⁻¹ V∇L, ρ)⟩
        = −η Σ_i (v_iᵀ∇L) · clip( σ_i⁻¹ v_iᵀ∇L, ρ ).

Per coordinate, (v_iᵀ∇L)·clip(σ_i⁻¹ v_iᵀ∇L, ρ): if unclipped it's σ_i⁻¹ (v_iᵀ∇L)²; if clipped it's ρ|v_iᵀ∇L| (signs match because σ_i>0). Either way it equals min{ σ_i⁻¹|v_iᵀ∇L|², ρ|v_iᵀ∇L| }. So f′(0) = −η Σ_i min{ ρ|v_iᵀ∇L|, σ_i⁻¹|v_iᵀ∇L|² }. Second derivative:

  f″(0) = η² ⟨Vᵀu, ∇²L Vᵀu⟩ = η² ⟨u, Σu⟩ = η² Σ_i u_i² σ_i.

Since |u_i| = min{ |v_iᵀ∇L|/σ_i, ρ }, we get u_i² σ_i ≤ min{ |v_iᵀ∇L|/σ_i, ρ }·(|v_iᵀ∇L|/σ_i)·σ_i = min{ |v_iᵀ∇L|²/σ_i, ρ|v_iᵀ∇L| }. So f″(0) ≤ η² Σ_i min{ ρ|v_iᵀ∇L|, σ_i⁻¹|v_iᵀ∇L|² }. Combining,

  L(θ_+) − L(θ) ≤ f′(0) + f″(0) ≤ −(η − η²) Σ_i min{ ρ|v_iᵀ∇L|, σ_i⁻¹|v_iᵀ∇L|² }.

Stare at the per-coordinate decrease, min{ ρ|v_iᵀ∇L|, σ_i⁻¹|v_iᵀ∇L|² }. The two arguments are exactly the two regimes I designed for. The unclipped term σ_i⁻¹|v_iᵀ∇L|² is the full Newton quadratic decrease — and crucially it has σ_i in the denominator, so a huge eigenvalue doesn't shrink the *guaranteed decrease*; the rate doesn't pay for sharpness. The clipped term ρ|v_iᵀ∇L| is the SignGD-style decrease, a safe bounded amount that doesn't depend on σ_i at all — that's the bound that holds even when the curvature is unreliable. The min just says: in each coordinate you're guaranteed at least the smaller of "real Newton progress" and "safe clipped progress." That is the descent guarantee that lets me be sloppy and infrequent about h, made precise.

The rest is two phases. In the far phase, the sum of decreases is bounded below, so the loss drops by a fixed amount each step; in O((L(θ₀) − minL)/(ημρ²)) steps it reaches a small region (loss − minL ≤ μρ²/8). Once there, one can show the Newton step never triggers the clip (near the minimum ‖(∇²L)⁻¹∇L‖ ≤ ρ), so the iterate becomes plain Newton, and the descent lemma's unclipped branch gives L(θ_t) − minL ≤ (1 − η(1−η))^{t−T}(L(θ_T) − minL) — exponential decay, so log(1/ε) steps to error ε. Setting η = 1/2 and ρ = R/(2√d), the total is

  T ≲ d · (L(θ₀) − minL)/(μR²) + ln( μR²/(32dε) ).

No condition number, no largest-eigenvalue/smoothness term anywhere. Compare that with the SignGD proxy for Adam on a two-coordinate quadratic L = ½μθ₁² + ½βθ₂²: pick two initializations on the axes; because SignGD moves each coordinate by exactly η per step, reaching loss ε in the flat coordinate forces η ≤ √(8ε/β), and then traversing the sharp coordinate's initial distance √(2Δ/μ) takes at least T ≥ ½(√(Δ/ε) − √2)·√(β/μ) steps. That √(β/μ) is the square root of the condition number — Adam's proxy provably pays for heterogeneity, and the clipped-curvature method provably doesn't. That's the whole thesis in one inequality.

Now land it in code, grounded in a nanoGPT-style training loop. The optimizer keeps two extra buffers per parameter, the gradient EMA and the curvature EMA, and refreshes the curvature on a schedule. The Gauss-Newton-Bartlett refresh is the cheap default: every k steps, do a forward pass for logits, sample labels from the model's softmax, compute cross-entropy against those sampled labels, backward, and let the curvature buffer absorb grad⊙grad (the batch-size factor B is folded into the denominator at step time). The update is the gradient EMA divided by the floored, γ-scaled curvature, clipped, applied with decoupled weight decay. Note that sign(m)·clip(|m|/d, 1) is identical to clip(m/d, 1) since the clip preserves sign, so the code can compute a magnitude ratio and reattach the sign.

```python
import torch
from torch.optim.optimizer import Optimizer

class Sophia(Optimizer):
    # betas = (beta1 for gradient EMA m, beta2 for curvature EMA h)
    # rho plays the role of gamma; weight_decay is decoupled (AdamW-style)
    def __init__(self, params, lr=1e-4, betas=(0.96, 0.99), rho=0.04,
                 weight_decay=1e-1):
        defaults = dict(lr=lr, betas=betas, rho=rho, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def update_hessian(self):
        # GNB curvature refresh: called after a backward on SAMPLED labels,
        # so p.grad is the gradient of the resampled-label loss. h <- EMA(grad ⊙ grad).
        # This is the always-nonnegative Gauss-Newton-Bartlett diagonal estimate.
        for group in self.param_groups:
            _, beta2 = group['betas']
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                if 'hessian' not in state:
                    state['hessian'] = torch.zeros_like(p)
                state['hessian'].mul_(beta2).addcmul_(p.grad, p.grad, value=1 - beta2)

    @torch.no_grad()
    def step(self, bs):
        # bs = total batch size B, folded into the denominator (the B in B·ĝ⊙ĝ).
        for group in self.param_groups:
            beta1, _ = group['betas']
            lr, rho, wd = group['lr'], group['rho'], group['weight_decay']
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                if 'exp_avg' not in state:
                    state['exp_avg'] = torch.zeros_like(p)
                    state['hessian'] = torch.zeros_like(p)
                m, h = state['exp_avg'], state['hessian']

                p.mul_(1 - lr * wd)                 # decoupled weight decay
                m.mul_(beta1).add_(p.grad, alpha=1 - beta1)   # gradient EMA (numerator)

                # clip(m / max(gamma*h, eps), 1): magnitude ratio capped at 1,
                # sign reattached. gamma == rho here, and B is inside the denom.
                ratio = (m.abs() / (rho * bs * h + 1e-15)).clamp(max=1.0)
                p.addcmul_(m.sign(), ratio, value=-lr)        # theta -= lr * clip(m/(gamma*h), 1)

# --- training loop (nanoGPT-style); k = hess_interval ---
opt = Sophia(model.parameters(), lr=peak_lr, betas=(0.96, 0.99), rho=0.05)
for it in range(max_iters):
    # ordinary step: real-label loss, then the clipped curvature-scaled update
    logits, loss = model(X, Y)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)   # global-norm clip (stability)
    opt.step(bs=total_bs * block_size)
    opt.zero_grad(set_to_none=True)
    X, Y = get_batch('train')

    # every k steps: refresh GNB curvature on SAMPLED labels (cost ~ one extra gradient)
    if it % k == k - 1:
        logits, _ = model(X, 0)
        y_sample = torch.distributions.Categorical(logits=logits).sample()
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), y_sample.view(-1), ignore_index=-1)
        loss.backward()
        opt.update_hessian()
        opt.zero_grad(set_to_none=True)
        X, Y = get_batch('train')
```

For the structure-agnostic variant, the only change is the refresh: instead of resampling labels and squaring the gradient, draw u ~ 𝒩(0, I), form the Hessian-vector product Hu = ∇⟨∇L, u⟩ by a double backward, and feed u ⊙ (Hu) into the same curvature buffer — unbiased for diag(H), at the price of the extra backward.

The causal chain, start to finish: Adam reduces to uniform-magnitude sign steps, which equalize step size but not loss decrease, so on the order-of-magnitude-heterogeneous curvature of language-model losses the flat directions starve and the rate pays the condition number; the object that equalizes loss decrease per coordinate is the curvature, i.e. Newton, whose rate is condition-number-free; but the full Hessian is too big, indefinite (so Newton climbs to saddles), and non-stationary (so its quadratic model goes stale); restrict to a cheap diagonal estimate — Hutchinson via a Hessian-vector product, or the always-positive Gauss-Newton-Bartlett via squaring the gradient on model-sampled labels — and bound the damage of any bad coordinate with a per-coordinate clip, which makes the optimizer default to safe SignGD steps exactly where curvature is untrustworthy and take real Newton steps where it isn't; the clip's bounded-decrease guarantee (the descent lemma) is what licenses estimating the curvature only every k steps, so the step-count win survives into wall-clock; and the convex analysis confirms the rate is free of both the condition number and the smoothness constant, the two quantities a heterogeneous landscape would otherwise charge for.
