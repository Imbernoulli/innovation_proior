Let me start from the thing that actually hurts. I'm training a deep net with SGD, and the same three complaints keep recurring: I can't turn the learning rate up without the thing diverging, I have to baby the initialization or it never gets going, and the moment I put a sigmoid or tanh anywhere deep the network just sits there learning nothing. The received wisdom treats these as three unrelated annoyances with three unrelated patches — ReLU for the saturation, Xavier init for the scale, a tiny learning rate for the divergence — and I keep using all three at once, but I have a nagging feeling they're one disease wearing three coats. If they are, there should be a single fix, and these three patches should fall out of it as side effects rather than as the main event.

So what is the disease. Write the net as a composition, ℓ = F₂(F₁(u, Θ₁), Θ₂), and stare at F₂. From its point of view, x = F₁(u, Θ₁) is just "the input," and a gradient step on Θ₂ — Θ₂ ← Θ₂ − (α/m) Σᵢ ∂F₂(xᵢ,Θ₂)/∂Θ₂ — is literally the same step a stand-alone network F₂ fed x would take. That equivalence is not a metaphor; it's an algebraic identity. So everything I know about training a learner applies to F₂ as a sub-learner — including the dull but real fact that a learner trains best when the distribution of its inputs doesn't keep moving. (That's why we whiten the data once at the bottom and why train/test distribution mismatch hurts.) But x is the output of the layers below, and every time Θ₁ updates, the distribution of x shifts: its mean wanders, its variance breathes, its very shape changes. So F₂ is forever chasing a moving target, re-spending capacity just to track where its own inputs went instead of fitting the actual task. And it compounds with depth, because a small nudge to Θ₁ gets amplified as it climbs through the layers above. Shimodaira 2000 named the train/test version of this "covariate shift" and people fix it with domain adaptation; here it's happening *inside* the network, between consecutive training steps. Internal covariate shift. That reframing is the whole game: if I could pin the distribution of each layer's inputs so it stays put across training, every sub-network would face a stationary problem, and I'd bet the three coats come off together.

Why is the drift specifically lethal with saturating nonlinearities? Take z = g(Wu+b) with g the logistic sigmoid. As |Wu+b| grows, g'(·) → 0 — the tails are flat. So for every coordinate of the pre-activation except those near zero, the gradient flowing down to u is multiplied by something tiny and effectively vanishes. Now add drift: as the lower parameters move, more and more coordinates of Wu+b get shoved out into those flat tails, and the layer goes dead. ReLU dodges this only because its positive branch never saturates — it's treating the symptom. But if I could hold the pre-activation distribution stable near the origin, the optimizer would simply never wander into the saturated regime, and sigmoids would be trainable again from the cause side, not the symptom side. That's the fix I actually want.

The classical result to lean on: LeCun, Bottou, Orr & Müller, "Efficient BackProp" (1998), and Wiesler & Ney (2011) — a net converges faster when its inputs are *whitened*: linearly transformed to zero mean, unit variance, decorrelated. Standard practice whitens the data once, at the bottom. But if every internal layer is a sub-learner with the same appetite, the obvious extrapolation is: whiten the inputs of *every* layer, and keep doing it as training proceeds. Fixed distributions all the way up.

First instinct, the cheap one: treat normalization as housekeeping. Every so often, sweep the training set, compute means and variances (or the full whitening), and subtract/rescale the activations as a separate step bolted on outside the gradient update. Before I commit, let me actually run the smallest version of this in my head, because something feels off about doing it "outside the gradient."

Concretely: a unit adds a learned bias, x = u + b, and I center it by subtracting the dataset mean, x̂ = x − E[x], with E[x] = (1/N) Σᵢ xᵢ over the training set X. Now I take a gradient step on b. Here's the trap, and I want to be precise about it: my gradient computation treats E[x] as a *constant* — it does not know that E[x] secretly depends on b, because E[u+b] = E[u] + b. So the step it computes is b ← b + Δb with Δb ∝ −∂ℓ/∂x̂. Fine. Now I recompute the normalization with the new bias and look at what the layer actually outputs:

    (u + b + Δb) − E[u + b + Δb] = (u + b + Δb) − (E[u] + b + Δb) = u + b − E[u] = u + b − E[u+b].

It is *identical* to before the step. The Δb I added to the raw activation is exactly the Δb that reappears inside E[·] when I re-center, and the two cancel. So the output didn't move; the loss didn't move. But b *did* move — and it'll move again next step, by the same Δb, because the loss gradient is unchanged and keeps asking for the same thing. So b marches off toward infinity while the loss sits perfectly flat. The model blows up. Wall. And it gets strictly worse if the normalization also rescales by a variance, because now there's a multiplicative parameter doing the same silent drift. This is exactly the pathology I'd expect from the whole family that normalizes outside the optimizer — mean-normalized SGD (Wiesler 2014), the reparameterization tricks of Raiko et al. 2012, natural-gradient schemes — they normalize as a side computation or by editing the optimizer, and the moment the gradient ignores how the normalization depends on the parameters, this cancellation bites.

So the lesson is sharp, and it is not "normalization is bad." It is: the normalization must be *part of the model*, so that the gradient of the loss accounts for it. If backprop knows the statistics depend on the parameters, the cancellation stops being a silent drift and becomes an honest gradient signal — the gradient w.r.t. b will correctly come out as "moving b does nothing," i.e. essentially zero, instead of "keep pushing b." Let me state the requirement honestly. The normalization is a transform x̂ = Norm(x, X) that depends not only on this example x but on the whole set X — and every element of X itself depends on Θ if x came from a layer below. For backprop I need *two* Jacobians:

    ∂Norm(x, X)/∂x   and   ∂Norm(x, X)/∂X.

Dropping the second is exactly what detonated b above. Keep both and I'm safe by construction.

Can I afford to keep both with *full* whitening? Let me cost it out rather than wave it away. Full whitening of a d-dimensional activation means: form the covariance Cov[x] = E_{x∈X}[xxᵀ] − E[x]E[x]ᵀ, which is a d×d matrix; compute its inverse square root Cov[x]^{−1/2}; produce the whitened activation Cov[x]^{−1/2}(x − E[x]); and then, for backprop, differentiate *through* that inverse-square-root. The matrix inverse-square-root in practice goes through an eigendecomposition (or SVD), which is O(d³) per step, and — worse — differentiating through an eigendecomposition is delicate: the derivatives blow up when eigenvalues collide, and the map isn't everywhere differentiable. And I have to redo this over the whole training set after every parameter update. That's already unusable, but there's a sharper, structural killer: I want this to live inside mini-batch SGD, and a mini-batch of size m is typically *smaller* than d, the number of activations I'm whitening. A sample covariance estimated from m < d points is singular — rank at most m−1 — so Cov^{−1/2} doesn't even exist without adding a regularizer, and the whitening direction becomes ill-defined and noisy. Wall, again. Full joint whitening is correct and unusable: too expensive, not everywhere differentiable, and singular exactly in the regime I need it.

Time to retreat to something cheaper. Two simplifications, and I owe a reason for each thing I give up.

Simplification one: don't whiten jointly — normalize each scalar feature independently. For a d-dimensional input x = (x⁽¹⁾…x⁽ᵈ⁾), make each coordinate zero-mean, unit-variance on its own:

    x̂⁽ᵏ⁾ = (x⁽ᵏ⁾ − E[x⁽ᵏ⁾]) / √(Var[x⁽ᵏ⁾]).

Is dropping decorrelation a sin? This is precisely where LeCun et al. 1998 pays the debt: they show per-coordinate mean/variance normalization speeds convergence *even when the features are not decorrelated*. The decorrelation is the most expensive part of whitening and, conveniently, the most dispensable. So I keep almost all the benefit and shed the cubic cost. And look at what the per-dimension move buys me on the structural problem: I never form a covariance matrix, so there is no d×d inverse, no eigendecomposition, no singularity when m < d. Each coordinate's mean and variance are scalars, estimable from even a small batch, and differentiable in closed form. The simplification doesn't merely make the math cheaper; it dissolves the exact obstruction (singular covariance, non-differentiable matrix root) that made the joint version impossible inside mini-batch SGD. That's the sign I'm pushing on the right variable.

But before I celebrate, a representational cost I have to check, because it's the obvious objection. If I force the input to a sigmoid to be zero-mean unit-variance, I've pinned it to the region around the origin — which for a sigmoid is exactly the *roughly-linear* part of the curve. I've stolen the layer's ability to use the nonlinear regime, or to sit at any other operating point. More generally, forcing every pre-activation to mean 0, variance 1 forbids the network from ever choosing a feature that is deliberately off-center or high-variance. Normalization just changed what the layer can represent. Unacceptable as stated — a fix that quietly shrinks the hypothesis class is not a free lunch, it's a tax.

So patch it: after normalizing, hand each activation back the freedom to be anything, with a learned per-feature scale and shift,

    y⁽ᵏ⁾ = γ⁽ᵏ⁾ x̂⁽ᵏ⁾ + β⁽ᵏ⁾,

γ and β learned alongside the rest of Θ. Does this genuinely restore full power, or is it cosmetic? Check the worst case, the one where normalization could only hurt: suppose the optimum really did want the original, un-normalized activation back. Can the layer get there? If I ignore ε, set γ⁽ᵏ⁾ = √(Var[x⁽ᵏ⁾]) and β⁽ᵏ⁾ = E[x⁽ᵏ⁾]; if I keep ε, set γ⁽ᵏ⁾ = √(Var[x⁽ᵏ⁾]+ε). The ε-free check is

    y⁽ᵏ⁾ = √(Var[x⁽ᵏ⁾]) · (x⁽ᵏ⁾ − E[x⁽ᵏ⁾])/√(Var[x⁽ᵏ⁾]) + E[x⁽ᵏ⁾] = x⁽ᵏ⁾,

exactly, and with the ε-adjusted γ the same cancellation uses √(Var[x⁽ᵏ⁾]+ε) in both numerator and denominator. The transform can represent the *identity*. So the affine pair isn't a hack to undo a side effect; it's what guarantees I've lost nothing. The network can recover any mean and any variance it wants per feature, including reverting to the raw activation, including parking the sigmoid input wherever it likes on the curve. Normalization becomes something the network can *undo* if undoing is optimal, while the default starting point is a clean, stable distribution. That's the right shape for a fix: it constrains nothing in the limit, it only changes the default — and a better default is the whole point. Two extra scalars per feature is a trivial price; the conclusion is that the transform preserves the network's capacity outright.

Simplification two: I can't sweep the whole training set every step to get E[x⁽ᵏ⁾] and Var[x⁽ᵏ⁾]. But I'm *already* drawing a fresh mini-batch every step. So let the mini-batch itself *define* the statistics — the mean and variance over the current m examples become the normalization mean and variance. This is the move that makes the "must be in the gradient" requirement free, and I want to see exactly why. Once the statistics are the batch mean and batch variance, they are a deterministic, differentiable function of the very batch I'm already differentiating through. The dreaded ∂Norm/∂X term is no longer some mysterious whole-set dependence I have to track separately; it's just ordinary calculus on a sum over m terms. The property that was a liability in the b-disaster (the statistics depend on the data) becomes the mechanism that makes the whole thing differentiable for free. There's no whole-set sweep, no special bookkeeping — backprop owns the statistics automatically.

And it dovetails with simplification one in a way I should call out: the per-dimension choice is what *enables* the mini-batch choice. If I'd kept joint whitening, the batch with m < d would give a singular covariance and I'd be forced into regularizing it. Because I only need per-coordinate scalar variances, a batch of m examples gives me m perfectly good samples per coordinate, and the estimates are well-defined. The two simplifications aren't independent; the first removes the obstruction that would have wrecked the second.

So fix a single activation x (drop the (k); every coordinate gets the identical treatment), and a mini-batch B = {x₁…x_m}. The transform is:

    μ_B  = (1/m) Σᵢ xᵢ                         (mini-batch mean)
    σ²_B = (1/m) Σᵢ (xᵢ − μ_B)²                (mini-batch variance)
    x̂ᵢ  = (xᵢ − μ_B) / √(σ²_B + ε)             (normalize)
    yᵢ  = γ x̂ᵢ + β                            (scale and shift)

with ε a small constant inside the square root for numerical stability — without it, a feature that happens to be near-constant across the batch gives σ²_B ≈ 0 and the normalization divides by zero. Note the transform is intrinsically *not* a per-example operation: yᵢ depends on xᵢ *and* on the other examples in the batch, through μ_B and σ²_B. That coupling is going to matter twice — once as a problem (inference) and once as a gift (regularization). As a sanity check that the normalization does what I claim: by construction Σᵢ x̂ᵢ = 0 and (1/m) Σᵢ x̂ᵢ² = 1 (neglecting ε), so across the batch each x̂ has mean 0 and variance 1 — exactly the stable, near-origin distribution I wanted to feed each sub-network.

The whole argument rests on the gradient flowing *through* the statistics, so I have to actually backprop ℓ through this transform and get gradients for γ and β. The forward graph branches: xᵢ feeds into μ_B; xᵢ and μ_B feed into σ²_B; then xᵢ, μ_B, σ²_B all feed into x̂ᵢ; and x̂ᵢ feeds yᵢ. So xᵢ reaches the loss by *three* routes — directly through x̂ᵢ, indirectly through μ_B, and indirectly through σ²_B. The chain rule says I must sum all three; forgetting any one is the same class of error as the b-blow-up. Let me grind every Jacobian term.

Top of the transform, ∂ℓ/∂yᵢ is handed down from above. Since yᵢ = γ x̂ᵢ + β,

    ∂ℓ/∂x̂ᵢ = ∂ℓ/∂yᵢ · γ.

Now σ²_B. It appears inside every x̂ⱼ via x̂ⱼ = (xⱼ − μ_B)(σ²_B + ε)^{−1/2}, so ∂x̂ⱼ/∂σ²_B = (xⱼ − μ_B) · (−½)(σ²_B + ε)^{−3/2}. Summing the chain over all j that σ²_B touches:

    ∂ℓ/∂σ²_B = Σⱼ ∂ℓ/∂x̂ⱼ · (xⱼ − μ_B) · (−½)(σ²_B + ε)^{−3/2}.

Now μ_B. It influences the loss two ways: directly inside each x̂ⱼ (where ∂x̂ⱼ/∂μ_B = −(σ²_B + ε)^{−1/2}), and *through* σ²_B, because σ²_B = (1/m) Σⱼ (xⱼ − μ_B)² depends on μ_B too, with ∂σ²_B/∂μ_B = (1/m) Σⱼ −2(xⱼ − μ_B). So both paths:

    ∂ℓ/∂μ_B = (Σⱼ ∂ℓ/∂x̂ⱼ · −(σ²_B + ε)^{−1/2}) + ∂ℓ/∂σ²_B · (1/m) Σⱼ −2(xⱼ − μ_B).

(The sum Σⱼ(xⱼ − μ_B) is identically 0, so that second term vanishes — but I'll keep it written, because it's the honest chain-rule term and dropping it by hand is the kind of "the gradient ignores a dependence" shortcut that started this whole mess.) Finally xᵢ itself, gathering its three paths — direct through x̂ᵢ (∂x̂ᵢ/∂xᵢ = (σ²_B + ε)^{−1/2}), through σ²_B (∂σ²_B/∂xᵢ = 2(xᵢ − μ_B)/m), and through μ_B (∂μ_B/∂xᵢ = 1/m):

    ∂ℓ/∂xᵢ = ∂ℓ/∂x̂ᵢ · (σ²_B + ε)^{−1/2} + ∂ℓ/∂σ²_B · 2(xᵢ − μ_B)/m + ∂ℓ/∂μ_B · (1/m).

And the parameters, summing over the batch because γ and β are shared across all m examples:

    ∂ℓ/∂γ = Σᵢ ∂ℓ/∂yᵢ · x̂ᵢ,
    ∂ℓ/∂β = Σᵢ ∂ℓ/∂yᵢ.

There it is — a fully differentiable transform, every path accounted for. The normalizer is welded into the model; the optimizer sees it. And this is the resolution of the b-disaster, not a coincidence: with normalization inside the model, the gradient w.r.t. a bias added just before it correctly reflects that shifting the bias does nothing, so the optimizer no longer pushes it. The b-blow-up was pointing at this conclusion the whole time — the lesson was never "don't normalize," it was "make the statistics differentiable functions of the batch so backprop owns them."

Now the coupling I flagged comes back as a problem: train versus inference. During training, normalizing by the *mini-batch's* statistics is what makes everything differentiable, and that's non-negotiable. But at inference the coupling is a liability — the prediction for one image must not depend on which other images happen to share its batch; I want a deterministic function of the single input, and I want the same answer whether I classify that image alone or in a crowd. So at test time I swap the batch statistics for fixed *population* statistics:

    x̂ = (x − E[x]) / √(Var[x] + ε),

with E[x] and Var[x] the means/variances over the training distribution, frozen. Neglecting ε these still deliver mean 0, variance 1, so the inference distribution matches what training produced.

Which estimate of Var[x] do I freeze? Here the Bessel subtlety bites, and I should get it right rather than reuse the training formula by reflex. During training I used the *biased* sample variance σ²_B = (1/m) Σ(xᵢ − μ_B)², which divides by m about the *sample* mean μ_B. That's the right thing to backprop through (it's the exact variance of the batch I normalized), but it systematically *underestimates* the population variance, because estimating the mean from the same m points consumes one degree of freedom — the data hug their own sample mean more tightly than they hug the true mean. The clean population estimate, if I explicitly average batch variances of fixed size m, applies the correction factor m/(m−1):

    Var[x] = (m/(m−1)) · E_B[σ²_B].

Using the biased σ²_B directly at inference would shrink the variance, hence slightly inflate every normalized x̂, and create a small train/inference mismatch; the m/(m−1) factor corrects that. But the simple running-average implementation I want to mirror takes the common pragmatic route: it updates `running_var` with σ²_B itself, the biased batch variance, because that lets me monitor the model as it trains without a separate estimation pass. If I wanted the exact Bessel-corrected running estimate for a fixed batch size, I would multiply σ²_B by m/(m−1) before the running update. Either way, once the stored mean and variance estimate are frozen, normalize-then-scale-shift is a single affine map in x, so I can fold the two affines into one,

    y = (γ / √(Var[x] + ε)) · x + (β − γ E[x] / √(Var[x] + ε)),

and inference costs one multiply-add per activation — no batch statistics, no per-batch dependence, nothing stochastic.

Now convolutions, because the host net is a conv net and the rule has to respect convolutional structure. Two sub-decisions: *where* to insert the transform, and *how* to pool the statistics.

Where. A layer is z = g(Wu + b). I could normalize the layer input u, or the pre-activation Wu + b. Normalizing u is tempting (it's the layer's literal input) but wrong here: u is typically the *output* of a previous nonlinearity, whose distribution shape is itself drifting and is often skewed/sparse, and pinning only its first two moments leaves the shape free to keep shifting, so it wouldn't actually kill the covariate shift. The pre-activation Wu + b, by contrast, is an affine mixture of many inputs, and a sum of many things tends toward a symmetric, non-sparse, "more Gaussian" distribution — Hyvärinen & Oja 2000 (ICA) make this concrete. For a more-Gaussian quantity, matching the mean and variance really does buy a stable distribution, because a Gaussian is pinned by its first two moments. So normalize x = Wu + b, right before the nonlinearity: z = g(BN(Wu + b)).

And a cleanup falls out of that placement: the bias b is now redundant. I'm about to subtract the batch mean of Wu + b, and any constant b just shifts that mean by the same b and gets canceled by the subtraction — exactly the cancellation from the b-disaster, except now it's harmless because it's intended. The bias's only job, providing a learnable additive offset, is already done by β. So drop b entirely: z = g(BN(Wu)), with its own (γ, β) per normalized dimension. This is the clean resolution of the earlier cautionary tale — I don't fight the bias's drift, I delete the bias, because β subsumes it.

How to pool, for conv layers. The defining property of a convolution is weight sharing: one filter slides across all spatial locations of a feature map, so every location of that map is produced by the *same* weights — that's what makes convolutions translation-equivariant and parameter-efficient. The normalization must honor this: different locations of one feature map are computed identically, so they must be *normalized* identically. If I instead treated each (channel, location) pair as its own activation with its own mean/variance and its own (γ, β), I'd apply a *different* affine at each spatial position, which breaks the translation equivariance the convolution was built to have — a feature appearing at location (i,j) versus (i′,j′) would be transformed differently, which is exactly the symmetry a conv layer exists to avoid. So instead, for a given feature map I pool *all* its values together — across the m examples in the batch *and* all spatial locations — into one mean and one variance, and learn a single (γ⁽ᵏ⁾, β⁽ᵏ⁾) per feature map (not per activation). For a batch of size m and feature maps of size p×q, the effective sample size is m′ = m·p·q, which is a bonus: far more samples per statistic than the fully-connected case, so the batch estimates are tighter. At inference the same per-feature-map affine is applied at every location, equivariance intact.

Now the question I opened with — why does all this finally let me crank the learning rate? "It's more stable" is too vague. With the pre-activation normalized, backprop becomes invariant to the *positive* scale of the weights, in the idealized algebra where ε is zero, and approximately whenever ε is tiny compared with a²σ²_B. For a scalar a > 0, BN(Wu) = BN((aW)u): scaling W by a multiplies both the value Wu and its batch standard deviation √σ²_B by a, and the normalization divides one by the other, so a cancels and the normalized output is unchanged. Differentiate this identity under the same ε-negligible approximation. Because BN((aW)u) equals BN(Wu) as a function of u, the Jacobian w.r.t. the layer input is untouched by the weight scale:

    ∂BN((aW)u)/∂u = ∂BN(Wu)/∂u.

So the gradient that flows *back to the layer below* doesn't depend on how big this layer's weights are at all — the explosion mechanism (big weights amplify the backprop signal, which makes weights bigger, which amplifies more) is cut. And for the weight itself, since aW sits exactly where W did,

    ∂BN((aW)u)/∂(aW) = (1/a) · ∂BN(Wu)/∂W,

so *larger* positive weight scales receive *smaller* gradients — the update on an over-large weight shrinks by 1/a, which pulls the parameter scale back rather than letting it run away. The weight scale self-stabilizes. That's the precise mechanism: in a plain net a too-high learning rate inflates the weight scale, which amplifies gradients, which inflates the scale further until it diverges; under BN the inflation feeds *negatively* back on the gradient, so I can raise the learning rate without that runaway. The scale invariance also means initialization stops being delicate — multiply all the weights of a normalized layer by any positive constant and, up to the ε caveat, the forward map is unchanged — which kills the second coat.

I can push the gradient-propagation story one conjectural step further. Consider the map between two consecutive *normalized* vectors, ẑ = F(x̂), and pretend it's locally linear, F(x̂) ≈ J x̂, with x̂ and ẑ Gaussian, unit-covariance, and uncorrelated. Then I = Cov[ẑ] = J Cov[x̂] Jᵀ = J·I·Jᵀ = JJᵀ, so JJᵀ = I and every singular value of J equals 1. Saxe et al. 2013 argue that Jacobian singular values near 1 are exactly what preserve gradient magnitudes through depth (a singular value far from 1 either amplifies or shrinks the gradient as it crosses the layer, compounding catastrophically over many layers). It's only a heuristic — the map isn't really linear and the activations aren't really Gaussian or independent — but it says the right thing: normalizing the activations should drive the layer Jacobians toward the well-conditioned regime, so gradients neither explode nor vanish through depth. The precise effect is more than I can prove, but the direction is clear.

And a gift I didn't design for, from the batch coupling I flagged earlier. Because yᵢ depends on the whole batch through μ_B and σ²_B, a given example's representation is *jittered* by whichever other examples happen to land in its mini-batch — the same example normalized in two different batches comes out slightly differently. The network therefore never sees a deterministic function of a single example during training; it sees the example perturbed by batch-dependent noise. That is noise injection, which is regularization — the same job Dropout (Srivastava et al. 2014) does by randomly masking units, except here it falls out of the normalization for free, sourced from the random batch composition rather than an explicit mask. Concretely, each activation gets shifted by a random μ_B and scaled by a random 1/√σ²_B that fluctuate batch to batch, so downstream layers must learn to be robust to that variation — a Dropout-like pressure without an explicit mask. So I'd expect to be able to cut or drop Dropout in a normalized net, which is welcome because Dropout's noise slows convergence and I'm trying to go *faster*.

Let me lay it down as code, the way I'd actually build it. In my harness every transform is one more layer with a forward that caches what the backward needs and a backward that returns the input gradient plus gradients for any learnable parameters; my new transform is exactly such a layer, slotted between the affine producer and the nonlinearity, with γ and β registered so the same SGD updates them like any W. I'll write the staged form first, because it makes the three-path backward graph obvious. Forward, training mode:

```python
import numpy as np

def batchnorm_forward(x, gamma, beta, bn_param):
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)   # decay for the running stats
    N, D = x.shape
    running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
    running_var  = bn_param.get('running_var',  np.zeros(D, dtype=x.dtype))

    if mode == 'train':
        mu      = 1.0 / N * np.sum(x, axis=0)        # mu_B, per feature
        xmu     = x - mu                             # center
        carre   = xmu ** 2
        var     = 1.0 / N * np.sum(carre, axis=0)    # sigma^2_B (biased, /N)
        sqrtvar = np.sqrt(var + eps)                 # eps guards near-constant features
        invvar  = 1.0 / sqrtvar
        va2     = xmu * invvar                       # x_hat = (x-mu)/sqrt(var+eps)
        va3     = gamma * va2
        out     = va3 + beta                         # y = gamma*x_hat + beta
        # accumulate running batch-stat estimates for inference
        running_mean = momentum * running_mean + (1.0 - momentum) * mu
        running_var  = momentum * running_var  + (1.0 - momentum) * var  # biased /N estimate
        cache = (mu, xmu, carre, var, sqrtvar, invvar, va2, va3,
                 gamma, beta, x, bn_param)
    elif mode == 'test':
        # inference: frozen running stats -> a single affine map, no batch dependence
        xhat = (x - running_mean) / np.sqrt(running_var + eps)
        out  = gamma * xhat + beta
        cache = (running_mean, running_var, gamma, beta, bn_param)
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    bn_param['running_mean'] = running_mean
    bn_param['running_var']  = running_var
    return out, cache
```

The staged backward walks that graph in reverse, each block being one chain-rule term I derived, with the two indirect paths into xᵢ folded in by accumulating into `dxmu` (the σ²_B path) and `dmu` (the μ_B path):

```python
def batchnorm_backward(dout, cache):
    mu, xmu, carre, var, sqrtvar, invvar, va2, va3, gamma, beta, x, bn_param = cache
    eps = bn_param.get('eps', 1e-5)
    N, D = dout.shape

    dbeta   = np.sum(dout, axis=0)             # dl/dbeta = sum_i dl/dy
    dva2    = gamma * dout                     # dl/dx_hat = dl/dy * gamma
    dgamma  = np.sum(va2 * dout, axis=0)       # dl/dgamma = sum_i dl/dy * x_hat

    dxmu    = invvar * dva2                     # direct path: dl/dx_hat * 1/sqrt(var+eps)
    dinvvar = np.sum(xmu * dva2, axis=0)
    dsqrtvar = -1.0 / (sqrtvar ** 2) * dinvvar
    dvar    = 0.5 * (var + eps) ** (-0.5) * dsqrtvar   # dl/dsigma^2_B
    dcarre  = 1.0 / N * np.ones(carre.shape) * dvar
    dxmu   += 2 * xmu * dcarre                  # add the sigma^2_B path into x_i

    dx      = dxmu
    dmu     = -np.sum(dxmu, axis=0)             # dl/dmu_B
    dx     += 1.0 / N * np.ones(dout.shape) * dmu     # add the mu_B path into x_i
    return dx, dgamma, dbeta
```

If I grind the algebra and collapse those three paths into one closed form — substitute dl/dsigma^2_B and dl/dmu_B back into dl/dx_i, use Σ(xᵢ−μ_B)=0 to kill the spurious term, and factor out γ(σ²+ε)^{−1/2}/m — the whole dx becomes a single expression, same result, far cheaper (no graph to walk, no intermediate arrays):

```python
def batchnorm_backward_alt(dout, cache):
    mu, xmu, carre, var, sqrtvar, invvar, va2, va3, gamma, beta, x, bn_param = cache
    eps = bn_param.get('eps', 1e-5)
    N, D = dout.shape
    dbeta  = np.sum(dout, axis=0)
    dgamma = np.sum((x - mu) * (var + eps) ** (-0.5) * dout, axis=0)
    dx = (1.0 / N) * gamma * (var + eps) ** (-0.5) * (
            N * dout
            - np.sum(dout, axis=0)
            - (x - mu) * (var + eps) ** (-1.0) * np.sum(dout * (x - mu), axis=0))
    return dx, dgamma, dbeta
```

The first term N·dout is the direct path, the −Σdout is the μ_B path pulling out the batch mean of the upstream gradient, and the last term is the σ²_B path pulling out its correlation with x̂ — the three paths, now visible as three terms in one line. The convolutional version is the identical operation with the statistics pooled per channel over batch *and* the two spatial axes, one (γ,β) per channel, which I get by just reducing over axes (0,2,3) on an (N,C,H,W) tensor — the effective batch is m′ = N·H·W per channel:

```python
def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    N, C, H, W = x.shape
    eps = bn_param.get('eps', 1e-5); momentum = bn_param.get('momentum', 0.9)
    running_mean = bn_param.get('running_mean', np.zeros(C, dtype=x.dtype))
    running_var  = bn_param.get('running_var',  np.zeros(C, dtype=x.dtype))
    if bn_param['mode'] == 'train':
        mu   = (np.sum(x, axis=(0, 2, 3)) / (N*H*W)).reshape(1, C, 1, 1)
        var  = (np.sum((x - mu) ** 2, axis=(0, 2, 3)) / (N*H*W)).reshape(1, C, 1, 1)
        xhat = (x - mu) / np.sqrt(var + eps)            # shared per channel over all locations
        out  = gamma.reshape(1, C, 1, 1) * xhat + beta.reshape(1, C, 1, 1)
        bn_param['running_mean'] = momentum * running_mean + (1.0 - momentum) * np.squeeze(mu)
        bn_param['running_var']  = momentum * running_var  + (1.0 - momentum) * np.squeeze(var)
        cache = (mu, var, x, xhat, gamma, beta, bn_param)
    else:  # test branch applies the same per-channel affine using running stats
        mu   = running_mean.reshape(1, C, 1, 1); var = running_var.reshape(1, C, 1, 1)
        xhat = (x - mu) / np.sqrt(var + eps)
        out  = gamma.reshape(1, C, 1, 1) * xhat + beta.reshape(1, C, 1, 1)
        cache = (mu, var, x, xhat, gamma, beta, bn_param)
    return out, cache
```

So the causal chain, end to end: the real pain — can't raise the learning rate, can't use sigmoids, init is fragile — traces to one cause, internal covariate shift, the distribution of each layer's inputs drifting as the parameters below it move. Whitening every layer would fix it, but doing it as a side step makes the optimizer fight the normalizer and a bias drifts to infinity while the loss stays flat, so the normalization *must* be inside the model and differentiable, carrying both ∂Norm/∂x and ∂Norm/∂X. Full joint whitening carries them correctly but is O(d³), non-differentiable through the matrix root, and singular on mini-batches, so I retreat to per-dimension normalization — licensed by LeCun et al. (it helps even undecorrelated) and, crucially, the very simplification that dissolves the singularity. Per-dimension normalization would shrink the representable class, so a learnable γ,β restores it, recovering the identity at γ=√(Var+ε), β=E. Using the mini-batch's own statistics makes the dependence on the data a free, differentiable function the gradient owns; the three-path backprop (direct, via σ²_B, via μ_B) realizes it and collapses to a single dx. The batch coupling that makes training differentiable would make inference non-deterministic, so at test time I freeze stored mean and variance estimates; the exact population variance estimate has the m/(m−1) Bessel correction, while the common running-stat code stores the biased batch variance directly. Convolutions pool per feature map over batch and space to keep the translation equivariance weight sharing demands. And the positive-scale invariance BN((aW)u)≈BN(Wu) when ε is negligible — larger weights, smaller gradients, Jacobian singular values toward 1 — is exactly what unlocks the high learning rates, tames the saturating nonlinearities, and frees me from babying the initialization, with the batch-noise regularization throwing in a partial replacement for Dropout on the side.
