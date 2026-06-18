OK, let me think this through from scratch. The thing nagging me is a leaderboard fact: on bits-per-dimension for CIFAR-10 and ImageNet, autoregressive models keep winning, and the noise-corruption generative models — the ones that add Gaussian noise to an image and learn to walk it back — make gorgeous samples but their likelihood numbers are mediocre. So either these corruption models are fundamentally bad density estimators, or we're training and evaluating them with the wrong objective. I don't believe the first. Let me try to figure out what the *right* objective actually is, derived honestly, and see where the current practice deviates from it.

Here's the framing that I keep coming back to. A corruption model is a latent-variable model. We have a forward process that takes data `x` and produces noisier and noisier versions `z_t`, with `t` running from 0 (basically clean) to 1 (basically pure noise). We have a reverse process that's supposed to invert it. That is *literally* a VAE: there's an inference distribution `q(z|x)` and a generative distribution `p(x,z)`, and the natural objective is the evidence lower bound. The twist versus a plain VAE is that the inference side isn't learned — it's a fixed Gaussian corruption — and the latent isn't one layer, it's a whole Markov chain of layers. So this is a VAE with a fixed, infinitely-deep inference network. If I take that seriously, the ELBO should just fall out, and I should be able to ask the cleanest possible question: in the limit of infinite depth, what does the loss actually *depend on*? Not the algorithm — the dependency structure.

Let me pin the forward process. The most workable choice is to give the marginal in closed form rather than only the per-step kernel:

  q(z_t | x) = N(α_t x, σ_t² I),

with `α_t, σ_t` smooth positive scalar functions of `t`. This already feels right because it lets me jump to any noise level directly. The single number that says "how corrupted is `z_t`" is the signal-to-noise ratio

  SNR(t) = α_t² / σ_t²,

and I'll insist it's strictly decreasing in `t` — that's just the statement that things get noisier as `t` grows. Two familiar regimes live inside this: variance-preserving with `α_t² = 1 − σ_t²`, and variance-exploding with `α_t² = 1`. I'll keep it general for now and not commit.

Now the generative side inverts this. Discretize time into `T` steps, `s(i)=(i−1)/T`, `t(i)=i/T`. The model is

  p(x) = ∫ p(z_1) p(x|z_0) ∏_i p(z_{s(i)} | z_{t(i)}).

For the top, if `SNR(1)` is small enough then `q(z_1|x) ≈ N(0,I)`, so set `p(z_1) = N(0,I)`. For the bottom, I want `p(x|z_0)` close to the unknown `q(x|z_0)`; with `SNR(0)` large, `z_0` almost determines `x`, so a simple per-pixel factorized decoder works. The interesting part is the middle: each `p(z_s|z_t)`.

The VAE ELBO for a chain like this decomposes the way these always do:

  −log p(x) ≤ −VLB(x) = KL(q(z_1|x)‖p(z_1)) + E_{q(z_0|x)}[−log p(x|z_0)] + L_T,

a prior loss, a reconstruction loss, and a "diffusion loss" `L_T` that's the sum over the chain of transition KLs. The first two are standard VAE terms — reparameterize, Monte Carlo, done. Everything subtle is in `L_T`. So let me grind on it.

  L_T = Σ_{i=1}^T E_{q(z_t|x)} KL[ q(z_s | z_t, x) ‖ p(z_s | z_t) ].

I need both distributions inside the KL. First, what is `q(z_t|z_s)`? Since `q(z_t|x)=N(α_t x, σ_t²)` and the chain is Markov, `z_t` given `z_s` is Gaussian. Matching: if `z_s = α_s x + σ_s ε_s` and `z_t = α_t x + σ_t ε_t`, then `z_t = (α_t/α_s) z_s + (noise)`, so

  q(z_t|z_s) = N(α_{t|s} z_s, σ²_{t|s} I),  α_{t|s} = α_t/α_s,  σ²_{t|s} = σ_t² − α_{t|s}² σ_s².

(The variance is what's left over after the deterministic part `α_{t|s} z_s` already carries `α_{t|s}² σ_s²` of variance; total must be `σ_t²`.) Good.

Now `q(z_s | z_t, x)`. This is a Bayesian posterior: prior `q(z_s|x)=N(α_s x, σ_s²)`, likelihood `q(z_t|z_s)=N(α_{t|s} z_s, σ²_{t|s})`. Gaussian prior times linear-Gaussian likelihood gives a Gaussian posterior. The standard precision/mean update: for prior `N(μ_A, σ_A²)` and likelihood `N(a x, σ_B²)`, the posterior precision is `σ̃^{-2} = σ_A^{-2} + a² σ_B^{-2}` and `μ̃ = σ̃² (σ_A^{-2} μ_A + a σ_B^{-2} y)`. Plugging in (`μ_A = α_s x`, `σ_A² = σ_s²`, `a = α_{t|s}`, `σ_B² = σ²_{t|s}`, `y = z_t`):

  σ_Q^{-2}(s,t) = σ_s^{-2} + α_{t|s}² σ_{t|s}^{-2}.

Let me simplify that. Common denominator: `σ_Q^{-2} = (σ²_{t|s} + α_{t|s}² σ_s²)/(σ²_{t|s} σ_s²)`. The numerator is `σ²_{t|s} + α_{t|s}² σ_s² = (σ_t² − α_{t|s}² σ_s²) + α_{t|s}² σ_s² = σ_t²`. So

  σ_Q^{-2} = σ_t² / (σ²_{t|s} σ_s²),   σ_Q²(s,t) = σ²_{t|s} σ_s² / σ_t².

And the mean:

  μ_Q(z_t, x; s,t) = (α_{t|s} σ_s² / σ_t²) z_t + (α_s σ²_{t|s} / σ_t²) x.

Now the model side. I get to choose `p(z_s|z_t)`. The cleanest possible choice: make it *the same* posterior, but with the unknown `x` replaced by a prediction `x̂_θ(z_t, t)` from a denoising network. So

  p(z_s|z_t) = q(z_s | z_t, x = x̂_θ(z_t, t)) = N( μ_θ(z_t; s,t), σ_Q²(s,t) I ),

with `μ_θ` identical to `μ_Q` but with `x → x̂_θ`. The point of this choice is that the two distributions in the KL have *identical variance* — only their means differ — which collapses the Gaussian KL to a pure squared-error term:

  KL = (1 / 2σ_Q²) ‖ μ_Q − μ_θ ‖².

And the mean difference is gorgeous: the `z_t` term is common to both, so it cancels, leaving only the data term:

  μ_Q − μ_θ = (α_s σ²_{t|s} / σ_t²) ( x − x̂_θ ).

So

  KL = (1/2) · (σ_t² / (σ²_{t|s} σ_s²)) · (α_s² σ⁴_{t|s} / σ_t⁴) · ‖x − x̂_θ‖².

Let me crunch the coefficient, because I suspect it simplifies to something I'll recognize:

  = (1/2) · (α_s² σ²_{t|s} / (σ_s² σ_t²)) · ‖x − x̂_θ‖².

Substitute `σ²_{t|s} = σ_t² − α_{t|s}² σ_s²`:

  = (1/2) · (α_s² (σ_t² − α_{t|s}² σ_s²) / (σ_s² σ_t²)) · ‖·‖²
  = (1/2) · (α_s²/σ_s² − α_s² α_{t|s}² / σ_t²) · ‖·‖².

And `α_s² α_{t|s}² = α_s² (α_t/α_s)² = α_t²`, so the second term is `α_t²/σ_t²`. Therefore

  KL = (1/2) (α_s²/σ_s² − α_t²/σ_t²) ‖x − x̂_θ‖² = (1/2) ( SNR(s) − SNR(t) ) ‖x − x̂_θ‖².

There it is. The entire per-transition KL is *half the drop in SNR across the step, times the denoising squared error*. Every messy `σ_{t|s}`, every `α_{t|s}` cancelled, and what's left is written purely in SNR. Stare at this — it means the whole diffusion loss only ever sees the noise schedule through `SNR`. Summing over the chain and reparameterizing `z_t = α_t x + σ_t ε`:

  L_T = (1/2) E_{ε} Σ_{i=1}^T ( SNR(s) − SNR(t) ) ‖x − x̂_θ(z_t; t)‖².

To avoid computing all `T` terms per step, draw `i ~ Uniform{1,…,T}` and scale by `T`:

  L_T = (T/2) E_{ε, i} [ ( SNR(s) − SNR(t) ) ‖x − x̂_θ(z_t; t)‖² ],

an unbiased Monte Carlo estimator. Clean.

Now the obvious next question: how big should `T` be? Is more steps always better for the bound? Let me compare `L_T` to `L_{2T}` at a fixed SNR function. Insert a midpoint `t' = t − 0.5/T` into each segment so I can write `L_T` with the split telescoped:

  L_T = (1/2) E_ε Σ_i [ (SNR(s) − SNR(t')) + (SNR(t') − SNR(t)) ] ‖x − x̂_θ(z_t; t)‖²,

where crucially *both* sub-terms use the denoiser evaluated at the coarse point `z_t, t`. Doubling the steps gives

  L_{2T} = (1/2) E_ε [ Σ_i (SNR(s) − SNR(t')) ‖x − x̂_θ(z_{t'}; t')‖² + Σ_i (SNR(t') − SNR(t)) ‖x − x̂_θ(z_t; t)‖² ],

i.e. the first sub-term now uses the *finer* point `z_{t'}, t'`. Subtract:

  L_{2T} − L_T = (1/2) E_ε Σ_i (SNR(s) − SNR(t')) ( ‖x − x̂_θ(z_{t'}; t')‖² − ‖x − x̂_θ(z_t; t)‖² ).

Now `t' < t`, so `z_{t'}` is *less* noisy than `z_t`. Predicting `x` from a less-noisy input is strictly easier, so for a decent denoiser the first MSE is smaller than the second: the bracket is negative, and `SNR(s) − SNR(t') > 0`. Hence `L_{2T} − L_T < 0`. Doubling the steps *always* lowers the loss (given a good enough model). Intuitively the discrete loss is an upper Riemann sum of an integral of a decreasing function — refine the partition, the upper sum drops. So I should just take `T → ∞`. No more agonizing over a depth hyperparameter; the best depth is infinite, and I should derive the continuous-time loss directly.

Take the limit. Write `L_T` as a function of `τ = 1/T`:

  L_T = (1/2) E_{ε, i} [ ( (SNR(t − τ) − SNR(t)) / τ ) ‖x − x̂_θ(z_t; t)‖² ].

As `τ → 0`, the difference quotient `(SNR(t−τ) − SNR(t))/τ → −SNR'(t)`, and `i ~ Uniform{1,…,T}` becomes `t ~ Uniform[0,1]`:

  L_∞ = −(1/2) E_{ε, t~U[0,1]} [ SNR'(t) ‖x − x̂_θ(z_t; t)‖² ]
      = −(1/2) E_ε ∫_0^1 SNR'(t) ‖x − x̂_θ(z_t; t)‖² dt.

`SNR` is decreasing so `SNR' < 0` and the loss is positive. This is *short*. The infinite-depth bound is one integral: the squared denoising error, weighted by the rate at which SNR is falling.

Now the move that I think is the real payoff. `SNR(t)` is strictly monotonic, hence invertible. Change variables to `v ≡ SNR(t)`, so `t = SNR^{-1}(v)` and `dv = SNR'(t) dt`. Let `α_v, σ_v, z_v, x̃_θ(z_v, v)` denote the corresponding objects as functions of `v`. Then

  L_∞ = −(1/2) E_ε ∫_0^1 SNR'(t) ‖x − x̃_θ(z_v, v)‖² dt = (1/2) E_ε ∫_{SNR_min}^{SNR_max} ‖x − x̃_θ(z_v, v)‖² dv,

where the minus sign and the swap of limits (`t:0→1` maps to `v:SNR_max→SNR_min` since SNR decreases) combine to give a clean positive integral from `SNR_min = SNR(1)` to `SNR_max = SNR(0)`.

Let me read what this says, because it's startling. The integrand `‖x − x̃_θ(z_v, v)‖²` depends on the schedule only through `z_v`. And what is `z_v`? Since `v = α_v²/σ_v²`, we have `σ_v = α_v/√v`, so

  z_v = α_v x + σ_v ε = α_v ( x + ε/√v ).

The data `z_v` feeds into the network only up to that overall factor `α_v`, and the *informative content* — `x + ε/√v` — depends on `v` alone, not on `α_v` and `σ_v` separately. So once I fix the two endpoints `SNR_min` and `SNR_max`, the value of the continuous-time loss is *completely independent of the shape of `SNR(t)` in between*. The whole noise schedule, the thing everyone hand-tunes — β-linear, cosine — enters the bound only through its two endpoints. Everything between `t=0` and `t=1` is gauge; it washes out.

Hold on — does this mean different "diffusion processes" that people thought were genuinely different are the same model? Take two specs A and B with the same `(SNR_min, SNR_max)`. Since `z_v = α_v(x + ε/√v)` for any spec, `z_v^A = (α_v^A/α_v^B) z_v^B` — the latents are identical up to a deterministic rescaling. So if I define `x̃_θ^B(z, v) ≡ x̃_θ^A((α_v^A/α_v^B) z, v)`, the denoiser sees the same effective input, the reconstruction error matches at every `v`, the loss matches, and since the reverse-process conditionals are functions of the denoiser, the *whole generative distribution* `p(x)` matches — up to that trivial latent rescaling. The variance-preserving and variance-exploding processes, long treated as distinct, are the *same continuous-time model*. That's a real unification, and it falls straight out of the change of variables.

This invariance is not just pretty; it's *operationally useful*, and that's the part I want to exploit. The bound's value is fixed by the endpoints, so I'm free to use the interior shape of the schedule for something else. What else do I care about? The Monte Carlo estimator of `L_∞` — I sample `t` and `ε` and form a one-sample (or minibatch) estimate. Its *variance* controls how fast optimization goes: a high-variance loss gradient is a noisy gradient. And the variance of the estimator very much *does* depend on the schedule shape (it changes how my samples of `t` are distributed across SNR levels), even though the *expectation* does not. So: learn the schedule shape to *minimize the variance of the loss estimator*, while the bound itself is held by the endpoints. That's a free lunch handed to me by the invariance — I'd never get away with reshaping the schedule if it changed the objective, but it doesn't.

Let me set that up concretely. Parameterize the schedule so I learn it. I'll work with `γ_η(t) ≡ −log SNR(t)`, so `SNR(t) = exp(−γ_η(t))`, where `γ_η` is monotonically *increasing* (SNR decreasing). I need to *guarantee* monotonicity, otherwise the change-of-variables invertibility breaks. The trick: build `γ_η` from linear layers with non-negative weights — a monotone neural network. Something like `γ̃_η(t) = l_1(t) + l_3(φ(l_2(l_1(t))))` with all weights constrained positive and `φ` a sigmoid; composition of monotone-increasing maps is monotone-increasing. A wide middle layer (≈1024 units) gives it flexibility to bend the schedule. For variance minimization I then rescale this raw network to hit the endpoints exactly:

  γ_η(t) = γ_0 + (γ_1 − γ_0) ( γ̃_η(t) − γ̃_η(0) ) / ( γ̃_η(1) − γ̃_η(0) ),

with `γ_0 = −log SNR_max`, `γ_1 = −log SNR_min`. Now `γ_η(0)=γ_0`, `γ_η(1)=γ_1` no matter what the interior weights do. So I split the training: `γ_0, γ_1` (the endpoints, which *do* move the bound) are optimized w.r.t. the VLB; the interior weights `η` (which don't move the bound) are optimized to minimize the estimator's variance.

How do I optimize for variance without a separate, expensive backward pass? Here's the cute part. Do SGD on the *squared* Monte Carlo loss `L^{MC}_∞²`. Because `E[L^{MC}²] = (E[L^{MC}])² + Var[L^{MC}] = L_∞² + Var`, and `L_∞²` is independent of `η` (the bound doesn't depend on the interior shape!), we get

  ∇_η E[L^{MC}_∞²] = ∇_η Var[L^{MC}_∞].

So gradient descent on the squared loss is *exactly* gradient descent on the variance — for free, no separate variance estimator. And I can get this gradient without a second backprop through the denoiser by the chain rule through SNR: `d/dη [L^{MC}²] = d/dSNR[L^{MC}²] · d SNR/dη`, with `d/dSNR[L^{MC}²] = 2 (d L^{MC}/d SNR) ⊙ L^{MC}`, and `d L^{MC}/d SNR` is already produced as a byproduct of the main backward pass. Negligible overhead. I'll also sample `t` with a low-discrepancy / antithetic scheme — for a minibatch of `k`, draw one `u_0 ~ U[0,1]` and set `t^i = mod(u_0 + i/k, 1)` — each `t^i` is still marginally uniform but the batch tiles `[0,1]` evenly, which further cuts variance.

Now let me reconnect to how the field actually parameterizes the denoiser, because I want my objective to look like theirs and to be numerically sane. There are three equivalent views of the prediction target. The denoiser `x̂_θ` predicts the clean data. But from `z_t = α_t x + σ_t ε` I can read off the noise, `ε̂_θ = (z_t − α_t x̂_θ)/σ_t`, equivalently `x̂_θ = (z_t − σ_t ε̂_θ)/α_t`. And the score: `∇_{z_t} log q(z_t|x) = −ε/σ_t`, so a score model `s_θ = (α_t x̂_θ − z_t)/σ_t²` is just another linear reparameterization. Three faces of one model — denoising, noise-prediction, score — used interchangeably in the literature. Which to actually parameterize? Predicting `ε` is the well-behaved choice numerically (the target has fixed unit scale across noise levels), so I'll have the network output `ε̂_θ` and define `x̂_θ` from it. Plugging `x − x̂_θ = −σ_t(ε − ε̂_θ)/α_t`... let me just substitute into the continuous loss using a variance-preserving spec where things tidy up. With `α_t = √sigmoid(−γ)`, `σ_t = √sigmoid(γ)`, `SNR = exp(−γ)`, and `SNR'(t) = −γ'(t) exp(−γ(t))`, I get `SNR'(t)‖x−x̂‖² = −γ'(t) exp(−γ) · (σ_t²/α_t²)‖ε−ε̂‖² = −γ'(t) exp(−γ) · exp(γ)‖ε−ε̂‖² = −γ'(t)‖ε−ε̂‖²`. So

  L_∞ = (1/2) E_{ε,t} [ γ'_η(t) ‖ε − ε̂_θ(z_t; t)‖² ].

That's the loss I'll actually optimize: half the time-derivative of `γ` times the noise-prediction MSE. And the discrete-time version, by the same substitution into `(T/2)(SNR(s)−SNR(t))‖x−x̂‖²` with `SNR(s)−SNR(t) = exp(−γ(s)) − exp(−γ(t))` and the `σ_t²/α_t²` factor:

  L_T = (T/2) E_{ε,i} [ ( exp(γ_η(t) − γ_η(s)) − 1 ) ‖ε − ε̂_θ(z_t; t)‖² ],

where I deliberately wrote the coefficient as `exp(γ(t)−γ(s)) − 1`. Why that form? Because `γ(t)−γ(s)` is a small positive number for fine time steps, and `exp(small) − 1` computed naively in 32-bit floating point loses almost all its significant digits — the intermediate `exp(small)` rounds toward 1, exactly where floating point is worst. There's a stable primitive `expm1(x) = exp(x) − 1` for precisely this. Writing the loss with `expm1` lets me train in fp32 (or bf16) where naive implementations of the discrete loss needed fp64. Same story shows up in the sampler's posterior variance `σ²_{t|s} = −expm1(softplus(γ(s)) − softplus(γ(t)))` — using `expm1`/`softplus` keeps the arithmetic away from the bad region near 1.

The other two ELBO terms I should make concrete. The prior loss is `KL(q(z_1|x)‖N(0,I))`, closed form for two Gaussians: with `var_1 = sigmoid(γ(1))` and mean `√(1−var_1) x`, it's `(1/2) Σ ( (1−var_1)x² + var_1 − log var_1 − 1 )`. The reconstruction loss handles the discrete 8-bit data: choose `p(x_i|z_{0,i}) ∝ q(z_{0,i}|x_i)`, normalized over the 256 possible pixel values — a discretized Gaussian decoder. With large `SNR(0)` this is a tight approximation to `q(x|z_0)` and is exact to the extent dimensions are independent. So `−log p(x|z_0)` is just a categorical log-likelihood read off Gaussian logits over the discrete grid.

There's one wall this clean theory doesn't address: fine-scale detail. Likelihood is brutally sensitive to the exact low-order bits of each pixel, and my reconstruction decoder `p(x|z_0)` is deliberately weak, so the burden of fine detail falls on the denoiser `x̂_θ` at the low-noise end. But at low noise, `q(z_t)` is sharply peaked because the underlying 8-bit data is discrete; a smooth convolutional stack can have a hard time reacting to tiny scalar changes in `z_t`. I need to amplify small input changes before they enter the network. Fourier features do exactly that: append channels `sin(2^n π z)` and `cos(2^n π z)` for high integer `n`. The frequencies I want are `n ∈ {7, 8}`; if an implementation helper multiplies by `2π`, then the same band appears as `range(6, 8)`, because `2^6·2π = 2^7π` and `2^7·2π = 2^8π`. If these features let the denoiser carry the low-noise detail, then the high-SNR endpoint can be pushed upward without asking a smooth network to infer discrete spikes from almost raw pixels.

Let me also place the perceptual-quality recipes relative to my bound, because it clarifies why they trail on likelihood. Write a weighted version,

  L_∞(x, w) = (1/2) E_ε ∫_{SNR_min}^{SNR_max} w(v) ‖x − x̃_θ(z_v, v)‖² dv,

which in `γ`-form is `(1/2) E ∫_0^1 γ'(t) w(exp(−γ(t))) ‖ε − ε̂_θ‖² dt`. Setting `w(v) = 1` is exactly the VLB. The "simple" objective people minimize, `∫ ‖ε − ε̂_θ‖² dt`, has no `γ'(t)` factor — comparing, it corresponds to `w(SNR(t)) = 1/γ'(t)`. That reweighting puts relatively more emphasis on noisy, low-SNR levels (good for global coherence, hence good FID) and less on the fine high-SNR detail (bad for likelihood). The variance-exploding score-matching recipe, with `σ` geometric, has `γ'(t)` constant, so its implied `w` is constant — secretly consistent with the VLB — but it never *learns* the endpoints, leaving likelihood on the table. So the leaderboard gap was never about the model class being a bad density estimator; it was about training a quality-weighted surrogate and hand-fixing the schedule. Take `w=1`, learn the endpoints, add Fourier features, and the bound becomes competitive.

One more reassurance before code: is this objective even consistent — does minimizing it recover the true marginal? The score view says yes. Predicting `ε` is predicting `∇ log q(z_t|x)` up to the `−1/σ_t` factor, and the denoising-score-matching identity says, for any positive weighting `c`, that `(1/2)E_{q(z_t)}‖√c(∇log q(z_t) − s_θ)‖²` equals `(1/2)E_{q(x,z_t)}‖√c(∇log q(z_t|x) − s_θ)‖²` plus a constant independent of `θ` — the cross terms match after pulling the gradient through `q(z_t)=∫q(z_t|x)q(x)dx` and using `∇q = q∇log q`. So at the optimum `s_θ^*(z_t) = ∇log q(z_t)`: the trained model recovers the true marginal score at every noise level. Good — the bound I derived is the right thing to chase.

Now I need the implementation to respect every sign and square root I just derived. In the variance-preserving parameterization, I must sample `z_t` as `sqrt(1 - sigmoid(γ_t)) x + sqrt(sigmoid(γ_t)) ε`, not as the variances themselves times `x` and `ε`. The reconstruction path samples `z_0` the same way and then passes the rescaled value `z_0 / sqrt(1 - var_0) = x + exp(0.5γ_0)ε` to the discrete decoder. The prior KL is the diagonal-Gaussian expression `0.5 Σ[(1 - var_1)x² + var_1 - log var_1 - 1]`.

The forward pass now has the right center of gravity: it returns `loss_recon`, `loss_klz`, and `loss_diff`; antithetic time sampling draws a single offset and tiles the minibatch across `[0,1]`; the continuous branch uses a directional derivative through `γ(t)` to get `γ'(t)` and forms `0.5 γ'(t)‖ε - ε̂‖²`; the discrete branch snaps `t` to the grid, sets `s=t-1/T`, and forms `0.5 T expm1(γ_t - γ_s)‖ε - ε̂‖²`. The sampler has to use the opposite stable difference, `c=-expm1(γ_s-γ_t)`, then step with `sqrt(sigmoid(-γ_s)/sigmoid(-γ_t)) (z_t - sqrt(sigmoid(γ_t)) c ε̂) + sqrt((1 - sigmoid(-γ_s))c)ε`. The signs now agree: `γ_t>γ_s`, so the training coefficient `expm1(γ_t-γ_s)` is positive; `γ_s-γ_t<0`, so `c` is also positive.

For schedule learning I need to keep two roles separate. The endpoints move the actual bound, so they belong in the ordinary VLB gradient. The interior shape does not move the continuous-time objective, so if I use the full monotone schedule there, its natural job is variance reduction through the squared Monte Carlo loss. A simpler implementation can still train a scalar or fixed schedule through the summed BPD objective, but that is an implementation choice, not the reason the endpoint-invariance argument works.

Putting it together in JAX/Flax, the center of gravity is one forward call returning the three loss terms, with the continuous-time branch (`T=0`) using an autodiff `jvp` for `γ'(t)` and the discrete branch using `expm1`:

```python
import jax, jax.numpy as jnp
import flax.linen as nn

class GenerativeModel(nn.Module):
    config: object

    def setup(self):
        self.encdec = EncDec(self.config)            # discrete-data encode/decode
        self.score_model = ScoreUNet(self.config)    # predicts eps_hat from (z_t, gamma_t)
        self.gamma = NoiseSchedule(self.config)      # monotone gamma_eta(t) = -log SNR(t)

    def __call__(self, images, conditioning, deterministic=True):
        g_0, g_1 = self.gamma(0.), self.gamma(1.)
        var_0, var_1 = nn.sigmoid(g_0), nn.sigmoid(g_1)
        x = images
        n_batch = x.shape[0]
        f = self.encdec.encode(x)                    # x -> [-1, 1]

        # reconstruction loss: -log p(x | z_0), discretized-Gaussian decoder
        eps_0 = jax.random.normal(self.make_rng("sample"), f.shape)
        z_0_rescaled = f + jnp.exp(0.5 * g_0) * eps_0   # z_0 / sqrt(1 - var_0)
        loss_recon = - self.encdec.logprob(x, z_0_rescaled, g_0)

        # prior loss: KL( q(z_1|x) || N(0,I) ), closed form for Gaussians
        mean1_sqr = (1. - var_1) * jnp.square(f)
        loss_klz = 0.5 * jnp.sum(mean1_sqr + var_1 - jnp.log(var_1) - 1., axis=(1, 2, 3))

        # diffusion loss: sample t (antithetic for low variance), build z_t, predict eps
        rng1 = self.make_rng("sample")
        t0 = jax.random.uniform(rng1)
        t = jnp.mod(t0 + jnp.arange(0., 1., step=1. / n_batch), 1.)   # low-discrepancy
        T = self.config.sm_n_timesteps
        if T > 0:
            t = jnp.ceil(t * T) / T                  # snap to the discrete grid
        g_t = self.gamma(t)
        var_t = nn.sigmoid(g_t)[:, None, None, None]
        eps = jax.random.normal(self.make_rng("sample"), f.shape)
        z_t = jnp.sqrt(1. - var_t) * f + jnp.sqrt(var_t) * eps   # z_t = alpha_t x + sigma_t eps
        eps_hat = self.score_model(z_t, g_t, conditioning, deterministic)
        loss_diff_mse = jnp.sum(jnp.square(eps - eps_hat), axis=[1, 2, 3])

        if T == 0:
            # continuous time: L_inf = (1/2) E[ gamma'(t) ||eps - eps_hat||^2 ]
            _, g_t_grad = jax.jvp(self.gamma, (t,), (jnp.ones_like(t),))  # gamma'(t)
            loss_diff = .5 * g_t_grad * loss_diff_mse
        else:
            # discrete time: L_T = (T/2) E[ expm1(g_t - g_s) ||eps - eps_hat||^2 ]
            s = t - (1. / T)
            g_s = self.gamma(s)
            loss_diff = .5 * T * jnp.expm1(g_t - g_s) * loss_diff_mse

        return loss_recon, loss_klz, loss_diff

    def sample(self, i, T, z_t, conditioning, rng):
        # ancestral reverse step p(z_s | z_t), all in expm1/sigmoid form for fp32 safety
        eps = jax.random.normal(jax.random.fold_in(rng, i), z_t.shape)
        t = (T - i) / T; s = (T - i - 1) / T
        g_s, g_t = self.gamma(s), self.gamma(t)
        eps_hat = self.score_model(z_t, g_t * jnp.ones(z_t.shape[0]), conditioning, True)
        a = nn.sigmoid(-g_s); c = - jnp.expm1(g_s - g_t)
        sigma_t = jnp.sqrt(nn.sigmoid(g_t))
        z_s = jnp.sqrt(nn.sigmoid(-g_s) / nn.sigmoid(-g_t)) * (z_t - sigma_t * c * eps_hat) \
              + jnp.sqrt((1. - a) * c) * eps
        return z_s


class NoiseSchedule(nn.Module):
    """gamma_eta(t) = -log SNR(t), built monotone from positive-weight linear layers."""
    config: object
    n_features: int = 1024

    def setup(self):
        g0, g1 = self.config.gamma_min, self.config.gamma_max
        self.l1 = DenseMonotone(1, kernel_init=constant_init(g1 - g0),
                                   bias_init=constant_init(g0))
        self.l2 = DenseMonotone(self.n_features)
        self.l3 = DenseMonotone(1, use_bias=False)

    @nn.compact
    def __call__(self, t):
        t = jnp.reshape(t, (-1, 1))
        h = self.l1(t)                               # linear backbone (sets endpoints)
        _h = 2. * (t - .5)                           # interior bend, learned for variance
        _h = self.l3(2 * (nn.sigmoid(self.l2(_h)) - .5)) / self.n_features
        return jnp.squeeze(h + _h, axis=-1)


class DenseMonotone(nn.Dense):
    """Dense layer with non-negative weights -> monotonic increasing."""
    @nn.compact
    def __call__(self, x):
        kernel = jnp.abs(self.param('kernel', self.kernel_init, (x.shape[-1], self.features)))
        y = x @ kernel
        if self.use_bias:
            y += self.param('bias', self.bias_init, (self.features,))
        return y


class Base2FourierFeatures(nn.Module):
    """Append sin/cos(2^n * pi * z) channels to amplify fine pixel-level detail.
    The multiplier is 2*pi, so start=6, stop=8 yields the n in {7, 8} band of sin(2^n pi z)."""
    start: int = 6; stop: int = 8; step: int = 1

    @nn.compact
    def __call__(self, z):
        freqs = jnp.asarray(range(self.start, self.stop, self.step), z.dtype)
        w = 2. ** freqs * 2 * jnp.pi
        w = jnp.tile(w[None, :], (1, z.shape[-1]))
        h = jnp.repeat(z, len(freqs), axis=-1) * w
        return jnp.concatenate([jnp.sin(h), jnp.cos(h)], axis=-1)
```

And the training objective sums the three terms, converted to bits/dim; the endpoint scalars are optimized against this bound while the schedule interior is optimized to cut estimator variance:

```python
def loss_fn(params, batch, rng):
    loss_recon, loss_klz, loss_diff = model.apply(params, **batch, rngs={"sample": rng})
    to_bpd = 1. / (np.prod(batch["images"].shape[1:]) * np.log(2.))
    bpd = (loss_recon.mean() + loss_klz.mean() + loss_diff.mean()) * to_bpd
    return bpd
```

The whole causal chain in one breath: treat the corruption model as an infinitely-deep VAE and write its ELBO; the per-step KL collapses to half the SNR-drop times the denoising error, so the loss only ever sees the schedule through SNR; more steps always help, so push to continuous time, where the loss is one integral `−½∫SNR'(t)‖x−x̂‖²dt`; change variables to SNR and the bound depends on the schedule *only through its two endpoints*, which both unifies the variance-preserving/exploding processes and frees the interior shape to be tuned for low estimator variance (via SGD on the squared loss, which by `E[L²]=L²+Var` is exactly variance minimization); reparameterize to noise prediction for `½E[γ'(t)‖ε−ε̂‖²]` and write the discrete coefficient as `expm1` for fp32 stability; learn the endpoints against the bound, add Fourier features so the denoiser can carry the fine-scale detail that likelihood demands and so `SNR_max` can be pushed high; and the result is a corruption model that is, at last, a first-rate likelihood model.
