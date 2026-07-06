The loss is the whole task, and with the architecture, optimizer, schedule, and evaluation all frozen,
the floor I must start from is the simplest objective that trains this `AutoencoderKL` at all. Whatever
I eventually build will be a correction to *this*, so I want to get the floor exactly right and
understand precisely what it can and cannot do, because its measured failure is what the next rung
reacts to. The loop has already encoded the image into a diagonal-Gaussian posterior, drawn one
reparameterized code, and decoded it into `recon`; it exposes `posterior.kl()` and hands me `recon`,
`target`, the `posterior`, and `step`. My job is to turn those into one scalar to minimize. So let me
derive what that scalar should be from first principles, then read off the literal fill.

I am training a deep latent-variable model: a latent `z ~ p(z)` pushed through a decoder to paint an
image, `x ~ p_theta(x|z)`. I fix the prior to the most boring thing imaginable, `p(z) = N(0, I)`,
parameter-free, and let all the modeling happen in the decoder — a Gaussian pushed through a deep
enough network can manufacture whatever latent structure the data needs, so expressiveness is not my
problem. My problem is learning. To do maximum likelihood on `theta` I need the marginal
`p_theta(x) = ∫ p(z) p_theta(x|z) dz`, and the instant the decoder is a neural net that integral is
intractable; by Bayes the posterior `p_theta(z|x)` is intractable for the same reason, which kills EM
(its E-step *is* the posterior) and the conjugate mean-field machinery that the coordinate-ascent
variational-Bayes baselines rely on. Those baselines want analytic expectations under a conjugate
family, and a deep decoder is exactly the case where no such family exists. So I replace the true
posterior by a tractable approximation `q_phi(z|x)` and optimize a bound. For any `q` I can write
`log p_theta(x) = L(x) + D_KL(q_phi(z|x) || p_theta(z|x))`, where
`L(x) = E_{q}[log p_theta(x,z) - log q_phi(z|x)]`; since a KL is non-negative, `L(x) ≤ log p_theta(x)`
is a lower bound on the evidence, and the slack is exactly that intractable posterior KL — which I am
happy to leave implicit, because maximizing `L` both raises the marginal likelihood I care about and
tightens `q` toward the true posterior at once. Peeling out the prior gives the form I want to stare
at: `L(x) = E_{q}[log p_theta(x|z)] - D_KL(q_phi(z|x) || p(z))`. Structurally that *is* a regularized
autoencoder — an expected reconstruction log-likelihood minus a divergence pulling the encoder's output
toward the prior — and the regularizer is not an ad-hoc penalty I bolted on the way denoising or sparse
autoencoders need one; it falls out of the probability. That is already a better story than the other
reconstruction baselines on the table. A plain autoencoder minimizes pixel error through a bottleneck
with *no* pressure on the code's distribution, so it is free to scatter codes arbitrarily and learn a
latent that reconstructs the training set without any coherent geometry — fine for memorization, wrong
for a model I want to behave. Denoising and sparse autoencoders do shape the code, but with a *chosen*
device — a hand-set corruption process, or a sparsity penalty with its own coefficient — that I would
have to justify and tune out of nowhere. Mean-field variational Bayes would give me a principled bound,
but its coordinate-ascent updates need analytic expectations under a conjugate family, and a deep decoder
is exactly the setting where no conjugate family exists, so that machinery does not run here at all. The
ELBO's `-D_KL(q||p)` is the regularizer all three of those are reaching for, except it is not bolted on:
it is the price the bound charges for using an approximate posterior, and it happens to be computable in
closed form for the Gaussian I am handed. So the negative ELBO is not one option among the baselines —
it is the one that turns the baselines' ad-hoc code-shaping into a consequence of doing maximum
likelihood properly.

The sign discipline bites later, so I pin it now. The *bound* I maximize contains `-D_KL`; the *loss*
I minimize is `-L`, so it must contain `+D_KL`. Maximize the negative-divergence, equivalently minimize
the positive-divergence — I will *add* the KL to the loss, never subtract it there. It is the kind of
sign error that trains a model that runs, looks plausible, and quietly optimizes the encoder *away*
from the prior, so I would rather be pedantic now than debug a diverging KL later.

The diagonal Gaussian I am handed gives the prior divergence in closed form, no Monte-Carlo noise. For
`q = N(mu, sigma^2 I)` and `p = N(0, I)` over latent dimension `J`, the negative-KL term in the bound is
`-D_KL(q||p) = 0.5 Σ_j (1 + log sigma_j^2 - mu_j^2 - sigma_j^2)`, which is never positive and reaches
zero exactly at `mu_j = 0, sigma_j = 1` — the prior — as it must. Let me actually check that degenerate
point rather than assert it: plug `mu_j = 0, sigma_j^2 = 1` into the positive form
`0.5 Σ_j (mu_j^2 + sigma_j^2 - 1 - log sigma_j^2)` and every term is `0.5·(0 + 1 - 1 - 0) = 0`, so the
divergence is exactly zero at the prior, and its gradient there is `∂/∂mu_j = mu_j = 0` and
`∂/∂sigma_j^2 = 0.5(1 - 1/sigma_j^2) = 0`, a genuine stationary minimum — the KL pulls `mu` toward 0
and `sigma^2` toward 1 and rests there. That positive form is *precisely* the per-sample quantity
`posterior.kl()` already returns. So the KL half of my objective is just `posterior.kl().mean()` —
averaged over the minibatch, added to the loss. I do not recompute it from `mu` and `logvar`; the
loop's bottleneck has done that for me.

Now the reconstruction term `log p_theta(x|z)`, where the decoder-likelihood choice is more consequential
than it looks. For real-valued pixels the natural choice is a fixed-variance Gaussian,
`p_theta(x|z) = N(x; f_theta(z), sigma_x^2 I)`. Then `log p_theta(x|z)` equals, up to an additive
constant, `-(1/(2 sigma_x^2)) Σ_d (x_d - f_theta(z)_d)^2` — so maximizing the Gaussian likelihood is
*exactly* minimizing squared (Euclidean, L2) reconstruction error. That is the "L2" the rung is named
for, and it is not a heuristic distance: it is the negative log-likelihood of a fixed-variance Gaussian
decoder. But there is a sibling I should take seriously here, because the Gaussian's quadratic tail is
the very thing that will hurt me. If I instead model the decoder likelihood as a **Laplace** distribution,
the negative log-likelihood becomes the *absolute* error, `Σ_d |x_d - f_theta(z)_d|` — an L1
reconstruction term. The Laplace has heavier tails, so it penalizes large residuals *less* than the
quadratic does, and the practical consequence is that L1 tends to produce **sharper** reconstructions.

Let me make that concrete rather than hand-wave it, because "sharper" is doing all the work. Take a
single pixel whose reconstruction target is uncertain — the decoder cannot resolve, from the code,
whether this pixel should be bright or dark, so under its conditional it is 1 with probability 0.7 and 0
with probability 0.3, a stand-in for the ambiguity at a texture edge. A squared-error objective is
minimized by the conditional *mean*, `0.7·1 + 0.3·0 = 0.7` — a washed-out grey that is neither of the
plausible values. An absolute-error objective is minimized by the conditional *median*, and since more
than half the mass sits at 1, the median is 1 — L1 commits to the dominant value. Check the costs: at
`c = 0.7`, `E|X-c| = 0.7·0.3 + 0.3·0.7 = 0.42`, while at `c = 1`, `E|X-c| = 0.7·0 + 0.3·1 = 0.30`, so
L1 genuinely prefers the crisp commitment (0.30 < 0.42), whereas L2 genuinely prefers the grey
(`E(X-c)^2` is 0.21 at `c=0.7` versus 0.30 at `c=1`). That is the whole blur story in one pixel: squared
error, by quadratically punishing the worst misses, drives the decoder toward the smooth conditional
mean that hedges across plausible high-frequency completions, while L1's linear penalty tolerates a few
large misses in exchange for committing to an edge. On a reconstruction task scored by a perceptual
statistic like rFID, that sharpness matters, so the principled default to *start* from is the Laplace/L1
form: same derivation, heavier-tailed decoder, sharper output. This is the choice the scaffold's
`l2-kl` fill actually makes — despite the slug it uses `F.l1_loss`, the Laplace reconstruction term, not
MSE — and the reasoning lands exactly there.

Where does `sigma_x^2` sit? It multiplies only the reconstruction term, not the KL, so it is precisely
the knob weighing reconstruction against the prior regularizer. For continuous data this knob is
*necessary*, not optional: without a scale the two terms are in incommensurable units. One subtlety the
implementation forces on me — the exact decoder NLL is a per-image *sum* over pixels, but the convenient
`F.l1_loss` (like `F.mse_loss`) *averages* over batch, channels, height, and width. That averaging
matters quantitatively, so let me carry the constants through instead of waving at them. The image is
`3×32×32`, so the pixel count is `D = 3·1024 = 3072`; the exact per-image Laplace NLL (dropping the
additive constant) is `(1/b) Σ_d |x_d - f_d| = (D/b)·rec_mean`, where `rec_mean` is what `F.l1_loss`
returns and `b` is the Laplace scale. The exact negative bound per image is therefore
`(D/b)·rec_mean + KL_sum`, and dividing the whole objective by the positive constant `(D/b)` — which
does not move the argmin — gives `rec_mean + (b/D)·KL_sum`. So the effective KL coefficient in the form
I actually code, `rec_mean + beta·KL`, is `beta = b/D`: the decoder scale divided by the pixel count.

That single identity tells me how to set `beta`. The mean reduction has already folded a factor of
`1/D = 1/3072 ≈ 3.3e-4` into the relationship between the per-pixel-mean reconstruction term and the
per-sample KL sum; a unit-scale Laplace decoder (`b = 1`) would put `beta ≈ 3.3e-4`. For a task whose
entire goal is reconstruction quality I want the KL *weaker* than that natural balance, not stronger:
its job here is hygiene — keep the code near the prior and well-behaved — not to compress hard. Setting
`beta = 1e-6` puts it a further `≈ 300×` below even the reduction-natural value (`1e-6 / 3.3e-4 ≈ 3e-3`),
which is exactly the "faint leash" I want. Let me sanity-check the magnitude against what the KL term
will actually contribute. The latent grid is `8×8` at compression `f=4`, and the latent channel count
scales across the three runs, so `J` is `64·4 = 256` (small), `64·8 = 512` (medium), `64·16 = 1024`
(large). A latent that is being *used* carries on the order of a nat per active dimension, so `KL_sum`
lands somewhere in the hundreds to `~1000`; multiplied by `1e-6` the KL contributes `~5e-4` to the loss,
against a reconstruction term of order `0.05–0.1`. That is well under one percent — the KL is present,
but it is not steering. To feel the other end of the knob, imagine the textbook `beta = 1`: then `1·KL`
of a few hundred would swamp `rec_mean ≈ 0.1` by three to four orders of magnitude, the optimizer would
drive every `mu_j → 0, sigma_j → 1` to kill that dominant term, the posterior would collapse to the
prior, and the decoder would receive an information-free code and paint the dataset mean — catastrophic
rFID. So the small `beta` is not timidity; it is the only setting consistent with a reconstruction-scored
objective, and `kl_weight = 1e-6` is the value the scaffold fixes.

There is a dynamical wrinkle worth noting, because it changes what the leash even does over training.
`AutoencoderKL` initializes its bottleneck near `mean ≈ 0` and `logvar ≈ 0` (so `sigma ≈ 1`), which by
the check above is exactly where `KL_sum ≈ 0`. So at step zero the loss is essentially pure
reconstruction; the KL only starts to bite as the encoder pushes `mu` and `sigma` off the prior to carry
information. That is the right causal order for this task — let reconstruction pull the code wherever it
needs to go, and let the tiny `beta` merely discourage the code from wandering to numerically silly
places (enormous `mu`, vanishing `sigma`) that would make the sampler ill-conditioned. The leash tightens
in proportion to how hard the encoder is straining against the prior, which is precisely when a little
tension is healthy and never when it would fight reconstruction at the start.

How many noise samples per datapoint? In principle the bound averages over draws of `eps`, but the loop
already does stochastic gradient descent over a minibatch of images, and each step averages over many
`(x, eps)` pairs. With a batch in the hundreds, a single reparameterized draw per image (`L = 1`) is
enough; the minibatch supplies the rest, and the extra variance from `L = 1` versus `L = 10` is
swamped by the variance the minibatch already carries. And critically, that one draw has *already
happened* — the loop sampled `z = mean + std·eps` inside `posterior.sample()` before producing `recon`.
So I must not resample inside the loss; I consume `recon` as given. Resampling here would draw a *second*
`eps`, decode nothing new (I have no decoder handle), and simply desynchronize the code from the
reconstruction I am scoring — a silent bug. The reparameterization is also what makes the gradient flow,
and it is worth being explicit about why this particular estimator is the one that scales. Because the
sample is `z = mu + sigma·eps` with `eps` fixed and external, `∂z/∂mu = 1` and `∂z/∂sigma = eps`, so
`∂L/∂mu = ∂L/∂z` and `∂L/∂sigma = (∂L/∂z)·eps` — the gradient rides the decoder's own backward pass
straight into the encoder, a low-variance pathwise derivative. The alternative, differentiating the
sampling distribution directly (the score-function / REINFORCE estimator), would weight `∂ log q/∂phi`
by the raw loss value and carry variance that grows with the loss magnitude — unusable for training a
deep decoder pixel-by-pixel. The closed-form KL is a differentiable function of `mean` and `logvar`, so
one backward pass on one scalar trains both networks jointly: one feedforward encode, one decode, one
backward pass, with the recognition network amortizing inference across all images. That is the thing
per-datapoint MCMC and wake-sleep could not deliver at scale.

So the floor is settled, and the literal edit is the small one: a reconstruction term that is the
Laplace-decoder NLL — `F.l1_loss(recon, target)`, the sharp sibling of MSE — plus the bottleneck's
closed-form KL averaged over the batch and weighted by a tiny `kl_weight = 1e-6`, added (never
subtracted), returning the scalar and a metrics dict reporting the two pieces. No discriminator, no
perceptual term, no frequency term — the bare negative ELBO with the sharper reconstruction likelihood.
I report `rec_loss` and `kl_loss` separately rather than only the total, and that is not cosmetic: given
the initialization dynamics above, I want to *see* whether the KL is even active over training or sitting
near zero, and whether the reconstruction term plateaus while rFID keeps mattering — the split metrics
are how the next rung reads which term is the bottleneck rather than guessing. (The distilled module is
in the answer.)

Now reason about what this floor must do, because that is the entire point of running it. The objective
is dominated by per-pixel L1, and per-pixel distances treat the image as a bag of independent pixels —
which is exactly where they lie about reconstruction quality. Even L1, sharper than L2, is minimized in
the presence of residual uncertainty by committing to *a* plausible value per pixel without any notion
that a slightly-shifted-but-equally-sharp texture is perceptually as good as the target while a washed
out one is worse. It scores energy of the residual, and energy of the residual is only loosely coupled
to what the FID feature extractor — and a human — actually reports. The two-point calculation above
shows L1 beats L2 at committing, but it also shows the ceiling: L1's commitment is per-pixel and
context-free; nothing in it knows that a coherent high-frequency *pattern* matters more than any single
pixel's value. So I expect this rung to learn a *competent but soft* reconstructor: PSNR and SSIM should
be respectable, because those reward pixel agreement and local-statistic agreement, the very things L1
optimizes; but rFID, which compares the *distributions* of deep features between originals and
reconstructions, should lag, because the reconstructions will carry the tell-tale softness of a
pixel-only objective and their feature statistics will drift from the originals'. I will not pretend to
know the gap in advance — I expect PSNR/SSIM to look decent while rFID lags; the rFID column will tell
me how badly.

I also expect a sharp split across the three scales, and the direction is predictable from the capacity
numbers alone. The only thing that changes across the three runs is width and latent channels, and the
latent spatial grid is fixed at `8×8` for all of them, so the difference is purely how many channels the
`f=4` bottleneck carries: `J = 256` for small, `512` for medium, `1024` for large. I can read those as
an information-bottleneck squeeze directly: the encoder must funnel the image's `D = 3072` real degrees
of freedom through `J` latent values, so the compression ratio `D/J` is `3072/256 = 12:1` for small,
`3072/512 = 6:1` for medium, and `3072/1024 = 3:1` for large. The small model has to discard four times
as many degrees of freedom as the large one keeps, and what gets discarded first, under a pixel loss, is
the low-energy high-frequency detail — the very content a bag-of-pixels objective barely rewards. So the
small model, with the tightest squeeze, has the least room to encode high-frequency detail and should
hedge hardest toward the blurry conditional mean — the `c = 0.7` grey rather than the crisp `1` — and
post the *worst* rFID by a wide margin. The large model, at a gentle `3:1` squeeze, can carry most of the
detail, so even a pixel-only loss should let it reconstruct fairly crisply and its rFID should be far
lower; the medium, at `6:1`, sits in between. Whatever the exact numbers, the diagnosis
is already pointed at the next rung: this is not a learning-rate or a capacity problem — I am not allowed
to touch capacity anyway — it is a *measurement* problem. The loss is optimizing the wrong distance, and
it is optimizing it hardest into the ground exactly where the model can least afford to waste capacity on
pixel-mean-matching. The fix is to stop measuring reconstruction error purely in pixel space and start
measuring it where it tracks perception — a perceptual, feature-space distance layered on this same
L1 + KL skeleton, which is the direction I turn to from here.

The causal chain in one breath: a deep latent model has an intractable marginal and posterior, so I
maximize the variational lower bound `L = E_q[log p(x|z)] - D_KL(q||p)`, whose negation is my loss; the
diagonal-Gaussian bottleneck gives the KL in closed form as `posterior.kl()`, which I add (checked to be
zero at the prior with a stationary minimum there) with a tiny `beta = 1e-6` — some 300× below the
reduction-natural `1/D` balance, a faint leash rather than a compressor, because a large `beta` would
collapse the posterior and paint the dataset mean; the reconstruction term is a decoder NLL, and I take
the **Laplace/L1** form rather than the Gaussian/L2 form because a worked two-point residual shows L1
commits to the median (a crisp edge) where L2 hedges to the mean (a blur); I consume the loop's single
reparameterized `recon` without resampling, and the reparameterization's low-variance pathwise gradient
trains encoder and decoder in one backward pass — landing the bare `F.l1_loss(recon, target) + 1e-6·kl`
floor, and expecting decent PSNR/SSIM but a lagging, scale-dependent rFID (worst where latent capacity
is smallest) that a perceptual term must rescue next.
