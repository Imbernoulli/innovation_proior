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
(its E-step *is* the posterior) and the conjugate mean-field machinery. So I replace the true posterior
by a tractable approximation `q_phi(z|x)` and optimize a bound. For any `q` I can write
`log p_theta(x) = L(x) + D_KL(q_phi(z|x) || p_theta(z|x))`, where
`L(x) = E_{q}[log p_theta(x,z) - log q_phi(z|x)]`; since a KL is non-negative, `L(x) ≤ log p_theta(x)`
is a lower bound on the evidence, and the slack is exactly that intractable posterior KL — which I am
happy to leave implicit, because maximizing `L` both raises the marginal likelihood I care about and
tightens `q` toward the true posterior at once. Peeling out the prior gives the form I want to stare
at: `L(x) = E_{q}[log p_theta(x|z)] - D_KL(q_phi(z|x) || p(z))`. Structurally that *is* a regularized
autoencoder — an expected reconstruction log-likelihood minus a divergence pulling the encoder's output
toward the prior — and the regularizer is not an ad-hoc penalty I bolted on the way denoising or sparse
autoencoders need one; it falls out of the probability. That is already a better story than a plain
autoencoder, which is known not to learn a well-behaved code from reconstruction alone.

The sign discipline bites later, so I pin it now. The *bound* I maximize contains `-D_KL`; the *loss*
I minimize is `-L`, so it must contain `+D_KL`. Maximize the negative-divergence, equivalently minimize
the positive-divergence — I will *add* the KL to the loss, never subtract it there.

The diagonal Gaussian I am handed gives the prior divergence in closed form, no Monte-Carlo noise. For
`q = N(mu, sigma^2 I)` and `p = N(0, I)` over latent dimension `J`, the negative-KL term in the bound is
`-D_KL(q||p) = 0.5 Σ_j (1 + log sigma_j^2 - mu_j^2 - sigma_j^2)`, which is never positive and reaches
zero exactly at `mu_j = 0, sigma_j = 1` — the prior — as it must. Its negation is the positive KL I add
to the loss, `D_KL(q||p) = 0.5 Σ_j (mu_j^2 + sigma_j^2 - 1 - log sigma_j^2)`, and that is *precisely*
the per-sample quantity `posterior.kl()` already returns. So the KL half of my objective is just
`posterior.kl().mean()` — averaged over the minibatch, added to the loss. I do not recompute it from
`mu` and `logvar`; the loop's bottleneck has done that for me.

Now the reconstruction term `log p_theta(x|z)`, where the decoder-likelihood choice is more consequential
than it looks. For real-valued pixels the natural choice is a fixed-variance Gaussian,
`p_theta(x|z) = N(x; f_theta(z), sigma_x^2 I)`. Then `log p_theta(x|z)` equals, up to an additive
constant, `-(1/(2 sigma_x^2)) Σ_d (x_d - f_theta(z)_d)^2` — so maximizing the Gaussian likelihood is
*exactly* minimizing squared (Euclidean, L2) reconstruction error. That is the "L2" the rung is named
for, and it is not a heuristic distance: it is the negative log-likelihood of a fixed-variance Gaussian
decoder. But there is a sibling I should take seriously here. If I instead model the decoder likelihood
as a **Laplace** distribution rather than a Gaussian, the negative log-likelihood becomes the
*absolute* error, `Σ_d |x_d - f_theta(z)_d|` — an L1 reconstruction term. The Laplace has heavier tails,
so it penalizes large residuals *less* than the quadratic does, and the practical consequence is that
L1 tends to produce **sharper** reconstructions: squared error, by quadratically punishing the worst
pixels, drives the decoder toward the smooth conditional mean that hedges across plausible
high-frequency completions — a blur — whereas L1's linear penalty tolerates a few large misses in
exchange for committing to crisp edges. On a reconstruction task scored by a perceptual statistic like
rFID, that sharpness matters, so the principled default to *start* from is the Laplace/L1 form: same
derivation, heavier-tailed decoder, sharper output. This is the choice the scaffold's `l2-kl` fill
actually makes — despite the slug it uses `F.l1_loss`, the Laplace reconstruction term, not MSE — and
the reasoning lands exactly there.

Where does `sigma_x^2` sit? It multiplies only the reconstruction term, not the KL, so it is precisely
the knob weighing reconstruction against the prior regularizer. For continuous data this knob is
*necessary*, not optional: without a scale the two terms are in incommensurable units. One subtlety the
implementation forces on me — the exact decoder NLL is a per-image *sum* over pixels, but the convenient
`F.l1_loss` (like `F.mse_loss`) *averages* over batch, channels, height, and width. That averaging
rescales the reconstruction term relative to the per-image-sum form by `1/D` (with `D` the pixel count),
so the version I actually code, `mean_pixel_recon + beta · mean_batch_KL`, equals the exact negative
bound up to one positive global factor when `beta` absorbs both `sigma_x^2` and the `1/D` reduction.
Practically, then, `beta` (the `kl_weight`) is the *effective* KL coefficient under the chosen
reduction, not the raw decoder variance — and for a task whose entire goal is reconstruction quality I
want it *small*. The KL's job here is hygiene: keep the code near the prior and well-behaved, not to
compress hard. So `beta` on the order of `1e-6` is right — the objective is overwhelmingly
reconstruction-driven, with the KL present only as a faint leash. (Push `beta` up and I trade
reconstruction for a tidier latent, which is the wrong trade when rFID is what I am scored on.) That is
exactly the `kl_weight = 1e-6` the scaffold fixes.

How many noise samples per datapoint? In principle the bound averages over draws of `eps`, but the loop
already does stochastic gradient descent over a minibatch of images, and each step averages over many
`(x, eps)` pairs. With a batch in the hundreds, a single reparameterized draw per image (`L = 1`) is
enough; the minibatch supplies the rest. And critically, that one draw has *already happened* — the
loop sampled `z = mean + std·eps` inside `posterior.sample()` before producing `recon`. So I must not
resample inside the loss; I consume `recon` as given. The reparameterization is also what makes the
gradient flow: because the sample was a smooth function of `mean`, `std`, and a fixed `eps`, backprop
through the decoder carries `∂ log p_theta/∂z` back through the sampled latent into the encoder, and the
closed-form KL is a differentiable function of `mean` and `logvar`, so one backward pass on one scalar
trains both networks jointly. That is the thing per-datapoint MCMC and wake-sleep could not deliver at
scale: no inner sampling loop, just one feedforward encode, one decode, one backward pass, with the
recognition network amortizing inference across all images.

So the floor is settled, and the literal edit is the small one: a reconstruction term that is the
Laplace-decoder NLL — `F.l1_loss(recon, target)`, the sharp sibling of MSE — plus the bottleneck's
closed-form KL averaged over the batch and weighted by a tiny `kl_weight = 1e-6`, added (never
subtracted), returning the scalar and a metrics dict reporting the two pieces. No discriminator, no
perceptual term, no frequency term — the bare negative ELBO with the sharper reconstruction likelihood.
(The distilled module is in the answer.)

Now reason about what this floor must do, because that is the entire point of running it. The objective
is dominated by per-pixel L1, and per-pixel distances treat the image as a bag of independent pixels —
which is exactly where they lie about reconstruction quality. Even L1, sharper than L2, is minimized in
the presence of residual uncertainty by committing to *a* plausible value per pixel without any notion
that a slightly-shifted-but-equally-sharp texture is perceptually as good as the target while a washed
out one is worse. It scores energy of the residual, and energy of the residual is only loosely coupled
to what the FID feature extractor — and a human — actually reports. So I expect this rung to learn a
*competent but soft* reconstructor: PSNR and SSIM should be respectable, because those reward pixel
agreement and local-statistic agreement, the very things L1 optimizes; but rFID, which compares the
*distributions* of deep features between originals and reconstructions, should lag, because the
reconstructions will carry the tell-tale softness of a pixel-only objective and their feature
statistics will drift from the originals'.

I also expect a sharp split across the three scales, and the direction is predictable. The small model
(narrowest channels, only 4 latent channels) has the least capacity to encode high-frequency detail
through the `f=4` bottleneck, so under a pixel-only loss it should hedge hardest toward the blurry mean
and post the *worst* rFID by a wide margin. The large model (widest channels, 16 latent channels) has
the most room to carry detail, so even a pixel-only loss should let it reconstruct fairly crisply and
its rFID should be far lower. The medium scale should sit in between. Whatever the exact numbers, the
diagnosis is already pointed at the next rung: this is not a learning-rate or a capacity problem, it is
a *measurement* problem — the loss is optimizing the wrong distance. The fix is to stop measuring
reconstruction error purely in pixel space and start measuring it where it tracks perception, which is
the move the next rung makes by adding a learned perceptual term on top of this same L1 + KL skeleton.

The causal chain in one breath: a deep latent model has an intractable marginal and posterior, so I
maximize the variational lower bound `L = E_q[log p(x|z)] - D_KL(q||p)`, whose negation is my loss; the
diagonal-Gaussian bottleneck gives the KL in closed form as `posterior.kl()`, which I add with a tiny
`beta = 1e-6` because the task is reconstruction-driven; the reconstruction term is a decoder NLL, and I
take the **Laplace/L1** form rather than the Gaussian/L2 form because its heavier tails yield sharper
output; I consume the loop's single reparameterized `recon` without resampling, so one backward pass
trains encoder and decoder jointly — landing the bare `F.l1_loss(recon, target) + 1e-6 · kl` floor, and
expecting decent PSNR/SSIM but a lagging, scale-dependent rFID that a perceptual term must rescue next.
