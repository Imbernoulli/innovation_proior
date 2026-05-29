OK, let me think this through from scratch. I want high-resolution images of complex natural scenes, and I want the good behavior of likelihood-based generative models — broad coverage of the data, stable training, no mode collapse — without the brutal compute bill that the best of them are currently running up. The model class that's winning on quality right now is the denoising diffusion model. It beats GANs on sample quality, it covers the distribution instead of collapsing onto a few modes, and it's flexible: you can take an unconditional model and inpaint or colorize or guide it at test time. So I don't want to abandon it. I want to make it cheap.

But how expensive is it, really, and *why*? Let me actually stare at the cost. A diffusion model defines a fixed forward process that slowly turns an image into Gaussian noise, and it learns to reverse that, one denoising step at a time. Sampling is a long sequential walk back from noise — tens to a thousand steps — and every single step is a full forward pass of a big UNet over the entire pixel grid. Training is worse: every step needs a forward *and* backward pass, over the full-resolution RGB tensor, summed over the whole noise schedule. The numbers people are quoting for the strongest pixel-space models are on the order of hundreds of GPU-days to train, and days just to draw fifty thousand samples. That's not a model most people can train. So the question I actually care about isn't "can I make a slightly faster sampler" — people are already chipping at the step count with cleverer samplers, but every step is still a full pixel-space pass, and training cost is untouched by that. The question is: what is all this compute being *spent on*?

Likelihood-based models are mode-covering. They have to put probability mass on essentially the whole dataset, which means they spend modeling capacity in proportion to the actual information content of the signal. And what is the information content of a natural image, bit for bit? It's dominated by high-frequency texture — the exact pixel values of grass, skin pores, fabric weave — almost all of which is imperceptible. This is the same fact that pushed earlier autoregressive image models toward discretized likelihoods: the model burns enormous effort nailing down detail that no human would notice. So a pixel-space diffusion model, being likelihood-based, is doing the same thing. It's spending capacity on imperceptible detail.

Let me make that sharper by imagining the rate/distortion curve of a diffusion model that's already trained. As I add bits (as I let the model spend more capacity / denoise more carefully), distortion drops. But the *shape* of that drop has two regimes. Early on, a large chunk of the bits buys a big perceptual improvement — but it's all the high-frequency stuff; the semantic content, the layout, what's even in the picture, barely moves. The model is doing perceptual compression: throwing away imperceptible detail. Only after that, in a second regime, do the remaining bits go toward the actual semantics — which objects, how arranged, the conceptual composition. So the costly sequential model is paying full pixel-space price during a first phase whose entire job is to discard detail that doesn't matter. That's the waste. And the reweighted training objective people already use — the one that undersamples the easy, nearly-clean denoising steps — is gesturing at this, trying to focus effort on the perceptually meaningful steps. But it doesn't change the fact that the network is *evaluated in pixel space* at every step regardless. The compute per step is fixed by the resolution.

So the leverage is obvious once I see it this way. The dominant cost is set by how many spatial positions the expensive model has to chew through per layer — roughly quadratic in resolution for the convolutions, worse for any global attention. If the first regime is nearly semantics-free and it's eating most of the budget, then I should just not make the diffusion model do it. Hand the perceptual compression off to something cheap and one-time, and let the diffusion model — the part that's slow because it's sequential — run somewhere small, where every position it processes already carries semantic content.

What would "somewhere small" be? A lower-dimensional space that is *perceptually equivalent* to the image but has the imperceptible detail already squeezed out. That's an autoencoder. Train an encoder E that downsamples an image x of size H×W into a compact z of size h×w with h=H/f, w=W/f for some factor f, and a decoder D that reconstructs x from z. Train it once. Freeze it. Then run the entire diffusion process on z instead of x. The diffusion model keeps its exact form — same UNet, same noise schedule, same objective — it just sees z.

Let me check that this actually buys what I think it buys, quantitatively, before I get attached to it. One UNet pass costs work proportional to the number of spatial positions it processes. Downsample each side by f and you have h·w = HW/f² positions instead of HW. Convolutional work per layer drops by f². Any attention layer drops by f⁴ in its dominant term, because attention is quadratic in the number of positions. Multiply that per-step saving across all T sampling steps, and across the forward+backward of every training step, and across the whole dataset. For f=4 that's a 16× reduction in the convolutional work per step; for f=8, 64×. That is exactly the order of magnitude I need — turning hundreds of GPU-days into a single-GPU job — and it's a structural saving, not a sampler tweak. Good. The idea has legs. Let me now poke at every place it could go wrong.

First wall: doesn't "compress the image, then model the compressed thing with a generator" already exist? Yes — and I should be honest about why those approaches didn't already solve this, or I'll just reinvent their failure. The line is VQ-VAE, then VQ-VAE-2, then VQGAN, then the big text-to-image autoregressive models. The recipe there: learn a discrete latent with an autoencoder, then model the prior over the discrete codes with an autoregressive transformer that generates them one token at a time. The decomposition is exactly mine — compress, then model the compressed thing. So why didn't they get cheap high-fidelity diffusion? Because of the *second* stage. An autoregressive transformer's cost is quadratic in the sequence length, and it generates tokens one at a time. To make that tractable you have to keep the token sequence short, which means you have to compress *hard* — something like 16× per side. And the moment you compress that hard, your autoencoder has to throw away real detail to hit the rate, so reconstructions degrade and you've put a low ceiling on final quality. On top of that, the transformer needs billions of parameters to model the prior, and to feed a 2D grid of codes into a sequence model you flatten it into a 1D raster order, which throws away the spatial structure of the latent entirely. So in that world the compression level isn't chosen to preserve quality; it's dictated by what the transformer can stomach.

That diagnosis tells me exactly what to change. The villain is the autoregressive transformer in stage two. It's the thing that's quadratic in sequence length and forces brutal compression and ignores 2D structure. Replace it. With what? With the very thing I'm trying to make cheap: a convolutional diffusion model. A diffusion UNet is convolutional — its cost scales gently in the spatial size of its input, not quadratically in a flattened sequence length, and it natively respects the 2D layout because it *is* a 2D conv net. So if the prior over the latent is modeled by a conv diffusion model instead of an AR transformer, I'm no longer forced into extreme compression. I'm free to pick a *mild* compression — f=4 or f=8 — where the autoencoder still reconstructs almost perfectly, and let the diffusion model handle the rest. The whole tension in the older two-stage work dissolves: they compressed hard because their stage-two model demanded it; I can compress gently because mine doesn't.

Second wall: should I learn the autoencoder and the generative prior *together*? There's an approach that does — learn the encoder/decoder and a score-based prior over the latent jointly, end to end. It sounds appealing: the latent gets shaped to be easy to model. But think about what that costs me. If I'm training reconstruction and the generative prior at the same time, I have to weight the two losses against each other, and that balance is delicate — push toward prior-fit and reconstructions blur; push toward reconstruction and the prior is left modeling a messy space. Worse, the latent space is a *moving target*: the diffusion model is chasing a distribution that's still shifting under it as the encoder updates. I don't want any of that. The cleaner move is to fully decouple: train the autoencoder first, freeze it, and then train the diffusion model in a *fixed* latent space. Now there's no reconstruction-vs-prior weighting at all — the autoencoder is judged purely on reconstruction, so I get faithful reconstructions, and the diffusion model is judged purely on modeling a fixed distribution. As a bonus, one universal autoencoder can be reused across many different diffusion-model trainings and tasks — train it once, amortize it forever. That bonus is real: the expensive-to-explore part (many diffusion models for many tasks) now sits on top of a shared, cheap-to-reuse base.

So the architecture is settling into two stages, decoupled. Let me design each properly, starting with the autoencoder, because everything downstream depends on the latent it produces.

What objective do I train the autoencoder on? The naive choice is pixel L2 or L1 between x and D(E(x)). But I know what that gives me: blur. An L2 reconstruction loss is minimized by predicting the conditional mean of the pixel, and when there's any uncertainty about fine detail the mean is a smeared average — it literally optimizes toward spatial averaging. The whole point of this autoencoder is to be *perceptually equivalent* to the image, and blur is the opposite of that. So I need losses that care about perceptual quality and realism, not per-pixel error. Two ingredients fix this. A perceptual loss — compare deep features of x and the reconstruction rather than raw pixels, so the objective rewards matching the things a vision network finds salient (this is the LPIPS idea). And a patch-based adversarial loss — a PatchGAN discriminator that tries to tell real image patches from reconstructed ones, which forces the reconstruction to *look* like a real image locally, keeping it on the natural-image manifold instead of drifting to a blurry mean. Together: reconstructions that are sharp and plausible, not averaged. This is the same first-stage recipe the VQGAN line used, and it works for the same reason — I'm just going to pair it with a diffusion second stage instead of a transformer.

There's a balancing subtlety in combining the reconstruction term and the adversarial term: their gradient magnitudes can be wildly different and shift during training, so one can swamp the other. I'll set the adversarial weight adaptively — scale it by the ratio of the gradient norm of the reconstruction (NLL) term to the gradient norm of the adversarial term, both measured at the last decoder layer. That keeps the two forces comparable automatically: d_weight = ‖∇ L_rec‖ / (‖∇ L_adv‖ + ε), clamped to a sane range. No hand-tuned schedule.

Now, the latent itself. If I just let E produce an arbitrary z, the latent can drift to arbitrarily large scale and high variance, which makes it a nightmare for the diffusion model to model later. I need to regularize the latent toward something well-behaved — zero-centered, modest variance — but only *lightly*, because heavy regularization trades away reconstruction fidelity, and reconstruction fidelity is the hard ceiling on everything. Two natural regularizers fit, and I'll keep both as options because they have different characters.

One: make the encoder output the parameters of a diagonal Gaussian — a mean and a log-variance per latent location — and put a small KL penalty pulling that posterior toward a standard normal, exactly like a VAE. So E(x) gives (μ, σ), z is sampled as z = μ + σ·ε with ε ∼ N(0,1), and the regularizer is KL(N(μ,σ²) ‖ N(0,1)). For a diagonal Gaussian against a unit normal that KL has the closed form, per latent component, ½(μ² + σ² − 1 − log σ²); summed over the latent and pulled in with a tiny weight — something like 1e-6 — it's just enough to keep z centered and from blowing up, far too weak to hurt reconstruction. This is the KL-regularized variant; the latent is continuous.

Two: regularize by discretizing. Put a vector-quantization layer in the middle — learn a codebook of vectors and snap each latent location to its nearest codebook entry. With a large enough codebook this is also a mild regularizer (lots of capacity, little distortion). And here's a clean way to wire it: rather than treat the quantized codes as the latent the diffusion model sees, I take z *before* the quantization step and fold the quantization operation into the decoder — it becomes effectively the first layer of D. Then this variant is just a VQGAN whose quantizer has been absorbed into the decoder. This is the VQ-regularized variant; the latent the diffusion model works on is continuous (pre-quantization).

Either way the rule is the same: regularize as little as I can get away with — tiny KL weight, or a high-capacity codebook — so reconstructions stay near-perfect. The reason I can afford to barely regularize at all is the decision I already made: the prior is modeled by a powerful diffusion model, not by a weak prior or a sequence model that needs a tidy discrete latent. The diffusion model can handle a not-especially-tidy latent, so I don't have to pay for tidiness in reconstruction quality.

Good — stage one is a perceptual autoencoder, trained with L1 + LPIPS + adaptive PatchGAN, lightly regularized (KL or VQ), then frozen. Now stage two: diffusion in this latent space. Let me actually re-derive the diffusion objective so I know exactly what I'm running on z, and so I understand which knob is doing the "focus on the perceptual bits" work.

The forward process is fixed. Pick signal and noise schedules α_t and σ_t and define, from a clean sample x_0,

q(x_t | x_0) = N(x_t | α_t x_0, σ_t² I),

so x_t = α_t x_0 + σ_t ε with ε ∼ N(0,1). The signal-to-noise ratio at step t is SNR(t) = α_t²/σ_t². It has the Markov structure for s<t: q(x_t|x_s) = N(α_{t|s} x_s, σ_{t|s}² I) with α_{t|s} = α_t/α_s and, so the marginals stay consistent, σ_{t|s}² = σ_t² − α_{t|s}² σ_s². The generative model reverses this with the same Markov shape, p(x_0) = ∫ p(x_T) ∏_t p(x_{t-1}|x_t), and p(x_T) is a standard normal.

To train, write the evidence lower bound. Negative log-likelihood is bounded by

−log p(x_0) ≤ KL(q(x_T|x_0) ‖ p(x_T)) + Σ_t E_{q(x_t|x_0)} KL(q(x_{t-1}|x_t,x_0) ‖ p(x_{t-1}|x_t)).

The first term only depends on the final SNR(T) and is essentially constant if the schedule drives SNR(T) to zero, so the work is in the sum. Now I need to parameterize p(x_{t-1}|x_t). I know the *true* posterior q(x_{t-1}|x_t,x_0) in closed form — it's Gaussian — but it depends on the unknown x_0. So I let the network produce an estimate x_θ(x_t,t) of x_0 and define the reverse step as the true posterior with x_0 replaced by that estimate:

p(x_{t-1}|x_t) := q(x_{t-1}|x_t, x_θ(x_t,t)) = N(x_{t-1} | μ_θ(x_t,t), σ_{t|t-1}² σ_{t-1}²/σ_t² · I),

with the posterior mean coming out as

μ_θ(x_t,t) = (α_{t|t-1} σ_{t-1}² / σ_t²) x_t + (α_{t-1} σ_{t|t-1}² / σ_t²) x_θ(x_t,t).

Because both the true posterior and the model posterior are Gaussians with the same variance, each KL in the sum reduces to a scaled squared distance between their means, which — after the algebra of substituting μ_θ — is a scaled squared error between the true x_0 and the estimate x_θ. Carrying the constants through, the sum becomes

Σ_t E_{ε} ½ (SNR(t−1) − SNR(t)) ‖ x_0 − x_θ(α_t x_0 + σ_t ε, t) ‖².

So each step is a denoising regression — predict the clean signal from a noised version — weighted by the *drop in SNR* across that step. That weight is the crux of where capacity goes, so I'll hold onto it.

I still need to reparameterize the target from "predict x_0" to "predict the noise". Define ε_θ(x_t,t) = (x_t − α_t x_θ(x_t,t))/σ_t. Since x_t = α_t x_0 + σ_t ε, the injected noise is ε = (x_t − α_t x_0)/σ_t, so subtracting gives ε − ε_θ = (α_t/σ_t)(x_θ − x_0), and squaring,

‖ x_0 − x_θ(α_t x_0 + σ_t ε, t) ‖² = (σ_t²/α_t²) ‖ ε − ε_θ(α_t x_0 + σ_t ε, t) ‖².

Predicting the noise is the more stable target and it's what ties this to denoising score matching — ε_θ is, up to scale, an estimate of the score of the noised data.

The weighting is the other issue. In clean-signal space the ELBO weight on step t is ½(SNR(t−1)−SNR(t)); after I switch to the noise target, that weight is multiplied by the σ_t²/α_t² conversion factor. Either way it is a messy, t-dependent weight. But the strongest models throw it away and weight every step equally. Why is that the *right* thing rather than a cheat? Because of the perceptual/semantic split I started from. The high-SNR steps (t small, x_t nearly clean) are the easy, perceptual-detail steps; the principled weight actually emphasizes them, which spends capacity nailing imperceptible detail. Setting all weights equal de-emphasizes those trivially-denoisable steps relative to the harder mid-SNR steps where the semantic structure is actually decided. So the equal-weight, noise-prediction objective is precisely the lever that focuses the model on the perceptually relevant bits — the same instinct as the rate/distortion observation, now baked into the loss. Equal weights collapse everything to

L = E_{x, ε∼N(0,1), t} ‖ ε − ε_θ(x_t, t) ‖²,

t uniform over the steps. That's the whole training objective for a pixel-space model.

Now I just move it into the latent. Everything above holds verbatim with z in place of x. Because the forward process is fixed and has a closed form, I don't need to run any chain during training — I encode once, z_0 = E(x), then sample a single step directly, z_t = √(ᾱ_t) z_0 + √(1−ᾱ_t) ε (I'll write the cumulative signal as ᾱ_t so this matches the usual schedule buffers), and regress. The latent objective is

L_LDM = E_{E(x), ε∼N(0,1), t} ‖ ε − ε_θ(z_t, t) ‖².

The backbone ε_θ is a time-conditional UNet — and this is where keeping the 2D latent pays off again: the UNet is built mostly from 2D convolutions, so it uses the spatial structure of z that the old autoregressive approaches flattened away, with self-attention only at the coarse, low-resolution levels to keep cost bounded. The timestep enters through an embedding that scales and shifts the residual-block features. To sample, I draw z_T ∼ N(0,1), run the reverse steps, and decode the result with a single pass through D to get an image. One decode at the end — not per step — because D is cheap and the sequential cost was the latent-space part.

There's a scale subtlety I should not skip, because it bit me the moment I thought about generating bigger images. The diffusion forward process mixes signal and noise according to SNR(t) = α_t²/σ_t², and that's calibrated assuming the data has roughly unit variance. But my latent's variance is whatever the autoencoder happened to produce. If I use a KL latent whose variance is large, then relative to the noise schedule the effective SNR is high everywhere, and the model ends up committing semantic detail too early in the reverse process; when I then try to sample convolutionally at resolutions beyond what I trained on, this miscalibration shows up as degraded samples. The fix is simple: rescale the latent to unit variance. Estimate one empirical standard deviation σ̂ from the first batch of latents and divide: z ← z/σ̂, undoing the scale at decode time. (The VQ latent already has variance near 1, so it needs no rescaling.) In code this is a single scalar scale_factor applied right after encoding and inverted before decoding.

So far I have unconditional generation in latent space. But most of what I want — text-to-image, class-conditional, layout, super-resolution — is conditional, p(z|y). The clean cases first: when the conditioning y is *spatially aligned* with the image — a low-resolution version for super-resolution, a segmentation map, a mask for inpainting — there's a trivial answer. Downsample y to the latent grid and just concatenate it to the UNet's input along the channel axis. The conditioning lives on the same grid as z, so concatenation is the natural, almost-free injection; the "conditioning encoder" is the identity. That handles the dense, image-like conditions and even lets me run the model convolutionally to synthesize images larger than training resolution.

The hard case is conditioning that has *no* spatial correspondence to the image grid at all: a text prompt, a class label, a sequence of bounding boxes. I cannot concatenate a sentence to a feature map — there's no spatial alignment to exploit, and the modalities are different. I need a mechanism where each spatial location of the image features can pull information from anywhere in a conditioning representation, regardless of that representation's shape or modality. That is exactly what attention does. So: pre-process y with a domain-specific encoder τ_θ into a sequence τ_θ(y) ∈ R^{M×d_τ} — for text, an unmasked transformer over tokenized input (token plus positional embeddings, then blocks of self-attention and position-wise MLP); for a class label, just a single learnable embedding vector; for layout, encode each box as a tuple of discretized corner positions and class. Then inside the UNet, at several feature levels, take the flattened intermediate feature map φ_i(z_t) ∈ R^{N×d_ε^i} as the query side and the conditioning sequence as the key/value side, and cross-attend:

Attention(Q,K,V) = softmax(QKᵀ/√d) · V,  Q = W_Q^{(i)} φ_i(z_t),  K = W_K^{(i)} τ_θ(y),  V = W_V^{(i)} τ_θ(y),

with learnable projections W_Q, W_K, W_V at each level. Every image position attends over the whole conditioning sequence; the mechanism is blind to what modality y is, so the same machinery serves text, class, and layout. The 1/√d scaling on the logits is not decoration — the dot products Q·K grow in magnitude with the projection dimension d, and without rescaling they'd push the softmax into a saturated regime where one entry dominates and gradients through the others vanish; dividing by √d keeps the logits O(1) as d grows. Architecturally I drop this cross-attention layer into the UNet's transformer blocks alongside the existing self-attention and a position-wise MLP — and I notice that if I removed the cross-attention and MLP, the block is exactly the self-attention UNet I'd use unconditioned, so conditioning is a clean additive extension, not a redesign.

The conditional objective just carries y through:

L = E_{E(x), y, ε∼N(0,1), t} ‖ ε − ε_θ(z_t, t, τ_θ(y)) ‖²,

and I optimize τ_θ and ε_θ jointly — the conditioning encoder learns its representation in service of the denoising task, end to end with the UNet, while the autoencoder stays frozen underneath.

One more capability comes almost for free, and it's why staying in the diffusion family mattered. Diffusion models can be steered at test time without retraining. The classifier-guidance formula for an ε-parameterized model with fixed variance corrects the predicted noise by the gradient of a noise-aware classifier:

ε̂ ← ε_θ(z_t,t) + √(1−α_t²) · ∇_{z_t} log p_Φ(y|z_t).

I can read this more generally. Interpret the "classifier" as any differentiable guider on a decoded estimate of the clean latent: p_Φ(y | T(D(z_0(z_t)))), where T is any differentiable transform — identity, a downsampling, whatever the task needs. A Gaussian guider with fixed variance, log p_Φ = −½‖y − T(D(z_0(z_t)))‖², turns guidance into an L2 regression toward a target image — an upsampling/translation knob — and swapping the L2 for a perceptual (LPIPS) distance gives perceptual guidance for super-resolution. And because all of this happens in latent space, even a noise-aware classifier used for guidance is cheap to train. Separately, when I want to push conditional fidelity, classifier-free guidance applies directly: train ε_θ with the conditioning randomly dropped so it learns both ε_θ(z,c) and an unconditional ε_θ(z,∅), then at sampling extrapolate, ε̂ = ε_θ(z,c) + s·(ε_θ(z,c) − ε_θ(z,∅)), with s>1 trading diversity for fidelity — no separate classifier needed.

Let me trace the whole thing once more to be sure it closes. The waste in pixel-space diffusion is that the expensive sequential model spends most of its compute in a perceptual-compression regime that barely touches semantics. So I peel that regime off into a train-once autoencoder — perceptual + adversarial losses for sharp, on-manifold reconstructions, lightly regularized (KL or VQ) so the latent is well-behaved without sacrificing fidelity — and freeze it. I run an ordinary diffusion model, with the equal-weight noise-prediction objective that itself focuses capacity on the perceptually relevant steps, entirely inside that small latent space, on a 2D-convolutional UNet that exploits the latent's spatial structure (which the old AR-transformer two-stage models threw away by flattening and over-compressing). I rescale the latent to unit variance so the noise schedule is calibrated. For conditioning, spatially-aligned inputs concatenate to the UNet; arbitrary modalities enter through cross-attention from a domain-specific encoder, blind to modality. And diffusion's test-time guidance carries over, now cheap because it lives in latent space. The compute drops by roughly f² per step from leaving pixel space, the reconstruction stays faithful because I compress only mildly, and the quality and flexibility of diffusion are intact. Now the code.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ---------- Stage 1: perceptual autoencoder (train once, then freeze) ----------

class DiagonalGaussian:
    """Encoder output as a per-location diagonal Gaussian (KL variant)."""
    def __init__(self, parameters):
        self.mean, logvar = torch.chunk(parameters, 2, dim=1)   # split channels
        self.logvar = torch.clamp(logvar, -30.0, 20.0)
        self.var = torch.exp(self.logvar)
        self.std = torch.exp(0.5 * self.logvar)

    def sample(self):                       # z = mu + sigma * eps
        return self.mean + self.std * torch.randn_like(self.mean)

    def mode(self):
        return self.mean

    def kl(self):                           # KL( N(mu,sigma^2) || N(0,1) )
        return 0.5 * torch.sum(self.mean**2 + self.var - 1.0 - self.logvar,
                               dim=[1, 2, 3])


class AutoencoderKL(nn.Module):
    """Downsamples x by factor f to a low-dim, perceptually equivalent latent."""
    def __init__(self, encoder, decoder, z_channels, embed_dim):
        super().__init__()
        self.encoder = encoder              # conv encoder, downsamples by f
        self.decoder = decoder              # conv decoder, upsamples by f
        # encoder emits 2*z_channels (mean+logvar); project to 2*embed_dim
        self.quant_conv = nn.Conv2d(2 * z_channels, 2 * embed_dim, 1)
        self.post_quant_conv = nn.Conv2d(embed_dim, z_channels, 1)

    def encode(self, x):
        return DiagonalGaussian(self.quant_conv(self.encoder(x)))

    def decode(self, z):
        return self.decoder(self.post_quant_conv(z))

    def forward(self, x, sample_posterior=True):
        posterior = self.encode(x)
        z = posterior.sample() if sample_posterior else posterior.mode()
        return self.decode(z), posterior


class LPIPSWithDiscriminator(nn.Module):
    """L1 + perceptual (LPIPS) + adaptively-weighted PatchGAN; tiny KL."""
    def __init__(self, perceptual, discriminator, kl_weight=1e-6, perceptual_weight=1.0):
        super().__init__()
        self.perceptual = perceptual                 # LPIPS feature distance
        self.discriminator = discriminator           # PatchGAN
        self.kl_weight = kl_weight                   # mild: keep latent well-behaved
        self.perceptual_weight = perceptual_weight
        self.logvar = nn.Parameter(torch.zeros(()))  # learned output log-variance

    def adaptive_weight(self, rec_loss, g_loss, last_layer):
        # balance reconstruction vs adversarial gradients at the last decoder layer
        rec_grad = torch.autograd.grad(rec_loss, last_layer, retain_graph=True)[0]
        g_grad = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        w = torch.norm(rec_grad) / (torch.norm(g_grad) + 1e-4)
        return torch.clamp(w, 0.0, 1e4).detach()

    def forward(self, x, x_rec, posterior, last_layer):
        rec = torch.abs(x - x_rec)                                   # L1, not L2 (no blur)
        p_loss = self.perceptual(x, x_rec)
        while p_loss.ndim < rec.ndim:
            p_loss = p_loss[..., None]
        rec = rec + self.perceptual_weight * p_loss                  # stay on manifold
        nll = (rec / torch.exp(self.logvar) + self.logvar).mean()
        g_loss = -self.discriminator(x_rec).mean()                  # fool the patch critic
        d_w = self.adaptive_weight(nll, g_loss, last_layer)
        kl = posterior.kl().mean()
        return nll + d_w * g_loss + self.kl_weight * kl


# ---------- Conditioning: cross-attention for arbitrary modalities ----------

class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64):
        super().__init__()
        inner = dim_head * heads
        context_dim = context_dim or query_dim
        self.heads = heads
        self.scale = dim_head ** -0.5                 # 1/sqrt(d): keep logits O(1)
        self.to_q = nn.Linear(query_dim, inner, bias=False)    # Q from image feats
        self.to_k = nn.Linear(context_dim, inner, bias=False)  # K from tau(y)
        self.to_v = nn.Linear(context_dim, inner, bias=False)  # V from tau(y)
        self.to_out = nn.Linear(inner, query_dim)

    def forward(self, x, context=None):
        context = context if context is not None else x   # None -> self-attention
        q, k, v = self.to_q(x), self.to_k(context), self.to_v(context)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=self.heads),
                      (q, k, v))
        sim = torch.einsum('b i d, b j d -> b i j', q, k) * self.scale
        attn = sim.softmax(dim=-1)                     # softmax(QK^T/sqrt(d))
        out = torch.einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=self.heads)
        return self.to_out(out)


# ---------- Stage 2: diffusion in the (frozen) latent space ----------

class LatentDiffusion(nn.Module):
    def __init__(self, autoencoder, unet, cond_encoder=None,
                 timesteps=1000, scale_factor=1.0, conditioning='crossattn'):
        super().__init__()
        self.first_stage = autoencoder.eval()         # frozen perceptual compressor
        for p in self.first_stage.parameters():
            p.requires_grad_(False)
        self.unet = unet                              # time-conditional UNet (eps_theta)
        self.cond_encoder = cond_encoder              # tau_theta (None for unconditional)
        self.conditioning = conditioning              # 'crossattn' or 'concat'
        self.scale_factor = scale_factor              # unit-variance latent (SNR calib.)

        betas = torch.linspace(1e-4, 2e-2, timesteps)
        acp = torch.cumprod(1.0 - betas, dim=0)       # alpha-bar_t (cumulative signal)
        self.num_timesteps = timesteps
        self.register_buffer('sqrt_acp', torch.sqrt(acp))
        self.register_buffer('sqrt_one_minus_acp', torch.sqrt(1.0 - acp))

    def _extract(self, buffer, t, x):
        return buffer[t].view(-1, *([1] * (x.ndim - 1)))

    @torch.no_grad()
    def encode_to_latent(self, x):
        z = self.first_stage.encode(x).sample()       # z = E(x), then rescale
        return self.scale_factor * z                  # -> unit variance

    @torch.no_grad()
    def decode_latent(self, z):
        return self.first_stage.decode(z / self.scale_factor)  # single decode at the end

    def q_sample(self, z0, t, noise):
        # closed-form forward: z_t = sqrt(acp_t) z0 + sqrt(1-acp_t) eps
        a = self._extract(self.sqrt_acp, t, z0)
        b = self._extract(self.sqrt_one_minus_acp, t, z0)
        return a * z0 + b * noise

    def get_conditioning(self, y):
        if self.cond_encoder is None or y is None:
            return y
        return self.cond_encoder(y)

    def apply_model(self, z_t, t, cond):
        # spatial conditions concatenate; arbitrary modalities cross-attend
        if self.conditioning == 'concat' and cond is not None:
            return self.unet(torch.cat([z_t, cond], dim=1), t)
        return self.unet(z_t, t, context=cond)        # context feeds CrossAttention

    def forward(self, x, y=None):
        z0 = self.encode_to_latent(x)
        t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=x.device)
        noise = torch.randn_like(z0)
        z_t = self.q_sample(z0, t, noise)
        cond = self.get_conditioning(y)
        eps = self.apply_model(z_t, t, cond)
        return F.mse_loss(eps, noise)                 # L = || eps - eps_theta(z_t,t,tau(y)) ||^2

    @torch.no_grad()
    def sample(self, shape, y=None, device=None):
        # Deterministic DDIM-style update for the eps-parameterized model.
        device = device or self.sqrt_acp.device
        z = torch.randn(tuple(shape), device=device)
        cond = self.get_conditioning(y)
        for i in reversed(range(self.num_timesteps)):
            t = torch.full((z.shape[0],), i, device=device, dtype=torch.long)
            eps = self.apply_model(z, t, cond)
            a = self._extract(self.sqrt_acp, t, z)
            b = self._extract(self.sqrt_one_minus_acp, t, z)
            z0 = (z - b * eps) / a
            if i > 0:
                t_prev = torch.full((z.shape[0],), i - 1, device=device, dtype=torch.long)
                ap = self._extract(self.sqrt_acp, t_prev, z)
                bp = self._extract(self.sqrt_one_minus_acp, t_prev, z)
                z = ap * z0 + bp * eps
            else:
                z = z0
        return self.decode_latent(z)
```

The chain, end to end: pixel-space diffusion wastes its sequential budget on a near-semantics-free perceptual-compression regime, so I move that regime into a frozen, perceptual+adversarial autoencoder and run an ordinary equal-weight noise-prediction diffusion model in its small, 2D-structured latent — cutting per-step cost by about f² while compressing only mildly enough to keep reconstructions faithful — and I condition it by concatenation when the input is spatially aligned and by modality-agnostic cross-attention when it isn't, inheriting diffusion's cheap latent-space test-time guidance along the way.
