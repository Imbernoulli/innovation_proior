Let me start from what a lossy audio codec actually has to do, because the classical ones make a particular bet and I want to know whether a neural network can make a better one. Opus and EVS take the waveform, run a hand-designed transform, and throw away the components a psychoacoustic model says we won't hear, then entropy-code what's left. They're fast and mature. The bet I want to make instead: learn the transform end to end from data — an encoder that maps the waveform to a compact representation and a decoder that reconstructs it — and let the data tell me what to keep. The constraints are brutal, though: it has to run faster than real time on a single CPU core, it has to be low-latency and streaming, and it has to sound good across speech, noisy speech, and music, at bitrates from 1.5 up to 24 kbps. So the question is concrete: what encoder, what way of making the latents small, and what training objective gets me there?

Start with the backbone, because that part is the most settled. A fully convolutional symmetric encoder–decoder works well for audio: the encoder is a stack of strided convolutions that downsample the waveform and double the channel count at each step, the decoder mirrors it with transposed convolutions that upsample and halve the channels. Put an initial kernel-7 conv at the front, a few residual-unit-plus-downsample blocks in the middle, a couple of LSTM layers over the latent sequence to model temporal structure, and a final kernel-7 conv to the latent dimension. With strides (2, 4, 5, 8) the total downsampling is 2·4·5·8 = 320, so at 24 kHz the encoder emits 24000/320 = 75 latent frames per second. That's the spine; nothing controversial yet. The decoder reverses the strides.

The hard part is making the latent small. The encoder outputs floating-point vectors — that's not a bitrate, that's just real numbers. To get a bitstream I have to quantize. The cleanest learned way is vector quantization: keep a codebook of, say, 1024 vectors, and replace each latent vector with the index of its nearest codebook entry. 1024 entries is 10 bits per frame, and at 75 frames per second that's 750 bits/s = 0.75 kbps per codebook. The trouble is the `argmin` over codebook entries is not differentiable, so I can't backprop through it to train the encoder. VQ-VAE solved this: use a straight-through estimator — in the backward pass, pretend the quantizer was the identity, copying the gradient at the decoder's input straight back to the encoder's output — and add a commitment loss that pulls the encoder output toward the codebook vector it selected, so the encoder doesn't wander away from its codes. Maintain the codebook with an exponential-moving-average update of the selected entries (decay around 0.99) and re-seed any entry that goes unused by sampling a vector from the current batch, so no codebook capacity is wasted on dead codes.

But one codebook of 1024 entries is only 10 bits per frame, 0.75 kbps. To hit 6 kbps I'd need a codebook of 2^80 entries — absurd; nearest-neighbor search and storage both explode, and the codebook can't be trained. So a single big codebook is the wrong shape for the rate I need. I need many bits per frame without an astronomically large codebook.

The way out is to quantize in stages around the residual. Quantize the latent with the first codebook; that leaves an error — the residual between the latent and its quantization. Quantize *that residual* with a second codebook. The residual of that with a third. After `N_q` stages, the quantized latent is the sum of the `N_q` chosen entries, and the bits per frame are `N_q · 10`. Each stage refines what the previous ones missed — it's coarse-to-fine. Eight stages gives 80 bits/frame = 6 kbps, sixteen gives 12 kbps, and so on, all with codebooks of only 1024 entries each. This is residual vector quantization, and it has a second gift I almost overlooked: because the stages are ordered by importance — the first codebook does the heavy lifting, each subsequent one contributes less — I can simply *stop early at inference*, keeping only the first `N_q` codebooks, and get a lower bitrate from the exact same trained model. So if I train with a variable number of stages, sampling `N_q` per batch (a multiple of 4, giving 1.5/3/6/12/24 kbps at 24 kHz), one model serves every target bitrate. That kills the "one model per bitrate" problem outright.

Let me write the RVQ forward loop carefully, because the residual bookkeeping is where sign errors hide. I keep a running `residual`, initialized to the input latent `z`, and a running sum `quantized_out`, initialized to zero. At each stage I quantize the current residual to get `quantized`, then I *subtract* it from the residual — `residual = residual - quantized` — so the next stage sees what's still unexplained, and I *add* it to the output — `quantized_out = quantized_out + quantized`. After `N_q` stages, `quantized_out` is the sum of all chosen entries and approximates `z`; `residual` is the final leftover error. Subtract on the residual, add on the output — those are the two operations and they must point opposite ways. The commitment loss sums, over the stages actually used, the squared distance between each stage's input residual `z_c` and its chosen entry `q_c(z_c)`, with no gradient flowing into the quantized value:

  l_w = Σ_{c=1}^{C} ‖ z_c − q_c(z_c) ‖₂²

Now the reconstruction objective. The obvious thing is an ℓ₂ on the waveform, but I know from every generative-audio result that pure ℓ₂ reconstruction gives blurry, dull, artifact-laden output — it averages over the perceptual detail. So I'll use a combination. A time-domain ℓ₁ on the raw waveform, `ℓ_t(x, x̂) = ‖x − x̂‖₁`, to keep the sample-level alignment honest. And a frequency-domain term that's where perception really lives. Compute mel-spectrograms at several time scales and match them; matching at multiple resolutions catches both fine temporal and fine spectral structure that a single STFT window would miss. Concretely, over scales `i ∈ {5, ..., 11}` (STFT window 2^i, hop 2^i/4), 64-bin mels, combine an ℓ₁ and an ℓ₂ term per scale:

  ℓ_f(x, x̂) = (1/(|α|·|s|)) Σ_i [ ‖ S_i(x) − S_i(x̂) ‖₁ + α_i ‖ S_i(x) − S_i(x̂) ‖₂ ]

and I'll just take α_i = 1 so the two terms enter equally. The multi-scale spectral loss is what makes the reconstruction perceptually sharp rather than merely close in energy.

Even that isn't enough for the last bit of realism, and the lesson from neural vocoders is unambiguous: add a discriminator as a learned perceptual loss. What kind? Audio structure lives in the time-frequency plane, so a discriminator that operates on the STFT is natural. Take the complex-valued STFT, concatenate real and imaginary parts, and run a 2D-conv network over it — a first conv, then convs with increasing dilation in time (1, 2, 4) and stride 2 over the frequency axis, then a final conv to the prediction. Different structures live at different frequency resolutions, so use several such discriminators at different STFT window lengths — [2048, 1024, 512, 256, 128] — a multi-scale STFT discriminator. Train it with the hinge loss, the stable standard choice: the discriminator wants `D(x)` large for real and `D(x̂)` small for fake, the generator wants `D(x̂)` large. So

  ℓ_g(x̂) = (1/K) Σ_k max(0, 1 − D_k(x̂))     [generator]
  L_d(x, x̂) = (1/K) Σ_k [ max(0, 1 − D_k(x)) + max(0, 1 + D_k(x̂)) ]   [discriminator]

The hinge is asymmetric in the right way: for the discriminator, a real example only contributes loss while `D(x) < 1` and a fake only while `D(x̂) > −1`, so once examples are confidently on the right side they stop pushing — exactly the margin behavior I want. And add a feature-matching term: match the discriminator's intermediate feature maps between real and reconstructed, which gives the generator a dense target. I'll make it *relative* — normalize each layer's L1 difference by the mean magnitude of that layer's real features — so layers with naturally large activations don't dominate:

  ℓ_feat(x, x̂) = (1/(K·L)) Σ_k Σ_l ‖ D_k^l(x) − D_k^l(x̂) ‖₁ / mean(‖ D_k^l(x) ‖₁)

So the generator's total loss is a weighted sum of five things: time-domain, frequency-domain, adversarial, feature-matching, and the RVQ commitment loss:

  L_G = λ_t·ℓ_t + λ_f·ℓ_f + λ_g·ℓ_g + λ_feat·ℓ_feat + λ_w·l_w

Now I hit a real wall, and it's about these λ's. The five terms have completely different and *time-varying* gradient magnitudes. The discriminator's gradient, in particular, swings wildly as the adversarial game progresses — early on it's huge, then it collapses, then it spikes again — while the mel-loss gradient is comparatively steady. If I just pick fixed λ's and backprop Σ λ_i g_i, then whenever the discriminator gradient blows up it drowns out the reconstruction terms and training destabilizes; and worse, the λ's aren't *interpretable* — λ_g = 3 doesn't tell me anything because I don't know the natural scale of g_g relative to g_f. Tuning them is guesswork that has to be redone whenever a term's scale shifts.

So let me decouple the *weight* I assign a loss from the *natural magnitude* of its gradient. For each loss `ℓ_i` that depends on the model only through the decoder output `x̂`, I can compute its gradient with respect to `x̂` directly: `g_i = ∂ℓ_i/∂x̂`. Track the exponential moving average of its norm, `⟨‖g_i‖₂⟩_β`. Then instead of using `λ_i g_i`, *renormalize* each gradient to unit scale and re-weight it by its fraction of the total weight, scaled to a reference norm `R`:

  g̃_i = R · (λ_i / Σ_j λ_j) · ( g_i / ⟨‖g_i‖₂⟩_β )

and backpropagate Σ_i g̃_i into the decoder. Look at what this buys. Dividing by the EMA of the gradient norm makes every loss contribute a gradient of the same characteristic scale regardless of its natural magnitude, so a discriminator spike no longer hijacks the update. And the weights become genuinely interpretable: if I normalize so Σ_j λ_j = 1, then λ_i is literally the fraction of the model's gradient that comes from loss `i`. I take R = 1 and β = 0.999. The one term this can't apply to is the commitment loss l_w, because it isn't a function of the decoder output `x̂` — it lives on the encoder/quantizer side — so it stays outside the balancer and keeps a fixed weight. With the balancer in place the weights are tame and meaningful: λ_t = 0.1, λ_f = 1, λ_g = 3, λ_feat = 3 at 24 kHz.

A couple of details that fall out of the constraints. The discriminator tends to overpower the decoder, so I don't update it every step — update it with probability 2/3 at 24 kHz — to keep the game balanced. And the streaming requirement shapes the convolution padding: in the streamable setup I put *all* the padding before the current timestep (causal), and for a transposed conv with stride `s` I emit the first `s` output steps immediately and hold the remaining `s` in memory until the next frame arrives, so the model can output its first 320 samples (13 ms at 24 kHz) as soon as the first 320 samples come in. Layer normalization over the time dimension is ill-suited to a causal stream (it's sensitive to spikes), so in the streaming model I use weight normalization on the model instead, keeping a mild normalization that helps the objective metrics.

One more lever for bitrate. The RVQ codes are a discrete source, and they're not uniform — some codes are far more likely than others — so they have spare entropy I can squeeze with an entropy coder. Train a small Transformer language model over the codes: at each timestep, embed the previous step's codes (one learned embedding table per codebook, summed) and predict the distribution of the current step's codes through `N_q` separate linear heads, each with as many outputs as the codebook cardinality. I predict the `N_q` codebooks at a timestep *in parallel* from the previous timestep — neglecting the within-timestep mutual information between codebooks — because the alternative, predicting one codebook per step serially, multiplies the sequence length by `N_q` and kills the real-time budget; the cross-entropy cost of ignoring that intra-step dependence is small. Feed the predicted probabilities to a range-based arithmetic coder. There's a sharp practical hazard: arithmetic coding needs the encoder and decoder to agree on the probabilities *bit for bit*, but the same model evaluated in batch mode versus real streaming mode can differ by more than 10⁻⁸ due to floating-point, which corrupts the decode. So round the probabilities to a precision of 10⁻⁶ before coding, with a fixed total range width of 2^24 and a minimum range, to make the coder deterministic across evaluation modes.

Here is the structure in code — the convolutional backbone, the residual quantizer with its subtract-on-residual / add-on-output loop, the multi-scale STFT discriminator's place, and the balancer.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Residual Vector Quantization -------------------------------------------
class VectorQuantization(nn.Module):
    def __init__(self, dim, codebook_size=1024, decay=0.99):
        super().__init__()
        self.codebook = nn.Parameter(torch.randn(codebook_size, dim))  # EMA-updated in practice
    def forward(self, x):                      # x: [B, T, D]
        # nearest-entry lookup (argmin); straight-through gradient at the call site
        d = (x.pow(2).sum(-1, keepdim=True)
             - 2 * x @ self.codebook.t()
             + self.codebook.pow(2).sum(-1))
        idx = d.argmin(-1)
        q = F.embedding(idx, self.codebook)
        loss = F.mse_loss(q.detach(), x)       # commitment: gradient only to encoder side
        q = x + (q - x).detach()               # straight-through estimator
        return q, idx, loss

class ResidualVectorQuantization(nn.Module):
    def __init__(self, dim, n_q=32, codebook_size=1024):
        super().__init__()
        self.layers = nn.ModuleList([VectorQuantization(dim, codebook_size) for _ in range(n_q)])
    def forward(self, x, n_q=None):
        quantized_out = 0.0
        residual = x
        all_idx, all_loss = [], []
        n_q = n_q or len(self.layers)          # drop trailing codebooks -> lower bitrate
        for layer in self.layers[:n_q]:
            quantized, idx, loss = layer(residual)
            residual = residual - quantized    # next stage refines the leftover
            quantized_out = quantized_out + quantized   # codes sum to the reconstruction
            all_idx.append(idx); all_loss.append(loss)
        return quantized_out, torch.stack(all_idx), torch.stack(all_loss)

# --- Convolutional encoder / decoder (SEANet-style backbone) ----------------
class ResidualUnit(nn.Module):
    def __init__(self, dim, dilation):
        super().__init__()
        self.block = nn.Sequential(
            nn.ELU(), nn.Conv1d(dim, dim // 2, 3, padding=dilation, dilation=dilation),
            nn.ELU(), nn.Conv1d(dim // 2, dim, 1))
    def forward(self, x):
        return x + self.block(x)

class Encoder(nn.Module):
    def __init__(self, n_filters=32, strides=(2, 4, 5, 8), dimension=128):
        super().__init__()
        layers = [nn.Conv1d(1, n_filters, 7, padding=3)]
        mult = 1
        for s in strides:
            layers += [ResidualUnit(mult * n_filters, dilation=1), nn.ELU(),
                       nn.Conv1d(mult * n_filters, mult * 2 * n_filters, 2 * s, stride=s,
                                 padding=(2 * s - s) // 2)]
            mult *= 2                            # double channels on downsample
        layers += [nn.LSTM(mult * n_filters, mult * n_filters, 2)[0] if False else nn.Identity(),
                   nn.ELU(), nn.Conv1d(mult * n_filters, dimension, 7, padding=3)]
        self.model = nn.Sequential(*layers)
    def forward(self, x):
        return self.model(x)                     # -> z [B, dimension, T]

# (Decoder mirrors Encoder with ConvTranspose1d and reversed strides, halving channels.)

# --- Loss balancer ----------------------------------------------------------
class Balancer:
    def __init__(self, weights, R=1.0, beta=0.999):
        self.weights = weights                   # {name: lambda_i}, e.g. {'t':0.1,'f':1,'g':3,'feat':3}
        self.R, self.beta = R, beta
        self.ema = {}
    def backward(self, losses, x_hat):
        grads, norms = {}, {}
        for name, loss in losses.items():
            g, = torch.autograd.grad(loss, x_hat, retain_graph=True)
            grads[name] = g
            n = g.norm(p=2)
            self.ema[name] = self.beta * self.ema.get(name, n.item()) + (1 - self.beta) * n.item()
            norms[name] = self.ema[name]
        total_w = sum(self.weights[k] for k in losses)
        out = sum(self.R * self.weights[k] / total_w * grads[k] / (norms[k] + 1e-12)
                  for k in losses)               # g~_i, renormalized & re-weighted
        x_hat.backward(out)                       # commitment loss handled separately (not a fn of x_hat)
```

The whole chain: a convolutional encoder–decoder gives the trainable transform; residual vector quantization makes the latent discrete at a competitive bitrate and, by dropping trailing codebooks, lets one model span every target bitrate; the reconstruction loss is time-domain ℓ₁ plus a multi-scale mel term, sharpened by a multi-scale STFT discriminator with hinge and relative-feature-matching losses; a gradient balancer renormalizes each loss by the EMA of its gradient norm so the weights become interpretable fractions and the discriminator's swings stop destabilizing training; and a small Transformer over the codes plus a deterministic range coder squeezes the remaining entropy for free, all under a causal-padding scheme that keeps it streaming and faster than real time on one CPU core.
