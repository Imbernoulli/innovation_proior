Let me start from what a lossy audio codec actually has to do, because the classical ones make a particular bet and I want to know whether a neural network can make a better one. Opus and EVS take the waveform, run a hand-designed transform, and throw away the components a psychoacoustic model says we won't hear, then entropy-code what's left. They're fast and mature. The bet I want to make instead: learn the transform end to end from data — an encoder that maps the waveform to a compact representation and a decoder that reconstructs it — and let the data tell me what to keep. The constraints are brutal, though: the low-latency 24 kHz path has to run faster than real time on a single CPU core, the system has to cover speech, noisy speech, and music, and the bitrate ladder has to range from very low rates up to high-fidelity settings. So the question is concrete: what encoder, what way of making the latents small, and what training objective gets me there?

I start with the backbone, because that part is the most settled. A fully convolutional symmetric encoder-decoder works well for audio: the encoder is a stack of strided convolutions that downsample the waveform and double the channel count at each step, the decoder mirrors it with transposed convolutions that upsample and halve the channels. Put an initial kernel-7 conv at the front, then four residual-unit-plus-downsample blocks; each residual unit uses two kernel-3 convolutions with a skip connection, and each downsample uses a strided convolution whose kernel is twice the stride. A two-layer LSTM over the latent sequence gives temporal modeling before the final kernel-7 projection to the latent dimension. With strides (2, 4, 5, 8) the total downsampling is 2·4·5·8 = 320, so at 24 kHz the encoder emits 24000/320 = 75 latent frames per second, and at 48 kHz it emits 150. That's the spine; nothing controversial yet. The decoder reverses the strides and puts its own two-layer LSTM on the latent side.

The hard part is making the latent small. The encoder outputs floating-point vectors — that's not a bitrate, that's just real numbers. To get a bitstream I have to quantize. The cleanest learned way is vector quantization: keep a codebook of, say, 1024 vectors, and replace each latent vector with the index of its nearest codebook entry. 1024 entries is 10 bits per frame, and at 75 frames per second that's 750 bits/s = 0.75 kbps per codebook. The trouble is the `argmin` over codebook entries is not differentiable, so I can't backprop through it to train the encoder. VQ-VAE solved this: use a straight-through estimator — in the backward pass, pretend the quantizer was the identity, copying the gradient at the decoder's input straight back to the encoder's output — and add a commitment loss that pulls the encoder output toward the codebook vector it selected, so the encoder doesn't wander away from its codes. Maintain the codebook with an exponential-moving-average update of the selected entries (decay 0.99) and re-seed any unused or nearly unused entry by sampling a vector from the current batch, so no codebook capacity is wasted on dead codes.

But one codebook of 1024 entries is only 10 bits per frame, 0.75 kbps. To hit 6 kbps I'd need a codebook of 2^80 entries — absurd; nearest-neighbor search and storage both explode, and the codebook can't be trained. So a single big codebook is the wrong shape for the rate I need. I need many bits per frame without an astronomically large codebook.

The way out is to quantize in stages around the residual. Quantize the latent with the first codebook; that leaves an error — the residual between the latent and its quantization. Quantize *that residual* with a second codebook. The residual of that with a third. After `N_q` stages, the quantized latent is the sum of the `N_q` chosen entries, and the bits per frame are `N_q · 10`. Each stage refines what the previous ones missed — it's coarse-to-fine. Eight stages gives 80 bits/frame = 6 kbps, sixteen gives 12 kbps, and so on, all with codebooks of only 1024 entries each. This is residual vector quantization, and it has a second gift I almost overlooked: because the stages are ordered by importance — the first codebook does the heavy lifting, each subsequent one contributes less — I can simply *stop early at inference*, keeping only the first `N_q` codebooks, and get a lower bitrate from the exact same trained model. So if I train with a variable number of stages, sampling `N_q` per batch (a multiple of 4, giving 1.5/3/6/12/24 kbps at 24 kHz), one model serves every target bitrate. That kills the "one model per bitrate" problem outright.

Let me write the RVQ forward loop carefully, because the residual bookkeeping is where sign errors hide. I keep a running `residual`, initialized to the input latent `z`, and a running sum `quantized_out`, initialized to zero. At each stage I quantize the current residual to get `quantized`, then I *subtract* it from the residual — `residual = residual - quantized` — so the next stage sees what's still unexplained, and I *add* it to the output — `quantized_out = quantized_out + quantized`. After `N_q` stages, `quantized_out` is the sum of all chosen entries and approximates `z`; `residual` is the final leftover error. Subtract on the residual, add on the output — those are the two operations and they must point opposite ways. The commitment loss sums, over the stages actually used, the squared distance between each stage's input residual `z_c` and its chosen entry `q_c(z_c)`, with no gradient flowing into the quantized value:

  l_w = Σ_{c=1}^{C} ‖ z_c − q_c(z_c) ‖₂²

Now the reconstruction objective. The obvious thing is an ℓ₂ on the waveform, but I know from every generative-audio result that pure ℓ₂ reconstruction gives blurry, dull, artifact-laden output — it averages over the perceptual detail. So I'll use a combination. A time-domain ℓ₁ on the raw waveform, `ℓ_t(x, x̂) = ‖x − x̂‖₁`, to keep the sample-level alignment honest. And a frequency-domain term that's where perception really lives. Compute mel-spectrograms at several time scales and match them; matching at multiple resolutions catches both fine temporal and fine spectral structure that a single STFT window would miss. Concretely, over scales `i ∈ e = {5, ..., 11}` (STFT window 2^i, hop 2^i/4), 64-bin mels, and scalar balance coefficients `a ∈ α`, average the paired ℓ₁ and ℓ₂ terms:

  ℓ_f(x, x̂) = (1/(|α|·|e|)) Σ_{a∈α} Σ_{i∈e} [ ‖ S_i(x) − S_i(x̂) ‖₁ + a ‖ S_i(x) − S_i(x̂) ‖₂ ]

and I'll just take α = {1} so the two terms enter equally. The multi-scale spectral loss is what makes the reconstruction perceptually sharp rather than merely close in energy.

Even that isn't enough for the last bit of realism, and the lesson from neural vocoders is unambiguous: add a discriminator as a learned perceptual loss. What kind? Audio structure lives in the time-frequency plane, so a discriminator that operates on the STFT is natural. Take the complex-valued STFT, concatenate real and imaginary parts, and run a 2D-conv network over it — a first 3x8 convolution with 32 channels, then convolutions with increasing dilation in time (1, 2, 4) and stride 2 over the frequency axis, then a final 3x3 convolution to the prediction. Different structures live at different frequency resolutions, so use several such discriminators at different STFT window lengths — [2048, 1024, 512, 256, 128] at 24 kHz, doubled at 48 kHz — and process stereo channels separately. Train this multi-scale STFT discriminator with the hinge loss, the stable standard choice: the discriminator wants `D(x)` large for real and `D(x̂)` small for fake, the generator wants `D(x̂)` large. So

  ℓ_g(x̂) = (1/K) Σ_k max(0, 1 − D_k(x̂))     [generator]
  L_d(x, x̂) = (1/K) Σ_k [ max(0, 1 − D_k(x)) + max(0, 1 + D_k(x̂)) ]   [discriminator]

The hinge is asymmetric in the right way: for the discriminator, a real example only contributes loss while `D(x) < 1` and a fake only while `D(x̂) > −1`, so once examples are confidently on the right side they stop pushing — exactly the margin behavior I want. And add a feature-matching term: match the discriminator's intermediate feature maps between real and reconstructed, which gives the generator a dense target. I'll make it *relative* — normalize each layer's L1 difference by the mean magnitude of that layer's real features — so layers with naturally large activations don't dominate:

  ℓ_feat(x, x̂) = (1/(K·L)) Σ_k Σ_l ‖ D_k^l(x) − D_k^l(x̂) ‖₁ / mean(‖ D_k^l(x) ‖₁)

So the generator's total loss is a weighted sum of five things: time-domain, frequency-domain, adversarial, feature-matching, and the RVQ commitment loss:

  L_G = λ_t·ℓ_t + λ_f·ℓ_f + λ_g·ℓ_g + λ_feat·ℓ_feat + λ_w·l_w

Now I hit a real wall, and it's about these λ's. The five terms have completely different and *time-varying* gradient magnitudes. The discriminator's gradient, in particular, swings wildly as the adversarial game progresses — early on it's huge, then it collapses, then it spikes again — while the mel-loss gradient is comparatively steady. If I just pick fixed λ's and backprop Σ λ_i g_i, then whenever the discriminator gradient blows up it drowns out the reconstruction terms and training destabilizes; and worse, the λ's aren't *interpretable* — λ_g = 3 doesn't tell me anything because I don't know the natural scale of g_g relative to g_f. Tuning them is guesswork that has to be redone whenever a term's scale shifts.

So let me decouple the *weight* I assign a loss from the *natural magnitude* of its gradient. For each loss `ℓ_i` that depends on the model only through the decoder output `x̂`, I can compute its gradient with respect to `x̂` directly: `g_i = ∂ℓ_i/∂x̂`. Track the exponential moving average of its norm, `⟨‖g_i‖₂⟩_β`. Then instead of using `λ_i g_i`, *renormalize* each gradient to unit scale and re-weight it by its fraction of the total weight, scaled to a reference norm `R`:

  g̃_i = R · (λ_i / Σ_j λ_j) · ( g_i / ⟨‖g_i‖₂⟩_β )

and backpropagate Σ_i g̃_i into the decoder. Look at what this buys. Dividing by the EMA of the gradient norm makes every loss contribute a gradient of the same characteristic scale regardless of its natural magnitude, so a discriminator spike no longer hijacks the update. And the weights become genuinely interpretable: if I normalize so Σ_j λ_j = 1, then λ_i is literally the fraction of the model's gradient that comes from loss `i`. I take R = 1 and β = 0.999. The one term this can't apply to is the commitment loss l_w, because it isn't a function of the decoder output `x̂` — it lives on the encoder/quantizer side — so it stays outside the balancer and keeps a fixed weight. With the balancer in place the weights are tame and meaningful: λ_t = 0.1, λ_f = 1, λ_g = 3, λ_feat = 3 at 24 kHz, with λ_g and λ_feat raised to 4 at 48 kHz.

A couple of details that fall out of the constraints. The discriminator tends to overpower the decoder, so I don't update it every step — update it with probability 2/3 at 24 kHz and every two batches at 48 kHz — to keep the game balanced. And the streaming requirement shapes the convolution padding: in the streamable setup I put *all* the padding before the current timestep (causal), and for a transposed conv with stride `s` I emit the first `s` output steps immediately and hold the remaining `s` in memory until the next frame arrives, so the model can output its first 320 samples (13 ms at 24 kHz) as soon as the first 320 samples come in. Layer normalization over the time dimension is ill-suited to a causal stream because it needs future context and is sensitive to spikes, so in the streaming model I use weight normalization on the model instead, keeping a mild normalization without breaking causality.

One more lever for bitrate. The RVQ codes are a discrete source, and they're not uniform — some codes are far more likely than others — so they have spare entropy I can squeeze with an entropy coder. Train a small Transformer language model over the codes: five layers, eight heads, 200 channels, feed-forward width 800, no dropout. At each timestep, embed the previous step's codes (one learned embedding table per codebook, summed) and predict the distribution of the current step's codes through `N_q` separate linear heads, each with as many outputs as the codebook cardinality. I predict the `N_q` codebooks at a timestep *in parallel* from the previous timestep — neglecting the within-timestep mutual information between codebooks — because the alternative, predicting one codebook per step serially, multiplies the sequence length by `N_q` and kills the real-time budget; the cross-entropy cost of ignoring that intra-step dependence is small. Feed the predicted probabilities to a range-based arithmetic coder. There's a sharp practical hazard: arithmetic coding needs the encoder and decoder to agree on the probabilities *bit for bit*, but the same model evaluated in batch mode versus real streaming mode can differ by more than 10⁻⁸ due to floating-point, which corrupts the decode. So round the probabilities to a precision of 10⁻⁶ before coding, with a fixed total range width of 2^24 and a minimum range of 2, to make the coder deterministic across evaluation modes.

Here is the structure in code — the convolutional backbone, the residual quantizer with its subtract-on-residual / add-on-output loop, the hinge and feature losses for the discriminator, and the balancer.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class VectorQuantization(nn.Module):
    def __init__(self, dim, codebook_size=1024, decay=0.99, eps=1e-5, dead_threshold=2.0):
        super().__init__()
        embed = torch.randn(codebook_size, dim)
        self.decay, self.eps = decay, eps
        self.codebook_size, self.dead_threshold = codebook_size, dead_threshold
        self.register_buffer("embed", embed)
        self.register_buffer("cluster_size", torch.zeros(codebook_size))
        self.register_buffer("embed_avg", embed.clone())

    def forward(self, x):                                      # x: [B, D, T]
        b, d, t = x.shape
        flat = x.transpose(1, 2).reshape(-1, d)                # [B*T, D]
        dist = (flat.pow(2).sum(1, keepdim=True)
                - 2 * flat @ self.embed.t()
                + self.embed.pow(2).sum(1))
        idx = dist.argmin(1)
        q = F.embedding(idx, self.embed).view(b, t, d).transpose(1, 2)

        if self.training:
            with torch.no_grad():
                one_hot = F.one_hot(idx, self.codebook_size).type_as(flat)
                cluster = one_hot.sum(0)
                embed_sum = one_hot.t() @ flat
                self.cluster_size.mul_(self.decay).add_(cluster, alpha=1 - self.decay)
                self.embed_avg.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)
                n = self.cluster_size.sum()
                smoothed = (self.cluster_size + self.eps) / (n + self.codebook_size * self.eps) * n
                self.embed.copy_(self.embed_avg / smoothed.unsqueeze(1))
                dead = self.cluster_size < self.dead_threshold
                if dead.any():
                    samples = flat[torch.randint(0, flat.size(0), (int(dead.sum().item()),), device=flat.device)]
                    self.embed[dead] = samples
                    self.embed_avg[dead] = samples
                    self.cluster_size[dead] = 1.0

        loss = F.mse_loss(q.detach(), x)                       # commitment: gradient only to encoder side
        q = x + (q - x).detach()                               # straight-through estimator
        return q, idx.view(b, t), loss

class ResidualVectorQuantization(nn.Module):
    def __init__(self, dim, n_q=32, codebook_size=1024):
        super().__init__()
        self.layers = nn.ModuleList([VectorQuantization(dim, codebook_size) for _ in range(n_q)])
    def forward(self, x, n_q=None):
        residual, quantized_out = x, x.new_zeros(x.shape)
        all_idx, commitment = [], x.new_zeros(())
        for layer in self.layers[:(n_q or len(self.layers))]:
            quantized, idx, loss = layer(residual)
            residual = residual - quantized                    # next stage refines the leftover
            quantized_out = quantized_out + quantized          # codes sum to the reconstruction
            all_idx.append(idx)
            commitment = commitment + loss
        return quantized_out, torch.stack(all_idx, dim=1), commitment

class PaddedConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, causal=False):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, stride=stride)
        total = kernel_size - stride
        self.left = total if causal else (total + 1) // 2
        self.right = 0 if causal else total // 2

    def forward(self, x):
        return self.conv(F.pad(x, (self.left, self.right)))

class PaddedConvTranspose1d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, causal=False):
        super().__init__()
        self.conv = nn.ConvTranspose1d(in_ch, out_ch, kernel_size=kernel_size, stride=stride)
        total = kernel_size - stride
        self.left = 0 if causal else (total + 1) // 2
        self.right = total if causal else total // 2

    def forward(self, x):
        y = self.conv(x)
        end = -self.right if self.right else None
        return y[..., self.left:end]

class ResidualUnit(nn.Module):
    def __init__(self, channels, causal=False):
        super().__init__()
        self.block = nn.Sequential(
            nn.ELU(), PaddedConv1d(channels, channels // 2, 3, causal=causal),
            nn.ELU(), PaddedConv1d(channels // 2, channels, 3, causal=causal))

    def forward(self, x):
        return x + self.block(x)

class Encoder(nn.Module):
    def __init__(self, channels=1, n_filters=32, strides=(2, 4, 5, 8), dimension=128, causal=False):
        super().__init__()
        layers, mult = [PaddedConv1d(channels, n_filters, 7, causal=causal)], 1
        for s in strides:
            layers += [ResidualUnit(mult * n_filters, causal), nn.ELU(),
                       PaddedConv1d(mult * n_filters, 2 * mult * n_filters, 2 * s, s, causal)]
            mult *= 2
        hidden = mult * n_filters
        self.down = nn.Sequential(*layers)
        self.lstm = nn.LSTM(hidden, hidden, num_layers=2, batch_first=True)
        self.proj = nn.Sequential(nn.ELU(), PaddedConv1d(hidden, dimension, 7, causal=causal))

    def forward(self, x):
        y = self.down(x)
        y, _ = self.lstm(y.transpose(1, 2))
        return self.proj(y.transpose(1, 2))                      # -> z [B, dimension, T]

class Decoder(nn.Module):
    def __init__(self, channels=1, n_filters=32, strides=(2, 4, 5, 8), dimension=128, causal=False):
        super().__init__()
        hidden = n_filters * (2 ** len(strides))
        self.in_proj = PaddedConv1d(dimension, hidden, 7, causal=causal)
        self.lstm = nn.LSTM(hidden, hidden, num_layers=2, batch_first=True)
        layers = []
        for s in reversed(strides):
            layers += [nn.ELU(), PaddedConvTranspose1d(hidden, hidden // 2, 2 * s, s, causal),
                       ResidualUnit(hidden // 2, causal)]
            hidden //= 2
        layers += [nn.ELU(), PaddedConv1d(n_filters, channels, 7, causal=causal)]
        self.up = nn.Sequential(*layers)

    def forward(self, z_q):
        y = self.in_proj(z_q)
        y, _ = self.lstm(y.transpose(1, 2))
        return self.up(y.transpose(1, 2))

def hinge_generator_loss(fake_logits):
    return sum(F.relu(1 - y).mean() for y in fake_logits) / len(fake_logits)

def hinge_discriminator_loss(real_logits, fake_logits):
    pairs = zip(real_logits, fake_logits)
    return sum(F.relu(1 - r).mean() + F.relu(1 + f).mean() for r, f in pairs) / len(real_logits)

def relative_feature_loss(real_feats, fake_feats, eps=1e-8):
    losses = []
    for real_layers, fake_layers in zip(real_feats, fake_feats):
        for r, f in zip(real_layers, fake_layers):
            losses.append((r - f).abs().mean() / (r.abs().mean() + eps))
    return sum(losses) / len(losses)

class Balancer:
    def __init__(self, weights, R=1.0, beta=0.999):
        self.weights, self.R, self.beta, self.ema = weights, R, beta, {}

    def backward(self, losses, x_hat):
        grads = {}
        for name, loss in losses.items():
            (g,) = torch.autograd.grad(loss, x_hat, retain_graph=True)
            norm = g.norm(dim=tuple(range(1, g.dim()))).mean().item()
            self.ema[name] = self.beta * self.ema.get(name, norm) + (1 - self.beta) * norm
            grads[name] = g / (self.ema[name] + 1e-12)
        total_w = sum(self.weights[k] for k in losses)
        balanced = sum(self.R * self.weights[k] / total_w * grads[k] for k in losses)
        x_hat.backward(balanced)                                  # commitment loss handled separately
```

The whole chain: a convolutional encoder-decoder gives the trainable transform; residual vector quantization makes the latent discrete at a competitive bitrate and, by dropping trailing codebooks, lets one model span the bitrate ladder; the reconstruction loss is time-domain ℓ₁ plus a multi-scale mel term, sharpened by a multi-scale STFT discriminator with hinge and relative-feature-matching losses; a gradient balancer renormalizes each loss by the EMA of its gradient norm so the weights become interpretable fractions and the discriminator's swings stop destabilizing training; and a small Transformer over the codes plus a deterministic range coder squeezes the remaining entropy, with the 24 kHz causal-padding path preserving low-latency faster-than-real-time streaming.
