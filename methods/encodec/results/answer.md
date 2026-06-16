# EnCodec, distilled

EnCodec is an end-to-end neural audio codec: a convolutional encoder-decoder with a residual-vector-quantized latent, trained with a reconstruction loss plus a multi-scale STFT discriminator, a gradient balancer that makes the loss weights interpretable, and an optional Transformer entropy model over the codes. The 24 kHz mono model spans 1.5-24 kbps with low-latency streaming and faster-than-real-time CPU inference; the 48 kHz stereo model spans 3-24 kbps in the high-fidelity fullband setting.

## The problem

Compress speech and music to a small discrete bitstream and reconstruct them with minimal perceptual distortion, end to end, streaming, across a range of bitrates and audio types, with hand-engineered classical codecs (Opus/EVS) as the quality and latency bar.

## Key ideas

**Convolutional encoder-decoder.** Encoder: initial kernel-7 conv, then `B=4` blocks of (one residual unit with two kernel-3 convolutions + strided downsampling conv with kernel = 2×stride), channels doubling per block, then a 2-layer LSTM, then a kernel-7 conv to the latent dimension. Strides (2, 4, 5, 8) give 320x downsampling, so 24 kHz audio becomes 75 latent frames/s and 48 kHz audio becomes 150 latent frames/s. Decoder mirrors it with transposed convolutions, reversed strides, and a 2-layer LSTM. Streaming uses all-causal padding and emits the first `s` output steps per transposed convolution while buffering the rest; weight norm replaces time-dimension layer norm in the streamable model.

**Residual vector quantization (RVQ).** Quantize the latent with codebook 1, take the residual, quantize it with codebook 2, and so on for `N_q` stages (up to 32 codebooks of 1024 entries = 10 bits each; 16 at 48 kHz). The code is the sum of chosen entries. Codebooks use EMA updates (decay 0.99) and dead-entry re-seeding; trained through quantization with a straight-through estimator + commitment loss. Sampling `N_q` (a multiple of 4) per batch and **dropping trailing codebooks at inference** lets one model serve 1.5/3/6/12/24 kbps.

```
RVQ:  residual = z ; out = 0
      for c in 1..N_q:  q = quantize_c(residual)
                        residual = residual − q      # refine leftover
                        out      = out + q           # codes sum to reconstruction
commitment:  l_w = Σ_c ‖ z_c − q_c(z_c) ‖₂²          # no grad to q_c
```

**Objective.** Reconstruction = time-domain ℓ₁ plus multi-scale mel (windows 2^i, i∈e={5..11}; hop 2^i/4; ℓ₁ + a·ℓ₂ with a∈α and α={1}). Perceptual = multi-scale STFT (MS-STFT) discriminator over complex STFT (windows [2048,1024,512,256,128], doubled at 48 kHz; stereo channels processed separately), trained with hinge loss + relative feature matching:

```
ℓ_t      = ‖x − x̂‖₁
ℓ_f      = (1/|α||e|) Σ_{a∈α} Σ_{i∈e} [‖S_i(x) − S_i(x̂)‖₁ + a ‖S_i(x) − S_i(x̂)‖₂]
ℓ_g(x̂)  = (1/K) Σ_k max(0, 1 − D_k(x̂))                                  # generator
L_d      = (1/K) Σ_k [max(0, 1 − D_k(x)) + max(0, 1 + D_k(x̂))]          # discriminator (hinge)
ℓ_feat   = (1/KL) Σ_k Σ_l ‖D_k^l(x) − D_k^l(x̂)‖₁ / mean(‖D_k^l(x)‖₁)     # relative feature matching
L_G      = λ_t ℓ_t + λ_f ℓ_f + λ_g ℓ_g + λ_feat ℓ_feat + λ_w l_w
```

Weights: λ_t=0.1, λ_f=1, λ_g=3, λ_feat=3 (24 kHz; λ_g=λ_feat=4 at 48 kHz), λ_w=1. The discriminator update probability is 2/3 at 24 kHz and 0.5 at 48 kHz to avoid overpowering the decoder.

**Loss balancer.** For each loss that depends on the model only via the output `x̂`, with `g_i = ∂ℓ_i/∂x̂` and `⟨‖g_i‖₂⟩_β` its EMA norm (β=0.999, R=1):

```
g̃_i = R · (λ_i / Σ_j λ_j) · g_i / ⟨‖g_i‖₂⟩_β        # backprop Σ_i g̃_i
```

This renormalizes every loss to a common gradient scale (so the discriminator's swings can't destabilize training) and makes each λ_i interpretable as the fraction of the model gradient from loss i (when Σλ=1). The commitment loss l_w is excluded (not a function of x̂).

**Entropy coding.** A small Transformer (5 layers, 8 heads, 200 channels, FFN 800, no dropout) predicts each timestep's `N_q` codebook distributions in parallel from the previous step (one embedding table and one linear head per codebook), feeding a deterministic range coder (probabilities rounded to 1e-6, range width 2^24, minimum range 2) for further lossless compression. With entropy coding, the 24 kHz path remains faster than real time; the 48 kHz fullband path trades speed for stereo fidelity.

**Training:** Adam, batch 64 × 1-second clips, lr 3×10⁻⁴, β₁=0.5, β₂=0.9, 300 epochs × 2,000 updates. ELU activations.

## Working code

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

        commitment = F.mse_loss(q.detach(), x)                 # gradient only to encoder input
        q = x + (q - x).detach()                               # straight-through estimator
        return q, idx.view(b, t), commitment

class ResidualVectorQuantization(nn.Module):
    def __init__(self, dim, n_q=32, codebook_size=1024):
        super().__init__()
        self.layers = nn.ModuleList([VectorQuantization(dim, codebook_size) for _ in range(n_q)])

    def forward(self, x, n_q=None):
        residual, out = x, x.new_zeros(x.shape)
        codes, commitment = [], x.new_zeros(())
        for layer in self.layers[:(n_q or len(self.layers))]:
            q, idx, loss = layer(residual)
            residual = residual - q                            # next codebook sees the leftover
            out = out + q                                      # decoder receives the sum
            codes.append(idx)
            commitment = commitment + loss
        return out, torch.stack(codes, dim=1), commitment       # codes: [B, N_q, T]

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
        return self.proj(y.transpose(1, 2))

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
        x_hat.backward(balanced)                              # commitment loss is backpropagated separately
```
