Lossy audio compression is fundamentally a trade-off between bitrate and the distortion that humans actually hear. Classical codecs such as Opus and EVS rely on hand-designed time-frequency transforms and psychoacoustic rules, which makes them fast and robust but also limits how much they can adapt to the full diversity of speech, noisy speech, and music. Neural approaches can in principle learn this transform end to end, but the obstacles are severe: the latent representation must be made genuinely discrete and compact, the system must support many target bitrates without retraining, the objective must match human perception rather than simple sample error, and the whole pipeline must stream in real time on a single CPU core.

The main reason earlier neural ideas fall short is that they either use a single vector-quantization codebook, which cannot reach competitive bitrates without an impossibly large codebook, or they sum fixed weighted losses whose gradients have wildly different scales. A single VQ codebook of 1024 entries gives only 10 bits per frame, far below what high-fidelity audio needs, and adversarial discriminator gradients can swing orders of magnitude, drowning out reconstruction terms and making training unstable.

The method I propose is EnCodec. It is an end-to-end neural audio codec built around a convolutional encoder-decoder, residual vector quantization, a multi-scale STFT discriminator, and a gradient balancer that renormalizes loss contributions so their weights become interpretable.

The encoder is a fully convolutional network that downsamples the waveform by a total factor of 320. It starts with a kernel-7 convolution, then four blocks each containing a residual unit with two kernel-3 convolutions followed by a strided downsampling convolution whose kernel size is twice the stride. The strides are 2, 4, 5, and 8, and channels double at each block. A two-layer LSTM over the resulting latent sequence provides temporal modeling, and a final kernel-7 convolution projects to a 128-dimensional latent. The decoder mirrors this with transposed convolutions and a two-layer LSTM. For streaming, all padding is placed before the current timestep so the model is causal, and transposed convolutions emit their first output steps immediately while buffering the rest.

The discrete bottleneck uses residual vector quantization. Instead of one large codebook, there are up to 32 codebooks of 1024 entries each. The first codebook quantizes the latent, the second quantizes the residual of that quantization, and so on. The quantized representation is the sum of the selected entries across all active codebooks. Because later codebooks contribute progressively less detail, trailing codebooks can simply be dropped at inference, which means one trained model can serve multiple bitrates. For 24 kHz audio the frame rate is 75 Hz, so 16 codebooks give 160 bits per frame or 12 kbps, and the full 32 codebooks give 24 kbps; fewer codebooks give 1.5, 3, or 6 kbps. Training samples the number of active codebooks per batch, so the model learns the whole ladder jointly.

The reconstruction objective combines a time-domain L1 loss with a multi-scale mel-spectrogram loss computed at several STFT window lengths, which captures structure at different time-frequency resolutions. To push perceptual realism further, a multi-scale STFT discriminator processes the complex spectrogram of real and reconstructed audio at several frequency resolutions and is trained with a hinge loss plus a relative feature-matching term. The discriminator is updated less frequently than the generator so it does not overpower the decoder early in training.

Because these loss terms have very different gradient magnitudes, EnCodec uses a balancer. For each loss that depends on the model only through the decoder output, it computes the gradient with respect to that output, tracks an exponential moving average of its norm, and renormalizes each gradient to a common scale before combining them with the desired weights. This makes each weight represent the actual fraction of the final gradient it contributes, and it prevents discriminator spikes from destabilizing training.

An optional lightweight Transformer entropy model predicts the distribution of the next frame's codebook entries from the previous frame, and a deterministic range coder exploits the remaining redundancy for further lossless compression. Probabilities are rounded to a fixed precision so the encoder and decoder agree bit-for-bit across batch and streaming modes.

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

    def forward(self, x):
        b, d, t = x.shape
        flat = x.transpose(1, 2).reshape(-1, d)
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

        commitment = F.mse_loss(q.detach(), x)
        q = x + (q - x).detach()
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
            residual = residual - q
            out = out + q
            codes.append(idx)
            commitment = commitment + loss
        return out, torch.stack(codes, dim=1), commitment

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
        x_hat.backward(balanced)
```
