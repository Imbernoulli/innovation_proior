# WaveNet

## Problem

Generate raw audio waveforms directly — the sequence of amplitude samples at wideband rates (≥16 kHz) — as a tractable, fully probabilistic autoregressive model trained by maximum likelihood, with no vocoder or hand-built signal model in between. The central tension: a single audio sample needs thousands of samples of past context to be predicted well, yet sequences are 10^4–10^5 samples per second, so the model must reach a very large receptive field while remaining cheap to train.

## Key idea

Model the waveform with the chain rule, `p(x) = Π_t p(x_t | x_1,…,x_{t-1})`, and compute each conditional with a stack of **dilated causal convolutions**. Causal convolutions enforce the autoregressive ordering and allow all timesteps to be trained in parallel (unlike an RNN's sequential BPTT). Dilating the convolutions with a dilation that **doubles each layer** (1, 2, 4, …, 512, then repeated) makes the receptive field grow **exponentially with depth at linear cost**. The per-sample output is a 256-way categorical softmax over **μ-law-companded** 8-bit values. Each layer uses a **gated activation** and contributes both a **residual** and a **skip** connection. Conditioning information (speaker identity, linguistic features) is injected into the gates to turn `p(x)` into a controllable `p(x|h)`.

## Final method

**Output.** μ-law companding to 8 bits, then a 256-way categorical:
`f(x) = sign(x)·ln(1+μ|x|)/ln(1+μ)`, `μ = 255`, quantized to 256 levels. The log warp puts fine resolution at small amplitudes (where hearing is most sensitive), so 256 levels reconstruct speech near-perfectly; a categorical softmax makes no assumption about the distribution's shape (beats a Gaussian / mixture-density output).

**Causal convolutions.** Each output at `t` depends only on inputs at `≤ t` (left-pad by `(k-1)·dilation`, drop the tail). Training is one parallel forward pass over the ground-truth waveform; generation is sequential (sample fed back in).

**Dilated stack and receptive field.** For a stack of causal layers of filter width `k` with dilations `d_1,…,d_L`:

`RF = (k − 1)·Σ_ℓ d_ℓ + 1`.

With `k = 2` and a doubling block `1,2,4,…,512` (10 layers), `Σ d_ℓ = 2^10 − 1 = 1023`, so `RF = 1024` from 10 layers — exponential reach for linear depth/compute. Repeat the block `M` times to extend the field further (and restart at dilation 1 so fine structure is re-modeled at each block).

**Gated activation.** `z = tanh(W_f * x) ⊙ σ(W_g * x)`. The sigmoid is a learned, input-conditioned multiplicative gate on the tanh content; beats ReLU for audio.

**Residual + skip.** Each layer: gated unit → 1×1 conv added back to the trunk (residual, trainable depth) and a separate 1×1 conv to a skip output. All skips are summed, then `ReLU → 1×1 → ReLU → 1×1 → softmax`.

**Conditioning.** `p(x|h) = Π_t p(x_t | x_{<t}, h)`.
- Global (e.g. speaker id): `z = tanh(W_f*x + V_f^T h) ⊙ σ(W_g*x + V_g^T h)`, with `V^T h` broadcast over time.
- Local (e.g. linguistic features at a lower rate): upsample `h` to audio rate with a transposed (learned) convolution to get `y`, then `z = tanh(W_f*x + V_f*y) ⊙ σ(W_g*x + V_g*y)` with `V*y` a 1×1 conv. (Repeat-upsampling works slightly worse than the learned transposed conv.)

**Context stacks (optional).** A separate, smaller stack over a long span locally-conditions a larger main stack over a short span; it can use fewer units and pooling for the longest timescales.

## Code

```python
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

QUANT = 256  # 8-bit after mu-law companding


def mu_law_encode(audio, mu=QUANT - 1):
    audio = np.clip(audio, -1.0, 1.0)
    magnitude = np.log1p(mu * np.abs(audio)) / np.log1p(mu)
    signal = np.sign(audio) * magnitude
    return ((signal + 1) / 2 * mu + 0.5).astype(np.int64)


def mu_law_decode(quantized, mu=QUANT - 1):
    signal = 2 * (quantized.astype(np.float32) / mu) - 1
    magnitude = (1 / mu) * ((1 + mu) ** np.abs(signal) - 1)
    return np.sign(signal) * magnitude


class CausalConv1d(nn.Module):
    """1-D conv that depends only on current and past inputs."""
    def __init__(self, in_ch, out_ch, kernel_size=2, dilation=1):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation)

    def forward(self, x):
        return self.conv(F.pad(x, (self.pad, 0)))  # left-pad only -> causal


class ResidualBlock(nn.Module):
    """Dilated causal layer: gated activation -> 1x1 residual + 1x1 skip."""
    def __init__(self, res_ch, dil_ch, skip_ch, dilation, cond_ch=None):
        super().__init__()
        self.filter_conv = CausalConv1d(res_ch, dil_ch, 2, dilation)
        self.gate_conv   = CausalConv1d(res_ch, dil_ch, 2, dilation)
        if cond_ch is not None:
            self.cond_filter = nn.Conv1d(cond_ch, dil_ch, 1)
            self.cond_gate   = nn.Conv1d(cond_ch, dil_ch, 1)
        self.res_conv  = nn.Conv1d(dil_ch, res_ch, 1)
        self.skip_conv = nn.Conv1d(dil_ch, skip_ch, 1)

    def forward(self, x, cond=None):
        f, g = self.filter_conv(x), self.gate_conv(x)
        if cond is not None:
            f = f + self.cond_filter(cond)
            g = g + self.cond_gate(cond)
        z = torch.tanh(f) * torch.sigmoid(g)         # gated activation
        return x + self.res_conv(z), self.skip_conv(z)


class WaveNet(nn.Module):
    def __init__(self, n_blocks=3, n_layers=10,
                 res_ch=32, dil_ch=32, skip_ch=256, cond_ch=None):
        super().__init__()
        self.input_conv = CausalConv1d(QUANT, res_ch, 2, 1)
        self.dilations = [2 ** i for i in range(n_layers)] * n_blocks  # 1..512 repeated
        self.blocks = nn.ModuleList(
            ResidualBlock(res_ch, dil_ch, skip_ch, d, cond_ch)
            for d in self.dilations)
        self.head = nn.Sequential(
            nn.ReLU(), nn.Conv1d(skip_ch, skip_ch, 1),
            nn.ReLU(), nn.Conv1d(skip_ch, QUANT, 1))

    @property
    def receptive_field(self):
        # (k-1)*sum(dilations)+1 for the dilated stack, +(k-1) for the input conv
        return (2 - 1) * sum(self.dilations) + 1 + (2 - 1)

    def forward(self, x_onehot, cond=None):
        x = self.input_conv(x_onehot)
        skips = 0
        for block in self.blocks:
            x, skip = block(x, cond)
            skips = skips + skip
        return self.head(skips)  # (B, QUANT, T) next-sample logits


def loss_fn(model, waveform, cond=None):
    # cross-entropy == -Σ log p(x_t | x_<t)
    x = F.one_hot(waveform[:, :-1], QUANT).float().transpose(1, 2)
    logits = model(x, cond)
    return F.cross_entropy(logits, waveform[:, 1:])
```
