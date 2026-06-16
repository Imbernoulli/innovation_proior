# Conformer

## Problem

Build an end-to-end ASR encoder that captures both **global** (long-range,
content-based) and **local** (position-based, fine-grained) dependencies
parameter-efficiently. Self-attention is globally strong but a blunt local
extractor; convolution is locally sharp but needs depth/parameters for global
reach; and bolting a single averaged global summary onto a conv stack (squeeze-
and-excitation) cannot model dynamic, position-dependent global interactions.

## Key idea

Conformer is an end-to-end ASR encoder that puts content-based global modeling and
position-based local modeling in the same residual block. Self-attention supplies
utterance-wide interactions, convolution supplies efficient local acoustic feature
extraction, and the feed-forward layers are split into two half-weighted Macaron
steps around the attention-plus-convolution core.

## Final block

For input x_i to block i, the final block is:

```text
x_tilde_i = x_i + (1/2) * FFN(x_i)
x_prime_i = x_tilde_i + MHSA(x_tilde_i)
x_double_prime_i = x_prime_i + Conv(x_prime_i)
y_i = LayerNorm(x_double_prime_i + (1/2) * FFN(x_double_prime_i))
```

## Modules and configuration

The MHSA module uses relative sinusoidal positional encoding and pre-norm. The
convolution module is LayerNorm, pointwise convolution with expansion factor 2,
GLU, depthwise 1-D convolution, BatchNorm, Swish, pointwise convolution, and
dropout. The two FFNs use an inner width of 4d with Swish and dropout, and only
their residual contributions are weighted by 1/2; MHSA and convolution have full
residual weight.

The encoder uses convolutional subsampling over 80-channel log-mel filterbank
features before the block stack. The reported model sizes are S: 16 layers,
d = 144, 4 heads, 10.3M parameters; M: 16 layers, d = 256, 4 heads, 30.7M
parameters; and L: 17 layers, d = 512, 8 heads, 118.8M parameters. All use
depthwise convolution kernel size 32 and a single-LSTM-layer transducer decoder.
Training uses dropout 0.1 on each module output before the residual add, L2
regularization with weight 1e-6, variational noise, Adam with beta1 = 0.9,
beta2 = 0.98, epsilon = 1e-9, a Transformer learning-rate schedule with 10k
warmup steps and peak learning rate 0.05 / sqrt(d), and SpecAugment on the
filterbank inputs.

## Code

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class FeedForward(nn.Module):
    def __init__(self, d, expansion=4, p=0.1):
        super().__init__()
        self.ln = nn.LayerNorm(d)
        self.net = nn.Sequential(
            nn.Linear(d, expansion * d),
            Swish(),
            nn.Dropout(p),
            nn.Linear(expansion * d, d),
            nn.Dropout(p),
        )

    def forward(self, x):
        return self.net(self.ln(x))


class RelativeSinusoidalEncoding(nn.Module):
    def __init__(self, d_head):
        super().__init__()
        self.d_head = d_head

    def forward(self, length, device):
        positions = torch.arange(
            -(length - 1), length, device=device, dtype=torch.float32)
        freq = torch.arange(0, self.d_head, 2, device=device, dtype=torch.float32)
        inv_freq = 1.0 / (10000 ** (freq / self.d_head))
        angles = positions[:, None] * inv_freq[None, :]
        emb = torch.stack((angles.sin(), angles.cos()), dim=-1).flatten(-2)
        return emb[:, :self.d_head]


class RelMultiHeadSelfAttention(nn.Module):
    def __init__(self, d, heads, p=0.1):
        super().__init__()
        if d % heads:
            raise ValueError("d must be divisible by heads")
        self.heads = heads
        self.d_head = d // heads
        self.ln = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.rel = RelativeSinusoidalEncoding(self.d_head)
        self.content_bias = nn.Parameter(torch.zeros(heads, self.d_head))
        self.pos_bias = nn.Parameter(torch.zeros(heads, self.d_head))
        self.drop = nn.Dropout(p)
        self.out = nn.Linear(d, d)

    def forward(self, x):
        b, t, d = x.shape
        h = self.ln(x)
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        q = q.view(b, t, self.heads, self.d_head).transpose(1, 2)
        k = k.view(b, t, self.heads, self.d_head).transpose(1, 2)
        v = v.view(b, t, self.heads, self.d_head).transpose(1, 2)

        q_content = q + self.content_bias[None, :, None, :]
        q_pos = q + self.pos_bias[None, :, None, :]
        content_scores = torch.einsum("bhtd,bhsd->bhts", q_content, k)

        rel = self.rel(t, x.device)
        rel_scores = torch.einsum("bhtd,md->bhtm", q_pos, rel)
        offsets = (
            torch.arange(t, device=x.device)[None, :]
            - torch.arange(t, device=x.device)[:, None]
            + t - 1
        )
        offsets = offsets.view(1, 1, t, t).expand(b, self.heads, t, t)
        rel_scores = torch.gather(rel_scores, dim=-1, index=offsets)

        scores = (content_scores + rel_scores) / math.sqrt(self.d_head)
        weights = self.drop(torch.softmax(scores, dim=-1))
        y = torch.einsum("bhts,bhsd->bhtd", weights, v)
        y = y.transpose(1, 2).contiguous().view(b, t, d)
        return self.drop(self.out(y))


class ConvolutionModule(nn.Module):
    def __init__(self, d, kernel=32, p=0.1):
        super().__init__()
        self.kernel = kernel
        self.ln = nn.LayerNorm(d)
        self.pw1 = nn.Conv1d(d, 2 * d, 1)
        self.dw = nn.Conv1d(d, d, kernel, groups=d)
        self.bn = nn.BatchNorm1d(d)
        self.act = Swish()
        self.pw2 = nn.Conv1d(d, d, 1)
        self.drop = nn.Dropout(p)

    def forward(self, x):
        h = self.ln(x).transpose(1, 2)
        h = F.glu(self.pw1(h), dim=1)
        left = (self.kernel - 1) // 2
        right = self.kernel // 2
        h = self.dw(F.pad(h, (left, right)))
        h = self.act(self.bn(h))
        h = self.drop(self.pw2(h))
        return h.transpose(1, 2)


class ConformerBlock(nn.Module):
    def __init__(self, d, heads, kernel=32, p=0.1):
        super().__init__()
        self.ff1 = FeedForward(d, p=p)
        self.mhsa = RelMultiHeadSelfAttention(d, heads, p)
        self.conv = ConvolutionModule(d, kernel, p)
        self.ff2 = FeedForward(d, p=p)
        self.ln = nn.LayerNorm(d)

    def forward(self, x):
        x = x + 0.5 * self.ff1(x)
        x = x + self.mhsa(x)
        x = x + self.conv(x)
        x = x + 0.5 * self.ff2(x)
        return self.ln(x)


class ConvSubsampling(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.c1 = nn.Conv2d(1, channels, 3, stride=2)
        self.c2 = nn.Conv2d(channels, channels, 3, stride=2)

    def forward(self, x):
        x = F.relu(self.c1(x.unsqueeze(1)))
        x = F.relu(self.c2(x))
        b, c, t, f = x.shape
        return x.permute(0, 2, 1, 3).reshape(b, t, c * f)


class ConformerEncoder(nn.Module):
    def __init__(self, d=512, layers=17, heads=8, kernel=32):
        super().__init__()
        self.subsample = ConvSubsampling(d)
        self.proj = nn.LazyLinear(d)
        self.drop = nn.Dropout(0.1)
        self.blocks = nn.ModuleList(
            ConformerBlock(d, heads, kernel) for _ in range(layers))

    def forward(self, x):
        x = self.drop(self.proj(self.subsample(x)))
        for block in self.blocks:
            x = block(x)
        return x
```
