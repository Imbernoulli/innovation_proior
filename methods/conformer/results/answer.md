# Conformer

## Problem

Build an end-to-end ASR encoder that captures both **global** (long-range,
content-based) and **local** (position-based, fine-grained) dependencies
parameter-efficiently. Self-attention is globally strong but a blunt local
extractor; convolution is locally sharp but needs depth/parameters for global
reach; and bolting a single averaged global summary onto a conv stack (squeeze-
and-excitation) cannot model dynamic, position-dependent global interactions.

## Key idea

A block that **composes** self-attention and convolution — attention first to
establish global context, then a gated depthwise-convolution module to carve local
detail on the globalized features — and brackets that core with **two half-step
feed-forward modules** in a Macaron arrangement.

- **Self-attention with relative positional encoding**, in a pre-norm residual
  unit, for global content interactions robust to utterance length.
- **Convolution module**: LayerNorm → pointwise conv (expansion 2) → GLU →
  depthwise conv (kernel 32) → BatchNorm → Swish → pointwise conv → dropout, as a
  residual — the local, position-based operator.
- **Macaron FFNs**: two position-wise FFNs (4× inner, Swish) with **half-step
  residual weight 1/2**, one before and one after the attention+conv core,
  followed by a final LayerNorm.

## Final block

For input x_i to block i:

  x̃_i = x_i + (1/2)·FFN(x_i)
  x'_i = x̃_i + MHSA(x̃_i)
  x''_i = x'_i + Conv(x'_i)
  y_i = LayerNorm( x''_i + (1/2)·FFN(x''_i) )

FFNs carry half-weight residuals; MHSA and Conv carry full-weight residuals.

Config: convolutional subsampling front end (two stride-2 convs) over 80-channel
log-mel features. Sizes (layers / dim / heads, kernel 32): S 16/144/4 (~10M),
M 16/256/4 (~30M), L 17/512/8 (~118M); single-LSTM-layer transducer decoder.
Train with Adam (β₁=0.9, β₂=0.98, ε=1e-9), Transformer schedule (10k warmup, peak
0.05/√d), dropout 0.1 per residual, small L2, variational noise, SpecAugment.

## Code

```python
import torch, torch.nn as nn, torch.nn.functional as F

class Swish(nn.Module):
    def forward(self, x): return x * torch.sigmoid(x)

class FeedForward(nn.Module):                       # 4x inner, Swish, pre-norm
    def __init__(self, d, expansion=4, p=0.1):
        super().__init__()
        self.ln = nn.LayerNorm(d)
        self.net = nn.Sequential(nn.Linear(d, expansion * d), Swish(), nn.Dropout(p),
                                 nn.Linear(expansion * d, d), nn.Dropout(p))
    def forward(self, x):
        return self.net(self.ln(x))

class RelMultiHeadSelfAttention(nn.Module):
    def __init__(self, d, heads, p=0.1):
        super().__init__()
        self.ln = nn.LayerNorm(d)
        self.mha = nn.MultiheadAttention(d, heads, dropout=p, batch_first=True)
        self.drop = nn.Dropout(p)                   # relative positional encoding wired in
    def forward(self, x):
        h = self.ln(x)
        a, _ = self.mha(h, h, h)
        return self.drop(a)

class ConvolutionModule(nn.Module):                 # gated depthwise conv
    def __init__(self, d, kernel=32, p=0.1):
        super().__init__()
        self.ln  = nn.LayerNorm(d)
        self.pw1 = nn.Conv1d(d, 2 * d, 1)           # expansion 2
        self.dw  = nn.Conv1d(d, d, kernel, padding=kernel // 2, groups=d)
        self.bn  = nn.BatchNorm1d(d)
        self.act = Swish()
        self.pw2 = nn.Conv1d(d, d, 1)
        self.drop = nn.Dropout(p)
    def forward(self, x):
        h = self.ln(x).transpose(1, 2)
        h = F.glu(self.pw1(h), dim=1)               # GLU -> back to d channels
        h = self.act(self.bn(self.dw(h)))
        h = self.drop(self.pw2(h))
        return h.transpose(1, 2)

class ConformerBlock(nn.Module):
    def __init__(self, d, heads, kernel=32, p=0.1):
        super().__init__()
        self.ff1  = FeedForward(d, p=p)
        self.mhsa = RelMultiHeadSelfAttention(d, heads, p)
        self.conv = ConvolutionModule(d, kernel, p)
        self.ff2  = FeedForward(d, p=p)
        self.ln   = nn.LayerNorm(d)
    def forward(self, x):
        x = x + 0.5 * self.ff1(x)                   # half-step
        x = x + self.mhsa(x)                        # global
        x = x + self.conv(x)                        # then local
        x = x + 0.5 * self.ff2(x)                   # half-step
        return self.ln(x)

class ConvSubsampling(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.c1, self.c2 = nn.Conv2d(1, d, 3, 2), nn.Conv2d(d, d, 3, 2)
    def forward(self, x):
        x = F.relu(self.c1(x.unsqueeze(1)))
        x = F.relu(self.c2(x))
        b, c, t, f = x.shape
        return x.permute(0, 2, 1, 3).reshape(b, t, c * f)

class ConformerEncoder(nn.Module):
    def __init__(self, d=512, layers=17, heads=8, kernel=32, n_mels=80):
        super().__init__()
        self.subsample = ConvSubsampling(d)
        self.proj = nn.Linear(d * (((n_mels - 1) // 2 - 1) // 2), d)
        self.drop = nn.Dropout(0.1)
        self.blocks = nn.ModuleList(ConformerBlock(d, heads, kernel) for _ in range(layers))
    def forward(self, x):
        x = self.drop(self.proj(self.subsample(x)))
        for blk in self.blocks:
            x = blk(x)
        return x
```
