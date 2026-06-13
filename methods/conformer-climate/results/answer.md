**Problem.** Emulate a climate column's sub-grid tendencies from its 60-level vertical profiles plus
whole-column scalars, scored by NMSE. The map has two dependency types that must be captured at once: local
along height (neighbor-level gradients/curvature) and long-range vertical coupling (surface fluxes driving
heating aloft, cloud-top radiative coupling on a coupled column). Convolution is locally sharp but globally
expensive; attention is globally strong but a blunt local extractor; a multi-resolution net couples distant
levels only at coarse resolution.

**Key idea.** A Conformer encoder over the 60-level axis (Gulati et al. 2020, arXiv:2005.08100), used as a
pure regression encoder. Each level is a token (its 9 profile values + broadcast column scalars, linearly
embedded with a learned *absolute* per-level positional encoding). Each conformer block composes the two
operators at *full resolution*: a half-step macaron FFN, then multi-head self-attention (global, content-based
level-to-level coupling), then a gated depthwise conv module (local per-level detail on the globalized
features), then a half-step macaron FFN, with a closing LayerNorm. Block forward:
x̃ = x + ½·FFN₁(x); x' = x̃ + MHSA(x̃); x'' = x' + Conv(x'); y = LN(x'' + ½·FFN₂(x'')).

**Why it works.** Attention couples any level to any other in one hop at full resolution (the long-range
operator the conv stack and the coarse-bottleneck U-Net lacked), while the depthwise conv keeps the sharp
local-along-height structure — neither half is faked. The macaron sandwich is the ODE-step view of the block.
Absolute (not relative) positional encoding because the surface vs. tropopause distinction is physically real.

**Adaptations to the regression harness (what the speech recipe drops).** No convolutional subsampling front
end (60 levels is short; subsampling would discard per-level resolution), no transducer/attention decoder
(fixed-length target), no SpecAugment (inputs pre-normalized). Conv module = LN → pointwise(×2) → GLU →
depthwise(k=7) → BatchNorm → Swish → pointwise → dropout; FFN = LN → Linear(×4) → Swish → dropout → Linear →
dropout. Two structure-matched heads: per-level Linear→6 channels (var-major reshape to 360) and
mean-pool-over-levels→MLP→8. d_model 256, 4 blocks, 4 heads, kernel 7. Fixed AdamW + cosine LR + MSE.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class _Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class _FeedForward(nn.Module):
    """Macaron FFN: LayerNorm -> Linear(x4) -> Swish -> Dropout -> Linear -> Dropout."""
    def __init__(self, d, expansion=4, dropout=0.1):
        super().__init__()
        self.ln = nn.LayerNorm(d)
        self.net = nn.Sequential(
            nn.Linear(d, expansion * d), _Swish(), nn.Dropout(dropout),
            nn.Linear(expansion * d, d), nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(self.ln(x))


class _MHSA(nn.Module):
    """Pre-norm multi-head self-attention over the level axis (global coupling)."""
    def __init__(self, d, heads=4, dropout=0.1):
        super().__init__()
        self.ln = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        h = self.ln(x)
        out, _ = self.attn(h, h, h, need_weights=False)
        return self.drop(out)


class _ConvModule(nn.Module):
    """Gated depthwise conv: LN -> PW(x2) -> GLU -> DWConv(k7) -> BN -> Swish -> PW -> Dropout."""
    def __init__(self, d, kernel=7, dropout=0.1):
        super().__init__()
        self.ln = nn.LayerNorm(d)
        self.pw1 = nn.Conv1d(d, 2 * d, 1)
        self.dw = nn.Conv1d(d, d, kernel, padding=kernel // 2, groups=d)
        self.bn = nn.BatchNorm1d(d)
        self.act = _Swish()
        self.pw2 = nn.Conv1d(d, d, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        h = self.ln(x).transpose(1, 2)          # (B, d, L)
        h = F.glu(self.pw1(h), dim=1)
        h = self.dw(h)
        h = self.act(self.bn(h))
        h = self.drop(self.pw2(h))
        return h.transpose(1, 2)                # (B, L, d)


class _ConformerBlock(nn.Module):
    """Macaron FFN / MHSA / Conv / FFN; half-step FFN residuals; closing LayerNorm."""
    def __init__(self, d, heads=4, kernel=7, dropout=0.1):
        super().__init__()
        self.ff1 = _FeedForward(d, dropout=dropout)
        self.mhsa = _MHSA(d, heads, dropout)
        self.conv = _ConvModule(d, kernel, dropout)
        self.ff2 = _FeedForward(d, dropout=dropout)
        self.ln = nn.LayerNorm(d)

    def forward(self, x):
        x = x + 0.5 * self.ff1(x)
        x = x + self.mhsa(x)
        x = x + self.conv(x)
        x = x + 0.5 * self.ff2(x)
        return self.ln(x)


class Custom(nn.Module):
    """Conformer encoder over the 60 vertical levels for climate physics emulation."""

    N_LEVELS = 60
    N_PROFILE_IN = 9
    N_PROFILE_OUT = 6
    N_SCALAR_OUT = 8

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_scalar_in = input_dim - self.N_PROFILE_IN * self.N_LEVELS

        d_model, n_blocks, heads, kernel = 256, 4, 4, 7

        self.embed = nn.Linear(self.N_PROFILE_IN + self.n_scalar_in, d_model)
        self.pos = nn.Parameter(torch.zeros(1, self.N_LEVELS, d_model))

        self.blocks = nn.ModuleList(
            _ConformerBlock(d_model, heads, kernel) for _ in range(n_blocks)
        )
        self.out_norm = nn.LayerNorm(d_model)

        self.ml_head = nn.Linear(d_model, self.N_PROFILE_OUT)
        self.sl_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.ReLU(),
            nn.Linear(d_model // 2, self.N_SCALAR_OUT),
        )

    def forward(self, x):
        B = x.shape[0]
        x_profile = x[:, :self.N_PROFILE_IN * self.N_LEVELS].view(
            B, self.N_PROFILE_IN, self.N_LEVELS).transpose(1, 2)        # (B, 60, 9)
        x_scalar = x[:, self.N_PROFILE_IN * self.N_LEVELS:]             # (B, n_scalar)
        x_scalar = x_scalar.unsqueeze(1).expand(-1, self.N_LEVELS, -1)  # (B, 60, n_scalar)
        tokens = torch.cat([x_profile, x_scalar], dim=-1)              # (B, 60, 9+n_scalar)

        h = self.embed(tokens) + self.pos                              # (B, 60, d)
        for block in self.blocks:
            h = block(h)
        h = self.out_norm(h)                                           # (B, 60, d)

        ml_out = self.ml_head(h).transpose(1, 2).reshape(B, -1)        # (B, 6*60) var-major
        sl_out = self.sl_head(h.mean(dim=1))                           # (B, 8)
        return torch.cat([ml_out, sl_out], dim=-1)                     # (B, 368)
```
