**Problem (from step 4).** The U-Net is the strongest baseline (long NMSE 0.347, reaches its level by the
short budget) but leaves two things unspent. Its self-attention sits only at the *coarse* 16-position
bottleneck, so distant *fine* levels never couple directly — the small residual ml_nmse gain (0.373→0.347).
And its conv-then-pool scalar head regressed badly (sl_nmse 0.354–0.396 vs the CNN's 0.058, and seed-unstable).
Both point at one fix.

**Key idea.** A Conformer encoder (Gulati et al. 2020) over the 60 levels: run global
self-attention *and* a local depthwise-conv module at **full resolution in every block**, so any level couples
to any other in one hop (closing the fine-resolution coupling) while the conv keeps the sharp per-level
structure. Each level is a token (9 profile values + broadcast column scalars, embedded with a learned
*absolute* per-level positional encoding — height has real physical meaning). Block: half-step macaron FFN →
MHSA → conv module → half-step macaron FFN → LayerNorm. Restore the CNN's clean two structure-matched heads.

**Why it works.** Full-resolution attention is cheap on a 60-level axis (the U-Net coarsened only for cost,
which does not bite here) and is the operator the U-Net lacked at fine resolution. The pooled-MLP scalar head
is the CNN's design that produced sl_nmse 0.058, undoing the U-Net's regression. Risk: 4 blocks without
coarsening could underfit local structure relative to the U-Net's deep conv stack — ml_nmse could regress.

**Bar / what to validate (no feedback — this is the endpoint).** Clear the U-Net's long NMSE 0.347
consistently across all three budgets and seeds. Expect ml_nmse below 0.347 (fine coupling), a large sl_nmse
recovery toward 0.058 (pooled head), and skill reached early (full-column reach per block). The clean win is
all three: ml_nmse↓ and sl_nmse↓ and long NMSE < 0.347.

**Adaptations to the harness (what the speech recipe drops).** No conv-subsampling front end (60 levels is
short; subsampling discards per-level resolution), no decoder (fixed-length target), no augmentation
(pre-normalized). Conv module = LN→pointwise(×2)→GLU→depthwise(k7)→BatchNorm→Swish→pointwise→dropout;
FFN = LN→Linear(×4)→Swish→dropout→Linear→dropout. d_model 256, 4 blocks, 4 heads, kernel 7. AdamW + cosine
LR + MSE, all fixed.

```python
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
        h = F.glu(self.pw1(h), dim=1)           # gate channels
        h = self.dw(h)                          # local vertical window (k=7)
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
        x = x + 0.5 * self.ff1(x)               # half-step FFN
        x = x + self.mhsa(x)                     # global, content-based level coupling
        x = x + self.conv(x)                     # local per-level detail on globalized features
        x = x + 0.5 * self.ff2(x)               # half-step FFN
        return self.ln(x)


class Custom(nn.Module):
    """Conformer encoder over the 60 vertical levels for climate physics emulation.

    Each of the 60 levels is a token (its 9 profile values + broadcast column
    scalars, embedded with a learned absolute per-level positional encoding). A
    stack of conformer blocks couples distant levels by attention and sharpens
    local vertical structure by depthwise convolution, both at full resolution.
    Two structure-matched heads read off the 360 per-level tendencies and the 8
    whole-column diagnostics.
    """

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

        # Per-level token: 9 profile values at the level + broadcast column scalars.
        self.embed = nn.Linear(self.N_PROFILE_IN + self.n_scalar_in, d_model)
        # Learned ABSOLUTE per-level positional encoding (height has real meaning).
        self.pos = nn.Parameter(torch.zeros(1, self.N_LEVELS, d_model))

        self.blocks = nn.ModuleList(
            _ConformerBlock(d_model, heads, kernel) for _ in range(n_blocks)
        )
        self.out_norm = nn.LayerNorm(d_model)

        # Per-level head -> 6 tendency channels per level (the 360 multi-level targets).
        self.ml_head = nn.Linear(d_model, self.N_PROFILE_OUT)
        # Whole-column head: mean-pool over levels -> small MLP -> 8 scalar diagnostics.
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
