The U-Net is the strongest baseline, and its numbers tell me both why it won and exactly what it left on the table. At the long budget it lands at NMSE $0.347$, R² $0.653$ — better than the CNN's $0.365$ — and the most telling fact is *when* it gets there: its short-30ep NMSE is already $0.355$, almost its final $0.347$, where the CNN needed the full budget to crawl from $0.443$ to $0.365$. That is the signature of multi-resolution context plus bottleneck attention: it reaches skill fast because it does not have to slowly accrete a vertical receptive field over epochs. But "most" is not all. First, ml_nmse only moved from $0.373$ to $0.347$ — a real but *small* gain on the multi-level tendencies — and the reason is *where* the U-Net's attention acts: at the coarse bottleneck, after two downsamples, where sixty levels have been pooled to sixteen positions. So the only coupling the attention can express is between *coarse groups of ~4 levels* through their pooled summaries; distant *fine* levels never query each other directly. The full-resolution operator in the U-Net is still only the local convolution, and the global operator only ever sees a blurred sixteen-position version of the column. The sliver the U-Net still misses is the *fine-grained* part of the long-range coupling that survives only at full resolution. Second, the regression I explicitly flagged: its single-level head fell apart, sl_nmse going from the CNN's clean $0.058$ to $0.354$–$0.396$ — worse by a factor of six and wildly seed-dependent. The eight whole-column diagnostics, which the CNN nailed with a pooled MLP head, are mangled by the U-Net's conv-then-pool output path. So the strongest baseline leaves two things uncaptured: the fine-resolution long-range coupling (attention only acts coarse) and the single-level diagnostics (the scalar readout regressed). Both point at the same fix — stop confining the global operator to the coarse bottleneck, and run *both* operators, global content-based level-to-level coupling *and* local per-level convolution, at *full* sixty-level resolution in every block. Because the axis is only sixty long, full-resolution attention is cheap; the U-Net coarsened to make attention affordable, but on a sixty-level column I never needed to.

I propose a **Conformer encoder over the sixty levels** (Gulati et al. 2020): a stack of convolution-augmented transformer blocks that compose global self-attention with a local depthwise convolution at full resolution. The naive way to combine a global and a local operator is to run them as parallel branches and concatenate, but that just sets them side by side and leaves fusion to later layers — the convolution never acts on the globally-mixed features. I want them to *compose*: attention first, to make every level a content-weighted mixture of the whole column (the surface-drives-aloft coupling now present in each level's features), then a local convolution that sharpens the per-level vertical pattern *on those globalized features*. Establish the long-range coupling, then carve local detail on top. That ordering — self-attention then convolution, each in a pre-norm residual — is the spine of the block.

Each sublayer's internals follow from the setting. The attention is multi-head and pre-norm, with one choice that differs from the usual sequence deployment: positional encoding. In text or speech the absolute index is arbitrary and only offsets matter, arguing for relative encodings — but a climate column is *not* translation-invariant in height. Level 0 is the surface, level 59 the model top, and "near the surface" versus "near the tropopause" is a real physical distinction the tendencies depend on, so I use a learned *absolute* per-level positional embedding, one vector per height, added to the tokens — the model conditions on which level it is, not just on gaps, and this is cheap because there are only sixty positions. The convolution sublayer is a gated depthwise module: LayerNorm, a pointwise convolution that doubles the channels followed by a GLU gate (so the module can suppress irrelevant channels at a level), a depthwise convolution along the level axis with kernel seven (a level and three neighbors each side — local, leaving the long-range job to attention), BatchNorm and a Swish activation, a second pointwise convolution back to the model width, and dropout. Around this attention-then-convolution core I place two *half-step* feed-forward modules following the ODE/macaron view — one before, one after, each with a one-half residual weight — and close with a LayerNorm. The two FFNs are the symmetric half-steps; attention and convolution carry the full-weight residuals as the main mixing operations. So each block is

$$x \leftarrow x + \tfrac12\,\mathrm{FFN}(x), \quad x \leftarrow x + \mathrm{MHSA}(x), \quad x \leftarrow x + \mathrm{Conv}(x), \quad x \leftarrow x + \tfrac12\,\mathrm{FFN}(x), \quad x \leftarrow \mathrm{LayerNorm}(x).$$

I deliberately *cut* the rest of the speech-encoder recipe, exactly as I dropped the U-Net's segmentation machinery, because importing it wholesale would be wrong here. A speech conformer prepends a convolutional subsampling front end — stride-two convolutions to cut a long time axis before attention — and appends a transducer or attention decoder with audio augmentation. None of that fits: the axis is sixty levels, so there is nothing to subsample, and subsampling would throw away the per-level resolution I need to place tendencies at the right height — the opposite of what I want, since the U-Net's coarsening is the very limit I am escaping. There is no autoregressive output, so no decoder; the target is a fixed-length per-level-plus-scalar vector. The inputs are already normalized by the fixed harness. So I strip the front end and the decoder and build the minimal column version: tokenize each level as its nine profile values concatenated with the broadcast whole-column scalars, linearly embed to the model width, add the absolute per-level positional embedding, run four conformer blocks at full resolution, and read off.

The output directly fixes the U-Net's scalar regression by returning to the two structure-matched heads the CNN got right, now reading a full-resolution attention-conv representation. After the blocks and a final LayerNorm I have a per-level feature at every height. The six multi-level tendencies are per-level quantities, so the head is a per-level linear readout — a Linear from the model width to six outputs applied at every level — reshaped var-major to the 360 targets. The eight single-level diagnostics describe the whole column, so the head mean-pools over the levels and runs a small MLP to the eight numbers — the same clean pooled-MLP design that gave the CNN sl_nmse $0.058$, not the U-Net's noisy conv-then-pool path. So this rung restores the scalar head that worked *and* upgrades the multi-level path with full-resolution coupling.

The delta from the U-Net is two precise moves. First, lift the global operator out of the coarse bottleneck and run it at full sixty-level resolution in every block, composed with a local depthwise convolution — closing the fine-resolution long-range coupling the U-Net's sixteen-position bottleneck attention could not express. Second, drop the U-Net's noisy conv-then-pool scalar readout for the CNN's clean pooled-MLP head — undoing the sl_nmse regression. The bet is that the U-Net's remaining ml_nmse sliver is the fine part of the coupling, reachable only by full-resolution attention, and that its sl_nmse blowup is purely a readout artifact a better head removes. The bar this must clear: beat NMSE $0.347$ at the long budget, consistently across all three budgets and all three seeds. The falsifiable expectations are specific — ml_nmse below $0.347$ (fine coupling resolved; if full-resolution attention buys nothing over coarse attention, ml_nmse sits at or above $0.347$ and the "fine coupling" diagnosis is wrong), a large sl_nmse recovery toward the CNN's $0.058$ and far below the U-Net's $0.354$ (the pooled head is the CNN's design), and skill reached *early* like the U-Net because every block already has full-column reach. The risk I hold open: full-resolution attention over sixty levels with only four blocks and no coarsening could *underfit* the local structure relative to the U-Net's deep multi-resolution conv stack, in which case ml_nmse could come in worse than $0.347$ even as sl_nmse recovers. The clean win is all three at once — ml_nmse down *and* sl_nmse down *and* long NMSE below $0.347$.

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
