The convolutional rung was the big jump. At the long budget the CNN reaches NMSE $0.365$, R² $0.635$ — down from ED's $0.583$ — and the win is concentrated exactly on the metric ED throttled: ml_nmse fell from $0.595$ to $0.373$, sl_nmse from $0.133$ to $0.058$, and it keeps improving with budget ($0.443 \to 0.384 \to 0.365$). Putting the sixty-level axis back into the architecture and spending full representational width on local vertical detectors was the right call. But look at where it stalls: from medium to long it moves only $0.384 \to 0.365$, and ml_nmse at $0.373$ still leaves a third of the multi-level variance uncaptured. That is the ceiling I flagged — a width-three kernel sees only a level and its two neighbors per layer, and even eight residual blocks build a *limited* vertical receptive field. The convolution is excellent at local vertical structure (gradients, curvature, adjacent-level adjustments) and that is most of what it bought; what it cannot do cheaply is relate *distant* levels. But the physics has genuinely long-range vertical coupling — surface heat and moisture fluxes drive convection that deposits heating hundreds of hectopascals higher up, radiative cooling at a cloud top depends on what is below it, the column is a coupled system, not a stack of independent windows. A pure local-conv stack reaches distant levels only by stacking many windows, which is slow to train and is exactly the part flattening out. So the CNN's residual is, I suspect, the long-range vertical coupling its receptive field cannot span, and the next move must keep everything the convolution got right — the local per-level structure, the two structure-matched heads — and add a way to relate distant levels.

There are two ways to enlarge a receptive field over an ordered axis. Going deeper or wider is what is already flattening out, and it spends compute linearly for a receptive field that grows linearly. The other is to *change resolution*: pool the sixty levels to a coarse grid where a width-three kernel now spans a large fraction of the column, process there, then bring the resolution back — coarsening is the cheap way to get global reach, because at a 4× coarser grid a single window already sees a quarter of the column. But pooling buys context at the cost of localization: after I pool four levels into one I no longer know which of them an activation came from, and for this problem I need *both* — coarse whole-column context to capture surface-drives-aloft coupling, and fine per-level localization to place a tendency at the right height. So I cannot just pool and be done; I need an architecture that gets context from coarsening *and* recovers localization afterward. I propose a **1D U-Net over the level axis**: an encoder-decoder with skip connections — a U over the column.

A contracting path of conv-and-pool builds context — at each level residual conv blocks process the current resolution, then a stride-two downsample halves the number of levels while the channel count grows to preserve capacity as the axis shrinks. After two downsamples the sixty-ish levels reach a coarse grid where a single window spans most of the column, and that is where long-range coupling becomes cheap. An expanding path then upsamples back to full resolution, and here is the part that beats a plain pool-then-upsample: the upsampled coarse map knows *what* (whole-column context) but is blurry about *where* (the precise level), because pooling threw the fine localization away. That localization did not vanish from the network — it is still in the *shallow*, high-resolution features from before the pooling, which know *where* but lack context. So at each expanding step I upsample, then **concatenate** the matching contracting-path feature map, then run conv blocks that *learn* to fuse the fine per-level detail with the coarse context. Concatenate rather than add, so the network is not forced into a fixed linear mixture and can reason, per level, about how a fine local feature should modulate a coarse decision. The picture is a U: down the left building whole-column context at coarse resolution, up the right rebuilding per-level resolution, rungs across the middle carrying the localization the pooling discarded.

There is one operator a pure local conv stack fundamentally lacks even with pooling: a way to relate *every* coarse position to *every other* directly and content-dependently. Pooling gives global reach only through the blunt averaging of a wide window; what I want at the most-coarsened level, where there are only a handful of positions, is for each position to attend to all the others based on what they contain — a surface-regime position directly querying an upper-level position. That is self-attention. It is expensive at full sixty-level resolution, but at the bottleneck the axis is already coarse, so a single self-attention block there is cheap and is exactly where it pays: distant coarsened levels exchange information in one hop, the long-range coupling the CNN's local windows could only approximate by stacking. So the bottleneck is residual-conv $\to$ self-attention $\to$ residual-conv: process locally, mix globally, process locally again.

A few internals follow from the regression setting, where I deliberately diverge from how a U-shaped network is built for its original dense-labeling use. There is no segmentation loss here, no per-pixel weighted cross-entropy, no class-imbalance reweighting, no elastic-deformation augmentation, no valid-convolution tiling — all machinery for dense classification that the fixed scaffold neither needs nor exposes. This is regression over a short fixed-length axis scored by MSE, so I use **same-padded** convolutions (keep the length, no skip cropping) and **GroupNorm** rather than BatchNorm in the residual blocks — the normalization should not depend on batch statistics that shift across the three training budgets, and GroupNorm pairs cleanly with the SiLU activations. Each residual block **zero-initializes its second convolution** so the block starts as the identity and the deep U trains stably from the first step — the same $h \gets h + F(h)$ conditioning argument as the CNN, made explicit at init, and applied to the attention's output projection too. The level axis is sixty, not a power of two, and clean halving needs an even divisible length, so I pad $60 \to 64$ before the encoder and crop back to $60$ at the output; two downsamples take $64 \to 32 \to 16$, so the bottleneck attention runs over sixteen positions — cheap.

The input and output reuse the CNN's channels-over-levels insight with one deliberate change at the input. The CNN gave the ~16 whole-column scalars a *learned* linear projection onto the level axis as one extra channel; here I instead **broadcast** each scalar across all sixty levels as its own channel — nine profile channels plus sixteen scalar channels, twenty-five input channels — and let the U's first convolution and the skip-carried fine features sort out how each scalar should localize, rather than committing to a single learned per-level pattern up front. The wider channel count is affordable because the U immediately lifts to a fixed base width. The output reuses the two-structure-matched-heads idea adapted to the U: the final conv emits fourteen channels over sixty levels, the first six being the per-level tendencies reshaped to the 360 multi-level targets, and the last eight reduced to whole-column scalars by **mean-pooling over the levels**, with a ReLU on those eight because the diagnostics — net fluxes, precipitation, solar — are physically non-negative.

The delta from the CNN is precise and surgical: keep the channels-over-levels layout, the residual conv blocks, and the two-headed output split that the CNN got right, and add exactly the two things its flat local stack lacked — multi-resolution context via the contracting/expanding U with skip-concatenation (so coarsening reaches distant levels and the skips restore per-level localization), and a self-attention block at the coarse bottleneck (so distant levels couple directly in one hop). The bet is that the CNN's residual is long-range vertical coupling, and that these two additions span it. If the diagnosis is right, the gain shows up as a further drop in `ml_nmse` and a flatter, *less* budget-sensitive NMSE — the U-Net should be near its final level already at the short budget, because multi-resolution context plus bottleneck attention reach skill fast rather than slowly accreting receptive field over epochs. The risk I hold open: self-attention over only sixteen coarse positions, plus the broadcast-then-fuse handling of the scalars, could leave the single-level diagnostics *worse* than the CNN's clean learned scalar projection — sl_nmse is where I would expect a regression if the U's whole-column readout is noisier than the CNN's pooled MLP head. So I watch ml_nmse to confirm the long-range win and sl_nmse to check I have not traded away the scalar half.

```python
class ResBlock1d(nn.Module):
    """1D residual block: GroupNorm + Conv1d + SiLU + Conv1d + skip."""
    def __init__(self, channels, dropout=0.1):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(32, channels // 4), channels)
        self.conv1 = nn.Conv1d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(min(32, channels // 4), channels)
        self.conv2 = nn.Conv1d(channels, channels, 3, padding=1)
        self.drop = nn.Dropout(dropout)
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x):
        h = F.silu(self.norm1(x))
        h = self.conv1(h)
        h = self.drop(F.silu(self.norm2(h)))
        h = self.conv2(h)
        return (x + h) * (0.5 ** 0.5)


class AttnBlock1d(nn.Module):
    """Self-attention over the sequence (level) dimension."""
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.norm = nn.GroupNorm(min(32, channels // 4), channels)
        self.qkv = nn.Conv1d(channels, channels * 3, 1)
        self.proj = nn.Conv1d(channels, channels, 1)
        self.num_heads = num_heads
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x):
        B, C, L = x.shape
        h = self.norm(x)
        qkv = self.qkv(h).reshape(B, 3, self.num_heads, C // self.num_heads, L)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        # Scaled dot-product attention
        scale = (C // self.num_heads) ** -0.5
        attn = torch.einsum('bhcl,bhcm->bhlm', q, k) * scale
        attn = attn.softmax(dim=-1)
        out = torch.einsum('bhlm,bhcm->bhcl', attn, v)
        out = out.reshape(B, C, L)
        return (x + self.proj(out)) * (0.5 ** 0.5)


class Custom(nn.Module):
    """1D U-Net for climate physics emulation (adapted from ClimsimUnet v4).

    Architecture:
    - Reshape flat [B, 556] -> [B, num_profile_vars + num_scalar_vars, 60]
      (profile vars naturally span 60 levels; scalars broadcast to all levels)
    - Pad to 64 (power of 2) for clean downsampling
    - Encoder: 3 resolution levels with residual blocks + downsampling
    - Bottleneck: residual block + self-attention
    - Decoder: 3 levels with skip connections + upsampling
    - Output projection back to flat [B, 368]
    """
    N_LEVELS = 60
    N_PROFILE_IN = 9   # 9 multi-level input vars
    N_SCALAR_IN = 16   # 16 single-level input vars
    N_PROFILE_OUT = 6  # 6 multi-level output vars
    N_SCALAR_OUT = 8   # 8 single-level output vars

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        in_ch = self.N_PROFILE_IN + self.N_SCALAR_IN  # 25 channels
        base_ch = 128

        # Encoder
        self.enc_in = nn.Conv1d(in_ch, base_ch, 3, padding=1)
        self.enc1 = nn.ModuleList([ResBlock1d(base_ch) for _ in range(3)])
        self.down1 = nn.Conv1d(base_ch, base_ch * 2, 2, stride=2)  # 64->32
        self.enc2 = nn.ModuleList([ResBlock1d(base_ch * 2) for _ in range(3)])
        self.down2 = nn.Conv1d(base_ch * 2, base_ch * 2, 2, stride=2)  # 32->16

        # Bottleneck with attention
        self.mid1 = ResBlock1d(base_ch * 2)
        self.mid_attn = AttnBlock1d(base_ch * 2, num_heads=4)
        self.mid2 = ResBlock1d(base_ch * 2)

        # Decoder
        self.up2 = nn.ConvTranspose1d(base_ch * 2, base_ch * 2, 2, stride=2)  # 16->32
        self.dec2 = nn.ModuleList([ResBlock1d(base_ch * 4)] +
                                  [ResBlock1d(base_ch * 4) for _ in range(2)])
        self.dec2_proj = nn.Conv1d(base_ch * 4, base_ch * 2, 1)
        self.up1 = nn.ConvTranspose1d(base_ch * 2, base_ch, 2, stride=2)  # 32->64
        self.dec1 = nn.ModuleList([ResBlock1d(base_ch * 2)] +
                                  [ResBlock1d(base_ch * 2) for _ in range(2)])
        self.dec1_proj = nn.Conv1d(base_ch * 2, base_ch, 1)

        # Output
        self.out_norm = nn.GroupNorm(min(32, base_ch // 4), base_ch)
        self.out_conv = nn.Conv1d(base_ch, self.N_PROFILE_OUT + self.N_SCALAR_OUT, 3, padding=1)

    def forward(self, x):
        B = x.shape[0]

        # Reshape: split profile (9 vars x 60 levels) and scalar (16 vars)
        x_profile = x[:, :self.N_PROFILE_IN * self.N_LEVELS]
        x_scalar = x[:, self.N_PROFILE_IN * self.N_LEVELS:]

        x_profile = x_profile.reshape(B, self.N_PROFILE_IN, self.N_LEVELS)  # [B, 9, 60]
        x_scalar = x_scalar.unsqueeze(2).expand(-1, -1, self.N_LEVELS)      # [B, 16, 60]
        h = torch.cat([x_profile, x_scalar], dim=1)  # [B, 25, 60]

        # Pad 60 -> 64 for clean 2x downsampling
        h = F.pad(h, (0, 4))  # [B, 25, 64]

        # Encoder
        h = self.enc_in(h)
        for block in self.enc1:
            h = block(h)
        skip1 = h  # [B, 128, 64]
        h = self.down1(h)  # [B, 256, 32]
        for block in self.enc2:
            h = block(h)
        skip2 = h  # [B, 256, 32]
        h = self.down2(h)  # [B, 256, 16]

        # Bottleneck
        h = self.mid1(h)
        h = self.mid_attn(h)
        h = self.mid2(h)

        # Decoder
        h = self.up2(h)  # [B, 256, 32]
        h = torch.cat([h, skip2], dim=1)  # [B, 512, 32]
        for block in self.dec2:
            h = block(h)
        h = self.dec2_proj(h)  # [B, 256, 32]
        h = self.up1(h)  # [B, 128, 64]
        h = torch.cat([h, skip1], dim=1)  # [B, 256, 64]
        for block in self.dec1:
            h = block(h)
        h = self.dec1_proj(h)  # [B, 128, 64]

        # Output
        h = self.out_conv(F.silu(self.out_norm(h)))  # [B, 14, 64]

        # Remove padding and reshape
        h = h[:, :, :self.N_LEVELS]  # [B, 14, 60]

        y_profile = h[:, :self.N_PROFILE_OUT, :].reshape(B, self.N_PROFILE_OUT * self.N_LEVELS)
        y_scalar = h[:, self.N_PROFILE_OUT:, :].mean(dim=2)  # avg over levels
        y_scalar = F.relu(y_scalar)  # non-negative scalar outputs

        return torch.cat([y_profile, y_scalar], dim=1)
```
