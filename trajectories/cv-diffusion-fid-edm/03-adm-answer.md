**Problem (context rung).** Rungs 1–2 redesigned the *training* of a fixed-shape denoiser. This rung
fixes the other axis — the U-Net **backbone** $F_\theta$ — at a strong setting, so the training-design
redesign that follows runs on a substrate that is no longer the bottleneck. It changes no
training-design knob (target, weighting, $\sigma$-distribution, augmentation are untouched).

**Key idea (ADM architecture).** A heavily-tuned denoiser U-Net:
- **Self-attention at multiple resolutions** ($32\times32$, $16\times16$, $8\times8$) instead of one,
  so the network assembles global structure at coarse/medium/fine scales — directly aiding the
  high-noise regime FID is most sensitive to.
- **Fixed 64 channels per head** (head count grows with channels), keeping every head at a consistent,
  well-conditioned dimensionality.
- **Adaptive group normalization (AdaGN):** inject the (timestep, class) embedding as a learned
  per-channel scale and shift, $\mathrm{AdaGN}(\mathbf{h},\mathbf{e})=\mathbf{e}_s\cdot\mathrm{GroupNorm}(\mathbf{h})+\mathbf{e}_b$
  — multiplicative gating by noise level and class, not mere addition.
- **BigGAN-style residual blocks** for up/downsampling, resampling through a residual path rather than
  a lossy bottleneck.
- **Depth** traded for width at fixed budget helps FID but costs wall-clock, so variable width is kept
  in practice; **residual rescaling by $1/\sqrt2$** is available but is the one change that does *not*
  improve FID here.

**What to watch.** A clearly stronger bar, most pronounced at higher resolution where multi-scale
attention and AdaGN have the most to do — the backbone a serious training-design study starts from. It
sharpens the open problem: the architecture is no longer the limit, so remaining FID headroom must be
in *how the denoiser is trained*.

```python
# ADM U-Net backbone: multi-resolution attention, 64-ch heads, AdaGN conditioning, BigGAN up/down blocks.
class DhariwalUNet(torch.nn.Module):
    def __init__(self, img_resolution, in_channels, out_channels, label_dim=0, augment_dim=0,
                 model_channels=192, channel_mult=[1,2,3,4], channel_mult_emb=4, num_blocks=3,
                 attn_resolutions=[32,16,8], dropout=0.10, label_dropout=0):
        super().__init__()
        emb_channels = model_channels * channel_mult_emb
        init = dict(init_mode='kaiming_uniform', init_weight=np.sqrt(1/3), init_bias=np.sqrt(1/3))
        init_zero = dict(init_mode='kaiming_uniform', init_weight=0, init_bias=0)
        block_kwargs = dict(emb_channels=emb_channels, channels_per_head=64, dropout=dropout,
                            init=init, init_zero=init_zero)               # 64 channels per attention head
        # Mapping: timestep + (class) embedding feeds the per-block AdaGN scale/shift.
        self.map_noise = PositionalEmbedding(num_channels=model_channels)
        self.map_layer0 = Linear(model_channels, emb_channels, **init)
        self.map_layer1 = Linear(emb_channels, emb_channels, **init)
        self.map_label = Linear(label_dim, emb_channels, bias=False, init_mode='kaiming_normal',
                                init_weight=np.sqrt(label_dim)) if label_dim else None
        # Encoder / decoder built from UNetBlocks; attention enabled at attn_resolutions (32,16,8);
        # up/down use BigGAN-style residual resampling. (Block bodies omitted for brevity.)
        ...

    def forward(self, x, noise_labels, class_labels, augment_labels=None):
        emb = self.map_noise(noise_labels)
        emb = silu(self.map_layer0(emb)); emb = self.map_layer1(emb)
        if self.map_label is not None:
            emb = emb + self.map_label(class_labels)                      # class flows through AdaGN too
        emb = silu(emb)
        skips = []
        for block in self.enc.values():
            x = block(x, emb) if isinstance(block, UNetBlock) else block(x)
            skips.append(x)
        for block in self.dec.values():
            if x.shape[1] != block.in_channels:
                x = torch.cat([x, skips.pop()], dim=1)
            x = block(x, emb)                                             # UNetBlock applies AdaGN(h, emb)
        return self.out_conv(silu(self.out_norm(x)))
```
