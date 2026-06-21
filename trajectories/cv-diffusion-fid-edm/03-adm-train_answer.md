Two rungs of work brought the cosine-schedule, learned-variance recipe to a CIFAR-10 FID of $3.19$ (with the long-chain $L_{\text{simple}}$ baseline reaching $\sim 2.90$), and every one of those gains came from redesigning the *training* of a denoiser whose *shape* I never touched. The schedule, the target, the loss weighting, the variance machinery — all reworked around a fixed U-Net backbone. But the backbone itself, the network $F_\theta$ that actually does the regression, is still the competent-but-unremarkable U-Net I inherited: a few residual blocks per resolution, self-attention at a single middle resolution, the timestep embedding simply added into the residual stream. The limitation that forces this rung is that I have been optimizing how I train a network without ever asking whether that network is the right shape to train. So I set the training knobs aside and treat this purely as an architecture question: what should a diffusion denoiser's backbone look like to give the optimizer the most to work with? This rung changes no training-design knob — preconditioning, loss weighting, $\sigma$-distribution, and augmentation are all left exactly as the previous rung handed them — and rebuilds only $F_\theta$.

I propose **ADM**, a heavily-tuned denoiser U-Net built from five deliberate changes to that backbone, four of which carry the gain and one of which I keep but explicitly do not rely on.

The first is *attention placement*. A denoiser at high noise has to invent global structure — it sees something close to pure noise and must produce a coherent layout — and global structure is exactly what stacks of local convolutions cannot assemble on their own. That is the job of self-attention, which mixes information across the whole feature map. Putting attention at only the single middle resolution lets the network do that global mixing at one spatial scale and nowhere else. So I enable self-attention at *several* resolutions — $32\times32$, $16\times16$, and $8\times8$ — so the network can integrate context at coarse, medium, and fine scales. This is aimed squarely at the high-noise regime where coarse coherence is decided, which is the regime FID is most sensitive to.

The second is *how attention is sized*. Multi-head attention splits the channels into heads, and there are two natural ways to scale across resolutions of very different dimensionality: fix the number of heads and let head width grow with the channel count, or fix the head width and let the head count grow. What governs attention quality is the dimensionality each head reasons over, not how many heads there are, so I fix the *width per head* at **64 channels** and let the number of heads fall out of the channel count at each resolution. Every head then operates at a consistent, well-conditioned dimensionality regardless of where it sits in the pyramid, matching how head width is the held-constant quantity in the transformer literature.

The third change is the most consequential, and it is about *how the timestep and class label are injected*. Adding the embedding into the residual stream — what the inherited backbone does — can only express an additive shift. But a denoiser's behavior should depend on the noise level *multiplicatively* as well: the noise level should rescale how strongly each feature channel responds, and a pure addition cannot represent a per-channel gain. So I replace additive injection with **adaptive group normalization (AdaGN)**: after group-normalizing the activations, I apply a learned per-channel scale and shift that are *predicted from the timestep+class embedding*,
$$\mathrm{AdaGN}(\mathbf{h},\mathbf{e}) = \mathbf{e}_s \cdot \mathrm{GroupNorm}(\mathbf{h}) + \mathbf{e}_b,$$
where $(\mathbf{e}_s, \mathbf{e}_b)$ is a linear projection of the embedding $\mathbf{e}$. The conditioning can now both *gate* (scale) and *bias* (shift) each channel as a function of noise level — the right interface between "how noisy am I" and "how should I transform features." This is also what most directly improves the *conditional* denoiser, because the class signal flows through the same scale/shift path and can modulate features rather than merely being summed in.

The fourth is the *resampling blocks*. Down- and up-sampling inside a U-Net are where information is most easily lost or aliased, and a plain strided convolution or nearest-neighbor resize is a crude operator there. I replace them with the BigGAN-style residual up/down blocks proven out in large-scale image generators, which carry a learned skip across the resolution change so the network resamples through a residual path rather than a lossy bottleneck. This stabilizes the feature pyramid that the multi-resolution attention then operates on.

The fifth is *depth versus width* at a roughly fixed parameter budget. A deeper, thinner net composes more nonlinear transformations and extends the denoiser's representational reach; trading some width for depth at fixed budget does improve FID. But it costs wall-clock — more sequential layers, slower steps — and the substrate has to be trainable at scale, not merely optimal per-parameter, so I note depth as a genuine lever and keep the practical configuration at variable width rather than maximizing depth.

There is one modification I try and back off from: rescaling the residual connections by $1/\sqrt{2}$. The motivation is signal-variance control — keeping activation magnitudes from compounding through the residual stream, which is standard in score-SDE and BigGAN — so I add it and make the block support it. But on this task it is the single change in the set that does *not* improve FID, while every other change does. So it stays available and unrelied-upon; the win comes from the multi-resolution attention, the 64-channel heads, AdaGN, and the BigGAN resampling, not from the residual rescale.

What this buys, it buys as a *stronger substrate* rather than a better training recipe. The four training-design decisions are untouched — the preconditioning is still the inherited $\boldsymbol{\epsilon}$-style mapping, the loss is the chain-derived weighting, the $\sigma$-distribution is the schedule's, the variance machinery is as left. So this is a context rung: it fixes the denoiser at a strong setting — wider attention coverage, 64-channel heads, AdaGN conditioning, BigGAN resampling, depth held in reserve — and in doing so sharpens the open problem. The architecture is no longer the bottleneck, which means whatever FID headroom remains must live in *how the denoiser is trained*, not in *what it is*. The training-design redesign that follows runs on top of exactly this backbone.

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
