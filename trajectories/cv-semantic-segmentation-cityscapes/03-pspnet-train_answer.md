The pretrained dilated backbone with a local conv head reaches **72.25 mIoU**, a $+3.15$ gain over the from-scratch encoder–decoder. Now I want to find *where* it is still wrong, because that tells me what to add. The head's receptive field is a couple of $3\times3$ convs over the output-stride-$8$ feature map, so each pixel's decision is informed by a modest neighborhood — a few tens of pixels of context. For much of Cityscapes that is plenty, but the errors that survive are the ones where the *local* evidence is genuinely ambiguous and only the *whole scene* resolves it. A gray planar surface, seen through a small window, can be road or sidewalk — the texture is identical; what disambiguates it is the global layout, where it sits relative to the buildings and the curb. A tall narrow vertical can be a pole, a tree trunk, or a sign post — locally identical, separable only from what is around it across the frame. The classic failure is a patch of "boat" floating in a parking lot, or "building" on a stretch the rest of the scene makes obviously a wall: the head decided each pixel from too small a window and never consulted the global context that would have vetoed the mismatch. The backbone's deepest units have a large receptive field in principle, but the *effective* receptive field of a deep CNN is much smaller than its theoretical one, so scene-level context never reaches the per-pixel decision.

I propose the **Pyramid Pooling Module (PSPNet)**: summarize the feature map at a *pyramid* of region scales and feed all of them back to every pixel. The bluntest fix would be to global-average-pool the whole feature map to one $C$-dimensional vector — a scene descriptor — and broadcast it back to every position, so each pixel's decision can be conditioned on "what kind of scene is this overall." That is a real and cheap fix, but it is *too* blunt: a single global vector collapses the entire frame into one summary, discarding all spatial structure of the context. A driving scene has gross spatial organization — sky up top, road at the bottom, buildings flanking — and a single average throws it away. So instead I pool at *several* region scales at once. I adaptive-average-pool the OS-$8$ feature into a $1\times1$ grid (the single global vector, coarsest context), a $2\times2$ grid (four quadrant summaries), a $3\times3$ grid, and a $6\times6$ grid (finer sub-region summaries). Each grid cell is the average of the features over its region — a context summary at that region's scale — so the stack of grids is context at a pyramid of scales, from "the whole image" down to "this sixth of the image." A pixel that needs the whole-scene cue gets it from the $1\times1$ level; a pixel that needs to know "I am in the lower-left region where road usually is" gets it from the $2\times2$ or $3\times3$ level. This is adaptive in a way the per-patch baseline never was: the network leans on whichever scale of context actually disambiguates a given class, rather than being locked to one window size.

The mechanics make the context dense again before fusion. For each pooling scale I adaptive-average-pool the OS-$8$ feature down to that grid, run a $1\times1$ conv to reduce its channels (so the four levels together do not blow up the channel count), then *bilinearly upsample each pooled map back to full feature-map resolution* — so a $2\times2$ summary becomes a full-resolution map where every pixel carries the summary of its quadrant. I concatenate all four upsampled context maps with the original feature and run a $3\times3$ bottleneck conv to fuse local features with multi-scale global context, then the usual $1\times1$ classifier and $\times8$ upsample. The two load-bearing choices: the $1/2/3/6$ ladder of scales, and average rather than max pooling. One level (global) is structureless, as argued; too many fine levels would approach the local conv head I already have, adding cost without new information — the $1/2/3/6$ rungs span the useful range, coarse enough to bring genuinely global context, fine enough to keep some spatial organization of it. And *average* pooling because I want each region's *typical* content as a context descriptor — the prior of what the region usually is — not the single most-activated location that max pooling would return. The bet against $72.25$ is that the surviving errors are exactly the context-starved ones — road/sidewalk confusion, floating mislabeled patches, locally-ambiguous verticals — and that piping multi-scale global context into every pixel's decision clears them. The one risk is that a bilinear broadcast of a coarse summary could smear context across a real boundary, but the fusion bottleneck sees the sharp local feature alongside the broadcast context and can down-weight the latter at boundaries.

```python
class PPM(nn.ModuleList):
    """Pool the feature into several region grids, reduce, upsample back."""
    def __init__(self, pool_scales, in_channels, channels, conv_cfg, norm_cfg,
                 act_cfg, align_corners, **kwargs):
        super().__init__()
        self.pool_scales = pool_scales           # (1, 2, 3, 6)
        self.align_corners = align_corners
        for pool_scale in pool_scales:
            self.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(pool_scale),
                ConvModule(in_channels, channels, 1, conv_cfg=conv_cfg,
                           norm_cfg=norm_cfg, act_cfg=act_cfg, **kwargs)))

    def forward(self, x):
        ppm_outs = []
        for ppm in self:
            ppm_out = ppm(x)
            upsampled_ppm_out = resize(ppm_out, size=x.size()[2:],
                                       mode='bilinear',
                                       align_corners=self.align_corners)
            ppm_outs.append(upsampled_ppm_out)   # region summary -> full res
        return ppm_outs

class PSPHead(BaseDecodeHead):
    def __init__(self, pool_scales=(1, 2, 3, 6), **kwargs):
        super().__init__(**kwargs)
        self.psp_modules = PPM(pool_scales, self.in_channels, self.channels,
                               conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg,
                               act_cfg=self.act_cfg, align_corners=self.align_corners)
        self.bottleneck = ConvModule(
            self.in_channels + len(pool_scales) * self.channels, self.channels,
            3, padding=1, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)

    def _forward_feature(self, inputs):
        x = self._transform_inputs(inputs)
        psp_outs = [x]
        psp_outs.extend(self.psp_modules(x))     # + multi-scale global context
        psp_outs = torch.cat(psp_outs, dim=1)
        return self.bottleneck(psp_outs)         # fuse local + global

    def forward(self, inputs):
        return self.cls_seg(self._forward_feature(inputs))
```

Backbone R-50-D8 (output stride $8$); pool scales `(1, 2, 3, 6)`; the fused feature goes through the $1\times1$ classifier and an $\times8$ bilinear upsample.
