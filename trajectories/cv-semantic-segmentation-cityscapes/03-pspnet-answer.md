**Problem (from step 2).** The FCN head reaches 72.25 but decides each pixel from a local window (a few
3×3 convs over the OS-8 map). The errors that survive are context-starved: road vs. sidewalk (same
local texture, different scene layout), floating mislabeled patches, locally-identical verticals. The
deep CNN's *effective* receptive field is much smaller than its theoretical one, so global, scene-level
context never reaches the per-pixel decision.

**Key idea — Pyramid Pooling Module (PSPNet).** Summarize the feature map at a *pyramid* of region
scales and feed all of them back to every pixel. Adaptive-average-pool the OS-8 feature into 1×1, 2×2,
3×3, and 6×6 grids — context from "the whole image" down to "this sixth of the image" — reduce each
with a 1×1 conv, bilinearly upsample each back to full feature resolution (so every pixel carries its
region's summary), concatenate all four with the original feature, and fuse with a 3×3 bottleneck
before the 1×1 classifier.

**Why it works.** A single global average is global but structureless (it discards the gross spatial
organization of a driving scene); a local head has structure but no global reach. The pyramid gives
both: the 1×1 level supplies whole-scene context, the 2×2/3×3/6×6 levels supply spatially-organized
sub-region context, and the network leans on whichever scale disambiguates a given class. Average (not
max) pooling captures each region's *typical* content as a prior. This clears exactly the context
errors a local head leaves.

**Change / code.** The pyramid pooling module and the context-fusion head.

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

Backbone R-50-D8 (output stride 8); pool scales `(1, 2, 3, 6)`; the fused feature goes through the 1×1
classifier and an ×8 bilinear upsample.
