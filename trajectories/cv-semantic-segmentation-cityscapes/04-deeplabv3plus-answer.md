**Problem (from step 3).** Pyramid pooling reaches 77.85 but gets its context by *pooling resolution
away* — averaging each region into a tiny grid and bilinearly broadcasting it back, so within-region
detail is lost. And the prediction is still produced at output stride 8 and upsampled ×8, so the finest
boundaries stay blurry no matter how good the context.

**Key idea — DeepLabV3+ (separable ASPP + decoder).** Two changes. (1) **Atrous spatial pyramid
pooling (ASPP):** gather multi-scale context *densely*, without pooling, by running several 3×3 convs
in parallel over the OS-8 feature at different dilation rates (plus a 1×1 local branch and one
global-average-pool branch), so every pixel keeps its location while seeing far-apart context;
concatenate and fuse. Make the dilated branches **depthwise-separable** (depthwise 3×3 + pointwise 1×1)
to cut cost. (2) A **lightweight decoder:** upsample the context feature OS-8→OS-4 (a 2× jump, not 8×),
concatenate the channel-reduced low-level OS-4 backbone feature (one skip), refine with separable 3×3
convs, then upsample ×4 to full resolution.

**Why it works.** A rate-12 atrous conv sees wide context but still emits a distinct value at every
pixel, so context near a boundary comes from that pixel's real neighborhood rather than a regional
average — sharper context than pooling. (Very large rates degenerate toward 1×1, hence the explicit
global-pool branch for the honest whole-image cue.) The single OS-4 skip reinstates the U-Net idea
minimally: semantics from ASPP, boundary detail from the shallow feature, fused once — recovering the
thin/boundary classes the ×8 upsample blurs. Separable convs make the pyramid affordable.

**Change / code.** The depthwise-separable ASPP module and the decoder head.

```python
class DepthwiseSeparableASPPModule(ASPPModule):
    """ASPP with depthwise-separable convs for the dilated branches."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for i, dilation in enumerate(self.dilations):
            if dilation > 1:
                self[i] = DepthwiseSeparableConvModule(
                    self.in_channels, self.channels, 3,
                    dilation=dilation, padding=dilation,
                    norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)

class DepthwiseSeparableASPPHead(ASPPHead):
    def __init__(self, c1_in_channels, c1_channels, **kwargs):
        super().__init__(**kwargs)
        self.aspp_modules = DepthwiseSeparableASPPModule(
            dilations=self.dilations, in_channels=self.in_channels,
            channels=self.channels, conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)
        self.c1_bottleneck = ConvModule(c1_in_channels, c1_channels, 1,
                                        norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)
        self.sep_bottleneck = nn.Sequential(
            DepthwiseSeparableConvModule(self.channels + c1_channels, self.channels,
                                         3, padding=1, norm_cfg=self.norm_cfg,
                                         act_cfg=self.act_cfg),
            DepthwiseSeparableConvModule(self.channels, self.channels, 3, padding=1,
                                         norm_cfg=self.norm_cfg, act_cfg=self.act_cfg))

    def forward(self, inputs):
        x = self._transform_inputs(inputs)
        aspp_outs = [resize(self.image_pool(x), size=x.size()[2:],     # global branch
                            mode='bilinear', align_corners=self.align_corners)]
        aspp_outs.extend(self.aspp_modules(x))      # parallel atrous branches
        aspp_outs = torch.cat(aspp_outs, dim=1)
        output = self.bottleneck(aspp_outs)         # -> context feature
        if self.c1_bottleneck is not None:
            c1_output = self.c1_bottleneck(inputs[0])   # low-level OS-4 skip
            output = resize(output, size=c1_output.shape[2:],   # OS-8 -> OS-4
                            mode='bilinear', align_corners=self.align_corners)
            output = torch.cat([output, c1_output], dim=1)
        output = self.sep_bottleneck(output)        # fuse context + detail
        return self.cls_seg(output)
```

Backbone R-50-D8; ASPP dilations `(1, 12, 24, 36)` plus the global-pool branch; the OS-4 decoder skip
is channel-reduced to 48 before fusion; final ×4 bilinear upsample.
