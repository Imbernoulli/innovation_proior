The pyramid pooling lifts mIoU to **77.85** — the biggest jump on the ladder, $+5.60$ over the local FCN head — so context was the dominant problem. But look closely at *how* the pyramid gets its context and what that costs. Each level adaptive-average-pools the feature map down to a tiny grid — $1\times1$, $2\times2$, $3\times3$, $6\times6$ — then upsamples the summary back. Pooling to a $2\times2$ grid means averaging everything in each quadrant into one number; the *within-region* spatial detail of the context is gone, and when I bilinearly broadcast that quadrant summary back to full resolution it is a smooth blanket that knows nothing about where the boundaries inside the quadrant are. The pyramid buys global reach by *destroying resolution* to get it, then upsampling, so the context is right but spatially coarse — and that coarseness shows up most at object boundaries. A second, separate problem hides behind it: regardless of how I get context, the whole head still predicts at output stride $8$ and bilinearly upsamples by $8$, so the finest boundaries — thin poles, the exact silhouette of a pedestrian — are blurry no matter how good the context, because an $8\times$ bilinear upsample of an OS-$8$ map cannot draw a sharp edge. The U-Net solved that with a full decoder, which I dropped when I moved to the dilated backbone; perhaps I gave up too much.

I propose **DeepLabV3+**, which fixes both, with a *depthwise-separable atrous spatial pyramid pooling* module for context and a single lightweight decoder skip for boundaries. For the first problem, I want context at multiple scales but applied *densely*, at the feature map's own resolution, not via pooling. The tool I already have for "see farther without downsampling" is dilation — it is what makes the backbone output stride $8$ in the first place. So instead of pooling into grids, I run *several $3\times3$ convolutions in parallel over the same OS-$8$ feature map, each at a different dilation rate*: a rate-$12$ $3\times3$ conv samples context from a window roughly $24$ pixels across, rate-$24$ from $\sim48$, rate-$36$ from $\sim72$ — all at full feature resolution, no pooling, every output pixel keeping its location. That is a *pyramid of dilations* rather than a pyramid of pools — atrous spatial pyramid pooling. I add a rate-$1$ branch for the purely-local view and a single image-level global-average-pool branch (the one genuinely-global summary, broadcast back), so I do not lose the whole-scene cue that the pooling pyramid's $1\times1$ level gave me, then concatenate all branches and fuse with a $1\times1$ conv. The reason dilation-instead-of-pooling helps is precisely the within-region detail pooling threw away: a rate-$12$ atrous conv sees far-apart context but still produces a distinct value at *every* pixel, so context near a boundary is computed from that pixel's actual wide neighborhood rather than a single regional average. The trade is that very large dilation rates degenerate — when the rate is so big that the $3\times3$ kernel's three taps fall mostly outside the feature map, the conv reduces toward a $1\times1$ (the center tap dominates) — which is exactly why I keep the global-pool branch as the honest whole-image summary rather than pushing the rate ever higher. Three parallel $3\times3$ atrous convs over a $2048$-channel OS-$8$ map is heavy, so I cut the cost with *depthwise-separable* convolution: factor each $3\times3$ atrous conv into a depthwise $3\times3$ (one filter per channel — the spatial/atrous part) followed by a pointwise $1\times1$ (the cross-channel mixing). Same receptive field, a fraction of the multiply-adds, and the dilation lives entirely in the cheap depthwise part.

For the second problem — the boundaries — rather than reinstate the whole U-Net I add a *minimal* decoder, just enough to recover them. The boundary information lives in the *shallow* backbone stage: the OS-$4$ feature, sharp but low-level. So I take the rich ASPP context feature, bilinearly upsample it from OS-$8$ to OS-$4$ — a $2\times$ jump, not $8\times$ — grab the OS-$4$ backbone feature, run a $1\times1$ conv to squeeze its channel count down (it is a low-level feature; I do not want it to dominate the fusion, so it is reduced to $48$ channels), concatenate the two, and refine with a couple of $3\times3$ separable convs. Then the classifier, then a final $\times4$ upsample to full resolution. It is the U-Net skip idea, but a *single* skip at OS-$4$ bolted onto the dilated backbone — semantics from the context module, boundary detail from the shallow skip, fused once. The bet against $77.85$ is modest and honest: the atrous pyramid gathers the same multi-scale context as the pooling pyramid but without averaging away within-region resolution, so context near boundaries is sharper, and the single OS-$4$ skip recovers the fine boundaries an $8\times$ upsample blurs. The pyramid pooling already captured most of the available context gain, so the headroom here is smaller and comes mostly from boundary classes — poles, fences, thin riders — rather than a wholesale jump; the channel-reducing $1\times1$ on the skip and the separable-conv fusion are the hedges against an over-low-level skip sharpening noise.

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

Backbone R-50-D8; ASPP dilations `(1, 12, 24, 36)` plus the global-pool branch; the OS-$4$ decoder skip is channel-reduced to $48$ before fusion; final $\times4$ bilinear upsample.
