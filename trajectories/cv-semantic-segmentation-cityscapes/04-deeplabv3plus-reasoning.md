77.85 — the pyramid pooling fixed the context errors, and it's the biggest jump yet. So context was
the dominant problem. But look closely at *how* the pyramid gets its context and what that costs. Each
pyramid level adaptive-average-pools the feature map down to a tiny grid — 1×1, 2×2, 3×3, 6×6 — then
upsamples the summary back. Pooling to a 2×2 grid means I've averaged everything in each quadrant into
one number; the *within-region* spatial detail of the context is gone, and when I bilinearly broadcast
that quadrant summary back to full resolution it's a smooth blanket that knows nothing about where the
boundaries inside the quadrant are. So the pyramid buys global reach by *destroying resolution* to get
it, then upsampling. The context is right but spatially coarse, and that coarseness shows up most at
object boundaries.

Two problems are tangled together here, and I should pull them apart. First: can I gather multi-scale
context *without* pooling resolution away? Second: regardless of how I get context, the whole head
still produces its prediction at output stride 8 and then bilinearly upsamples by 8 to full
resolution — so the finest boundaries, the thin poles, the exact silhouette of a pedestrian, are blurry
no matter how good the context is, because an 8× bilinear upsample of an OS-8 map cannot draw a sharp
edge. The U-Net solved that with a full decoder; I dropped the decoder when I moved to the dilated
backbone. Maybe I gave up too much there.

Take the first problem. I want context at multiple scales, but applied *densely* at the feature map's
own resolution, not via pooling. The tool I already have for "see farther without downsampling" is
dilation — it's what makes the backbone output stride 8 in the first place. So instead of pooling into
grids, run *several convolutions in parallel over the same OS-8 feature map, each at a different
dilation rate*. A rate-6 3×3 conv samples context from a window ~12 pixels across; rate-12 from ~24;
rate-18 from ~36 — all at full feature resolution, no pooling, every output pixel keeping its location.
That's a *pyramid of dilations* rather than a pyramid of pools — atrous spatial pyramid pooling. Add a
rate-1 1×1 conv for the purely-local branch, and a single image-level global-average-pool branch (the
one genuinely-global summary, broadcast back) so I don't lose the whole-scene cue the pyramid pooling's
1×1 level gave me. Concatenate all the branches and fuse with a 1×1 conv. Now I have multi-scale
context *and* I've kept the resolution while gathering it — each branch is a dense per-pixel feature.

The reason dilation-instead-of-pooling helps is precisely the within-region detail that pooling threw
away: a rate-12 atrous conv sees far-apart context but still produces a distinct value at *every*
pixel, so context near a boundary is computed from that pixel's actual wide neighborhood rather than a
single regional average. The trade is that very large dilation rates start to degenerate — when the
rate gets so big that the 3×3 kernel's three taps fall mostly outside the feature map, the conv reduces
toward a 1×1 (the center tap dominates), which is why I keep the global-pool branch as the honest
whole-image summary rather than pushing the dilation rate ever higher.

There's a cost I should pay attention to: three parallel 3×3 atrous convs over a 2048-channel OS-8 map
is heavy. I can cut it sharply with *depthwise-separable* convolution — factor each 3×3 atrous conv
into a depthwise 3×3 (one filter per channel, the spatial/atrous part) followed by a pointwise 1×1
(the cross-channel mixing). Same receptive field, a fraction of the multiply-adds, and the dilation
lives entirely in the cheap depthwise part. So the atrous pyramid becomes a *separable* atrous pyramid.

Now the second problem — the boundaries. The context module gives me a strong OS-8 feature, but I'm
still one 8× bilinear jump from the labels. Rather than reinstate the whole U-Net, I add a *minimal*
decoder, just enough to recover boundaries. The boundary information lives in the *shallow* backbone
stage — the OS-4 feature, which is sharp but low-level. So: take the rich context feature, bilinearly
upsample it from OS-8 to OS-4 (a 2× jump, not 8×), grab the OS-4 backbone feature, run a 1×1 conv on it
to squeeze its channel count down (it's a low-level feature, I don't want it to dominate the fusion),
concatenate the two, and refine with a couple of 3×3 separable convs. Then the classifier, then a final
4× upsample to full resolution. It's the U-Net skip idea, but a *single* skip at OS-4, bolted onto the
dilated backbone — semantics from the context module, boundary detail from the shallow skip, fused once.

Let me write the head. The separable atrous pyramid (the dilation>1 branches become depthwise-separable):

```python
class DepthwiseSeparableASPPModule(ASPPModule):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for i, dilation in enumerate(self.dilations):
            if dilation > 1:
                self[i] = DepthwiseSeparableConvModule(
                    self.in_channels, self.channels, 3,
                    dilation=dilation, padding=dilation,
                    norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)
```

and the head: build the context feature from the image-pool branch plus the atrous branches, then the
single OS-4 decoder skip:

```python
def forward(self, inputs):
    x = self._transform_inputs(inputs)
    aspp_outs = [resize(self.image_pool(x), size=x.size()[2:],   # global branch
                        mode='bilinear', align_corners=self.align_corners)]
    aspp_outs.extend(self.aspp_modules(x))      # parallel atrous branches (rates 1,12,24,36)
    aspp_outs = torch.cat(aspp_outs, dim=1)
    output = self.bottleneck(aspp_outs)         # fuse the pyramid -> context feature
    if self.c1_bottleneck is not None:
        c1_output = self.c1_bottleneck(inputs[0])   # low-level OS-4 skip, channel-reduced
        output = resize(output, size=c1_output.shape[2:],   # context up OS-8 -> OS-4
                        mode='bilinear', align_corners=self.align_corners)
        output = torch.cat([output, c1_output], dim=1)
    output = self.sep_bottleneck(output)        # fuse context + detail (separable convs)
    return self.cls_seg(output)
```

The prediction, against 77.85. Two things should each add a little. The atrous pyramid gathers the same
multi-scale context as the pooling pyramid but *without* averaging away within-region resolution, so
the context near boundaries is sharper. And the single OS-4 decoder skip lets the prediction recover
the fine boundaries that an 8× bilinear upsample blurs. The bet is that keeping resolution while
gathering context, plus one cheap decoder skip for the boundaries, pushes Cityscapes mIoU above the
pooling pyramid. The risk is diminishing returns: the pyramid pooling already captured most of the
available context gain, so the headroom here is smaller and comes mostly from boundary classes (poles,
fences, thin riders) rather than a wholesale jump — and if the OS-4 skip is too low-level it could
sharpen noise. The channel-reducing 1×1 on the skip and the separable-conv fusion are the hedges. The
concrete change is the separable atrous spatial pyramid plus the single OS-4 decoder; the full head is
in the answer.
