72.25 with the pretrained dilated backbone and the local conv head. Now I want to find out *where* it's
still wrong, because that's what tells me what to add. Think about the head's receptive field. It's a
couple of 3×3 convs over the output-stride-8 feature map. Each output pixel's decision is informed by a
modest neighborhood around it — a few tens of pixels of context, give or take, through the dilated
backbone and the head convs. For a lot of Cityscapes that's plenty. But the errors that survive are the
ones where the *local* evidence is genuinely ambiguous and only the *whole scene* resolves it.

Concretely: a gray planar surface, looked at through a small window, can be road or sidewalk — the
texture is the same; what disambiguates it is the global layout, where it sits relative to the
buildings and the curb. A tall narrow vertical thing can be a pole, a tree trunk, or the leg of a
traffic-sign post — locally identical, separable only if you know what's around it across the frame.
The classic failure of a purely-local segmenter is a patch of "boat" floating in a parking lot, or
"building" predicted on a stretch that the rest of the scene makes obviously a wall: the head decided
each pixel from too small a window and never consulted the global context that would have vetoed the
mismatch. The backbone's deepest units *do* have a large receptive field in principle, but the
*effective* receptive field of a deep CNN is much smaller than its theoretical one, and the head on top
only sees a local window — so global, scene-level context is in practice not reaching the per-pixel
decision.

So the deficiency is global context. I need each pixel's prediction to be able to see a summary of the
*entire* feature map, not just its neighborhood. How do I summarize the whole map and feed it back to
every location? The bluntest version: global-average-pool the feature map to a single C-dimensional
vector — the scene descriptor — and broadcast it back to every spatial position, concatenated with the
local features. Now every pixel's decision can be conditioned on "what kind of scene is this overall."
That's a real fix, and it's cheap. But it's *too* blunt: a single global vector collapses the entire
frame into one summary, losing all spatial structure of the context. A driving scene has gross spatial
organization — sky up top, road at the bottom, buildings flanking — and a single global average throws
that away. I want context that is global in reach but not totally structureless.

The move is to pool at *several* region scales at once, not just one. Pool the feature map into a 1×1
grid (the single global vector — coarsest context), and *also* into a 2×2 grid (four region summaries,
quadrants of the scene), a 3×3 grid, and a 6×6 grid (finer sub-region summaries). Each grid cell is the
average of the features over its region, so it's a context summary at that region's scale. Stack of
grids = context at a *pyramid* of scales, from "the whole image" down to "this sixth of the image." A
pixel that needs the whole-scene cue gets it from the 1×1 level; a pixel that needs to know "I'm in the
lower-left region where road usually is" gets it from the 2×2 or 3×3 level. This is adaptive in the
sense the per-patch baseline never was: the network can lean on whichever scale of context actually
disambiguates a given class, instead of being locked to one window size.

The mechanics. For each pooling scale, adaptive-average-pool the OS-8 feature map down to that grid
(1×1, 2×2, 3×3, 6×6), run a 1×1 conv to reduce its channels (so the four pyramid levels together don't
blow up the channel count), then *upsample each pooled map back to the full feature-map resolution* by
bilinear interpolation — so a 2×2 summary becomes a full-resolution map where every pixel carries the
summary of its quadrant. Concatenate all four upsampled context maps with the original feature map, and
run a 3×3 bottleneck conv to fuse local features with multi-scale global context. Then the usual 1×1
classifier and upsample.

Let me write the pyramid pooling module:

```python
class PPM(nn.ModuleList):
    def __init__(self, pool_scales, in_channels, channels, **kw):
        super().__init__()
        self.pool_scales = pool_scales        # (1, 2, 3, 6)
        for pool_scale in pool_scales:
            self.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(pool_scale),     # summarize each region
                ConvModule(in_channels, channels, 1, **kw)))  # reduce channels

    def forward(self, x):
        ppm_outs = []
        for ppm in self:
            ppm_out = ppm(x)
            upsampled = resize(ppm_out, size=x.size()[2:],
                               mode='bilinear', align_corners=self.align_corners)
            ppm_outs.append(upsampled)        # broadcast region summary to full res
        return ppm_outs
```

and the head: concatenate the input feature with all the pyramid context maps, then a 3×3 bottleneck:

```python
def _forward_feature(self, inputs):
    x = self._transform_inputs(inputs)        # OS=8 deep feature
    psp_outs = [x]
    psp_outs.extend(self.psp_modules(x))      # + 1x1, 2x2, 3x3, 6x6 context, upsampled
    psp_outs = torch.cat(psp_outs, dim=1)
    feats = self.bottleneck(psp_outs)         # fuse local + multi-scale global
    return feats
```

Why these four scales and not, say, just 1×1, or a dozen levels? One level (global) is structureless,
as I argued; too many fine levels would approach the local conv head I already have and add cost
without new information. The 1/2/3/6 ladder spans the useful range — whole image, halves, thirds,
sixths — coarsely enough to bring genuinely global context, finely enough to keep some spatial
organization of it. And average pooling (not max) because I want each region's *typical* content as a
context descriptor, not its single most-activated location; the prior is what the region usually is.

The prediction, against 72.25. The errors I expect to clear are exactly the context-starved ones: the
road/sidewalk confusion, the floating mislabeled patches, the locally-ambiguous verticals — anywhere
the right answer was determined by the scene rather than the neighborhood. The bet is that piping
multi-scale global context into every pixel's decision lifts the Cityscapes mIoU well above the local
head, because those context errors are a big share of what's left. The risk is double-counting context
the backbone already half-captures, or that the bilinear-broadcast of a coarse summary smears
context across a real boundary — but the fusion bottleneck sees the sharp local feature alongside the
broadcast context and can down-weight the latter at boundaries. The concrete change is the pyramid
pooling module feeding multi-scale context into the fusion bottleneck; the full head is in the answer.
