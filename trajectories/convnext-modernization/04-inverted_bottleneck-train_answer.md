Both predictions from the last rung landed: depthwise convolution alone cost
1.23 points (79.51 to 78.28) while nearly halving FLOPs, and widening to 96
channels more than recovered it, 78.28 to 80.50. But the FLOPs line deserves
an honest look before calling that an unqualified win — 5.27G, above the
4.5G Swin-T reference I've been tracking throughout, and above every prior
point on this ladder. The accuracy gain is real, but the budget discipline
that held since rung 2 has slipped, and I want a move that recovers it
without giving back accuracy, for reasons that live inside the current
block's own shape rather than a general desire to save compute.

The bottleneck pattern, inherited unchanged from vanilla ResNet through
every rewrite so far, is wide-at-the-boundary, narrow-in-the-middle: input
and output channel count is wide, the residual branch reduces down for its
spatial mixing, then expands back. That means every downsampling block's
shortcut — the 1x1 conv that reshapes the identity path when resolution and
channel count change between stages — connects wide-to-wide, since the
shortcut has to match the block's wide boundary on both ends. That is the
single most expensive 1x1 conv at each stage transition, purely because of
which end of the block is wide.

The documented alternative is to invert it: narrow-wide-narrow instead of
wide-narrow-wide. Expand up from the boundary width with the first 1x1
conv, do the spatial mixing at that expanded width, project back down to
the narrow boundary with the second 1x1 conv. One motivation is efficiency
precedent from mobile-oriented ConvNets, which use roughly this shape with a
4x expansion factor. But there's a second, independent reason to like this
shape specifically now: it's also the shape of a Transformer feedforward
sublayer — narrow residual-stream width, project up 4x, nonlinearity,
project back down to narrow. Two lines of reasoning, mobile efficiency and
Transformer-block structure, converge on the same shape, which is a
stronger signal than either alone would be; if the shape only showed up in
mobile networks I might read it as latency-specific over-optimization
irrelevant here, and if it only showed up in Transformers I might read it
as a sequence-model convention with no reason to transfer. That it appears
independently under two different pressures is why I'm adopting it now.

Concretely: keep the depthwise conv from rung 3, but relocate it inside the
inverted shape. The block boundary — what the shortcut connects — moves to
the current bottleneck's narrower "planes" width; the first 1x1 conv
expands 4x from there, matching the Transformer MLP's expansion ratio since
I have no separate reason yet to pick a different multiplier; the depthwise
conv now operates at this expanded width; the second 1x1 conv projects back
down to the boundary width. The downsampling shortcut, which used to
connect wide-to-wide, now connects narrow-to-narrow.

Two forces should move FLOPs in opposite directions and I don't have a
confident prior on which wins outright, though I have a directional guess.
Against savings: the depthwise conv now operates at 4x the channel count it
used to, and depthwise FLOPs scale linearly with channels, so that single
layer gets meaningfully more expensive. Toward savings: every stage-
transition shortcut, now narrow-to-narrow instead of wide-to-wide, and
there are several of these, each processing full spatial resolution at that
stage. My guess is the shortcut savings dominate, since the shortcuts touch
every spatial position while the depthwise conv's added cost is confined to
one already-cheap operation — so I expect FLOPs to come down from 5.27G,
hopefully toward the ~4.5G band, though I won't know the exact landing
point without running it. On accuracy, this is a pure rearrangement of
where the same operations sit relative to the shortcut, not a change in
what operations exist or the block's total capacity, so I don't expect a
large swing; if anything a mild positive, since the narrower shortcut is a
cleaner identity path with less to project, but I hold that loosely as a
secondary effect. The number that will tell me whether this rung was worth
doing is FLOPs relative to 5.27G: a real reduction without giving back the
80.50% accuracy point would mean the rearrangement bought back budget
discipline for free.

```python
# rung 4: inverted bottleneck -- narrow -> wide(4x) -> narrow, depthwise at the wide point.
import torch.nn as nn

class InvertedBottleneck(nn.Module):
    """narrow -> 1x1 expand 4x -> depthwise -> 1x1 project -> narrow (+ shortcut)."""
    expand_ratio = 4

    def __init__(self, dim, stride=1, downsample=None):
        super().__init__()
        hidden = dim * self.expand_ratio
        self.pw_expand = nn.Conv2d(dim, hidden, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden)
        self.dw = nn.Conv2d(hidden, hidden, 3, stride=stride, padding=1,
                             groups=hidden, bias=False)          # depthwise, now at 4x width
        self.bn2 = nn.BatchNorm2d(hidden)
        self.pw_project = nn.Conv2d(hidden, dim, 1, bias=False)  # back to narrow boundary
        self.bn3 = nn.BatchNorm2d(dim)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample     # now narrow -> narrow, cheaper than rung 3's shortcut

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.pw_expand(x)))
        out = self.relu(self.bn2(self.dw(out)))
        out = self.bn3(self.pw_project(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)

def make_stage(dim_in, dim_out, blocks, stride=1):
    downsample = None
    if stride != 1 or dim_in != dim_out:
        downsample = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 1, stride, bias=False),   # narrow -> narrow now
            nn.BatchNorm2d(dim_out),
        )
    layers = [InvertedBottleneck(dim_out, stride=stride, downsample=downsample)
              if dim_in == dim_out else
              InvertedBottleneck(dim_in, stride=stride, downsample=downsample)]
    for _ in range(1, blocks):
        layers.append(InvertedBottleneck(dim_out))
    return nn.Sequential(*layers)

dims = [96, 192, 384, 768]      # matches rung 3's width-96 base
depths = [3, 3, 9, 3]           # unchanged since rung 2
```
