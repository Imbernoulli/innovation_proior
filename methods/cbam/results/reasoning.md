Let me think about where there's still slack in a convolutional network, because depth, width, and cardinality have all been pushed hard and they all do the same kind of thing — add raw capacity that gets applied uniformly to every channel and every pixel. But a feature map isn't uniformly useful. Some channels are firing on patterns that don't matter for this particular image; some spatial locations are just background clutter. A convolution computes over all of them with equal weight. So there's a different lever: let the network look at its own intermediate feature map and decide, adaptively per input, which parts to amplify and which to damp. That's attention, and I want it as a tiny module I can staple into every conv block without redesigning the backbone.

The thing I'm refining is a feature map F with C channels over an H×W grid. What does it even mean to "attend" here? There are two genuinely different axes. One is *channel*: each channel is essentially a feature detector, so asking "which channels matter" is asking *what* is meaningful in this image. The other is *spatial*: asking "which locations matter" is asking *where* the informative region is. A convolution mixes both axes at once, so if I want to refine its output I should be able to refine along both. Let me design a piece for each.

Start with channel attention, because there's already a known way to do it and I want to understand exactly what it does and does not capture. The squeeze-and-excitation idea is: collapse each channel's whole H×W map down to a single number by global average pooling, giving a C-vector that summarizes "how much is this channel responding overall"; push that vector through a small bottleneck MLP — reduce to C/r channels with a ReLU, then expand back to C — and squash with a sigmoid to get a per-channel gate in [0,1]; multiply it back into the channels. The bottleneck with reduction ratio r keeps the parameter cost tiny, and r=16 is a sensible default. The whole gate is a function of one statistic per channel: the spatial mean. So the question I should actually press on is whether the mean is a sufficient summary of "is this channel detecting something I care about."

Let me try to break it. Take a 4×4 channel A that is uniformly 0.5 everywhere — a diffuse, mid-level response with no structure. Its mean is 0.5. Now take a 4×4 channel B that is 0 everywhere except a single cell at 8.0 — one sharp, localized peak, the rest silence. B's mean is 8/16 = 0.5 as well. So A and B have *identical* average-pooled descriptors, 0.5 each, and the SE gate — which sees only that number — must assign them the same importance. But these are exactly opposite situations: B has a strong, distinctive activation that very plausibly marks a specific object part, while A is just texture-less background fill. Averaging destroyed the one thing that distinguishes them. That's a real failure of the summary, not a stylistic complaint — I just exhibited two channels the gate provably cannot tell apart.

What statistic *would* separate them? The max. max(A) = 0.5, max(B) = 8.0. The peak immediately distinguishes the localized strong detector from the diffuse one. So max pooling gathers a genuinely different clue: not "how much does this channel respond on average" but "does this channel have a strong activation somewhere." And the two are complementary in the other direction too — a channel that is 0.5 everywhere and a channel that is 0.5 at one cell and 0 elsewhere share a max of 0.5 but have means 0.5 vs 0.03, so the mean separates *those*. Neither statistic dominates; each sees what the other misses. So rather than choose, use both: run global average pooling and global max pooling over the spatial dimension to get two C-vectors, the average-pooled descriptor and the max-pooled descriptor.

Now, do I give each its own MLP? That doubles parameters and, more importantly, the two descriptors live in the same channel space — they're two views of the same channels' importance — so it makes sense to score them with the *same* function. So push both through one *shared* MLP (the same W_0 reduce, ReLU, W_1 expand), then merge by element-wise summing the two output vectors before the sigmoid. Summing in logit space lets each pooling vote on each channel's gate, with the sigmoid combining the votes. So:

  M_c(F) = σ( MLP(AvgPool(F)) + MLP(MaxPool(F)) ) = σ( W_1·ReLU(W_0·F^c_avg) + W_1·ReLU(W_0·F^c_max) ),

with W_0 ∈ R^{C/r × C}, W_1 ∈ R^{C × C/r}, shared across both inputs, output a C×1×1 gate. Apply it as F' = M_c(F) ⊗ F, broadcasting the C-vector across all spatial positions.

I want to make sure I haven't quietly broken backward compatibility with the gate I started from. Set pool_types to ['avg'] only — drop the max branch — and the sum has a single term: M_c(F) = σ(MLP(AvgPool(F))) = σ(W_1·ReLU(W_0·F^c_avg)). That is *literally* the SE channel gate, character for character. So the max branch is a strict addition: with it off I recover the baseline exactly, and with it on I supply the peak statistic the baseline was blind to in the A-vs-B example above. Good — this is a superset of SE, not a replacement that might be worse on cases SE already handled.

Now spatial attention, and I'll build it symmetric to the channel module — same design philosophy, opposite axis. For channel attention I squeezed over space to get a per-channel statistic; for spatial attention I want a per-location statistic, so I squeeze over *channels*. At each spatial position, summarize the C channels into a compact descriptor. By the same argument I just made — and the A-vs-B check tells me it's the same argument, just transposed — use both pooling types: average over the channel axis and max over the channel axis, each producing a single H×W map. Pooling along the channel axis is a known way to highlight informative regions — collapse the channels and where the network is responding strongly across features lights up; the max map flags locations with at least one sharply-firing detector, the mean map flags locations with broad cross-channel response. Concatenate the avg-over-channels map and the max-over-channels map into a 2×H×W descriptor.

From that 2-channel descriptor I need a single H×W attention map that says where to emphasize. A spatial gate should consider a neighborhood — "is this a coherent informative region" — not just one pixel, so a convolution with a reasonably large receptive field fits. Apply one convolution layer over the 2-channel map producing 1 channel, with a 7×7 kernel so it sees enough context, then sigmoid:

  M_s(F) = σ( f^{7×7}( [AvgPool_chan(F); MaxPool_chan(F)] ) ),

giving a 1×H×W gate. Apply it as F'' = M_s(F') ⊗ F', broadcasting the single map across all channels.

Let me sanity-check the shapes end to end on a concrete tensor before I commit, because broadcasting bugs here would be silent. Take F of shape (1, 8, 16, 16) and r=16. Channel branch: avg-pool over 16×16 gives (1,8,1,1), flatten to (1,8); MLP maps 8 → 8//16 = 0 ... that's a problem — integer division sends C/r to zero when C < r. With C=8, r=16, the bottleneck width is 0 and the Linear layer is degenerate. So the reduction ratio isn't free: it presumes the gated layer has at least r channels. In a real ResNet the gated feature maps have 64/128/256/512 channels (or 256+ at the bottleneck-expansion points where I'd insert this), so C ≥ r=16 always holds and C/r is 4/8/16/32 — fine. I'll keep r=16 but note it's only valid where C ≥ r, which is everywhere I actually place the module. Redo the trace with C=64: avg and max pool each give (1,64), MLP 64→4→64 gives (1,64), the two sum to (1,64), sigmoid, unsqueeze to (1,64,1,1), expand to (1,64,16,16), multiply F — output (1,64,16,16), same shape. Spatial branch on that: ChannelPool maxes and means over dim=1 giving two (1,1,16,16) maps, concat to (1,2,16,16), 7×7 conv pad=3 to (1,1,16,16), sigmoid, multiply — output (1,64,16,16). Shapes close. Same as input. Good — it's genuinely a same-shape refinement.

Now I have two complementary gates — one says *what*, one says *where* — and I have to decide how they relate. Two natural arrangements. Parallel: compute both from the same input and combine (e.g. add the two and sigmoid into one 3-D mask). Sequential: apply one, then compute the other on the already-refined map. Parallel is appealing as "one 3-D attention map," but it forces the two attentions to be inferred independently from the unrefined features and then reconciled. Sequential lets the second module operate on features the first has already cleaned up: the channel gate suppresses irrelevant feature types first, and the spatial gate then computes its cross-channel pooling on a map where the junk channels are already damped, so its descriptor isn't polluted by them. I can't decide this from the shape check — both arrangements typecheck — and I can't compute the win without training, so I'll commit to sequential on this cooperation argument and flag it as the thing to verify by ablation: I expect channel-then-spatial to beat the parallel combination, but that's a hypothesis the ImageNet sweep has to confirm, not something I can settle on paper.

Within sequential, which order? Here I do have an argument with teeth rather than just a preference. Channel attention is *global* — one scalar per channel, applied identically at every location — while spatial attention is *local* — a per-location map. If I run spatial first, I damp some locations, and *then* the channel gate pools (avg and max) over space including those already-damped locations, so the channel statistic is computed on a spatially-distorted field. If I run channel first, the spatial pooling later runs over channels at full spatial resolution, and the channel reweighting it sees is a clean global rescaling. Applying the global "what" filter first and then localizing "where" composes more cleanly. I'd still want the ablation to confirm channel-first edges out spatial-first, but the composition argument points that way, so: channel attention, then spatial attention.

Putting it together, given F ∈ R^{C×H×W}:

  F' = M_c(F) ⊗ F      (refine channels: what)
  F'' = M_s(F') ⊗ F'   (refine locations: where)

with broadcasting on each multiply. F'' is the refined output, same shape as F — I verified that on the (1,64,16,16) trace — so I can drop this block into any conv stage. Concretely, apply it to a residual block's conv output before the skip addition, where it costs almost nothing in parameters (one small shared MLP plus one 7×7-conv-to-1-channel) yet re-weights the whole feature map.

Let me write it, grounded in a clean module structure.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelGate(nn.Module):
    # M_c(F) = sigma( MLP(AvgPool(F)) + MLP(MaxPool(F)) ), shared MLP, applied per channel.
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super().__init__()
        self.mlp = nn.Sequential(                       # shared bottleneck MLP: C -> C/r -> C
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels),
        )
        self.pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:               # both avg (smooth) and max (peak) descriptors
            if pool_type == 'avg':
                pooled = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
            elif pool_type == 'max':
                pooled = F.max_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
            att = self.mlp(pooled)
            channel_att_sum = att if channel_att_sum is None else channel_att_sum + att  # sum votes
        scale = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale                                # broadcast over spatial dims

class ChannelPool(nn.Module):
    # squeeze the channel axis with max and mean -> 2 x H x W
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)

class SpatialGate(nn.Module):
    # M_s(F) = sigma( f^{7x7}( [AvgPool_chan(F); MaxPool_chan(F)] ) ), applied per location.
    def __init__(self):
        super().__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = nn.Sequential(                   # 7x7 conv: 2 -> 1, no ReLU
            nn.Conv2d(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, bias=False),
            nn.BatchNorm2d(1, eps=1e-5, momentum=0.01),
        )

    def forward(self, x):
        x_compress = self.compress(x)
        scale = torch.sigmoid(self.spatial(x_compress))  # 1 x H x W, broadcast over channels
        return x * scale

class CBAM(nn.Module):
    # channel attention, then spatial attention (sequential, channel-first)
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super().__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial = no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()

    def forward(self, x):
        x_out = self.ChannelGate(x)                     # F' = M_c(F) . F
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)             # F'' = M_s(F') . F'
        return x_out
```

The causal chain: depth/width/cardinality scale capacity but spend it uniformly, so the open lever is input-adaptive feature selection — attention — as a cheap plug-in. A feature map has two axes worth refining: channel ("what") and spatial ("where"). For channels, the average-pool-then-bottleneck-MLP gate is blind to localized peaks — I showed two 4×4 channels (uniform-0.5 vs single-8.0-spike) with the same mean 0.5 that it cannot distinguish, whereas their maxes 0.5 vs 8.0 do — so I add a parallel max-pool descriptor through the *same* shared MLP and sum the two before the sigmoid; dropping the max branch reduces the gate exactly to SE, so it's a strict superset. For space, I mirror the design on the opposite axis: pool over channels with avg and max, concatenate the two maps, and run a single 7×7 conv to a sigmoid gate so a neighborhood decides where to focus. The two gates are complementary; I apply them sequentially, channel first (global) then spatial (local) on the already-refined map, because that composition is cleaner — though the sequential-beats-parallel and channel-first-beats-spatial-first claims are hypotheses I'd settle by ablation on ImageNet, not on paper. The result, F'' = M_s(M_c(F)⊗F) ⊗ (M_c(F)⊗F), is a same-shape refined map (verified by tracing shapes through a (1,64,16,16) input): CBAM, droppable into every conv block.
