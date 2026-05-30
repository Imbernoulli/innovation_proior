Let me think about where there's still slack in a convolutional network, because depth, width, and cardinality have all been pushed hard and they all do the same kind of thing — add raw capacity that gets applied uniformly to every channel and every pixel. But a feature map isn't uniformly useful. Some channels are firing on patterns that don't matter for this particular image; some spatial locations are just background clutter. A convolution computes over all of them with equal weight. So there's a different lever: let the network look at its own intermediate feature map and decide, adaptively per input, which parts to amplify and which to damp. That's attention, and I want it as a tiny module I can staple into every conv block without redesigning the backbone.

The thing I'm refining is a feature map F with C channels over an H×W grid. What does it even mean to "attend" here? There are two genuinely different axes. One is *channel*: each channel is essentially a feature detector, so asking "which channels matter" is asking *what* is meaningful in this image. The other is *spatial*: asking "which locations matter" is asking *where* the informative region is. A convolution mixes both axes at once, so if I want to refine its output I should be able to refine along both. Let me design a piece for each.

Start with channel attention, because there's already a known way to do it and I can see exactly where it falls short. The squeeze-and-excitation idea is: collapse each channel's whole H×W map down to a single number by global average pooling, giving a C-vector that summarizes "how much is this channel responding overall"; push that vector through a small bottleneck MLP — reduce to C/r channels with a ReLU, then expand back to C — and squash with a sigmoid to get a per-channel gate in [0,1]; multiply it back into the channels. The bottleneck with reduction ratio r keeps the parameter cost tiny, and r=16 is a sensible default. Fine. But stare at the summarization step: averaging the whole channel. Average pooling is smooth and dense, every spatial position contributes to the gradient, but it *blurs*. A channel that has one sharp, distinctive peak — exactly the kind of localized evidence that signals a specific object part — gets averaged down toward a channel with diffuse mid-level response everywhere. The average can't tell those apart, and yet for inferring "is this channel detecting something distinctive" they're very different.

So average pooling throws away the peak. What captures the peak? Max pooling — the single strongest response in the channel. Max pooling gathers a different clue: not "how much does this channel respond on average" but "does this channel have a strong distinctive activation somewhere." Those two statistics are complementary, not redundant. Rather than choose, use both. Run global average pooling and global max pooling over the spatial dimension to get two C-vectors, the average-pooled descriptor and the max-pooled descriptor.

Now, do I give each its own MLP? That doubles parameters and, more importantly, the two descriptors live in the same channel space — they're two views of the same channels' importance — so it makes sense to score them with the *same* function. So push both through one *shared* MLP (the same W_0 reduce, ReLU, W_1 expand), then merge by element-wise summing the two output vectors before the sigmoid. Summing in logit space lets each pooling vote on each channel's gate, with the sigmoid combining the votes. So:

  M_c(F) = σ( MLP(AvgPool(F)) + MLP(MaxPool(F)) ) = σ( W_1·ReLU(W_0·F^c_avg) + W_1·ReLU(W_0·F^c_max) ),

with W_0 ∈ R^{C/r × C}, W_1 ∈ R^{C × C/r}, shared across both inputs, output a C×1×1 gate. When I drop the max branch, this collapses exactly to the average-only channel gate — so I haven't broken anything; I've added the missing peak information. Apply it as F' = M_c(F) ⊗ F, broadcasting the C-vector across all spatial positions.

Now spatial attention, and I'll build it symmetric to the channel module — same design philosophy, opposite axis. For channel attention I squeezed over space to get a per-channel statistic; for spatial attention I want a per-location statistic, so I squeeze over *channels*. At each spatial position, summarize the C channels into a compact descriptor. By the same argument I just made, use both pooling types: average over the channel axis and max over the channel axis, each producing a single H×W map. Pooling along the channel axis is a known way to highlight informative regions — collapse the channels and where the network is responding strongly across features lights up. Concatenate the avg-over-channels map and the max-over-channels map into a 2×H×W descriptor.

From that 2-channel descriptor I need a single H×W attention map that says where to emphasize. A spatial gate should consider a neighborhood — "is this a coherent informative region" — not just one pixel, so a convolution with a reasonably large receptive field fits. Apply one convolution layer over the 2-channel map producing 1 channel, with a 7×7 kernel so it sees enough context, then sigmoid:

  M_s(F) = σ( f^{7×7}( [AvgPool_channel(F); MaxPool_channel(F)] ) ),

giving a 1×H×W gate. Apply it as F'' = M_s(F') ⊗ F', broadcasting the single map across all channels.

Now I have two complementary gates — one says *what*, one says *where* — and I have to decide how they relate. Two natural arrangements. Parallel: compute both from the same input and combine (e.g. add the two and sigmoid into one 3-D mask). Sequential: apply one, then compute the other on the already-refined map. Parallel is appealing as "one 3-D attention map," but it forces the two attentions to be inferred independently from the unrefined features and then reconciled. Sequential lets the second module operate on features the first has already cleaned up, so they cooperate rather than compete — the channel gate suppresses irrelevant feature types first, and the spatial gate then localizes within the better-conditioned map. That cooperation should win, and it does: sequential beats parallel. Within sequential, which first? Channel attention is *global* (one value per channel, applied everywhere) while spatial attention is *local* (per location); applying the global "what" filter first and then localizing "where" is the more natural order, and channel-first comes out slightly ahead of spatial-first. So the module is: channel attention, then spatial attention.

Putting it together, given F ∈ R^{C×H×W}:

  F' = M_c(F) ⊗ F      (refine channels: what)
  F'' = M_s(F') ⊗ F'   (refine locations: where)

with broadcasting on each multiply. F'' is the refined output, same shape as F, so I can drop this block into any conv stage — concretely, apply it to a residual block's conv output before the skip addition, where it costs almost nothing in parameters (one small shared MLP plus one 7×7-conv-to-1-channel) yet re-weights the whole feature map.

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

The causal chain: depth/width/cardinality scale capacity but spend it uniformly, so the open lever is input-adaptive feature selection — attention — as a cheap plug-in. A feature map has two axes worth refining: channel ("what") and spatial ("where"). For channels, the existing average-pool-then-bottleneck-MLP gate blurs away distinctive peaks, so I add a parallel max-pool descriptor through the *same* shared MLP and sum the two before the sigmoid, recovering peak information at no real cost (and reducing to the average-only gate if the max branch is removed). For space, I mirror the design on the opposite axis: pool over channels with avg and max, concatenate the two maps, and run a single 7×7 conv to a sigmoid gate so a neighborhood decides where to focus. The two gates are complementary, and applying them sequentially — channel first (global), then spatial (local) on the already-refined map — lets them cooperate, beating a parallel combination. The result, F'' = M_s(M_c(F)⊗F) ⊗ (M_c(F)⊗F), is a same-shape refined map: CBAM, droppable into every conv block.
