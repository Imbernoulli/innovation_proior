Convolutional networks have been pushed mainly through three capacity levers: depth, width, and cardinality. All of them add representational power, but they spend that power uniformly, treating every channel and every spatial location of a feature map as equally important. For any particular input that is clearly wasteful: some channels respond to patterns that are irrelevant, and some spatial regions are just background clutter. A standard convolution cannot suppress those uninformative parts because it blends information across channels and space with fixed weights.

Attention offers a way out, but the existing choices are either too narrow or too expensive. Squeeze-and-Excitation (SE) is the closest successful template: it performs channel attention by global average pooling followed by a small bottleneck MLP and a sigmoid gate. It is lightweight and effective, yet it has two important weaknesses. First, average pooling over the whole spatial extent of a channel produces a smooth summary in which every location contributes equally; a channel with one sharp, distinctive peak is treated the same as a channel with diffuse, mediocre responses everywhere. Second, SE attends only to channels, not to locations, so it can reweight "what" is present but not "where" the network should focus inside the map.

I propose a Convolutional Block Attention Module, or CBAM. It is a small, plug-and-play refinement unit that takes an intermediate feature map and returns a reweighted map of the same shape. CBAM reasons about attention along two complementary axes: channel and spatial. Given a feature map F of shape C × H × W, it first refines the channels, then refines the spatial locations on top of that already-refined map.

For channel attention, I compute two per-channel descriptors by pooling over the spatial dimensions. Global average pooling gives the smooth overall response of each channel, preserving the SE-style signal. Global max pooling gives the strongest response anywhere in the channel, capturing distinctive peaks that average pooling blurs away. Both descriptors live in the same C-dimensional space, so they can share a single bottleneck MLP that reduces to C/r channels, applies ReLU, and expands back to C. The two MLP outputs are added together in logit space before a sigmoid, so each pooling type votes on the final per-channel gate. Dropping the max branch recovers the original SE channel gate, while keeping both yields a richer attention signal for almost no extra cost. The gate is broadcast across all spatial positions and multiplied back into the feature map.

For spatial attention, I apply the same philosophy along the opposite axis. I squeeze the channel dimension at each spatial location by computing both the average and the maximum over channels, producing two single-channel maps of size H × W. These are concatenated into a two-channel descriptor and passed through a 7 × 7 convolution that produces one attention map, followed by batch normalization and a sigmoid. The large kernel lets the gate depend on a local neighborhood rather than a single pixel, so it can highlight coherent informative regions and suppress background clutter. This map is broadcast across all channels and multiplied into the features.

The two gates could be combined in parallel, but arranging them sequentially works better: the channel gate first filters which feature detectors matter for this input, and the spatial gate then localizes attention within the cleaner feature map. Within sequential arrangements, channel-first also slightly outperforms spatial-first, matching the intuition that deciding what to attend to globally should come before deciding where to attend locally. The final module is therefore F' = channel_gate(F) ⊗ F followed by F'' = spatial_gate(F') ⊗ F', with broadcasting on each multiplication. CBAM can be inserted at the end of the convolutional branch inside any residual block, just before the skip connection is added. Its overhead is tiny: one shared bottleneck MLP plus one 7 × 7 convolution, with the reduction ratio r typically set to 16.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ChannelGate(nn.Module):
    """Shared bottleneck MLP over avg and max spatial-pooled descriptors."""
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super().__init__()
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels),
        )
        self.pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                pooled = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
            elif pool_type == 'max':
                pooled = F.max_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
            att = self.mlp(pooled)
            channel_att_sum = att if channel_att_sum is None else channel_att_sum + att
        scale = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat(
            (torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)),
            dim=1,
        )


class SpatialGate(nn.Module):
    """7x7 conv over avg/max channel-pooled maps."""
    def __init__(self):
        super().__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, bias=False),
            nn.BatchNorm2d(1, eps=1e-5, momentum=0.01),
        )

    def forward(self, x):
        x_compress = self.compress(x)
        scale = torch.sigmoid(self.spatial(x_compress))
        return x * scale


class CBAM(nn.Module):
    """Channel attention, then spatial attention."""
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super().__init__()
        self.channel_gate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial = no_spatial
        if not no_spatial:
            self.spatial_gate = SpatialGate()

    def forward(self, x):
        x_out = self.channel_gate(x)
        if not self.no_spatial:
            x_out = self.spatial_gate(x_out)
        return x_out
```
