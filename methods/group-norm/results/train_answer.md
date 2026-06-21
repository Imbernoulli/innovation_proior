The normalization layer in a deep convnet is the component that keeps the scale of hidden activations stable enough for gradient descent to propagate through many layers. The dominant version normalizes each channel by its mean and variance computed over the mini-batch, but that makes the whole network depend on batch size. When the per-device batch is forced down to two or four images by high-resolution inputs, video, or detection backbones, the sample mean and variance become noisy and biased, and accuracy collapses. On ResNet-50 / ImageNet the error rises from about 23.6 % at batch 32 to 34.7 % at batch 2. The same batch dependence also creates a train/test mismatch, because inference has to use frozen running averages instead of the current batch, and those frozen statistics can be wrong after transfer or under non-i.i.d. sampling. So the real target is a normalization whose statistic is estimated inside a single sample, never across the batch.

The existing batch-free choices sit at two unsatisfying extremes. Layer normalization pools all channels and spatial positions of one image into a single mean and variance; that is too coarse for convolutional features, whose channels detect edges, colors, textures, and frequencies with genuinely different scales. Instance normalization normalizes each channel of each image independently; that removes the batch dependence but discards all cross-channel structure, leaving each channel blind to its correlated neighbors. What is needed is a granularity between "all channels" and "one channel" that normalizes correlated channels together without collapsing unlike channels into the same statistic.

The method is Group Normalization, or GN. It splits the C channels into G contiguous groups of C / G channels each. For each sample and each group, it computes one mean and one variance over that group's channels and all spatial positions, then standardizes every position in the group by those statistics and finally applies a learnable per-channel affine transform. In the same unified notation as the other normalization layers, the set of positions that share one statistic is S_i = { k : k_N = i_N, floor(k_C / (C/G)) = floor(i_C / (C/G)) }. The condition k_N = i_N keeps the computation inside one image, so the batch dimension never enters; GN is the same function in training, evaluation, transfer, or any batch size. Because the spatial axes are always part of S_i, the estimate is based on thousands of values even on a single image, so it is stable without the batch. Setting G = 1 recovers layer normalization and G = C recovers instance normalization, so GN literally interpolates between the two batch-free extremes; the interior works better than either boundary.

The practical default is 32 groups, and the accuracy is a broad plateau across a wide range of G. Grouping mirrors the structure of classical vision descriptors such as HOG and SIFT, which normalize orientation histograms within local blocks, and it matches the observation that convolutional channels form correlated families. After the shared per-group standardization, the per-channel affine lets each channel recover its own scale, so representational power is preserved. The layer also generalizes to video with no extra code, because the reduction simply includes the temporal axis along with the spatial axes inside one sample.

```python
import torch
import torch.nn as nn


class GroupNorm(nn.Module):
    """Group Normalization: normalize each group of channels inside each sample.

    Drop-in replacement for BatchNorm2d that is independent of batch size.
    """

    def __init__(self, num_groups, num_channels, eps=1e-5, affine=True):
        super().__init__()
        assert num_channels % num_groups == 0
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        if affine:
            self.weight = nn.Parameter(torch.ones(num_channels))
            self.bias = nn.Parameter(torch.zeros(num_channels))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x):
        N, C = x.shape[0], x.shape[1]
        spatial = x.shape[2:]                       # (H, W) for images, (T, H, W) for video
        x = x.reshape(N, self.num_groups, C // self.num_groups, *spatial)
        dims = tuple(range(2, x.dim()))             # within-group channels + all spatial axes
        mean = x.mean(dim=dims, keepdim=True)
        var = x.var(dim=dims, keepdim=True, unbiased=False)
        x = (x - mean) / torch.sqrt(var + self.eps)
        x = x.reshape(N, C, *spatial)
        if self.affine:
            shape = [1, C] + [1] * len(spatial)
            x = x * self.weight.reshape(shape) + self.bias.reshape(shape)
        return x
```
