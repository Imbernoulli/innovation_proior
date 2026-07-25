The set-prediction detector DETR showed that object detection can be end-to-end, anchor-free, and NMS-free, but in practice it is hard to use: it converges roughly ten times slower than standard detectors, and its small-object accuracy is weak. Both problems trace back to the same place. DETR's Transformer attention treats every spatial location in a feature map as a key, so the encoder self-attention cost scales quadratically with feature-map area. That alone makes high-resolution or multi-scale feature maps unaffordable, which is exactly what small objects need. At the same time, with standard initialization the attention logits are near-zero, so each query starts by attending almost uniformly over tens of thousands of keys. The gradient that should teach the query to focus on a few relevant locations is diluted across all those keys, and it takes an enormous number of epochs for the attention to sharpen. The fix, therefore, is not a separate training trick for convergence or a separate architectural add-on for scale; it is to redesign the attention operator itself so that each query looks at only a small, learned set of locations instead of everywhere.

The method is Deformable DETR. Its core is multi-scale deformable attention. Each query has a content feature and a normalized 2-d reference point in the image. Instead of dot-producting against every key, the query predicts both a small set of 2-d sampling offsets around its reference point and an attention weight for each sample. The values at those fractional locations are read off by bilinear interpolation, and the result is a weighted sum. Formally, for a query with content z_q and reference point p̂_q, attending over L multi-scale feature maps with M heads and K sampling points per level, the output is the sum over heads, levels, and points of W_m [ A_mlqk W'_m x^l(φ_l(p̂_q) + Δp_mlqk) ], where the offsets Δp_mlqk and weights A_mlqk are produced by a single linear projection on z_q with 3MLK outputs. The weights are normalized by a softmax over the LK points per head. Because the number of attended locations is tiny compared with the feature-map area, the encoder cost drops from quadratic in spatial size to linear, and decoder cross-attention becomes independent of spatial size altogether. More importantly, the gradient is now concentrated on a handful of sampled points, so attention can sharpen quickly instead of crawling out of a uniform initialization.

The architecture keeps the set-prediction formulation intact. The backbone is ResNet, and four multi-scale feature maps are built from stages C3, C4, C5 via 1×1 convolutions to 256 channels plus a fourth level from a 3×3 stride-2 convolution on C5. Notably, no feature-pyramid top-down pathway is needed, because cross-scale exchange happens inside the deformable attention itself. The encoder replaces self-attention with multi-scale deformable attention, where each feature-map pixel queries with itself as its reference point. A learned scale-level embedding is added to the sinusoidal positional encoding so each pixel knows which pyramid level it belongs to. The decoder keeps ordinary multi-head self-attention among the object queries so they can still coordinate and suppress duplicates, but replaces cross-attention into the feature maps with multi-scale deformable attention. Each object query's reference point is predicted from its embedding by a linear layer followed by a sigmoid. The box head then predicts an offset relative to that reference point in logit space, so when the correction is zero the box center equals the reference point; this ties where the attention samples and what the head predicts, removing optimization slack.

Initialization is part of the design and matters a lot. The projection weights that produce offsets and weights are initialized to zero, so at start-up the behavior depends only on the biases and is query-independent. The attention-weight biases are set so every sampled point gets equal weight 1/(LK). The offset biases are set so the eight heads point in the eight compass directions at radii 1 through K, giving a small symmetric local receptive field that training then deforms. Two optional refinements are natural once the attention is cheap enough. Iterative box refinement lets every decoder layer correct the previous layer's box, with the reference point updated to the current box center and the sampling offsets modulated by the current box size; gradients are blocked through the previous box to keep the chain stable. A two-stage variant adds an encoder-only first stage that predicts a box per pixel, takes the top-scoring proposals as initial boxes, and feeds them into the decoder, grounding the object queries in actual image content instead of generic learned slots. Training uses the same bipartite-matching set loss as DETR, but classification is focal loss, the number of object queries is increased to 300, and the schedule shrinks to 50 epochs with a 10× learning-rate drop at epoch 40.

```python
import torch
from torch import nn
import torch.nn.functional as F


def ms_deform_attn_core(value, value_spatial_shapes, sampling_locations, attention_weights):
    """value: (N, sum(HW), M, D); sampling_locations: (N, Lq, M, L, K, 2) in [0,1];
    attention_weights: (N, Lq, M, L, K). Bilinear-sample LK points/query and weight-sum."""
    N, _, M, D = value.shape
    _, Lq, _, L, K, _ = sampling_locations.shape
    value_list = value.split([H * W for H, W in value_spatial_shapes], dim=1)
    sampling_grids = 2 * sampling_locations - 1                       # [0,1] -> [-1,1]
    out_levels = []
    for lid, (H, W) in enumerate(value_spatial_shapes):
        v_l = value_list[lid].flatten(2).transpose(1, 2).reshape(N * M, D, H, W)
        grid_l = sampling_grids[:, :, :, lid].transpose(1, 2).flatten(0, 1)
        out_levels.append(F.grid_sample(v_l, grid_l, mode='bilinear',
                                        padding_mode='zeros', align_corners=False))
    attn = attention_weights.transpose(1, 2).reshape(N * M, 1, Lq, L * K)
    out = (torch.stack(out_levels, dim=-2).flatten(-2) * attn).sum(-1)
    return out.view(N, M * D, Lq).transpose(1, 2)


class MSDeformAttn(nn.Module):
    def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=4):
        super().__init__()
        self.M, self.L, self.K, self.d_model = n_heads, n_levels, n_points, d_model
        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.constant_(self.sampling_offsets.weight, 0.)
        thetas = torch.arange(self.M) * (2 * torch.pi / self.M)
        grid = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid = (grid / grid.abs().max(-1, keepdim=True)[0]).view(self.M, 1, 1, 2)
        grid = grid.repeat(1, self.L, self.K, 1)
        for k in range(self.K):
            grid[:, :, k] *= (k + 1)
        self.sampling_offsets.bias = nn.Parameter(grid.view(-1))
        nn.init.constant_(self.attention_weights.weight, 0.)
        nn.init.constant_(self.attention_weights.bias, 0.)               # softmax -> 1/(LK)

    def forward(self, query, reference_points, value, value_spatial_shapes):
        N, Lq, _ = query.shape
        value = self.value_proj(value).view(N, -1, self.M, self.d_model // self.M)
        offsets = self.sampling_offsets(query).view(N, Lq, self.M, self.L, self.K, 2)
        weights = self.attention_weights(query).view(N, Lq, self.M, self.L * self.K)
        weights = weights.softmax(-1).view(N, Lq, self.M, self.L, self.K)
        norm = torch.stack([torch.tensor([W, H]) for H, W in value_spatial_shapes], 0)
        loc = reference_points[:, :, None, :, None, :] + offsets / norm[None, None, None, :, None, :]
        out = ms_deform_attn_core(value, value_spatial_shapes, loc, weights)
        return self.output_proj(out)
```
