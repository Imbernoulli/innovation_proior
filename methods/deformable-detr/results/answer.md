# Deformable DETR

## Problem
The Transformer-based set-prediction detector (DETR) is elegant — end-to-end, NMS-free — but converges
~10× slower than standard detectors and detects small objects poorly. Both deficits stem from one
cause: Transformer attention over a feature map treats every spatial location as a key, which (a) is
O(H²W²C), forbidding the high-resolution / multi-scale maps small objects need, and (b) starts from a
near-uniform attention A_mqk ≈ 1/N_k whose diluted gradient takes very long to sharpen.

## Key idea
Replace dense attention with **(multi-scale) deformable attention**: each query attends only to a small
set of K learned sampling points around a reference point, on each of L feature levels — predicting both
the sampling offsets and the attention weights directly from the query (no query-key dot product) and
reading feature values by bilinear interpolation. This makes attention linear in feature-map size in
the encoder, independent of it in decoder cross-attention, and concentrates each gradient on a handful
of points, so it converges fast and affords multi-scale maps.

## Multi-scale deformable attention
For a query with content z_q and normalized reference point p̂_q ∈ [0,1]² over L feature maps {x^l}:

  MSDeformAttn(z_q, p̂_q, {x^l}) = Σ_{m=1}^{M} W_m [ Σ_{l=1}^{L} Σ_{k=1}^{K} A_mlqk · W'_m · x^l(φ_l(p̂_q) + Δp_mlqk) ],

- M heads, L levels, K points per level per head; φ_l rescales p̂_q to level l's grid.
- Δp_mlqk (offsets) and A_mlqk (weights) are produced by a single linear layer on z_q with 3MLK channels:
  2MLK channels are the offsets, MLK channels go through softmax over the LK points (Σ_{l,k} A_mlqk = 1).
- x^l(·) read by bilinear interpolation. Defaults M = 8, K = 4, L = 4, C = 256.
- Complexity O(2N_q C² + min(HWC², N_q K C²)): O(HWC²) in the encoder (N_q = HW), O(NKC²) in decoder
  cross-attention (N_q = N), independent of HW.
- Reduces to deformable convolution when L = K = 1 and W'_m = I; reduces to full attention if the K
  points cover all locations.

## Architecture
- **Multi-scale features (L = 4):** ResNet stages C3, C4, C5 each via 1×1 conv to C = 256, plus a 4th
  level from a 3×3 stride-2 conv on C5. No FPN top-down pathway — deformable attention exchanges
  cross-scale information itself.
- **Encoder:** self-attention replaced by MSDeformAttn; query = each feature-map pixel, reference point
  = itself. A learned **scale-level embedding** e_l is added (with the sinusoidal positional embedding)
  so a pixel knows which level it is on.
- **Decoder:** keep ordinary self-attention among the N object queries; replace cross-attention with
  MSDeformAttn. Each query's reference point p̂_q is predicted from its embedding by a linear layer + sigmoid.
- **Box prediction relative to the reference** (in logit space):
  b̂_q = ( σ(b_qx + σ⁻¹(p̂_qx)), σ(b_qy + σ⁻¹(p̂_qy)), σ(b_qw), σ(b_qh) ),
  so attention sampling and the predicted box share a reference, easing optimization.

## Initialization (important)
Zero the offset/weight projection weights (init depends only on biases). Bias the weights so
A_mlqk = 1/(LK) (uniform); bias the offsets so the M = 8 heads point in 8 directions at radii k = 1..K
— a symmetric local receptive-field prior that training then deforms.

## Variants
- **Iterative box refinement (D = 6 layers):** layer d refines the previous box,
  b̂^d_q = σ(Δb^d_q + σ⁻¹(b̂^{d-1}_q)) componentwise; per-layer (unshared) heads; initial box at the
  reference with w = h = 0.1. Gradient blocked at σ⁻¹(b̂^{d-1}) (RAFT-style stabilization). Layer d
  samples around the previous box: reference = its center, offsets modulated by its (w, h).
- **Two-stage:** an encoder-only first stage predicts a box per pixel (3-layer FFN + binary fg/bg
  classifier; per-level base scale s = 0.05), trained with the Hungarian loss; top boxes become region
  proposals (no NMS), fed to the decoder as initial boxes with their coordinates as query positions.

## Training
Bipartite-matching set loss (one prediction per object → no NMS). Classification uses **focal loss**
(weight 2); N = 300 object queries. Adam, base lr 2e-4 (β₁ = 0.9, β₂ = 0.999, weight decay 1e-4), 50
epochs with 10× decay at epoch 40. The reference-point and sampling-offset projections use 10× smaller
lr (2e-5), since they control the sampling geometry.

## Code
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
