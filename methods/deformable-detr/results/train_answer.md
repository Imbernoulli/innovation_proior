The set-prediction detector DETR showed that object detection can be end-to-end, anchor-free, and NMS-free, but in practice it is hard to use: it converges roughly ten times slower than standard detectors, and its small-object accuracy is weak. Both problems trace back to the same place. DETR's Transformer attention treats every spatial location in a feature map as a key, so the encoder self-attention cost scales quadratically with feature-map area. That alone makes high-resolution or multi-scale feature maps unaffordable, which is exactly what small objects need. At the same time, with standard initialization the attention logits are near-zero, so each query starts by attending almost uniformly over tens of thousands of keys. The gradient that should teach the query to focus on a few relevant locations is diluted across all those keys, and it takes an enormous number of epochs for the attention to sharpen. The fix, therefore, is not a separate training trick for convergence or a separate architectural add-on for scale; it is to redesign the attention operator itself so that each query looks at only a small, learned set of locations instead of everywhere.

The method is Deformable DETR. Its core is multi-scale deformable attention. Each query has a content feature and a normalized 2-d reference point in the image. Instead of dot-producting against every key, the query predicts both a small set of 2-d sampling offsets around its reference point and an attention weight for each sample. The values at those fractional locations are read off by bilinear interpolation, and the result is a weighted sum. Formally, for a query with content z_q and reference point p̂_q, attending over L multi-scale feature maps with M heads and K sampling points per level, the output is the sum over heads, levels, and points of W_m [ A_mlqk W'_m x^l(φ_l(p̂_q) + Δp_mlqk) ], where the offsets Δp_mlqk and weights A_mlqk are produced by a single linear projection on z_q with 3MLK outputs. The weights are normalized by a softmax over the LK points per head. Because the number of attended locations is tiny compared with the feature-map area, the encoder cost drops from quadratic in spatial size to linear, and decoder cross-attention becomes independent of spatial size altogether. More importantly, the gradient is now concentrated on a handful of sampled points, so attention can sharpen quickly instead of crawling out of a uniform initialization.

The architecture keeps the set-prediction formulation intact. The backbone is ResNet, and four multi-scale feature maps are built from stages C3, C4, C5 via 1×1 convolutions to 256 channels plus a fourth level from a 3×3 stride-2 convolution on C5. Notably, no feature-pyramid top-down pathway is needed, because cross-scale exchange happens inside the deformable attention itself. The encoder replaces self-attention with multi-scale deformable attention, where each feature-map pixel queries with itself as its reference point. A learned scale-level embedding is added to the sinusoidal positional encoding so each pixel knows which pyramid level it belongs to. The decoder keeps ordinary multi-head self-attention among the object queries so they can still coordinate and suppress duplicates, but replaces cross-attention into the feature maps with multi-scale deformable attention. Each object query's reference point is predicted from its embedding by a linear layer followed by a sigmoid. The box head then predicts an offset relative to that reference point in logit space, so when the correction is zero the box center equals the reference point; this ties where the attention samples and what the head predicts, removing optimization slack.

Initialization is part of the design and matters a lot. The projection weights that produce offsets and weights are initialized to zero, so at start-up the behavior depends only on the biases and is query-independent. The attention-weight biases are set so every sampled point gets equal weight 1/(LK). The offset biases are set so the eight heads point in the eight compass directions at radii 1 through K, giving a small symmetric local receptive field that training then deforms. Two optional refinements are natural once the attention is cheap enough. Iterative box refinement lets every decoder layer correct the previous layer's box, with the reference point updated to the current box center and the sampling offsets modulated by the current box size; gradients are blocked through the previous box to keep the chain stable. A two-stage variant adds an encoder-only first stage that predicts a box per pixel, takes the top-scoring proposals as initial boxes, and feeds them into the decoder, grounding the object queries in actual image content instead of generic learned slots. Training uses the same bipartite-matching set loss as DETR, but classification is focal loss, the number of object queries is increased to 300, and the schedule shrinks to 50 epochs with a 10× learning-rate drop at epoch 40.

```python
import torch
from torch import nn
import torch.nn.functional as F


def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=eps, max=1 - eps)
    return torch.log(x / (1 - x))


def pos_embed_2d(h, w, c, device):
    assert c % 4 == 0
    dim = c // 2
    y_embed = torch.arange(1, h + 1, dtype=torch.float32, device=device).unsqueeze(1).repeat(1, w)
    x_embed = torch.arange(1, w + 1, dtype=torch.float32, device=device).unsqueeze(0).repeat(h, 1)
    div = torch.exp(-torch.arange(0, dim, 2, dtype=torch.float32, device=device) * (torch.log(torch.tensor(10000.0)) / dim))
    pe = torch.zeros(h, w, c, device=device)
    pe[:, :, 0:dim:2] = torch.sin(y_embed[:, :, None] * div)
    pe[:, :, 1:dim:2] = torch.cos(y_embed[:, :, None] * div)
    pe[:, :, dim + 0::2] = torch.sin(x_embed[:, :, None] * div)
    pe[:, :, dim + 1::2] = torch.cos(x_embed[:, :, None] * div)
    return pe


def ms_deform_attn_core(value, spatial_shapes, sampling_locations, attention_weights):
    """Bilinear-sample LK points per query from L feature maps and weight-sum.
    value: (N, sum(HW), M, D); sampling_locations: (N, Lq, M, L, K, 2) in [0,1];
    attention_weights: (N, Lq, M, L, K)."""
    N, _, M, D = value.shape
    _, Lq, _, L, K, _ = sampling_locations.shape
    value_list = value.split([H * W for H, W in spatial_shapes], dim=1)
    sampling_grids = 2 * sampling_locations - 1  # [0,1] -> [-1,1]
    sampled = []
    for lid, (H, W) in enumerate(spatial_shapes):
        v_l = value_list[lid].flatten(2).transpose(1, 2).reshape(N * M, D, H, W)
        grid_l = sampling_grids[:, :, :, lid].transpose(1, 2).flatten(0, 1)  # (N*M, Lq, K, 2)
        sampled.append(F.grid_sample(v_l, grid_l, mode='bilinear', padding_mode='zeros', align_corners=False))
    attn = attention_weights.transpose(1, 2).reshape(N * M, 1, Lq, L * K)
    out = (torch.stack(sampled, dim=-2).flatten(-2) * attn).sum(-1)
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
        thetas = torch.arange(self.M, dtype=torch.float32) * (2 * torch.pi / self.M)
        grid = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid = (grid / grid.abs().max(-1, keepdim=True)[0]).view(self.M, 1, 1, 2).repeat(1, self.L, self.K, 1)
        for k in range(self.K):
            grid[:, :, k] *= (k + 1)
        self.sampling_offsets.bias = nn.Parameter(grid.view(-1))
        nn.init.constant_(self.attention_weights.weight, 0.)
        nn.init.constant_(self.attention_weights.bias, 0.)

    def forward(self, query, reference_points, value, spatial_shapes):
        N, Lq, _ = query.shape
        value = self.value_proj(value).view(N, -1, self.M, self.d_model // self.M)
        offsets = self.sampling_offsets(query).view(N, Lq, self.M, self.L, self.K, 2)
        weights = self.attention_weights(query).view(N, Lq, self.M, self.L * self.K)
        weights = weights.softmax(-1).view(N, Lq, self.M, self.L, self.K)
        norm = torch.stack([torch.tensor([W, H], dtype=offsets.dtype, device=offsets.device) for H, W in spatial_shapes])
        loc = reference_points[:, :, None, :, None, :] + offsets / norm[None, None, None, :, None, :]
        out = ms_deform_attn_core(value, spatial_shapes, loc, weights)
        return self.output_proj(out)


class EncoderLayer(nn.Module):
    def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=4, dim_ffn=1024, dropout=0.1):
        super().__init__()
        self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.ffn = nn.Sequential(nn.Linear(d_model, dim_ffn), nn.ReLU(inplace=True),
                                 nn.Dropout(dropout), nn.Linear(dim_ffn, d_model))
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, pos, reference_points, spatial_shapes):
        q = src + pos
        src2 = self.self_attn(q, reference_points, src, spatial_shapes)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src = src + self.dropout2(self.ffn(src))
        return self.norm2(src)


class DecoderLayer(nn.Module):
    def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=4, dim_ffn=1024, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.ffn = nn.Sequential(nn.Linear(d_model, dim_ffn), nn.ReLU(inplace=True),
                                 nn.Dropout(dropout), nn.Linear(dim_ffn, d_model))
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, tgt, query_pos, reference_points, memory, spatial_shapes):
        q = k = tgt + query_pos
        tgt2, _ = self.self_attn(q, k, tgt)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        q2 = tgt + query_pos
        tgt2 = self.cross_attn(q2, reference_points, memory, spatial_shapes)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt = tgt + self.dropout3(self.ffn(tgt))
        return self.norm3(tgt)


class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers):
        super().__init__()
        layers = []
        for i in range(num_layers):
            layers.append(nn.Linear(in_dim if i == 0 else hidden_dim, hidden_dim if i < num_layers - 1 else out_dim))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < len(self.layers) - 1 else layer(x)
        return x


class DeformableDETR(nn.Module):
    def __init__(self, backbone, num_classes, num_queries=300, d_model=256, n_levels=4, n_heads=8, n_points=4):
        super().__init__()
        self.backbone = backbone
        self.num_queries = num_queries
        self.d_model = d_model
        # Project C3, C4, C5 and build C6; simplified here as one conv per backbone output.
        in_chs = backbone.out_channels if hasattr(backbone, 'out_channels') else [512, 1024, 2048]
        self.input_proj = nn.ModuleList([nn.Conv2d(c, d_model, 1) for c in in_chs[:3]] +
                                        [nn.Sequential(nn.Conv2d(in_chs[2], d_model, 3, stride=2, padding=1))])
        self.level_embed = nn.Parameter(torch.randn(n_levels, d_model))
        enc_layer = EncoderLayer(d_model, n_levels, n_heads, n_points)
        self.encoder = nn.ModuleList([enc_layer for _ in range(6)])
        dec_layer = DecoderLayer(d_model, n_levels, n_heads, n_points)
        self.decoder = nn.ModuleList([dec_layer for _ in range(6)])
        self.query_embed = nn.Embedding(num_queries, d_model * 2)
        self.reference_point = nn.Linear(d_model, 2)
        self.class_head = nn.Linear(d_model, num_classes)
        self.box_head = MLP(d_model, d_model, 4, 3)

    def forward(self, images):
        feats = self.backbone(images)
        if isinstance(feats, torch.Tensor):
            feats = [feats]
        srcs, shapes, pos = [], [], []
        refs = []
        for l, proj in enumerate(self.input_proj):
            f = proj(feats[min(l, len(feats) - 1)] if l < 3 else feats[-1])
            if l == 3:
                f = self.input_proj[3](feats[-1])
            B, _, H, W = f.shape
            srcs.append(f.flatten(2).transpose(1, 2))
            shapes.append((H, W))
            pe = pos_embed_2d(H, W, self.d_model, f.device).unsqueeze(0) + self.level_embed[l]
            pos.append(pe.flatten(2).transpose(1, 2))
            yy, xx = torch.meshgrid(torch.linspace(0, 1, H, device=f.device),
                                    torch.linspace(0, 1, W, device=f.device), indexing='ij')
            ref = torch.stack([xx, yy], -1).flatten(0, 1).unsqueeze(0).expand(B, -1, -1)
            refs.append(ref)
        src = torch.cat(srcs, dim=1)
        pos_enc = torch.cat(pos, dim=1)
        reference_points = torch.cat(refs, dim=1)
        memory = src
        for layer in self.encoder:
            memory = layer(memory, pos_enc, reference_points, shapes)

        query_pos, tgt = self.query_embed.weight.split(self.d_model, dim=1)
        query_pos = query_pos.unsqueeze(0).expand(images.size(0), -1, -1)
        tgt = tgt.unsqueeze(0).expand(images.size(0), -1, -1)
        reference = self.reference_point(query_pos).sigmoid()

        for layer in self.decoder:
            tgt = layer(tgt, query_pos, reference, memory, shapes)

        logits = self.class_head(tgt)
        box = self.box_head(tgt)
        ref_logit = inverse_sigmoid(reference)
        box[..., :2] += ref_logit
        boxes = box.sigmoid()
        return {'pred_logits': logits, 'pred_boxes': boxes}
```
