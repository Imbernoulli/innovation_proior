## Research question

A recently introduced detector (DETR; Carion et al.) casts object detection as direct set prediction
with a Transformer encoder-decoder and a bipartite-matching loss, removing the anchors, the hand-tuned
assignment rules, and the non-maximum-suppression post-process that prior detectors rely on. It is
competitive on large objects. It uses many more training epochs than a standard detector, and operates
on a single low-resolution feature map: the encoder self-attention is quadratic in the number of
spatial locations, so the model is run at one coarse scale.

The setting: in a Transformer-based set-prediction detector, the attention mechanism aggregates, for
each query, over the whole feature map. The question is how to design the attention that processes the
feature maps so that a Transformer set-prediction detector can be trained on a normal budget and can
operate on high-resolution, multi-scale feature maps, while keeping the end-to-end, NMS-free,
set-prediction formulation intact.

## Background

**Multi-head attention.** Given a query element with feature z_q and a set of key elements with
features x_k, multi-head attention aggregates the keys' (value-projected) contents weighted by
query-key compatibility:

  MultiHeadAttn(z_q, x) = Σ_{m=1}^{M} W_m [ Σ_{k∈Ω_k} A_mqk · W'_m x_k ],

with attention weights A_mqk ∝ exp(z_qᵀ U_mᵀ V_m x_k / √C_v) normalized to sum to one over keys, M
heads of width C_v = C/M, and learnable projections U_m, V_m, W'_m, W_m. With N_q queries and N_k
keys the cost is O(N_q C² + N_k C² + N_q N_k C). When queries and keys are both image pixels,
N_q = N_k = HW ≫ C, so the O(N_q N_k C) = O(H²W²C) term dominates: attention grows quadratically
with feature-map area. With standard initialization, the projected query U_m z_q and key V_m x_k have
roughly zero mean and unit variance, so before training the attention logits are near-uniform and
A_mqk ≈ 1/N_k — each query spreads its attention almost evenly over the whole map at the start of
training.

**Sampling-based, content-adaptive operators.** Deformable convolution (Dai et al.) augments ordinary
convolution. A regular conv samples its input on a fixed grid R; deformable conv augments each grid
point with a learned offset, y(p₀) = Σ_n w(p_n)·x(p₀ + p_n + Δp_n), where the offsets Δp_n are
predicted from the features by a sibling convolution and vary per location. Because the offset
locations are fractional, the sampled values are computed by bilinear interpolation,
x(p) = Σ_q G(q, p)·x(q), which is differentiable in p.

**Multi-scale features.** Modern detectors handle the range of object sizes with multi-scale feature
maps — a feature pyramid (FPN; Lin et al.) builds high-resolution, semantically strong maps by a
top-down pathway with lateral connections, so small objects are detected on fine maps and large
objects on coarse maps.

**Sigmoid box parameterization.** Predicting normalized box coordinates in [0,1] is conveniently done
by passing raw predictions through a sigmoid; predicting a coordinate relative to a reference is
naturally done in logit space, σ(Δ + σ⁻¹(reference)), which stays in [0,1] and reduces to the
reference when Δ = 0.

## Baselines

**DETR (Carion et al.).** The detector this work builds on. A CNN backbone (ResNet) produces a single
C×H×W feature map; a Transformer encoder runs self-attention over its HW pixels; a Transformer decoder
takes N learnable object queries (e.g. N = 100), self-attends among them and cross-attends into the
encoder memory; and per-query heads — a 3-layer FFN regressing a normalized box b ∈ [0,1]⁴ and a
linear classifier — produce the detections. A set-based Hungarian loss with bipartite matching forces
one prediction per ground-truth object, so no NMS is needed. Encoder self-attention costs O(H²W²C);
decoder cross-attention O(HWC² + NHWC) and decoder self-attention O(2NC² + N²C).

**Anchor / proposal detectors with FPN (e.g. Faster R-CNN + FPN).** The detectors DETR is measured
against. They train in a short schedule and detect small objects well using multi-scale feature
pyramids, and use hand-designed components — anchors, IoU-based assignment, NMS. They are the standard
convergence and small-object yardstick.

## Evaluation settings

The benchmark is COCO 2017 object detection (~118k train images, 5k val, 80 classes, many small
objects per image). The metric is average precision (AP) averaged over IoU thresholds 0.50–0.95,
reported overall and broken out by object size (AP_S / AP_M / AP_L) and at fixed thresholds (AP_50,
AP_75). Detectors are also compared on training cost (epochs, GPU-hours), parameter count, FLOPs, and
inference frames per second. Backbones are ResNet variants pretrained on ImageNet with frozen
batch-norm statistics; an optimizer suited to transformers (Adam) and a step learning-rate schedule
are standard.

## Code framework

The available primitives are a ResNet backbone, standard multi-head attention and Transformer
encoder/decoder blocks, a bilinear-sampling operator (e.g. grid_sample) for gathering features at
fractional locations, the set-based bipartite-matching detection loss, and routine box utilities
(normalized-coordinate conversion, sigmoid/inverse-sigmoid). The scaffold wires a backbone, a
multi-scale feature stack, an encoder/decoder, and the detection heads together, and leaves empty the
slots the method must fill.

```python
import torch
from torch import nn
import torch.nn.functional as F


def sigmoid_inverse(x):
    return torch.log(x / (1 - x))


class AttentionOverFeatureMaps(nn.Module):
    """The attention operator that processes feature maps for each query."""
    def __init__(self, d_model=256, n_heads=8, n_levels=4):
        super().__init__()
        self.d_model, self.n_heads, self.n_levels = d_model, n_heads, n_levels
        # TODO

    def forward(self, query, reference_point, feature_maps, spatial_shapes):
        # TODO: produce an updated query feature from the feature maps
        raise NotImplementedError


class Encoder(nn.Module):
    """Refine multi-scale feature maps; each map pixel is a query over the maps."""
    def __init__(self, layer, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([layer for _ in range(num_layers)])

    def forward(self, feature_maps, spatial_shapes, pos):
        # TODO: each pixel attends over the feature maps
        raise NotImplementedError


class Decoder(nn.Module):
    """Object queries self-attend, then attend into the encoded feature maps."""
    def __init__(self, layer, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([layer for _ in range(num_layers)])

    def forward(self, query, reference_points, memory, spatial_shapes):
        # TODO: query self-attention + attention over feature maps; predict boxes
        raise NotImplementedError


class SetDetector(nn.Module):
    def __init__(self, backbone, num_classes, num_queries=300, d_model=256, n_levels=4):
        super().__init__()
        self.backbone = backbone
        self.num_queries = num_queries
        # TODO: build a multi-scale feature stack from the backbone (channels -> d_model)
        self.input_proj = None
        self.encoder = None
        self.decoder = None
        self.query_embed = nn.Embedding(num_queries, d_model)
        self.class_head = None               # TODO: per-query classifier
        self.box_head = None                 # TODO: per-query box

    def forward(self, images):
        feats = self.backbone(images)
        # TODO: project to multi-scale maps, encode, decode, predict {logits, boxes}
        raise NotImplementedError
```
