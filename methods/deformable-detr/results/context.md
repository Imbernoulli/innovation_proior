## Research question

A recently introduced detector (DETR; Carion et al.) had shown that object detection can be cast as
direct set prediction with a Transformer encoder-decoder and a bipartite-matching loss, removing the
anchors, the hand-tuned assignment rules, and the non-maximum-suppression post-process that every
prior detector relied on. It was elegant and competitive on large objects — but it had two crippling
practical deficits, and both trace back to the same source: a Transformer attention layer, applied to
image feature maps, treats *every* spatial location as a key. First, it converges painfully slowly,
needing roughly an order of magnitude more training epochs than a standard detector. Second, it
detects small objects poorly, because the natural fix for small objects — high-resolution and/or
multi-scale feature maps — is unaffordable: encoder self-attention is quadratic in the number of
spatial locations, so high resolution blows up compute and memory.

The precise question: can the attention mechanism at the heart of this set-prediction detector be
redesigned so that it (a) converges in a normal training budget and (b) affords high-resolution,
multi-scale feature maps for small-object detection — while keeping the end-to-end, NMS-free,
set-prediction formulation intact? A solution must attack the "attend to all locations" property
that causes both problems at once.

## Background

**Multi-head attention.** Given a query element with feature z_q and a set of key elements with
features x_k, multi-head attention aggregates the keys' (value-projected) contents weighted by
query-key compatibility:

  MultiHeadAttn(z_q, x) = Σ_{m=1}^{M} W_m [ Σ_{k∈Ω_k} A_mqk · W'_m x_k ],

with attention weights A_mqk ∝ exp(z_qᵀ U_mᵀ V_m x_k / √C_v) normalized to sum to one over keys, M
heads of width C_v = C/M, and learnable projections U_m, V_m, W'_m, W_m. With N_q queries and N_k
keys the cost is O(N_q C² + N_k C² + N_q N_k C). When queries and keys are both image pixels,
N_q = N_k = HW ≫ C, so the O(N_q N_k C) = O(H²W²C) term dominates: attention grows *quadratically*
with feature-map area.

**Why all-location attention also converges slowly.** With standard initialization, the projected
query U_m z_q and key V_m x_k have roughly zero mean and unit variance, so before training the
attention logits are near-uniform and A_mqk ≈ 1/N_k. When N_k is large (image pixels), each query
spreads its attention almost evenly over the whole map; the gradient that should sharpen attention
onto the few relevant locations is diluted across N_k keys and is ambiguous. The model needs a very
long schedule to move from this near-uniform state to the sparse, object-focused attention it
eventually learns. In practice the attention maps start essentially uniform and only become sparse —
concentrated on object boundaries — after many epochs.

**Sampling-based, content-adaptive operators.** Deformable convolution (Dai et al.) attacks a related
rigidity in ordinary convolution. A regular conv samples its input on a fixed grid R; deformable conv
augments each grid point with a *learned* offset, y(p₀) = Σ_n w(p_n)·x(p₀ + p_n + Δp_n), where the
offsets Δp_n are predicted from the features by a sibling convolution and vary per location. Because
the offset locations are fractional, the sampled values are computed by bilinear interpolation,
x(p) = Σ_q G(q, p)·x(q), which is differentiable in p. The lesson: a layer can adaptively gather
information from a *small, learned set of locations* anywhere in the map, rather than from a fixed or
exhaustive neighborhood.

**Multi-scale features.** Modern detectors handle the huge range of object sizes with multi-scale
feature maps — a feature pyramid (FPN; Lin et al.) builds high-resolution, semantically strong maps
by a top-down pathway with lateral connections, so small objects are detected on fine maps and large
objects on coarse maps.

**Sigmoid box parameterization.** Predicting normalized box coordinates in [0,1] is conveniently done
by passing raw predictions through a sigmoid; predicting a coordinate *relative to a reference* is
naturally done in logit space, σ(Δ + σ⁻¹(reference)), which stays in [0,1] and reduces to the
reference when Δ = 0.

## Baselines

**DETR (Carion et al.).** The detector this work reacts to and builds on. A CNN backbone (ResNet)
produces a single C×H×W feature map; a Transformer encoder runs self-attention over its HW pixels; a
Transformer decoder takes N learnable object queries (e.g. N = 100), self-attends among them and
cross-attends into the encoder memory; and per-query heads — a 3-layer FFN regressing a normalized box
b ∈ [0,1]⁴ and a linear classifier — produce the detections. A set-based Hungarian loss with bipartite
matching forces one prediction per ground-truth object, so no NMS is needed. Encoder self-attention
costs O(H²W²C); decoder cross-attention O(HWC² + NHWC) and decoder self-attention O(2NC² + N²C). The
gaps it leaves: ~10× the training epochs of a standard detector (from the near-uniform-attention
initialization problem above) and weak small-object accuracy (the quadratic encoder forbids
high-resolution or multi-scale maps).

**Anchor / proposal detectors with FPN (e.g. Faster R-CNN + FPN).** The detectors DETR was measured
against. They converge in a short schedule and detect small objects well using multi-scale feature
pyramids, but reintroduce exactly the hand-designed pieces — anchors, IoU-based assignment, NMS — that
the set-prediction formulation removed. They are the convergence-and-small-object yardstick a
redesigned attention would need to approach without giving the hand-design back.

## Evaluation settings

The benchmark is COCO 2017 object detection (~118k train images, 5k val, 80 classes, many small
objects per image). The metric is average precision (AP) averaged over IoU thresholds 0.50–0.95,
reported overall and broken out by object size (AP_S / AP_M / AP_L) and at fixed thresholds (AP_50,
AP_75). Detectors are also compared on training cost (epochs, GPU-hours), parameter count, FLOPs, and
inference frames per second — convergence speed is itself a headline axis here. Backbones are ResNet
variants pretrained on ImageNet with frozen batch-norm statistics; an optimizer suited to
transformers (Adam) and a step learning-rate schedule are standard.

## Code framework

The available primitives are a ResNet backbone, standard multi-head attention and Transformer
encoder/decoder blocks, a bilinear-sampling operator (e.g. grid_sample) for gathering features at
fractional locations, the set-based bipartite-matching detection loss, and routine box utilities
(normalized-coordinate conversion, sigmoid/inverse-sigmoid). The scaffold wires a backbone, a
multi-scale feature stack, an encoder/decoder, and the detection heads together, and leaves empty the
slots the method must fill: the attention operator that processes feature maps (the piece whose
all-location cost has to be replaced), how each query gets a spatial reference, and how a box is
predicted relative to it.

```python
import torch
from torch import nn
import torch.nn.functional as F


def sigmoid_inverse(x):
    return torch.log(x / (1 - x))


class AttentionOverFeatureMaps(nn.Module):
    """The attention operator whose 'attend to every pixel' cost must be redesigned.
    Aggregates feature-map content for each query around a spatial reference."""
    def __init__(self, d_model=256, n_heads=8, n_levels=4):
        super().__init__()
        self.d_model, self.n_heads, self.n_levels = d_model, n_heads, n_levels
        # TODO: how does a query gather feature-map content efficiently?

    def forward(self, query, reference_point, feature_maps, spatial_shapes):
        # TODO: produce an updated query feature from the feature maps
        raise NotImplementedError


class Encoder(nn.Module):
    """Refine multi-scale feature maps; each map pixel is a query over the maps."""
    def __init__(self, layer, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([layer for _ in range(num_layers)])

    def forward(self, feature_maps, spatial_shapes, pos):
        # TODO: each pixel attends over the feature maps around its own location
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
        self.reference_point = None          # TODO: a spatial reference per query
        self.class_head = None               # TODO: per-query classifier
        self.box_head = None                 # TODO: per-query box, relative to its reference

    def forward(self, images):
        feats = self.backbone(images)
        # TODO: project to multi-scale maps, encode, decode, predict {logits, boxes}
        raise NotImplementedError
```
