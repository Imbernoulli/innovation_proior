The set-prediction detector is the design I want to keep — a Transformer reads a CNN feature map, a handful of object queries cross-attend into it, a bipartite-matching loss forces one prediction per object, and the whole apparatus of anchors, IoU assignment, and NMS just evaporates. The trouble is it's almost unusable in practice for two reasons, and I want to find out whether both reasons share a root cause, because if they do I might fix them with one change.

Deficit one: it trains forever — something like an order of magnitude more epochs than an ordinary detector to reach comparable accuracy. Deficit two: it's bad at small objects. Let me hold these up next to the attention mechanism and see where they come from.

Start with the small-object problem, because it's the more mechanical of the two. Small objects need high-resolution feature maps — you can't localize a 20-pixel object on a feature map downsampled by 32. Modern detectors handle this with multi-scale feature pyramids: detect small things on fine maps, big things on coarse maps. But the encoder here runs self-attention over the feature-map pixels, and let me count the cost. With queries and keys both being the HW pixels of the map, attention is O(N_q C² + N_k C² + N_q N_k C), and since N_q = N_k = HW ≫ C, the dominating term is O(N_q N_k C) = O(H²W²C). Quadratic in the *area* of the feature map. So if I double the resolution along each axis, attention cost goes up sixteen-fold. High-resolution maps, let alone a multi-scale pyramid of them, are simply unaffordable. The small-object weakness isn't a modeling choice; it's forced by the quadratic encoder.

Now the slow convergence. Why does attention over image pixels take so long to train? Look at the attention weights at initialization. With sensible init, the projected query U_m z_q and projected key V_m x_k each come out with about zero mean and unit variance, so the logits z_qᵀ U_mᵀ V_m x_k / √C_v are small and roughly equal across keys, which means after the softmax every key gets weight A_mqk ≈ 1/N_k. The query attends almost uniformly to *everything*. Now think about the gradient that's supposed to teach the query to focus. It's spread across all N_k keys, each contributing a 1/N_k-sized share — the signal that "this key matters, those don't" is buried under the near-uniform averaging. With N_k = HW being tens of thousands, the gradient is badly diluted and ambiguous, and it takes an enormous number of steps for the weights to crawl from "uniform over the whole image" to "sharp on the object." Indeed the learned attention does end up sparse — focused on object boundaries — it just takes forever to get there from a uniform start.

So both deficits have the *same* root: attention treats every one of the HW spatial locations as a key. That's what makes it quadratic (cost over all keys) and that's what makes it slow (uniform-over-all-keys initialization, diluted gradient). If I could make each query attend to only a *small, fixed* number of locations — say K of them, with K ≪ HW — then the cost stops depending on HW the bad way, and the gradient is concentrated on K things instead of smeared over HW. One change, both problems. The question is which K locations, and how a query picks them.

There's a precedent for exactly this kind of "attend to a few learned locations" idea, from convolution. Ordinary convolution samples its input on a fixed grid around each position. Deformable convolution makes that grid *movable*: it predicts, from the features, an offset for each sample point, so the conv gathers from y(p₀) = Σ_n w(p_n)·x(p₀ + p_n + Δp_n), where the offsets Δp_n are learned and vary per location. The offsets are fractional, so the sampled value is read off by bilinear interpolation, which is differentiable in the offset. The relevant idea I want to steal: instead of a fixed or exhaustive set of locations, let the model *predict a small set of sampling locations* around a reference, content-adaptively, and read the feature there by bilinear interpolation.

Let me design that as an attention module. A query q has a content feature z_q and a 2-d reference point p_q in the feature map — think of p_q as "where this query is looking." For each head m and each of K sampling points k, I want a location p_q + Δp_mqk and a weight A_mqk, and I aggregate the value-projected features sampled at those locations:

  DeformAttn(z_q, p_q, x) = Σ_{m=1}^{M} W_m [ Σ_{k=1}^{K} A_mqk · W'_m x(p_q + Δp_mqk) ],

with x(p_q + Δp_mqk) read by bilinear interpolation since the location is fractional, A_mqk ∈ [0,1] normalized so Σ_k A_mqk = 1, and Δp_mqk ∈ ℝ² an unconstrained 2-d offset. The key set is now K points per head, not HW pixels.

Here's the part where I have to deviate from standard attention, and it's worth being careful about why. In ordinary attention the weight A_mqk comes from a dot product between the query and the *key's* content. But my "keys" are at arbitrary fractional sampled locations that I haven't even computed yet — I'd have to interpolate a key feature at every candidate location to dot against the query, which defeats the whole point of being cheap, and there's no fixed key set to dot against anyway. So I don't compute weights from query-key compatibility. Instead I predict *both* the offsets and the weights directly from the query feature z_q, just like deformable conv predicts its offsets from the features. Concretely, feed z_q through one linear projection with 3MK output channels: the first 2MK channels are the offsets Δp_mqk (two numbers per point per head), and the remaining MK channels go through a softmax (over the K points, per head) to give the weights A_mqk. No query-key interaction at all — the query decides where to look and how much to weight each look, in one shot. This is the deformable-conv analogue of attention, and it's what keeps the cost from ever touching N_k.

Does this actually generalize attention or is it a different animal? If I let K range over *all* spatial locations and predicted a weight for each, this would reduce to ordinary attention (a weighted sum over all positions). So deformable attention is an efficient variant of attention with a learned pre-filter: it picks K locations worth attending to instead of attending to everything. That reassures me I'm not throwing away expressiveness, just the exhaustiveness that caused the trouble.

Now extend it to multi-scale, because the small-object fix needs a pyramid. Suppose I have L feature maps {x^l} at different resolutions. I want a query to sample across *all* levels, so it can pull fine detail and coarse context together. Use a normalized reference point p̂_q ∈ [0,1]² (so (0,0) is top-left, (1,1) bottom-right of the image), and a per-level rescaling φ_l that maps the normalized point into level l's pixel grid. Then sample K points per level per head:

  MSDeformAttn(z_q, p̂_q, {x^l}) = Σ_{m=1}^{M} W_m [ Σ_{l=1}^{L} Σ_{k=1}^{K} A_mlqk · W'_m x^l(φ_l(p̂_q) + Δp_mlqk) ],

with the weights now normalized over both level and point, Σ_{l,k} A_mlqk = 1, so LK weights per head sum to one. It samples LK points total from the multi-scale stack instead of K from one map. Notice the single-scale version is just L = 1, and — checking the limit — if I further set K = 1 and fix the value projection W'_m to identity, this is exactly one deformable-convolution sample per head. So multi-scale deformable attention sits cleanly between deformable convolution (one point, one scale) and full attention (all points), which is a good sign the formulation is the right shape.

Let me verify the cost actually fixed both problems. The expensive pieces: predicting the offsets and weights is O(3 N_q C M K) (the 3MK-channel projection over N_q queries); computing the aggregation, with the value projection done up front so it's shared, is O(N_q C² + HW C² + 5 N_q K C), where the 5 is four taps for bilinear interpolation plus the weighted sum. Putting it together the overall cost is O(N_q C² + min(HWC², N_q K C²) + 5 N_q K C + 3 N_q C M K). With the defaults I'll use — M = 8 heads, K ≤ 4 points, C = 256 — the small terms 5K + 3MK = 20 + 96 = 116 are less than C = 256, so they're absorbed and it's O(2 N_q C² + min(HWC², N_q K C²)). Now plug in the two places it's used. In the encoder, queries are the pixels, N_q = HW, giving O(HW C²) — *linear* in the spatial size, not quadratic. That's the fix: I can now afford high-resolution, multi-scale maps. In the decoder cross-attention, queries are the object queries, N_q = N, giving O(NK C²) — completely independent of HW. And the slow-convergence cause is gone too, because each query's gradient is now concentrated on K = a handful of sampled points rather than diluted over HW, so attention can sharpen from the start instead of crawling out of a uniform initialization.

Now wire it into the architecture. First the multi-scale feature maps. Take a ResNet backbone and pull the output of stages C3, C4, C5 (strides 8, 16, 32), each through a 1×1 convolution to a common width C = 256 — that's three levels. For very large objects I want one more, even coarser level, so add a fourth map by a 3×3 stride-2 convolution on C5 (call it C6), also 256 channels. Four levels, L = 4. Notice I am deliberately *not* adding a feature-pyramid top-down pathway. The whole point of multi-scale deformable attention is that a query samples across all levels in one operation, so cross-scale information already gets exchanged inside the attention — bolting an FPN on top would be redundant, and indeed it doesn't help.

The encoder replaces its self-attention with multi-scale deformable attention. Both queries and keys are pixels of the multi-scale maps; for a query pixel, the natural reference point is *itself* — its own normalized location. One subtlety: a query pixel needs to know which of the four levels it lives on, because the same normalized (x, y) appears on every level and the positional embedding alone can't disambiguate level. So I add a scale-level embedding e_l on top of the positional embedding. The positional embedding is the usual fixed sinusoidal encoding, but there's no canonical "level encoding," so I make e_l a learned vector per level, trained jointly.

The decoder has two attention types: self-attention among the N object queries (so they coordinate and avoid claiming the same object) and cross-attention from queries into the feature maps. Deformable attention is built for feature-map keys, so I replace only the *cross*-attention with multi-scale deformable attention; the query-to-query self-attention I leave as ordinary multi-head attention, since N is small (a few hundred) and that interaction is cheap and is exactly the dedup channel I want to keep intact. Each object query needs a reference point to sample around — but unlike encoder pixels, a query has no inherent location. So predict it: from the query's embedding, a learned linear projection followed by a sigmoid gives a normalized reference point p̂_q ∈ [0,1]².

That reference point suggests a nice change to how I predict boxes. The deformable cross-attention gathers image features from *around* the reference point, so the reference point is effectively the query's best guess of where its object is — treat it as the initial box center. Then have the detection head predict the box as an *offset relative to the reference* rather than in absolute coordinates. Concretely, in logit space:

  b̂_q = ( σ(b_qx + σ⁻¹(p̂_qx)), σ(b_qy + σ⁻¹(p̂_qy)), σ(b_qw), σ(b_qh) ),

where b_q{x,y,w,h} are the head's raw outputs and σ, σ⁻¹ are sigmoid and inverse sigmoid. The σ⁻¹(p̂) turns the reference center into a logit, the head adds a correction, and σ maps back into [0,1] — so when the correction is zero the box center *is* the reference point, and everything stays in normalized coordinates. Why bother? Because now the location the attention samples around and the box the head predicts are tied to the same reference — the learned attention ends up strongly correlated with the predicted box, which removes a lot of optimization slack and speeds convergence further.

The initialization of the offset-and-weight projection matters more than it looks, so let me reason it out rather than init randomly. If I randomly initialize the linear projection that produces Δp and A, the K sampling points start at random offsets with random weights — chaotic, and the module has to climb out of nonsense. I'd rather start from a sensible prior that resembles a uniform local receptive field. So: initialize the projection *weights* to zero, which makes the offsets and weights at initialization depend only on the *biases* (independent of the query). Set the weight biases so every sampled point starts with equal attention A_mlqk = 1/(LK). And set the offset biases so the M = 8 heads start pointing in the eight compass directions at increasing radii — head 1 to (−k, −k), head 2 to (−k, 0), …, head 8 to (k, k), for the k-th point at radius k. So at initialization each query looks at a small symmetric spread of points around its reference, in eight directions, with uniform weights — a reasonable conv-like starting receptive field that training then deforms. Starting from this prior rather than randomness is part of why it trains quickly.

I have an efficient, fast-converging set detector. Now I can afford a couple of refinements that were impractical when each forward pass was so expensive. The first: iterative box refinement, borrowing the idea from iterative refinement in optical-flow estimation. Instead of only the last decoder layer predicting a box, let *every* decoder layer refine the previous layer's box. With D = 6 layers, layer d takes the box b̂^{d-1}_q from layer d−1 and corrects it, again in logit space:

  b̂^d_q = σ( Δb^d_q + σ⁻¹(b̂^{d-1}_q) )  componentwise for x, y, w, h,

where Δb^d_q is layer d's predicted correction and the heads are *not* shared across layers (each layer learns its own refinement). The initial box b̂^0 has center at the reference point and a small fixed size (w = h = 0.1; the system turns out robust to this — 0.05 through 0.5 all work similarly). Two stabilizers. First, like the optical-flow refinement, I block the gradient from flowing back through σ⁻¹(b̂^{d-1}) — gradients pass only through the current correction Δb^d — so the chain of refinements doesn't create an exploding backward path through all D layers. Second, since the box is being refined, the cross-attention at layer d should sample around the *current* box, not the original reference: set the reference point to the previous box's center (b̂^{d-1}_qx, b̂^{d-1}_qy) and modulate the sampling offsets by the box size, (Δp_x · b̂^{d-1}_qw, Δp_y · b̂^{d-1}_qh), so the sampled locations scale with the box — a big box samples over a wide span, a small box over a narrow one. (And the offset-bias init for refinement is further scaled by 1/(2K) so the initial sampling points sit inside the previous box.)

The second refinement is a two-stage variant, by analogy to two-stage detectors that generate region proposals first. In the original set detector the object queries are learned vectors unrelated to the actual image — generic "slots." It would help to initialize them from image-dependent proposals. So a first stage generates proposals: apply a detection head (a 3-layer FFN regressing a box, plus a linear binary foreground/background classifier) to *every pixel* of the encoded multi-scale maps, where each pixel directly predicts a box. To keep box sizes sensible per level, initialize each pixel's predicted width and height around a base scale that grows with the level, σ⁻¹(2^{l−1}·s) with base s = 0.05, so coarse levels propose larger boxes. Train this first stage with the same bipartite-matching loss. But there's a cost trap: if I literally used every pixel as an object query in the decoder, the decoder's query-to-query self-attention is quadratic in the number of queries and would blow up. So I make the first stage *encoder-only* — no decoder — just pixel-wise box prediction, and pick the top-scoring boxes as region proposals. Because the matching loss already enforces one-prediction-per-object, I don't need NMS before passing proposals on. The second stage feeds those proposals into the decoder as the initial boxes for iterative refinement, with each object query's positional embedding set from its proposal's coordinates. Now the queries start from image-grounded guesses instead of generic slots.

A couple of training choices follow the prior set detector but with two deliberate changes. The classification loss: rather than a plain cross-entropy over classes, use a focal loss (weight 2) for the box classification, which down-weights the many easy background predictions and is better suited to the large foreground/background imbalance. And increase the number of object queries from 100 to 300, since the now-cheaper attention makes more queries affordable and more slots help recall. Optimize with Adam at base learning rate 2e-4 (β₁ = 0.9, β₂ = 0.999, weight decay 1e-4), 50 epochs with a 10× decay at epoch 40 — already roughly an order of magnitude shorter than the original detector's schedule. And give the linear projections that predict the reference points and the sampling offsets a 10× smaller learning rate (2e-5): they control *where* the module looks, and a large learning rate there destabilizes the sampling geometry.

Let me write it, grounded in how the multi-scale deformable attention actually gets implemented — the offset/weight projection, the bilinear-sampling core via grid_sample, and the encoder/decoder wiring.

```python
import torch
from torch import nn
import torch.nn.functional as F


def ms_deform_attn_core(value, value_spatial_shapes, sampling_locations, attention_weights):
    """Pure-pytorch core: bilinear-sample LK points per query from L feature maps and
    weight-sum them. value: (N, sum(HW), M, D); sampling_locations: (N, Lq, M, L, K, 2) in
    [0,1]; attention_weights: (N, Lq, M, L, K)."""
    N, _, M, D = value.shape
    _, Lq, _, L, K, _ = sampling_locations.shape
    value_list = value.split([H * W for H, W in value_spatial_shapes], dim=1)
    sampling_grids = 2 * sampling_locations - 1          # [0,1] -> [-1,1] for grid_sample
    sampling_value_list = []
    for lid, (H, W) in enumerate(value_spatial_shapes):
        # (N, HW, M, D) -> (N*M, D, H, W)
        v_l = value_list[lid].flatten(2).transpose(1, 2).reshape(N * M, D, H, W)
        # (N, Lq, M, K, 2) -> (N*M, Lq, K, 2)
        grid_l = sampling_grids[:, :, :, lid].transpose(1, 2).flatten(0, 1)
        sampled = F.grid_sample(v_l, grid_l, mode='bilinear',
                                padding_mode='zeros', align_corners=False)  # (N*M, D, Lq, K)
        sampling_value_list.append(sampled)
    # (N*M, 1, Lq, L*K)
    attn = attention_weights.transpose(1, 2).reshape(N * M, 1, Lq, L * K)
    out = (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attn).sum(-1)
    return out.view(N, M * D, Lq).transpose(1, 2)        # (N, Lq, C)


class MSDeformAttn(nn.Module):
    def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=4):
        super().__init__()
        self.M, self.L, self.K, self.d_model = n_heads, n_levels, n_points, d_model
        # one linear over the query -> 2*M*L*K offsets and M*L*K (pre-softmax) weights
        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, d_model)    # W'_m stacked over heads
        self.output_proj = nn.Linear(d_model, d_model)   # W_m stacked over heads
        self._reset_parameters()

    def _reset_parameters(self):
        # zero the projection WEIGHTS so init depends only on biases (query-independent)
        nn.init.constant_(self.sampling_offsets.weight, 0.)
        # offset biases: M heads point in distinct directions at radii k=1..K
        thetas = torch.arange(self.M) * (2 * torch.pi / self.M)
        grid = torch.stack([thetas.cos(), thetas.sin()], -1)             # 8 compass dirs
        grid = (grid / grid.abs().max(-1, keepdim=True)[0])              # unit-infinity
        grid = grid.view(self.M, 1, 1, 2).repeat(1, self.L, self.K, 1)
        for k in range(self.K):
            grid[:, :, k] *= (k + 1)                                     # radius grows with k
        self.sampling_offsets.bias = nn.Parameter(grid.view(-1))
        nn.init.constant_(self.attention_weights.weight, 0.)
        nn.init.constant_(self.attention_weights.bias, 0.)               # -> uniform 1/(LK)

    def forward(self, query, reference_points, value, value_spatial_shapes):
        N, Lq, _ = query.shape
        value = self.value_proj(value).view(N, -1, self.M, self.d_model // self.M)
        offsets = self.sampling_offsets(query).view(N, Lq, self.M, self.L, self.K, 2)
        weights = self.attention_weights(query).view(N, Lq, self.M, self.L * self.K)
        weights = weights.softmax(-1).view(N, Lq, self.M, self.L, self.K)  # sum_{l,k}=1
        # reference (N,Lq,L,2) normalized; offsets are in normalized units / per-level scale
        offset_norm = torch.stack([torch.tensor([W, H]) for H, W in value_spatial_shapes], 0)
        sampling_locations = reference_points[:, :, None, :, None, :] \
            + offsets / offset_norm[None, None, None, :, None, :]
        out = ms_deform_attn_core(value, value_spatial_shapes, sampling_locations, weights)
        return self.output_proj(out)


class DeformableDetector(nn.Module):
    def __init__(self, backbone, num_classes, num_queries=300, d_model=256, n_levels=4):
        super().__init__()
        self.backbone = backbone
        self.num_queries = num_queries
        # C3,C4,C5 -> 1x1 conv to d_model; C6 -> 3x3 stride-2 conv on C5. No FPN top-down.
        self.input_proj = build_multiscale_proj(backbone.out_channels, d_model, n_levels)
        self.level_embed = nn.Parameter(torch.randn(n_levels, d_model))   # learned scale-level emb
        self.encoder = DeformableEncoder(d_model, n_levels, num_layers=6)
        self.decoder = DeformableDecoder(d_model, n_levels, num_layers=6)
        self.query_embed = nn.Embedding(num_queries, 2 * d_model)         # content + query pos
        self.reference_point = nn.Linear(d_model, 2)                      # query pos -> p_hat
        self.class_head = nn.Linear(d_model, num_classes)                 # focal-loss classifier
        self.box_head = MLP(d_model, d_model, 4, num_layers=3)            # raw box offsets

    def forward(self, images):
        feats = self.backbone(images)                          # C3..C5
        srcs, shapes, pos = [], [], []
        for l, f in enumerate(self.input_proj(feats)):         # build L multi-scale maps
            srcs.append(f.flatten(2).transpose(1, 2))
            shapes.append((f.shape[2], f.shape[3]))
            pos.append(sinusoidal_pos_embed(f) + self.level_embed[l])     # pos + scale-level
        src = torch.cat(srcs, 1); pos = torch.cat([p.flatten(2).transpose(1,2) for p in pos], 1)
        memory = self.encoder(src, shapes, pos)                # reference = each pixel itself

        query_pos, tgt = self.query_embed.weight.split(self.query_embed.weight.size(1)//2, 1)
        query_pos = query_pos[None].expand(images.size(0), -1, -1)
        tgt = tgt[None].expand(images.size(0), -1, -1)
        reference = self.reference_point(query_pos).sigmoid()  # p_hat in [0,1]^2
        hs, refs = self.decoder(tgt, reference, query_pos, memory, shapes)

        # box predicted RELATIVE to reference, in logit space
        out_logits = self.class_head(hs[-1])
        ref_logit = inverse_sigmoid(refs[-1])
        box = self.box_head(hs[-1])
        box[..., :2] += ref_logit                              # center offset from reference
        out_boxes = box.sigmoid()                              # -> [0,1]^4
        return {"pred_logits": out_logits, "pred_boxes": out_boxes}
```

The causal chain: the set-prediction detector both converged ~10× too slowly and detected small objects poorly, and tracing both back showed a single root — its attention treats every one of the HW feature-map pixels as a key, which is quadratic in spatial size (forbidding high-resolution multi-scale maps) and starts from a near-uniform 1/N_k attention whose gradient is too diluted to sharpen quickly. So, borrowing the learned-sampling idea from deformable convolution, I replace it with a deformable attention module that attends to only K learned sampling points around each query's reference point, predicting the offsets and weights directly from the query (no query-key dot product) and reading values by bilinear interpolation; extended across L levels this becomes multi-scale deformable attention, which is linear in HW in the encoder and HW-independent in the decoder cross-attention, and concentrates each gradient on a handful of points so training is fast. I build four-level multi-scale maps from ResNet (no FPN, since the attention exchanges scale information itself), tag each level with a learned scale embedding, predict a per-query reference point and the box as a logit-space offset from it (tying attention to the box), initialize the offsets to a uniform eight-direction prior, and add iterative box refinement (gradient-blocked across layers) and an encoder-only two-stage proposal mechanism — yielding an end-to-end, NMS-free detector that converges in a normal budget and handles small objects.
