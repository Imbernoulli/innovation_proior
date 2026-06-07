# DETR — synthesis (design-decision → why), written before Phase 2

## The pain point at the time (2019–2020)
Object detection = predict a *set* of (class, box). But no detector predicts a set directly. They all
turn it into dense surrogate regression/classification on *many candidates* (proposals / anchors / center
grid), each candidate independently classified + box-regressed. Three hand-designed pieces fall out of this:
1. **Anchors / proposals / centers** — a dense tiling of reference boxes; design (scales, aspect ratios,
   strides) heavily affects performance (Zhang et al. 2019 "bridging the gap" shows this).
2. **Target-assignment heuristic** — IoU thresholds decide which anchor is "responsible" for which GT
   (e.g. IoU>0.7 positive, <0.3 negative). Many anchors map to one GT → many near-duplicate positives.
3. **NMS** — because (2) produces duplicates by construction, you need non-maximum suppression at inference
   to collapse near-identical boxes. NMS is greedy, non-differentiable, has its own threshold, and is not
   trained jointly.

So the system is not end-to-end: the loss optimizes per-anchor surrogates, but the actual quantity we care
about (a clean set) is produced by a separate non-learned post-process. End-to-end set prediction worked in
translation/speech but "not yet in detection."

## Load-bearing ancestors

### Faster R-CNN (Ren et al. 2015) — the baseline to match/replace
- Two stages: RPN proposes regions (objectness + box deltas over anchors), then per-RoI head classifies +
  refines. Box regression is a **delta** Δ w.r.t. an anchor/proposal (parametrized t_x,t_y,t_w,t_h).
- Positive/negative proposals balanced by **subsampling** (e.g. 1:3) — this is the class-imbalance fix that
  DETR mirrors with the no-object down-weight.
- Gap: needs anchors, assignment heuristic, NMS. Many design iterations (FPN etc.) needed to make it strong;
  performance sensitive to all the hand-design.

### NMS + anchors — the thing to remove
- NMS exists *only* because training admits many predictions per object. If training enforced one-to-one,
  there'd be nothing to suppress. This is the key lever: change the loss so duplicates are penalized, and
  NMS becomes unnecessary.

### The Transformer (Vaswani et al. 2017) — the architecture
- Encoder-decoder, self-attention models all pairwise interactions in a set/sequence; permutation-invariant
  up to positional encoding. Originally autoregressive decoding (one token at a time).
- Self-attention is exactly the global pairwise-interaction machinery you need to let predictions "talk to
  each other" and not collude on the same object → de-duplication happens *inside* the model.
- Attention: softmax(QK^T / sqrt(d_k)) V. Multi-head with M heads, d_k = d_model/M. The 1/sqrt(d_k) keeps
  dot products from saturating softmax. Need positional encoding because attention is set-symmetric.

### Parallel (non-autoregressive) decoding (Gu et al. 2017; Oord et al. 2017; BERT)
- A set has no natural order, so autoregressive ordering is artificial and slow (cost ∝ output length).
  Decode all N slots in parallel — possible precisely because the matching loss is permutation-invariant.

### Set prediction + Hungarian matching (Kuhn 1955; Stewart & Andriluka 2015; Erhan 2014)
- To train a set predictor you need a permutation-invariant loss. The canonical device: find a bipartite
  matching between predictions and GT (Hungarian algorithm, O(n^3)), then supervise each matched pair.
- Stewart 2015 / RomeraParedes 2015 did this with RNN decoders on small datasets — never beat strong
  baselines, never used transformers+parallel decoding.

### GIoU (Rezatofighi et al. 2018) — the box metric
- Plain IoU loss = 0 gradient when boxes don't overlap (no signal to move them together). GIoU adds a term
  that's defined even for disjoint boxes: GIoU = IoU − |C \ (A∪B)| / |C|, C = smallest enclosing box.
  GIoU ∈ [−1,1]; loss = 1 − GIoU. Scale-invariant (a ratio), unlike L1 which scales with box size.

## The derivation chain
1. We want a *set*. Cast as: fix N slots (N ≫ typical #objects), pad GT to N with ∅. Predict N (class,box).
2. To train without an order, score each candidate assignment σ (a permutation) and pick the cheapest →
   bipartite matching, solved by Hungarian. Matching cost per pair = class term + box term.
3. Matching cost: use **probability** p̂(c_i) not log-prob, so it's commensurate with the box term (both
   O(1)); empirically better. For ∅ rows the cost is constant (drop them). Box term in the *cost* uses the
   same L1+GIoU as the loss.
   L_match(y_i, ŷ_σ(i)) = −1[c_i≠∅] p̂_σ(i)(c_i) + 1[c_i≠∅] L_box(b_i, b̂_σ(i)).
4. Given σ̂, the **Hungarian loss** is the actual training loss, now with **−log p̂** (proper NLL) for class:
   L = Σ_i [ −log p̂_σ̂(i)(c_i) + 1[c_i≠∅] L_box(b_i, b̂_σ̂(i)) ].
   Down-weight the ∅ class log-prob by 10 (class imbalance, analogous to Faster R-CNN subsampling).
5. Box loss: L_box = λ_iou·(1−GIoU) + λ_L1·||b−b̂||_1, normalized by #objects in batch.
   Why both: L1 alone has scale-dependent magnitude (big boxes dominate); GIoU is scale-invariant but its
   gradient geometry differs — combining gets the best of both. Why GIoU not IoU: IoU has no gradient for
   disjoint boxes. λ_L1=5, λ_iou=2.
   Why absolute box prediction (not Δ vs anchor): there are no anchors anymore; predict cx,cy,w,h in [0,1]
   directly via sigmoid.
6. Architecture: CNN backbone (ResNet-50, frozen BN, C=2048) → 1×1 conv to d=256 → flatten HW to a
   sequence → transformer **encoder** (self-attention, fixed sine 2D positional encoding added at every
   layer) → transformer **decoder** that takes N learned input embeddings = **object queries** (each query
   = one slot; they must differ to produce different outputs since decoder is permutation-invariant) and
   cross-attends to encoder memory → each output embedding → shared FFN head: 3-layer MLP→box (sigmoid),
   linear→class (incl. ∅).
7. Why this de-duplicates: decoder self-attention lets the N queries see each other and the matching loss
   gives exactly one positive per object, so the model learns to spread queries over distinct objects → no
   NMS needed.
8. **Auxiliary decoding losses (deep supervision):** add a prediction FFN + Hungarian loss after *every*
   decoder layer (shared FFN params, one shared LayerNorm). Helps the model output the right *number* of
   objects per class; stabilizes training of the deep decoder.

## Design-decision → why (+ rejected alternatives)
| decision | why | rejected alt & failure |
|---|---|---|
| N fixed slots + ∅ padding | turn variable-size set into fixed-size; ∅ = "no detection" | dynamic count → no clean parallel head |
| bipartite (one-to-one) matching | unique assignment ⇒ no duplicates ⇒ no NMS | many-to-one (Faster R-CNN) ⇒ duplicates ⇒ needs NMS |
| Hungarian algorithm | exact min-cost assignment, O(n^3), standard | greedy match suboptimal |
| prob (not log-prob) in matching cost | commensurate w/ box term; better empirically | log-prob unbounded, dominates cost |
| −log p (NLL) in the actual loss | proper classification gradient | prob-based loss = weaker class gradient |
| down-weight ∅ by 10 | N≫#obj ⇒ ∅ dominates; rebalance | no weight ⇒ collapse to all-∅ (mirrors FRCNN subsampling) |
| L1 + GIoU box loss | L1 = good coords, scale-dependent; GIoU = scale-invariant overlap | L1 only: big boxes dominate; IoU only: no grad when disjoint |
| GIoU not IoU | gradient even for non-overlapping boxes | IoU loss: zero grad, can't pull boxes together |
| absolute box (cx,cy,w,h sigmoid) | no anchors to offset from | Δ-vs-anchor reintroduces anchors |
| transformer encoder-decoder | self-attn = global pairwise reasoning ⇒ de-dup in-model | conv/FC (early set detectors) can't model global relations ⇒ still need NMS |
| object queries = learned embeddings | break decoder permutation symmetry; each = a slot | identical queries ⇒ identical outputs |
| parallel decoding | set has no order; fast; matching loss is perm-invariant | autoregressive (RNN) slow, imposes fake order, never beat baselines |
| fixed sine 2D pos enc, added every layer | attention needs spatial info; 2D generalization of 1D sine | no pos enc ⇒ spatially blind |
| 3-layer MLP for box, linear for class | small heads suffice on rich decoder embeddings | — |
| ResNet-50, frozen BN | standard detection backbone; BN stats unstable at small batch | trainable BN ⇒ instability |
| backbone lr 1e-5 ≪ 1e-4 | pretrained backbone, transformer from scratch; stabilizes early epochs | equal lr ⇒ unstable |
| AdamW, wd 1e-4, grad clip 0.1 | transformers train poorly w/ SGD; clip controls spikes | SGD ⇒ worse |
| aux losses at every decoder layer | deep supervision; correct object count | last-layer only ⇒ harder optim |
| long schedule | transformer + bipartite matching slow to converge | short ⇒ undertrained |

## Code grounding (facebookresearch/detr)
- `models/detr.py`: DETR module (backbone→input_proj→transformer→class_embed/bbox_embed); SetCriterion
  (loss_labels NLL w/ empty_weight, loss_boxes L1+GIoU); MLP head; aux loss loop.
- `models/matcher.py`: HungarianMatcher — cost_class = −p[target], cost_bbox = cdist L1, cost_giou =
  −GIoU; C = w_bbox·bbox + w_class·class + w_giou·giou; scipy linear_sum_assignment.
- `util/box_ops.py`: generalized_box_iou = iou − (enclosing_area − union)/enclosing_area.
- `models/transformer.py`: encoder/decoder, pos added to q,k each layer; query_pos added in decoder;
  return_intermediate for aux losses.
- `models/backbone.py`: ResNet w/ FrozenBatchNorm2d; `models/position_encoding.py`: sine + learned.
- Inference toy in src/python/detr.py (50-line version).
