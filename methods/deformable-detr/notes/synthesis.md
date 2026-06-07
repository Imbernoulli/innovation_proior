# Deformable DETR synthesis (grounded in 2010.04159 src + DETR + deformable conv 1703.06211 + canonical repo)

## Pain point: DETR's two deficits (both from Transformer attention on image feature maps as keys)
1. SLOW CONVERGENCE: needs many more epochs than modern detectors (~500 vs ~12 for Faster R-CNN). At init, cross-attention is ~uniform over the whole feature map; only after long training do attention maps become sparse (focusing on object extremities). Learning that change is slow.
   - WHY (the math): with proper init, U_m z_q and V_m x_k ~ mean 0 var 1, so attention weights A_mqk ≈ 1/N_k when N_k large → ambiguous gradients for input features → long schedule needed for weights to focus on specific keys. In images N_k = HW is huge → convergence tedious.
2. LOW SMALL-OBJECT PERFORMANCE / can't use high-res: modern detectors use high-res / multi-scale feature maps for small objects, but DETR encoder self-attention is O(H²W²C) — quadratic in spatial size — so high-res is unaffordable. DETR uses only single C5 (stride 32) feature map.

## Standard multi-head attention recap (Vaswani)
MultiHeadAttn(z_q, x) = Σ_m W_m [ Σ_k A_mqk · W'_m x_k ]
- A_mqk ∝ exp( z_q^T U_m^T V_m x_k / sqrt(C_v) ), normalized Σ_k A_mqk = 1.
- W'_m ∈ R^{C_v×C}, W_m ∈ R^{C×C_v}, U_m,V_m ∈ R^{C_v×C}. C_v = C/M.
- complexity O(N_q C² + N_k C² + N_q N_k C); in images N_q=N_k=HW≫C → dominated by O(N_q N_k C) = O(H²W²C).

## DETR recap
- CNN backbone (ResNet) → feature map x ∈ R^{C×H×W} → Transformer encoder-decoder → N object queries (learnable pos embeddings, N=100).
- 3-layer FFN regression branch → box b∈[0,1]^4 (normalized cx,cy,w,h); linear projection classification branch.
- Encoder self-attn O(H²W²C). Decoder cross-attn O(HWC² + NHWC) linear in HW; self-attn O(2NC² + N²C).
- Set-based Hungarian loss, bipartite matching, unique predictions per GT.

## KEY IDEA: Deformable Attention Module
Core issue: Transformer attention looks over ALL spatial locations. Fix: only attend to a small set K of key SAMPLING POINTS around a reference point, regardless of feature map size. Inspired by deformable convolution (Dai 2017).

### Deformable conv (Dai 2017) ancestor
- Regular conv samples on fixed grid R. Deformable conv augments with learned offsets: y(p0)=Σ_n w(p_n) x(p0+p_n+Δp_n). Offsets Δp_n learned by a sibling conv, fractional → bilinear interpolation x(p)=Σ_q G(q,p) x(q). Data-driven, per-location.

### Single-scale deformable attention (Eq 2)
DeformAttn(z_q, p_q, x) = Σ_{m=1}^M W_m [ Σ_{k=1}^K A_mqk · W'_m · x(p_q + Δp_mqk) ]
- q query w/ content z_q & 2-d reference point p_q. m head, k sampled key, K = total sampled points (K ≪ HW).
- Δp_mqk = sampling offset (2-d real, unconstrained), A_mqk = scalar attention weight ∈[0,1], Σ_k A_mqk = 1.
- x(p_q+Δp_mqk) by bilinear interpolation (fractional location).
- Both Δp and A from LINEAR PROJECTION of z_q: linear op with 3MK channels → first 2MK = offsets Δp, last MK → softmax → weights A.
- KEY DIFFERENCE from standard attn: attention weights come DIRECTLY from query (not query-key dot product). No key-query interaction; weights predicted. Pre-filtering by deformable sampling.
- complexity O(2N_q C² + min(HWC², N_q K C²)). Encoder N_q=HW → O(HWC²) LINEAR. Decoder cross-attn N_q=N → O(NKC²) INDEPENDENT of HW.

### Multi-scale deformable attention (Eq 3)
MSDeformAttn(z_q, p̂_q, {x^l}) = Σ_m W_m [ Σ_{l=1}^L Σ_{k=1}^K A_mlqk · W'_m · x^l(φ_l(p̂_q) + Δp_mlqk) ]
- p̂_q ∈ [0,1]² normalized reference point. φ_l rescales normalized coords to level-l feature map.
- A_mlqk normalized Σ_{l,k} A_mlqk = 1 (over all L levels and K points = LK total).
- Samples LK points from L-level multi-scale maps. (single-scale = special case L=1.)
- Degenerates to deformable conv when L=1, K=1, W'_m = identity.
- When sampling points traverse all locations → equivalent to Transformer attention. So it's an efficient variant of attention w/ deformable pre-filtering.

## Architecture
### Encoder
- Replace ALL Transformer attention (self-attn) w/ MSDeformAttn. Input & output = multi-scale maps, same resolutions.
- Multi-scale features {x^l}_{l=1}^{L-1} (L=4) from ResNet stages C3,C4,C5 (each via 1×1 conv to C=256). x^L (lowest res) = 3×3 stride-2 conv on C5 (= "C6"). All C=256.
- NO FPN top-down: MSDeformAttn itself exchanges info across scales (ablation: FPN doesn't help).
- Query & key = pixels of multi-scale maps. For each query pixel, reference point = itself.
- Scale-level embedding e_l added (in addition to positional embedding) to identify which level a query pixel is in. e_l randomly initialized, jointly trained (positional embedding is fixed sinusoidal).

### Decoder
- Cross-attn + self-attn, both query = object queries.
- Only replace CROSS-attention w/ MSDeformAttn (it processes feature maps as keys); leave self-attention unchanged (standard MHA among queries — small N, cheap).
- Reference point p̂_q for each object query: predicted from its query embedding via learnable linear projection + sigmoid.
- Detection head predicts box as RELATIVE OFFSETS w.r.t. reference point (reference = initial box center guess). → decoder attention strongly correlated w/ predicted boxes → faster convergence.

## Bounding box prediction (appendix bbox_pred)
b̂_q = { σ(b_qx + σ⁻¹(p̂_qx)), σ(b_qy + σ⁻¹(p̂_qy)), σ(b_qw), σ(b_qh) }
- b_q{x,y,w,h} ∈ R predicted by head. σ, σ⁻¹ = sigmoid, inverse sigmoid. Keeps b̂ ∈ [0,1]^4.
- The σ⁻¹(p̂) + offset, then σ: predict offset in logit space relative to reference point.

## Complexity derivation (appendix)
- Computing Δp & A: O(3 N_q C M K).
- Computing Eq2 given them: O(N_q C² + N_q K C² + 5 N_q K C) — factor 5 = bilinear interp (4) + weighted sum.
- Or compute W'_m x before sampling (query-independent): O(N_q C² + HW C² + 5 N_q K C).
- Overall: O(N_q C² + min(HWC², N_q K C²) + 5 N_q K C + 3 N_q C M K).
- With M=8, K≤4, C=256: 5K + 3MK = 20+96 = 116 < 256 = C, so → O(2 N_q C² + min(HWC², N_q K C²)).

## Initialization (appendix — CRITICAL for it to work)
- M=8 heads. W'_m, W_m random init.
- Linear projection WEIGHTS for predicting A_mlqk and Δp_mlqk init to ZERO.
- BIASES init so that at init: A_mlqk = 1/(LK) (uniform), and Δp offsets spread in 8 compass directions scaled by k: Δp for the 8 heads = (-k,-k),(-k,0),(-k,k),(0,-k),(0,k),(k,-k),(k,0),(k,k) for k∈{1..K}. → heads start looking in different directions at increasing radii.
- For iterative refinement: offset-bias init further ×1/(2K) so initial sampling points within previous-layer box.

## Variants
### Iterative bounding box refinement (from RAFT optical flow)
- Each decoder layer refines box from previous layer. D=6 layers.
- b̂^d_q = σ( Δb^d_q + σ⁻¹(b̂^{d-1}_q) ) componentwise for x,y,w,h.
- Heads per layer NOT shared. Initial box b̂^0 = (p̂_qx, p̂_qy, 0.1, 0.1). Robust to w0,h0 choice (0.05-0.5 similar).
- Gradient ONLY through Δb^d, blocked at σ⁻¹(b̂^{d-1}) (stabilize, like RAFT).
- In refinement, layer d samples keys relative to box b̂^{d-1}: reference point = (b̂^{d-1}_qx, b̂^{d-1}_qy), sampling offset MODULATED by box size: (Δp_x·b̂^{d-1}_qw, Δp_y·b̂^{d-1}_qh). → sampling locations tied to box center & size.

### Two-stage Deformable DETR
- First stage: encoder-only (remove decoder). Each pixel = object query, directly predicts a box (3-layer FFN regression + linear binary fg/bg classification). Box: b̂_i with w,h base init σ⁻¹(2^{l_i-1} s), base scale s=0.05. Hungarian loss.
- Top-scoring boxes = region proposals (NO NMS before stage 2).
- Second stage: proposals → decoder as initial boxes for iterative refinement; object-query positional embeddings = pos embeddings of proposal coordinates.

## Training config (experiment)
- Follow DETR except: Focal Loss (Lin 2017, weight 2) for box classification (vs DETR cross-entropy); N=300 object queries (vs 100).
- 50 epochs, lr decay ×0.1 at epoch 40. Adam, base lr 2e-4, β1=0.9, β2=0.999, wd 1e-4.
- lr ×0.1 for linear projections predicting reference points & sampling offsets.

## Canonical implementation
github.com/fundamentalvision/Deformable-DETR. MSDeformAttn module: linear(C, 3MK*L... actually offsets MK*L*2 + weights MK*L), value proj W'; pure-pytorch core ms_deform_attn_core_pytorch:
- value split by levels [H_l*W_l]; reshape each to (N*M, D, H_l, W_l).
- sampling_grids = 2*sampling_locations - 1 (to [-1,1]); F.grid_sample(value_l, grid_l, mode='bilinear', padding_mode='zeros', align_corners=False).
- attention_weights reshape (N*M,1,Lq,L*K); output = (stack(sampling_values,-2).flatten(-2) * attention_weights).sum(-1); then reshape & output proj W_m.
- (CUDA kernel for speed; pytorch core is the reference.)

## Design-decision → why table
- Deformable attn (K sampled points not all HW): kills both quadratic complexity AND slow convergence (small fixed key set → focused gradients from start). Inspired by deformable conv.
- Attention weights from query directly (not Q·K): there are no "keys" to dot with at arbitrary fractional sampled locations cheaply; predicting weights from query is the deformable-conv analogue and avoids N_k blow-up.
- Multi-scale (L=4) without FPN: small objects need high res; MSDeformAttn samples across levels so cross-scale info exchange is built in → FPN top-down redundant.
- C6 via stride-2 conv on C5: add an extra low-res level for large objects.
- Scale-level embedding e_l (learned): query pixel must know which level it's in (positional embedding alone is per-level ambiguous); learned because there's no canonical encoding.
- Only replace decoder CROSS-attn, keep self-attn: deformable attn is for feature-map keys; query-query self-attn is small-N standard MHA, cheap & needed for dedup.
- Reference point + box-as-offset: deformable attn samples around reference, so predicting box relative to it ties attention to box → easier optimization, faster convergence.
- sigmoid/inverse-sigmoid box param: keep coords in [0,1], add offsets in logit space.
- Zero-init weight + structured bias for offsets/weights: start as uniform attention spread in 8 directions (mimics a sensible conv-like receptive field) so training begins from a reasonable prior, not random chaos.
- N=300 queries, focal loss: more queries + focal loss handle the many-candidate / class-imbalance better; modest changes following DETR.
- 0.1× lr for ref-point & offset projections: these control sampling geometry; large lr destabilizes where the module looks.
- Iterative refinement (grad blocked at prev box, RAFT-style): progressive box correction; gradient blocking stabilizes.
- Two-stage (encoder-only proposals, no NMS): better init boxes for decoder; one-to-one Hungarian means no NMS needed.
```
