# Swin Transformer — synthesis notes (superseded)

This older synthesis slot is retained for compatibility. The strict source table is `notes/source_matrix.md`, and the active reconstruction/audit notes for the repaired deliverables are `notes/discovery_synthesis.md`.

## Pain point / research question
We want ONE Transformer backbone that works for *all* vision tasks (classification, detection, segmentation), the way ResNet does and the way the Transformer does for NLP. Two obstacles when porting a language Transformer (ViT) to vision:
1. **Scale variation.** Words are atomic tokens of one "scale"; visual entities (a car, a person, a hand) appear at wildly different scales. Detection/segmentation need *multi-scale* features (FPN, U-Net pyramids). ViT produces a single low-resolution (16× downsampled) feature map → no pyramid.
2. **Resolution / cost.** Global self-attention is O((hw)²) in #tokens. A dense-prediction image at stride-4 has ~3136 tokens at 224²; at detection resolutions (1333 long side) it explodes. Quadratic cost makes global attention intractable for high-res dense prediction.

A solution must: (a) produce a hierarchical, multi-scale feature map at the same strides as a ConvNet (4,8,16,32) so it drops into FPN/U-Net/Mask-R-CNN unchanged; (b) have cost **linear** in image size; (c) retain enough global modeling power to beat ConvNets.

## Load-bearing ancestors (lineage)

- **ViT (Dosovitskiy et al. 2020).** Split image into 16×16 non-overlapping patches, linear-embed each as a token, add learned absolute position embedding + class token, run a *standard* Transformer encoder with **global** MSA. One scale (stride 16), quadratic cost. Needs JFT-300M to beat ConvNets. *Gap:* single-scale, quadratic, data-hungry. This is the thing we react against.
- **DeiT (Touvron et al. 2020).** Same architecture as ViT but trained on ImageNet-1K with strong aug + distillation; shows ViT-style nets can train on 1.28M images. *Gap:* still single-scale, still quadratic. Gives us the training recipe (AdamW, RandAug, Mixup/CutMix, stochastic depth, rand-erasing).
- **CNN pyramids: VGG / ResNet (He et al. 2015).** Hierarchical feature maps via stride-2 downsampling between stages → strides 4/8/16/32. This is the *target shape* a vision backbone must produce; it's why FPN/U-Net work. Tells us we need patch-merging downsampling to mimic this.
- **FPN (Lin et al. 2017) / U-Net (Ronneberger 2015).** Dense-prediction heads consume a *pyramid* of feature maps. They presuppose a multi-scale backbone. Motivates requirement (a).
- **Standard Transformer (Vaswani et al. 2017).** Scaled dot-product attention `softmax(QKᵀ/√d)V`, multi-head, residual+LN, 2-layer MLP FFN with GELU. The block we'll reuse; only the MSA is replaced.
- **Local/sliding-window self-attention backbones (Hu et al. 2019 local-relation; Ramachandran et al. 2019 stand-alone self-attention; Zhao 2020 SAN).** Replace conv with attention computed in a local window *around each pixel* (sliding window). Slightly better accuracy/FLOPs than ResNet but **terrible real latency**: each query pixel has a *different* key set → poor memory access, no batched matmul. *Gap:* local attention is the right idea for linear cost, but *sliding* windows are hardware-hostile. This is the precise failure that motivates *non-overlapping* windows.
- **Relative position encoding (Shaw 2018; T5 Raffel 2019; Hu 2018 relation-net; Hu 2019 local-relation).** Add a learned bias depending on the *relative* offset between query and key, instead of (or in addition to) absolute position. Encodes translation-equivariant geometry. Motivates the relative-position-bias term.
- **Efficient/linear attention (Performer, Choromanski 2020).** Kernel-approximate softmax → linear cost. A baseline alternative for cost reduction. *Gap:* approximation loses accuracy; we beat it.
- **PVT (Wang 2021, concurrent).** Also builds a multi-resolution Transformer pyramid, but attention is still global-ish (spatial-reduction) → still quadratic. We're linear and local.

## The complexity argument (must be derived inline in reasoning.md)

Image of h×w patch-tokens, channel dim C. One MSA layer:
- QKV projection: each of Q,K,V is `(hw)×C @ C×C` = `hw·C²` mults, ×3 = `3hw C²`.
- attention scores `QKᵀ`: `(hw)×C @ C×(hw)` = `(hw)²·C`.
- weighted sum `(attn)V`: `(hw)×(hw) @ (hw)×C` = `(hw)²·C`.
- output proj: `hw·C²`.
Total Ω(MSA) = 4hwC² + 2(hw)²C. The `2(hw)²C` term is quadratic in hw.

Now partition into non-overlapping windows of M×M tokens → there are hw/M² windows, each with M² tokens. Per window the two attention terms cost `2(M²)²C = 2M⁴C`; times hw/M² windows = `2M²hwC`. The projection terms are unchanged (they're per-token): 4hwC². So
Ω(W-MSA) = 4hwC² + 2M²hwC.
With M fixed (=7), the second term is **linear** in hw. Quadratic (hw)² → linear hw. That's the whole point.

## Design decisions → why (with rejected alternatives)

| Decision | Why / what breaks otherwise | Rejected alternative & its failure |
|---|---|---|
| Small 4×4 patches at stem (vs ViT's 16×16) | Need a high-res stride-4 feature map to start the pyramid (dense prediction needs fine features) | 16×16 → only stride-16, no pyramid |
| Local **non-overlapping** window attention | Makes cost linear (above); fixed M² tokens/window → batched matmul, one shared key set per window → great memory access | (i) global attention = quadratic, intractable; (ii) **sliding** window = linear FLOPs but per-pixel key set → hardware-hostile, huge real latency |
| Window size M=7 | Big enough for useful local context, fits stage resolutions (56,28,14,7 all divisible-ish; stage-4 is 7×7 = one window) | M=1 → no context; large M → cost grows as M² per window, loses locality benefit |
| **Shifted** windows (alternate regular / shifted by ⌊M/2⌋) in successive blocks | Plain windows never exchange info across window borders → receptive field frozen at M. Shifting by half a window makes the next layer's windows straddle the previous borders → cross-window connections, growing receptive field, *while keeping* non-overlapping efficiency | (i) no shift → blocked receptive field (ablation: −1.1% top-1, −2.8 box AP); (ii) sliding window for cross-window = latency death |
| **Cyclic shift + masking** to batch the shifted config | Naive shift makes 9 windows (some < M²) instead of 4 → pad to M² and you'd compute 3×3=9 windows = 2.25× cost. Cyclic-roll the feature map top-left by ⌊M/2⌋ so wrapped-around tokens fill the partial windows; re-partition → exactly the same 4 windows. But a batched window now mixes tokens from non-adjacent image regions, so apply an attention **mask** (−∞ on cross-region pairs) so each sub-region only self-attends. Reverse-roll after. | naive padding = 2.25× compute; this keeps #windows constant |
| **Relative position bias** B added inside softmax: `softmax(QKᵀ/√d + B)V` | Translation-equivariant geometric prior; within a window, what matters is relative offset, not absolute index. Ablation: rel-pos beats no-pos (+1.2%) and abs-pos (+0.8%); abs-pos even *hurts* detection/seg | absolute pos embedding (ViT-style) abandons translation invariance, helps cls slightly but hurts dense tasks |
| Parameterize **B̂ ∈ ℝ^{(2M−1)×(2M−1)}**, index into it | Relative coord on each axis ∈ [−(M−1), M−1] → 2M−1 distinct values per axis. So only (2M−1)² distinct biases exist; storing full M²×M² B wastes params and ignores the sharing. Index: shift coords by +(M−1) to [0,2M−2], multiply the row-axis by (2M−1), sum the two axes → unique flat index ∈ [0,(2M−1)²−1] | full M²×M² table = redundant, doesn't tie equal-offset pairs |
| **Patch merging** between stages: concat 2×2 neighbors (→4C), LN, linear 4C→2C | Builds the hierarchy: 2× spatial downsample, 2× channel up — exactly the ResNet stage pattern → strides 4/8/16/32 | strided conv would work too but concat+linear is the natural token-space analog and keeps it all in token form |
| Per-stage depths {2,2,6,2} (T) / {2,2,18,2} (S/B/L); always **even** | Even so each (W-MSA, SW-MSA) pair is complete (a regular block then a shifted block). Most depth in stage-3 (stride-16) where most semantic work happens, à la ResNet | odd depth → dangling unpaired window config |
| head_dim d = 32 fixed; #heads grows 3→6→12→24 | Keep per-head dim constant as C doubles each stage → heads double. 32 is the standard sweet spot | — |
| MLP ratio α = 4 | Standard Transformer FFN width; same as ViT/BERT | — |
| LN **pre**-norm, residual after each sublayer | Stable deep-Transformer training (pre-LN) | post-LN harder to train deep |
| GELU, AdamW, weight decay 0.05, drop-path (stochastic depth, scaled by depth), no abs-pos, GAP head (no class token) | DeiT recipe; GAP found as accurate as class token; drop-path is the main regularizer for big models | class token works too but unnecessary |
| qk scale = d^{-1/2} | standard scaled dot-product (keeps logits ~unit variance) | unscaled → softmax saturates |

## Code structure (canonical microsoft/Swin-Transformer)
- `Mlp`: fc→GELU→drop→fc→drop.
- `window_partition(x,M)` / `window_reverse`: (B,H,W,C) ↔ (nW·B, M, M, C) via view+permute.
- `WindowAttention`: qkv linear, q*scale, attn=q@kᵀ, + relative_position_bias (gathered from table via relative_position_index), optional mask (added reshaped per nW), softmax, @v, proj. Builds `relative_position_index` buffer with the meshgrid/shift/multiply/sum trick.
- `SwinTransformerBlock`: shift_size = 0 (even block) or M//2 (odd). Pre-builds `attn_mask` via the slice-trick (img_mask with 9 region ids → window_partition → pairwise diff → −100 where different). forward: LN → roll(−shift) → partition → attention(mask) → reverse → roll(+shift) → residual; then LN→MLP→residual.
- `PatchMerging`: gather 4 strided sub-grids, concat (4C), LN, linear 4C→2C.
- `BasicLayer`: depth blocks alternating shift 0/M//2, optional downsample (PatchMerging) at end.
- `PatchEmbed`: Conv2d(3,C,kernel=4,stride=4) → flatten → optional LN.
- `SwinTransformer`: patch_embed → (optional abs pos) → 4 BasicLayers (dims C,2C,4C,8C) → LN → GAP → linear head.

## In-frame discipline
Never name "Swin", ViT-as-target, or cite this paper. May name prior art (ViT, ResNet, FPN, DeiT, Vaswani, T5) by author/year. reasoning.md: continuous first-person, no headers, derive complexity + masking + rel-pos indexing inline, walls-and-corrections, code at end.
