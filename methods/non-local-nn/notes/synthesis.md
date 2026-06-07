# Non-local Neural Networks synthesis (grounded in 1711.07971 src + non-local means + self-attention + ResNet/I3D)

## Pain point
- Capturing LONG-RANGE dependencies is central. Sequential data: recurrent ops. Image: large receptive fields from deep conv stacks.
- BOTH conv & recurrent process a LOCAL neighborhood (space or time). Long-range only captured by REPEATING local ops, propagating signal progressively. Problems with repeating local ops:
  1. Computationally inefficient.
  2. Optimization difficulties (need careful handling — cf vanishing gradients, residual nets).
  3. Multi-hop dependency (messages back & forth between distant positions) hard.
- Want: an efficient, simple, generic component capturing long-range deps DIRECTLY (interaction between ANY two positions regardless of distance), in space/time/spacetime.

## Ancestors
### Non-local means (Buades 2005) — the seed
- Classical denoising filter: response at a location = weighted mean of ALL pixels, weighted by patch appearance similarity. Distant pixels contribute. Later → BM3D (block-matching 3D, groups similar non-local patches). Non-local matching underlies texture synthesis (Efros 1999), super-res, inpainting.

### Self-attention (Vaswani 2017) — sibling
- Response at a position in a sequence = attend to all positions, weighted average in embedding space. KEY INSIGHT: self-attention is a SPECIAL CASE of non-local means (embedded Gaussian + softmax). This work bridges self-attention (NLP) to non-local filtering (vision), generalizing to space/spacetime for images/video.

### Relation/Interaction networks (Santoro, Battaglia)
- Process all pairs of positions; concatenation pairwise function. Insight: the NON-LOCALITY (orthogonal to attention/relation framing) is the key to success, not attention per se.

### vs fully-connected layer
- fc uses LEARNED weights (relationship NOT a function of input data); non-local relationship IS a function of input (f(x_i,x_j)). fc needs fixed size, loses positional correspondence; non-local supports variable size, preserves position i→i.

### vs conv/recurrent
- conv sums weighted input in LOCAL neighborhood (e.g. i-1≤j≤i+1, kernel 3); recurrent at time i uses current+latest (j=i or i-1). Non-local: ∀j (all positions).

## Generic formulation (Eq 1)
y_i = (1/C(x)) Σ_{∀j} f(x_i, x_j) g(x_j)
- i = output position index (space/time/spacetime), j enumerates ALL positions. x input, y output same size.
- f = pairwise function → scalar (affinity/relationship between i and all j).
- g = unary function → representation of input at j.
- C(x) = normalization factor.

## Instantiations (NOT sensitive to choice — generic non-local behavior is what matters)
- g(x_j) = W_g x_j (linear embedding), implemented as 1×1 conv (space) / 1×1×1 conv (spacetime).
### f choices:
1. Gaussian: f = e^{x_i^T x_j} (dot-product similarity; Euclidean also works but dot-product is implementation-friendly). C(x) = Σ_j f.
2. Embedded Gaussian: f = e^{θ(x_i)^T φ(x_j)}, θ=W_θ x_i, φ=W_φ x_j. C(x) = Σ_j f.
   - SELF-ATTENTION SPECIAL CASE: (1/C) f = softmax over j → y = softmax(x^T W_θ^T W_φ x) g(x), exactly self-attention form.
3. Dot product: f = θ(x_i)^T φ(x_j). C(x) = N (number of positions), NOT Σf — simplifies gradient; needed because input variable size. Difference from embedded Gaussian = NO softmax.
4. Concatenation (from Relation Networks): f = ReLU(w_f^T [θ(x_i), φ(x_j)]). C(x) = N.
- Softmax (attentional behavior) NOT essential — dot-product/concat versions work too. Non-locality is the key, not attention.

## Non-local Block (Eq 6)
z_i = W_z y_i + x_i
- Residual connection (+x_i). W_z = position-wise embedding on y_i, matching channels back to x.
- KEY: if W_z initialized as ZERO, the block is an IDENTITY mapping initially → can insert into ANY pretrained model without breaking it.
- Pairwise computation by matrix multiplication (for Gaussian/embedded/dot-product); concat straightforward.

## Implementation
- Bottleneck: W_g, W_θ, W_φ have HALF the channels of x (cf ResNet bottleneck). W_z maps back up. ~½ the compute.
- Subsampling trick: y_i = (1/C(x̂)) Σ f(x_i, x̂_j) g(x̂_j), x̂ = subsampled (pooled) x. Spatial pooling → ¼ the pairwise computation. Doesn't alter non-local behavior, just sparser. = max pool after φ and g.
- BN: add BN right after the last 1×1×1 (W_z); BN scale param init ZERO (Goyal) → block starts as identity. (Same identity-init goal via zero-init.) Other layers in block: no BN. Weight init He 2015.
- Lightweight when on high-level subsampled maps (T=4, H=W=14 or 7); comparable to a conv layer.

## Video models
### C2D baseline (ResNet-50/101)
- 2D conv processing frame-by-frame (1×k×k kernels). Temporal handled ONLY by pooling. Init from ImageNet ResNet. Input 32×224×224.
### I3D (inflated 3D)
- Inflate 2D k×k → 3D t×k×k spanning t frames. Init from 2D: each of t planes = pretrained k×k weights rescaled by 1/t (so static repeated frame → same as 2D model). Inflate either 3×3→3×3×3 or first 1×1→3×1×1, one kernel per 2 residual blocks (more = diminishing). conv1 inflated to 5×7×7.
### Non-local net
- Insert non-local blocks into C2D or I3D. Study 1, 5, 10 blocks.

## Diagnostic findings (in-frame, drive design)
- Instantiation insensitivity: Gaussian/embedded/dot/concat all similar → non-locality is the cause, not the f choice.
- Stage placement: res2/res3/res4 good; res5 worse (small spatial 7×7 → too few positions for non-local to gather rich info).
- More blocks help (1<5<10), saturating; deeper non-local model better.
- space/time/spacetime: spacetime best (long-range in both).
- Non-local vs I3D: non-local more accurate AND cheaper than 3D conv (1.2× FLOPs vs I3D 1.5-1.8×). Complementary to 3D conv (NL I3D best).
- Add blocks to res2/3/4 (every other residual block for 5/10-block models), to keep on higher-res maps.

## Training (video)
- ImageNet pretrain. 32-frame clips (crop 64 consecutive, drop every other). Spatial 224×224 from scaled video shorter side ∈ [256,320].
- 8-GPU, 8 clips/GPU = batch 64. 400k iters, lr 0.01, /10 every 150k. Momentum 0.9, wd 1e-4. Dropout 0.5 after global pool.
- BN ENABLED during fine-tuning (unlike usual frozen-BN ResNet fine-tuning) — reduces overfitting.

## COCO extension
- Add non-local blocks to Mask R-CNN (1 block before last res-block of res4). Improves detection/segmentation/keypoint at small extra cost. Shows generality beyond video.

## Canonical implementation
facebookresearch/video-nonlocal-net (Caffe2). NLNet block: theta/phi/g = 1×1×1 conv to C/2; affinity = matmul(theta_flat, phi_flat); softmax (embedded Gaussian) or scale 1/N (dot-product); out = matmul(softmax, g_flat); W_z = 1×1×1 conv back to C + BN(scale init 0); residual add. Subsample via maxpool on phi,g. Phase-2 code mirrors in PyTorch.

## Design-decision → why table
- Non-local op (all positions ∀j): captures long-range deps directly in one layer, vs repeating local conv/recurrent (inefficient, hard optimization, poor multi-hop).
- f = pairwise affinity, g = unary embedding: separate "how related" from "what content", mirroring non-local means.
- g = W_g x (1×1 conv): cheapest learned per-position embedding.
- Multiple f options, softmax not essential: shows non-locality (not attention) is the source of gains; dot-product C=N avoids softmax & eases gradient.
- C=N for dot-product/concat (vs Σf): variable input size needs normalization; N is simplest & gradient-friendly.
- Residual block z = W_z y + x with W_z (or BN scale) zero-init: lets block be dropped into any pretrained net as identity, breaking nothing.
- Bottleneck (half channels): halve compute, ResNet-style.
- Subsampling (pool φ, g): ¼ pairwise compute, non-local behavior preserved (still attends across whole map, just sparser keys).
- Place on res3/res4 (subsampled high-level maps): pairwise cost ∝ N² so cheap when N small; res5 too small (few positions).
- BN enabled in fine-tuning: reduces overfitting (large video model).
- Dot-product similarity over Euclidean: implementation-friendly (matmul).
```
