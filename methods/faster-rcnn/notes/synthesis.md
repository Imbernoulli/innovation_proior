# Faster R-CNN — synthesis notes (Phase 1.5)

## Pain point at the time (2015)
- State-of-the-art detection = region-based CNNs. Pipeline: (1) external region proposal algorithm produces ~2000 candidate boxes; (2) a CNN classifies/refines each.
- SPPnet (He 2014) and Fast R-CNN (Girshick 2015) crushed the per-region CNN cost by sharing one full-image conv pass and pooling per-RoI (SPP / RoI pooling). Detection conv+region-wise now ~150-300ms on GPU.
- This EXPOSED the proposal step as the bottleneck. Selective Search = ~2s/image on CPU (order of magnitude slower than the detector). EdgeBoxes = ~0.2s, best quality/speed tradeoff, but still ~= the whole detector's time.
- Naive fix: reimplement Selective Search on GPU. But that's pure engineering; it ignores the detector entirely and misses the chance to SHARE computation. The proposer and detector both need image features — recomputing them twice is waste.

## Key insight / discovery order
1. The conv feature map Fast R-CNN already computes for detection ALSO contains everything needed to propose regions. So compute proposals as a small head ON TOP of the shared conv features → marginal proposal cost ~10ms.
2. Make the proposer fully convolutional (FCN, Long 2015): slide a tiny network (n×n=3×3 conv → ReLU → two sibling 1×1 convs) over the feature map. Sharing the FC across positions = sliding window = conv. This is the Region Proposal Network (RPN).
3. The multi-scale problem. Old ways: (a) image pyramid (resize image to many scales, recompute features each — slow); (b) filter pyramid (multiple filter sizes, e.g. DPM 5×7,7×5). Both enumerate scales explicitly = costly. New: ANCHORS — at each sliding location predict k boxes parameterized RELATIVE to k reference boxes of fixed scales/aspect ratios. "Pyramid of regression references." Single image scale, single filter size. This is what makes feature sharing free of extra scale cost.
4. reg layer: 4k outputs (box deltas). cls layer: 2k scores (object/not). k=9 (3 scales {128²,256²,512²} × 3 ratios {1:1,1:2,2:1}).
5. Anchor box regression parameterization (from R-CNN): t_x=(x−x_a)/w_a, t_y=(y−y_a)/h_a, t_w=log(w/w_a), t_h=log(h/h_a). Center offsets normalized by anchor size; size in log space. Scale/translation invariant. k SEPARATE regressors (one per anchor shape), not weight-shared across shapes — because features are a fixed 3×3, can't tell which scale, so each anchor "owns" its regressor.
6. Multi-task loss (follows Fast R-CNN): L = (1/N_cls)Σ L_cls(p_i,p*_i) + λ(1/N_reg)Σ p*_i L_reg(t_i,t*_i). L_cls = log loss over 2 classes. L_reg = smooth-L1 (robust, less sensitive to outliers than L2). p*_i gates reg to positives only. N_cls=256 (minibatch), N_reg~2400 (anchor locations), λ=10 to balance. Insensitive to λ.
7. Label assignment: positive if (i) highest IoU with a GT, OR (ii) IoU>0.7 with any GT. Negative if IoU<0.3 for all GT. In between = ignored. Rule (i) is a fallback for when no anchor clears 0.7.
8. Translation invariance: anchors + functions are translation invariant (up to network stride). Contrast MultiBox: 800 k-means anchors, NOT translation invariant; huge FC output layer (6.1M params) → overfitting risk. RPN output layer tiny (~2.8e4 params).
9. Training RPN: image-centric sampling (1 image/minibatch), sample 256 anchors, up to 1:1 pos:neg (pad with neg if <128 pos). New layers init N(0,0.01); shared convs from ImageNet pretrain. lr 0.001 (60k) then 0.0001 (20k), momentum 0.9, wd 0.0005. Cross-boundary anchors IGNORED in training (else huge uncorrectable error terms → no convergence); at test, clip to image.
10. NMS on proposals at IoU 0.7 → ~2000; take top-N for detection. NMS doesn't hurt mAP.

## Feature sharing / training schemes
- Problem: RPN and Fast R-CNN trained independently each modify the shared convs differently. Need a way to actually share one set of convs.
- (i) Alternating training (4-step, used in paper): Step1 train RPN from ImageNet init. Step2 train Fast R-CNN using step1 proposals, also from ImageNet init (convs NOT yet shared). Step3 init RPN from detector, FIX shared convs, fine-tune only RPN-unique layers (now shared). Step4 FIX shared convs, fine-tune only FR-CNN-unique layers. Now both share convs = unified net.
- (ii) Approximate joint training: merge into one net; forward generates proposals treated as fixed precomputed RoIs; backward combines RPN-loss and FR-CNN-loss into shared convs. Ignores gradient wrt proposal coordinates (they're network outputs too) → approximate. ~25-50% faster than alternating, close results. In released Python code.
- (iii) Non-approximate joint: would need RoI pooling differentiable wrt box coords (RoI warping, Dai 2015). Out of scope.

## Implementation details
- Single scale, shorter side s=600. Total conv stride 16 for ZF and VGG. ~60×40×9 ≈ 20000 anchors per 1000×600 image; ~6000 after dropping cross-boundary.
- ZF: 5 conv layers, 256-d intermediate. VGG-16: 13 conv, 512-d intermediate. n=3 receptive fields 171 (ZF), 228 (VGG); predictions can exceed receptive field (infer extent from middle).

## One-stage vs two-stage (design rationale, not proposed eval)
- OverFeat = one-stage class-specific: regressor+classifier on sliding windows over a scale pyramid; one aspect ratio per window; simultaneously locates AND categorizes.
- RPN = first stage of a two-stage cascade: class-AGNOSTIC proposals from square 3×3 windows + anchors of varied scale/ratio, then Fast R-CNN attends/refines with adaptively pooled features from proposal boxes. Two-stage cascade with adaptive pooling on the actual proposal region → more accurate.

## Canonical code (py-faster-rcnn, rbgirshick) — grounded files in code/
- generate_anchors.py: base_size=16, ratios [0.5,1,2], scales 2**[3,4,5]=[8,16,32] → 9 anchors. _ratio_enum keeps area constant while changing ratio (ws=round(sqrt(size/r)), hs=round(ws*r)); _scale_enum multiplies w,h by scale.
- bbox_transform.py: bbox_transform (compute t* targets), bbox_transform_inv (apply deltas to anchors), clip_boxes.
- anchor_target_layer.py: build labels (1/0/-1), targets, inside/outside weights; filter inside-image anchors; overlaps; subsample to RPN_BATCHSIZE=256, fg fraction 0.5; pos overlap 0.7, neg 0.3.
- proposal_layer.py: shift anchors over grid, apply deltas, clip, filter min_size, sort by score, pre_nms_topN (6000), NMS 0.7, post_nms_topN (300).
- proposal_target_layer.py: samples RoIs for the Fast R-CNN head (fg/bg, regression targets per class).

## Design-decision → why table
- Compute proposals on shared conv map (not separate net / GPU-SS): reuse features already computed → near-free proposals + jointly optimizable.
- FCN sliding 3×3 window: weight sharing across positions = translation invariance + tiny param count; 3×3 keeps params low while receptive field is large.
- Two sibling 1×1 convs (cls,reg): conv form of the per-position FC pair; FCN-friendly.
- Anchors (vs image/filter pyramid): single-scale image + single filter size → cheap; scale handled by reference boxes in the regression, not by recomputation. Enables free sharing.
- 3 scales × 3 ratios (k=9): covers PASCAL object range; not tuned per dataset; robust.
- Relative log-space box parameterization: scale-invariant targets, bounded/normalized; standard from R-CNN.
- k separate (non-shared) regressors: features are fixed 3×3 size → can't infer scale → each anchor shape needs its own regressor.
- smooth-L1 reg loss: robust to outliers vs L2 (no exploding gradients on large errors).
- λ=10, N_cls=256, N_reg~2400: balances the two unequally-normalized terms; results insensitive to λ.
- pos/neg IoU 0.7/0.3 + highest-IoU fallback: clean positives, clean negatives, ambiguous ignored; fallback guarantees ≥1 positive per GT.
- ignore cross-boundary anchors in training: else large uncorrectable error terms break convergence.
- NMS 0.7 on proposals: dedup highly-overlapping proposals; doesn't hurt mAP, cuts proposal count.
- image-centric 256 sample, ≤1:1: negatives dominate (~6000 anchors mostly bg); balance to avoid bias.
- 4-step alternating: pragmatic way to actually share one conv set between two tasks that each want to move the convs; converges fast.
