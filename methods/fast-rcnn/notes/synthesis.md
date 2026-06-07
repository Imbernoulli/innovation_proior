# Fast R-CNN synthesis (grounded in 1504.08083 src + SPPnet 1406.4729 + R-CNN)

## Pain point (~2015)
R-CNN works (great mAP) but its pipeline is ugly and slow:
1. Multi-stage training: (a) fine-tune ConvNet on proposals w/ log loss, (b) fit per-class SVMs to features (replacing softmax), (c) learn bbox regressors. 3 separate stages.
2. Expensive in space & time: features extracted from EVERY proposal in EVERY image, written to disk. VGG16 on VOC07 trainval (5k imgs) = 2.5 GPU-days, hundreds of GB.
3. Test slow: ConvNet forward pass per proposal, no sharing → VGG16 47s/image (GPU).
Root cause of slowness: forward pass per proposal, no shared computation.

## SPPnet (He et al. 2014, 1406.4729) — the key ancestor / partial fix
- Compute conv feature map for WHOLE image ONCE. Then for each proposal, extract fixed-length feature by spatial-pyramid-pooling the portion of the feature map inside the proposal (multiple grid sizes e.g. concatenated, like SPP / Lazebnik 2006). Shares conv computation → 10-100x faster test, 3x faster training.
- DRAWBACK: still multi-stage (features→FT log loss→SVMs→bbox reg), features still written to disk. AND fine-tuning CANNOT update conv layers below the SPP layer. → limits accuracy of very deep nets.
- WHY can't backprop through SPP: when each training sample (RoI) comes from a DIFFERENT image (R-CNN/SPPnet sampling), each RoI has huge receptive field (often whole image). Forward must process the entire receptive field → training inputs are huge → backprop through SPP from disparate images is wildly inefficient.

## Fast R-CNN contributions
1. Higher mAP than R-CNN/SPPnet.
2. Single-stage training w/ multi-task loss.
3. Training updates ALL layers (incl conv).
4. No disk caching of features.

## Architecture
- Input: whole image + set of object proposals (RoIs).
- Whole image → conv+maxpool layers → conv feature map.
- For each RoI: RoI pooling layer → fixed HxW feature map (e.g. 7x7 for VGG16).
- → fc layers → branch into TWO sibling outputs:
  (a) softmax over K+1 classes (K object + background),
  (b) bbox regression offsets, 4 numbers PER class (4K total).

## RoI pooling layer
- Max-pool features inside any RoI into fixed HxW (H,W are hyperparams independent of RoI).
- RoI = (r,c,h,w): top-left (r,c), height/width (h,w) in conv-feature-map coords.
- Divide h×w RoI into H×W grid of sub-windows of approx size (h/H)×(w/W), max-pool each into corresponding output cell. Per-channel independently.
- = special case of SPP with a single pyramid level. Uses SPPnet's sub-window calc.

## RoI pooling backward
- x_i = i-th input activation; y_rj = j-th output of r-th RoI. y_rj = x_{i*(r,j)}, i*(r,j) = argmax_{i'∈R(r,j)} x_{i'}.
- A single x_i may feed several outputs y_rj.
- ∂L/∂x_i = Σ_r Σ_j [i = i*(r,j)] ∂L/∂y_rj   (route gradient through argmax switches, accumulate).

## Initializing from pretrained ImageNet net (AlexNet/VGG_CNN_M/VGG16)
Three transforms:
1. Replace last max-pool with RoI pooling layer (H=W=7 for VGG16, compatible with first fc).
2. Replace last fc + 1000-way softmax with two sibling layers ((K+1)-softmax + per-class bbox regressors).
3. Modify net to take two inputs: images + list of RoIs.

## Fine-tuning: hierarchical sampling (the enabling trick)
- SGD minibatches sampled HIERARCHICALLY: sample N images, then R/N RoIs from each.
- RoIs from same image SHARE computation/memory in fwd & bwd passes.
- Small N → less minibatch compute. N=2, R=128 → ~64x faster than sampling 1 RoI from 128 different images (R-CNN/SPPnet way). THIS is what makes backprop through RoI pooling (and into conv layers) efficient.
- Concern: RoIs from same image correlated → slow convergence? Not an issue in practice; good results w/ N=2,R=128 in fewer SGD iters than R-CNN.

## Multi-task loss (single-stage joint training)
- Two sibling outputs per RoI: p=(p_0..p_K) softmax over K+1; bbox offsets t^k=(t^k_x,t^k_y,t^k_w,t^k_h) per class k.
- t^k parameterization = same as R-CNN (scale-invariant translation + log-space w/h shift relative to proposal).
- Each RoI labeled w/ GT class u and GT bbox target v.
- L(p,u,t^u,v) = L_cls(p,u) + λ[u≥1] L_loc(t^u,v).
- L_cls(p,u) = −log p_u (log loss).
- L_loc(t^u,v) = Σ_{i∈{x,y,w,h}} smooth_L1(t^u_i − v_i).
- smooth_L1(x) = 0.5x² if |x|<1, else |x|−0.5. Robust L1, less sensitive to outliers than L2 (R-CNN/SPPnet used L2). L2 w/ unbounded targets needs careful LR tuning to avoid exploding gradients; smooth_L1 eliminates this.
- [u≥1] Iverson bracket: background (u=0) has no GT box → L_loc ignored.
- Normalize GT regression targets v_i to zero mean unit variance.
- λ=1 in all experiments.

## Mini-batch sampling
- N=2 images uniformly random per minibatch, R=128 (64 RoIs/image).
- 25% of RoIs = foreground: from proposals w/ IoU≥0.5 to a GT box (labeled u≥1).
- 75% = background: proposals w/ max IoU in [0.1, 0.5). Labeled u=0. Lower threshold 0.1 = heuristic for hard-example mining (lsvm/DPM).
- Horizontal flip prob 0.5. No other augmentation.

## SGD hyperparams
- fc softmax init N(0, 0.01²); bbox reg fc init N(0, 0.001²); biases 0.
- Per-layer LR 1 for weights, 2 for biases; global LR 0.001.
- VOC07/12 trainval: 30k iters at 0.001, then 10k at 0.0001.
- Momentum 0.9, weight decay 0.0005.

## Scale invariance
- (1) brute force single-scale: process image at fixed size, net learns scale invariance. s = shortest side = 600px, cap longest at 1000, keep aspect ratio. (VGG16 fits GPU.)
- (2) image pyramid multi-scale: 5 scales {480,576,688,864,1200} (SPPnet's), cap 2000. RoI assigned to scale where scaled RoI closest to 224² area. Multi-scale training = random pyramid scale per image (augmentation).
- Finding: single-scale ≈ multi-scale (deep nets learn scale invariance); multi-scale small mAP gain at large compute cost. Use single-scale s=600 everywhere.

## Truncated SVD for faster detection
- For detection, many RoIs → ~half forward time in fc layers (unlike classification).
- fc layer W (u×v) ≈ U Σ_t V^T (SVD, top t singular vals/vecs).
- Param count uv → t(u+v). Replace single fc by two fc (no nonlinearity between): first = Σ_t V^T (no bias), second = U (original bias).
- VGG16: top 1024 of fc6 (25088×4096), top 256 of fc7 (4096×4096). fc6+fc7 were 45% of time.

## Motivating/diagnostic findings (in-frame, drive design)
- Multi-task training helps: joint > stage-wise > cls-only. Shared rep, tasks help each other (+0.8 to +1.1 mAP on pure cls).
- Single-scale ≈ multi-scale (deep nets learn scale invariance).
- Softmax ≈ SVM (softmax +0.1 to +0.8): one-shot FT sufficient; softmax adds inter-class competition. → drop SVMs.
- Fine-tuning conv layers matters for VGG16: freezing 13 conv (only fc learn, = SPPnet single-scale) drops 66.9→61.4. But conv1 generic (no effect); only fine-tune conv3_1 and up for VGG16 (9 of 13) for speed/memory.
- More proposals not always better: sparse proposals act as a cascade (Viola-Jones); mAP rises then falls as proposal count grows; dense 45k boxes → 52.9 mAP (SVM even worse 49.3). AR doesn't correlate w/ mAP as #proposals varies.

## Test-time detection
- Forward pass: image + ~2000 proposals (sometimes ~45k). Per RoI → p + per-class bbox offsets.
- Detection confidence for class k = p_k. Per-class NMS (R-CNN's algo/settings).

## Canonical implementation
github.com/rbgirshick/fast-rcnn (Python + Caffe). Structure: ROIPooling layer (Caffe C++/CUDA), VGG16 backbone w/ RoIPool replacing pool5, two sibling fc heads (cls_score softmax + bbox_pred), SmoothL1Loss layer, hierarchical RoI minibatch sampler (roidb), multi-task loss. Phase-2 code mirrors this in PyTorch-style.

## Design-decision → why table
- Shared conv map (RoI pooling, not per-proposal forward): R-CNN recomputes CNN per proposal (47s/img); share to be fast.
- RoI pooling = single-level SPP: SPP gives fixed-length feature from variable region; single level is simplest, enough.
- Hierarchical sampling (N=2, R=128): makes backprop through RoI pooling INTO conv layers efficient (RoIs share receptive field/compute) — fixes SPPnet's frozen-conv limitation. ~64x faster than cross-image sampling.
- Single-stage multi-task loss: removes the 3-stage R-CNN pipeline; tasks share rep & help each other (multi-task > stage-wise).
- smooth_L1 over L2: robust to outliers, no exploding-gradient LR tuning on unbounded targets.
- λ=1, normalize targets to zero-mean-unit-var: balance two losses on comparable scale.
- 25% fg / 75% bg, bg IoU∈[0.1,0.5): class balance; 0.1 floor = implicit hard-neg mining.
- softmax not SVM: one-shot FT suffices, softmax adds inter-class competition, no disk caching.
- Fine-tune conv3_1+ for VGG16: conv FT matters for deep nets (66.9 vs 61.4 frozen); conv1 generic; balance speed/memory.
- Truncated SVD on fc: fc dominates detection time w/ many RoIs; cheap rank-t factorization.
- Single-scale s=600: deep nets learn scale invariance; multi-scale not worth compute.
```
