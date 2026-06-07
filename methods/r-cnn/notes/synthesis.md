# R-CNN synthesis (grounded in 1311.2524 src + selective search + DPM)

## Pain point (in-frame, ~2012-2013)
- PASCAL VOC detection progress stalled 2010-2012; gains from ensembling HOG/SIFT-based systems (DPM v5 ~33% mAP VOC2010).
- AlexNet (Krizhevsky 2012) showed CNNs dominate ImageNet *classification*. Open question (ILSVRC2012 workshop debate): does classification accuracy transfer to *detection*?
- Two sub-problems: (1) localizing objects with a deep net; (2) training high-capacity CNN with scarce detection data.

## Localization options considered (dead ends)
- Regression of box coords directly: Szegedy et al concurrent — fares poorly (30.5% VOC07 vs R-CNN 58.5%).
- Sliding window: CNNs used for faces/pedestrians with only 2 conv+pool layers to keep resolution. AlexNet has 5 conv layers; high units have receptive field 195x195, stride 32x32 → precise localization in sliding-window is open challenge. Rejected.
- Chosen: "recognition using regions" paradigm (Gu 2009). Region proposals.

## Three modules
1. Region proposals: category-independent. Use selective search (fast mode), ~2000 per image. (Method-agnostic but SS chosen for controlled comparison with UVA/Regionlets.)
2. CNN feature extractor: Caffe AlexNet. 4096-d feature from each warped 227x227 RGB region (mean-subtracted), 5 conv + 2 fc.
3. Class-specific linear SVMs.

## Warping (appendix A)
- CNN needs fixed 227x227. Options: tightest square w/ context (B), tightest square w/o context (C), anisotropic warp (D). Plus context pad p.
- Chosen: anisotropic warp with context pad p=16 px. Beat alternatives by 3-5 mAP. Missing data (region beyond image) filled with image mean (subtracted later).

## Selective search (Uijlings 2013) — ancestor
- Start: Felzenszwalb-Huttenlocher graph-based segmentation → initial regions.
- Greedy hierarchical grouping: compute similarities of neighboring regions (color, texture, size, fill/shape compatibility), merge two most similar, recompute, repeat until single region. Emit all regions across the hierarchy. Diversification over color spaces & similarity measures. High recall (~99% at ~10k locations), class-independent, data-driven. Fast mode ~2000.

## DPM (Felzenszwalb) — baseline being beaten
- Star model: root HOG filter (Dalal-Triggs-like) + part filters at higher resolution + deformation cost (spring) per part. Score = root + sum parts placed at best deformed positions − deformation penalties. Latent SVM training (part positions are latent). HOG features. mAP ~33% VOC2010. Also does bbox regression from part locations.

## UVA / spatial pyramid BoW — most germane baseline (same SS proposals)
- 4-level spatial pyramid, densely sampled SIFT + Extended OpponentSIFT + RGB-SIFT, each vector-quantized w/ 4000-word codebooks → ~360k-dim. Histogram intersection kernel SVM. 35.1% mAP VOC2010. Two orders of magnitude higher-dim features than R-CNN's 4096 → slow.

## OverFeat — ILSVRC2013 rival
- Sliding-window CNN detector. Best on ILSVRC2013 detection at 24.3% mAP. R-CNN beats with 31.4%.

## Training pipeline (three stages — KEY design decisions)
### Supervised pre-training
- Pre-train CNN on ILSVRC2012 classification (image-level labels only, no boxes). ~2.2 pts top-1 worse than Krizhevsky due to simplifications.
### Domain-specific fine-tuning
- Continue SGD on warped proposals. Replace 1000-way classif layer with random (N+1)-way (N classes + background). VOC N=20, ILSVRC N=200.
- Positives for fine-tuning: proposal with >=0.5 IoU with a GT box → positive for that box's class. Rest = background.
- SGD lr = 0.001 (1/10 of pre-train initial rate) so as not to clobber init.
- Mini-batch 128: 32 positive windows (over all classes) + 96 background. Bias toward positives (rare).
### SVM training
- Positives = ONLY ground-truth boxes for each class.
- Negatives = proposals with <0.3 IoU with all instances of a class. 0.3 threshold from grid search over {0,0.1,...,0.5} on val. Setting 0.5 → -5 mAP; setting 0 → -4 mAP.
- Grey zone (0.3<IoU<1, not GT) ignored.
- Hard negative mining (Sung & Poggio 1994; Felzenszwalb lsvm) — too much data for memory. Converges in 1 pass.

### Why pos/neg defined differently FT vs SVM (appendix B)
- Historical: started with SVMs on pretrained features (no FT yet); the SVM label def was optimal among options tried.
- When FT added, reusing SVM defs gave much worse results than current FT defs.
- Hypothesis: not fundamental; arises because FT data limited. Current FT scheme adds many "jittered" examples (0.5-1 IoU, not GT) → ~30x more positives → needed to avoid overfitting when fine-tuning ENTIRE net. But jittered = suboptimal for precise localization.

### Why SVM at all (vs softmax)?
- Cleaner to just use the 21-way softmax of FT'd net. Tried it: VOC07 mAP dropped 54.2 → 50.9.
- Drop likely from: FT positive def doesn't emphasize precise localization; softmax trained on randomly sampled negs not hard negs.

## Test-time detection
- SS → ~2000 proposals → warp → CNN → 4096-d feats.
- For each class, score each feature vector with that class's SVM (matrix-matrix product: 2000x4096 feats · 4096xN weights).
- Greedy NMS per class: reject region if IoU with a higher-scoring selected region > learned threshold.
- Run-time: feature compute 13s/img GPU, 53s/img CPU, amortized over classes. Only class-specific cost = dot products + NMS. Scales to 100k classes (10s matmul, 1.5GB).

## Bounding-box regression (appendix C) — full derivation
- Input: N training pairs {(P^i, G^i)}. P=(P_x,P_y,P_w,P_h) center+w+h pixels. G same.
- Learn transform mapping P→G via four functions d_x(P),d_y(P),d_w(P),d_h(P).
- d_x,d_y = scale-invariant translation of center; d_w,d_h = log-space translation of w,h.
- Predicted box:
  Ĝ_x = P_w d_x(P) + P_x
  Ĝ_y = P_h d_y(P) + P_y
  Ĝ_w = P_w exp(d_w(P))
  Ĝ_h = P_h exp(d_h(P))
- Each d_*(P) = w_*^T φ5(P), linear on pool5 features φ5(P).
- Learn w_* by ridge regression:
  w_* = argmin Σ_i (t_*^i − ŵ_*^T φ5(P^i))^2 + λ||ŵ_*||^2
- Targets:
  t_x = (G_x − P_x)/P_w
  t_y = (G_y − P_y)/P_h
  t_w = log(G_w / P_w)
  t_h = log(G_h / P_h)
- Closed-form (standard regularized least squares).
- Two subtle issues: (1) regularization important, λ=1000 from val. (2) only use pairs where P is *near* a GT: assign P to max-IoU GT iff IoU > 0.6 (val). Discard unassigned. Per-class regressors.
- Test: predict new window once per proposal. Iterating doesn't help.
- WHY this parameterization: t_x,t_y normalized by P_w,P_h → scale/translation invariant (a fixed pixel error on a big box is a smaller relative target). Log-space for w,h → invariant to scale & keeps Ĝ_w,Ĝ_h positive (exp). Same spirit as DPM bbox regression but from CNN features not DPM part geometry.

## Feature visualization (sec 5.1) — diagnostic, in-frame
- pool5 feature map 6x6x256 = 9216-d. Each pool5 unit receptive field 195x195 in 227x227 input.
- Non-parametric viz: pick a unit, compute activations over ~10M held-out proposals, sort, NMS, show top regions. Units align to concepts (people, text), textures (dot arrays), materials (specular). → pool5 = small set of class-tuned features + distributed shape/texture/color/material rep.

## Canonical implementation
- Official: github.com/rbgirshick/rcnn (MATLAB/Caffe). Structure: selective_search → im_crop/warp → caffe forward (fc7 features) → per-class SVM (liblinear) → bbox regression (ridge) → nms. Code in Phase 2 mirrors this structure in Python/numpy/torch-style pseudocode faithful to that pipeline.

## Design-decision → why table
- Region proposals not sliding window: deep net receptive field/stride too coarse for sliding-window localization.
- Warp (anisotropic) + p=16 context: simplest; beat tightest-square variants by 3-5 mAP; context helps.
- Supervised pre-train then FT: detection data too scarce to train high-cap CNN from scratch; ImageNet supervised pretrain transfers; FT adapts to detection domain (+8 mAP).
- FT positives IoU>=0.5 (loose): generate ~30x jittered positives to avoid overfitting whole net on scarce data.
- SVM positives GT-only, negs IoU<0.3: precise positives + hard negatives → better localization than softmax (54.2 vs 50.9).
- 0.3 neg threshold: grid-searched; 0.5 → -5, 0 → -4.
- SVM over softmax: +3.3 mAP from precise positives + hard-neg mining.
- Hard negative mining: bg overwhelms; can't fit all in memory.
- bbox reg parameterization (center normalized by w/h, log w/h): scale/translation invariance + positivity.
- λ=1000 ridge: regularization critical (val).
- 0.6 IoU for bbox-reg training pairs: far P → hopeless mapping.
- Greedy per-class NMS: many proposals fire on same object → dedup.
```
