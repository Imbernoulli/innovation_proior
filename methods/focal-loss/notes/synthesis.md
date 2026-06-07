# Synthesis — Focal Loss / RetinaNet

## Pain point
Two-stage detectors (R-CNN line) top COCO. One-stage (YOLO, SSD) are faster but ~10-40% relatively worse AP. Question: *why* the gap, and can a one-stage detector close it?

## Diagnostic finding (pre-method, in scope for context Background)
A one-stage detector densely enumerates ~10^4–10^5 candidate locations (anchors) per image; only a handful contain objects. Extreme foreground:background imbalance (e.g. 1:1000). Two problems: (1) training inefficient — most locations are easy negatives, no signal; (2) en masse, easy negatives overwhelm the loss/gradient → degenerate model. This is the central cause of the accuracy gap.

## Ancestors (load-bearing)
- **R-CNN (Girshick 2014)**: region proposals + CNN classifier. Modern era start.
- **Fast R-CNN (Girshick 2015)**: RoI pooling, single-stage training of the classifier; smooth-L1 box loss.
- **Faster R-CNN + RPN (Ren 2015)**: RPN integrates proposal generation; "anchors" — translation-invariant reference boxes at each location, classified obj/not-obj and regressed. The proposal stage *filters* most background → second stage sees a balanced (~1:3) set. This cascade is how two-stage detectors dodge imbalance.
- **OHEM (Shrivastava 2016)**: online hard example mining — score all RoIs by loss, NMS, keep highest-loss ones for the minibatch. Emphasizes hard examples but *completely discards* easy examples; needs sampling machinery, batch-size/NMS knobs.
- **SSD (Liu 2016) / YOLO (Redmon)**: one-stage, dense. SSD uses hard-negative mining keeping a fixed 1:3 neg:pos ratio per batch. Faster, lower AP.
- **FPN (Lin 2017)**: top-down pathway + lateral connections → multi-scale feature pyramid from one image; each level detects one scale band. Built on ResNet.
- **ResNet (He 2016)**: backbone.
- **Bootstrapping / hard-negative mining (Sung&Poggio 1994, Viola-Jones 2001, DPM Felzenszwalb 2010)**: classic fixes for the same imbalance.
- **Robust losses (Huber)**: down-weight outliers (hard examples) — focal loss does the OPPOSITE (down-weights easy inliers).

## Derivation chain
1. CE binary: CE(p,y) = −log p if y=1 else −log(1−p). Define p_t = p if y=1 else 1−p. CE(p_t) = −log p_t.
2. Even easy examples (p_t ≫ .5) incur non-trivial loss: at p_t=0.9, −log(0.9)≈0.105 nats. Sum over ~10^5 easy negs: 0.105·10^5 ≈ 10^4, vs a handful of positives at O(1) each. Easy negs dominate.
3. α-balanced CE: CE = −α_t log p_t. Balances pos/neg *frequency* but NOT easy/hard. Negatives are still mostly easy → still dominate within the negative class.
4. Need a factor that depends on how well-classified the example is. Modulating factor (1−p_t)^γ: →0 as p_t→1 (easy, down-weight), →1 as p_t→0 (hard, untouched). FL(p_t) = −(1−p_t)^γ log p_t.
5. γ focusing parameter. γ=0 ⇒ CE. Quantitative: γ=2, p_t=0.9 ⇒ factor (0.1)^2=0.01 ⇒ 100× down-weight. p_t≈0.968 ⇒ (0.032)^2≈10^-3 ⇒ 1000×. Hard p_t≤0.5 ⇒ factor ≥(0.5)^2=0.25 ⇒ at most 4× down. So easy examples crushed relative to hard. γ=2 best.
6. α-balanced FL: FL = −α_t (1−p_t)^γ log p_t. α=0.25 with γ=2 (α drops as γ rises because down-weighting negs already reduces their pull, so less need to up-weight positives).
7. Numerical stability: combine sigmoid + loss in one layer.

## Prior-π initialization (stability)
Default init → P(foreground)≈0.5 at start. With imbalance, first-iter loss from the frequent (background) class is huge → instability/divergence. Set a prior π for the rare class: bias of final cls conv b = −log((1−π)/π), π=0.01. Then σ(b)=π so every anchor starts as foreground-prob ≈ 0.01. This is an *initialization* change, not a loss change. CE without it diverges; with it, 30.2 AP.

## Appendix A — FL* (form not crucial)
x_t = y·x, p_t = σ(x_t). p_t* = σ(γ x_t + β). FL* = −log(p_t*)/γ. Two knobs γ (steepness), β (shift). Comparable AP. Shows any loss that diminishes weight for well-classified (x_t>0) works.

## Appendix B — derivatives (wrt x)
- dCE/dx = y(p_t − 1)
- dFL/dx = y(1−p_t)^γ (γ p_t log p_t + p_t − 1)   [verified by hand]
- dFL*/dx = y(p_t* − 1)
For high-confidence preds derivative → 0 or −1; FL/FL* derivative small as soon as x_t>0 (unlike CE).

## RetinaNet architecture
FPN-on-ResNet backbone, levels P3–P7 (P3-5 from C3-5 top-down+lateral; P6 = 3×3 s2 conv on C5; P7 = ReLU then 3×3 s2 conv on P6), C=256 channels. A=9 anchors/level (3 aspect {1:2,1:1,2:1} × 3 scales {2^0,2^{1/3},2^{2/3}}), areas 32²–512². Two shared subnets (separate params): cls subnet = 4×(3×3 conv C, ReLU) + 3×3 conv KA + sigmoid; box subnet = same but 4A linear outputs, class-agnostic. IoU≥0.5 = fg, [0,0.4)=bg, [0.4,0.5)=ignore. Focal loss on ALL ~100k anchors, normalized by # anchors assigned to a gt box. Box: smooth-L1. SGD, prior π=0.01 bias init on final cls conv.

## Code grounding
- fvcore `sigmoid_focal_loss`: p=σ(inputs); ce=BCE_with_logits; p_t = p·t + (1−p)(1−t); loss = ce·(1−p_t)^γ; if α≥0: α_t = α·t+(1−α)(1−t); loss*=α_t.
- detectron2 RetinaNetHead: cls/bbox subnets, prior_prob=0.01 → bias = −log((1−p)/p); losses() one-hots labels (drop bg col), sigmoid_focal_loss reduction=sum, normalized by EMA of num_pos_anchors.

## Code-framework scaffold (pre-method, generic one-stage dense detector)
- ResNet/FPN backbone exists; anchor generator exists; smooth-L1 exists; SGD exists.
- Slots: classification subnet head (stub), the classification LOSS (stub — currently CE/BCE), head bias init (stub), loss normalization (stub).
