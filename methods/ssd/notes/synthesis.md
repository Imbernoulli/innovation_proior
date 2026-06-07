# SSD synthesis notes

## Pain point at the time (2015)
- SOTA detection = "propose-then-classify": Selective Search / RPN proposes ~thousands of boxes, then resample pixels/features per box, classify + refine. Accurate but slow. Faster R-CNN = 7 FPS even on Titan X. Too slow for real-time/embedded.
- The expensive part is the *resampling* (RoI pooling per proposal) and the *two-network* dependency.
- Single-shot attempts existed but were less accurate: OverFeat (sliding window, single topmost feature map, one box per location), YOLO (whole topmost 7x7 feature map → FC layer → 2 boxes/cell + class probs per cell). YOLO is fast (45 FPS) but 63.4 mAP vs Faster 73.2. Two weaknesses: (1) single coarse feature map → bad on small objects and on dense/small clusters; (2) FC layer to predict boxes loses spatial structure and is parameter-heavy; box coords predicted from whole-image features.

## Load-bearing ancestors
- **R-CNN (Girshick 2014)**: region proposals (selective search) + CNN classify each crop. Slow (per-crop CNN).
- **SPPnet (He 2014)**: spatial pyramid pooling → share conv features across proposals, classify from feature map crops. Faster but not end-to-end, multi-stage.
- **Fast R-CNN (Girshick 2015)**: RoI pooling, single-stage training of classifier+bbox regressor with multi-task loss (softmax conf + smooth-L1 loc). Still needs external proposals. **Source of the smooth-L1 loc loss and the bbox regression parameterization.**
- **Faster R-CNN (Ren 2015)**: RPN — slide a small conv net over the last feature map; at each location, *k* **anchor boxes** of multiple scales/aspect ratios; predict objectness + 4 box offsets per anchor relative to the anchor. Then RoI-pool and reclassify with Fast R-CNN head. **Source of: anchor boxes, conv predictors over a feature map, the offset encoding (cx,cy,w,h with log for w,h).** Limitation it leaves: anchors only on ONE feature map (single scale); plus the second RoI-pool/classify stage.
- **MultiBox (Erhan 2014 / Szegedy 2015)**: a CNN directly regresses a fixed set of box proposals (priors clustered from data) + a confidence each, trained with a matching loss (each prior matched to GT by best overlap, bipartite). Class-agnostic objectness only → still needs a second classifier. **Source of: the loss structure (matching + confidence + localization), the "default/prior box" idea, the matching strategy.** Limitation: priors on one feature map; single confidence not per-class; needs follow-up classification net.
- **OverFeat (Sermanet 2013)**: sliding-window detection, fully-convolutional, one box per location from topmost map. Single scale.
- **YOLO (Redmon 2015)**: single net, whole-image FC predicts SxS grid each with B boxes + class probs. Fast, single scale, coarse grid.
- **VGG-16 (Simonyan 2014)**: 3x3 conv stack base network. Truncate before classifier → base feature extractor.
- **FCN / Hypercolumns / ParseNet**: showed lower (higher-res) conv layers carry fine detail useful for localization → motivation for using *multiple* feature maps including earlier ones.
- **Atrous / à trous (Holschneider 1990; DeepLab Chen 2014)**: dilated conv to keep resolution while enlarging receptive field, cheaply. Used to convert VGG fc6/fc7 to conv (conv6 dilation=6) without losing too much resolution / speed.

## The method (derived objects)
1. **Multi-scale feature maps**: take base net (VGG truncated), keep conv4_3, fc7(conv7), then append extra conv layers conv8_2, conv9_2, conv10_2, conv11_2 that progressively shrink (38→19→10→5→3→1 for SSD300). Predict from all 6. Early/large map → small objects; deep/small map → large objects.
2. **Convolutional predictors**: for each chosen feature map (m×n×p), apply a 3×3×p conv. Per location, k default boxes; each predicts c class scores + 4 offsets → (c+4)k filters → (c+4)kmn outputs. (vs YOLO's FC layer.)
3. **Default boxes** (= anchors but on multiple maps): tiled per cell. Center = ((j+0.5)/|f_k|, (i+0.5)/|f_k|). Scales:
   s_k = s_min + (s_max−s_min)/(m−1)·(k−1), s_min=0.2, s_max=0.9.
   Aspect ratios a_r ∈ {1,2,3,1/2,1/3}: w=s_k√a_r, h=s_k/√a_r. For a_r=1 add extra box s'_k=√(s_k·s_{k+1}) → 6 boxes/loc (4 when dropping 3,1/3).
4. **Matching**: each GT → best-jaccard default box (guarantees ≥1 match, from MultiBox bipartite); THEN every default box with jaccard>0.5 to any GT also matched (SSD's relaxation → multiple positives, easier learning). x_ij^p ∈ {0,1}, Σ_i x_ij^p ≥ 1.
5. **Loss**: L = (1/N)(L_conf + α L_loc), N = #matched, α=1.
   L_loc = Σ_{i∈Pos} Σ_{m∈{cx,cy,w,h}} x_ij^k smooth_L1(l_i^m − ĝ_j^m), with
   ĝ_cx=(g_cx−d_cx)/d_w, ĝ_cy=(g_cy−d_cy)/d_h, ĝ_w=log(g_w/d_w), ĝ_h=log(g_h/d_h).
   L_conf = −Σ_{Pos} x_ij^p log(ĉ_i^p) − Σ_{Neg} log(ĉ_i^0), softmax over classes (incl. background class 0).
6. **Hard negative mining**: after matching, most defaults negative → imbalance. Sort negatives by confidence loss (highest = hardest), keep top so neg:pos ≤ 3:1. Faster, more stable.
7. **Data augmentation**: sample patches with min-jaccard {0.1,0.3,0.5,0.7,0.9} / random / whole; patch size [0.1,1] of image, AR [1/2,2]; flip 0.5; photometric. Plus "zoom out" expansion (place on 16× canvas) for small objects.
8. **Atrous base**: VGG fc6→conv6 dilation 6, fc7→conv7 1×1; pool5 3×3-s1; à trous. ~20% faster than full VGG, same accuracy.
9. **L2Norm on conv4_3**: conv4_3 has larger feature magnitudes/different scale → L2-normalize per location and learn a scale (init 20).
10. **Inference**: conf threshold 0.01 to prune, per-class NMS jaccard 0.45, keep top 200.

## Implementation-detail notes (canonical ssd.pytorch)
- Encoding adds *variance* division: g_cxcy /= variance[0]*prior_wh (variance[0]=0.1), g_wh=log(...)/variance[1] (variance[1]=0.2). The variances are not in the paper's eq (they come from MultiBox/Faster R-CNN code); decode multiplies back. I'll include them as the standard impl does, noting they're a normalization of the regression targets.
- mbox = [4,6,6,6,4,4] boxes per location for the 6 maps. Total = 38²·4+19²·6+10²·6+5²·6+3²·4+1²·4 = 8732.
- VGG source layers indices 21 (conv4_3 relu) and -2 (conv7 relu).
- L_loc uses smooth_l1 sum over positives; L_conf cross-entropy over pos+selected neg; both divided by N=num_pos.

## Design-decision → why
- *Multiple feature maps* not image pyramid: share params, one forward pass, cheap. Each map specializes to a scale band.
- *Conv predictor* not FC (YOLO): preserves spatial layout, far fewer params, predicts offsets *relative to local default box* not global coords → easier.
- *Default boxes of several AR per cell*: discretize output box space; separate predictors per AR specialize. Ablation: dropping {1/3,3} −0.6, dropping {1/2,2} another −2.1.
- *Match-by-overlap>0.5 (not just best)*: lets several overlapping defaults fire → smoother target, faster learning (MultiBox only kept best → harder).
- *log encoding for w,h*: width/height positive & multiplicative; log makes regression target additive/scale-invariant, smooth-L1 well-behaved. (From Fast/Faster R-CNN.)
- *Hard neg mining 3:1*: 8732 boxes, almost all background → without it the conf loss is swamped; sampling hardest negatives keeps signal, stabilizes.
- *Atrous*: keep resolution + big receptive field cheaply.
- *s_min=0.2..s_max=0.9 linear*: regular coverage of scales across the 6 maps; smaller (0.1/0.07) on conv4_3 because it's the only high-res map for small objects.
