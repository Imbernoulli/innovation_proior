# YOLO — Synthesis Notes (Phase 1.5)

## The pain point (research question)
Object detection in 2014-15 = "where + what" for every object. State of the art (R-CNN family, DPM) was accurate but SLOW and built as a pipeline of separately-trained stages. None ran in real time. Goal: a detector that is (a) fast enough for streaming video (>=30 fps), (b) trained end-to-end against a single objective that corresponds to detection, (c) reasons about the whole image at once (global context) so it makes fewer background false positives.

## Field state / load-bearing ancestors

### Detection = repurposed classifier (the prevailing paradigm)
The dominant idea: take a classifier, slide/evaluate it over many locations & scales, then post-process.
- **DPM (Felzenszwalb, Girshick, McAllester, Ramanan 2010)**: HOG features + a root filter + part filters with a deformable (star) spatial model, scored by a latent-SVM, run as a **sliding window** over an image pyramid. Disjoint stages: static hand-engineered features, classify each window, predict boxes for high-scoring regions, NMS. Strong *spatial* model (why it generalizes to artwork) but low absolute accuracy and slow. Only "30Hz DPM" (Sadeghi & Forsyth 2014) reached real time, by heavy GPU/cascade engineering, and at low mAP.
- **R-CNN (Girshick, Donahue, Darrell, Malik 2014)**: Selective Search (Uijlings 2013) proposes ~2000 category-agnostic regions; warp each; run a CNN per region to extract features; per-class linear SVMs score them; a linear regressor refines boxes; NMS. Huge accuracy jump (CNN features) but: ~2000 forward passes/image, selective search is 1-2 s/image on CPU, every stage trained separately, >40 s/image. Sees only the cropped region — no global context, so it hallucinates objects in background patches.
- **Fast R-CNN (Girshick 2015)**: one conv pass over the whole image, RoI-pooling per proposal, softmax + box regressor jointly trained. Faster (~0.5 fps) and more accurate, but **still depends on Selective Search** for proposals (the ~2 s/image bottleneck), so not real time. Top error mode (per Hoiem diagnostic methodology): background false positives (13.6% of top detections) — a consequence of classifying isolated proposals.
- **Faster R-CNN (Ren, He, Girshick, Sun 2015)**: learns proposals with an RPN, shares conv features. Faster (7-18 fps) but still two-stage, still below real time at high accuracy.

### Detection = direct regression with a CNN (the minority lineage YOLO grows from)
- **OverFeat (Sermanet et al. 2013)**: one ConvNet does classification + localization; the FC layers are applied convolutionally so a sliding-window of box regressions is computed efficiently in one net. BUT optimizes localization, not detection; localizer sees only local windows; needs heavy post-processing to merge into coherent detections; disjoint.
- **MultiBox / Scalable detection (Erhan/Szegedy et al. 2014)**: a CNN directly regresses a fixed set of class-agnostic region proposals + confidences (instead of Selective Search). Shows a net can *predict boxes* directly. But class-agnostic, still just a proposal stage inside a larger pipeline.
- **MultiGrasp (Redmon & Angelova 2014)** — the direct seed of the grid idea: regress a single graspable rectangle from an image with a CNN, using a coarse grid. YOLO generalizes the grid-regression idea to many objects of many classes.

### Backbone lineage
- **Network-in-Network (Lin, Chen, Yan 2013)**: 1x1 conv "mlpconv" layers add representational power cheaply and reduce channel dimensionality.
- **GoogLeNet / Inception (Szegedy et al. 2014)**: 22 layers, inception modules, 1x1 reductions before expensive 3x3/5x5 to kill the compute bottleneck. YOLO drops inception, keeps the cheap idea: alternate 1x1 reduction + 3x3 conv. 24 conv + 2 FC. Pretrained on ImageNet at 224, then detection at 448.

## The method as a chain of decisions → why

| Decision | Why this, not the alternative |
|---|---|
| Frame detection as ONE regression from pixels → boxes+classes | Pipelines are slow & can't be optimized end-to-end on detection; each stage tuned separately. A single net = one objective = directly optimize detection + real-time speed. |
| Single forward pass, no proposals | Selective Search (~2 s) is the bottleneck; ~2000 region passes are wasteful. One pass over the full image gives global context (fewer background FPs) and speed. |
| Divide image into S×S grid; cell whose center contains object is "responsible" | Need a fixed-size output tensor for a CNN regressor + a way to spatially divide labor so predictions don't collapse to one box. Center-assignment makes each object owned by exactly one cell → enforces spatial diversity, suppresses duplicate detections (cheap built-in NMS-like effect). From MultiGrasp's grid. |
| Each cell predicts B=2 boxes (x,y,w,h,confidence) | One box/cell underfits when a cell must cover varied aspect ratios; B>1 lets predictors **specialize** (one for tall, one for wide). B=2 is the cheap minimum that gives specialization; output stays small. |
| Confidence ≡ Pr(Object)·IOU(pred,truth) | Want a score that is high only when (a) there IS an object and (b) the box is accurate. Product of "is there an object" and "how good is the box" = exactly IOU when object present, 0 when absent. The target for confidence is the *actual IOU*, not a constant 1 — teaches calibrated localization quality. |
| ONE set of C class probs per cell, Pr(Class_i\|Object), shared across the B boxes | Class is a property of the *location/object*, not of each candidate box; the B boxes are competing hypotheses for the *same* object in that cell. Sharing keeps the output tensor small (S·S·(B·5+C) not S·S·B·(5+C)) and matches the "one object per cell" constraint. |
| Test-time class-specific confidence = Pr(Class_i\|Object)·Pr(Object)·IOU | Multiply the cell's class prob by each box's confidence → per-box, per-class score encoding both class likelihood and localization quality, usable for thresholding/NMS. |
| S=7, B=2, C=20 → 7×7×30 tensor | VOC has 20 classes; 7×7 is the spatial resolution after the conv stack downsampling; 2 boxes. 98 boxes/image total (7·7·2) vs ~2000 proposals. |
| Backbone: 24 conv (1x1 reductions + 3x3) + 2 FC, leaky ReLU(0.1) | GoogLeNet-inspired but simpler (no inception). 1x1 reductions = cheap NIN/Inception trick. FC layers do the box/class regression. Leaky ReLU avoids dead units. |
| Pretrain conv on ImageNet @224, then detect @448 | Detection data is scarce; ImageNet pretraining gives good features. Detection needs fine detail → double resolution. Add 4 conv + 2 FC randomly-initialized on top (Ren et al. show adding conv+FC to pretrained nets helps). |
| Normalize w,h by image size; x,y as offset within cell → all in [0,1] | Bounded targets stabilize regression; x,y relative to cell ties the box to the responsible cell. |
| Loss = sum-squared error | Easy to optimize. But misaligned with mAP, so it needs the fixes below. |
| λ_coord = 5 on coordinate terms | Localization and classification weighted equally is wrong; most cells are empty so localization signal is rare — upweight it. |
| λ_noobj = 0.5 on no-object confidence | Most cells have no object → their "push confidence to 0" gradient swamps the few object cells, driving the net to predict ~0 everywhere → divergence. Downweight the empty-cell confidence loss. |
| Predict sqrt(w), sqrt(h) instead of w,h | SSE weights equal absolute errors equally; a 10px error matters far more on a 30px box than a 300px box. sqrt compresses large values so the same absolute error in a big box contributes less gradient — partially equalizes the IOU impact. |
| At train time only the responsible predictor (highest current IOU with truth) gets coord+obj gradient | Want one box/object → pick the box already fitting best so it keeps improving; the other is freed to specialize elsewhere. Drives specialization, improves recall. |
| Confidence target for responsible box = its IOU (rescore) | Matches the confidence definition; teaches the net to predict its own localization quality. (Darknet `rescore=1`.) |
| NMS at test | Mostly handled by grid spatial constraint, but large/border objects can be claimed by multiple cells; NMS adds 2-3% mAP. Not critical (unlike for R-CNN/DPM). |
| Linear activation on final layer, leaky elsewhere | Outputs are coordinates/probs in [0,1] regression targets; linear final lets it regress freely; sigmoid not used in v1. |
| Dropout 0.5 after first FC; random scale/translate ±20%, HSV exposure/sat ×1.5 | Detection data scarce → overfitting risk; standard regularization + augmentation. |
| LR warmup 1e-3→1e-2, then 1e-2 (75) →1e-3 (30) →1e-4 (30); 135 epochs; momentum 0.9, decay 5e-4, batch 64 | High initial LR diverges (unstable gradients from the SSE + many empty cells); warm up then step down. |

## Limitations (in-frame, knowable at design time)
- Strong spatial constraint: each cell → 2 boxes, 1 class → struggles with groups of small objects (flocks of birds).
- Coarse features (many downsampling layers) → trouble with unusual aspect ratios.
- SSE treats large/small box errors the same except for the sqrt patch → localization is the dominant error source.

## Canonical code
- Darknet `detection_layer.c` (official, C): memory layout per batch = [all class probs (S·S·C)] then [all confidences (S·S·B)] then [all box coords (S·S·B·4)]. Responsible box = max IOU (RMSE fallback when no overlap). `rescore=1` → confidence target = IOU. coord_scale=5, noobject_scale=0.5, object_scale=1, class_scale=1, sqrt=1.
- Clean PyTorch reimpl (aladdinpersson): per-cell layout [C | conf1,x,y,w,h | conf2,x,y,w,h]; YoloLoss picks bestbox by IOU, sqrt on w,h (with sign/abs guard for stability), λ_coord=5, λ_noobj=0.5. Yolov1 model = architecture_config (24 conv) + FC(→4096→S·S·30). This is the basis for the deliverable code.

## In-frame discipline
- Do NOT name the target paper, authors, "YOLO paper", arXiv, etc. May name the method "YOLO" as the thing being built (mainly answer.md).
- Prior-art citations (Girshick 2014, Felzenszwalb 2010, Sermanet 2013, Szegedy 2014, Erhan 2014, Redmon&Angelova 2014, Lin 2013, Uijlings 2013, Ren 2015, Hoiem 2012) stay and get elaborated.
- No proposed-method eval results (52.7/63.4 mAP, fps wins). Background FP rate of Fast R-CNN (13.6%) is a *diagnostic finding about an existing system* → allowed in context.
