# Mask R-CNN — synthesis notes (Phase 1.5)

## The pain point / research question
Instance segmentation = detect every object AND produce a pixel-accurate mask for *each* instance.
It sits between two mature-but-separate tasks:
- **Object detection** (bounding boxes + class): Fast/Faster R-CNN are fast, accurate, flexible.
- **Semantic segmentation** (per-pixel class, no instances): FCN is the clean fully-conv baseline.
The open problem: a *simple, fast, general* framework for instance segmentation, on par with what
Fast/Faster R-CNN did for detection and FCN did for semantic segmentation. Prior instance-seg systems
were complex multi-stage cascades, slow, and segmentation-first (segment then classify).

## Load-bearing ancestors (verified against paper)

### R-CNN (Girshick 2014)
Attend to ~2k region proposals (selective search), run a CNN on each warped crop, classify with SVM,
regress box. Core idea: regions + CNN features. Limitation: warps each crop and runs the full CNN per
region — extremely slow, multi-stage training (CNN, SVM, regressor separately).

### Fast R-CNN (Girshick 2015) — RoIPool + parallel heads
Run the conv backbone ONCE over the whole image to get a feature map. For each RoI, **RoIPool** extracts
a fixed small feature map (e.g. 7×7) from the shared map; then two sibling **fc** heads predict
(a) softmax class over K+1 classes, (b) per-class box regression. Trained end-to-end with a multi-task
loss L = L_cls + λ L_box (smooth-L1 box loss on the GT-class box only). Key shift: **parallel** class+box
in one net, single-stage training, shared computation. This "parallel siblings" pattern is the template
Mask R-CNN copies.
- **RoIPool mechanics**: a floating RoI [x1,y1,x2,y2] in image coords is mapped to the feature map by
  dividing by stride and ROUNDING: `[x/16]`. The quantized RoI is then split into a 7×7 grid of bins;
  each bin boundary is again rounded; max-pool inside each bin. Two rounding steps → the pooled features
  are misaligned with the actual RoI by up to ~half a bin (several pixels at stride 16/32). Fine for
  classification (translation-robust), DESTRUCTIVE for pixel-accurate masks.

### Faster R-CNN (Ren 2015) — RPN
Replaces selective search with a **Region Proposal Network**: a small conv net slides over the shared
feature map; at each location, k anchors (multi-scale, multi-aspect); two 1×1 conv siblings predict
objectness (2k) and box deltas (4k). Trained with the same cls+box multi-task loss. Two-stage detector:
stage 1 = RPN proposes; stage 2 = Fast R-CNN classifies/refines on RoIPool features. The two stages
share the backbone. This is THE base framework Mask R-CNN extends — identical stage 1, augmented stage 2.

### FCN (Long 2015) — fully convolutional semantic segmentation
Drop the fc layers; make the whole net convolutional so output is a spatial map. Predict per-pixel class
with a **per-pixel softmax** over C classes + multinomial cross-entropy. Upsample (deconv/bilinear) back
to input resolution. Key idea Mask R-CNN borrows: **use convolutions to keep spatial layout** — a mask
is spatial, so predict it with a small FCN, NOT by collapsing to an fc vector. Key thing Mask R-CNN
REJECTS: the per-pixel softmax couples segmentation with classification (pixels compete across classes).

### FPN (Lin 2017) — feature pyramid
Top-down pathway + lateral connections build a multi-scale in-network pyramid {P2..P5} from a single-scale
input, all at 256 channels. RoIs are assigned to a pyramid level by size: k = floor(k0 + log2(sqrt(wh)/224)).
Gives strong multi-scale features cheaply; an efficient detector head (shared across levels). Mask R-CNN
uses ResNet-FPN as its best backbone; the mask RoIs are pooled from the assigned level.

### Spatial Transformer Networks (Jaderberg 2015) — differentiable bilinear sampling
Introduced **differentiable bilinear sampling** of a feature map at arbitrary (non-integer) coordinates.
This is the primitive RoIAlign reuses: sample feature values at exact float locations via bilinear
interpolation from the 4 nearest grid points — no rounding, fully differentiable w.r.t. the sampled values.

### DeepMask / SharpMask (Pinheiro 2015/2016) — segment proposals, fc masks
Learn to PROPOSE class-agnostic segment candidates with a net that outputs a mask via an **fc** layer,
then classify each segment with Fast R-CNN. Segmentation-FIRST. Limitations: fc mask loses spatial
structure / needs many params; segment-then-classify is slow and less accurate; masks not aligned to
detection.

### MNC (Dai 2016) — multi-task cascade + RoIWarp
Instance-aware seg via a 3-stage cascade: box proposals → mask instances → categorize. Introduced
**RoIWarp**: also uses bilinear resampling (à la STN) — BUT still quantizes the RoI first (like RoIPool),
so it overlooks the alignment issue. Empirically RoIWarp ≈ RoIPool, much worse than RoIAlign. The cascade
makes later stages DEPEND on mask predictions (coupling) and is complex/multi-stage.

### FCIS (Li 2017) — fully convolutional instance seg
Position-sensitive output channels predict class+box+mask fully convolutionally → fast. But shares a
single set of position-sensitive maps across tasks; exhibits systematic artifacts on overlapping
instances and spurious edges. The shared-channels coupling hurts on hard overlap cases.

## The method that drops out

Add a THIRD sibling branch to Faster R-CNN's stage 2: a small FCN that outputs masks, in **parallel**
with the existing class + box siblings. Three design decisions are load-bearing.

### Design-decision → why table

| Decision | Why this, why not the alternative |
|---|---|
| **Parallel mask branch** (not segment-then-classify) | Follows Fast R-CNN's parallel-siblings win. Segment-first (DeepMask, MNC, FCIS) makes classification depend on masks → multi-stage, slow, error-coupling. Parallel = simple, fast, each task independent. |
| **Mask = small FCN, not fc** (m×m conv output) | A mask is *spatial*; fc collapses spatial layout into a vector (DeepMask), losing structure and needing more params. FCN keeps explicit m×m correspondence pixel-to-pixel. Ablation: FCN +2.1 mask AP over MLP. |
| **Per-class binary masks + per-pixel SIGMOID + binary CE** (Km² output, decoupled) | The mask branch should only answer "is this pixel part of THE object", not "which class" — the box branch already classifies. Softmax (FCN-style) makes classes COMPETE per pixel, coupling seg with classification. Decoupling: predict K masks, apply L_mask only on the GT-class channel k. Ablation: sigmoid 30.3 vs softmax 24.8 (+5.5 AP). Class-agnostic (single mask) nearly as good (29.7) → confirms classification is the box branch's job. |
| **L = L_cls + L_box + L_mask, equal weights, L_mask only on positive RoIs & only channel k** | Multi-task training; mask loss is avg binary CE over the m² pixels of the k-th mask. Defined only on positive RoIs (IoU≥0.5 with a GT). Other K-1 channels get no gradient → no inter-class competition. Multi-task even *helps* box AP (+0.9). |
| **RoIAlign: no quantization + bilinear sampling** | RoIPool's two roundings (`[x/16]`, bin rounding) misalign features with the RoI by up to ~half a bin → fine for class (translation-robust) but destroys pixel-accurate masks. RoIAlign uses `x/16` (no rounding), samples 4 points per bin by bilinear interp (STN primitive), then max/avg. Ablation: +3 AP at stride16, **+7.3** at stride32, +4.4 keypoint AP. RoIWarp (quantizes then bilinear) ≈ RoIPool → proves ALIGNMENT, not bilinear, is the cause. Insensitive to #points / max-vs-avg as long as no quantization. |
| **m=14 then deconv→28 (FPN) / 14 (C4)** | Output mask resolution; 2× deconv (transpose conv stride 2) upsamples 14→28 keeping it learnable+spatial. Keypoints need finer → 56×56. |
| **4×(conv 3×3, 256) FCN head (FPN)** | Keep 256-d spatial features through 4 convs, then 2×2 deconv stride2 + ReLU, then 1×1 conv to K channels. Straightforward; more complex designs not the focus. |
| **Sampling: 1:3 pos:neg, N=64 (C4)/512 (FPN); RoI pos iff IoU≥0.5** | Inherited from Fast/Faster R-CNN; system robust to these. |
| **Inference: box branch → NMS → top-100 boxes → mask branch on those; pick channel k=argmax class; resize m×m to box; threshold 0.5** | Decouples train (masks on positive proposals) from test (masks on FINAL detections) → fewer, more accurate RoIs, ~20% overhead only. Only the predicted-class mask is used. |
| **Keypoints: one-hot m×m mask per keypoint, m²-way SOFTMAX (not sigmoid), 56×56** | A keypoint is a SINGLE location → exactly one foreground pixel → softmax over m² (one-hot target) is the right model, unlike masks where many pixels are foreground (sigmoid). High res needed for localization. |
| **Backbone backbone/head split; ResNet-C4, ResNet-FPN, ResNeXt** | Decoupling backbone from head shows generality; FPN > C4 (multi-scale), deeper/ResNeXt better. |

## Key derivations to live out in reasoning.md
1. RoIPool quantization arithmetic: float RoI → `[x/16]` → bin rounding → misalignment magnitude (≈ stride/2 px), and why classification tolerates it but masks don't.
2. RoIAlign: continuous bin boundaries `x/16`; 4 sample points per bin at bin/4, 3·bin/4 fractions; bilinear interpolation weights from 4 neighbors; aggregate (avg). Show it's the STN sampler. Show RoIWarp = quantize-then-bilinear is the control proving alignment is the cause.
3. The mask loss: Km² output; per-pixel sigmoid σ(z); L_mask = -(1/m²) Σ_pixels [ y log σ(z_k) + (1-y) log(1-σ(z_k)) ] on channel k only; contrast with FCN softmax+multinomial that couples classes.
4. Why decoupling works: the "division of labor" — box branch owns classification, mask branch owns the binary shape; class-agnostic ≈ class-specific confirms it.
5. Multi-task loss equal weighting; mask only on positives; box loss on GT-class column only (smooth-L1).
6. Inference vs training discrepancy and why it's a net win.

## Canonical implementation (torchvision, grounded)
- `MaskRCNNHeads`: 4× Conv2d(3×3, 256) + ReLU.
- `MaskRCNNPredictor`: ConvTranspose2d(256,256,2,2,0) + ReLU + Conv2d(256, num_classes, 1).
- `maskrcnn_loss`: project GT masks onto proposals via roi_align(M,M); BCE-with-logits on channel = GT label only.
- `maskrcnn_inference`: sigmoid; pick channel = predicted label; one mask per detection.
- `MultiScaleRoIAlign(output_size=14, sampling_ratio=2)` for masks (box uses 7).
- Box head `TwoMLPHead` (fc6/fc7 1024) + `FastRCNNPredictor` (cls + per-class box).
- `RoIHeads.forward`: train → masks on positive proposals; test → masks on post-NMS top detections.
