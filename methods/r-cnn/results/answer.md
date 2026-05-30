# R-CNN: Regions with CNN features

## Problem
Carry the large gains that deep convolutional networks brought to image *classification* over to
*object detection*, which additionally requires localizing possibly many objects per image — while
training a high-capacity CNN from the small amount of box-annotated data available.

## Key idea
Decouple localization from recognition. Use a cheap, category-independent region proposer to
generate ~2000 candidate boxes per image; classify each with a deep CNN (transferred from
ImageNet); and refine boxes with a lightweight regressor. Two principles make it work:
(1) supervised pre-training on a large auxiliary classification task (ImageNet) followed by
domain-specific fine-tuning lets a high-capacity CNN be trained despite scarce detection data;
(2) a CNN applied to bottom-up region proposals localizes and classifies objects without dense
sliding windows.

## The pipeline
1. **Region proposals.** Selective search (fast mode) → ~2000 category-independent boxes.
2. **Warp.** Dilate each proposal by p = 16 px of context, then anisotropically warp it to the
   CNN's fixed 227×227 input (off-image pixels filled with the image mean).
3. **Feature extraction.** Forward each warped region through a CNN pre-trained on ImageNet
   classification (5 conv + 2 fc) → a 4096-d feature vector (shared across all classes).
4. **Classification.** One linear SVM per class scores every feature vector (a single
   2000×4096 by 4096×N matrix product).
5. **Non-maximum suppression.** Greedy NMS per class removes duplicate detections.
6. **Bounding-box regression.** A per-class ridge regressor on the conv features predicts a
   scale-invariant correction that tightens each box.

## Training (three stages)
- **Supervised pre-training:** train the CNN on ImageNet (ILSVRC2012) classification, image-level
  labels only.
- **Domain-specific fine-tuning:** replace the 1000-way head with a random (N+1)-way head (N
  object classes + background), continue SGD on warped proposals at lr = 0.001 (1/10 of the
  pre-training rate). Positives = proposals with ≥ 0.5 IoU to a ground-truth box (loose, to
  manufacture ~30× more "jittered" positives); rest = background. Minibatch of 128 = 32 positives
  + 96 background (biased toward rare positives).
- **Per-class SVMs:** positives = ground-truth boxes only (strict); negatives = proposals with
  < 0.3 IoU to every instance of the class (0.3 chosen by grid search; 0.5 → −5 mAP, 0 → −4 mAP);
  the 0.3–1.0 grey zone is ignored. Trained with hard-negative mining (converges in ~1 pass).
  SVMs beat using the fine-tuned softmax directly (54.2 → 50.9 mAP without them) because of the
  precise positives and hard negatives.

## Bounding-box regression
For a proposal P = (P_x, P_y, P_w, P_h) and target G = (G_x, G_y, G_w, G_h), learn four functions
d_⋆(P) = w_⋆ᵀ φ₅(P), linear in the conv (pool5) features φ₅(P), to predict the refined box:

  Ĝ_x = P_w·d_x(P) + P_x,  Ĝ_y = P_h·d_y(P) + P_y,
  Ĝ_w = P_w·exp(d_w(P)),   Ĝ_h = P_h·exp(d_h(P)).

The regression targets are scale-invariant (center shift normalized by box size) and log-space
(width/height ratios):

  t_x = (G_x − P_x)/P_w,  t_y = (G_y − P_y)/P_h,
  t_w = log(G_w/P_w),     t_h = log(G_h/P_h).

Weights are fit per class by ridge regression,
  w_⋆ = argmin_ŵ Σ_i (t_⋆^i − ŵᵀφ₅(P^i))² + λ‖ŵ‖²,  with λ = 1000,
solved in closed form. Only proposals with max-IoU > 0.6 to a ground-truth box are used as
training pairs (a far-away proposal gives a hopeless target). Applied once per detection at test
time (iterating does not help).

## Why it is efficient
All CNN computation is shared across classes; the only per-class cost is a matrix product against
the SVM weights plus NMS. Because the feature is just 4096-d (vs. ~360k-d for spatial-pyramid
encodings), the approach scales to thousands of classes with no approximation.

## Code
```python
import numpy as np

def prepare_region(image, box, pad=16, size=227):
    """Dilate by `pad` px of context, then anisotropically warp to size x size;
    off-image pixels filled with the image mean (subtracted before the CNN)."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    cx_pad, cy_pad = pad * w / size, pad * h / size
    src = clip_to_image(x1 - cx_pad, y1 - cy_pad, x2 + cx_pad, y2 + cy_pad, image, fill="mean")
    return anisotropic_resize(src, (size, size))


def finetune(cnn, dataset, num_classes, lr=0.001, n_pos=32, n_bg=96):
    cnn.replace_head(num_classes + 1)            # drop 1000-way, add random (N+1)-way head
    opt = SGD(cnn.parameters(), lr=lr)           # 1/10 pretraining LR
    for images, proposals, gts, gt_cls in dataset:
        lab = np.zeros(len(proposals), dtype=int)
        for i, p in enumerate(proposals):
            ious = np.array([iou(p, g) for g in gts] or [0.0])
            if ious.max() >= 0.5:                # LOOSE positives
                lab[i] = gt_cls[ious.argmax()] + 1
        pos, bg = np.where(lab > 0)[0], np.where(lab == 0)[0]
        idx = np.r_[sample(pos, n_pos), sample(bg, n_bg)]   # bias toward positives
        x = np.stack([prepare_region(images, proposals[j]) for j in idx])
        step(opt, cross_entropy(cnn.classify(x), lab[idx]))
    return cnn


def train_svms(feat_cache, proposals, gts, gt_cls, num_classes):
    svms = []
    for c in range(num_classes):
        X_pos = features_of(feat_cache, gts, cls=c)                  # GT-only positives
        X_neg = features_of(feat_cache, proposals, cls=c, rule="iou<0.3")
        w = init_linear_svm()
        for _ in range(num_passes):                                 # hard-negative mining
            w = fit_linear_svm(np.vstack([X_pos, X_neg]),
                               np.r_[np.ones(len(X_pos)), -np.ones(len(X_neg))])
            X_neg = np.vstack([X_neg, X_neg[score(w, X_neg) > -1.0]])
        svms.append(w)
    return np.stack(svms)                                           # (num_classes, 4096)


def bbox_reg_targets(P, G):                       # P,G = (cx,cy,w,h)
    return np.array([(G[0]-P[0])/P[2], (G[1]-P[1])/P[3],
                     np.log(G[2]/P[2]), np.log(G[3]/P[3])])


def fit_bbox_regressors(pool5, proposals, gts, gt_cls, num_classes, lam=1000.0):
    regs = {}
    for c in range(num_classes):
        X, T = [], []
        for f, p in zip(pool5, proposals):
            cand = [g for g in gts if gt_cls_of(g) == c]
            if cand and max(iou(p, g) for g in cand) > 0.6:         # NEARBY proposals only
                X.append(f); T.append(bbox_reg_targets(to_cwh(p), to_cwh(nearest_gt(p, cand))))
        if X:
            regs[c] = ridge_solve(np.array(X), np.array(T), lam)    # closed-form ridge
    return regs


def apply_bbox_reg(W_c, feat, P):
    d = feat @ W_c
    cx, cy, w, h = to_cwh(P)
    return [w*d[0] + cx, h*d[1] + cy, w*np.exp(d[2]), h*np.exp(d[3])]


def detect(image, cnn, svms, regs, num_classes, nms_iou=0.3):
    boxes = selective_search(image, mode="fast")          # ~2000 proposals
    inp = np.stack([prepare_region(image, b) for b in boxes])
    feats = cnn_forward(inp, layer="fc7")                 # 2000 x 4096 (shared)
    pool5 = cnn_forward(inp, layer="pool5")
    scores = feats @ svms.T                               # 2000 x num_classes
    dets = []
    for c in range(num_classes):
        refined = np.array([apply_bbox_reg(regs[c], pool5[i], boxes[i]) if c in regs
                            else to_cwh(boxes[i]) for i in range(len(boxes))])
        keep = nms(cwh_to_xyxy(refined), scores[:, c], nms_iou)
        dets += [(c, scores[i, c], refined[i]) for i in keep]
    return dets
```
