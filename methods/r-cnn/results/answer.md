# R-CNN: Regions with CNN features

## Problem

Turn a high-capacity CNN trained for image classification into an object detector: for each image, output class-labeled boxes, despite scarce box-level training data and a CNN architecture whose high-level features are too coarse for precise dense sliding-window localization.

## Method

1. Generate category-independent region proposals, using selective search fast mode in the reference system.
2. Expand each proposal to include `p = 16` pixels of target-frame context, then anisotropically warp it to the CNN's fixed `227 x 227` input.
3. Forward each warped proposal through an ImageNet-pretrained CNN, fine-tuned on warped detection windows, and use the `fc7` feature as a 4096-dimensional region descriptor.
4. Train one linear SVM per object class on those features.
5. Score all proposals for all classes with one feature-by-weight matrix product, then run greedy NMS per class.
6. Apply a class-specific bounding-box regressor from `pool5` features to tighten scored detections.

## Training

Pretrain the CNN on ImageNet classification with image-level labels. For detection fine-tuning, replace the 1000-way ImageNet head with an `(N+1)`-way head for `N` object classes plus background. Continue SGD on warped proposals with learning rate `0.001`, batch size `128`, and a foreground/background sampling split of `32/96`. A proposal is foreground for fine-tuning if its maximum IoU with a ground-truth box is at least `0.5`; otherwise it is background.

For SVM training, use a stricter class-specific rule. Positives for class `c` are only ground-truth boxes of class `c`. Negatives are proposals whose IoU with every instance of class `c` is below `0.3`; proposals in the grey zone are ignored for that class. Train with hard-negative mining.

## Bounding-Box Regression

For proposal `P = (P_x, P_y, P_w, P_h)` and matched ground-truth box `G = (G_x, G_y, G_w, G_h)`, define targets

```text
t_x = (G_x - P_x) / P_w
t_y = (G_y - P_y) / P_h
t_w = log(G_w / P_w)
t_h = log(G_h / P_h)
```

Learn four class-specific linear functions `d_*(P) = w_*^T phi_5(P)` from `pool5` features by ridge regression:

```text
w_* = argmin_w sum_i (t_*^i - w^T phi_5(P_i))^2 + lambda ||w||^2,
lambda = 1000.
```

Invert the transform as

```text
Ghat_x = P_w d_x(P) + P_x
Ghat_y = P_h d_y(P) + P_y
Ghat_w = P_w exp(d_w(P))
Ghat_h = P_h exp(d_h(P)).
```

The signs and constants are fixed by exact inversion: if `d_* = t_*`, then `Ghat = G`. Train regressors only on proposals whose max IoU with a ground-truth box is greater than `0.6`, once per class. At test time, apply the regressor once, clip the adjusted box to the image, and keep the original SVM score.

## Reference-Faithful Pseudocode

```python
import numpy as np


def prepare_region(image, box, image_mean, crop_size=227, padding=16):
    """Match the R-CNN warp crop rule.

    `padding` is measured in the transformed 227 x 227 frame. Missing
    off-image pixels are mean-valued before mean subtraction, represented
    as zeros after subtraction.
    """
    x1, y1, x2, y2 = map(float, box)

    if padding > 0:
        scale = crop_size / float(crop_size - 2 * padding)
        w = x2 - x1
        h = y2 - y1
        cx = x1 + 0.5 * w
        cy = y1 + 0.5 * h
        half_w = 0.5 * w * scale
        half_h = 0.5 * h * scale
        src = np.round([cx - half_w, cy - half_h,
                        cx + half_w, cy + half_h]).astype(int)
    else:
        src = np.round([x1, y1, x2, y2]).astype(int)

    clipped = clip_box_to_image(src, image.shape)
    patch = image[clipped.y1:clipped.y2, clipped.x1:clipped.x2]

    sx = crop_size / float(src[2] - src[0])
    sy = crop_size / float(src[3] - src[1])
    pad_left = int(round(max(0, -src[0]) * sx))
    pad_top = int(round(max(0, -src[1]) * sy))
    crop_w = min(crop_size - pad_left, int(round(patch.shape[1] * sx)))
    crop_h = min(crop_size - pad_top, int(round(patch.shape[0] * sy)))

    resized = resize_bilinear_no_antialias(patch, (crop_h, crop_w))
    resized = resized - image_mean[pad_top:pad_top + crop_h,
                                   pad_left:pad_left + crop_w]
    window = np.zeros((crop_size, crop_size, 3), dtype=np.float32)
    window[pad_top:pad_top + crop_h, pad_left:pad_left + crop_w] = resized
    return window


def finetune_labels(proposals, gt_boxes, gt_classes):
    labels = np.zeros(len(proposals), dtype=np.int32)  # 0 = background
    for i, box in enumerate(proposals):
        overlaps = np.array([iou(box, gt) for gt in gt_boxes])
        if len(overlaps) and overlaps.max() >= 0.5:
            labels[i] = gt_classes[overlaps.argmax()] + 1
    return labels


def train_svms(feature_cache, roidb, classes, feat_norm_mean):
    svms = {}
    for c in classes:
        X_pos = gt_box_features(feature_cache, roidb, c, layer="fc7")
        X_pos = scale_features(X_pos, feat_norm_mean)

        cache = NegativeCache()
        initialized = False
        for image_id in roidb.image_ids:
            feats, overlaps = proposal_features(feature_cache, image_id,
                                                layer="fc7")
            feats = scale_features(feats, feat_norm_mean)

            if not initialized:
                neg = np.where(overlaps[:, c] < 0.3)[0]
            else:
                scores = feats @ svms[c].w + svms[c].b
                neg = np.where((overlaps[:, c] < 0.3) &
                               (scores > -1.0001))[0]

            cache.add_unique(image_id, neg, feats[neg])
            if cache.needs_retrain() or roidb.is_last(image_id):
                svms[c] = fit_linear_svm(X_pos, cache.features())
                cache.evict_easy(svms[c], threshold=-1.2)
                initialized = True
    return svms


def bbox_targets(src_box, gt_box):
    sx1, sy1, sx2, sy2 = map(float, src_box)
    gx1, gy1, gx2, gy2 = map(float, gt_box)
    sw, sh = sx2 - sx1 + np.finfo(float).eps, sy2 - sy1 + np.finfo(float).eps
    gw, gh = gx2 - gx1 + np.finfo(float).eps, gy2 - gy1 + np.finfo(float).eps
    scx, scy = sx1 + 0.5 * sw, sy1 + 0.5 * sh
    gcx, gcy = gx1 + 0.5 * gw, gy1 + 0.5 * gh
    return np.array([(gcx - scx) / sw, (gcy - scy) / sh,
                     np.log(gw / sw), np.log(gh / sh)])


def train_bbox_regressors(pool5_cache, roidb, classes, feat_norm_mean,
                          lam=1000.0):
    regs = {}
    for c in classes:
        X, Y = [], []
        for image_id in roidb.image_ids:
            feats, boxes, gt_boxes = pool5_cache[image_id]
            for feat, box in zip(feats, boxes):
                gt, ov = nearest_gt_of_class(box, gt_boxes, c)
                if gt is not None and ov > 0.6:
                    X.append(feat)
                    Y.append(bbox_targets(box, gt))
        if not X:
            continue

        X = scale_features(np.asarray(X), feat_norm_mean)
        X = np.c_[X, np.ones(len(X))]  # bias feature, as in the MATLAB code
        Y = np.asarray(Y)
        mu = Y.mean(axis=0)
        Y0 = Y - mu
        T, T_inv = target_whitener(Y0)
        beta = ridge_solve(X, Y0 @ T, lam)
        regs[c] = (beta, mu, T_inv)
    return regs


def apply_bbox_regressor(reg, feat, box):
    beta, mu, T_inv = reg
    y = (np.r_[feat, 1.0] @ beta) @ T_inv + mu
    dx, dy, dw, dh = y

    x1, y1, x2, y2 = map(float, box)
    w, h = x2 - x1 + np.finfo(float).eps, y2 - y1 + np.finfo(float).eps
    cx, cy = x1 + 0.5 * w, y1 + 0.5 * h
    pred_cx = dx * w + cx
    pred_cy = dy * h + cy
    pred_w = np.exp(dw) * w
    pred_h = np.exp(dh) * h
    return np.array([pred_cx - 0.5 * pred_w, pred_cy - 0.5 * pred_h,
                     pred_cx + 0.5 * pred_w, pred_cy + 0.5 * pred_h])


def detect(image, cnn, svms, bbox_regs, svm_feat_norm_mean,
           bbox_feat_norm_mean, image_mean):
    boxes = selective_search(image, mode="fast")
    crops = np.stack([prepare_region(image, b, image_mean) for b in boxes])
    fc7 = scale_features(cnn_forward(crops, layer="fc7"), svm_feat_norm_mean)
    pool5 = scale_features(cnn_forward(crops, layer="pool5"),
                           bbox_feat_norm_mean)

    detections = []
    for c, svm in svms.items():
        scores = fc7 @ svm.w + svm.b
        candidate_boxes = boxes.copy()
        if c in bbox_regs:
            candidate_boxes = np.vstack([
                apply_bbox_regressor(bbox_regs[c], pool5[i], boxes[i])
                for i in range(len(boxes))
            ])
            candidate_boxes = clip_boxes_to_image(candidate_boxes, image.shape)

        keep = nms(candidate_boxes, scores, iou_thresh=0.3)
        detections.extend((c, scores[i], candidate_boxes[i]) for i in keep)
    return detections
```
