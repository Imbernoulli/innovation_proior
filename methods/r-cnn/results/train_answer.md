Object detection on PASCAL VOC demands more than an image-level label: a detector must name every object of interest and return a tight box for each, where a prediction only counts when it has the right class and overlaps an unmatched ground-truth box by at least $0.5$ intersection-over-union. The large convolutional networks that just transformed ImageNet classification are too strong to ignore, but two facts block a naive transfer. First, a classifier can discard object position once it has enough evidence for a label, whereas a detector is judged on geometry; recognition and localization have to happen together. Second, a high-capacity ImageNet CNN has tens of millions of parameters while detection datasets with box labels are small, so training such a network from scratch on a few thousand boxed images invites severe overfitting. The standing alternatives each fail in a specific way. Regressing object coordinates directly from the whole image is conceptually clean, but the signal is too global and ambiguous — an image holds an unknown number of instances — and concurrent deep-regression results on VOC are not competitive. A dense sliding-window classifier is familiar, but a deep ImageNet-style network has units with large receptive fields and a coarse effective stride, so its high-level features are excellent for recognition invariance and poor for precise window placement; shrinking the network to recover resolution throws away the very hierarchy that motivated the attempt. Meanwhile the hand-designed-feature systems that lead the leaderboard — deformable part models on HOG, selective-search spatial-pyramid bag-of-words detectors — gain mostly through ensembles, context rescoring, and minor variants over a visual code that is fixed by the designer rather than learned.

I propose R-CNN: regions with CNN features. What makes it work is not a single formula but a deliberate division of labor that lets each older tool do the part it is good at while the CNN supplies the missing learned representation. The first move is to stop asking the CNN to search over all positions and scales. A category-independent proposal process answers "where might an object be?" without knowing the class, and the CNN answers "what does this region look like?" Selective search in fast mode supplies the proposals: it oversegments the image, then greedily merges neighboring regions using plural similarity terms — color, texture, size, and fill — over several color spaces, yielding roughly two thousand high-recall, class-independent boxes per image instead of hundreds of thousands of sliding windows. Each arbitrary proposal rectangle must then become the CNN's fixed $227 \times 227$ input. I considered preserving aspect ratio by enclosing the box in a tight square (which admits a lot of background) and masking out the surrounding context (which produces an unnatural input unlike ImageNet crops); the simpler and better choice is to anisotropically warp the proposal, accepting shape distortion in exchange for filling the input with the candidate, while adding a small band of context because objects are not recognized in isolation. Concretely I expand the proposal so the original box carries a border of $p = 16$ pixels in the transformed frame. Because the padding is defined at the target scale, the source crop is grown by the factor $227 / (227 - 2p)$ with rounding and clipping — not by a naive $p/227$ fraction of the original box — and off-image pixels are filled with the image mean before mean subtraction, becoming zeros after it.

With proposals and warping fixed, the architecture is nearly forced: for each image generate the proposals, warp and mean-subtract each, forward it through the CNN, and read out the $4096$-dimensional $fc7$ vector as the region descriptor. This shares the expensive convolutional work across all classes — one feature matrix per image, then a single feature-by-weight matrix product against the per-class linear classifiers, then per-class duplicate suppression. The scarcity of box labels is solved by supervised transfer: pretrain the CNN on ImageNet classification, then adapt it by replacing the $1000$-way head with an $(N+1)$-way head for $N$ object classes plus background and continuing SGD on warped proposals at the reduced learning rate $0.001$, so fine-tuning adjusts the initialization without destroying it, with batch size $128$ and a foreground/background split of $32/96$ that compensates for how rare positive proposals are. Fine-tuning uses a deliberately loose label rule — a proposal is foreground if its maximum IoU with any ground-truth box is at least $0.5$, otherwise background — because updating the whole network benefits from many jittered positive examples around each object. That loose rule is good for learning features but wrong for the final decision boundary, so I do not trust the fine-tuning softmax as the detector. Instead I train one linear SVM per class on the frozen features, where the clean positives for class $c$ are only the ground-truth boxes of $c$ and the negatives are proposals whose IoU with every instance of $c$ falls below $0.3$, with the grey-zone proposals ignored for that class. The $0.3$ threshold is load-bearing: validation shows that moving it to $0.5$ or to $0$ costs several AP points, because the classifier then either swallows too many imprecise near-positives or treats too many ambiguous regions as hard negatives. Since most windows are background and most are trivially easy, the SVMs are fit with hard-negative mining — start from low-overlap negatives, scan images adding only margin-violating or high-scoring negatives, retrain, and evict negatives that have become safely easy — so memory is spent on confusing examples. Greedy non-maximum suppression then removes duplicate hits per class from the redundant proposal set.

The residual errors point to the last component. If the features are doing their job, many false positives are not background confusions but boxes that sit on or near the right object yet are not tight enough — exactly the regime where regression is sensible as a small correction from a plausible proposal to a nearby ground-truth box, even though regression failed as the whole localization mechanism. The transform must be scale-invariant, because a fixed pixel error means very different things for a tiny box and a large one. So for a proposal $P = (P_x, P_y, P_w, P_h)$ in center/width/height coordinates and its matched ground truth $G$, I define the center targets as fractions of the proposal size and the size targets in log space, the latter so that doubling and halving are symmetric and predicted sizes stay positive:
$$t_x = (G_x - P_x)/P_w, \quad t_y = (G_y - P_y)/P_h, \quad t_w = \log(G_w/P_w), \quad t_h = \log(G_h/P_h).$$
I learn four class-specific linear functions $d_*(P) = w_*^{\top}\phi_5(P)$ of the proposal's $pool5$ feature by ridge regression,
$$w_* = \arg\min_w \sum_i \big(t_*^{\,i} - w^{\top}\phi_5(P_i)\big)^2 + \lambda\,\lVert w\rVert^2, \qquad \lambda = 1000,$$
and invert the parameterization at test time as
$$\hat G_x = P_w\, d_x(P) + P_x, \quad \hat G_y = P_h\, d_y(P) + P_y, \quad \hat G_w = P_w\,e^{d_w(P)}, \quad \hat G_h = P_h\,e^{d_h(P)}.$$
The signs and constants are fixed by exact inversion: if $d_* = t_*$, then $\hat G = G$ — the center shift carries the $P_w$ factor and the size correction is exponential on width and height only, with no stray sign or scale constant. The fit adds practical conditioning — scaling the features, appending a bias feature, and centering and decorrelating the four targets before solving, then unwhitening at test time — none of which changes the target definitions or the inverse. I also restrict the training pairs: asking a linear model to drag a proposal that is nowhere near any object across the image is meaningless, so a proposal is assigned to its max-overlap ground truth only when that overlap exceeds $0.6$, regressors are class-specific, the correction is applied once per scored detection, the adjusted box is clipped to the image, and the original SVM score is kept; validation says a single pass suffices. Region proposals handle the coarse search, warped windows let a fixed-input CNN score arbitrary regions, ImageNet pretraining plus fine-tuning supplies the representation when box labels are scarce, strict SVMs with hard-negative mining sharpen the boundary, NMS removes duplicates, and the regressor tightens the boxes.

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
