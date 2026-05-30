Let me start from what's actually stuck. Detection on PASCAL VOC has barely moved for two or three years. The strong systems all rest on SIFT and HOG — blockwise histograms of gradient orientation, a single hand-designed feature stage — and the leaderboard creeps forward only by ensembling and tweaking. At the same time a large convolutional network just blew the doors off ImageNet classification, training on 1.2 million images with ReLUs and dropout. So the obvious thing to ask, the thing everyone was arguing about, is whether that classification gain transfers to detection. Classification takes one image and emits one label. Detection has to say *where* every object is, and there might be many. That extra demand — localization — is the whole gap, and there's a second gap underneath it: a network with tens of millions of weights wants on the order of a million labeled examples, and detection datasets have thousands of boxed images. Two problems then: localize with a deep net, and train a high-capacity model on scarce boxed data. Let me take them one at a time and see where each pushes me.

Localization first. How do I get a network that classifies whole images to instead tell me where things are? The cleanest idea is to regress box coordinates straight out of the image — feed the image, predict four numbers per object. But concurrent results on exactly this are discouraging: directly regressing boxes lands around 30% mAP on VOC2007, well short of where I'd want to be. Regression of geometry from a single global feature seems to be a weak signal. Set it aside.

Second idea, the classic one: sliding window. Slide the classifier across every position and scale, and wherever it fires, that's a detection. CNNs have actually been used this way for a long time on narrow categories — faces, pedestrians — but always with shallow nets, two convolution-and-pooling layers, *specifically* so that the spatial resolution stays high and a window's location is pinned down precisely. My network is five conv layers deep. Let me actually compute what that does to localization. By the time I'm at the top conv layer, a single unit is looking at about a 195×195 pixel patch of the input and the units step in strides of about 32 pixels. So the feature map is coarse: positions are quantized to a 32-pixel grid and each carries a receptive field nearly the size of a small object. Sliding a classifier on that grid can tell me roughly *there's a cat in this region*, but "precise localization within the sliding-window paradigm" with that receptive field and stride is an unsolved problem. I could shrink the network to recover resolution, but then I throw away the very depth that makes the features good. That's the wrong trade. So sliding window is out for a deep net.

What's left is to decouple *where might something be* from *what is it*. Don't ask the network to localize at all — hand it candidate regions and let it only classify them. This is "recognition using regions," and it's already worked for both detection and segmentation. The move is: some external, cheap, category-independent process proposes a few thousand boxes that probably contain objects, and the expensive deep network just scores each one. Localization becomes the job of the proposal stage; recognition becomes the job of the CNN. That division feels right — it plays to the network's strength (rich classification) and routes around its weakness (coarse spatial grid).

So I need a proposal generator. There's a good one: selective search. It starts from a fast graph-based over-segmentation into many small regions, then greedily merges the two most-similar neighbouring regions — similarity being a blend of color-histogram match, texture-histogram match, a size term that prefers merging small regions early, and a fill/shape-compatibility term — recomputes similarities involving the merged region, and repeats all the way up until the image is a single region. Every region produced along that hierarchy becomes a candidate box, and running the whole thing under a few different color spaces and similarity weightings diversifies the set. The payoff: a couple thousand class-independent windows per image with recall around 99%. It doesn't have to be selective search specifically — any category-independent proposer would slot in — but selective search lets me compare head-to-head against prior systems that used the same proposals, so I'll use it in fast mode, ~2000 boxes.

Now the architecture of the detector falls into three pieces almost by itself: (1) generate ~2000 category-independent proposals; (2) push each through a CNN to get a fixed-length feature vector; (3) score each feature vector with a classifier, one per class. Let me build each and watch for where it breaks.

Piece two has an immediate snag. The CNN demands a fixed 227×227 input, and a proposal is an arbitrary rectangle — any size, any aspect ratio. I have to map the rectangle to 227×227. How? A few options. I could enclose the proposal in the tightest *square* and isotropically scale that square to 227 — preserving aspect ratio but pulling in whatever surrounding pixels the square covers. I could do the same but blank out everything outside the original proposal. Or I could just anisotropically *warp* the rectangle to fill 227×227, squashing or stretching its aspect ratio. There's a second knob crossed with these: how much surrounding context to include — a border of p pixels of image around the proposal in the warped frame. Intuitively, warping distorts shape, which sounds bad; but a tight square that includes a lot of background can dilute the object. I don't get to reason this one all the way to a single answer from first principles; it's an empirical call, and warping with a context band of p = 16 pixels comes out ahead of the square-crop variants by 3–5 mAP. The context band evidently helps the network see a little of the object's surroundings. So: anisotropic warp, dilate the box by 16 px of context first, and when the dilated box runs off the image edge fill the missing pixels with the image mean (which gets subtracted off anyway). Each warped 227×227 region forward-propagates through five conv layers and two fully-connected layers to a 4096-dimensional feature vector. That 4096-d vector is two orders of magnitude smaller than the ~360k-dim spatial-pyramid encodings prior region-based systems used, which is going to matter a lot for speed later.

Now the data-scarcity problem, which I parked. I cannot train this 60-million-parameter network from scratch on a few thousand boxed VOC images; it will memorize and overfit. The usual fix is unsupervised pre-training then supervised fine-tuning. But I have something better lying around: a giant *labeled* auxiliary task — ImageNet classification, 1.2 million images, 1000 classes, image-level labels (no boxes, but I don't need boxes for this step). So pre-train the CNN *supervised* on ImageNet classification first. That gives me a network whose features already encode a rich visual hierarchy. Then adapt it to detection. This supervised-pretrain-then-finetune recipe is the lever that makes a high-capacity CNN trainable on scarce detection data.

How do I adapt it? The pre-trained net ends in a 1000-way ImageNet classifier. Detection has N object classes plus a "background" class, so I rip off the 1000-way layer and bolt on a randomly initialized (N+1)-way classification layer — N=20 for VOC, 200 for ILSVRC, plus 1 for background — and keep everything else. Then continue SGD, but on *warped proposal windows* rather than whole ImageNet images, so the network sees the actual distribution it'll be tested on (warped regions, with their distortions and context bands). One thing I must be careful about: I'm starting from good pre-trained weights and I don't want to wreck them. So I drop the learning rate to 0.001, a tenth of the rate used at the start of pre-training — fine-tuning makes progress without clobbering the initialization.

Fine-tuning needs labels for proposals. Which proposals are positive for, say, "car"? A proposal that overlaps a car ground-truth box a lot is clearly a car; one that overlaps nothing is background; the ambiguous middle is the problem. I'll set a rule: map each proposal to the ground-truth box it overlaps most (by IoU), and if that overlap is at least 0.5, call the proposal a positive for that box's class; otherwise call it background. And the minibatch: backgrounds vastly outnumber objects among ~2000 proposals, so if I sample uniformly the gradient is all background. I bias the sampling — each minibatch of 128 is 32 positives (across all classes) and 96 background windows. That 1:3 lean toward positives keeps the rare object signal alive.

Now piece three, the classifier on top of the features. The fine-tuned network already *has* a softmax classifier — its new (N+1)-way layer. The clean thing would be to just use that softmax as the detector and be done. Let me hold that thought, because there's a competing idea: train a separate linear SVM per class on the fixed CNN features. Why would I bother with SVMs when I already have a softmax? Two reasons surface once I think about what each is trained on.

Consider training a binary "car" detector. A region tightly on a car is obviously positive; a region of sky is obviously negative; the trouble is a region that partially overlaps a car — half-car, half-background. For a *precise* detector I want my positive examples to be precise — really just the cars — and my negatives to include those tricky partial overlaps so the detector learns to reject near-misses. So for the SVMs I'll define positives as *only the ground-truth boxes* for each class, and negatives as proposals with less than 0.3 IoU with any instance of that class. Proposals in between (≥0.3 but not ground truth) I simply ignore — they're too car-like to be clean negatives, too imprecise to be clean positives. Where does 0.3 come from? It's not free; I grid-searched the negative threshold over {0, 0.1, …, 0.5} on a validation set, and 0.3 won. And the surrounding values genuinely hurt: pushing it up to 0.5 (so more partial overlaps count as positives) costs 5 mAP, and dropping it to 0 (every non-positive is a negative) costs 4 mAP. So the threshold is doing real work, and carelessness here is expensive.

Now I can see *why* the fine-tuning labels and the SVM labels are defined differently — and they are, jarringly: fine-tuning calls everything ≥0.5 IoU a positive (loose, includes lots of partial overlaps), while the SVM calls only ground-truth boxes positive (tight) and uses 0.3 for negatives. Why the mismatch? It's the scarce-data problem biting again. When I fine-tune the *entire* deep network, I need a lot of positive examples or it overfits; the loose ≥0.5 rule manufactures many "jittered" positives — proposals overlapping a true box between 0.5 and 1.0 but not equal to it — roughly 30× more positives than ground-truth alone. That volume is what keeps fine-tuning from overfitting the whole net. But those jittered positives are imprecise by construction, so they're bad for teaching *precise* localization. The SVMs, sitting on fixed features, aren't at overfitting risk the same way and can afford to be strict — ground-truth-only positives — which is exactly what precise detection wants. The two stages have different data budgets, so they get different label rules.

So should I drop the SVMs and just use the softmax? Let me reason about what the softmax was trained on versus what the SVM is trained on. The softmax saw the loose, jittered, imprecise positives and was trained against randomly sampled negatives; the SVM sees tight ground-truth positives and, crucially, *hard* negatives. I'd bet the SVM localizes better. And the test bears it out — swapping the SVMs for the fine-tuned softmax drops VOC2007 mAP from 54.2 to 50.9. Three-plus points. So the SVMs stay. (It's not that softmax is hopeless — 50.9 is close — but the precise-positive, hard-negative SVM is worth the extra stage.)

That phrase "hard negatives" needs unpacking, because there are far too many background regions to train on naively — millions of them across the dataset, they won't fit in memory. Hard negative mining is the standard answer: don't train on all negatives, train on the ones the current classifier gets *wrong* (background regions it scores as positive). Run the classifier, collect its high-scoring false positives, add them to the negative set, retrain, repeat. The model spends its capacity on the confusing negatives rather than the trivially-easy sky-and-grass. It converges fast — mAP stops climbing after a single pass over the images.

Let me check that this whole thing is even efficient enough to be practical, because I'm running a deep net on 2000 windows per image. The expensive part — proposals plus 2000 CNN forward passes — is *shared across all classes*: I compute the 4096-d feature for each proposal once. The only per-class work is scoring, which is a dot product of features against SVM weights, plus NMS. Stack the features into a 2000×4096 matrix and the per-class weights into a 4096×N matrix and the scoring is one matrix–matrix product. Because the feature is only 4096-d (not 360k-d like the spatial-pyramid systems), this is tiny — even 100k classes is a ~10-second matmul and 1.5 GB of weights, versus the hundreds of GB and orders-of-magnitude-slower scoring a high-dimensional feature would demand. So the design scales to thousands of classes with no approximation. Good.

At test time, then: selective search gives ~2000 proposals, warp each and forward through the CNN to features, score every feature with every class's SVM. Now I have many scored boxes, and lots of them overlap the same object — several proposals fire on the same cat. I need to collapse duplicates. Greedy non-maximum suppression, per class: sort the boxes by score, take the top one, throw away any lower-scored box whose IoU with it exceeds a threshold, repeat. Standard, and necessary because the proposal set is redundant.

I think the system works, but when I look at *where* it fails — using an error-analysis tool that buckets mistakes into poor localization, confusion with similar classes, or firing on background — the dominant error is *localization*: the box is roughly on the object but not tight enough to clear the 0.5-IoU bar. That makes sense given everything above: my boxes are inherited from selective search and only coarsely refined, and my fine-tuning positives were deliberately loose. So the boxes are systematically a bit off. Can I nudge a detected box to fit the object better, using the features I already computed? This is where a regression stage earns its place — not regressing boxes from scratch (that failed earlier as a *localization* method), but regressing a small *correction* to an already-decent proposal box. DPM does something analogous, refining its box from inferred part geometry; here I'll refine from the CNN features.

Let me derive the box regressor carefully, because the parameterization is the whole game. I have training pairs (P, G): P = (P_x, P_y, P_w, P_h) is a proposal's center coordinates, width, height in pixels; G = (G_x, G_y, G_w, G_h) is the ground-truth box it should become. I want to learn a mapping P → Ĝ. The naive thing is to regress the raw differences G − P. But think about scale: a 10-pixel center error on a tiny box is a gross miss; the same 10 pixels on a huge box is nothing. If I regress raw pixel offsets, the loss is dominated by large boxes and the targets aren't comparable across scales. I want the *correction* expressed in units that are invariant to the proposal's size. So for the center, normalize the translation by the proposal's own width and height:

  t_x = (G_x − P_x) / P_w,    t_y = (G_y − P_y) / P_h.

Now t_x is "shift the center by this fraction of the box width," which means the same target for a small box and a large box that are *proportionally* equally misaligned. Scale-invariant, exactly what I wanted. For width and height, a *difference* is again scale-dependent and could also drive a predicted width negative. The natural move is to predict the log-ratio:

  t_w = log(G_w / P_w),    t_h = log(G_h / P_h).

Log-space makes "double the width" and "halve the width" symmetric corrections (+log2 and −log2), it's scale-invariant (depends only on the ratio), and when I invert it the exponential guarantees a positive predicted size. So the four functions I'll learn are d_x(P), d_y(P) for the normalized center shift and d_w(P), d_h(P) for the log-size correction, and I reconstruct the predicted box by inverting the target definitions:

  Ĝ_x = P_w · d_x(P) + P_x,
  Ĝ_y = P_h · d_y(P) + P_y,
  Ĝ_w = P_w · exp(d_w(P)),
  Ĝ_h = P_h · exp(d_h(P)).

Let me sanity-check the inversion. If the regressor predicts d_x = t_x = (G_x−P_x)/P_w, then Ĝ_x = P_w·(G_x−P_x)/P_w + P_x = G_x. And Ĝ_w = P_w·exp(log(G_w/P_w)) = P_w·(G_w/P_w) = G_w. So a perfect regressor reproduces G exactly — the parameterization is consistent.

What do I regress *from*? The features I already have. Specifically the last-conv-layer features of the proposal — call them φ5(P) — and a plain linear function of them, d_⋆(P) = w_⋆ᵀ φ5(P), one weight vector per coordinate ⋆ ∈ {x, y, w, h}. Linear is enough because the CNN features are already rich; I'm only fitting a small correction. Fitting the weights is then least squares with regularization — ridge regression:

  w_⋆ = argmin_ŵ  Σ_i (t_⋆^i − ŵᵀ φ5(P^i))² + λ ‖ŵ‖².

Two things bite in practice. First, regularization is not optional here — with high-dimensional features and limited pairs, an unregularized fit overfits badly; λ = 1000 (set on validation) is large, which tells me the targets are noisy enough that I want a heavily shrunk solution. Second, *which pairs* (P, G) do I even train on? If P is nowhere near any object, asking the regressor to teleport it onto a distant box is a hopeless, meaningless task that will only corrupt the fit. So I only train on a proposal P if it actually sits near a ground-truth box: assign P to its maximum-IoU ground-truth G, and keep the pair only if that IoU exceeds 0.6; discard the rest. The regressor is class-specific (one set of weights per class), and since it's plain regularized least squares it solves in closed form. At test time I apply it once per detected box — I could iterate (refine, re-extract features, refine again), but iterating doesn't help, so once it is.

I want to step back and look at what the features themselves learned, partly as a sanity check that the whole region-classification premise is sound. Take a single unit in the last conv layer — the pooled fifth-layer map is 6×6×256, so 9216 numbers, and each unit has that ~195×195 receptive field. Treat one unit as if it were its own little detector: compute its activation over ~10 million held-out proposal regions, sort from highest to lowest, NMS to avoid near-duplicate crops, and look at the top-firing regions. They're interpretable — some units lock onto people, some onto text, some onto dot-array textures, some onto specular highlights and material properties. So the representation is a mix of a few class-tuned detectors and a distributed code for shape, texture, color, material — exactly the kind of rich, learned hierarchy that hand-designed HOG can't be, and the next fully-connected layer can compose these into class scores. That's the qualitative reason the whole approach should beat HOG-based systems: the features are doing work HOG never could.

So the full method: selective search proposes ~2000 category-independent regions; each is dilated by 16 px of context and anisotropically warped to 227×227; a CNN — supervised-pretrained on ImageNet classification, then fine-tuned on warped proposals with ≥0.5-IoU positives and a 32/96 positive/background minibatch at lr 0.001 — extracts a 4096-d feature per region; per-class linear SVMs (ground-truth positives, <0.3-IoU hard-mined negatives) score every region; per-class greedy NMS removes duplicates; and a per-class ridge-regression stage on the conv features applies a scale-invariant box correction to fix the dominant localization errors. Regions with CNN features.

Let me put it down as code, mirroring how this actually gets built — proposals, warp, fine-tune, feature cache, SVM training with hard negatives, box regression, and test-time detection.

```python
import numpy as np

# ---- existing primitives ------------------------------------------------
# selective_search(image, mode="fast") -> ~2000 boxes (x1,y1,x2,y2)
# cnn_forward(batch_227, layer)         -> features (pretrained AlexNet via Caffe)
# nms(boxes, scores, iou_thresh)        -> kept indices
# ridge_solve(X, t, lam)                -> closed-form (X^T X + lam I)^-1 X^T t


def prepare_region(image, box, pad=16, size=227):
    # dilate the box by `pad` px of context, then ANISOTROPICALLY warp to size x size.
    # off-image pixels are filled with the image mean (subtracted before the CNN).
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    # context band measured in the warped frame -> expand in source proportionally
    cx_pad = pad * w / size
    cy_pad = pad * h / size
    src = clip_to_image(x1 - cx_pad, y1 - cy_pad, x2 + cx_pad, y2 + cy_pad, image, fill="mean")
    return anisotropic_resize(src, (size, size))  # squash aspect ratio to fit


def iou(a, b):  # standard intersection-over-union on (x1,y1,x2,y2)
    ...


# ---- stage 1: fine-tune the CNN on warped proposals ----------------------
def finetune_labels(proposals, gts, gt_classes, num_classes):
    # FT positives are LOOSE: any proposal with max-IoU >= 0.5 is a positive for
    # that gt's class; everything else is background. This manufactures ~30x more
    # ("jittered") positives -> enough data to fine-tune the whole net without overfit.
    labels = np.zeros(len(proposals), dtype=int)  # 0 = background
    for i, p in enumerate(proposals):
        ious = np.array([iou(p, g) for g in gts])
        if len(gts) and ious.max() >= 0.5:
            labels[i] = gt_classes[ious.argmax()] + 1   # +1: 0 reserved for bg
    return labels  # (N+1)-way targets


def finetune(cnn, dataset, num_classes, lr=0.001, batch=128, n_pos=32, n_bg=96):
    cnn.replace_head(num_classes + 1)        # drop 1000-way, add (N+1)-way random head
    opt = SGD(cnn.parameters(), lr=lr)       # 1/10 of pretraining LR: don't clobber init
    for images, proposals, gts, gt_cls in dataset:
        lab = finetune_labels(proposals, gts, gt_cls, num_classes)
        pos = np.where(lab > 0)[0]; bg = np.where(lab == 0)[0]
        idx = np.r_[sample(pos, n_pos), sample(bg, n_bg)]   # bias toward rare positives
        x = np.stack([prepare_region(images, proposals[j]) for j in idx])
        loss = cross_entropy(cnn.classify(x), lab[idx])
        opt.zero_grad(); loss.backward(); opt.step()
    return cnn


# ---- stage 2: per-class linear SVMs on cached features -------------------
def svm_labels(proposals, gts, cls):
    # SVM labels are STRICT: positives are ONLY ground-truth boxes; negatives are
    # proposals with < 0.3 IoU with every instance of `cls`; the 0.3-1.0 grey zone
    # is IGNORED. (0.3 chosen by grid search; 0.5 -> -5 mAP, 0.0 -> -4 mAP.)
    pos, neg = [], []
    for p in proposals:
        max_iou = max((iou(p, g) for g in gts), default=0.0)
        if max_iou < 0.3:
            neg.append(p)
    pos = list(gts)                  # ground-truth boxes are the positives
    return pos, neg


def train_svms(feat_cache, proposals_per_image, gts_per_image, num_classes):
    svms = []
    for c in range(num_classes):
        X_pos = features_of(feat_cache, gts_per_image, cls=c)        # gt-only positives
        X_neg = features_of(feat_cache, proposals_per_image, cls=c, rule="<0.3")
        w = init_linear_svm()
        # hard-negative mining: train, find current false positives, add, retrain.
        # converges in ~1 pass; the bulk of easy background is never needed.
        for _ in range(num_passes):
            w = fit_linear_svm(np.vstack([X_pos, X_neg]),
                               np.r_[np.ones(len(X_pos)), -np.ones(len(X_neg))])
            hard = X_neg[score(w, X_neg) > -1.0]      # margin-violating negatives
            X_neg = np.vstack([X_neg, hard])
        svms.append(w)
    return np.stack(svms)            # (num_classes, 4096)


# ---- stage 3: bounding-box regression (per class, ridge) -----------------
def bbox_reg_targets(P, G):
    # P,G = (cx, cy, w, h). Scale-invariant center shift; log-space size correction.
    tx = (G[0] - P[0]) / P[2]
    ty = (G[1] - P[1]) / P[3]
    tw = np.log(G[2] / P[2])
    th = np.log(G[3] / P[3])
    return np.array([tx, ty, tw, th])


def fit_bbox_regressors(pool5_feats, proposals, gts, gt_cls, num_classes, lam=1000.0):
    regressors = {}
    for c in range(num_classes):
        X, T = [], []
        for f, p in zip(pool5_feats, proposals):
            ious = np.array([iou(p, g) for g in gts if gt_cls_of(g) == c] or [0])
            if ious.max() > 0.6:                      # only train from NEARBY proposals
                g = nearest_gt(p, gts, c)
                X.append(f); T.append(bbox_reg_targets(to_cwh(p), to_cwh(g)))
        if X:
            regressors[c] = ridge_solve(np.array(X), np.array(T), lam)  # closed form
    return regressors


def apply_bbox_reg(W_c, pool5_feat, P):
    d = pool5_feat @ W_c                  # [dx, dy, dw, dh]
    cx, cy, w, h = to_cwh(P)
    return [w * d[0] + cx, h * d[1] + cy, w * np.exp(d[2]), h * np.exp(d[3])]


# ---- test-time detection -------------------------------------------------
def detect(image, cnn, svms, regressors, num_classes, nms_iou=0.3):
    boxes = selective_search(image, mode="fast")            # ~2000 proposals
    inp = np.stack([prepare_region(image, b) for b in boxes])
    feats = cnn_forward(inp, layer="fc7")                   # 2000 x 4096, shared cost
    pool5 = cnn_forward(inp, layer="pool5")                 # for box regression
    scores = feats @ svms.T                                 # 2000 x num_classes matmul
    dets = []
    for c in range(num_classes):
        sc = scores[:, c]
        refined = np.array([apply_bbox_reg(regressors[c], pool5[i], boxes[i])
                            if c in regressors else to_cwh(boxes[i])
                            for i in range(len(boxes))])
        keep = nms(cwh_to_xyxy(refined), sc, nms_iou)       # per-class greedy NMS
        dets += [(c, sc[i], refined[i]) for i in keep]
    return dets
```

The causal chain, end to end: detection had stalled on hand-designed HOG/SIFT features while CNNs were transforming classification, so I want CNN features for detection — but a deep net localizes poorly under sliding windows (coarse receptive field and stride) and regressing boxes outright is weak, so I decouple localization from recognition by classifying ~2000 category-independent selective-search proposals; each arbitrary region is warped (with 16 px of context) to the CNN's fixed input; the high-capacity CNN can't be trained on scarce boxed data, so I pre-train it supervised on ImageNet classification and fine-tune it on warped proposals with loose ≥0.5-IoU positives (to manufacture enough positives) at a tenth the learning rate; per-class linear SVMs with strict ground-truth positives and hard-mined <0.3-IoU negatives score the shared 4096-d features cheaply; greedy per-class NMS removes the duplicate firings the redundant proposal set produces; and because the dominant residual error is loose localization, a per-class ridge regressor on the conv features applies a scale-invariant, log-space box correction to tighten each detection.
