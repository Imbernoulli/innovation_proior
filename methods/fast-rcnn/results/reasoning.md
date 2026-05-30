The region-based ConvNet detector works — classify a few thousand bottom-up proposals with a deep net and detection accuracy jumps. But living with it day to day, the thing that bothers me isn't the accuracy, it's the machinery. Training is three separate stages bolted together: first fine-tune the ConvNet on warped proposals with a log loss; then throw away the softmax it learned and fit per-class SVMs to the network's features; then, as a third pass, fit bounding-box regressors. Each stage has its own objective, its own data handling. And to feed the SVMs and the regressors I have to extract a feature vector for *every proposal of every image* and write them all to disk — hundreds of gigabytes, and on the order of days of GPU time for a very deep network on a few thousand images. Test time is just as bad: the network does a full forward pass for *each* of the ~2000 proposals, with no computation shared between them, so a deep net spends the better part of a minute per image. Let me see if I can collapse this into one network, trained once, that shares its work.

Where is all the test-time cost going? It's the per-proposal forward pass. Two proposals in the same image overlap enormously, yet I recompute all the convolutional features for each from scratch. That's obviously wasteful — the convolutional feature map of an image is a function of the image, not of the proposal. So compute it once. Run the conv layers over the whole image, get a single conv feature map, and then for each proposal just look at the corresponding sub-region of that shared map. The conv work — the expensive part — happens once per image instead of ~2000 times.

That immediately raises the problem the fully connected layers create. The conv feature map region for a proposal is some variable h×w×C slab — different proposals have different sizes — but the fully connected layers downstream need a fixed-length vector. I need to turn an arbitrary-size region of the feature map into a fixed-size feature. The known device is spatial pyramid pooling: pool the region into one or more fixed grids and concatenate. But do I need the full pyramid of multiple grid resolutions? For a clean, single mechanism, take the simplest case — a single grid. Fix an output size H×W (say 7×7, chosen to match what the first fully connected layer expects). For a proposal region of size h×w in the feature map, divide it into an H×W grid of sub-windows each about (h/H)×(w/W), and max-pool the feature values within each sub-window into the corresponding output cell, independently per channel. Out comes a fixed H×W×C feature for any region. That's a region-of-interest pooling layer — spatial pyramid pooling collapsed to a single pyramid level. It sits where the last max-pool used to be.

Now I have a fast *test-time* path: image → conv map (once) → per-proposal region pooling → fc layers → outputs. The shared-conv idea alone gets the 10-to-100× test speedup. But it leaves the training pipeline just as ugly and, worse, there's a known accuracy ceiling lurking. The shared-feature-map approach, as it had been trained, could *not* update the convolutional layers — fine-tuning only touched the fully connected layers above the pooling. And freezing the conv trunk caps the accuracy of a very deep network, because the deep conv features are exactly what I'd most want to adapt to detection. So I have to ask: why couldn't the conv layers be fine-tuned, and can I fix it?

Let me trace the inefficiency. Back-propagating through the region-pooling layer is fine in principle. The trouble is *how the training samples were drawn*: one proposal per image, with each proposal in the mini-batch coming from a different image. Each proposal can have a receptive field that spans nearly the whole image, so to compute its features and gradients I have to run the forward (and backward) pass over its entire receptive field — effectively the whole image. If my 128 mini-batch samples come from 128 different images, I'm processing 128 (nearly) full images per update. That's ruinous, which is why people just froze the conv layers and trained the fc part on cached features.

The fix is staring at me once I phrase it that way: don't sample proposals from different images — sample them from a *few* images, and many proposals per image. Sample hierarchically: pick N images, then R/N proposals from each. Proposals from the same image share the same conv feature map, so they share the forward and backward computation through the conv trunk. With N=2 images and R=128 proposals, I process 2 images per update instead of 128 — roughly 64× less conv computation than the one-proposal-per-image scheme — and the gradient flows all the way back through the region-pooling layer into the conv layers. Now I *can* fine-tune the whole network, conv layers included, in normal training time. There's a worry: proposals from the same image are correlated, so the mini-batch gradient is less diverse, which could slow convergence. In practice it doesn't bite — good results come with N=2, R=128 in fewer SGD iterations than the per-proposal scheme used. The correlation just isn't strong enough to matter, and SGD momentum smooths it out.

Let me make the region-pooling backward pass concrete, because it's the piece that has to actually carry gradient into the conv trunk. The layer just does max-pooling, so it has argmax switches like any max-pool, only the pooling windows are defined per proposal. Let x_i be an input activation to the layer and y_rj the j-th pooled output of the r-th proposal, so y_rj = x_{i*(r,j)} where i*(r,j) = argmax over the sub-window R(r,j) of the inputs. One input x_i can be the argmax for several outputs — different proposals' windows can overlap and select the same location. So the gradient to x_i accumulates over every output that selected it:

  ∂L/∂x_i = Σ_r Σ_j [i = i*(r,j)] ∂L/∂y_rj,

i.e. for each proposal r and each pooled cell y_rj, route the incoming derivative ∂L/∂y_rj back to whichever input was its argmax. Standard max-pool routing, summed over all proposals sharing the feature map. That's what lets two images' worth of proposals jointly update the conv layers.

So the network is: pre-trained ImageNet ConvNet as the trunk, its last max-pool swapped for a region-pooling layer (H=W=7 to feed the first fc), and its 1000-way classification head swapped out. Swapped for what? Detection wants both a class label and a refined box per region. So branch into two sibling outputs after the fc layers: one that produces a softmax distribution over K object classes plus a background class — p = (p_0, …, p_K) — and one that produces bounding-box regression offsets, four numbers for *each* of the K classes, t^k = (t^k_x, t^k_y, t^k_w, t^k_h). Per-class boxes, because the right refinement of a box depends on what class it is. The offset parameterization is the standard scale-invariant translation plus log-space size shift relative to the proposal.

Now the part that lets me kill the multi-stage pipeline: train both heads at once with a single loss. There's a real reason to expect this to *help*, not just to be convenient — when two tasks share a representation, training them jointly shapes the shared trunk by both objectives at once, and they can reinforce each other. So put a multi-task loss on each labeled proposal. Label each training proposal with a ground-truth class u and, if it's an object, a ground-truth box target v. Then

  L(p, u, t^u, v) = L_cls(p, u) + λ [u ≥ 1] L_loc(t^u, v),

with the classification term the ordinary log loss L_cls(p, u) = −log p_u, and the localization term applied only to the box offsets for the *true* class u. The [u ≥ 1] indicator is doing something important: background regions (u = 0, by convention) have no ground-truth box, so they get no localization loss — only object regions are asked to refine a box. And I only penalize the offsets t^u for the true class, not all K box predictions, since the other classes' boxes are irrelevant for this region.

For L_loc itself, the natural choice is L2 on the four offsets, but L2 has a problem here. The regression targets can be large, and an L2 loss on a large residual produces a large gradient — with unbounded targets, training can blow up and you end up babysitting the learning rate to keep gradients from exploding. I want something that behaves like L2 for small residuals (smooth, well-behaved near the optimum) but only grows linearly for large residuals (bounded gradient, robust to outliers). That's a smooth L1:

  smooth_L1(x) = 0.5 x²        if |x| < 1,
                  |x| − 0.5     otherwise,

and L_loc(t^u, v) = Σ_{i ∈ {x,y,w,h}} smooth_L1(t^u_i − v_i). Let me check it's sensible: at x = 0 it's 0 with zero derivative; at |x| = 1 the two branches meet (0.5·1 = 1 − 0.5 = 0.5) and the derivatives match (x = 1 vs. constant 1), so it's C¹ — continuous value and slope at the seam. For |x| ≥ 1 the gradient is just ±1, bounded, so a wildly-off box can't produce a runaway gradient. Exactly the robustness I wanted, and it removes the learning-rate fragility that L2 had.

That leaves λ, which trades off the two losses. For it to be a meaningful "1," the two terms have to be on comparable scales, so I normalize the ground-truth box targets to zero mean and unit variance first; then λ = 1 balances them. With the targets normalized, neither term dominates by accident.

One more thing this multi-task setup buys me: I can finally ask whether the SVMs were even necessary. The reason the earlier system used per-class SVMs instead of the fine-tuned softmax was historical and tied to the cached-feature pipeline. But now I'm fine-tuning the whole network end to end with a softmax already in the loss. If the softmax is good enough, I drop a whole training stage and the disk cache with it. The intuition cuts in softmax's favor, actually: a softmax over K+1 classes introduces *competition* between classes when scoring a region — the scores must sum to one, so confidently calling something a "cat" suppresses "dog" — whereas one-vs-rest SVMs score each class independently with no such competition. So the single-stage softmax should be at least as good. And it is: across networks it edges out post-hoc SVMs slightly. So: no SVMs, no disk cache, one training stage.

Now mini-batch construction in detail. N = 2 images per batch, R = 128 proposals total, 64 per image. Which proposals? I want a mix of objects and background. Take 25% of the proposals (32) as foreground: proposals whose IoU with a ground-truth box is at least 0.5, labeled with that box's class (u ≥ 1). The other 75% (96) as background, labeled u = 0 — but not *any* background. I draw them from proposals whose maximum IoU with ground truth lies in [0.1, 0.5). The upper bound 0.5 is just "not foreground." The lower bound of 0.1 is the interesting choice: it excludes the trivially-easy background (proposals overlapping nothing) and keeps the background that at least partly grazes an object, which are the *hard* negatives — the ones the classifier is most likely to get wrong. So the 0.1 floor is an implicit hard-example-mining heuristic, achieved by sampling rather than by a separate mining stage. Plus horizontal flips with probability 0.5 for a little augmentation, nothing else.

The SGD details, since they matter for a network that's part-pretrained, part-fresh. Initialize the two new sibling fc layers from zero-mean Gaussians — std 0.01 for the classification layer, 0.001 for the box-regression layer (smaller, because its outputs are offsets that should start near zero) — and biases at 0. Use a global learning rate of 0.001 with per-layer multipliers (1 for weights, 2 for biases), train 30k iterations, then drop to 0.0001 for another 10k. Momentum 0.9, weight decay 0.0005. The pretrained conv layers come along and get fine-tuned through the region-pooling gradient.

Do I actually need to fine-tune *all* the conv layers, though? Two sub-questions. First, does conv fine-tuning matter at all for a very deep net? If I freeze all the conv layers and only let the fc layers learn — which is exactly the frozen-conv regime that capped accuracy before — I should see a drop. And I do: it falls several points (66.9 down to 61.4 for the deep net). So training through the region-pooling layer into the conv trunk is the thing that recovers the accuracy a very deep network is capable of. Second, does *every* conv layer need to learn? No. The very first conv layer learns generic edge/color filters that are task-independent — letting conv1 learn or not has no real effect on mAP. And fine-tuning from too low costs memory and time (updating from the second conv block is 1.3× slower; updating from the first runs out of GPU memory) for only a fraction of a point. So fine-tune from a middle conv block upward — for the deep net, the upper 9 of 13 conv layers — and freeze the bottom. Pragmatic, not principled, but it's the right speed/accuracy point.

I should think about scale, since objects come at wildly different sizes. Two options. Brute force: process every image at a single fixed size and let the network learn scale invariance from the data. Or finesse: build an image pyramid, run the conv net at several scales, and assign each proposal to the pyramid level where its scaled size is closest to the canonical training size (around 224² area). The pyramid is more compute. Which wins? The brute-force single scale comes out almost as well — deep conv nets are evidently good at directly learning scale invariance from the training distribution — and the pyramid adds only a small mAP gain at large cost. So I'll fix the shortest image side to 600 pixels (capping the longest at 1000 to fit the deep net in GPU memory) and run single-scale. The multi-scale option exists for the smaller, less memory-bound models, but single-scale is the speed/accuracy sweet spot, especially for deep nets.

Now profiling the fast network reveals a new bottleneck I didn't have before. In whole-image *classification*, the fully connected layers are cheap relative to the conv layers. But in detection I run the fc layers once *per proposal* — ~2000 times — so they now eat nearly half the forward-pass time. The conv layers are shared and cheap per image; the fc layers are not shared and there are thousands of them. I can compress the big fc layers without retraining. A fc layer is a matrix W of size u×v; take its truncated SVD, W ≈ U Σ_t V^T, keeping the top t singular values. Replace the single layer by two layers with no nonlinearity between them: the first applies Σ_t V^T (no bias), the second applies U (carrying W's original bias). The composition is U Σ_t V^T ≈ W, but the parameter and compute count drops from u·v to t·(u+v), a big win when t ≪ min(u,v). For the deep net's two huge fc layers (the 25088×4096 and 4096×4096 matrices that dominated), keeping the top ~1024 and ~256 singular values cuts the fc time substantially with only a small mAP loss. A pure inference-time speedup, no retraining required.

One more thing worth checking, now that training+testing a model is cheap (a couple hours), is whether feeding more proposals always helps — because sparse proposals are really acting as a cascade: the proposal stage rejects the overwhelming majority of windows, leaving the deep classifier a small, high-quality set. If proposals were purely a computational convenience, more of them should never hurt accuracy. But sweeping from 1k to 10k proposals, mAP rises and then *falls* slightly — swamping the classifier with more candidates doesn't help and even hurts. So the sparse proposal set is doing real work as a cascade, not just trimming compute, and there's such a thing as too many. (Average Recall, the popular proposal-quality proxy, tracks mAP only at a fixed proposal count and misleads when the count varies — more proposals can raise AR while mAP drops.) Worth knowing so I don't chase recall blindly.

Putting it together: one network takes an image and its proposals, runs the conv trunk once to a shared feature map, region-pools each proposal to a fixed 7×7×C feature, runs fc layers, and branches into a (K+1)-way softmax and per-class box offsets; it's trained in a single stage with the multi-task log-loss-plus-smooth-L1 objective, on hierarchically sampled mini-batches (2 images, 128 proposals, 25% foreground, hard-ish background from IoU∈[0.1,0.5)), fine-tuning the conv layers through the region-pooling gradient; at test time it's a single forward pass with truncated-SVD-compressed fc layers and per-class NMS.

Let me write it down, mirroring how it actually gets built — the region-pooling module, the two-headed detector, the multi-task loss, the hierarchical sampler, and the SVD compression.

```python
import torch
from torch import nn
import torch.nn.functional as F


class RegionPooling(nn.Module):
    """Single-level spatial-pyramid (RoI) max-pool: any region -> fixed HxW x C.
    Replaces the backbone's last max-pool; gradients flow back into the conv trunk
    via the max-pool argmax switches (shared across overlapping regions)."""
    def __init__(self, output_h=7, output_w=7, spatial_scale=1/16.):
        super().__init__()
        self.oh, self.ow, self.scale = output_h, output_w, spatial_scale

    def forward(self, feat, rois):
        # rois: (R, 5) = (image_index, x1, y1, x2, y2) in input-image pixels
        out = feat.new_zeros(rois.size(0), feat.size(1), self.oh, self.ow)
        for r, (n, x1, y1, x2, y2) in enumerate(rois):
            x1, y1, x2, y2 = [int(round(c.item() * self.scale)) for c in (x1, y1, x2, y2)]
            h, w = max(y2 - y1, 1), max(x2 - x1, 1)
            for i in range(self.oh):
                for j in range(self.ow):
                    # sub-window of approx size (h/H) x (w/W) tiling the region
                    sy1 = y1 + (i * h) // self.oh; sy2 = y1 + ((i + 1) * h) // self.oh
                    sx1 = x1 + (j * w) // self.ow; sx2 = x1 + ((j + 1) * w) // self.ow
                    region = feat[int(n), :, sy1:max(sy2, sy1 + 1), sx1:max(sx2, sx1 + 1)]
                    out[r, :, i, j] = region.amax(dim=(1, 2))   # max-pool (autograd routes grad)
        return out


class FastDetector(nn.Module):
    def __init__(self, backbone, head, num_classes, feat_dim=4096):
        super().__init__()
        self.backbone = backbone           # conv trunk (last max-pool removed)
        self.region_pool = RegionPooling(7, 7, spatial_scale=1/16.)
        self.head = head                   # the fc layers (e.g. fc6, fc7)
        self.cls = nn.Linear(feat_dim, num_classes + 1)   # K+1 softmax scores
        self.box = nn.Linear(feat_dim, 4 * num_classes)   # per-class box offsets
        nn.init.normal_(self.cls.weight, std=0.01); nn.init.zeros_(self.cls.bias)
        nn.init.normal_(self.box.weight, std=0.001); nn.init.zeros_(self.box.bias)

    def forward(self, images, rois):
        feat = self.backbone(images)             # whole-image conv map, computed ONCE
        pooled = self.region_pool(feat, rois)    # (R, C, 7, 7)
        x = self.head(pooled.flatten(1))         # (R, feat_dim)
        return self.cls(x), self.box(x)          # class scores, box offsets


def smooth_l1(x):
    ax = x.abs()
    return torch.where(ax < 1.0, 0.5 * x * x, ax - 0.5)


def multitask_loss(cls_scores, box_offsets, labels, box_targets, lam=1.0):
    # classification: log loss over K+1 classes
    L_cls = F.cross_entropy(cls_scores, labels)
    # localization: smooth-L1 on the TRUE class's offsets, only for fg (u >= 1)
    fg = labels >= 1
    R = box_offsets.size(0)
    box_offsets = box_offsets.view(R, -1, 4)
    pred_u = box_offsets[torch.arange(R)[fg], labels[fg]]        # offsets for true class
    L_loc = smooth_l1(pred_u - box_targets[fg]).sum(dim=1).sum()
    L_loc = L_loc / max(R, 1)
    return L_cls + lam * L_loc                                   # [u>=1] enforced by masking


def sample_minibatch(images_db, num_images=2, rois_per_image=64,
                     fg_frac=0.25, fg_iou=0.5, bg_iou=(0.1, 0.5)):
    """Hierarchical sampling: few images, many RoIs each -> RoIs share the conv pass."""
    batch_imgs, batch_rois, batch_labels, batch_targets = [], [], [], []
    for n in random.sample(range(len(images_db)), num_images):
        rec = images_db[n]                                       # proposals + gt for image n
        max_iou, gt_idx = roi_gt_overlaps(rec.proposals, rec.gt_boxes)
        n_fg = int(round(fg_frac * rois_per_image))
        fg = where(max_iou >= fg_iou)                            # foreground proposals
        bg = where((max_iou >= bg_iou[0]) & (max_iou < bg_iou[1]))  # HARD-ish background
        sel = cat(sample(fg, n_fg), sample(bg, rois_per_image - n_fg))
        for k in sel:
            label = rec.gt_classes[gt_idx[k]] + 1 if max_iou[k] >= fg_iou else 0  # 0 = bg
            target = encode_boxes(rec.proposals[k], rec.gt_boxes[gt_idx[k]]) if label else 0
            batch_rois.append((len(batch_imgs), *rec.proposals[k]))
            batch_labels.append(label); batch_targets.append(target)
        batch_imgs.append(maybe_hflip(rec.image, p=0.5))         # only augmentation
    return stack(batch_imgs), tensor(batch_rois), tensor(batch_labels), stack(batch_targets)


def truncated_svd_fc(fc, t):
    """Replace one fc layer W (u x v) by two: (Sigma_t V^T) then (U), no nonlinearity.
    Cuts params u*v -> t*(u+v); used on the big fc layers at detection time."""
    W = fc.weight.data                          # (u, v)
    U, S, Vt = torch.linalg.svd(W, full_matrices=False)
    first = nn.Linear(W.size(1), t, bias=False)
    first.weight.data = torch.diag(S[:t]) @ Vt[:t]
    second = nn.Linear(t, W.size(0), bias=True)
    second.weight.data = U[:, :t]
    second.bias.data = fc.bias.data
    return nn.Sequential(first, second)


@torch.no_grad()
def detect(model, image, proposals, num_classes, score_thresh=0.05, nms_iou=0.3):
    rois = torch.tensor([(0, *p) for p in proposals])
    cls_scores, box_offsets = model(image[None], rois)
    probs = cls_scores.softmax(dim=1)                            # inter-class competition
    box_offsets = box_offsets.view(len(proposals), num_classes, 4)
    dets = []
    for k in range(num_classes):                                # background (0) excluded
        boxes = apply_offsets(proposals, box_offsets[:, k])
        scores = probs[:, k + 1]
        keep = nms_per_class(boxes, scores, nms_iou)
        dets += [(k, scores[i], boxes[i]) for i in keep if scores[i] > score_thresh]
    return dets
```

The causal chain: the region-based detector was accurate but slow and multi-stage because it ran a full ConvNet per proposal and trained classifier and box-regressor separately; so I share all convolutional work by computing one whole-image feature map and region-pooling each proposal to a fixed size; to recover the accuracy a deep net loses when its conv layers are frozen, I sample mini-batches hierarchically (few images, many proposals) so gradients flow efficiently through region-pooling into the conv trunk and all layers train; I fold classification and box regression into one network with a single multi-task loss (log loss plus a robust smooth-L1, weighted by λ=1 on normalized targets, applied only to the true class of foreground regions), which lets the softmax replace the post-hoc SVMs and eliminates the disk cache; I sample hard-ish background by an IoU floor as implicit hard-negative mining; and at test time I compress the now-dominant fc layers with truncated SVD and run per-class NMS, yielding one network trained in one stage that is far faster to train and test while at least matching the prior accuracy.
