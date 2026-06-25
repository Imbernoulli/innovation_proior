OK, let me think this through from scratch. I have a detection pipeline that, in principle, should be fast now, and yet in practice it is not, and I want to understand exactly where the time goes before I touch anything.

The pipeline has two halves. First, an external module — Selective Search, say — looks at the image and hands me a couple thousand candidate boxes that *might* contain objects. Then a convolutional network looks at each candidate, decides what category it is (or that it's background), and nudges its box to fit better. A year or two ago the second half was the disaster: R-CNN literally ran the whole CNN once per proposal, two thousand forward passes per image, tens of seconds. But that's been fixed. SPP, and then Fast R-CNN, made the observation that you don't need to run the convolutions per region at all — you run them once over the whole image, get a single convolutional feature map, and then for each proposal you just pool the sub-window of that map that the proposal covers into a fixed-size little feature and send *that* through a couple of cheap layers. One shared conv pass, then a thin per-region tail. On a GPU the conv pass is ~150 ms and the region-wise part is another few tens of ms. Call the detector a couple hundred milliseconds.

So where's the two seconds? It's the proposer. Selective Search runs on the CPU, over hand-engineered low-level features — superpixels merged by color and texture — and it takes about 2 seconds an image. EdgeBoxes is the fast one and it's still ~0.2 s, which is *the same order as the entire detection network*. Stare at that for a second: the part of the system that's supposed to be the cheap pre-filter is now the expensive part. We optimized the CNN until it disappeared, and what's left standing is the thing we always treated as a free black box.

What do I do about it? The obvious move is: Selective Search is on the CPU, the detector is on the GPU, that's not a fair fight — just reimplement the proposer on the GPU. And sure, that might shave the wall clock. But it bothers me, because it's pure plumbing. It treats the proposer as a fixed algorithm I'm porting to faster hardware, and in doing so it keeps the proposer completely walled off from the detector. The detector computes this rich convolutional feature map of the whole image. The proposer, meanwhile, recomputes its own low-level features from scratch, on the CPU, ignoring everything the conv net knows. Two passes over the image, two feature representations, and they never talk. If I'm paying for a deep feature map anyway, why am I generating proposals from color and texture?

That reframes the whole thing. The question isn't "how do I make the proposer faster" — it's "the detector already computes a feature map that encodes where the stuff in the image is; can't I read the proposals straight off of *that*?" If I can, then the proposals cost almost nothing, because the expensive part — the convolutions — is already being paid for by the detector. The marginal cost of proposing would just be whatever thin layers I add on top of the shared map. So the thing to chase isn't a faster proposer, it's a proposer that shares computation with the detector instead of duplicating it — and I won't know how nearly-free it really is until I've sized the head I'd have to add. Let me follow that thread and see if it actually closes.

So let me try to build a proposer that lives on top of the shared conv feature map. What does it need to output? For every plausible object in the image, a box and a score saying "this looks like an object (versus background)." Class-agnostic is fine — the downstream detector will sort out *which* object. So I need, densely over the image, box coordinates and an objectness score.

How do I read that off a feature map of size W×H×C? The feature map is a grid; each spatial cell summarizes a patch of the image through the receptive field. The natural thing is to slide a small network over the grid: at each location, take a little window of the feature map — say n×n — feed it through a small net, and have it emit, for that location, a box and an objectness score. And because I want this to be cheap and uniform, the small net should have the *same weights at every location*. That's just a convolution. A small network slid with shared weights over a feature map is a fully convolutional network. So concretely: an n×n conv layer (with a ReLU) that maps the C-channel input to some intermediate vector at each location, then two sibling 1×1 conv layers reading off that vector — one producing the box coordinates ("reg"), one producing the objectness score ("cls"). 1×1 convs because the per-location prediction is just a linear map of the per-location feature, applied identically everywhere — which is exactly what a 1×1 conv is.

Why fully convolutional, and not, say, a fully-connected layer that ingests the whole feature map and spits out a fixed list of boxes? An FC layer over the whole map would have a gigantic parameter count and would only work at a fixed input size. And there's a deeper reason: sliding the same little net everywhere is *translation invariant* — if an object moves twenty pixels to the right, its features move with it on the grid, the same shared filter fires at the new location, and out comes the same box (shifted accordingly). The function that produces proposals doesn't depend on absolute position. That's the right inductive bias for "find objects anywhere," and it's automatic with convolution and would have to be painstakingly faked otherwise.

Why n=3 for the window and not something big? I want the receptive field on the original image to be large enough to judge whole objects, but I want the parameter count and compute small. The trick is that I'm at the *last* conv layer of a deep backbone, where each cell already has an enormous receptive field — for ZF it's 171 pixels, for VGG 228 — so even a 3×3 window over this map sees a big chunk of the image. I get a large effective receptive field essentially for free, so 3×3 is plenty and keeps the head tiny.

Now the hard part, the one that's been quietly nagging: objects come in wildly different sizes and shapes, and I've committed to a *single* feature map at a *single* scale with a *single* filter size. A 3×3 window on the conv map is one fixed footprint. How does one fixed-size predictor handle a tiny faraway sign and a huge nearby bus?

Let me look at how everyone else handles scale, because I'm clearly fighting the standard playbook. There are two classic answers. One: image pyramids — resize the image to many scales and run the whole conv stack at each scale, so a big object at the coarse scale looks the same size as a small object at the fine scale. It works, and it's exactly what I can't afford: it multiplies the convolution cost, which is the whole thing I'm trying to amortize. Two: filter pyramids — keep one image but run windows/filters of several sizes and aspect ratios over the feature map (DPM trains a 5×7 template and a 7×5 template separately, and so on). Also works, also pays linearly per scale and per aspect ratio in filters. Both schemes share the same shape: you *enumerate* scales somewhere — in the images or in the filters — and you pay for each one. If I do either, I lose the free-sharing property I just bought.

So I want scale-awareness *without* enumerating it in the images or the filters. Where else can it live? It can live in the *output parameterization*. At a single location, instead of predicting one box, let me predict several — but each one is predicted *relative to a different reference box*. I'll pin down, at each sliding position, a fixed family of reference boxes — call them anchors — say one at 128², one at 256², one at 512², each in a few aspect ratios. The predictor at that location outputs, for each anchor, a small correction (a center shift and a size scaling) that turns that anchor into a proposal. So the "small" anchor's predictor learns to refine small things, the "large" anchor's predictor learns to refine large things, and they all read from the *same* fixed 3×3 feature. The image is one scale, the filter is one size; the multiple scales and ratios are carried entirely by the *reference boxes the regressors point at*. It's a pyramid, but a pyramid of regression references, not of images or filters. That's the piece that lets scale ride along for free on top of feature sharing.

Let me make that concrete. If I have k anchors at each location, the reg layer needs 4k outputs — four numbers (dx, dy, dw, dh) per anchor — and the cls layer needs 2k outputs, an object/not-object score per anchor (two-class softmax; could equivalently be k logistic scores). By default I'll take 3 scales × 3 aspect ratios = 9 anchors. For a feature map of, say, 60×40, that's 60×40×9 ≈ 20000 anchors over the image. Each one is a tiling of a reference box at one grid cell.

Now, how exactly do I encode a box as a correction to its anchor? I don't want the network regressing raw pixel coordinates — those aren't scale-invariant, and a 10-pixel error means something different for a 30-pixel object than for a 500-pixel one. The R-CNN parameterization is the right thing: encode the center as an offset *normalized by the anchor's size*, and the width/height as a *log ratio* to the anchor's size:

  t_x = (x − x_a)/w_a,  t_y = (y − y_a)/h_a,  t_w = log(w/w_a),  t_h = log(h/h_a),

with the matching starred targets t* computed from the ground-truth box (x*, y*, w*, h*) the same way. Dividing the center offset by the anchor width makes the target dimensionless and comparable across scales. Putting size in log space means the regressor predicts a multiplicative factor, which is symmetric (halving and doubling are equal and opposite) and keeps w positive automatically when I invert with w = w_a·exp(t_w). To go from prediction back to a box I just invert these: x = t_x·w_a + x_a, w = w_a·exp(t_w), and so on. This is bounding-box regression from an anchor to a nearby ground-truth box.

There's a subtlety here that's worth pausing on, because it's different from how Fast R-CNN does its box regression. In Fast R-CNN, the features for regression are pooled from an *arbitrary* RoI, and a *single* set of regression weights handles all region sizes — the pooling already normalized away the size. In my setup the features for all k anchors at a location are the *same* fixed 3×3 patch; nothing told the regressor whether it's looking at a small thing or a large thing. So I must *not* share regression weights across anchors. I learn k separate regressors, one per anchor shape — each one is responsible for one scale and one aspect ratio. The anchor identity is what supplies the missing scale information: regressor #1 always refines the 128²,2:1 box, regressor #5 always refines the 256²,1:1 box, and so on. That's how a fixed-size feature can still produce boxes of very different sizes.

Good. Now I need to *train* this proposer, which means I need labels on the anchors and a loss.

Labels first. Each anchor is either a positive (it's on an object and should be refined toward it), a negative (it's background), or neither (ambiguous — ignore it). The clean rule: an anchor is positive if it has IoU at least 0.7 with some ground-truth box, negative if its IoU is below 0.3 with *all* ground-truth boxes, and don't-care in between. The don't-care band matters — forcing the network to call a half-overlapping anchor object-or-not is asking it to learn noise. But there's an edge case: what if some ground-truth box is so awkwardly shaped that *no* anchor clears 0.7? Then it would get no positive anchor at all and contribute nothing. So I add a fallback: for each ground-truth box, every anchor tied for the highest IoU to it is also labeled positive, regardless of the 0.7 threshold. In practice the 0.7 rule handles almost everything; the fallback just guarantees every object gets at least one positive.

Now the loss. This is exactly the multi-task shape Fast R-CNN uses, adapted to anchors. For each anchor i, let p_i be the predicted objectness probability and p*_i its label (1 positive, 0 negative); t_i the predicted box correction and t*_i the target. I want

  L = (1/N_cls) Σ_i L_cls(p_i, p*_i) + λ (1/N_reg) Σ_i p*_i L_reg(t_i, t*_i).

L_cls is the log loss over the two classes. L_reg is the smooth-L1 (robust) loss on the four-vector difference t_i − t*_i. Two design points hide in here. The first is that p*_i multiplying L_reg: the regression loss is *gated to positive anchors only*. That's obvious once you say it — there is no meaningful box to regress for a background anchor; "where is the object" is undefined when there's no object — but it's easy to get wrong, and including background anchors in the box loss would pour gradient into nonsense targets. The second is why smooth-L1 and not plain L2 for the box loss. Some anchors will start far from their targets, and a squared loss on a large error produces a large gradient that can dominate a minibatch and destabilize training; smooth-L1 is quadratic near zero (precise when you're close) but only linear far out (bounded gradient, robust to the few wild outliers). That robustness is the whole reason it's the standard choice here.

The two normalizers and λ are there for balancing. If I just summed everything, the cls term — counted over the minibatch of sampled anchors — and the reg term — naturally counted over all the anchor locations — would be on completely different scales, and one would swamp the other. So I normalize cls by the minibatch size (N_cls = 256) and reg by the number of anchor locations (N_reg ≈ 2400), and then I scale reg by λ to bring the two onto roughly equal footing; λ ≈ 10 does it. This is a balancing knob, not a change to the meaning of objectness or box geometry.

Sampling. If I take the loss over *all* anchors in an image, the background ones — there are thousands, almost everything is background — completely dominate and the network just learns to say "not an object." So I follow the image-centric strategy: one image per minibatch, and from it sample 256 anchors with a positive-to-negative ratio of up to 1:1. If fewer than 128 positives exist in the image, I pad the rest with negatives. The 1:1 cap keeps the gradient from being a wall of negatives.

There's a nasty practical detail about anchors near the image edge. Many anchors, especially the big 512² ones tiled near the border, stick out past the image. During training, if I keep these cross-boundary anchors, they introduce large, badly conditioned error terms — a reference box mostly outside the frame is a poor target for fitting an object inside it — and these terms are big enough that the objective doesn't converge. So I *ignore* all cross-boundary anchors during training.

How many does that actually leave? Let me count for a 1000×600 image. Stride 16 gives a feature grid of 1000//16 × 600//16 = 62 × 37 = 2294 cells, and 2294 × 9 = 20646 anchors total — so "~20000" is right. Tiling the nine base anchors over the grid and keeping only those fully inside [0,1000)×[0,600), I'm left with about 8000 (≈ 8150 with this grid). So the cull throws away well over half of them — the dropped ~12000 are almost all the large anchors hanging off the four borders — but it still leaves several thousand fully-inside anchors per image, which is plenty to sample a 256 minibatch from. The point I needed to confirm is that dropping cross-boundary anchors doesn't starve training; it doesn't. At test time I don't have this luxury (I want proposals everywhere), so I keep applying the RPN fully convolutionally and simply *clip* any out-of-bounds proposal to the image boundary afterward.

Initialization and schedule, briefly: the new layers (the 3×3 conv and the two 1×1 convs) get small Gaussian weights, N(0, 0.01); the shared conv layers come from an ImageNet-pretrained classifier, as is standard, because there isn't nearly enough detection data to learn good convs from scratch. Then SGD with momentum 0.9, weight decay 0.0005, learning rate 0.001 for the first 60k minibatches and 0.0001 for the next 20k.

At inference the RPN spits out a box and score for every anchor, ~20000 of them, and many of these overlap heavily — the same object gets nominated by neighboring locations and by several anchors. I don't want to hand the detector 20000 nearly-duplicate boxes. So I run non-maximum suppression on the proposals by their objectness score, with an IoU threshold of 0.7: keep the top-scoring box, drop everything that overlaps it by more than 0.7, repeat. The threshold is high enough to be conservative around nearby objects while still removing the worst redundancy; after that I keep the top-N proposals for the detector.

Now I have a proposer that reads off the shared feature map. But I've been hand-waving "shared," and that word is doing a lot of work, because here's the problem. I have two networks that both want to use the *same* convolutional layers: the RPN (proposer) and the Fast R-CNN (detector). If I train the RPN by itself, it will tune the convs one way. If I train the detector by itself, it'll tune them a different way. They don't agree. So I can't just train them separately and hope the convs they each want are the same convs — they aren't. I need a procedure that actually produces *one* set of convolutional layers that serves both tasks.

Let me think about what would happen if I just merged them into a single network and backpropped both losses at once — approximate joint training. Forward pass: the convs produce the feature map; the RPN head produces proposals; those proposals are fed (as if they were fixed, precomputed boxes) into the RoI-pooling + detection head; both the RPN loss and the detection loss are computed; backward pass combines the gradients into the shared convs. This is clean and it's fast — one network, one optimization. The catch: when I treat the proposals as "fixed, precomputed boxes" going into RoI pooling, I'm pretending the proposal coordinates aren't themselves outputs of the network. But they *are* — the RoI-pooling layer takes the predicted box coordinates as input, so a fully correct gradient would have to flow back through those coordinates too. This scheme drops that term. So it's an approximation. Making it non-approximate would require an RoI-pooling layer that's differentiable with respect to the box coordinates — an "RoI warping" layer — which is a separate nontrivial piece of machinery. I'll set that aside.

There's a more pragmatic route that sidesteps the differentiability question entirely while still forcing the two heads onto one shared backbone: alternate, and freeze the shared convs during the second half. Concretely, four steps.

Step 1: train the RPN alone, starting from the ImageNet-pretrained backbone, fine-tuning end to end for the proposal task. Now I have good proposals, but the backbone has been pulled toward the RPN's wishes.

Step 2: train a *separate* Fast R-CNN detector, also starting fresh from the ImageNet backbone, using the proposals that step-1's RPN produced. At this point the two networks have two different copies of the conv layers — nothing is shared yet — but I now have a detector whose convs are tuned for detection.

Step 3: take the detector's conv layers, *freeze* them, and re-train the RPN on top of those frozen convs — only the RPN-specific layers (the 3×3 and the two 1×1 convs) get updated. Now the RPN and the detector are reading from the *same* convs, because I built the RPN head on top of the detector's frozen backbone. Sharing has begun.

Step 4: keep those same convs frozen, and fine-tune only the detector-specific layers (the RoI head). Now both networks share one identical set of convolutional layers and form a single unified network — the RPN as a head and the detector as a head, on one backbone. The proposer "tells the detector where to look," and they cost one conv pass between them.

Why freeze in steps 3 and 4 instead of letting everything move? Because the freeze is exactly the mechanism that *forces* sharing. If I let the convs move in step 3, they'd drift toward the RPN again and the detector head I trained in step 2 would no longer match them. By pinning the convs to one fixed state and only adapting each head to that state, both heads end up genuinely on the same backbone. It's pragmatic rather than elegant, but it converges fast and it actually delivers the shared features, which is the whole point.

Let me step back and make sure the cascade structure is right, because there's a tempting simpler design I should rule out. Why two stages — propose, then detect — instead of one stage that, like OverFeat, just runs class-specific classifiers and regressors directly on sliding windows and skips the proposal step? In a one-stage design the region features come from a sliding window of one aspect ratio over a scale pyramid, and the same features have to decide *both* where the object is *and* what category it is, in one shot. My two-stage cascade does something the one-stage version can't: the first stage produces class-agnostic proposals cheaply, and then the second stage *re-pools features from the predicted proposal box* — features that actually cover the region the first stage pointed at — and uses *those* to classify and refine. The detector attends to the proposal. That second look, on features adaptively pooled from the candidate region rather than from a generic sliding window, is what I expect to make detections more accurate. So the proposal stage isn't a cost to be eliminated; it's the thing that lets the detector focus.

Let me also sanity-check the anchor scheme against the closest prior attempt, MultiBox, because it also predicts a fixed set of boxes from a net. MultiBox learns ~800 anchor centers by k-means and predicts them from a fully-connected layer applied to a whole image crop. Two things are wrong with that for my purposes. First, those 800 learned boxes are tied to absolute positions — they're *not* translation invariant, so the same object shifted isn't guaranteed the same proposal; my anchors, being a fixed family replicated at every grid cell with a convolutional (shared-weight) head, are translation invariant by construction. Second, MultiBox's output layer is enormous, and it's worth putting an actual number on the gap rather than waving at it. MultiBox emits (4+1)×800 = 3200 outputs from a fully-connected layer; on top of a feature vector that's at least a few thousand wide, that final layer alone is (a few thousand)×3200 ≈ millions of parameters — say roughly 1500×3200 ≈ 4.8M as a conservative floor. My output layer is two 1×1 convs reading a 512-channel input: cls is 512×(2×9) = 9216 weights and reg is 512×(4×9) = 18432, totaling 27648 ≈ 2.8×10⁴. So the comparison is 2.8×10⁴ vs several×10⁶ — roughly two orders of magnitude, exactly as I'd hoped, and now it's a count rather than a vibe. (The shared 3×3 conv ahead of these adds 3·3·512·512 ≈ 2.4M, but that's backbone-side feature mixing shared with nothing to overfit per-box; the *per-box prediction* layer, the part that grows the overfitting risk, is the tiny 2.8×10⁴.) Smaller prediction head, translation invariance for free, and it shares the backbone — that's the position I want to be in.

So the whole thing collapses to a satisfying picture: I was trying to make a proposer faster, hit the wall that a faster-but-separate proposer still wastes the detector's features, and the way out was to stop treating proposal as a separate algorithm and make it a thin convolutional head on the detector's own feature map. The only thing standing between that idea and a single fixed-size head was scale, and anchors — a pyramid of regression references rather than of images or filters — dissolved it. Train it with a gated multi-task loss, share the backbone by alternating-and-freezing, and the proposals come essentially for free.

Now let me write the load-bearing pieces as real code. First, the anchors — generate the k=9 reference boxes for one location, by enumerating aspect ratios (holding area fixed) and then scales.

```python
import numpy as np

def generate_anchors(base_size=16, ratios=[0.5, 1, 2], scales=2 ** np.arange(3, 6)):
    """k reference boxes for one feature-map cell.
    base_size = the network's total stride (one cell ~ 16 px of image)."""
    base_anchor = np.array([1, 1, base_size, base_size]) - 1   # (x1,y1,x2,y2)
    ratio_anchors = _ratio_enum(base_anchor, ratios)           # 3 aspect ratios
    anchors = np.vstack([_scale_enum(ratio_anchors[i, :], scales)  # x3 scales
                         for i in range(ratio_anchors.shape[0])])
    return anchors   # shape (9, 4)

def _whctrs(a):
    w = a[2] - a[0] + 1; h = a[3] - a[1] + 1
    return w, h, a[0] + 0.5 * (w - 1), a[1] + 0.5 * (h - 1)

def _mkanchors(ws, hs, x_ctr, y_ctr):
    ws = ws[:, None]; hs = hs[:, None]
    return np.hstack((x_ctr - 0.5 * (ws - 1), y_ctr - 0.5 * (hs - 1),
                      x_ctr + 0.5 * (ws - 1), y_ctr + 0.5 * (hs - 1)))

def _ratio_enum(anchor, ratios):
    # change aspect ratio while keeping area ~constant: w*h = size, h = w*ratio
    w, h, xc, yc = _whctrs(anchor)
    size = w * h
    ws = np.round(np.sqrt(size / ratios))
    hs = np.round(ws * ratios)
    return _mkanchors(ws, hs, xc, yc)

def _scale_enum(anchor, scales):
    # multiply both sides by each scale -> the 128/256/512 family
    w, h, xc, yc = _whctrs(anchor)
    return _mkanchors(w * scales, h * scales, xc, yc)
```

Before I trust this I should actually run it once in my head (or on paper) and read off the nine boxes, because two things could be wrong: the area might not stay constant across aspect ratios, and the scales might not land where I think. Start from the base cell, the 16-px stride square: as (x1,y1,x2,y2) that's (0,0,15,15), so w = h = 16, center (7.5, 7.5), area 256. Now `_ratio_enum` for ratios [0.5, 1, 2]: it sets ws = round(sqrt(256/ratio)), hs = round(ws·ratio). For ratio 0.5: ws = round(sqrt(512)) = round(22.6) = 23, hs = round(11.5) = 12, area 23·12 = 276. For ratio 1: ws = hs = 16, area 256. For ratio 2: ws = round(sqrt(128)) = round(11.3) = 11, hs = round(22) = 22, area 242. So the areas are 276 / 256 / 242 — *approximately* constant but not exactly, and that's worth noticing rather than papering over: the rounding to integer pixels perturbs the area by a few percent. That's fine — these are reference boxes the regressor will correct anyway — but I should say "area roughly preserved," not "area preserved." The aspect ratios h/w come out 0.52, 1.0, 2.0, which is what I wanted.

Now `_scale_enum` multiplies each of those three by scales 2^3, 2^4, 2^5 = 8, 16, 32. Tiling all nine and reading off widths/heights: I get widths {184, 368, 736} for the wide anchor, {128, 256, 512} for the square, {88, 176, 352} for the tall one, with the matching heights — nine boxes whose sqrt-areas are about {133, 266, 532}, {128, 256, 512}, {125, 249, 498}. So the square anchors hit the {128, 256, 512} family exactly and the rounded ones sit a hair off, which is the same approximation as before showing through. Good: the code genuinely produces a 3-scale × 3-ratio fan of reference boxes centered on the cell, which is exactly the family I argued for — and now I know its quirk (area only roughly fixed, square scales exact) instead of assuming it.

The box parameterization — targets from a reference box to a gt box, and the inverse that turns predicted deltas back into boxes. This is the t = ((x−x_a)/w_a, (y−y_a)/h_a, log(w/w_a), log(h/h_a)) encoding, vectorized.

```python
def bbox_transform(ex_rois, gt_rois):
    ew = ex_rois[:, 2] - ex_rois[:, 0] + 1.0
    eh = ex_rois[:, 3] - ex_rois[:, 1] + 1.0
    ecx = ex_rois[:, 0] + 0.5 * ew; ecy = ex_rois[:, 1] + 0.5 * eh
    gw = gt_rois[:, 2] - gt_rois[:, 0] + 1.0
    gh = gt_rois[:, 3] - gt_rois[:, 1] + 1.0
    gcx = gt_rois[:, 0] + 0.5 * gw; gcy = gt_rois[:, 1] + 0.5 * gh
    dx = (gcx - ecx) / ew          # center offset normalized by anchor size
    dy = (gcy - ecy) / eh
    dw = np.log(gw / ew)           # size correction in log space
    dh = np.log(gh / eh)
    return np.vstack((dx, dy, dw, dh)).T

def bbox_transform_inv(boxes, deltas):
    if boxes.shape[0] == 0:
        return np.zeros((0, deltas.shape[1]), dtype=deltas.dtype)
    w = boxes[:, 2] - boxes[:, 0] + 1.0
    h = boxes[:, 3] - boxes[:, 1] + 1.0
    cx = boxes[:, 0] + 0.5 * w; cy = boxes[:, 1] + 0.5 * h
    dx, dy, dw, dh = deltas[:, 0::4], deltas[:, 1::4], deltas[:, 2::4], deltas[:, 3::4]
    px = dx * w[:, None] + cx[:, None]      # invert: x = t_x*w_a + x_a
    py = dy * h[:, None] + cy[:, None]
    pw = np.exp(dw) * w[:, None]            # invert: w = w_a*exp(t_w)
    ph = np.exp(dh) * h[:, None]
    out = np.zeros_like(deltas)
    out[:, 0::4] = px - 0.5 * pw; out[:, 1::4] = py - 0.5 * ph
    out[:, 2::4] = px + 0.5 * pw; out[:, 3::4] = py + 0.5 * ph
    return out

def clip_boxes(boxes, im_shape):
    boxes[:, 0::4] = np.clip(boxes[:, 0::4], 0, im_shape[1] - 1)
    boxes[:, 1::4] = np.clip(boxes[:, 1::4], 0, im_shape[0] - 1)
    boxes[:, 2::4] = np.clip(boxes[:, 2::4], 0, im_shape[1] - 1)
    boxes[:, 3::4] = np.clip(boxes[:, 3::4], 0, im_shape[0] - 1)
    return boxes
```

The encode/decode pair has to be a clean inverse, or training targets and test-time boxes won't mean the same thing, so let me run one box through both and check it comes back. Take an anchor (100,100,355,355) — that's the 256-px square, w = h = 256, center (227.5, 227.5) — and a ground-truth box (120,140,300,360), w = 181, h = 221, center (210, 250). Encoding: t_x = (210−227.5)/256 = −0.068, t_y = (250−227.5)/256 = +0.088, t_w = log(181/256) = −0.347, t_h = log(221/256) = −0.147. Now feed (anchor, t) through the inverse: it recomputes center 227.5 + (−0.068)·256 = 210 and 227.5 + 0.088·256 = 250 — center recovered exactly — and size 256·exp(−0.347) = 181, 256·exp(−0.147) = 221 — size recovered exactly. Reassembling corners from center and size gives (120, 140, 301, 361). Against the true (120,140,300,360) that's off by exactly 1 px on the far corner.

That 1-px miss is not a bug to chase, but I want to know where it comes from before I move on. The center and the width/height come back exact; the discrepancy is entirely in turning (center, w) back into x2. The encoder measured w as x2−x1+1 = 256 (the "+1" pixel-count convention), but the decoder lays the box down as center ± 0.5·w, which spans 256 pixels symmetric about the center and so lands x2 half a pixel out, and the same on x1 — netting a 1-px shift on the far corner after the asymmetric rounding of the center. So the parameterization is *exact in the space it actually optimizes* — (center offset, log size) — and the only slack is a sub-pixel corner artifact from the inclusive width convention, which at proposal granularity (boxes feeding an RoI-pool that quantizes to the 16-px grid anyway) is below the noise floor. Good — the inverse is faithful where it matters, and I now know the one place it isn't perfectly so. As a second check, a zero delta on that anchor should return the anchor; it returns (100,100,356,356), the same harmless +1 on the far corner, confirming the artifact is the width convention and nothing in the math.

Assigning targets to anchors for training: tile the 9 anchors over the W×H grid (the convolutional sliding window made explicit), drop the cross-boundary ones, compute IoU to the ground-truth boxes, apply the 0.7 / 0.3 / highest-IoU labeling, subsample to 256 at up-to-1:1, and gate the regression targets to positives.

```python
# Existing detector-stack primitive: bbox_overlaps(boxes, query_boxes).
def anchor_targets(anchors, gt_boxes, im_info, feat_h, feat_w,
                   feat_stride=16, pos_thr=0.7, neg_thr=0.3,
                   batch=256, fg_frac=0.5):
    # tile the k base anchors over every feature-map cell (the FCN sliding window)
    sx = np.arange(feat_w) * feat_stride
    sy = np.arange(feat_h) * feat_stride
    sx, sy = np.meshgrid(sx, sy)
    shifts = np.vstack((sx.ravel(), sy.ravel(), sx.ravel(), sy.ravel())).T
    A, K = anchors.shape[0], shifts.shape[0]
    all_anchors = (anchors.reshape(1, A, 4) +
                   shifts.reshape(1, K, 4).transpose(1, 0, 2)).reshape(K * A, 4)

    # ignore cross-boundary anchors during training (else training won't converge)
    inside = np.where((all_anchors[:, 0] >= 0) & (all_anchors[:, 1] >= 0) &
                      (all_anchors[:, 2] < im_info[1]) &
                      (all_anchors[:, 3] < im_info[0]))[0]
    anc = all_anchors[inside]

    labels = np.full((len(inside),), -1, dtype=np.float32)   # -1 = ignore
    ov = bbox_overlaps(anc, gt_boxes)                        # IoU matrix
    argmax = ov.argmax(axis=1); max_ov = ov[np.arange(len(inside)), argmax]
    gt_argmax = np.where(ov == ov.max(axis=0))[0]

    labels[max_ov < neg_thr] = 0          # negatives: IoU < 0.3 with all gt
    labels[gt_argmax] = 1                 # fallback: best anchor per gt is positive
    labels[max_ov >= pos_thr] = 1         # positives: IoU >= 0.7 with some gt

    # subsample so positives:negatives is up to 1:1 within a 256 minibatch
    num_fg = int(fg_frac * batch)
    fg = np.where(labels == 1)[0]
    if len(fg) > num_fg:
        labels[np.random.choice(fg, len(fg) - num_fg, replace=False)] = -1
    num_bg = batch - np.sum(labels == 1)
    bg = np.where(labels == 0)[0]
    if len(bg) > num_bg:
        labels[np.random.choice(bg, len(bg) - num_bg, replace=False)] = -1

    targets = bbox_transform(anc, gt_boxes[argmax, :4])      # regress to matched gt
    inside_w = np.zeros((len(inside), 4), dtype=np.float32)
    inside_w[labels == 1, :] = 1.0        # gate reg loss to positive anchors only
    return labels, targets, inside_w, inside
```

And inference: turn the dense predictions into a short proposal list — apply the deltas to the tiled anchors, clip to the image, drop tiny boxes, take the top pre-NMS by score, NMS at 0.7, keep the top-N.

```python
# Existing detector-stack primitive: nms(dets, thresh).
def generate_proposals(scores, bbox_deltas, anchors, im_info,
                       feat_h, feat_w, feat_stride=16,
                       pre_nms=6000, post_nms=300, nms_thr=0.7, min_size=16):
    sx = np.arange(feat_w) * feat_stride
    sy = np.arange(feat_h) * feat_stride
    sx, sy = np.meshgrid(sx, sy)
    shifts = np.vstack((sx.ravel(), sy.ravel(), sx.ravel(), sy.ravel())).T
    A, K = anchors.shape[0], shifts.shape[0]
    anc = (anchors.reshape(1, A, 4) +
           shifts.reshape(1, K, 4).transpose(1, 0, 2)).reshape(K * A, 4)

    if scores.shape[1] == 2 * A:                                # bg/fg softmax layout
        scores = scores[:, A:, :, :]
    deltas = bbox_deltas.transpose(0, 2, 3, 1).reshape(-1, 4)
    scores = scores.transpose(0, 2, 3, 1).reshape(-1, 1)     # objectness (fg) scores

    proposals = bbox_transform_inv(anc, deltas)              # anchors -> boxes
    proposals = clip_boxes(proposals, im_info[:2])           # test-time: clip, don't drop
    keep = _filter_boxes(proposals, min_size * im_info[2])
    proposals, scores = proposals[keep], scores[keep]

    order = scores.ravel().argsort()[::-1]                   # keep strongest first
    if pre_nms > 0:
        order = order[:pre_nms]
    proposals, scores = proposals[order], scores[order]
    keep = nms(np.hstack((proposals, scores)), nms_thr)       # dedup -> top-N
    if post_nms > 0:
        keep = keep[:post_nms]
    return proposals[keep], scores[keep]

def _filter_boxes(boxes, min_size):
    ws = boxes[:, 2] - boxes[:, 0] + 1
    hs = boxes[:, 3] - boxes[:, 1] + 1
    return np.where((ws >= min_size) & (hs >= min_size))[0]
```

The causal chain, end to end: the detector's convolutions got cheap, which made the external proposer the bottleneck; a faster-but-separate proposer would still waste the detector's feature map, so I moved proposal generation onto that shared map as a tiny fully-convolutional head; that head is translation-invariant by construction and nearly free because the convolutions are already paid for; the one obstacle — handling many scales from a single fixed-size feature — fell to anchors, a pyramid of reference boxes that carry scale in the regression targets instead of in the images or filters; I trained the head with a gated classification-plus-smooth-L1 multi-task loss, balanced sampling, and cross-boundary anchors dropped; and I welded the proposer and detector onto one shared backbone by alternating training with the convs frozen in the later steps, so at test time a single conv pass serves both heads.
