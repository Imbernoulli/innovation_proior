OK, let me think this through from scratch. The state of object detection right now is lopsided. The accurate detectors are all two-stage: a first stage proposes a sparse set of candidate regions, a second stage classifies and refines them. R-CNN started it, Fast R-CNN sped it up with RoI pooling, Faster R-CNN folded the proposal step into a Region Proposal Network so the whole thing is one differentiable pipeline — and these top the COCO leaderboard. Meanwhile the one-stage detectors, SSD and YOLO, do the obvious simple thing: a fully convolutional net sweeps a dense grid of boxes over the image and predicts a class and a box offset at every location, in a single shot, no proposals. They're faster and simpler. And they're worse — something like 10–40% lower relative AP. People have basically accepted this as the price of speed.

I don't want to accept it. I want to know *why*. If I take a one-stage detector and just make it bigger or train it longer, does the gap close? Prior speed/accuracy sweeps say no — even when you handicap the two-stage detector down to comparable compute, the one-stage one still trails. So it isn't raw capacity. Something about the one-stage setup is structurally harder to train. Let me figure out what.

The difference that jumps out is the set of examples that reaches the classifier. A two-stage detector's first stage — selective search, EdgeBoxes, an RPN — takes the essentially infinite set of possible boxes and hands the classifier only one or two thousand candidates, and crucially those candidates are *not random*: they're the ones that look object-like, so most of the trivial background has already been thrown away. The second-stage classifier then trains on minibatches deliberately built with a fixed ratio, say 1:3 positives to negatives. So by the time a loss is computed, the example set the classifier sees is roughly balanced.

A one-stage detector has no such filter. It predicts over a *regular, dense* sampling of the image — every position, several scales, several aspect ratios. Tile all that out and you're enumerating on the order of 10^4 to 10^5 candidate boxes per image. And in a typical image, how many of those contain an object? A handful. Dozens, maybe. So the foreground-to-background ratio is something like 1 to 1000. That's the asymmetry. The two-stage cascade *is* an imbalance filter; the one-stage detector eats the full imbalance raw.

Let me make sure imbalance is really the culprit and not just a suspicious correlate. What goes wrong when almost everything is an easy negative? Two things. One, it's wasteful — the vast majority of locations are background patches the model already gets right, contributing no useful gradient. That alone just slows things down. But two, and this is the one I suspect actually breaks training: *en masse* the easy negatives might swamp the loss. Each easy negative individually contributes almost nothing, but there are 10^5 of them, and "almost nothing" times 10^5 need not be small. If the total loss — and therefore the total gradient — ends up dominated by the sum over easy background, the optimizer spends its capacity nudging already-correct background predictions and the rare foreground examples barely register, and you'd converge to a degenerate model that's very good at saying "background." But "might" and "need not be" are hand-waves; I should put numbers on it before I believe it.

Binary cross entropy: for a label y ∈ {±1} and the model's probability p for the y=1 class,

CE = −log p if y=1, and −log(1−p) if y=−1.

Let me collapse the cases by defining the probability assigned to the *correct* class: p_t = p when y=1, and p_t = 1−p when y=−1. Then both branches are just

CE(p_t) = −log p_t.

Clean. Now put numbers through it. Take an easy, well-classified example — a background patch the model is confident about, say p_t = 0.9. Its loss is −log(0.9) ≈ 0.105. Take a really easy one, p_t = 0.99: loss ≈ 0.010. These are small. But the property I want to test is whether they're small *enough*, in bulk, to stay out of the way. Suppose a single foreground example at p_t = 0.5 (genuinely uncertain): its loss is −log(0.5) ≈ 0.693. Now picture a realistic image: a few dozen foreground anchors at, say, O(1) loss apiece — call it 30 × 0.7 ≈ 21 total — against the sea of background. If 10^5 background anchors each sit at p_t = 0.9, their total is 10^5 × 0.105 ≈ 1.05×10^4. So the background outweighs the foreground by a factor of ~500 even when the background is *already confidently correct*. The easy negatives don't just contribute — they are essentially the entire loss, and the gradient inherits the same lopsidedness: each easy negative's push is tiny, but 10^5 tiny pushes in the "background" direction add up to a total gradient that points almost entirely at making background-prediction marginally better. That's the swamping made concrete: not from a special pathological case, but from background anchors the model has *already solved*. The culprit is the −log p_t never decaying to zero — a solved example still pays a 0.1-nat toll, and 0.1 times 10^5 buries everything.

So how do I fix it? Let me first reach for the standard tool, because if it works I'm done. The textbook fix for class imbalance is to weight the classes. Introduce α ∈ [0,1] for the positive class and 1−α for the negative class — set α by inverse frequency, or just tune it. Define α_t by the same case-split I used for p_t. Then

CE(p_t) = −α_t log p_t.

Does this solve it? Think about what α does: it rescales the positive examples relative to the negative examples. If positives are rare, crank α up and each positive counts for more. That rebalances *positive versus negative*. But walk through what's actually drowning me. The flood isn't "negatives" as a category — it's *easy* negatives. Within the negative class, the trouble is that almost all of them are trivially-correct background. α multiplies every negative by the same 1−α regardless of whether it's a hard, ambiguous, object-like patch or a flat piece of sky. It cannot tell easy from hard. It moves the positive/negative knob but leaves the easy/hard distribution within each class untouched — so the easy negatives, now scaled by a constant, still vastly outnumber and outweigh everything else. α is the wrong axis. I need to discriminate not by *class membership* but by *how well-classified the example already is*.

So what does "this example is already well-classified, ignore it" look like as a number? An example is well-classified exactly when p_t is large. So I want a factor that multiplies the loss, sits near 1 when p_t is small (the example is hard or wrong — leave its loss alone, I care about it), and shrinks toward 0 as p_t → 1 (the example is easy — kill its contribution). And it should do this smoothly and aggressively. The most natural thing that goes from 1 at p_t=0 to 0 at p_t=1 is (1−p_t). Raise it to a power to control how sharply: (1−p_t)^γ, with γ ≥ 0. Multiply it onto cross entropy:

FL(p_t) = −(1−p_t)^γ log p_t.

Let me check the two ends, because this only earns its keep if it behaves right at both. Hard example, p_t small — say p_t = 0.1: the modulating factor (1−0.1)^γ = 0.9^γ ≈ 1 for moderate γ. The loss is essentially unchanged. Good — I keep full attention on hard examples. Misclassified example, p_t → 0: factor → 1, untouched. Easy example, p_t → 1: factor → 0, the loss is suppressed. That's the qualitative shape I was after, but the question is whether the *quantitative* gap is big enough to flip the swamping I just computed.

So let me pin down what γ buys and put numbers on it. Take γ = 2. An easy example at p_t = 0.9: the factor is (1−0.9)^2 = (0.1)^2 = 0.01. So its loss is 100× smaller than under plain CE. Push to p_t ≈ 0.968: (0.032)^2 ≈ 10^−3, a 1000× reduction. Now a hard example at p_t = 0.5: factor (0.5)^2 = 0.25 — only a 4× reduction, and for anything harder than that (p_t < 0.5) the down-weighting is at most 4×. So the well-classified examples get crushed by two or three orders of magnitude while the hard ones are still present.

Let me redo the swamping arithmetic under this factor and see if the balance actually flips. The 10^5 easy background at p_t = 0.9 had total CE ≈ 1.05×10^4; multiply each by 0.01 and the focal total drops to ≈ 1.05×10^2. The few dozen foreground at O(1) loss barely move (a foreground at p_t=0.5 is only divided by 4, ~5 total). So I've gone from background outweighing foreground ~500:1 to roughly 100:5 ≈ 20:1 — and that's with the background still sitting at a *modest* 0.9 confidence; in practice the easy negatives sit much closer to p_t=0.99, where the factor is 10^−4 and they essentially vanish (10^5 × 0.01005 × 10^−4 ≈ 0.1 total). The flood is drained. The *relative* weight of hard examples shoots up by orders of magnitude even though I never explicitly sampled or selected anything — the loss reweights itself, continuously, per example, every iteration. And γ = 0 recovers (1−p_t)^0 = 1, i.e. plain CE, so γ is a dial that smoothly turns the focusing on. Something around γ = 2 has the right scale: small enough that hard examples are not erased (≤ 4×), large enough that the easy negatives collapse to near-irrelevance.

Hard example mining — OHEM — has the same *spirit*: it scores examples by loss, runs NMS, and builds the minibatch from the highest-loss ones, so it too pushes attention onto hard examples. But it does it by *selection*: it throws the easy examples away entirely. That discards whatever residual signal they carry, and it drags in machinery and knobs — batch size, NMS threshold, the 1:3 ratio — that I now have to tune. The focal loss is the continuous version of the same idea: instead of a hard keep/discard decision per minibatch, every example stays in, weighted by a smooth function of how hard it currently is. No sampling, no selection, applied to *all* ~10^5 anchors at once. That feels strictly cleaner.

And there's a nice way to see what I am and am not doing. There's a whole literature on robust losses — Huber loss and friends — built to *down-weight outliers*, i.e. examples with large error, the hard ones, so a few bad points don't wreck a regression. What I've written does the *opposite*: it down-weights the easy *inliers* so their sheer number doesn't wreck training, and it leaves the hard examples at full strength. Same machinery — a multiplicative reweighting by error — pointed the other way. The problem here isn't a few outliers, it's a flood of inliers.

Now, do I still want α? Let me not throw it away too fast. α and the modulating factor address different things — α is positive-vs-negative, (1−p_t)^γ is easy-vs-hard — so they're not redundant; they compose. Put them together:

FL(p_t) = −α_t (1−p_t)^γ log p_t.

In practice this α-balanced form is slightly better than the bare one, so I'll keep it. But note the interaction: once γ is doing its job and the easy negatives are down-weighted by ~100×, the negative class as a whole is already much quieter, so I need *less* of the positive up-weighting that α provides. The two have to be picked together, and α should come *down* as γ goes up. For γ=2, something like α=0.25 sits right. That coupling falls straight out of the fact that both are turning down the same flood from different directions.

Now a separate problem that's going to bite before the loss even gets a chance to work: initialization. A freshly-initialized classifier outputs roughly p ≈ 0.5 for everything — equal odds of foreground and background at every one of the 10^5 locations. Let me see how bad the first step is. At p = 0.5 a background anchor has p_t = 0.5, and even with γ = 2 the focal factor is only (0.5)^2 = 0.25 — the modulating factor barely helps, because the model isn't confident yet, so it can't tell these are "easy." So 10^5 background anchors each contribute 0.25 × (−log 0.5) ≈ 0.17, for a total ≈ 1.7×10^4 on the very first iteration. That's an enormous, abrupt loss spike from the frequent class right at step one — exactly the regime where a large gradient can blow the weights up and the net diverges. The focal loss only saves me *after* the model has learned to be confident about easy background; at initialization it hasn't, so the modulating factor is near its maximum and offers little protection. So I'd expect plain CE with default init to die fast, and even focal loss to be fragile at step one.

The fix shouldn't touch the loss — the loss is fine; it's the *starting point* that's pathological. I want the model to *start* believing, correctly, that almost everything is background. So I'll bias the initialization toward the rare class being rare: pick a prior probability π that the model assigns to foreground at the start — small, like π = 0.01 — and set the final classification layer's bias so its initial output is exactly that. With a sigmoid, output p = σ(b) = π means I solve for the bias: σ(b) = 1/(1+e^{−b}) = π gives e^{−b} = (1−π)/π, so

b = −log((1−π)/π).

Let me check the value and what it does to that first-step total. For π = 0.01, b = −log(0.99/0.01) = −log(99) ≈ −4.595, and σ(−4.595) = 0.01 back, good. Now a background anchor starts at p = 0.01, so p_t = 0.99, its focal loss is (0.99)^2 × (−log 0.99) ≈ 0.9801 × 0.01005 ≈ 9.85×10^−3 — and 10^5 of them sum to ≈ 985, down from ~1.7×10^4. The first-iteration background loss has dropped by ~17× *just from initialization*, and only the few real foregrounds — which start at p ≈ 0.01 and so *are* badly mispredicted (p_t = 0.01, factor ≈ 0.98, loss ≈ 4.5 apiece) — generate the meaningful initial gradient. The spike is gone and the signal points at the right examples. This is a one-line change to model init, not to the objective, and it gives the loss a stable starting point from which it can decide which anchors deserve attention.

Let me also convince myself about the gradient, since that's what actually trains the thing — I want to confirm the focal loss does the right thing to the gradient and not just the loss value. Work in the logit. Let x be the logit, x_t = y·x, so p_t = σ(x_t), and recall σ'(x_t) = p_t(1−p_t). For plain CE, dCE/dx works out to y(p_t − 1): for a confidently-correct example p_t → 1 it goes to 0, for a confidently-wrong one p_t → 0 it goes to −y, i.e. magnitude 1. Now the focal loss. Take y = 1 so p_t = p = σ(x). FL = −(1−p)^γ log p. Differentiate with respect to p:

dFL/dp = γ(1−p)^{γ−1} log p − (1−p)^γ / p.

Chain through dp/dx = p(1−p):

dFL/dx = [γ(1−p)^{γ−1} log p − (1−p)^γ/p] · p(1−p) = γ p (1−p)^γ log p − (1−p)^{γ+1}.

Factor out (1−p)^γ:

dFL/dx = (1−p)^γ (γ p log p + p − 1),

and restoring the sign-carrying y for both labels,

dFL/dx = y(1−p_t)^γ(γ p_t log p_t + p_t − 1).

Sanity check: set γ = 0 and it collapses to y(p_t − 1), the CE gradient — good. And the new structure is the (1−p_t)^γ prefactor sitting in front: for an easy example with p_t near 1, that prefactor is tiny, so the gradient contribution is tiny *on top of* whatever CE already did. The down-weighting happens in the gradient, not just the printed loss value — which is the only place it could matter. As soon as an example is on the right side and confident, its push on the weights collapses, and the optimizer's budget flows to the examples still getting it wrong.

This also makes me less attached to the exact algebraic form. The real requirement is a loss whose derivative becomes small soon after the signed logit crosses onto the correct side. Let x_t = yx and p_t = σ(x_t). I can make a second family by steepening and shifting that signed logit before applying the ordinary log loss:

p_t^* = σ(γ x_t + β),

FL^* = −log(p_t^*) / γ.

Different γ and β values change where the curve turns down and how sharp the turn is. Differentiate it to see the condition directly. Since d[−log σ(z)]/dz = σ(z) − 1 and z = γx_t + β, the outer division by γ cancels the γ from dz/dx_t; then dx_t/dx = y, so

dFL^*/dx = y(p_t^* − 1).

That derivative is the same CE-shaped derivative, but applied to the shifted, sharpened signed logit. If x_t is comfortably positive, p_t^* is close to 1 and the gradient is tiny; if x_t is negative, the gradient is still large. So the main loss is not a magic single expression. The design principle is: preserve pressure on wrong or ambiguous anchors and rapidly remove pressure from already-correct ones.

One implementation note before architecture: computing p with a sigmoid and then feeding it into log can be numerically nasty at the extremes. I'll fuse the sigmoid and the loss into a single layer — compute the loss directly from the logits — so the log and the sigmoid cancel in a stable way. That's a standard binary-cross-entropy-with-logits underneath, with the modulating factor multiplied on.

Now I need a detector to hang this on, and the whole point is that the *loss* is the contribution, so the network should be deliberately plain — no architectural cleverness to muddy the comparison. Build it dense and fully convolutional. For features, a single-resolution backbone won't cut it for multi-scale objects; I'll use a feature pyramid — a top-down pathway with lateral connections on top of a ResNet — so I get a stack of feature maps at several resolutions, each responsible for one band of object scales. Build pyramid levels P3 through P7 (P3–P5 from the corresponding ResNet stages via the top-down/lateral construction; P6 a strided conv on the last ResNet stage; P7 a ReLU then another strided conv on P6, which extends the reach to large objects and is cheap). All levels carry the same channel width, 256.

At each level, tile anchors — translation-invariant reference boxes, the idea from RPN. To cover scale and shape densely I put A = 9 anchors at each location: three aspect ratios {1:2, 1:1, 2:1} times three size multipliers {2^0, 2^{1/3}, 2^{2/3}}, with base areas growing from 32² at P3 to 512² at P7. Assign each anchor to a ground-truth box by IoU: ≥ 0.5 is foreground (one-hot to its class), in [0, 0.4) is background, and the gap [0.4, 0.5) is ignored so I don't train on ambiguous half-overlaps.

Two small subnets, one per task, attached to every pyramid level with weights shared across levels. The classification subnet: four 3×3 convs (256 filters, ReLU) then a final 3×3 conv emitting K·A channels — K class scores for each of the A anchors per location — with a sigmoid, so K independent binary predictions per anchor. The box subnet is the same little FCN but ends in 4·A linear outputs, the box offsets, class-agnostic. Classification and localization are related but not the same problem: the classifier needs to separate object-like hard negatives from true objects, while the regressor only trains on positives and needs geometric precision. Sharing the feature tower would let those gradients fight inside the one component whose loss behavior I am trying to isolate, so I keep the tower pattern shared but the parameters separate. And the final cls conv gets that prior-π bias from before.

Then the loss. This is the part that breaks from common practice: I apply the focal loss to *every* anchor in the image — all ~100k of them — not to a sampled subset of 256 the way RPN or OHEM or SSD's mining would. That's the whole bet: the loss is robust enough to imbalance that I don't need to sample at all. One catch on normalization: if I divide the summed focal loss by the *total* number of anchors, I'd be dividing by ~100k, almost all of which are easy negatives contributing near-zero loss under the modulating factor — that would make the per-example loss vanishingly small and scale-dependent on the image's anchor count. So normalize instead by the number of anchors *assigned to a ground-truth box* — the positives. That's the count of examples actually carrying signal, and it keeps the loss scale sane. Box regression rides along as a standard smooth-L1 on the positives. Train with SGD.

Before I commit to writing it all out, let me trace the loss expression I'll implement on two anchors by hand, to be sure the code does what the algebra promised — `ce_loss * (1 − p_t)^γ`, with `p_t = p·target + (1−p)·(1−target)` to fold the two label cases. Take both anchors at the post-init logit −4.595 (p ≈ 0.01). Anchor A is background (target 0): p_t = 1 − 0.01 = 0.99, factor (1−0.99)^2 = 10^−4, CE = −log 0.99 ≈ 0.01005, so focal ≈ 1.0×10^−6. Anchor B is foreground the model currently gets wrong (target 1): p_t = 0.01, factor (1−0.01)^2 = 0.9801, CE = −log 0.01 ≈ 4.605, so focal ≈ 4.51. The hard foreground outweighs the easy background by ≈ 4.5×10^6 to 1. Now compare what plain CE would have done on the *same* two anchors: 0.01005 vs 4.605, a ratio of only ≈ 458. So switching the loss shape buys four extra orders of magnitude of relative emphasis on the example that's actually wrong — without touching the data, the sampling, or the network. That four-orders-of-magnitude gap between the focal ratio and the CE ratio is the whole mechanism, and it's exactly the `(1−p_t)^γ` prefactor doing it. Good — the implementation just has to reproduce this expression numerically stably.

Let me write the pieces so the implementation follows the same chain: dense heads first, then the per-entry classification loss, then the shared normalizer for classification and box regression.

```python
import math
import torch
from torch import nn
from torch.nn import functional as F


class DenseHead(nn.Module):
    def __init__(self, in_channels, num_anchors, num_classes, conv_dims=None, initial_prob=0.01):
        super().__init__()
        conv_dims = conv_dims or [in_channels] * 4
        cls_layers = []
        bbox_layers = []
        prev_channels = in_channels
        for out_channels in conv_dims:
            cls_layers += [nn.Conv2d(prev_channels, out_channels, 3, padding=1), nn.ReLU()]
            bbox_layers += [nn.Conv2d(prev_channels, out_channels, 3, padding=1), nn.ReLU()]
            prev_channels = out_channels

        self.cls_subnet = nn.Sequential(*cls_layers)
        self.bbox_subnet = nn.Sequential(*bbox_layers)
        self.cls_score = nn.Conv2d(prev_channels, num_anchors * num_classes, 3, padding=1)
        self.bbox_pred = nn.Conv2d(prev_channels, num_anchors * 4, 3, padding=1)

        for module in [self.cls_subnet, self.bbox_subnet, self.cls_score, self.bbox_pred]:
            for layer in module.modules():
                if isinstance(layer, nn.Conv2d):
                    nn.init.normal_(layer.weight, mean=0, std=0.01)
                    nn.init.constant_(layer.bias, 0)

        bias = -math.log((1 - initial_prob) / initial_prob)
        nn.init.constant_(self.cls_score.bias, bias)

    def forward(self, features):
        logits = []
        bbox_reg = []
        for feature in features:
            logits.append(self.cls_score(self.cls_subnet(feature)))
            bbox_reg.append(self.bbox_pred(self.bbox_subnet(feature)))
        return logits, bbox_reg


def dense_binary_loss(inputs, targets, loss_params, reduction="sum"):
    inputs = inputs.float()
    targets = targets.float()
    loss_params = {"alpha": 0.25, "gamma": 2.0} if loss_params is None else loss_params
    alpha = loss_params.get("alpha", -1)
    gamma = loss_params.get("gamma", 2.0)

    p = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()
    return loss


class PositiveAnchorNormalizer:
    def __init__(self, momentum=100):
        self.momentum = momentum
        self.value = None

    def update(self, num_pos):
        num_pos = max(float(num_pos), 1.0)
        if self.value is None:
            self.value = num_pos
        else:
            self.value = self.value * (self.momentum - 1) / self.momentum + num_pos / self.momentum
        return self.value


def classification_loss(pred_logits, gt_labels, num_classes, loss_normalizer, loss_params):
    logits = torch.cat(pred_logits, dim=1) if isinstance(pred_logits, (list, tuple)) else pred_logits
    gt_labels = torch.stack(gt_labels) if isinstance(gt_labels, (list, tuple)) else gt_labels

    valid_mask = gt_labels >= 0
    pos_mask = valid_mask & (gt_labels != num_classes)
    normalizer_value = loss_normalizer.update(pos_mask.sum().item())

    targets = F.one_hot(gt_labels[valid_mask], num_classes + 1)[:, :-1]
    loss_cls = dense_binary_loss(
        logits[valid_mask], targets.to(logits.dtype), loss_params, reduction="sum"
    )
    return loss_cls / normalizer_value, normalizer_value, pos_mask


def detector_losses(anchors, pred_logits, pred_deltas, gt_labels, gt_boxes,
                    num_classes, loss_normalizer, loss_params):
    loss_cls, normalizer_value, pos_mask = classification_loss(
        pred_logits, gt_labels, num_classes, loss_normalizer, loss_params
    )
    deltas = torch.cat(pred_deltas, dim=1) if isinstance(pred_deltas, (list, tuple)) else pred_deltas
    target_deltas = encode_boxes(anchors, gt_boxes)

    if pos_mask.any():
        loss_box = smooth_l1_loss(deltas[pos_mask], target_deltas[pos_mask], reduction="sum")
    else:
        loss_box = deltas.sum() * 0

    return {"loss_cls": loss_cls, "loss_box_reg": loss_box / normalizer_value}
```

So the whole chain, in one breath: one-stage dense detectors trail two-stage ones, and the reason isn't capacity — it's that the dense grid hands the loss a 1:1000 flood of easy background, and cross entropy's refusal to forget well-classified examples lets that flood dominate the gradient and produce a degenerate model. Class weighting α can't fix it because it can't tell easy from hard. So reshape the loss itself: multiply cross entropy by (1−p_t)^γ, a factor that's ~1 for hard or wrong examples and collapses toward zero — by 100× at p_t=0.9, 1000× at p_t≈0.97 for γ=2 — for the easy ones, which re-weights training toward the hard examples continuously and per-example without any sampling. Add α back for positive/negative balance, lower it as γ rises, fix the first-iteration explosion with a prior-π bias on the output layer, and hang the whole thing on the plainest dense detector I can build — a feature pyramid on ResNet, anchors, two small shared subnets — applying the focal loss to every one of the ~100k anchors and normalizing by the assigned positives. The loss is the contribution; the network is deliberately ordinary.
