OK, let me think about detection from scratch, starting from what actually bugs me about how we do it. The thing I want out of a detector is a set: for an image, give me the objects — each one a class and a box — and give me each object exactly once. That's the whole task. Unordered, variable in size, no element allowed to name the same object twice. And yet every strong detector I know does something that, when I say it out loud, sounds like a workaround rather than the task. We tile the image with tens of thousands of reference boxes — anchors, or proposals, or candidate centers — and then we ask, for each reference independently, "is there an object here, and how should I nudge this reference to fit it?" We've turned one set-prediction problem into a huge bag of independent per-location classification-plus-regression problems.

Why does that bother me? Because of everything that has to be bolted on to make it work, none of which is the task. First, the reference field itself: somebody has to pick the anchor scales, the aspect ratios, the strides. And it's not a cosmetic choice — the diagnostics on current detectors show accuracy moves a lot with exactly how those initial guesses are laid down; when you control for how references get assigned to ground truth, the supposed gap between anchor-based and center-based detectors mostly evaporates. So a big chunk of "detector performance" is really a property of the hand-designed reference grid, not of the learned weights. That's a smell.

Second, the assignment rule. To train per-reference, every reference needs a target. The standard rule is overlap-based: if a reference's IoU with a ground-truth box clears some threshold it's a positive for that object, below another it's a negative, in between it's ignored. But the field is dense, so *many* references clear the threshold for the same object. Assignment is many-to-one by construction. We are deliberately telling a dozen references "you are all responsible for this one dog."

And that forces the third thing: non-maximum suppression. If a dozen references each predict a box for the dog, inference spits out a dozen overlapping dog boxes, and we collapse them with a greedy procedure — sort by confidence, keep the top one, throw away anything overlapping it past a threshold, repeat. NMS is not learned, not differentiable, has yet another threshold, and runs outside the model entirely. NMS exists *only because we created duplicates during training*. It's a cleanup for a mess we made on purpose. If training had somehow forced exactly one prediction per object, there'd be nothing to suppress — NMS would have no job.

So the actual situation is that the loss optimizes a surrogate (per-reference classification and box regression) while the thing I care about — a clean set, one box per object — is manufactured by a separate non-learned stage. That's not end-to-end. Translation and speech became end-to-end years ago; you predict the structured thing directly and backprop through the whole pipeline. Detection just... didn't. People tried, but the attempts either smuggled in some other prior, or never got competitive with strong baselines on a hard benchmark. I want to know whether you can predict the set *directly* — no surrogate task, no hand-tuned reference grid, no assignment heuristic, no NMS — and still match the best detectors.

Let me try to design that and see where it breaks.

If I'm going to predict a set directly, the first problem is variable size. Images have anywhere from zero to dozens of objects. Networks like fixed-size outputs. The cleanest move: commit to a fixed number of output slots N, chosen comfortably larger than the most objects I'd ever expect in one image, and let the model fill the rest with "nothing here." So I'll predict exactly N tuples (class, box), and I'll add a special class symbol — call it ∅, "no object" — that a slot emits when it isn't claiming anything. Now the output is always N predictions; an image with three objects should come out as three real detections and N−3 slots saying ∅. This ∅ is doing the same job the "background" class does in ordinary detectors, just at the level of a whole output slot instead of a reference box.

Now the hard part, and it's the part where the duplicates problem lives: how do I score these N predictions against the ground-truth objects to get a training loss? The ground truth is a set of, say, m objects. My output is a set of N predictions. There's no canonical ordering on either side. I can't just say "prediction 1 should match ground-truth object 1" — which object is "object 1"? Any ordering I impose is arbitrary, and if I impose one and train against it, I'm punishing the model for permuting its own outputs, which it shouldn't care about. The loss has to be invariant to the order of the predictions.

Let me think about what invariance buys me, and what goes wrong without it. Suppose I just pad ground truth to N with ∅ and compute a per-slot loss against a fixed pairing. Then two different slots that both produce a good "dog" box get scored against whatever happened to be in their fixed pairing slot — one of them is "right," the other is "wrong" even though both found the dog. The model has no consistent signal about who should own the dog. Worse, nothing in this setup discourages two slots from producing the same dog box; if anything, the fixed pairing rewards whichever slot lines up. I'm back to making duplicates. So the per-slot fixed loss is exactly the wrong thing.

What I want instead: pair up predictions and ground-truth objects so that each ground-truth object is owned by *exactly one* prediction, and choose the pairing that makes the predictions look as good as possible. If I can find that one-to-one pairing and then only supervise each matched pair, what happens to a duplicate? Let me actually trace it on a tiny case rather than trust my intuition, because the precise mechanism matters.

Take one ground-truth object — a dog at box (0.2,0.2,0.5,0.5) in xyxy — and N=3 slots. Slot A predicts a near-perfect dog box (0.21,0.19,0.49,0.51) with p(dog)=0.9; slot B predicts another good dog box (0.18,0.22,0.52,0.48) with p(dog)=0.8 — a duplicate; slot C predicts junk far away with p(dog)=0.1. Pad the ground truth to N with two ∅'s, so the targets are [dog, ∅, ∅]. The match cost of a slot to the single real object is −p(dog) + L1 + (1−GIoU); the cost of matching any slot to a ∅ is a constant (take it 0). Building that 3×3 cost matrix (rows = slots, columns = [dog, ∅, ∅]) and running the Hungarian solve:

  rows A,B,C × cols [dog,∅,∅]:
    [[-0.7345,  0,  0],
     [-0.4812,  0,  0],
     [ 3.8000,  0,  0]]
  assignment: A→dog, B→∅, C→∅.

So A, the best dog, owns the dog; the duplicate B is parked on a ∅. Now the important subtlety, which the trace of the numbers forces me to be honest about. Is the duplicate ruled out because it's *expensive*? No — if I were allowed to put both A and B on the dog the total cost would be −0.7345 + −0.4812 = −1.2157, which is *lower* (better) than the one-to-one total of −0.7345. The two-dog configuration is cheaper. It is excluded not by cost but because the assignment is constrained to be a *permutation*: the dog appears as a single column, and `linear_sum_assignment` cannot send two rows to one column (I checked — the columns it returns are all distinct). So the one-to-one *constraint*, not any intrinsic penalty on duplicates, forces the runner-up onto a ∅. And once B is matched to a ∅, the training loss will push it to predict "no object." That is the actual mechanism: the matching can crown only one owner per object, and everyone else is told to go silent. The de-duplication that NMS used to do by hand falls out of the assignment being a permutation.

So I need: a one-to-one assignment between the N predictions and the (ground-truth, padded to N with ∅) set. Let σ be a permutation of the N indices, so prediction σ(i) is assigned to ground-truth element i. I want the σ that minimizes the total assignment cost,

  σ̂ = argmin_σ Σ_i L_match(y_i, ŷ_σ(i)),

where L_match measures how badly prediction ŷ_σ(i) serves as the detection of ground-truth y_i. This is a bipartite matching — ground-truth nodes on one side, prediction nodes on the other, an edge cost for every pairing, find the min-cost perfect matching. And there's an exact, polynomial algorithm for this: the Hungarian algorithm, cubic in N. I don't have to be clever or greedy; I can get the optimal assignment directly. (A greedy "assign each object to its currently-best prediction" can collide — two objects wanting the same prediction — and gives a worse total. The Hungarian solve avoids that.) Bipartite-matching set losses aren't new — recurrent set predictors used them on small data — so the device itself buys me nothing on its own; whatever lets this reach a strong benchmark is going to have to come from somewhere else, in how the N predictions are produced. I'll hold that thought.

Now I have to write down L_match. A ground-truth element is y_i = (c_i, b_i): a class label c_i (possibly ∅) and a box b_i = (center_x, center_y, width, height), normalized to [0,1] relative to the image. A prediction at slot σ(i) gives a probability distribution over classes — write p̂_σ(i)(c_i) for the probability it assigns to the true class c_i — and a box b̂_σ(i). A good match is one where the prediction puts high probability on the right class *and* its box is close to the true box. So the matching cost should go down when p̂_σ(i)(c_i) is high and when the box is close:

  L_match(y_i, ŷ_σ(i)) = −1[c_i ≠ ∅] · p̂_σ(i)(c_i) + 1[c_i ≠ ∅] · L_box(b_i, b̂_σ(i)).

Let me check the pieces. The indicator 1[c_i ≠ ∅] zeros out both terms when the ground-truth element is one of the padding ∅'s — and that's right, because for a padding slot there's no box to fit and no "correct object" to claim; the cost of assigning any prediction to a ∅ shouldn't depend on the prediction at all. Notice the consequence: the cost of matching a prediction to ∅ is a constant, so it doesn't influence which real object gets which prediction. The matching is really only deciding, among the predictions, which m of them get the m real objects; the rest are dumped onto ∅ at constant cost. Good — that's the behavior I want.

One subtlety I almost glossed. For the class term in the matching cost I wrote the probability p̂(c_i), not the negative log-probability. Why not the same negative log-likelihood I will train with? Because here the class term sits *next to* a box term in a sum, and the two have to be commensurate for the matching to balance them sensibly. The box term L_box is an O(1) quantity (an L1 distance on [0,1] coordinates plus an overlap term in [0,2]). A probability is also O(1), bounded in [0,1]. A negative log-probability is unbounded above — a confidently-wrong slot contributes a huge −log p that swamps the box term and distorts which slot gets matched. So in the *matching cost* I use the probability directly; it keeps the class and box contributions on the same scale. I'll keep this distinct from the loss I actually backprop, which is next.

Once σ̂ is fixed by the Hungarian solve, the training loss — the thing I actually take gradients of — is computed over the matched pairs. Here I do want a proper classification loss, so I switch to negative log-likelihood for the class, and I keep the box loss on the matched real objects:

  L_Hungarian(y, ŷ) = Σ_{i=1}^{N} [ −log p̂_σ̂(i)(c_i) + 1[c_i ≠ ∅] · L_box(b_i, b̂_σ̂(i)) ].

Two things to notice about why matching-cost and loss differ. In the loss I use −log p̂ (real NLL — that gives the right gradient pressure on the classifier, strong when the model is confidently wrong, which is exactly when I want a big gradient). In the cost I used p̂ (bounded, commensurate with the box term, so the *assignment* is well-balanced). They're solving different jobs — one picks the pairing, the other trains the weights — so it's fine, even good, for them to differ. And the class term in the loss runs over *all* N slots including the ones matched to ∅: a slot matched to ∅ gets −log p̂(∅), i.e. it's trained to say "nothing here." That's the supervision that turns extra slots off and, together with the one-to-one matching, kills duplicates.

Now a problem I can see coming. N is much larger than m. Most slots get matched to ∅, so most of the class loss is "predict ∅." If I weight every slot's class term equally, the gradient is dominated by the ∅ class and the model can collapse to predicting ∅ everywhere — it would be right most of the time and the few real objects would barely register. This is the same class imbalance two-stage detectors face between background and foreground proposals, and they fix it by subsampling negatives to a fixed ratio. I don't have proposals to subsample, but I can do the analogous thing softly: down-weight the log-probability term when c_i = ∅. A factor of about 10 — divide the ∅ slots' classification loss by ten — rebalances it so the rare real objects aren't drowned out. Same goal as Faster R-CNN's positive/negative balancing, achieved by reweighting instead of subsampling.

Let me now pin down L_box, because I waved at it twice. The obvious choice is L1 on the four box coordinates, ||b_i − b̂_σ(i)||_1. Simple, and since I'm predicting boxes in absolute normalized coordinates (more on that in a second), L1 directly penalizes coordinate error. But L1 has a scale problem: its magnitude grows with box size. A 10% error on a huge box produces a much larger L1 than a 10% error on a tiny box, even though *relatively* they're equally wrong. So L1 alone would make the loss obsess over large boxes and neglect small ones. What I want alongside it is something scale-invariant — something that measures overlap quality as a ratio, independent of absolute size.

The natural scale-invariant box quantity is Intersection over Union: |A ∩ B| / |A ∪ B|, a pure ratio. But IoU has a fatal gradient property for a regression loss: when the predicted box and the target don't overlap at all, IoU is zero *everywhere* in that regime — the gradient is zero, and the loss gives no information about which direction would bring the boxes together. Early in training, when predicted boxes are often nowhere near their targets, that's exactly when I need a signal and IoU gives me none.

The fix is to add a term that keeps growing even when the boxes are disjoint, pulling them toward each other. Take C, the smallest axis-aligned box that encloses both A and B. When A and B are far apart, C is large and mostly empty; as they approach, C shrinks toward A ∪ B. So |C \ (A ∪ B)| / |C| — the fraction of the enclosing box that is wasted, neither A nor B — is a piecewise-differentiable, informative quantity even with no overlap. Subtract it from IoU:

  GIoU(A, B) = IoU(A, B) − |C \ (A ∪ B)| / |C|.

I shouldn't just trust the algebra; let me put real boxes through it and watch the numbers, because the whole reason for this term is the disjoint-box behavior and I want to see that it actually does what I claim. Take A = the unit square (0,0,1,1) and slide a second unit square B horizontally away from it, computing IoU and GIoU at each step:

  dx=0 (B=A):   IoU = 1.0000,  GIoU =  1.0000
  dx=1:         IoU = 0.0000,  GIoU =  0.0000
  dx=2:         IoU = 0.0000,  GIoU = -0.3333
  dx=3:         IoU = 0.0000,  GIoU = -0.5000
  dx=5:         IoU = 0.0000,  GIoU = -0.6667
  dx far:       IoU = 0.0000,  GIoU → -0.98...

This is exactly the property I was after and couldn't get from IoU. From dx=1 onward the boxes don't overlap, so IoU is pinned at 0 and gives no signal — but GIoU keeps decreasing as the boxes separate, so its gradient with respect to the box coordinates is nonzero and points back toward overlap. The flat-zero region that made IoU useless as a loss is gone. At dx=0 (coincident) GIoU = 1, the maximum, as it should be.

Now the two cases I want to be sure *don't* misbehave. Containment: a small box (1,1,2,2) inside a big one (0,0,4,4) gives IoU = 0.0625 and GIoU = 0.0625 — equal, because the enclosing box C *is* the big box and so is the union, leaving no wasted region. Good: GIoU doesn't invent an extra reward when one box contains the other. Partial overlap with neither containing the other: A=(0,0,2,2), B=(1,1,3,3) gives IoU = 0.1429 but GIoU = −0.0794. Here C is the 3×3 enclosing box, larger than the union, and the wasted-fraction penalty drags GIoU *below* IoU. So GIoU ≤ IoU in general, with equality exactly when C equals the union (coincidence and containment), and it stays informative — with gradient — when the boxes are disjoint. The range works out to [−1, 1], and since it's a ratio of areas throughout, it's scale-invariant. As a loss I want to *minimize*, I use 1 − GIoU, which lives in [0, 2] and is zero exactly when the boxes coincide.

So the box loss is a linear combination of the two, getting L1's direct coordinate fidelity and GIoU's scale-invariance:

  L_box(b_i, b̂) = λ_iou · (1 − GIoU(b_i, b̂)) + λ_L1 · ||b_i − b̂||_1,

normalized by the number of objects in the batch so the scale doesn't drift with how crowded the images are. The raw terms live on different numerical scales, so I need explicit weights rather than assuming the units will line up: something like λ_L1 = 5 and λ_iou = 2 keeps coordinate accuracy from being washed out while still giving the scale-invariant overlap term real force. In the matcher I can drop the constant part of 1 − GIoU and use −GIoU instead; that changes every possible assignment by the same amount per real target and leaves the argmin unchanged. The assignment-relevant box terms are otherwise the same as the ones the Hungarian loss will reward.

One more box decision I slipped past: I'm predicting boxes *directly* — absolute center, width, height in [0,1] — not as a delta off some reference box. That's deliberate and it's actually forced: the whole premise is that there are no anchors. The classic detectors regress an offset (t_x, t_y, t_w, t_h) relative to an anchor precisely because the anchor gives a starting point near the answer; predicting a delta off a reference *is* the anchor mechanism. If I predicted deltas I'd have to define references to take deltas from, and I'd be right back where I started. So I predict the box outright, squashing the four numbers through a sigmoid to keep them in [0,1]. The scale-balancing problem this creates (L1 on absolute coordinates) is exactly why I needed GIoU; the two choices are linked.

Good — I have a loss that, by construction, wants one prediction per object and needs no NMS. But a loss is only half of it. The loss can *want* unique predictions, but the architecture has to be *able* to produce them. What kind of network turns image features into N predictions that don't collude on the same object?

If the N output slots are computed independently from the image — say N parallel heads each looking at the features on its own — there's nothing stopping two of them from latching onto the same salient dog. They can't coordinate. The matching loss would punish the duplicate, but each head, in isolation, has no way to *know* another head already took the dog; it only sees the image. To avoid duplicates the slots have to be able to see *each other* and divide up the objects. I need a mechanism for global, all-pairs interaction among the predictions, and between the predictions and the whole image.

That description — "update every element by aggregating information from every other element, with learned weights" — is exactly self-attention. A self-attention layer takes a set of vectors, and for each one computes a weighted sum over all the others, the weights being a softmax over dot-product similarities. More concretely, for query sequence X_q and key-value sequence X_kv, I project to Q, K, V, add positional information to Q and K, and compute each head as softmax(QK^T / sqrt(d_k))V. The division by sqrt(d_k) is not cosmetic: if the query and key coordinates have roughly unit variance, their dot product has variance d_k, and without the scaling the softmax saturates as the head dimension grows. M heads with d_k = d / M give different relation subspaces and concatenate back to width d.

Self-attention models all pairwise interactions in one shot, and it's permutation-equivariant unless I inject position — permuting the inputs permutes the outputs the same way. That is perfect on the output side, because my predictions *are* a set, and dangerous on the image side, because pixels are not. The Transformer is built entirely out of these pieces: an encoder that self-attends over the image tokens, and a decoder that self-attends over its own outputs and cross-attends into the encoder's memory. If I let my N prediction slots be the decoder's outputs, then decoder self-attention is precisely the channel through which slot A can learn "slot B is already covering the dog, I'll take the cat," and cross-attention is how each slot pulls the image evidence it needs. So the two halves line up: the matching loss forbids duplicates, and decoder self-attention is the mechanism that gives the slots a way to actually obey it by seeing each other and dividing the objects. A loss that punishes collusion is toothless if the slots can't communicate; this architecture is what removes that excuse.

Let me build it concretely. Start with a standard CNN backbone — a ResNet — to turn the H₀×W₀ image into a compact feature map f of shape C×H×W, with C around 2048 and H, W about 1/32 of the input. A 1×1 convolution drops the channels from C to a smaller working width d (say 256), giving a d×H×W map. The encoder wants a sequence, so I flatten the spatial dimensions: d×HW, i.e. HW tokens of width d, one per spatial location. Run the Transformer encoder over these tokens. The cost is acceptable at this reduced resolution: a self-attention layer over HW image tokens spends O(d^2 HW) on the projections and O(d(HW)^2) on attention weights and value aggregation. In the decoder the N output slots are much fewer than HW, so slot self-attention is O(d^2 N + dN^2), and cross-attention into the image memory is O(d^2(N + HW) + dNHW). The expensive quadratic term stays in the encoder, where the CNN has already shrunk the spatial grid.

But attention without positional input is permutation-equivariant, and the image is emphatically not — token (3,7) is in a specific place. So I have to inject position. In the 1D Transformer this is done with sinusoids of geometrically-spaced frequencies; I generalize to 2D by building a sinusoidal encoding for the row coordinate and one for the column coordinate and concatenating them into a d-dimensional spatial code, half the channels for each axis. With padding masks in a batch, I should count only valid pixels, normalize each coordinate to a fixed range such as [0, 2π], and then use the usual temperature schedule: for channel index k, divide by 10000^{2 floor(k/2) / (d/2)}, take sin on even channels and cos on odd channels, once for x and once for y, then concatenate. I add these positional encodings to the queries and keys at every attention layer rather than only once at the input — attention only ever uses position through the query-key comparison, and feeding it in at each layer keeps the spatial signal from washing out as the stack deepens.

Now the decoder, and the piece that makes this whole thing tick. The decoder is also permutation-equivariant — if I feed it N identical input vectors, every slot does identical computation and produces identical outputs, which is useless. The N inputs have to *differ* to produce N different predictions. So I make them N learned vectors, one per slot, learned as parameters of the model. Each is, in effect, a learned "position" in output space — a standing question the slot asks of the image. I'll call them object queries. The decoder takes zero content vectors plus these query positions, lets the slots self-attend (so they coordinate and split up the objects), and cross-attends them into the encoder memory (so each gathers the image evidence for its object), producing N output embeddings. Because the first decoder layer begins with zero content, its first self-attention has almost nothing useful to mix and can be skipped as an optimization; after cross-attention has written image evidence into the slots, self-attention becomes the communication channel that matters. Because the slots are decoded all at once, not one after another, this is parallel decoding — which is the right call for a set, where there's no natural order to decode in anyway, and it's far faster than the autoregressive, one-box-at-a-time RNN decoders that earlier set detectors used and that never scaled to a benchmark like this. The queries being distinct learned vectors is what breaks the symmetry; the self-attention among them is what prevents collusion; doing it in parallel is what's both principled (sets are orderless) and fast.

Each of the N decoder output embeddings then goes through a small shared head, applied independently to every slot, to produce that slot's prediction: a 3-layer MLP with ReLU and hidden width d outputs the four box numbers (through a sigmoid, since they're normalized coordinates), and a single linear layer plus softmax outputs the class distribution over the real classes *plus* ∅. The head is shared across slots — every slot decodes the same way; what makes them differ is the embedding the decoder produced, which already encodes which object that slot grabbed. The class output has the extra ∅ entry so a slot that found nothing can say so, which is what the Hungarian loss trains the unmatched slots to do.

That's a complete system: backbone → project → flatten → encoder (with spatial position) → decoder (N object queries, parallel) → per-slot shared head → N predictions; trained by Hungarian-matching the N predictions to the padded ground truth and applying the NLL-plus-(L1+GIoU) loss on the matched pairs. No anchors (boxes are absolute), no assignment heuristic (the Hungarian solve *is* the assignment, and it's one-to-one and learned-into), no NMS (the loss forbids duplicates and self-attention lets the model obey).

Let me poke at what might not train well, because a clean idea that won't optimize is no good. The decoder is six layers deep, and I'm only supervising the very last layer's outputs. Deep stacks supervised only at the end can be slow and hard to train, and there's a specific failure I'd worry about here: the model getting the *number* of objects wrong, since each layer is silently refining a guess and only the top gets feedback. The standard remedy for hard-to-train deep stacks is deep supervision — attach a loss to intermediate layers too. So I add a prediction head and a full Hungarian loss after *every* decoder layer, not just the last. The heads share parameters across layers (one head, applied at every depth), with a shared layer-norm to put the differently-scaled intermediate activations on a common footing before the shared head sees them. These auxiliary decoding losses give every layer a direct training signal and, in particular, help the model converge to outputting the right count of objects per class. At inference I just read off the last layer; the auxiliary heads are a training scaffold.

And the optimization details, which for transformers are not optional. Train with AdamW — plain SGD trains transformers poorly — with weight decay around 1e-4, and clip the gradient norm (to about 0.1) because attention models throw occasional gradient spikes that destabilize training. Initialize the transformer with Xavier; the ResNet backbone comes from ImageNet pretraining. Two learning rates: the transformer is learned from scratch and needs a normal rate (1e-4), but the backbone is already trained and only needs gentle fine-tuning — give it roughly an order of magnitude less (1e-5), which matters specifically for stability in the first few epochs, when a large backbone rate would wreck the pretrained features before the transformer has learned to use them. Freeze the backbone's batch-norm statistics, the standard detection practice, since detection batches are small and BN estimates would be noisy. And expect to train *long* — bipartite matching plus a from-scratch transformer converges slowly, far slower than a standard detector schedule.

Let me put it in code, grounded in how this actually gets built. First the geometry helpers and GIoU.

```python
import torch
from torch import nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torchvision.ops.boxes import box_area


def box_cxcywh_to_xyxy(b):
    cx, cy, w, h = b.unbind(-1)
    return torch.stack([cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h], dim=-1)


def box_iou(boxes1, boxes2):
    area1, area2 = box_area(boxes1), box_area(boxes2)
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area1[:, None] + area2 - inter
    return inter / union, union


def generalized_box_iou(boxes1, boxes2):
    # GIoU = IoU - |C \ (A u B)| / |C|, with C the smallest enclosing box.
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)
    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])   # top-left of enclosing box
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])   # bottom-right of enclosing box
    wh = (rb - lt).clamp(min=0)
    area = wh[:, :, 0] * wh[:, :, 1]                      # |C|
    return iou - (area - union) / area                   # subtract the wasted fraction
```

Now the model. The N object queries are learned embeddings; the decoder is fed zeros plus those queries as positional input, and the encoder gets the flattened features plus a 2D positional encoding. I'll lean on the library Transformer and keep the wiring explicit, returning the stack of all decoder layers so I can apply the auxiliary losses.

```python
class DETR(nn.Module):
    def __init__(self, backbone, transformer, num_classes, num_queries, aux_loss=False):
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        # class head: a real class for each id, plus one extra slot for "no object"
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        # box head: 3-layer MLP -> 4 numbers (cx, cy, w, h), later squashed by sigmoid
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        # the N object queries: distinct learned vectors that break decoder symmetry
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        # 1x1 conv: drop backbone channels (C=2048) to the transformer width d
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.backbone = backbone
        self.aux_loss = aux_loss

    def forward(self, samples):
        features, pos = self.backbone(samples)        # CNN map + 2D sinusoidal position
        src, mask = features[-1].decompose()
        assert mask is not None
        # encoder over flattened HW tokens (+pos), decoder over N object queries (parallel);
        # hs holds every decoder layer's output -> [n_layers, batch, N, d]
        hs = self.transformer(self.input_proj(src), mask, self.query_embed.weight, pos[-1])[0]
        outputs_class = self.class_embed(hs)          # per slot, distribution over classes + no-object
        outputs_coord = self.bbox_embed(hs).sigmoid() # per slot, box in [0,1]
        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # deep supervision: expose each intermediate layer's predictions for its own loss
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k)
                                    for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x
```

For the matcher I build the cost matrix between every prediction and every ground-truth box, then Hungarian-solve it per image. The class cost is −p[true class] (probability, not log — kept commensurate with the box terms), and the box cost is L1 plus −GIoU.

```python
class HungarianMatcher(nn.Module):
    def __init__(self, cost_class=1, cost_bbox=5, cost_giou=2):
        super().__init__()
        self.cost_class, self.cost_bbox, self.cost_giou = cost_class, cost_bbox, cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):
        bs, num_queries = outputs["pred_logits"].shape[:2]
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [bs*N, num_classes+1]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)               # [bs*N, 4]
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])
        # class cost: -prob of the true class (the +1 constant is dropped, irrelevant to argmin)
        cost_class = -out_prob[:, tgt_ids]
        # box costs: L1 between coords, and -GIoU between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox),
                                         box_cxcywh_to_xyxy(tgt_bbox))
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = C.view(bs, num_queries, -1).cpu()
        sizes = [len(v["boxes"]) for v in targets]
        # one Hungarian solve per image over its own block of the cost matrix
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
                for i, j in indices]
```

For the criterion I match, then apply NLL on classes for *all* slots (matched slots get their true class, the rest get ∅, with ∅ down-weighted), and L1+GIoU on the matched boxes. The box terms are divided by the number of target objects, averaged across workers when training is distributed, so the loss scale does not change with the batch split. I keep cardinality error as a logging signal, rematch each auxiliary decoder layer during training, and at inference only convert normalized boxes and class logits back to image-scale detections; there is still no overlap-based filtering.

```python
class SetCriterion(nn.Module):
    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        # down-weight the no-object class to counter the N >> #objects imbalance
        empty_weight = torch.ones(num_classes + 1)
        empty_weight[-1] = eos_coef   # ~ 0.1, i.e. divide the no-object class loss by ~10
        self.register_buffer('empty_weight', empty_weight)

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        src_logits = outputs['pred_logits']
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        # default every slot to "no object"; matched slots get their true class
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_ce': loss_ce}
        if log and target_classes_o.numel() > 0:
            pred = src_logits[idx].argmax(-1)
            losses['class_error'] = 100 - 100 * (pred == target_classes_o).float().mean()
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        return {'cardinality_error': F.l1_loss(card_pred.float(), tgt_lengths.float())}

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        loss_giou = 1 - torch.diag(generalized_box_iou(box_cxcywh_to_xyxy(src_boxes),
                                                       box_cxcywh_to_xyxy(target_boxes)))
        return {'loss_bbox': loss_bbox.sum() / num_boxes,    # normalize by #objects in batch
                'loss_giou': loss_giou.sum() / num_boxes}

    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'boxes': self.loss_boxes,
            'cardinality': self.loss_cardinality,
        }
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def _normalized_num_boxes(self, outputs, targets):
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float,
                                    device=next(iter(outputs.values())).device)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(num_boxes)
            world_size = torch.distributed.get_world_size()
        else:
            world_size = 1
        return torch.clamp(num_boxes / world_size, min=1).item()

    def forward(self, outputs, targets):
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}
        indices = self.matcher(outputs_without_aux, targets)   # match the final layer
        num_boxes = self._normalized_num_boxes(outputs_without_aux, targets)
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))
        # deep supervision: rematch and reapply the loss at every intermediate decoder layer
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    kwargs = {'log': False} if loss == 'labels' else {}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    losses.update({k + f'_{i}': v for k, v in l_dict.items()})
        return losses


class PostProcess(nn.Module):
    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']
        prob = F.softmax(out_logits, -1)
        scores, labels = prob[..., :-1].max(-1)
        boxes = box_cxcywh_to_xyxy(out_bbox)
        img_h, img_w = target_sizes.unbind(1)
        scale = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale[:, None, :]
        return [{'scores': s, 'labels': l, 'boxes': b}
                for s, l, b in zip(scores, labels, boxes)]
```

Stepping back to make sure the chain holds together. The task is set prediction; the reason detectors weren't end-to-end was the trio of anchors, the overlap-assignment heuristic, and NMS, all of which trace back to a single root cause — training admits many predictions per object. Cut that root by demanding a one-to-one assignment between a fixed set of N slots and the ground truth, found by an exact bipartite (Hungarian) matching; now duplicates are penalized by the objective itself and NMS has no work left. Score each matched pair with NLL on the class and, for the box, L1 plus 1−GIoU — L1 for coordinate fidelity, GIoU because plain IoU gives no gradient for disjoint boxes and L1 alone isn't scale-invariant — and down-weight the no-object class to survive the slot imbalance. Predict boxes absolutely because dropping anchors leaves nothing to take deltas from. To let the slots actually realize a unique-prediction solution, produce them with a Transformer: encoder self-attention for global image context (with 2D positional encoding so image tokens retain location), a decoder over N distinct learned object queries whose self-attention lets them divide up the objects and whose cross-attention gathers image evidence, decoded in parallel because a set has no order. Add auxiliary losses at every decoder layer so the deep stack trains and gets the object count right. The loss wants unique detections; the attention architecture can deliver them; together they make detection a direct, end-to-end set prediction.
