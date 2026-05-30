OK, let me think this through from scratch. I want representations from unlabeled images, and the structure everyone is using is the Siamese net: one image, two augmented views, push the two feature vectors to agree. The trouble with "agree" is that the laziest way to agree is for everything to map to the same constant vector — perfect agreement, zero information. Collapse. So the entire game in label-free Siamese learning is: keep the "two views agree" signal, and bolt on *something* that forbids the constant solution. Every method I know is some particular choice of that something, and each choice drags in a pile of machinery.

Let me lay the choices side by side and ask what each one's anti-collapse part actually costs. Contrastive learning — SimCLR — keeps direct weight-sharing between the two branches and prevents collapse with *negatives*: besides pulling the two views of one image together, it pushes views of *different* images apart. The repulsion is what kills the constant solution, because if everything collapsed to one point the negative term would scream. But the catch is the representation quality scales with the *number* of negatives, so SimCLR needs a huge batch — thousands of images at once — just to have enough other-image features in the denominator, plus a large-batch optimizer to make that batch trainable. MoCo keeps the negatives but gets them more cheaply: a queue of ~65k stored features and a *momentum encoder* — a slowly moving average of the trained net — to keep that queue consistent. So MoCo's price is a whole second encoder and a long queue. Clustering methods like SwAV drop explicit negatives but cluster the batch online: compute a cluster assignment from one view, predict it from the other, and force the assignment to spread evenly across clusters with a Sinkhorn balance step so it can't dump everything into one cluster. That balance constraint is the anti-collapse part, and it costs an online clustering / Sinkhorn machine plus, again, large batches to have enough samples per assignment.

Then there's BYOL, which is the strange one. BYOL throws out negatives *entirely* and still doesn't collapse. It has a prediction MLP — a small head `h` on one branch that maps that view's embedding toward the other view's embedding — and the other branch is a momentum encoder. The loss is just the negative cosine similarity of the two normalized vectors (equivalently the MSE of ℓ₂-normalized vectors, up to a factor of 2), symmetrized over the two views. No repulsion anywhere. The standard story is that the *momentum encoder* is the thing preventing collapse — and indeed if you rip the momentum encoder out, BYOL is reported to fall apart, like 0.3% accuracy, total collapse.

That story bugs me. Stare at it for a second. The momentum encoder isn't *just* a moving average — it's a moving average, which means its parameters are *not* updated by gradients from the loss. The target branch never receives backprop. So whenever you use a momentum encoder you are *also*, unavoidably, applying a stop-gradient on that branch: the gradient of the loss does not flow back through the target. Those two properties are welded together. When people say "the momentum encoder prevents collapse," they have two suspects handcuffed to each other — the moving-average *and* the stop-gradient — and they've credited the moving-average. But maybe it's the stop-gradient. The way to find out is to separate them, and the way to separate them is brutal: kill the momentum encoder, share the weights directly between the two branches like SimCLR does, keep the predictor `h`, keep the symmetrized cosine loss — and then I can toggle the stop-gradient on and off independently and watch.

So let me build the most stripped-down Siamese net I can and see what's load-bearing. One image `x`, two augmented views `x1`, `x2`. One encoder `f` — backbone plus a projection MLP — with weights *shared* between the two views; no momentum, no second encoder. A prediction MLP `h` on one branch. Define `p1 = h(f(x1))` and `z2 = f(x2)`, and minimize their negative cosine similarity. Let me write `D` cleanly. I ℓ₂-normalize both and take the negative dot product:

    D(p, z) = - (p/‖p‖₂) · (z/‖z‖₂).

This is bounded in `[-1, 1]`, minimized at `-1` when `p` and `z` point the same way. And because both are unit vectors, `‖p̂ - ẑ‖² = 2 - 2 p̂·ẑ`, so `D` is just the MSE of the normalized vectors shifted and scaled by 2 — the cosine and the squared-error views are the same thing, which I'll want later. To use both views symmetrically I average the two directions:

    L = ½ D(p1, z2) + ½ D(p2, z1),

so each image gets a predictor applied to each of its two views, each matched against the *other* view's projection. Minimum possible value `-1`.

Now: what is the stop-gradient, and where does it go? When I compute `D(p1, z2)`, the `z2` on the right is acting as a *target* for the prediction `p1`. The idea is to treat that target as fixed — `z2` is a constant, gradients do not flow back into the encoder through it:

    D(p1, stopgrad(z2)).

And symmetrically the full loss becomes

    L = ½ D(p1, stopgrad(z2)) + ½ D(p2, stopgrad(z1)).

So in the first term the encoder applied to `x2` gets no gradient from `z2` (it's frozen as a target), but in the *second* term the same encoder applied to `x2` does get gradient through `p2`. Each branch is alternately "the thing being predicted" (frozen) and "the predictor" (trained). Now I can run the experiment that BYOL couldn't cleanly run, because I have no momentum encoder confounding things: train this exact architecture, all hyperparameters identical, and the *only* difference is whether `stopgrad` is there.

With the stop-gradient: the loss decreases steadily, the kNN monitor climbs, and the per-channel std of the ℓ₂-normalized output sits near `1/√d`. That `1/√d` number is the tell — if `z` were roughly a zero-mean isotropic Gaussian, normalizing it would put the std of each channel at about `1/√d` (each coordinate is `z_i` divided by a norm of order `√d`). So the outputs are scattered over the unit hypersphere, not piled up. It learns; final linear-eval accuracy lands around 67.7%, stable across trials.

Without the stop-gradient: the optimizer *immediately* finds the degenerate solution. The loss drops straight to its minimum, `-1`. The std of the normalized output goes to zero — every sample maps to the same unit vector. kNN accuracy is chance. Linear eval is 0.1%, i.e. random guessing. So collapse *does* exist for this loss and architecture — there was never any guarantee it wouldn't — and removing the single stop-gradient operation is enough to fall into it. Conversely, putting that one operation back is enough to escape it. No negatives, no momentum encoder, no Sinkhorn, no special batch size. Just `stopgrad`.

I want to be careful not to over-credit `stopgrad` by under-testing the architectural pieces. Maybe one of the heads, or BN, is secretly the anti-collapse part and stop-gradient is incidental. So let me ablate each piece with stop-gradient *kept on*, and separately check whether any piece alone prevents collapse without stop-gradient.

Take the predictor `h` first. Remove it — make `h` the identity. With stop-gradient kept, the model collapses (0.1%). I can actually see *why* this one is expected for the symmetric loss. With `h` gone, the loss is `½ D(z1, stopgrad(z2)) + ½ D(z2, stopgrad(z1))`. Look at the gradient. The first term contributes a gradient only through `z1` (its `z2` is frozen); the second only through `z2`. Add them and you get, up to the `½`, exactly the gradient you'd get from `D(z1, z2)` with *no* stop-gradient at all — the stop-gradient just splits the two-sided derivative into two one-sided pieces that sum back to the full thing, scaled by `1/2`. So in the no-predictor symmetric case, "with stop-gradient" is *identical* to "without stop-gradient, loss halved." Stop-gradient has been rendered vacuous, and we're back to the naked agreement loss whose minimizer is the constant. Collapse, as observed. So `h` is not optional — without it the stop-gradient stops doing anything.

But is `h` itself the anti-collapse mechanism? Two more probes say no. Freeze `h` at its random initialization and never train it: the model gets 1.5% — but watching the loss, this is *not* collapse. The loss stays *high* and never converges; the std doesn't crater. It's just that a random fixed `h` can't track the representation, so optimization stalls. So `h` has to be *trained*, but its role isn't to prevent collapse — when it fails it fails by not converging, not by collapsing. And a nice corollary: give `h` a *constant* learning rate (don't decay it) and it does slightly *better*, ~68.1%. That makes sense once I think about what `h` is chasing — it's trying to match the *current* representation, which is itself moving the whole time, so forcing `h` to converge early (by decaying its lr) is counterproductive; let it keep adapting to the latest features. I'll keep `h` on a non-decayed lr.

BN next. Strip all BN out of the MLP heads: accuracy drops to 34.6% — low, but, watching the std, *not collapsed*. It's an optimization difficulty, not a collapse. Add BN back to the hidden layers and you recover 67.4%; add it to the projection output too and you get 68.1% (and the learnable affine in that output BN isn't even necessary). Push it too far — BN on the predictor's *output* — and the training goes unstable, loss oscillates; again not collapse, just instability. And the decisive point: the stop-gradient comparison earlier used the *same* BN configuration in both arms, yet the one without stop-gradient collapsed. So BN helps *optimization*, exactly like it does in supervised nets, but there's no evidence it touches collapse.

Similarity function: swap the cosine `D` for a cross-entropy similarity (softmax the target, dot with log-softmax of the prediction). Still no collapse — 63.2% vs the cosine's 68.1%. So collapse prevention isn't a property of the cosine form either.

Symmetrization: drop it, use only the asymmetric `D(p1, stopgrad(z2))`. Still no collapse — 64.8%, and ~67.3% if I roughly compensate by sampling two augmentation pairs per image instead of one. So symmetrization buys accuracy (it's one more prediction per image, denser sampling) but it is *not* what prevents collapse.

Every dial I turn — predictor, BN, similarity, symmetrization, batch size — moves accuracy around, and *none* of them is the difference between collapsing and not. The one operation that flips collapse on and off, with everything else fixed, is the stop-gradient. Negatives: not needed. Momentum encoder: not needed. Large batches: not needed — SGD at batch 256 (and anywhere from 64 to 2048) works fine; the only batch that hurts is 4096, and that's the familiar large-batch-SGD problem, not collapse, fixable with a specialized optimizer that I don't otherwise need.

Which then forces a question I can't dodge. If a single stop-gradient is the hinge between collapse and a good representation, then I am not really minimizing the loss I wrote down. Gradient descent on `L(θ)` would *not* put a stop-gradient in the middle of it; the stop-gradient changes which gradients exist. The fact that this particular non-gradient is *necessary* tells me the thing being solved is not "minimize this similarity loss by SGD." There is some *other* optimization problem for which stop-gradient is the natural, correct move, and SimSiam is an implementation of *that*. Let me try to reverse-engineer it.

Where do you naturally see "freeze this quantity, optimize that one, then swap"? That is alternating optimization — fix one block of variables, solve for the other, alternate. k-means is the textbook case: alternate between the cluster centers and the per-point assignments. EM is the probabilistic cousin: an E-step computing expected assignments and an M-step updating parameters. In all of them, when you optimize one block the *other* block is a constant — and a constant contributes no gradient. That is precisely a stop-gradient. So let me hypothesize that there's a second set of variables hiding in this problem, and that SimSiam is alternating between them and the network weights.

Posit a loss with two arguments — the network parameters `θ`, and a *new* set of variables `η`:

    L(θ, η) = E_{x, T} [ ‖ F_θ(T(x)) − η_x ‖² ].

`F_θ` is the network, `T` an augmentation drawn from the augmentation distribution, `x` an image, and the expectation runs over images and augmentations. I'll use squared error here for clarity — by the normalized-MSE-equals-cosine identity from before, this is the cosine loss up to normalization. The key object is `η`. It is *one vector per image*, indexed by the image, so its size grows with the dataset. Crucially `η` is *not* the output of any network — it's a free variable, an argument of the optimization. Intuitively `η_x` is "the representation of image `x`," but untethered: we get to choose it to minimize the loss.

This is structurally k-means. There, `θ` would be the cluster centers — here it's the encoder weights. There, the assignment of each point (a one-hot pick of a center) is the per-point free variable — here `η_x` is that variable, a representation of `x` instead of a one-hot code. And just like k-means, I solve `min_{θ, η} L(θ, η)` by alternating:

    θ^t  ← argmin_θ  L(θ, η^{t−1})
    η^t  ← argmin_η  L(θ^t, η).

Now watch the stop-gradient *fall out*. In the first subproblem I'm optimizing `θ` with `η^{t−1}` held fixed. `η^{t−1}` is a constant here, so when I take an SGD step on `θ`, no gradient flows to `η`. The "target" `η` is frozen by definition of the subproblem — that *is* the stop-gradient. It isn't a trick I inserted to dodge collapse; it's the inevitable consequence of solving for `θ` while `η` is the other, currently-fixed block.

The second subproblem: with `θ^t` fixed, minimize over `η`. It separates per image, since `η_x` only appears in image `x`'s terms. For each `x`:

    min_{η_x} E_T [ ‖ F_{θ^t}(T(x)) − η_x ‖² ].

This is just "find the point closest in expected squared distance to `F_{θ^t}(T(x))` over random augmentations," and the minimizer of expected squared distance is the mean:

    η^t_x ← E_T [ F_{θ^t}(T(x)) ].

So the optimal `η_x` is the *average* embedding of image `x` over the augmentation distribution. (With cosine instead of MSE, I'd ℓ₂-normalize, same idea.) This is the E-step analogue: given the current network, compute each image's target as its expected representation across augmentations.

Now collapse it to one step to see SimSiam emerge. Computing `E_T[·]` exactly is hopeless, so approximate the expectation by sampling the augmentation just *once* — call that draw `T'`:

    η^t_x ← F_{θ^t}(T'(x)).

Substitute into the `θ`-subproblem:

    θ^{t+1} ← argmin_θ  E_{x, T} [ ‖ F_θ(T(x)) − F_{θ^t}(T'(x)) ‖² ].

Stare at that. `θ^t` on the right is a constant (stop-gradient). `T` and `T'` are two independent augmentation draws — two views of the same image. So the term is "embed view `T(x)` with the trainable net, embed view `T'(x)` with a frozen copy, pull them together" — that is *exactly* a Siamese network with shared weights and a stop-gradient on one branch. And if I take just *one* SGD step on this subproblem rather than fully solving it before recomputing `η`, the whole alternating scheme degenerates into: each minibatch, embed two views, treat one as the frozen target, step the encoder toward it. That is the SimSiam update. The siamese shape, the shared weights, and the stop-gradient are all *consequences* of one-step alternating optimization on `L(θ, η)`.

That accounts for everything except the predictor `h`, which I established empirically is mandatory. Where does `h` belong in this picture? The slippage is in the one-sample approximation `η^t_x ← F_{θ^t}(T'(x))`. The *correct* target is the expectation `E_T[F(T(x))]`, but I replaced it with a single noisy sample. The predictor's job is to repair exactly that. By definition `h` is trained to minimize `E_z [ ‖ h(z1) − z2 ‖² ]`, and the optimal `h` satisfies

    h(z1) = E_z[z2] = E_T[ f(T(x)) ],

the expectation over augmentations — the very quantity the single-sample approximation dropped. I can't compute `E_T` explicitly, but a small network *can learn to predict* it: across many epochs the augmentations of each image are sampled again and again, and `h` can average over that implicit distribution. So `h` is filling the gap between "the target should be the expected embedding" and "I only ever sample one augmentation." That's why removing `h` is fatal in a way distinct from collapse-by-no-stopgrad — without `h` there's nothing approximating the expectation, and (in the symmetric case) the stop-gradient becomes algebraically vacuous as I showed. And it's why `h` wants a constant, non-decayed lr: it's chasing a moving target (the expectation under an ever-improving `θ`), so it should keep adapting rather than freeze early.

Symmetrization slots in too: the loss `L(θ, η)` is an expectation over images *and* augmentations, which SGD estimates by sampling a batch of images and a pair of augmentations `(T1, T2)`. Symmetrizing — also using `(T2, T1)` — is just denser sampling of that expectation. More samples, a better estimate, slightly better accuracy; but nothing in the alternating structure *requires* it. Exactly matching the observation that symmetrization helps accuracy but isn't tied to collapse.

Let me stress-test the alternating hypothesis with a couple of probes. If SimSiam really is one-step alternation, then *multi-step* alternation should also work: treat `t` as an outer loop, pre-compute the targets `η_x` for a whole block of `k` inner SGD steps using the one-sample assignment, cache them, take `k` steps, then re-assign. Try `k = 1, 10, 100`, and a full epoch. They all train fine; `10` and `100` steps even come out slightly *better* than 1-step (at the cost of pre-computing and caching the `η`'s), and a full epoch is a touch worse. SimSiam (`k=1`) is just the cheap special case of a valid family. Good — the alternating frame predicts this and it holds. Second probe, aimed straight at the predictor's role: instead of using `h` to approximate `E_T`, approximate it a different way — maintain a *moving average* of the per-image target, `η^t_x ← m·η^{t−1}_x + (1−m)·F_{θ^t}(T'(x))` with `m=0.8`, like a memory bank accumulating views across iterations. With that moving-average target I can *drop the predictor entirely* and still get ~55% — whereas dropping `h` *without* the moving average collapses to 0.1%. So an explicit approximation of `E_T` substitutes for `h`, which is exactly what the hypothesis says `h` is doing.

One honest limit. This whole construction explains *what* SimSiam is implicitly optimizing — alternating minimization of `L(θ, η)` over the encoder and the per-image targets — and it explains *why the stop-gradient is there* (it's the consequence of holding the other block fixed, not a hack). It does *not* by itself prove collapse can't happen; `L(θ, η)` still has the constant solution `η_x ≡ const`, `F ≡ const`, loss zero. What I can say is why the alternating optimizer is unlikely to *reach* it. The trajectory depends on initialization, and `η` is initialized from the output of a randomly initialized network — which is essentially never constant. And critically, the alternating scheme never takes a gradient with respect to `η` *jointly across all images at once* — each `η_x` is set independently from its own image's embedding. So there's no single gradient direction that would pull all the `η_x` toward a shared constant simultaneously; to collapse, the optimizer would have to herd every image's target to the same point through this decoupled update, starting from a scattered init. It doesn't; it finds the other trajectory, the one where the outputs stay scattered on the sphere (std near `1/√d`). The non-collapse stays an empirical fact, but now it's a *reasonable* one: scattered init plus per-image decoupled targets plus the alternation makes the constant hard to fall into, whereas plain joint SGD on the naked similarity loss — no stop-gradient, no second block — has nothing stopping it and dives straight in.

So the picture closes. Strip the Siamese net to nothing — one shared encoder, no momentum, no negatives, ordinary batches, a predictor head, symmetrized negative-cosine loss — and add a single stop-gradient on the target branch. That stop-gradient isn't cosmetic: it's the signature of solving an alternating optimization `min_{θ,η} E‖F_θ(T(x)) − η_x‖²` one step at a time, with `η` the per-image expected-representation targets and `h` standing in for the expectation over augmentations that one-sample assignment throws away. Now the code.

```python
import torch
import torch.nn as nn
import torchvision.models as models

class SimSiam(nn.Module):
    def __init__(self, base_encoder=models.resnet50, dim=2048, pred_dim=512):
        super().__init__()
        # encoder f = backbone + projection MLP. The backbone's final fc is the
        # first layer of a 3-layer projection MLP; the projection output is
        # z (dim-d). zero_init_residual is the standard ResNet trick.
        self.encoder = base_encoder(num_classes=dim, zero_init_residual=True)
        prev_dim = self.encoder.fc.weight.shape[1]
        self.encoder.fc = nn.Sequential(
            nn.Linear(prev_dim, prev_dim, bias=False),
            nn.BatchNorm1d(prev_dim), nn.ReLU(inplace=True),     # proj layer 1
            nn.Linear(prev_dim, prev_dim, bias=False),
            nn.BatchNorm1d(prev_dim), nn.ReLU(inplace=True),     # proj layer 2
            self.encoder.fc,
            nn.BatchNorm1d(dim, affine=False))                   # output BN, no affine
        self.encoder.fc[6].bias.requires_grad = False           # bias absorbed by BN

        # predictor h: a 2-layer bottleneck MLP (hidden = dim/4). Bottleneck makes
        # h "digest" the representation; BN on the hidden layer, none on the output,
        # no ReLU on the output (output must be able to point anywhere).
        self.predictor = nn.Sequential(
            nn.Linear(dim, pred_dim, bias=False),
            nn.BatchNorm1d(pred_dim), nn.ReLU(inplace=True),     # hidden
            nn.Linear(pred_dim, dim))                            # output

    def forward(self, x1, x2):
        z1 = self.encoder(x1)            # projection of view 1
        z2 = self.encoder(x2)            # projection of view 2
        p1 = self.predictor(z1)          # prediction of view 1
        p2 = self.predictor(z2)          # prediction of view 2
        # stop-gradient on the targets: z is the frozen "eta" block of the
        # alternating problem -> .detach(). p is the trainable branch.
        return p1, p2, z1.detach(), z2.detach()

# negative cosine similarity; D(p, z) = -(p/||p||)·(z/||z||)
criterion = nn.CosineSimilarity(dim=1)

def simsiam_loss(p1, p2, z1, z2):
    # z1, z2 already detached -> stop-gradient on the target branch.
    # symmetrized: each view predicts the other's (frozen) projection.
    return -(criterion(p1, z2).mean() + criterion(p2, z1).mean()) * 0.5

# --- optimization ---
# SGD (no LARS needed), lr scaled linearly with batch size, cosine decay on the
# encoder; the PREDICTOR keeps a CONSTANT lr (it must keep chasing the moving
# representation, so we don't decay it).
def build_optimizer(model, base_lr=0.05, batch_size=512,
                    momentum=0.9, weight_decay=1e-4):
    init_lr = base_lr * batch_size / 256
    param_groups = [
        {'params': model.encoder.parameters(),   'fix_lr': False},
        {'params': model.predictor.parameters(), 'fix_lr': True},   # constant lr
    ]
    return torch.optim.SGD(param_groups, init_lr,
                           momentum=momentum, weight_decay=weight_decay), init_lr

def adjust_learning_rate(optimizer, init_lr, epoch, total_epochs):
    import math
    cur_lr = init_lr * 0.5 * (1.0 + math.cos(math.pi * epoch / total_epochs))
    for g in optimizer.param_groups:
        g['lr'] = init_lr if g.get('fix_lr', False) else cur_lr

def train_one_epoch(loader, model, optimizer):
    for images, _ in loader:                     # images = [view1, view2]
        p1, p2, z1, z2 = model(images[0], images[1])
        loss = simsiam_loss(p1, p2, z1, z2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

So the causal chain: I want a Siamese net that doesn't need negatives, a momentum encoder, or huge batches, so I strip BYOL down to a single shared encoder and toggle the stop-gradient — and find that the lone stop-gradient is what stands between collapse and a working representation, while every other knob only moves accuracy. That necessity says I'm secretly solving a different problem, and reverse-engineering it gives an alternating minimization `min_{θ,η} E‖F_θ(T(x))−η_x‖²` over the encoder weights and a set of per-image expected-representation targets, where holding one block fixed *is* the stop-gradient, the one-sample target approximation *is* the Siamese second view, the predictor `h` *is* the learned stand-in for the augmentation expectation that approximation drops, and symmetrization *is* just denser sampling — landing on a method that is BYOL without the momentum encoder, SimCLR without negatives, SwAV without online clustering, kept off the constant solution by nothing more than alternating optimization.
