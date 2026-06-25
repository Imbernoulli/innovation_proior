Let me start from the thing that actually bugs me. In language, the win was self-supervised pretraining — predict a masked word, predict the next token — and the reason it works is that a sentence hands you a dense, structured prediction problem: thousands of little supervised tasks fall out of the text itself, far richer than one label for the whole sentence. In vision we mostly still do the opposite: take an image full of texture, objects, layout, relations, and crush it down to a single category index from a fixed list of a few thousand classes. That throws away almost everything. And the patch-based Transformer encoders that just showed up are exactly the architectures that suffer most from a thin signal — they have weak built-in priors, they're data-hungry, and trained with plain supervision on a mid-sized labeled set their features don't look special. So the question I want to chase: can I learn good features from images with no labels at all, with something simple enough that I don't have to redesign it per architecture?

The obvious label-free idea is invariance: two augmented views of the same image should map to nearly the same thing. Crop differently, jitter color, blur — the content is the same, so make the embeddings agree. Clean. Except it has a catastrophic trivial optimum: map *every* image to the same constant vector. Then all views of all images agree perfectly, loss zero, features useless. This is collapse, and it's not a corner case I can ignore — it's the attractor that "make views agree" walks straight into. Every method I know in this space is really defined by the one trick it uses to dodge collapse. So before anything else, let me get clear on what those tricks are and what each one actually costs, because the design I want is going to be "which anti-collapse device is cheapest and most architecture-agnostic."

First family: contrastive / instance discrimination. Treat each image as its own class. Pull the two views of an image together, and push views of *other* images — negatives — apart. Concretely a noise-contrastive / InfoNCE loss: for a query, softmax over the similarity to its positive against a pile of negatives, maximize the positive. Here collapse is impossible by construction — a constant encoder makes every pair equally similar, so the contrastive softmax is maximally confused and the loss is huge. Negatives are the anti-collapse device. But the cost is real: you need *many* negatives simultaneously to get a useful gradient, which means either enormous batches or an auxiliary memory of past embeddings, a queue. And once you keep a queue, the embeddings in it go stale as the encoder drifts, so people encode the keys with a slowly-moving copy of the encoder — a momentum encoder, weights updated as an exponential moving average `θ_k ← λ θ_k + (1−λ) θ_q`, no gradient through it. Note that: the momentum encoder enters here only as a consistency hack for the queue. Keep it in mind, I think it's more important than that.

Second family: clustering with a balance constraint. Map features onto `K` learned prototypes, get a soft assignment, but *force the assignment to be balanced across the batch* — every prototype gets roughly equal mass — by solving a little optimal-transport problem with the Sinkhorn-Knopp iteration. Then swap: predict one view's assignment from the other view. Balance is the anti-collapse device: if no prototype is allowed to hog the mass, the encoder can't dump everything onto one point. This family also dragged in a component I like a lot independently of the clustering: multi-crop. Instead of two equal-size crops, take two big "global" crops and several small low-resolution "local" crops, and match across them. It's cheap because the local crops are tiny, and it's a big free accuracy gain. I'm going to want that regardless of how I resolve collapse. The cost of the clustering itself, though: the Sinkhorn balancing *couples all the samples in a batch* and runs an iterative normalization. That's batch-dependent and a bit heavy.

Third family is the one that nags at me. Drop negatives entirely. Just regress one view's output onto the other view's output — but the target side is a slowly-moving copy of the network (an EMA, like the momentum encoder again) with the gradient stopped, and you put an extra little prediction head on only one branch so the two branches aren't symmetric. Mean-squared error on ℓ2-normalized embeddings. No negatives, no queue, no clustering — and it works. Which is bizarre, because "make views agree, no negatives" is *exactly* the objective that should collapse to a constant. So why doesn't it? The going explanation is: the predictor asymmetry plus the EMA target. The target moves slowly, so it's effectively a fixed function of the recent past; the online branch has to predict it through that extra predictor head; and the predictor can't instantaneously become the constant map, so the system stays off the trivial solution and gets dragged along by its own slowly-updated history. A sibling method even shows the EMA isn't strictly necessary — predictor plus stop-gradient alone keep it off collapse. Honestly the mechanism here is subtle and a little mysterious, the architecture is asymmetric, the objective is a feature-space regression with no probabilistic meaning, and it's been reported to lean on batch normalization inside the heads to stay alive. That's a lot of moving parts whose individual roles nobody is fully sure of.

So here's the landscape: contrastive needs negatives (batch/queue), clustering needs Sinkhorn balancing (batch-coupled, iterative), the negative-free regressor needs a predictor + EMA + maybe BN and has a murky mechanism. I want something simpler and architecture-agnostic. Let me look at this from a completely different lineage and see if it reframes the problem.

Knowledge distillation. There you have a fixed pretrained teacher and you train a student to reproduce the teacher's *output distribution* — soften the teacher's logits with a temperature, soften the student's, match them with cross-entropy over the output dimensions. Self-training is the same shape: a teacher emits *soft pseudo-labels* on unlabeled data and the student learns from those. The catch, of course, is that distillation assumes a teacher that already exists. I have no labels and no pretrained teacher. But stare at the *form* of the distillation loss for a second, because it's different from everything above. It's not "are these two feature vectors close" — it's "does the student's *probability distribution over K outputs* match the teacher's distribution." That's a categorical cross-entropy, `H(P_t, P_s) = −Σ P_t log P_s`, where `P` is a softmax over a `K`-dim head output. And the matching is across views: feed the teacher one view, the student another, make the student's distribution match the teacher's. If I could conjure a teacher, self-supervised learning would just *be* distillation with no labels — predicting a teacher's softened output instead of a one-hot label.

So the question becomes: where does the teacher come from with no labels? It has to be built from the student itself, from its own past. And now the momentum encoder I flagged twice comes back, but with a totally different job. Forget "consistency device for a queue." There's a separate, old, well-established fact: averaging a model's weights over training iterations gives a model *better* than any single iterate — Polyak-Ruppert averaging — and people use a "mean teacher," an EMA of the student's weights, to provide targets in semi-supervised learning precisely because the averaged teacher is a running ensemble that tends to be *ahead* of the student in quality. That's exactly what a distillation target should be: something a bit better than the student, that the student can chase. So let me set the teacher to be an EMA of the student: `θ_t ← λ θ_t + (1−λ) θ_s`, stop-gradient on the teacher, gradient only into the student. The teacher isn't a peer distilling back (that would be codistillation, bidirectional); it's literally the time-average of the student. And because it's an average, I'd expect it to consistently lead the student through training and hand it higher-quality targets — a free, always-improving teacher.

Let me also throw out the predictor. The predictor only existed in the negative-free regressor to break the symmetry and dodge collapse there. If I'm going to handle collapse a different way (I'll get to it), then I don't need the asymmetry, and the student and teacher can be the *exact same architecture* `g = h ∘ f` — a backbone `f` plus a projection head `h` — with different weights. That's much cleaner, and it makes the mean-teacher reading literal: same network, teacher = time-average of student.

Now write the objective. Build distributions with a temperature softmax on the `K`-dim head output: `P_s(x)^(i) = softmax(g_θs(x)/τ_s)_i`, and similarly `P_t` with its own temperature `τ_t`. Cross-entropy `H(P_t, P_s)`. Bring in multi-crop: from one image make a set `V` of two global crops `{x_1^g, x_2^g}` and several small local crops. Push *all* crops through the student, but only the two global crops through the teacher — I want the teacher's target to come from a high-information global view, and I want the student to learn to predict that global distribution even from a tiny local crop. That's a "local-to-global" task baked into the loss. So:

`min_θs  Σ_{x ∈ {x_1^g, x_2^g}}  Σ_{x' ∈ V, x' ≠ x}  H(P_t(x), P_s(x'))`,

summing over every (teacher-global view, student-other view) pair, skipping the case where they're the same view. Let me unroll the double sum in the bare two-crop case (`V = {x_1, x_2}`, no locals yet) to make sure the bookkeeping does what I intend. Teacher view `x_1`: student views except `x_1` is just `{x_2}` → term `H(P_t(x_1), P_s(x_2))`. Teacher view `x_2`: student views except `x_2` is just `{x_1}` → term `H(P_t(x_2), P_s(x_1))`. Two terms, so after the `/n_terms` averaging it is `H(P_t(x_1), P_s(x_2))/2 + H(P_t(x_2), P_s(x_1))/2` — symmetric, as it should be, and the same-view diagonal (predicting a view's teacher distribution from its own student distribution, which would be trivially easy and uninformative) is correctly excluded. When I add local crops later, the same `x' ≠ x` rule will pair every local crop with both teacher globals, which is the local-to-global signal I want; I'll re-check that count when I write the loop.

Good — but I've quietly deferred the only question that matters. I threw away negatives, I threw away Sinkhorn balancing, I threw away the predictor. Those were *the* anti-collapse devices in the three families. So what stops *this* from collapsing? Nothing yet. If I run it as written, the EMA teacher and the student will happily agree on a constant output and the loss goes to zero with garbage features. The teacher being a slow average might delay it, but a slow path to collapse is still collapse. I have to put an anti-collapse mechanism back in, and I'd like the cheapest possible one — ideally something that touches only the teacher's output and doesn't couple the batch the way Sinkhorn does.

Let me think hard about *what collapse even looks like* in this distributional setup, because the answer should tell me what to prevent. The teacher's output is a distribution `P_t(x)` over `K` dimensions. Collapse means the features carry no information about the input — i.e. `P_t(x)` is essentially the *same distribution regardless of `x`*. There are two ways for a distribution to become input-independent:

One: it concentrates on a single dimension. One coordinate's logit runs away, the softmax is near one-hot on that coordinate for *every* image. Every image maps to "dimension 7," say. Constant output, collapse.

Two: it spreads out to uniform. Every coordinate gets `1/K`, for *every* image. Also constant, also collapse.

These are genuinely different failure modes — opposite ends, even: maximally peaked versus maximally flat — and that's the lever. If I can find one operation that pushes *away* from "one dimension dominates" and another that pushes *away* from "uniform," and they pull in opposite directions, then balancing the two might hold me in the stable middle where the output actually depends on the input.

Take the "one dimension dominates" mode first. The runaway happens because some coordinate of the teacher logits drifts persistently high relative to the others, and the softmax amplifies that into near-one-hot. The cheapest thing that kills a persistent per-coordinate bias is to *subtract the running mean of the teacher logits* before the softmax. Maintain a center `c` — an exponential moving average of the teacher's raw outputs over batches — and use `g_t(x) − c`. This is just adding a bias term with value `−c` to the teacher logits, and it only needs a *first-order* statistic, the mean. That's a big deal versus Sinkhorn: the mean is cheap, it doesn't couple samples in a fancy iterative way, and I can EMA it across batches:

`c ← m c + (1−m) (1/|T|) Σ_{z∈T} g_θt(z)`,

where `T` is the set of teacher global-view outputs in the minibatch, all-reduced across workers in distributed training. The claim I'm making is that subtracting `c` kills the dominant-dimension collapse, but I should not just assert that — let me set up the smallest case that actually exhibits the failure and watch what centering does to it. Take `K = 8` and a batch where coordinate 0 has drifted persistently high: each image's teacher logits are a large *shared* bias of `+4` on coordinate 0 plus a small per-image term of std `0.3` on every coordinate. That per-image term is the only thing that depends on the image — the real signal — and the `+4` is the runaway bias I'm worried about. Sharpen with `τ_t = 0.04` and *no* centering, and over a batch of 2000 images the softmax argmax lands on coordinate 0 for 100% of images and the mean output entropy is ≈ 0. Every image maps to the same near-one-hot vector; the output has thrown away the per-image signal entirely. That is exactly mode one, and now I can see *why* it happens — a per-coordinate bias shared across all images survives the softmax and swamps the small image-specific differences.

Now subtract the center (here just the batch mean of the logits, which removes that shared `+4`) and sharpen the same way: the argmax now lands on coordinate 0 for only 13% of images, all 8 coordinates get used as argmaxes across the batch, and the mean entropy rises to ≈ 0.34. So centering doesn't merely "recenter" — it strips off the component of the logits that is *common to every image*, and what's left for the softmax to amplify is the per-image signal. The output starts depending on the input again. Good: centering genuinely cures mode one, and I now believe it for a concrete reason rather than by analogy.

But the same example warns me about the catch. What is centering's *limit*? It keeps subtracting whatever the mean carries; if I lean on it without a counterforce, it flattens the per-coordinate differences. Let me check that directly: keep the centering but weaken the sharpening to `τ_t = 1.0` on the same batch. The mean entropy jumps to ≈ 2.04, and `log K = log 8 ≈ 2.08` — within rounding, the teacher has gone *uniform*. So centering cures mode one and, left to itself, walks straight into mode two. Measured, not hand-waved.

That measurement also tells me what the second operation has to be. The thing that pulled the centered output away from uniform in the working case was the low temperature: dividing logits by a small `τ_t` before the softmax concentrates mass. Sharpening is the cure for mode two. And symmetrically — visible in the first run above — sharpening *with no centering* is precisely what let coordinate 0 dominate; a peaky distribution with nothing recentering it collapses to a single coordinate. So the two operations are not just "both helpful": each one's cure is the other one's disease. Centering kills the dominant dimension but tends toward uniform; sharpening kills uniform but tends toward a dominant dimension. The three runs I just did are the three corners — sharpen-only (entropy 0, one dimension), center+weak-sharpen (entropy ≈ log K, uniform), center+sharpen (entropy 0.34, input-dependent and away from both extremes). Applying *both* — center the teacher logits, then sharpen with a low `τ_t` — the two pressures push in opposite directions and the middle run sits in the stable regime. And neither operation needs negatives, a predictor, or batch coupling. That's the anti-collapse mechanism I wanted: two cheap teacher-only knobs.

I want to make the complementarity precise, not just hand-wave "opposite directions," and there's a clean decomposition that does it. The loss is a cross-entropy between two distributions, and cross-entropy splits as entropy plus KL:

`H(P_t, P_s) = h(P_t) + D_KL(P_t ‖ P_s)`,

where `h(P_t) = −Σ P_t log P_t` is the teacher's own entropy and `D_KL` is the divergence from teacher to student. Let me confirm I have the identity right before I lean on it, with a concrete pair of distributions over `K = 4`: take `P_t = (0.9211, 0.0459, 0.0023, 0.0307)` and `P_s = (0.4756, 0.2137, 0.1934, 0.1173)`. Then `H(P_t, P_s) = −Σ P_t log P_s = 0.8249`. And `h(P_t) = −Σ P_t log P_t = 0.3380`, `D_KL(P_t‖P_s) = Σ P_t log(P_t/P_s) = 0.4869`; their sum is `0.3380 + 0.4869 = 0.8249`. The two agree to four places, so the split holds and I can read the loss as "teacher self-entropy plus teacher-to-student divergence."

This matters because it separates two things the loss is doing. `D_KL → 0` alone only says the student is matching the teacher, and a perfect noncollapsed student could in principle do that — driving the KL down is *not* by itself a collapse signal. The collapse diagnostic in these ablations is the pair of facts: the KL falls toward zero *and* the teacher entropy `h(P_t)` lands at an extreme, showing an input-independent teacher. The two extremes are computable. If the teacher went one-hot, `h = −1·log 1 = 0`. If it went uniform over `K`, every coordinate is `1/K` and `h = −Σ (1/K) log(1/K) = log K`; for the actual `K = 65536` that is `log 65536 = 11.09`. So the two broken configurations should leave fingerprints at two *specific, different* entropy values — `0` and `11.09` — and that difference is what makes them distinguishable rather than just "both bad." I can predict each: sharpening but *no centering* should send the teacher peaky and one coordinate takes over, so the KL falls and `h → 0` — and indeed in the `K = 8` batch above that configuration gave mean entropy ≈ 0, the small-`K` echo of this. Centering but *no sharpening* should flatten the teacher, so the KL falls and `h → log K` — again matching the ≈ log 8 I measured above. Two different collapse entropies from dropping two different operations are measurable evidence that they induce *opposite* failure modes. With both on, `h` should stay away from both extremes (the middle run sat at 0.34, well clear of 0 and of log 8 ≈ 2.08) and the KL should not show the same degenerate fall.

Let me sanity-check the directions of the knobs and the hyperparameters, because a sign error here would be silent. `τ_s = 0.1` for the student; I want the *teacher* sharper than the student so the target is a confident thing the student is pulled toward, so `τ_t < τ_s`, around 0.04-0.07. And I should watch the boundary: too *high* a teacher temperature means too little sharpening, so centering wins and I'd expect the loss to drift toward `ln K` — the uniform-collapse signature; in practice a fixed value much above ~0.06 falls into this. That argues for keeping a conservative target around `τ_t = 0.04`, and for warming `τ_t` up from a small value (say 0.04) toward 0.07 over the first several epochs rather than slamming it high immediately, since early on the teacher is essentially random and very fragile to collapse. The other end, `τ_t → 0`, is the argmax — a hard one-hot target — which is the dominant-dimension danger zone, fine only because centering counterbalances it. And the center's EMA rate `m`: it just has to track the moving mean; too slow (m far too close to 1) and the center lags reality and lets a dimension run away before it's corrected, so a moderate `m` like 0.9 is the safe regime. The teacher momentum `λ` I'll put on a cosine schedule from 0.996 toward 1: early on the student is changing fast so the teacher should follow a bit more loosely; late in training I want a very stable, heavily-averaged teacher.

A couple of architecture choices in the head are load-bearing for stability, so let me reason them out rather than just declare them. The head `h` maps the backbone feature to the `K`-dim output. I'll make it a small MLP — three linear layers with GELU and hidden width 2048 — and crucially put an *ℓ2-normalization bottleneck* near the end: project down to a low-dimensional bottleneck (say 256), ℓ2-normalize it onto the unit sphere, then a final linear layer to `K`. Why the ℓ2 bottleneck? Without it, as I deepen the head the output magnitudes are free to blow up or shrink and the whole thing destabilizes — a deep head trained without the normalization just collapses outright. Normalizing onto the sphere bounds the bottleneck representation and lets me use a *deep* head (which helps accuracy) and a very *large* `K` (also helps) without the training falling apart. The final `K`-dim layer I weight-normalize and fix its gain to 1, and I freeze that last layer for the first epoch — early on, before the features mean anything, letting the output layer chase the target just adds instability, so I hold it still while the backbone finds its footing. Big `K` (tens of thousands, e.g. 65536) helps and the bottleneck keeps the parameter cost moderate.

And batch normalization: the negative-free regressor was reported to need BN in its heads, which is an architecture-specific dependency I'd rather not carry — BN couples the batch and, worse, has to be synchronized across GPUs, which is slow, and it's not even native to the Transformer encoder (it uses no BN by default). Since my anti-collapse mechanism is centering + sharpening, not BN, I can just *not put BN in the head*. For the Transformer that makes the whole system BN-free, which is a genuine simplification and removes a cross-device sync bottleneck. The fact that I *can* drop BN is itself evidence that BN was never doing essential work here — centering already handles the first-order statistics that BN would.

Let me also pin down the teacher-construction choice, since "EMA" was a guess motivated by the Polyak-Ruppert story and I should make sure simpler options don't suffice. If I set the teacher to a *copy of the current student* (or the student from the previous iteration), it doesn't converge — the target moves in lockstep with the student and there's nothing stable to chase, so it needs heavier normalization to even work. If I use the student from a *previous epoch* as a frozen teacher, that actually does train and gives decent features — interesting, it confirms "a stale-but-fixed student makes a usable target." But the EMA teacher beats it clearly, and the reason lines up with the Polyak-Ruppert intuition: the EMA is a *continuously* updated running ensemble, better than any single iterate, so it leads the student by a margin throughout training and feeds it steadily improving targets. That's the configuration I'll keep, and I'd expect — and want to verify — that the teacher's own accuracy sits *above* the student's for essentially all of training, which is the signature that it's acting as a leading ensemble rather than a lagging copy.

Now let me put it in code, because the loss has a few indexing subtleties (which views the teacher sees, which pairs get matched) that I want to be exact about. The heart is the loss module and the train step.

```python
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
import numpy as np


class DINOHead(nn.Module):
    # backbone feature -> 3-layer MLP (GELU) -> l2-normalized bottleneck -> weight-normed K-dim layer
    def __init__(self, in_dim, out_dim, use_bn=False, norm_last_layer=True,
                 nlayers=3, hidden_dim=2048, bottleneck_dim=256):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)
        # weight-normalized last layer; gain initialized to 1 and optionally frozen
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        x = F.normalize(x, dim=-1, p=2)   # the l2 bottleneck that lets the head be deep and K huge
        return self.last_layer(x)


class DINOLoss(nn.Module):
    def __init__(self, out_dim, ncrops, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs,
                 student_temp=0.1, center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp          # tau_s = 0.1
        self.center_momentum = center_momentum    # m for the center EMA
        self.ncrops = ncrops                       # 2 globals + locals
        self.register_buffer("center", torch.zeros(1, out_dim))   # the center c
        # warm up tau_t from a small value: the early teacher is fragile to collapse
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp, teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp,
        ))

    def forward(self, student_output, teacher_output, epoch):
        # student: sharpen-by-tau_s, split per view (ALL crops)
        student_out = (student_output / self.student_temp).chunk(self.ncrops)

        # teacher: CENTER then SHARPEN, softmax, stop-gradient; only the 2 global views
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)

        total_loss, n_terms = 0, 0
        for iq, q in enumerate(teacher_out):          # over the 2 teacher global views
            for v in range(len(student_out)):         # over every student view
                if v == iq:
                    continue                          # skip same-view (teacher view vs its own student view)
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)  # H(P_t, P_s)
                total_loss += loss.mean()
                n_terms += 1
        total_loss /= n_terms
        self.update_center(teacher_output)            # update c AFTER, on raw teacher logits
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(batch_center)
            batch_center = batch_center / (len(teacher_output) * dist.get_world_size())
        else:
            batch_center = batch_center / len(teacher_output)
        self.center = self.center * self.center_momentum \
            + batch_center * (1 - self.center_momentum)           # EMA of raw teacher logits
```

Note that the centering uses the *raw* teacher logits (before the center subtraction and softmax) to update `c`, and in distributed training I all-reduce that mean across workers so the center reflects the global batch. The update happens after computing the loss; the softmax in the loss subtracts the current `c` and divides by the low teacher temperature — center, then sharpen, exactly the two complementary operations.

Let me run the double loop in my head for a realistic crop set to confirm the count I promised earlier — 2 globals (student/teacher indices 0, 1) plus 2 locals (student indices 2, 3), so `ncrops = 4` and `teacher_out` has 2 entries. Teacher index 0 pairs with student indices {1, 2, 3} (skipping 0); teacher index 1 pairs with student indices {0, 2, 3} (skipping 1). That's the pairs `(0,1), (0,2), (0,3), (1,0), (1,2), (1,3)` — `n_terms = 6`. Of those, the four with student index 2 or 3 — `(0,2), (0,3), (1,2), (1,3)` — are local crops being asked to reproduce a teacher *global* distribution; that is the local-to-global signal, and it's present because every local crop is in `student_out` and `v == iq` only ever fires for the two globals. The two same-resolution global-global terms `(0,1), (1,0)` are the symmetric pair I unrolled earlier. So the loop does exactly what the objective specifies, with the diagonal correctly excluded.

The train step ties it together: student gets all crops, teacher gets only the two globals, gradient flows into the student only, then the teacher is nudged by EMA.

```python
def train_one_epoch(student, teacher, dino_loss, data_loader, optimizer,
                    lr_schedule, wd_schedule, momentum_schedule, epoch):
    for it, (views, _) in enumerate(data_loader):
        it_g = len(data_loader) * epoch + it
        for i, g in enumerate(optimizer.param_groups):
            g["lr"] = lr_schedule[it_g]
            if i == 0:
                g["weight_decay"] = wd_schedule[it_g]

        views = [v.cuda(non_blocking=True) for v in views]
        teacher_output = teacher(views[:2])    # only the 2 global views -> high-quality targets
        student_output = student(views)        # all crops -> local-to-global
        loss = dino_loss(student_output, teacher_output, epoch)

        optimizer.zero_grad()
        loss.backward()
        clip_gradients(student, clip=3.0)
        cancel_last_layer_grads(epoch, student, freeze_last_layer=1)  # hold the output layer the 1st epoch
        optimizer.step()

        # teacher = EMA of student (momentum encoder); stop-gradient is implicit (no backprop here)
        with torch.no_grad():
            m = momentum_schedule[it_g]    # cosine 0.996 -> 1
            for ps, pt in zip(student.parameters(), teacher.parameters()):
                pt.data.mul_(m).add_((1 - m) * ps.detach().data)
```

So the whole causal chain: I want label-free features and the natural "match two views" objective collapses to a constant, so I reframe matching as distillation of a teacher's softened output distribution; with no labels the teacher has to be built from the student, and a weight-EMA teacher is a Polyak-Ruppert running ensemble that leads the student and supplies improving targets, so I set `θ_t ← λ θ_t + (1−λ) θ_s` with stop-gradient and the same architecture for both (no predictor needed); the loss is cross-entropy between teacher-on-a-global-view and student-on-a-different-view, summed over multi-crop pairs so small local crops learn to predict the global distribution; and collapse, which now means an input-independent teacher distribution, comes in exactly two opposite modes — one dimension dominating versus uniform — which I counter with two complementary teacher-only operations, centering (subtract an EMA of the teacher logits, cheap first-order, batch-robust; kills the dominant dimension but tends toward uniform) and sharpening (low `τ_t`; kills uniform but tends toward a dominant dimension), the two balancing each other. The `H = h + D_KL` decomposition gives the diagnostic: missing centering sends teacher entropy to 0, missing sharpening sends it to `−log(1/K)=log K`, and both failures show the same degenerate KL fall. An ℓ2-bottleneck head lets it go deep with huge `K`, and dropping BN makes it architecture-agnostic and, on the Transformer, entirely BN-free.
