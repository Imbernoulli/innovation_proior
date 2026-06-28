OK, let me think this through from scratch. I want representations from unlabeled images, and the thing that actually works right now is the contrastive instance-discrimination recipe: take an image, make two augmented views, push their features to agree, and push features of *other* images apart. The whole reason it learns anything nontrivial is the "push apart" — without it, the trivial minimizer is to map every image to the same vector, agreement perfect, information zero. So negatives are doing the real work; they are what prevents collapse.

But negatives are expensive, and not a little expensive. The quality of the features grows with the *number* of negatives in the denominator of the softmax. SimCLR gets them from a giant batch — you need thousands of images in a batch just so each one has enough other-image features to contrast against. MoCo gets them from a queue of about sixty-five thousand stored features plus a second momentum-updated encoder to keep those features consistent. NPID stores a running feature for every image in the dataset in a memory bank. Look at all three: the encoder is incidental, and an enormous fraction of the system is scaffolding that exists for one purpose — to have a large, consistent pile of negative vectors lying around to compare against. That's the cost I want to kill. Can I keep the "two views should agree" signal, keep something that stops collapse, but *never* do an explicit feature-to-feature negative comparison?

Let me stare at what a negative comparison even *is*. In InfoNCE the loss for view `z` with positive `z+` is `-log[ exp(z·z+/τ) / Σ_j exp(z·z_j/τ) ]`, the sum running over the positive and a bunch of negatives `z_j`. The negatives are other *images' features*. The signal I get from a negative `z_j` is just one number: how similar `z` is to that particular sampled instance. It is noisy — `z_j` is one random image — and to get a stable signal I have to average over many of them, which is exactly why I need thousands. What if, instead of comparing `z` to a thrashing crowd of random instance vectors, I compared it to a small, *stable* set of reference vectors that summarize where features tend to live? A fixed-ish set of anchors, say `K` of them, `{c_1, ..., c_K}` — call them prototypes. Then "what does `z` look like" becomes "how does `z` distribute over the `K` prototypes" — a `K`-vector of similarities, `z^T C`, instead of a list of comparisons to random images. That's bounded in size and doesn't grow with the dataset.

Grouping images by their features and using the group as a target — that's clustering, and clustering for representations already exists, so let me see how far the existing recipe gets me before reinventing it. DeepCluster runs k-means on the features of the *whole dataset* once an epoch, calls the cluster index of each image its pseudo-label, and trains a classifier to predict those labels. No explicit negatives — the cross-entropy over `K` clusters implicitly does the contrasting (predicting one class pushes down the others). Good. But two things bite. First, it is offline: to make the pseudo-labels you must forward the entire dataset and re-cluster every epoch — that's about a third of the training time, and it cannot scale to a stream of unlimited data. Second, collapse is right there waiting: nothing stops k-means from emptying clusters or the network from sending everything to one cluster, so DeepCluster needs re-assignment heuristics and balanced sampling to limp around it. And a sneaky third problem: the cluster *identities* are arbitrary from one epoch to the next — cluster 5 this epoch has nothing to do with cluster 5 last epoch — so the classification head has to be thrown away and relearned at every re-clustering, which jolts the network each time.

So I want the clustering *benefit* (no explicit negatives, a fixed-size target space) but online, and without those collapse hacks. Let me hold onto the prototypes idea and rebuild the objective so it never needs the full dataset.

In DeepCluster the code (the cluster assignment) is treated as a *target* you regress onto: compute it offline, freeze it, predict it. That's what forces the offline pass. But the original instance-discrimination insight is subtler than "predict a target" — it's "two views of the same image should map to the same thing." I don't actually need the code to be a ground truth about the image. I only need it to be *consistent across views*. So let me make the consistency explicit and symmetric: from view `s` compute a code `q_s` (its soft assignment over the prototypes), and from the *other* view `t`'s embedding `z_t`, try to *predict* `q_s`. And vice versa. If the two views genuinely carry the same content, the embedding of one should be enough to recover the code of the other. Swap them:

    L(z_t, z_s) = ℓ(z_t, q_s) + ℓ(z_s, q_t).

Predict the code of one view from the *embedding of the other*. I never compare `z_t` to `z_s` directly; I route the comparison through the codes. Where contrastive learning compares features to features, this compares a feature to the *cluster assignment* of its sibling view — and the assignment is a fixed-size `K`-vector, so the comparison cost no longer scales with how many negatives I can afford to drag around.

What is `ℓ(z, q)` concretely? The natural thing, mirroring the clustering classifier: turn the embedding's prototype similarities into a probability with a softmax, and take the cross-entropy against the code. So with `p^(k)` the softmax over prototypes,

    p_t^(k) = exp( (1/τ) z_t^T c_k ) / Σ_{k'} exp( (1/τ) z_t^T c_{k'} ),
    ℓ(z_t, q_s) = - Σ_k q_s^(k) log p_t^(k).

I keep the temperature `τ` from the contrastive recipe because it sharpens the softmax — without it the prototype logits are tiny (the embeddings are unit-norm, so `z^T c_k` lives in [-1,1]) and the gradient is mush; `τ ≈ 0.1` makes the distribution decisive enough to learn from. I'll L2-normalize both `z` and the prototypes `c_k` so the similarity is a clean cosine and the scale is controlled by `τ` alone, not by drifting norms.

Now if I expand the full objective over the batch and both views, each `ℓ` term is `-Σ_k q^(k) log p^(k)`, and `log p_t^(k) = (1/τ) z_t^T c_k - log Σ_{k'} exp((1/τ) z_t^T c_{k'})`. Since `Σ_k q^(k)=1`, summing gives, per image `n` and pair `(s,t)`,

    -(1/τ) z_{nt}^T C q_{ns} - (1/τ) z_{ns}^T C q_{nt} + log Σ_k exp(z_{nt}^T c_k/τ) + log Σ_k exp(z_{ns}^T c_k/τ),

and I minimize this jointly over the prototypes `C` and the encoder parameters `θ`. The prototypes are just a bias-free linear layer `z ↦ z^T C`, learned by backprop along with everything else. So far nothing here touches the full dataset — every quantity is computed from features *in the current batch only*. That was the whole point of routing through codes; it seems to have held.

But wait. I've removed every explicit negative. What stops the collapse now? Let me actually pin down how bad the failure is rather than wave at it. Suppose `K=2` prototypes and a batch of `B=4` images, and let the network be free to choose both the codes and where it sends embeddings. The degenerate solution: send every embedding to one point, put prototype 0 right there, and set every code to `q = (1,0)`. Then for every image the prediction `p` also concentrates on prototype 0, and the cross-entropy `-Σ_k q^(k) log p^(k) = -1·log 1 = 0`. I can confirm this numerically — take scores `M = C^T Z` where prototype 0 wins for every image:

    M = [[ 2, 2, 2, 2],
         [-1,-1,-1,-1]]

and assign each image by the plain per-column softmax of `M/ε` with `ε=0.05`. The result is

    [[1, 1, 1, 1],
     [0, 0, 0, 0]]

— every image to prototype 0, prototype 1 receiving total mass 0.0, starved. Loss zero, features garbage. So the collapse is not hypothetical; the unconstrained assignment walks straight into it. The negatives were the only thing standing between me and this in the contrastive world, and I just threw them out. The collapse pressure didn't go away; I have to block it some *other* way, structurally, when I compute the codes.

Look again at the failure: all the assignment mass piling onto one prototype. The clean way to forbid that is to *forbid the pile-up by construction* — require that within a batch the codes spread evenly across the prototypes, each prototype receiving roughly the same total share of the batch. If every prototype is forced to claim its fair fraction of the batch's assignment mass, then "everything to one code" is simply not in the feasible set. That's an **equipartition** constraint. Let me check it actually defeats the degenerate case above. With the same scores `M`, but now solving for codes that obey the equipartition (each of the two prototypes must carry mass `B/K = 2`), the codes come out

    [[0.5, 0.5, 0.5, 0.5],
     [0.5, 0.5, 0.5, 0.5]],

row sums `[2, 2]` exactly, column sums `[1, 1, 1, 1]`. The constraint refused the all-to-prototype-0 target: when the scores give it no way to tell the images apart, it splits each image's mass evenly rather than collapse. And crucially these are *targets* the prediction side then has to match — the network can no longer make the loss zero by sending everything to one place, because the supervision it's handed is no longer "everything to one place." This is the role negatives were playing, re-expressed as a constraint on the assignment rather than as pairwise repulsion.

Let me make the constraint precise. For a batch with embedding matrix `Z = [z_1, ..., z_B]` (each column a `D`-vector on the sphere) and prototype matrix `C = [c_1, ..., c_K]`, I want a code matrix `Q ∈ R^{K×B}`, column `q_n` being image `n`'s soft assignment over the `K` prototypes. In the optimal-transport solve I represent `Q` as a joint distribution: every column sums to `1/B`, every row sums to `1/K`, and the whole matrix sums to one. After multiplying this solution by `B`, each column becomes an ordinary per-image code summing to one and each prototype receives `B/K` total assignment weight. I want that joint `Q` to put mass where similarity `C^T Z` is high — maximize `Tr(Q^T C^T Z) = Σ_n q_n^T (C^T z_n)` — subject to those marginals. That is the transportation polytope

    Q = { Q ∈ R_+^{K×B} :  Q 1_B = (1/K) 1_K ,  Q^T 1_K = (1/B) 1_B }.

A pure linear program over this polytope would give a hard, vertex assignment — a permutation-like matching — which is brittle and, as I'll see, too aggressive online. So I soften it: add an entropy term and maximize

    max_{Q ∈ 𝒬}  Tr(Q^T C^T Z) + ε H(Q),   H(Q) = -Σ_{ij} Q_{ij} log Q_{ij}.

The `ε H(Q)` makes the problem strictly concave so the solution is unique and smooth, and it gives me a closed-form structure (coming next) that I can iterate cheaply. But `ε` cuts both ways: it pulls `Q` toward the *maximum-entropy* point of the polytope, which is the uniform matrix where every image is assigned equally to every prototype — itself a collapse, just a different one. So `ε` has to stay small, and I should check how small actually matters. On separable scores

    M = [[ 2.0, 1.5, -1.0, -0.5],
         [-1.0,-0.5,  2.0,  1.0]]

(images 0,1 lean to prototype 0; images 2,3 to prototype 1), I solve the entropy-regularized OT at three values of `ε` and look at how decisive the resulting codes are (mean of the largest code entry per image):

    ε = 0.05 :  mean max-code = 1.00 ,  q for image 0 = (1.00, 0.00)
    ε = 0.5  :  mean max-code = 0.98 ,  q for image 0 = (0.997, 0.003)
    ε = 5.0  :  mean max-code = 0.62 ,  q for image 0 = (0.64, 0.36)

So small `ε` produces near-hard codes that follow the similarities; at `ε=5` the codes drift toward uniform `(0.5, 0.5)` *regardless* of the scores — the similarity term has been washed out and I'm back to the everything-to-everything collapse. That fixes the regime I want: `ε` small (I'll use 0.05), strong adherence to the data, with just enough smoothing to be stable.

Now solve for the form of `Q`. Lagrangian, with multipliers `α ∈ R^K` for the row constraints and `β ∈ R^B` for the column constraints, writing `M = C^T Z`:

    L = Σ_{ij} Q_{ij} M_{ij} - ε Σ_{ij} Q_{ij} log Q_{ij} + α^T (Q 1_B - 1_K/K) + β^T (Q^T 1_K - 1_B/B).

Differentiate in `Q_{ij}`: `M_{ij} - ε(log Q_{ij} + 1) + α_i + β_j = 0`, so

    log Q_{ij} = (M_{ij} + α_i + β_j)/ε - 1
    ⇒  Q_{ij} = exp(α_i/ε - 1/2) · exp(M_{ij}/ε) · exp(β_j/ε - 1/2).

The `α_i` part depends only on the row, the `β_j` part only on the column. Fold them into positive vectors `u ∈ R^K` (rows) and `v ∈ R^B` (columns):

    Q^* = Diag(u) exp(C^T Z / ε) Diag(v).

So the optimum should be just the exponentiated similarity matrix `exp(C^T Z/ε)` rescaled by one factor per row and one per column. Let me confirm that the matrix I actually compute has this structure and I didn't drop a term in the Lagrangian. Run the row/column normalization to convergence on the separable `M` above (with `ε=0.5`), call the result `Q`, and form the ratio `R = Q / exp(M/ε)`. If the closed form is right, `R` must be rank-one, `R_{ij} = u_i v_j`, so every row of `R` is the same up to scale — dividing each row by its first entry should give the *same* row of column-ratios for `i=0` and `i=1`. It does, to machine precision:

    R / R[:,0]  =  [[1, 2.6528, 0.6409, 4.5964],
                    [1, 2.6528, 0.6409, 4.5964]]

identical across the two rows (max spread per column `0.0`). So the matrix really factors as `Diag(u) exp(M/ε) Diag(v)`; the derivation checks out. And `u, v` are exactly the scalings that make the row marginals hit `1/K` and the column marginals hit `1/B`. That's a matrix-balancing problem with a dead-simple iterative solution: fix the kernel `K_mat = exp(C^T Z/ε)`, then alternate — rescale the rows so each sums to `1/K`, then rescale the columns so each sums to `1/B`, repeat. This is Sinkhorn–Knopp. In distributed training the row sums and total sum are all-reduced across workers, and after the normalized OT solve the implementation multiplies by the global `B` so each local returned code column sums to one.

Let me trace the actual `distributed_sinkhorn` on a small input to make sure those marginals and the final rescale come out where I claim. One GPU (world size 1), `K=2`, `B=4`, separable scores `M` as above, `ε=0.05`, 3 iterations. Following the code exactly — exponentiate, normalize by the total to a joint distribution, then three times {divide rows by their sums, divide by `K`; divide columns by their sums, divide by `B`}, then multiply the whole thing by `B` — I get

    Q = [[1, 1, 0, 0],
         [0, 0, 1, 1]],

column sums `[1, 1, 1, 1]` (so each column is a usable per-image code), row sums `[2, 2]` which is `B/K = 2` (each prototype carries its fair share), and as a fraction of the total mass `B` that's `[0.5, 0.5] = 1/K`. Exactly the marginals the polytope prescribes. Good — the `Q *= B` at the end is what converts the joint-distribution columns (summing to `1/B`) into ordinary codes (summing to 1), and it does.

It converges geometrically; about **3** iterations is already enough — fewer (one) and the marginals aren't enforced and the loss won't converge; many more barely changes anything and, oddly, *slightly hurts*, which is a hint I'll come back to. And it's all matrix multiplies, trivial on a GPU, and across machines I only need to share normalization statistics, not the whole feature matrix — so it distributes far more cheaply than gathering everyone's features for pairwise negatives.

Two subtleties before I trust this. First, hard versus soft codes. SeLa, the offline ancestor that first casts this assignment as optimal transport, *rounds* `Q^*` to a hard assignment. If I take the same online analog and round to discrete codes, it converges faster but lands at a clearly worse model; the soft `Q^*` wins. The reason lines up with the Sinkhorn-iterations observation: rounding is an extremely aggressive optimization step, far sharper than a gradient update, and online (where the codes are recomputed every batch on a moving network) over-committing to a hard code early locks in mistakes. The soft, lightly-smoothed code is gentler and tracks the slowly-improving features better. So I keep `Q^*` soft, with small `ε` and just 3 Sinkhorn steps — exactly enough structure, not enough to over-commit.

Second, and this is the one that almost slipped past me: the codes are *targets*. If gradients flow back through `q` into the prototypes and the encoder, the model can cheat by warping the codes to make the loss small rather than improving the features. So I compute `Q^*` under stop-gradient — it is the supervision signal for this batch — and let gradients flow only through the *prediction* side, the softmax `p`. The encoder and prototypes learn to predict a code they didn't get to tamper with.

There's one thing about the prototypes I want to settle, because it changes how much I should fuss over `K`. Are they semantic class centroids (in which case their number and their learning matter a great deal) or just contrast anchors (in which case neither does)? The way to tell is to vary their number over an order of magnitude — three thousand, thirty thousand, a hundred thousand — and see whether the features move. They barely do. Even *fixing* the prototypes at random and never training them works almost as well. If they were class centroids, that would be impossible; that it's true tells me their real job is to be a stable reference frame against which views are made consistent, not to discover the "true" categories. I'll still learn them (a small gain) and pick a few thousand, but I now read `K` as the size of a reference frame, not a guess at the number of classes.

One more practical hole. The equipartition only makes sense if the batch is at least as big as the number of prototypes I'm spreading across — I need `B` comparable to `K` to partition `B` images into `K` parts at all. With a small batch (256) and a few thousand prototypes, `B ≪ K` and the constraint is degenerate. Contrastive methods hit the same wall and answered it with a memory bank or a 65k queue. I can answer it far more cheaply: since the codes only need *enough features to balance the assignment*, I keep a small rolling queue of recent embeddings — about three thousand, the same order as `K` — and prepend them to `Z` *only for the Sinkhorn assignment step*. I still compute the loss on the current batch's codes alone. Three thousand stored vectors is fifteen batches of 256, versus MoCo's sixty-five thousand from two hundred and fifty batches — a tiny tail, no momentum encoder, no second network. (I'll also wait a few epochs before switching the queue on, because in the very first epochs the network changes so fast that stale queued features would just confuse the assignment.)

Now the augmentation side, separately. The cropping is doing heavy lifting — comparing two crops forces the model to relate parts of a scene — and there's evidence that comparing *more* than two views helps. But adding more full-resolution `224×224` views is quadratically expensive in memory and compute, which is why everyone uses two. What if the extra views are *small*? Take the two standard full-resolution crops plus `V` extra low-resolution crops — say `96×96` — that each cover only a small part of the image. The small crops are cheap (resolution drives cost), so I get many more views for almost no extra memory. The training signal becomes: map these little local patches to the global codes — predict the full-image code from a tiny zoomed-in piece, which is a strong "the part should agree with the whole" pressure.

But which views produce codes? My first instinct is symmetry — let every view compute a code and predict every other. That breaks. A `96×96` crop covers a sliver of the image; its assignment is computed from partial information and is low quality, and using those as targets actually degrades transfer. So I keep the asymmetry: **only the two full-resolution crops produce codes**; all `V+2` views must *predict* those two codes. With `z_{t_1}, ..., z_{t_{V+2}}` and codes from views `i ∈ {1,2}`,

    L = Σ_{i ∈ {1,2}} Σ_{v=1}^{V+2} 1_{v≠i} ℓ(z_{t_v}, q_{t_i}).

There's a second reason to mix sizes rather than just shrink everything: training purely on downsized crops biases the features (a train/test resolution mismatch), so I keep two full-resolution crops in the mix and only *add* the small ones. Simple, and it doesn't increase memory the way more big crops would.

Let me assemble the loop and make sure every piece traces back. Per batch: augment each image into `2 + V` views; forward them through the ConvNet, a 2-layer MLP projection head, L2-normalize to get embeddings `z`; the prototype layer gives scores `z^T C`. For each of the two full-res crops, under no-grad, run Sinkhorn on its scores (with the queue prepended if batches are small) to get the soft code `q`. Then for every *other* view, take its softmax-over-prototypes prediction `p` at temperature `τ` and accumulate the cross-entropy `-Σ_k q^(k) log p^(k)`. Backprop through the predictions only (codes are detached), SGD-update the encoder and prototypes, and normalize the prototype vectors before they are used on the next iteration. A couple of stabilizers fall out of the structure: freeze the prototypes for the first epoch so the assignment has something stable to balance against while the encoder is still random; and since I'm training with very large batches for throughput, use LARS with a warmup-then-cosine learning-rate schedule.

Causal chain, end to end: contrastive learning works but only because of many negatives, and negatives are the expensive part → replace per-instance negatives with a small fixed set of prototype anchors and contrast *cluster assignments* instead of features → keep it online by computing codes from the current batch and making the objective a *swapped* prediction (predict one view's code from the other's embedding) rather than regressing offline targets → that removes the only thing preventing collapse (and a tiny numeric check confirmed the unconstrained assignment really does drive everything to one prototype), so re-introduce anti-collapse *structurally* as an equipartition constraint on the batch assignment, which the same check shows refuses that degenerate target → solve that as an entropy-regularized optimal-transport problem whose optimum factors as `Diag(u) exp(C^T Z/ε) Diag(v)` (verified rank-one numerically), computed by a few Sinkhorn row/column normalizations that I traced to the right marginals, kept soft and detached, with small `ε` (verified that large `ε` washes out to uniform) → patch small batches with a tiny feature queue instead of a 65k bank → and amplify the view signal cheaply with multi-crop, computing codes only from the full-resolution crops. Now the code.

```python
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import torch.distributed as dist

def _dist_ready():
    return dist.is_available() and dist.is_initialized()

def _all_reduce(x):
    if _dist_ready():
        dist.all_reduce(x)
    return x

def _world_size():
    return dist.get_world_size() if _dist_ready() else 1

# ---------- model: backbone + projection head + prototypes ----------
class Encoder(nn.Module):
    # the prototypes C are a bias-free linear layer z -> z^T C (scores over K prototypes)
    def __init__(self, output_dim=128, hidden_mlp=2048, nmb_prototypes=3000):
        super().__init__()
        self.backbone = resnet50_trunk()                      # standard ConvNet, global-avg-pooled -> 2048-D
        self.projection_head = nn.Sequential(                 # 2-layer MLP (kept from contrastive practice)
            nn.Linear(2048, hidden_mlp), nn.BatchNorm1d(hidden_mlp),
            nn.ReLU(inplace=True), nn.Linear(hidden_mlp, output_dim),
        )
        self.prototypes = nn.Linear(output_dim, nmb_prototypes, bias=False)

    def forward(self, inputs):                                # inputs: list of view-batches (mixed resolutions)
        if not isinstance(inputs, list): inputs = [inputs]
        # group crops by resolution so each resolution is one forward pass
        idx = torch.cumsum(torch.unique_consecutive(
            torch.tensor([x.shape[-1] for x in inputs]), return_counts=True)[1], 0)
        start, out = 0, None
        for end in idx:
            feats = self.backbone(torch.cat(inputs[start:end]))
            out = feats if out is None else torch.cat((out, feats)); start = end
        z = self.projection_head(out)
        z = F.normalize(z, dim=1, p=2)                        # embeddings on the unit sphere
        return z, self.prototypes(z)                          # (embeddings, prototype scores z^T C)

# ---------- the codes: entropy-regularized OT with equipartition, via Sinkhorn ----------
@torch.no_grad()
def distributed_sinkhorn(out, args):
    # out: local-B x K scores for one full-resolution crop, plus optional queued scores
    Q = torch.exp(out / args.epsilon).t()                    # K x local-B, matching paper notation
    B = Q.shape[1] * _world_size()                           # global number of samples being assigned
    K = Q.shape[0]

    Q /= _all_reduce(torch.sum(Q))                           # normalized joint distribution
    for _ in range(args.sinkhorn_iterations):
        sum_of_rows = _all_reduce(torch.sum(Q, dim=1, keepdim=True))
        Q /= sum_of_rows
        Q /= K                                               # row marginal: 1/K, globally

        Q /= torch.sum(Q, dim=0, keepdim=True)
        Q /= B                                               # column marginal: 1/B

    Q *= B                                                   # columns sum to 1, i.e. usable soft codes
    return Q.t()

# ---------- training: swapped prediction across (multi-crop) views ----------
def train(loader, model, optimizer, epoch, lr_schedule, args, queue=None):
    model.train()
    use_queue = False
    n_crops = int(np.sum(args.nmb_crops))
    prototypes = model.module.prototypes if hasattr(model, "module") else model.prototypes
    for it, views in enumerate(loader):                      # views: list of [2 full-res] + [V small] crop-batches
        iteration = epoch * len(loader) + it
        for g in optimizer.param_groups: g["lr"] = lr_schedule[iteration]

        with torch.no_grad():                                # keep prototypes on the unit sphere
            w = F.normalize(prototypes.weight.data.clone(), dim=1, p=2)
            prototypes.weight.copy_(w)

        embedding, output = model(views)                     # output: (sum_crops * B) x K scores
        embedding = embedding.detach()
        bs = views[0].size(0)

        loss = 0
        for i, crop_id in enumerate(args.crops_for_assign):  # compute codes ONLY from full-res crops {0,1}
            with torch.no_grad():
                out = output[bs * crop_id: bs * (crop_id + 1)].detach()
                if queue is not None:                        # small-batch patch: enlarge Z for the assignment only
                    if use_queue or not torch.all(queue[i, -1] == 0):
                        use_queue = True
                        out = torch.cat((queue[i] @ prototypes.weight.t(), out))
                    queue[i, bs:] = queue[i, :-bs].clone()
                    queue[i, :bs] = embedding[crop_id * bs:(crop_id + 1) * bs]
                q = distributed_sinkhorn(out, args)[-bs:]      # detached target code

            subloss = 0                                      # predict q from every OTHER view's embedding
            for v in np.delete(np.arange(n_crops), crop_id):
                x = output[bs * v: bs * (v + 1)] / args.temperature
                subloss -= torch.mean(torch.sum(q * F.log_softmax(x, dim=1), dim=1))
            loss += subloss / (n_crops - 1)
        loss /= len(args.crops_for_assign)

        optimizer.zero_grad(); loss.backward()
        if iteration < args.freeze_prototypes_niters:         # freeze prototypes early for a stable assignment
            for n, prm in model.named_parameters():
                if "prototypes" in n: prm.grad = None
        optimizer.step()
    return queue
```
