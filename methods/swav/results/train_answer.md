We want to learn good visual representations from unlabeled images, and we want to do it *online* — a single streaming pass over the data, with no separate phase that revisits the whole dataset — so that the method scales to arbitrarily large, even unlimited, collections and stays cheap in memory and compute on commodity GPUs. The recipe that actually works today is contrastive instance discrimination: take an image, make two augmented views, pull their features together, and push the features of *other* images apart. The "push apart" is what does the real work, because without it the trivial minimizer maps every image to one constant vector — perfect agreement, zero information. So negatives are the engine, and they are also the cost. The quality of the representation grows with the *number* of negatives in the softmax denominator, which is precisely why every practical system carries machinery whose sole purpose is to supply more of them: SimCLR needs batches of thousands; MoCo maintains a second momentum-updated encoder and a queue of about sixty-five thousand stored features; NPID keeps a running feature for every image in a memory bank. In all three the encoder is almost incidental and the bulk of the system is scaffolding for a large, consistent pile of negative vectors. The cost I want to kill is exactly this explicit feature-to-feature comparison.

Clustering already offers an alternative to per-instance negatives — group similar features and use the group identity as the target, so the cross-entropy over groups does the contrasting implicitly. But the existing clustering recipes fail on the very axes I care about. DeepCluster runs k-means over the features of the *entire* dataset once an epoch, which is offline (about a third of training time) and cannot consume a stream; it is prone to collapse (nothing stops k-means from emptying clusters or the network from sending everything to one cluster), so it needs re-assignment heuristics and balanced sampling; and because cluster identities are arbitrary from epoch to epoch, its classification head must be thrown away and relearned at every re-clustering, jolting the network. SeLa fixes the principled side by casting the assignment as an optimal-transport problem with an equal-partition constraint solved by Sinkhorn–Knopp, but it is still offline over the whole dataset and *rounds* the soft transport solution to a hard label. I want the clustering benefit — no explicit negatives, a fixed-size target space — but online, and without the collapse hacks.

I propose SwAV — Swapping Assignments between Views. The first move is to stop comparing features to a thrashing crowd of random instance vectors and instead compare each view's L2-normalized embedding $z$ to a small, *stable* set of $K$ trainable reference vectors, the **prototypes** $C = [c_1, \dots, c_K]$. "What does $z$ look like" becomes "how does $z$ distribute over the prototypes," a $K$-vector of similarities $z^\top C$ that is bounded in size and does not grow with the dataset. This is implemented as a bias-free linear layer $z \mapsto z^\top C$, learned by backprop along with everything else. The second, defining move is how the views are tied together. The original instance-discrimination insight is not "regress an offline target" but "two views of the same image should map to the same thing" — I only need the assignment to be *consistent across views*, not to be ground truth about the image. So I make consistency explicit and symmetric: from view $s$ compute a soft code $q_s$ (its assignment over the prototypes), from the *other* view $t$'s embedding $z_t$ predict $q_s$, and swap. The objective for a pair is

$$L(z_t, z_s) = \ell(z_t, q_s) + \ell(z_s, q_t), \qquad \ell(z, q) = -\sum_k q^{(k)} \log p^{(k)}, \qquad p^{(k)} = \frac{\exp\!\big(\tfrac{1}{\tau}\, z^\top c_k\big)}{\sum_{k'} \exp\!\big(\tfrac{1}{\tau}\, z^\top c_{k'}\big)}.$$

The comparison is never $z_t$ against $z_s$ directly; it is routed through the codes — a feature against the *cluster assignment* of its sibling view. The temperature $\tau \approx 0.1$ is carried over from the contrastive recipe and is load-bearing: the embeddings and prototypes are unit-norm, so $z^\top c_k$ lives in $[-1,1]$ and the raw logits are tiny; dividing by $\tau$ sharpens the softmax enough that the gradient is decisive rather than mush. L2-normalizing both $z$ and the $c_k$ makes the similarity a clean cosine so the scale is controlled by $\tau$ alone, not by drifting norms. Because $\sum_k q^{(k)} = 1$, each $\ell$ term expands to $-\tfrac{1}{\tau} z_t^\top C q_s + \log \sum_k \exp(z_t^\top c_k/\tau)$, and this is minimized jointly over the prototypes $C$ and the encoder parameters — fully online, with the codes $q$ computed from features in the current batch only.

Removing every explicit negative reopens the collapse I started with: if the network were free to choose the codes that minimize this loss, the global optimum is grotesque — send every embedding to one point, put one prototype there, set every code to it, and the cross-entropy is zero while the features are garbage. The negatives were the only thing blocking this, so I must re-introduce the anti-collapse pressure *structurally* when computing the codes. Collapse is all the assignment mass piling onto one prototype, so I forbid the pile-up by construction: within a batch the codes must spread evenly across the prototypes, each prototype claiming its fair share of the assignment mass. This **equipartition** constraint replaces pairwise repulsion with a global insistence that the batch be partitioned evenly. Concretely, for a batch embedding matrix $Z = [z_1, \dots, z_B]$ and prototypes $C$, I seek a code matrix $Q \in \mathbb{R}_+^{K \times B}$ that puts mass where similarity is high subject to fixed marginals, and I soften a pure linear assignment (which would give a brittle, over-aggressive hard matching) with an entropy term:

$$\max_{Q \in \mathcal{Q}} \ \mathrm{Tr}(Q^\top C^\top Z) + \varepsilon\, H(Q), \qquad H(Q) = -\sum_{ij} Q_{ij} \log Q_{ij}, \qquad \mathcal{Q} = \Big\{ Q \in \mathbb{R}_+^{K \times B} : Q \mathbf{1}_B = \tfrac{1}{K}\mathbf{1}_K,\ Q^\top \mathbf{1}_K = \tfrac{1}{B}\mathbf{1}_B \Big\}.$$

Here $Q$ is represented as a joint distribution — every row sums to $1/K$, every column to $1/B$ — and after the solve is multiplied by $B$, so each column becomes an ordinary per-image code summing to one and each prototype receives $B/K$ of the assignment weight; "all images to one code" is simply outside the feasible set. The entropy term makes the problem strictly concave (unique, smooth solution) and hands me a closed form, but $\varepsilon$ must stay *small*: it pulls $Q$ toward the maximum-entropy point of the polytope, the uniform matrix where every image is assigned equally to every prototype — itself a collapse — so a large $\varepsilon$ would wash out the data and return uniform-everything. Solving with multipliers $\alpha \in \mathbb{R}^K$ for the rows and $\beta \in \mathbb{R}^B$ for the columns, setting $M = C^\top Z$, the stationarity condition $M_{ij} - \varepsilon(\log Q_{ij} + 1) + \alpha_i + \beta_j = 0$ separates into a row factor and a column factor, giving

$$Q^\ast = \mathrm{Diag}(u)\, \exp\!\big(C^\top Z / \varepsilon\big)\, \mathrm{Diag}(v).$$

The optimal soft code is just the exponentiated similarity kernel rescaled by one positive factor $u \in \mathbb{R}^K$ per row and one $v \in \mathbb{R}^B$ per column, and $u, v$ are exactly the scalings that enforce the marginals — a matrix-balancing problem solved by **Sinkhorn–Knopp**: fix $\exp(C^\top Z/\varepsilon)$, then alternately rescale rows to sum to $1/K$ and columns to sum to $1/B$. About **3 iterations** suffice; one is too few (the marginals aren't enforced and the loss won't converge) and many more barely change anything (a hint, since the soft solution is what I want). It is all matrix multiplies, and across machines only the row sums and total sum need all-reducing — normalization statistics, not the whole feature matrix — so it distributes far more cheaply than gathering everyone's features for pairwise negatives.

Two choices are essential to make this work online. First, I keep $Q^\ast$ **soft** rather than rounding to a hard assignment as the offline ancestor does: rounding is an extremely aggressive optimization step, much sharper than a gradient update, and online — where codes are recomputed every batch on a moving network — over-committing early locks in mistakes; the lightly smoothed code tracks the slowly improving features. Second, the codes are **targets**, so I compute $Q^\ast$ under stop-gradient; otherwise the model would cheat by warping the codes to shrink the loss instead of improving the features. Gradients flow only through the prediction side $p$. A satisfying check confirms the prototypes are anchors, not class centroids: varying their number over an order of magnitude (three thousand to a hundred thousand) barely moves the features, and even fixing them at random works nearly as well — their job is to be a stable reference frame, so I learn them for a small gain and default to $K = 3000$.

Two practical holes remain. The equipartition only makes sense when the batch is comparable in size to the number of prototypes; with a small batch ($256$) and a few thousand prototypes, $B \ll K$ and the constraint degenerates. Rather than a 65k bank, I keep a small rolling **queue** of recent embeddings (about three thousand, the order of $K$) and prepend it to $Z$ *only for the Sinkhorn assignment*; the loss is still computed on the current batch's codes. That is fifteen batches of stored vectors against MoCo's two hundred and fifty, with no momentum encoder, switched on only after a few epochs so that the fast-changing early features don't poison the assignment. Finally, cropping does heavy lifting and comparing more than two views helps, but full-resolution views are quadratically expensive — so I use **multi-crop**: the two standard $224$ crops plus $V$ cheap low-resolution $96$ crops covering small parts of the image. The signal becomes "predict the full-image code from a tiny local patch," a strong part-to-whole pressure. Symmetry tempts me to let every view produce a code, but a $96$ crop's assignment is computed from partial information and using those as targets degrades transfer, so I keep the asymmetry: **only the two full-resolution crops produce codes**, and all $V+2$ views predict those two,

$$L = \sum_{i \in \{1,2\}} \sum_{v=1}^{V+2} \mathbf{1}_{v \neq i}\, \ell(z_{t_v}, q_{t_i}).$$

Mixing sizes rather than shrinking everything also avoids the train/test resolution bias. A couple of stabilizers fall out: freeze the prototypes for the first epoch so the assignment has something stable to balance against while the encoder is random, and since throughput favors very large batches, use LARS with warmup-then-cosine learning-rate scheduling. The end-to-end chain: contrastive learning works only because of many negatives, and negatives are the expensive part; replace per-instance negatives with a small set of prototype anchors and contrast cluster assignments; keep it online by swapping — predict one view's code from the other's embedding; re-introduce anti-collapse as an equipartition constraint; solve it as entropy-regularized optimal transport whose optimum is $\mathrm{Diag}(u)\exp(C^\top Z/\varepsilon)\mathrm{Diag}(v)$, computed by a few Sinkhorn normalizations, kept soft and detached with small $\varepsilon = 0.05$; patch small batches with a tiny queue; and amplify the view signal cheaply with multi-crop.

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

class Encoder(nn.Module):
    def __init__(self, output_dim=128, hidden_mlp=2048, nmb_prototypes=3000):
        super().__init__()
        self.backbone = resnet50_trunk()                       # ConvNet -> 2048-D (global avg pool)
        self.projection_head = nn.Sequential(
            nn.Linear(2048, hidden_mlp), nn.BatchNorm1d(hidden_mlp),
            nn.ReLU(inplace=True), nn.Linear(hidden_mlp, output_dim),
        )
        self.prototypes = nn.Linear(output_dim, nmb_prototypes, bias=False)  # C: z -> z^T C

    def forward(self, inputs):
        if not isinstance(inputs, list): inputs = [inputs]
        idx = torch.cumsum(torch.unique_consecutive(           # one forward pass per resolution
            torch.tensor([x.shape[-1] for x in inputs]), return_counts=True)[1], 0)
        start, out = 0, None
        for end in idx:
            feats = self.backbone(torch.cat(inputs[start:end]))
            out = feats if out is None else torch.cat((out, feats)); start = end
        z = F.normalize(self.projection_head(out), dim=1, p=2)  # unit-sphere embeddings
        return z, self.prototypes(z)                            # (embeddings, scores z^T C)

@torch.no_grad()
def distributed_sinkhorn(out, args):
    Q = torch.exp(out / args.epsilon).t()                       # K x local-B scores
    B = Q.shape[1] * _world_size()                              # global number of assigned samples
    K = Q.shape[0]

    Q /= _all_reduce(torch.sum(Q))                              # normalized joint mass
    for _ in range(args.sinkhorn_iterations):
        sum_of_rows = _all_reduce(torch.sum(Q, dim=1, keepdim=True))
        Q /= sum_of_rows
        Q /= K                                                  # rows -> 1/K globally

        Q /= torch.sum(Q, dim=0, keepdim=True)
        Q /= B                                                  # columns -> 1/B

    Q *= B                                                      # columns now sum to 1: assignment codes
    return Q.t()                                                # local-B x K soft codes

def train(loader, model, optimizer, epoch, lr_schedule, args, queue=None):
    model.train(); use_queue = False
    n_crops = int(np.sum(args.nmb_crops))
    prototypes = model.module.prototypes if hasattr(model, "module") else model.prototypes
    for it, views in enumerate(loader):                         # [2 full-res] + [V small] crop-batches
        iteration = epoch * len(loader) + it
        for g in optimizer.param_groups: g["lr"] = lr_schedule[iteration]
        with torch.no_grad():                                  # keep prototypes on the sphere
            w = F.normalize(prototypes.weight.data.clone(), dim=1, p=2)
            prototypes.weight.copy_(w)

        embedding, output = model(views)
        embedding = embedding.detach(); bs = views[0].size(0)

        loss = 0
        for i, crop_id in enumerate(args.crops_for_assign):     # codes from full-res crops only
            with torch.no_grad():
                out = output[bs*crop_id: bs*(crop_id+1)].detach()
                if queue is not None:                          # small-batch queue (assignment only)
                    if use_queue or not torch.all(queue[i, -1] == 0):
                        use_queue = True
                        out = torch.cat((queue[i] @ prototypes.weight.t(), out))
                    queue[i, bs:] = queue[i, :-bs].clone()
                    queue[i, :bs] = embedding[crop_id*bs:(crop_id+1)*bs]
                q = distributed_sinkhorn(out, args)[-bs:]        # detached target

            subloss = 0
            for v in np.delete(np.arange(n_crops), crop_id):
                x = output[bs*v: bs*(v+1)] / args.temperature
                subloss -= torch.mean(torch.sum(q * F.log_softmax(x, dim=1), dim=1))
            loss += subloss / (n_crops - 1)
        loss /= len(args.crops_for_assign)

        optimizer.zero_grad(); loss.backward()
        if iteration < args.freeze_prototypes_niters:           # freeze prototypes early
            for n, prm in model.named_parameters():
                if "prototypes" in n: prm.grad = None
        optimizer.step()
    return queue
```
