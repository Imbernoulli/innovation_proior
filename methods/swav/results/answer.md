# SwAV — Swapping Assignments between Views

## Problem

Learn visual representations from unlabeled images **online** (single streaming pass, scalable to unlimited data) and **cheaply**, without the large-negative-set machinery — big batches, memory banks, or a momentum encoder with a long queue — that contrastive instance-discrimination methods need, and without the offline full-dataset re-clustering and collapse heuristics of clustering methods.

## Key idea

Don't compare image features pairwise. Map each view's L2-normalized embedding `z` to a set of `K` trainable **prototypes** `C = [c_1, ..., c_K]`, producing a soft **code** `q` (a cluster assignment). Then solve a **swapped prediction** problem: predict the code of one view from the embedding of another view, and vice versa.

    L(z_t, z_s) = ℓ(z_t, q_s) + ℓ(z_s, q_t),
    ℓ(z, q) = - Σ_k q^(k) log p^(k),   p^(k) = softmax_k( z^T c_k / τ ).

The contrast that prevents collapse is supplied not by negatives but by an **equipartition constraint** when computing the codes online: within a batch, the assignment mass must be spread evenly across all prototypes, so the trivial "all images → one code" solution is infeasible.

## Computing codes: entropy-regularized optimal transport

For a batch of embeddings `Z = [z_1,...,z_B]` (`D×B`) and prototypes `C` (`D×K`), find the soft code matrix `Q` (`K×B`):

    max_{Q ∈ 𝒬}  Tr(Q^T C^T Z) + ε H(Q),   H(Q) = -Σ_{ij} Q_{ij} log Q_{ij},
    𝒬 = { Q ∈ ℝ_+^{K×B} : Q 1_B = (1/K) 1_K,  Q^T 1_K = (1/B) 1_B }.

The row constraint gives each prototype `1/K` of the normalized OT mass, while each column has mass `1/B`; after the implementation multiplies the solution by `B`, each sample's code sums to one and each prototype receives `B/K` total assignment weight. The optimum has closed form

    Q* = Diag(u) exp( C^T Z / ε ) Diag(v),

where `u ∈ ℝ^K`, `v ∈ ℝ^B` are obtained by the **Sinkhorn–Knopp** algorithm — alternately normalizing rows to `1/K` and columns to `1/B` on the kernel `exp(C^T Z/ε)`. About **3 iterations** suffice. The code `Q*` is kept **soft** (rounding to a hard assignment optimizes too aggressively and hurts online) and is **detached** (a target; gradients flow only through the prediction `p`). `ε` is kept **small** (0.05) — large `ε` collapses `Q` toward the uniform-everything solution. `τ = 0.1`.

## Practical pieces

- **Prototypes** are a bias-free linear layer whose prototype vectors are renormalized to unit norm at the start of each iteration (equivalently, before using the weights after the previous step); frozen for the first epoch. Their number barely matters (3k–100k equivalent) and even fixed-random prototypes work nearly as well — they are contrast anchors, not class centroids. Default `K = 3000`.
- **Small batches:** prepend a small rolling **queue** of recent embeddings (~3k vectors) to `Z` *for the Sinkhorn assignment only* (loss still on batch codes). Far cheaper than a 65k memory bank or a momentum encoder. Turned on after a few epochs.
- **Multi-crop:** use 2 full-resolution crops (224) plus `V` low-resolution crops (96) for many cheap extra views. Compute codes **only from the full-resolution crops**; every view predicts those codes:

      L = Σ_{i ∈ {1,2}} Σ_{v=1}^{V+2} 1_{v≠i} ℓ(z_{t_v}, q_{t_i}).

  Small crops have partial information, so they predict but never produce codes; mixing sizes also avoids the train/test resolution bias.
- **Optimization:** projection head = 2-layer MLP (with BatchNorm) to 128-D; LARS, lr warmup + cosine decay, synchronized BatchNorm; large batches for throughput.

## Code

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

Defaults: `K=3000` prototypes, `τ=0.1`, `ε=0.05`, 3 Sinkhorn iterations, 128-D projection, ResNet-50 backbone, multi-crop `2×224 + 6×96`.
