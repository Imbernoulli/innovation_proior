# DiffPool: differentiable hierarchical graph pooling

## Problem

Message-passing GNNs emit one embedding per node, but graph classification needs a single vector for
the whole graph. The standard global readout (sum / mean / set-pool over all node embeddings) is
*flat*: it collapses every node in one step and never represents the intermediate scales that real
graphs have (atoms → functional groups → molecule; nodes → communities → network). DiffPool supplies
the missing analogue of CNN spatial pooling — a *learned, differentiable* operator that coarsens a
graph into a smaller graph — so GNN layers can be stacked into a genuine hierarchy and trained
end-to-end with the classification task.

## Key idea

Pooling on a graph = clustering nodes into a coarser graph. A hard clustering is non-differentiable,
so DiffPool learns a **soft cluster assignment matrix** S^(l) ∈ R^{n_l × n_{l+1}}, each row a softmax
distribution over next-layer clusters. Given S, embeddings Z, and adjacency A at layer l:

  X^(l+1) = S^(l)ᵀ Z^(l)            (pool node embeddings into cluster-node features)
  A^(l+1) = S^(l)ᵀ A^(l) S^(l)       (coarsened weighted adjacency between clusters)

S and Z come from **two separate GNNs** on the same inputs (distinct parameters, distinct roles):

  Z^(l) = GNN_{l,embed}(A^(l), X^(l))                  — the content to carry forward
  S^(l) = softmax( GNN_{l,pool}(A^(l), X^(l)) )         — the partition; output width = n_{l+1}

Stack L such embed-then-pool blocks (n_{l+1} ≈ α·n_l, α ≈ 0.1–0.5), coarsening each layer; at the
final layer set S to a single all-ones column so all remaining nodes pool into one node → a single
graph vector → MLP + softmax classifier. The construction is permutation invariant whenever the
component GNNs are equivariant, since for any permutation P, X' = (PS)ᵀ(PZ) = SᵀZ and A' =
(PS)ᵀ(PAPᵀ)(PS) = SᵀAS (using PᵀP = I).

## Auxiliary objectives

The task gradient alone cannot reliably train the pooling GNN — the soft clustering is a non-convex,
factorization-like object with many spurious minima. Two auxiliary losses, added at every layer:

- **Link prediction:** L_LP = ‖ A^(l) − S^(l) S^(l)ᵀ ‖_F. Since (S Sᵀ)_{ij} = S_i·S_j is the soft
  probability that nodes i and j share a cluster, this makes connected nodes get grouped together.
- **Entropy:** L_E = (1/n) Σ_i H(S_i), minimized — pushes each assignment row toward one-hot so cluster
  membership is crisp rather than diffuse.

Total loss = classification cross-entropy + Σ_layers (L_LP + L_E). This same link objective collapses
dense, small-diameter cliques into single hypernodes (where message passing loses nothing) while
keeping sparse path/cycle/tree regions split across clusters (where structure must be preserved).

## Implementation

Grounded in the canonical dense pooling operator (softmax inside; Frobenius link loss; mean-entropy
loss). Graphs are batched as dense, max-node-padded tensors.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

def diff_pool(x, adj, s, normalize=True):
    # x: [B,n,d] embeddings (Z) ; adj: [B,n,n] ; s: [B,n,m] raw assignment scores
    s = torch.softmax(s, dim=-1)                                      # S = softmax(scores), row-wise
    out     = torch.matmul(s.transpose(1, 2), x)                      # X' = Sᵀ Z      -> [B,m,d]
    out_adj = torch.matmul(torch.matmul(s.transpose(1, 2), adj), s)   # A' = Sᵀ A S    -> [B,m,m]
    link_loss = adj - torch.matmul(s, s.transpose(1, 2))              # A - S Sᵀ
    link_loss = torch.norm(link_loss, p=2)                            # ‖A - S Sᵀ‖_F
    if normalize:
        link_loss = link_loss / adj.numel()
    ent_loss = (-s * torch.log(s + 1e-15)).sum(dim=-1).mean()         # mean_i H(S_i)
    return out, out_adj, link_loss, ent_loss

class GNN(nn.Module):
    """GCN-style message passing on a dense (adj, x); permutation-equivariant."""
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.W1 = nn.Linear(in_dim, hidden_dim)
        self.W2 = nn.Linear(hidden_dim, out_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)

    def norm_adj(self, adj):
        adj = adj + torch.eye(adj.size(-1), device=adj.device)        # Ã = A + I
        dinv = adj.sum(-1, keepdim=True).clamp(min=1).pow(-0.5)
        return dinv * adj * dinv.transpose(1, 2)                       # D̃^{-1/2} Ã D̃^{-1/2}

    def forward(self, adj, x):
        a = self.norm_adj(adj)
        h = F.relu(torch.matmul(a, self.W1(x)))
        b, n, c = h.shape
        h = self.bn1(h.reshape(b * n, c)).reshape(b, n, c)
        h = torch.matmul(a, self.W2(h))
        return F.normalize(h, p=2, dim=-1)                            # ℓ2-normalize per node

class DiffPoolNet(nn.Module):
    def __init__(self, in_dim, hidden, num_classes, max_nodes, assign_ratio=0.25, num_pool=1):
        super().__init__()
        self.embed_gnns, self.pool_gnns = nn.ModuleList(), nn.ModuleList()
        n, d_in = max_nodes, in_dim
        for _ in range(num_pool):
            m = max(int(assign_ratio * n), 1)                         # n_{l+1} = α · n_l
            self.embed_gnns.append(GNN(d_in, hidden, hidden))
            self.pool_gnns.append(GNN(d_in, hidden, m))
            n, d_in = m, hidden
        self.final_embed = GNN(d_in, hidden, hidden)
        self.classifier = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, num_classes))

    def forward(self, adj, x):
        link_total = ent_total = 0.0
        for embed_gnn, pool_gnn in zip(self.embed_gnns, self.pool_gnns):
            z = embed_gnn(adj, x)
            s = pool_gnn(adj, x)
            x, adj, lp, ent = diff_pool(z, adj, s)
            link_total, ent_total = link_total + lp, ent_total + ent
        z = self.final_embed(adj, x)
        graph_vec = z.sum(dim=1)                                      # final S = ones -> single graph vector
        return self.classifier(graph_vec), link_total, ent_total

def train_step(model, batch, opt):
    adj, x, y = batch
    logits, link_loss, ent_loss = model(adj, x)
    loss = F.cross_entropy(logits, y) + link_loss + ent_loss
    opt.zero_grad(); loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 2.0)
    opt.step()
    return loss.item()
```

Training details: Adam; gradient norm clipped at 2.0; ℓ2-normalize and batch-norm after each graph
conv; per-layer hidden representations may be concatenated for the final graph vector; node features
augmented with degree and clustering coefficient. The embedding GNN uses structural + node features
while the pooling GNN drops structural features to cluster by homophily; embedding width grows and
assignment width shrinks at deeper layers, mirroring CNN channel growth under spatial downsampling.
