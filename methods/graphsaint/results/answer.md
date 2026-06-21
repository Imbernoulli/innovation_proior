# GraphSAINT: graph-sampling-based minibatch training of GCNs

## Problem

Training a graph convolutional network by minibatches on a large graph is defeated by *neighbor
explosion*: producing one node's output under an L-layer GCN requires its L-hop neighborhood, on the
order of d^L support nodes, so cost grows multiplicatively with depth. Prior methods sample
nodes/edges *per layer* to bound this, but either still blow up with depth (GraphSAGE, S-GCN),
shred inter-layer connectivity into too-sparse minibatches (FastGCN), pay for a learned sampler
(AS-GCN), or use a biased, uncorrected estimator (ClusterGCN).

## Key idea

Sample the **training graph**, not the GCN. Each minibatch is an independently sampled subgraph
G_s(V_s, E_s) with |V_s| ≪ |V|, on which a *complete* L-layer GCN is built. Support never leaves the
subgraph (no explosion) and connectivity never degrades with depth (an edge in G_s is present at all
layers). Because a connectivity-preserving sampler is non-uniform, the minibatch estimator is biased;
two analytically-derived normalizations remove the bias, and a variance analysis selects the sampler.

## Normalization (unbiased estimator)

Analyze each layer independently. For a sampled node v at layer ℓ+1, with x̃_u^(ℓ) = (W^(ℓ))ᵀx_u^(ℓ):

  ζ_v^(ℓ+1) = Σ_{u∈V} (Â_{v,u} / α_{u,v}) x̃_u^(ℓ) 1_{u|v}.

Since E[1_{u|v}] = P((u,v) sampled | v sampled) = p_{u,v}/p_v, choosing the **aggregator
normalization** α_{u,v} = p_{u,v}/p_v makes ζ_v^(ℓ+1) an unbiased estimator of the full aggregation
Σ_u Â_{v,u} x̃_u^(ℓ).

For the loss, L_batch = Σ_{v∈V_s} L_v/λ_v with **loss normalization** λ_v = |V|·p_v gives
E[L_batch] = (1/|V|) Σ_{v∈V} L_v (since each node contributes with probability p_v).

Probabilities are estimated by preprocessing: run the sampler N times, count node appearances C_v and
edge appearances C_{u,v}; then α_{u,v} = C_{u,v}/C_v and λ_v = |V|·C_v/N. In the reference
implementation the stored quantities are reciprocals: each directed edge leaving v is scaled by
C_v/C_{u,v}, and each training node's loss is scaled by N/(C_v·|V_train|). Zero-count fallbacks and
clipping are implementation guards, not part of the unbiased estimator.

## Variance reduction (choosing the sampler)

With b_e^(ℓ) = Â_{v,u}x̃_u^(ℓ-1) + Â_{u,v}x̃_v^(ℓ-1) and ζ = Σ_ℓ Σ_{v∈V_s} ζ_v^(ℓ)/p_v =
Σ_ℓ Σ_e (b_e^(ℓ)/p_e) 1_e^(ℓ), under independent edge sampling (Σ_e p_e = m), and using that an edge
once sampled is present at all layers (1_e^(ℓ) = 1_e), Cov over distinct edges is 0 and
Cov(1_e^(ℓ1),1_e^(ℓ2)) = p_e − p_e², so

  Var(ζ) = Σ_e (Σ_ℓ b_e^(ℓ))²/p_e − Σ_e (Σ_ℓ b_e^(ℓ))².

Minimizing the first term s.t. Σ_e p_e = m via Cauchy–Schwarz gives

  p_e = m · ‖Σ_ℓ b_e^(ℓ)‖ / Σ_{e'} ‖Σ_ℓ b_{e'}^(ℓ)‖.

Dropping the activation dependence for a cheap topology-only sampler:

  p_e ∝ Â_{v,u} + Â_{u,v} = 1/deg(u) + 1/deg(v),

which matches the intuition that connected low-degree nodes most influence each other and should be
co-sampled.

## Samplers

All return a **node-induced** subgraph (induction adds back intra-subgraph edges, speeding convergence):
- **Edge:** sample m undirected edges (with replacement) ∝ 1/deg(u)+1/deg(v); take endpoints.
- **Node:** paper-level distribution ∝ ‖Â_{:,v}‖²; the public code path samples from a cumulative
  distribution built from training-CSR row counts.
- **Random walk:** r roots, each an h-hop uniform walk (proxy for an h-layer GCN, B = Â^L).
- **Multi-dimensional random walk (frontier):** r-node frontier, repeatedly pick u ∝ deg(u), step to a
  random neighbor, swap in.

## Implementation

```python
import numpy as np
import scipy.sparse as sp
import torch, torch.nn as nn, torch.nn.functional as F

def sampler_edge(adj_train, budget_m, deg):
    coo = sp.triu(adj_train, k=1).tocoo()                  # undirected edges counted once
    rows, cols = coo.row, coo.col
    pe = 1.0 / deg[rows] + 1.0 / deg[cols]; pe /= pe.sum()        # p_e ∝ 1/deg(u)+1/deg(v)
    pick = np.random.choice(len(rows), size=budget_m, replace=True, p=pe)
    return np.unique(np.concatenate([rows[pick], cols[pick]]))

def sampler_rw(adj_train, num_roots, walk_len):
    nodes = set(np.random.randint(adj_train.shape[0], size=num_roots))
    for r in list(nodes):
        u = r
        for _ in range(walk_len):
            nbrs = adj_train.indices[adj_train.indptr[u]:adj_train.indptr[u+1]]
            if len(nbrs) == 0: break
            u = np.random.choice(nbrs); nodes.add(u)
    return np.array(sorted(nodes))

def induce_with_edge_ids(adj_train, sub):
    sub = np.array(sorted(np.unique(sub)))
    pos = {v: i for i, v in enumerate(sub)}
    indptr, indices, edge_ids = [0], [], []
    for v in sub:
        for eid in range(adj_train.indptr[v], adj_train.indptr[v + 1]):
            u = adj_train.indices[eid]
            if u in pos:
                indices.append(pos[u]); edge_ids.append(eid)
        indptr.append(len(indices))
    return sp.csr_matrix((np.ones(len(indices)), indices, indptr), shape=(len(sub), len(sub))), np.array(edge_ids)

def estimate_norms(adj_train, sampler, train_nodes, coverage=50):
    n = adj_train.shape[0]
    num_train = len(train_nodes)
    C_v = np.zeros(n); C_uv = np.zeros(adj_train.nnz); subgraphs = []; total = 0
    while total <= coverage * num_train:
        sub = sampler(adj_train); subgraphs.append(sub); total += len(sub); C_v[sub] += 1
        _, edge_ids = induce_with_edge_ids(adj_train, sub); C_uv[edge_ids] += 1
    N = len(subgraphs)
    rows = np.repeat(np.arange(n), np.diff(adj_train.indptr))
    aggr_norm = np.divide(C_v[rows], C_uv, out=np.full_like(C_uv, 0.1), where=C_uv > 0)
    aggr_norm = np.clip(aggr_norm, 0, 1e4)                       # 1/alpha = C_v/C_uv
    loss_norm = np.zeros(n)
    loss_norm[train_nodes] = np.where(C_v[train_nodes] > 0, N / C_v[train_nodes] / num_train, 0.1)
    return subgraphs, aggr_norm, loss_norm

class GCN(nn.Module):
    def __init__(self, in_dim, hidden, num_classes, num_layers):
        super().__init__()
        dims = [in_dim] + [hidden] * (num_layers - 1)
        self.weights = nn.ModuleList(nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:]))
        self.out = nn.Linear(dims[-1], num_classes)
    def forward(self, adj_norm, x):
        h = x
        for W in self.weights:
            h = F.relu(torch.sparse.mm(adj_norm, W(h)))           # complete GCN on the subgraph
        return self.out(h)

def to_torch_sparse(adj):
    coo = adj.tocoo()
    idx = torch.LongTensor(np.vstack([coo.row, coo.col]))
    val = torch.FloatTensor(coo.data)
    return torch.sparse.FloatTensor(idx, val, torch.Size(coo.shape))

def train(adj_train, features, labels, model, opt, sampler, train_nodes, loss_fn, steps):
    deg_train = np.asarray(adj_train.sum(1)).ravel()
    subgraphs, aggr_norm, loss_norm = estimate_norms(adj_train, sampler, train_nodes)
    loss_norm = torch.tensor(loss_norm, dtype=torch.float32)
    i = 0
    for _ in range(steps):
        if i >= len(subgraphs):
            subgraphs += [sampler(adj_train) for _ in range(len(subgraphs))]
        sub = subgraphs[i]; i += 1
        sub_adj, edge_ids = induce_with_edge_ids(adj_train, sub)
        sub_adj.data = aggr_norm[edge_ids]                        # each edge × C_v/C_uv
        adj_norm = sp.diags(1.0/np.clip(deg_train[sub], 1, None)).dot(sub_adj)  # D_train^{-1}(A_s/alpha)
        adj_norm = to_torch_sparse(adj_norm)
        logits = model(adj_norm, features[sub])
        per_node = loss_fn(logits, labels[sub], reduction='none')
        loss = (per_node * loss_norm[sub]).sum()                 # loss norm: each L_v ÷ lambda_v
        opt.zero_grad(); loss.backward(); opt.step()
```

Training uses Adam; hidden dim and dropout swept per dataset; evaluation is full-batch (no sampling).
Each minibatch GCN being complete makes the method a drop-in backbone for jumping-knowledge networks,
graph attention (applied within the subgraph), and graph-classification pooling (replace A with A_s).
