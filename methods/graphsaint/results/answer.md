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
edge appearances C_{u,v}; then α_{u,v} = C_{u,v}/C_v and λ_v = |V|·C_v/N (in code, their reciprocals:
each edge scaled by C_v/C_{u,v}, each node's loss by N/(C_v·|V|)). The N subgraphs are reused as
minibatches.

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
- **Edge:** sample m edges (with replacement) ∝ 1/deg(u)+1/deg(v); take endpoints.
- **Node:** sample n nodes ∝ ‖Â_{:,v}‖².
- **Random walk:** r roots, each an h-hop uniform walk (proxy for an h-layer GCN, B = Â^L).
- **Multi-dimensional random walk (frontier):** r-node frontier, repeatedly pick u ∝ deg(u), step to a
  random neighbor, swap in.

## Implementation

```python
import numpy as np
import scipy.sparse as sp
import torch, torch.nn as nn, torch.nn.functional as F

def sampler_edge(adj_train, budget_m, deg):
    rows, cols = adj_train.nonzero()
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

def induce(adj_train, sub):
    return adj_train[sub][:, sub].tocsr()                         # node-induced subgraph

def estimate_norms(adj_train, sampler, num_train, coverage=50):
    n = adj_train.shape[0]
    C_v = np.zeros(n); subgraphs = []; total = 0
    while total <= coverage * num_train:
        sub = sampler(adj_train); subgraphs.append(sub); total += len(sub); C_v[sub] += 1
    N = len(subgraphs)
    loss_norm = np.where(C_v > 0, N / np.clip(C_v, 1, None) / num_train, 0.1)   # 1/lambda_v = N/(C_v|V|)
    return subgraphs, loss_norm                                  # aggr 1/alpha = C_v/C_uv computed per edge

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

def train(adj_train, features, labels, model, opt, sampler, num_train, steps):
    subgraphs, loss_norm = estimate_norms(adj_train, sampler, num_train)
    loss_norm = torch.tensor(loss_norm, dtype=torch.float32)
    i = 0
    for _ in range(steps):
        if i >= len(subgraphs):
            subgraphs += [sampler(adj_train) for _ in range(len(subgraphs))]
        sub = subgraphs[i]; i += 1
        sub_adj = induce(adj_train, sub)
        sub_adj = sp.diags(1.0/np.clip(sub_adj.sum(1).A1, 1, None)).dot(sub_adj)  # Â = D^{-1}A on subgraph
        adj_norm = scale_by_aggr_norm(sub_adj, sub)               # each edge × C_v/C_uv  (aggregator norm)
        logits = model(adj_norm, features[sub])
        per_node = F.cross_entropy(logits, labels[sub], reduction='none')
        loss = (per_node * loss_norm[sub]).sum()                 # loss norm: each L_v ÷ lambda_v
        opt.zero_grad(); loss.backward(); opt.step()
```

Training uses Adam; hidden dim and dropout swept per dataset; evaluation is full-batch (no sampling).
Each minibatch GCN being complete makes the method a drop-in backbone for jumping-knowledge networks,
graph attention (applied within the subgraph), and graph-classification pooling (replace A with A_s).
