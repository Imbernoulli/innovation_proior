Training a graph convolutional network on a large graph with minibatches is blocked by neighbor explosion: to compute one node's output through L GCN layers you need its L-hop neighborhood, which can be on the order of d^L nodes. A naive node minibatch therefore drags in a huge and ever-growing support set as the network gets deeper. Existing scalable methods try to beat this by sampling nodes or edges at each layer, but each approach pays a different price. GraphSAGE caps per-node fanout, yet the budget still multiplies across layers so deep networks remain expensive. FastGCN samples each layer independently, which gives constant sample size but severs inter-layer connectivity: a node chosen at layer ℓ+1 often has no sampled neighbor at layer ℓ, so its representation degrades as depth increases. AS-GCN repairs connectivity with a learned sampler, but that adds parameters and cost. ClusterGCN avoids layer sampling by using dense graph clusters, yet it provides no inverse-probability correction for the unequal node and edge inclusion probabilities it introduces. What is needed is a way to keep minibatches small without thinning the connections a GCN needs across layers, while also keeping the minibatch gradient an unbiased estimate of the full-graph gradient.

The solution is GraphSAINT, a graph-sampling-based inductive training method for GCNs. Instead of sampling the GCN layer by layer, GraphSAINT samples a small subgraph G_s from the full graph and builds a complete L-layer GCN on that subgraph. Because the subgraph is fixed before propagation begins, every node sees all of its sampled neighbors at every layer, so connectivity never degrades with depth and the support of every node is confined to the subgraph. The key design questions then become how to sample the subgraphs, how to remove the bias created by non-uniform sampling, and how to minimize the variance of the resulting estimator.

GraphSAINT answers the bias question with two inverse-probability normalizations derived by treating each layer independently. For a sampled node v, the aggregated pre-activation over its neighbors in the full graph is Σ_u Â_{v,u} x̃_u, where x̃_u is the transformed neighbor feature. In the subgraph only the neighbors that were sampled are present, so each neighbor contribution must be rescaled by the reciprocal of the conditional probability that the edge survives given that v is sampled. This aggregator normalization is α_{u,v} = p_{u,v}/p_v, where p_v is the probability that node v appears in a sampled subgraph and p_{u,v} is the probability that edge (u,v) appears. With this correction the subgraph aggregation becomes an unbiased estimator of the full-graph aggregation. The loss must be reweighted as well, because frequently sampled nodes would otherwise dominate the gradient. The loss normalization is λ_v = |V|·p_v, so each sampled node's loss is divided by its expected number of appearances; this makes the minibatch loss an unbiased estimate of the average full-graph loss. The probabilities are unknown for complex samplers, so GraphSAINT estimates them once during preprocessing by running the sampler many times and counting how often each node and edge appears, then computing α_{u,v} = C_{u,v}/C_v and λ_v = |V|·C_v/N.

For variance reduction, GraphSAINT derives the optimal edge sampling probability under an independent-edge sampling budget. Writing b_e^(ℓ) for the contribution of edge e at layer ℓ and using the fact that a sampled edge persists across all layers, the variance of the combined estimator is minimized when p_e is proportional to the magnitude of the edge's total contribution across layers. Since computing that exactly requires activations that change every step, GraphSAINT drops the activation dependence and uses only topology. With the standard row-normalized adjacency Â = D^{-1}A, this yields p_e ∝ 1/deg(u) + 1/deg(v). This matches the intuition that connected low-degree nodes strongly influence each other and should be co-sampled. Several practical samplers instantiate this idea. The edge sampler picks budget_m undirected edges with probability proportional to 1/deg(u)+1/deg(v) and returns their endpoints. The node sampler samples nodes according to a FastGCN-style distribution and induces a subgraph. Random-walk and multi-dimensional random-walk samplers collect nodes that are mutually reachable, which is a natural proxy for the receptive field of an L-layer GCN. In every case the returned subgraph is node-induced, meaning all edges of the original graph between sampled nodes are added back; this densifies connectivity and speeds convergence.

```python
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

def sampler_edge(adj_train, budget_m, deg):
    coo = sp.triu(adj_train, k=1).tocoo()
    rows, cols = coo.row, coo.col
    pe = 1.0 / deg[rows] + 1.0 / deg[cols]
    pe = pe / pe.sum()
    pick = np.random.choice(len(rows), size=budget_m, replace=True, p=pe)
    return np.unique(np.concatenate([rows[pick], cols[pick]]))

def sampler_rw(adj_train, num_roots, walk_len):
    nodes = set(np.random.randint(adj_train.shape[0], size=num_roots))
    for r in list(nodes):
        u = r
        for _ in range(walk_len):
            nbrs = adj_train.indices[adj_train.indptr[u]:adj_train.indptr[u + 1]]
            if len(nbrs) == 0:
                break
            u = np.random.choice(nbrs)
            nodes.add(u)
    return np.array(sorted(nodes))

def induce_with_edge_ids(adj_train, sub):
    sub = np.array(sorted(np.unique(sub)))
    pos = {v: i for i, v in enumerate(sub)}
    indptr, indices, edge_ids = [0], [], []
    for v in sub:
        for eid in range(adj_train.indptr[v], adj_train.indptr[v + 1]):
            u = adj_train.indices[eid]
            if u in pos:
                indices.append(pos[u])
                edge_ids.append(eid)
        indptr.append(len(indices))
    return sp.csr_matrix((np.ones(len(indices)), indices, indptr), shape=(len(sub), len(sub))), np.array(edge_ids)

def estimate_norms(adj_train, sampler, train_nodes, coverage=50):
    n = adj_train.shape[0]
    num_train = len(train_nodes)
    C_v = np.zeros(n)
    C_uv = np.zeros(adj_train.nnz)
    subgraphs = []
    total = 0
    while total <= coverage * num_train:
        sub = sampler(adj_train)
        subgraphs.append(sub)
        total += len(sub)
        C_v[sub] += 1
        _, edge_ids = induce_with_edge_ids(adj_train, sub)
        C_uv[edge_ids] += 1
    N = len(subgraphs)
    rows = np.repeat(np.arange(n), np.diff(adj_train.indptr))
    aggr_norm = np.divide(C_v[rows], C_uv, out=np.full_like(C_uv, 0.1), where=C_uv > 0)
    aggr_norm = np.clip(aggr_norm, 0, 1e4)
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
            h = F.relu(torch.sparse.mm(adj_norm, W(h)))
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
        sub = subgraphs[i]
        i += 1
        sub_adj, edge_ids = induce_with_edge_ids(adj_train, sub)
        sub_adj.data = aggr_norm[edge_ids]
        adj_norm = sp.diags(1.0 / np.clip(deg_train[sub], 1, None)).dot(sub_adj)
        adj_norm = to_torch_sparse(adj_norm)
        logits = model(adj_norm, features[sub])
        per_node = loss_fn(logits, labels[sub], reduction='none')
        loss = (per_node * loss_norm[sub]).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
```
