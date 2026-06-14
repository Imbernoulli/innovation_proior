# GRAN (Graph Recurrent Attention Networks), distilled

GRAN is an efficient auto-regressive deep generative model of graphs. It generates the
lower-triangular adjacency matrix `L^pi` **one block of `B` rows at a time** (so `O(N)`
sequential steps instead of `O(N^2)`). At each step it runs a fresh **GNN with attentive GRU
message passing** over the already-generated subgraph augmented with candidate edges, and emits
the block's edges from a **mixture of Bernoulli** output distribution that captures intra-block
edge correlations cheaply. Node-ordering is handled by maximizing a **log-sum over a small
family of canonical orderings**, which is a valid and tighter lower bound on the true
log-likelihood and is the optimal-posterior ELBO under that family.

## Problem it solves

Learn `p(G)` over simple undirected graphs of varying size and sample new ones, when (a) the
likelihood is a sum over `N!` node orderings, (b) edge generation must be correlated to be
realistic, and (c) the model must scale to graphs with hundreds–thousands of nodes. Prior deep
models give at most two of: expressive (correlated) edges, principled permutation handling, and
scalable/fast generation.

## Key ideas

1. **Block-wise auto-regression over the lower triangle.** With block `b_t = {B(t-1)+1,...,Bt}`
   and `T = ceil(N/B)`,
   ```
   p(L^pi) = prod_{t=1}^{T} p(L_{b_t}^pi | L_{b_1}^pi, ..., L_{b_{t-1}}^pi),    A^pi = L^pi + L^pi^T.
   ```
   Exact chain rule grouped into blocks; `B` trades quality for speed (fewer steps).

2. **A fresh GNN per step, no state carried across steps.** The conditioning subgraph is a known
   prefix at training time, so all step-conditionals are evaluated in parallel (PixelCNN-style),
   while sampling stays sequential. Running a GNN per step makes each decision depend *directly*
   on the existing topology and removes the RNN long-term bottleneck (graph-adjacent nodes are
   one hop apart in the GNN).

3. **Attentive GRU message passing on the augmented graph.** Per round `r`:
   ```
   m_ij^r   = f(h_i^r - h_j^r)                                  (f: 2-layer MLP, ReLU)
   h~_i^r   = [h_i^r, x_i]                                      (x_i: B-dim marker)
   a_ij^r   = Sigmoid( g(h~_i^r - h~_j^r) )                     (g: 2-layer MLP, ReLU)
   h_i^{r+1}= GRU( h_i^r, sum_{j in N(i)} a_ij^r m_ij^r )
   ```
   `x_i = 0` for existing nodes, a one-of-`B` encoding of block-position for new nodes. It tells
   existing from new nodes and **breaks symmetry** among the new nodes (all initialized to
   `h^0 = 0`) so they can connect differently. Initial node rep `h^0 = W L_b + b` is a linear
   dimension reduction of the adjacency row(s).

4. **Mixture-of-Bernoulli output (intra-block edge correlation, fully parallel).**
   ```
   p(L_{b_t}^pi | ...) = sum_{k=1}^{K} alpha_k  prod_{i in b_t} prod_{1<=j<=i} theta_{k,i,j}
   alpha_1..alpha_K     = Softmax( sum_{i in b_t, 1<=j<=i} MLP_alpha(h_i^R - h_j^R) )
   theta_{1,i,j},...,theta_{K,i,j} = Sigmoid( MLP_theta(h_i^R - h_j^R) )
   ```
   In the product, `theta_{k,i,j}` denotes the probability assigned by component `k` to the
   realized binary edge slot; equivalently, with raw Bernoulli parameter `rho_{k,i,j}`, the factor
   is `rho_{k,i,j}^{y_{i,j}} (1-rho_{k,i,j})^{1-y_{i,j}}`.
   Within a component edges are factorial (parallel); the shared latent index `k` couples them
   marginally. `K=1` degenerates to independent Bernoulli.

5. **Canonical-ordering objective (the permutation handle).** With a structure-defined family
   `Q = {pi_1,...,pi_M}` (no two giving the same `A^pi`),
   ```
   log p(G) >= log sum_{pi in Q} p(A^pi),
   ```
   valid (a strict subset of the `N!` orderings; nonnegative terms) and tighter than any single
   `log p(G,pi)`. Variationally it is the ELBO `log p(G) >= E_{q(pi|G)}[log p(G,pi)] + H(q)` at the
   Lagrangian-optimal posterior
   ```
   q*(pi|G) = p(G,pi) / sum_{pi' in Q~} p(G,pi'),
   ```
   so optimizing it implicitly picks the best (soft combination of) orderings per graph.
   Orderings used: default, degree-descending, BFS-tree and DFS-tree rooted at the max-degree
   node, and a **k-core descending** ordering (k-core = maximal subgraph with all degrees >= k;
   decomposition in `O(|E|)`; partition by largest core number, cores descending, degree-descending
   within a core).

6. **Strided sampling (train once, dial quality at test).** Train with block `B`, stride 1
   (learn next-`B`-rows from every sub-prefix). At test use any stride `1<=S<=B`: generate a
   `B`-block, keep the first `S` rows, advance by `S`. `S=B` fastest (`T=ceil(N/B)`); `S<B`
   overlaps blocks by `B-S`, modeling intra-block dependency over more steps (`T=floor((N-B)/S)+1`),
   no retraining.

## Derivation of the optimal ordering posterior (Lagrange)

Maximize the ELBO over a categorical `q` subject to `sum_pi q(pi)=1`:
```
J = sum_pi q(pi)[log p(G,pi) - log q(pi)] + lambda(sum_pi q(pi) - 1)
dJ/dq(pi) = log p(G,pi) - log q(pi) - 1 + lambda = 0  =>  q(pi) ∝ p(G,pi).
```
Normalizing gives `q*`. Substituting back, the bracket equals the constant
`log sum_{pi'} p(G,pi')`, so the ELBO collapses to
`log sum_{pi in Q~} p(G,pi) = log sum_{pi in Q} p(A^pi)` — the objective.

## Training loss (stable mixture-of-Bernoulli likelihood)

Per block-step (subgraph) and component `k`, the negative of binary cross-entropy summed over the
block's candidate edge slots is `sum_e log p_k(label_e)`. Combine components stably:
```
log p(block) = logsumexp_k ( log_softmax(alpha)_k + sum_e [ -BCE(logit_{k,e}, label_e) ] ),
```
sum block log-probabilities for each ordering, then `logsumexp` over the `C` orderings in `Q` to
get `log sum_{pi in Q} p(A^pi)`. The exact negative log-probability is the negative of this
quantity, with the canonical code multiplying by `2` when reporting full-graph probability from
lower-triangular edges. The optimized training loss uses the same edge-to-subgraph-to-ordering
reductions after normalizing each ordering's accumulated log-probability by its edge-slot count for
stability. The implementation averages the per-edge mixture-weight logits before `log_softmax`,
which keeps one block-level mixture vector while normalizing its scale across different numbers of
edge slots.

## Defaults

Canonical configs use `7` GNN layers, `num_prop = 1`, `K = 20` mixture components, block size and
sample stride `1`, Adam with learning rate `1e-4`, `dimension_reduce = true`, and attention on.
The hidden/embedding width is dataset-configured (`128` for grid/lobster, `512` for DD, `256` for
FIRSTMM-DB). The runner owns the optimizer; gradient clipping is not active in the canonical
training loop.

## Working code

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class GNN(nn.Module):
    """Attentive GRU message passing on the augmented graph (one generation step).
       m_ij = f(h_i - h_j); a_ij = sigmoid(g([h_i,x_i] - [h_j,x_j]));
       the implementation carries x_i/x_j as an edge feature;
       h_i <- GRU(h_i, sum_j a_ij m_ij). No state carried across steps."""

    def __init__(self, msg_dim, node_dim, edge_feat_dim,
                 num_prop=1, num_layer=7, att_hidden_dim=128):
        super().__init__()
        self.num_prop, self.num_layer, self.edge_feat_dim = num_prop, num_layer, edge_feat_dim
        self.update_func = nn.ModuleList([
            nn.GRUCell(msg_dim, node_dim) for _ in range(num_layer)])
        self.msg_func = nn.ModuleList([
            nn.Sequential(nn.Linear(node_dim + edge_feat_dim, msg_dim),
                          nn.ReLU(), nn.Linear(msg_dim, msg_dim)) for _ in range(num_layer)])
        self.att_head = nn.ModuleList([
            nn.Sequential(nn.Linear(node_dim + edge_feat_dim, att_hidden_dim),
                          nn.ReLU(), nn.Linear(att_hidden_dim, msg_dim), nn.Sigmoid())
            for _ in range(num_layer)])

    def _prop(self, state, edge, edge_feat, layer):
        diff = state[edge[:, 0], :] - state[edge[:, 1], :]
        ei = torch.cat([diff, edge_feat], dim=1) if self.edge_feat_dim > 0 else diff
        msg = self.msg_func[layer](ei) * self.att_head[layer](ei)
        agg = torch.zeros(state.shape[0], msg.shape[1], device=state.device)
        agg = agg.scatter_add(0, edge[:, [1]].expand(-1, msg.shape[1]), msg)
        return self.update_func[layer](agg, state)

    def forward(self, node_feat, edge, edge_feat):
        state = node_feat
        for ii in range(self.num_layer):
            if ii > 0:
                state = F.relu(state)
            for _ in range(self.num_prop):
                state = self._prop(state, edge, edge_feat, ii)
        return state


class GRANMixtureBernoulli(nn.Module):
    """Block-wise auto-regressive graph generator with GNN-attention per-step
       conditional and a mixture-of-Bernoulli output over the block's edges."""

    def __init__(self, max_nodes, hidden_dim=128, embedding_dim=128,
                 num_GNN_layers=7, num_GNN_prop=1, num_mix_component=20,
                 num_canonical_order=1, block_size=1, sample_stride=1,
                 att_edge_dim=64, edge_weight=1.0, **kwargs):
        super().__init__()
        self.max_nodes = max_nodes
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.block_size = block_size            # B
        self.sample_stride = sample_stride      # S (<= B)
        self.num_mix = num_mix_component        # K
        self.num_canonical_order = num_canonical_order
        self.att_edge_dim = att_edge_dim
        self.decoder_input = nn.Linear(max_nodes, embedding_dim)   # h^0 = W L_b + b
        self.decoder = GNN(hidden_dim, hidden_dim, 2 * att_edge_dim,
                           num_prop=num_GNN_prop, num_layer=num_GNN_layers)
        self.output_theta = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_mix_component))
        self.output_alpha = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_mix_component))
        pos_weight = torch.ones([1]) * edge_weight
        self.adj_loss_func = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='none')
        self.num_nodes_pmf = None

    def _augmented_graph(self, A_prefix, jj, K, device):
        adj = F.pad(A_prefix[:jj, :jj], (0, K, 0, K), 'constant', value=1.0)  # candidates = 1
        adj = torch.tril(adj, diagonal=-1)
        adj = adj + adj.transpose(0, 1)
        edges = adj.to_sparse().coalesce().indices().t()
        att_idx = torch.cat([torch.zeros(jj).long(),
                             torch.arange(1, K + 1)]).to(device).view(-1, 1)
        ef = torch.zeros(edges.shape[0], 2 * self.att_edge_dim, device=device)
        ef = ef.scatter(1, att_idx[edges[:, 0]], 1.0)
        ef = ef.scatter(1, att_idx[edges[:, 1]] + self.att_edge_dim, 1.0)
        return edges, ef

    def _step_logits(self, node_state, edges, ef, idx_row, idx_col):
        h = self.decoder(node_state, edges, ef)
        diff = h[idx_row, :] - h[idx_col, :]
        return self.output_theta(diff), self.output_alpha(diff)

    def _mixture_bernoulli_logp(self, label, log_theta, log_alpha):
        bce = torch.stack([self.adj_loss_func(log_theta[:, k], label)
                           for k in range(log_theta.shape[1])], dim=1)
        comp_logp = -bce.sum(dim=0)                                # sum_e log p_k(edge_e)
        log_alpha = F.log_softmax(log_alpha.mean(dim=0), dim=-1)
        return torch.logsumexp(comp_logp + log_alpha, dim=0)       # log sum_k alpha_k prod_e p_k(label_e)

    def _fit_num_nodes_pmf(self, node_counts, N):
        pmf = torch.zeros(N + 1, device=node_counts.device)
        for c in node_counts.long():
            pmf[c] += 1.0
        self.num_nodes_pmf = pmf / pmf.sum().clamp_min(1.0)

    def training_loss(self, adj, node_counts):
        self.train()
        device = adj.device
        B, C, N, _ = adj.shape
        K = self.block_size
        if self.num_nodes_pmf is None:
            self._fit_num_nodes_pmf(node_counts, N)
        L = torch.tril(adj, diagonal=-1)
        graph_logps = []
        for b in range(B):
            n = int(node_counts[b].item())
            order_logps = []
            for c in range(C):
                step_logps = []
                for jj in range(0, n - K + 1):                    # stride-1 teacher forcing
                    edges, ef = self._augmented_graph(L[b, c], jj, K, device)
                    node_state = torch.zeros(jj + K, self.embedding_dim, device=device)
                    if jj > 0:
                        node_state[:jj] = self.decoder_input(L[b, c, :jj, :N])
                    ir, ic = np.meshgrid(np.arange(jj, jj + K), np.arange(jj + K))
                    idx_row = torch.from_numpy(ir.reshape(-1)).long().to(device)
                    idx_col = torch.from_numpy(ic.reshape(-1)).long().to(device)
                    label = adj[b, c, idx_row, idx_col]
                    lt, la = self._step_logits(node_state, edges, ef, idx_row, idx_col)
                    step_logps.append(self._mixture_bernoulli_logp(label, lt, la))
                if step_logps:
                    order_logps.append(torch.stack(step_logps).sum())
            if order_logps:
                graph_logps.append(torch.logsumexp(torch.stack(order_logps), dim=0))
        return -torch.stack(graph_logps).mean()

    @torch.no_grad()
    def sample(self, n_samples, device):
        self.eval()
        K, S, N = self.block_size, self.sample_stride, self.max_nodes
        A = torch.zeros(n_samples, N, N, device=device)
        for ii in range(0, N - K + 1, S):
            jj = ii + K
            A[:, ii:, :] = 0.0
            A = torch.tril(A, diagonal=-1)
            for b in range(n_samples):
                edges, ef = self._augmented_graph(A[b], ii, K, device)
                node_state = torch.zeros(ii + K, self.embedding_dim, device=device)
                if ii > 0:
                    node_state[:ii] = self.decoder_input(A[b, :ii, :N])
                ir, ic = np.meshgrid(np.arange(ii, jj), np.arange(jj))
                idx_row = torch.from_numpy(ir.reshape(-1)).long().to(device)
                idx_col = torch.from_numpy(ic.reshape(-1)).long().to(device)
                keep = idx_row > idx_col
                idx_row, idx_col = idx_row[keep], idx_col[keep]
                if idx_row.numel() == 0:
                    continue
                lt, la = self._step_logits(node_state, edges, ef, idx_row, idx_col)
                k = torch.multinomial(F.softmax(la.mean(dim=0), -1), 1).item()
                A[b, idx_row, idx_col] = torch.bernoulli(torch.sigmoid(lt[:, k]))
        A = torch.tril(A, diagonal=-1)
        A = A + A.transpose(1, 2)                                  # A = L + L^T
        if self.num_nodes_pmf is not None:
            node_counts = torch.multinomial(self.num_nodes_pmf, n_samples, replacement=True)
            node_counts = node_counts.clamp(min=2).to(device)
        else:
            node_counts = (A.sum(dim=-1) > 0).long().sum(dim=-1).clamp(min=2)
        return A, node_counts


def mixture_bernoulli_loss(label, log_theta, log_alpha, adj_loss_func,
                           subgraph_idx, subgraph_idx_base, num_canonical_order,
                           sum_order_log_prob=False, return_neg_log_prob=False,
                           reduction="mean"):
    """Batched canonical loss: edge slots -> subgraphs -> canonical orderings -> graphs."""
    num_subgraph = int(subgraph_idx_base[-1].item())
    batch_size = subgraph_idx_base.shape[0] - 1
    num_order = num_canonical_order
    num_mix = log_theta.shape[1]

    edge_bce = torch.stack(
        [adj_loss_func(log_theta[:, k], label) for k in range(num_mix)], dim=1)

    edge_count = torch.zeros(num_subgraph, device=label.device)
    edge_count = edge_count.scatter_add(
        0, subgraph_idx, torch.ones_like(subgraph_idx).float())

    subgraph_bce = torch.zeros(num_subgraph, num_mix, device=label.device)
    subgraph_bce = subgraph_bce.scatter_add(
        0, subgraph_idx.unsqueeze(1).expand(-1, num_mix), edge_bce)

    subgraph_alpha = torch.zeros(num_subgraph, num_mix, device=label.device)
    subgraph_alpha = subgraph_alpha.scatter_add(
        0, subgraph_idx.unsqueeze(1).expand(-1, num_mix), log_alpha)
    subgraph_alpha = F.log_softmax(subgraph_alpha / edge_count.view(-1, 1), dim=-1)

    subgraph_logp = torch.logsumexp(-subgraph_bce + subgraph_alpha, dim=1)

    order_logp = torch.zeros(batch_size * num_order, device=label.device)
    order_count = torch.zeros(batch_size * num_order, device=label.device)
    subgraphs_per_graph = ((subgraph_idx_base[1:] - subgraph_idx_base[:-1]) // num_order).to(label.device)
    order_idx = torch.repeat_interleave(
        torch.arange(batch_size * num_order, device=label.device),
        torch.repeat_interleave(subgraphs_per_graph, num_order))
    order_logp = order_logp.scatter_add(0, order_idx, subgraph_logp)
    order_count = order_count.scatter_add(0, order_idx, edge_count)

    order_logp = order_logp.reshape(batch_size, num_order)
    normalized_order_logp = (order_logp.reshape(-1) / order_count).reshape(batch_size, num_order)
    if sum_order_log_prob:
        graph_logp = torch.sum(order_logp, dim=1)
        graph_loss_logp = torch.sum(normalized_order_logp, dim=1)
    else:
        graph_logp = torch.logsumexp(order_logp, dim=1)
        graph_loss_logp = torch.logsumexp(normalized_order_logp, dim=1)

    neg_log_prob = -2 * graph_logp
    loss = -graph_loss_logp
    if reduction == "mean":
        loss, neg_log_prob = loss.mean(), neg_log_prob.mean()
    elif reduction == "sum":
        loss, neg_log_prob = loss.sum(), neg_log_prob.sum()
    else:
        assert reduction == "none"
    return (loss, neg_log_prob) if return_neg_log_prob else loss
```

For efficient training, all subgraphs and orderings are batched into a single large graph, and
sampling caches node states. The structure above (dimension-reduce embedding -> attentive GRU GNN
with the existing/new marker as an edge feature -> mixture-of-Bernoulli head with a scatter-add
logsumexp likelihood -> strided autoregressive sampling) is the same model and loss.
