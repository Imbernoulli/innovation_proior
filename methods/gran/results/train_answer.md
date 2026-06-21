The problem is to learn a distribution p(G) over simple undirected graphs of varying size and sample new graphs that are statistically indistinguishable from the training set. Because a graph has no canonical node labeling, any adjacency matrix A^pi depends on an ordering pi, and the same graph corresponds to as many as N! different matrices. The honest likelihood is therefore p(G) = sum_pi p(A^pi), which is intractable to compute exactly. At the same time, realistic graphs require correlated edge generation, principled handling of this permutation ambiguity, and scalable training and sampling to graphs with hundreds or thousands of nodes. Existing approaches each satisfy only some of these demands.

One-shot latent-variable models such as GraphVAE decode edges independently given a latent and require approximate graph matching to align nodes, which costs roughly O(k^4) and limits them to small graphs. Fully autoregressive models like DeepGMG run a full GNN before every individual node or edge decision, giving strong expressiveness but scaling only to tens of nodes because each decision pays for multi-round message passing. GraphRNN drops the GNN and uses an RNN over breadth-first-search orderings, scaling further but still making O(N^2) sequential decisions and forcing each new edge to depend on the graph only through a recurrent hidden state, which loses direct topological conditioning and creates a long-term dependency bottleneck.

I propose GRAN, the Graph Recurrent Attention Network. GRAN generates the lower-triangular adjacency matrix L^pi one block of B rows at a time, so the number of sequential steps drops from O(N^2) to O(N). This is an exact chain-rule factorization: p(L^pi) = prod_t p(L_{b_t}^pi | L_{<b_t}^pi), where b_t indexes the rows in block t, and the full adjacency is recovered as A^pi = L^pi + L^pi^T. The block size B is an explicit speed-quality knob.

At each step GRAN runs a fresh GNN on the already-generated subgraph augmented with the B new nodes and all candidate edges connecting them. Because the prefix is known during training, every step conditional can be evaluated in parallel, just as in PixelCNN-style teacher forcing. No hidden state is carried across steps, so each decision sees the real graph topology directly and graph-adjacent nodes interact in one message-passing hop regardless of their distance in the generation sequence. Initial node representations are obtained by linearly embedding the existing adjacency rows, while new nodes start from zero and are distinguished by a B-dimensional marker that breaks symmetry and tags existing versus new nodes.

The GNN uses attentive GRU message passing. Messages are computed as m_ij = f(h_i - h_j) with a small MLP f, and attention weights a_ij = sigmoid(g([h_i, x_i] - [h_j, x_j])) with another small MLP g, where x_i is the marker. Messages are aggregated with these gates and fed through a GRU update to keep multi-round propagation stable. To model correlations among the edges generated in the same block without introducing a per-entry sequential model, GRAN emits a mixture of K Bernoulli products. Within each component edges are factorial and parallel, but the shared latent mixture index couples them marginally. The block log-probability is computed stably as logsumexp_k (log alpha_k - sum_e BCE(logit_{k,e}, label_e)).

To handle permutations without enumerating N! orderings, GRAN optimizes the log-sum of p(A^pi) over a small family Q of structure-defined canonical orderings, such as the default ordering, degree-descending, BFS and DFS trees rooted at the maximum-degree node, and a linear-time k-core ordering. This is a valid lower bound on the true log-likelihood and tighter than any single ordering. Variationally it is the ELBO at the optimal categorical posterior q*(pi|G) proportional to p(G,pi), so the model learns to weight orderings by how well they explain each graph. At test time, strided sampling lets one trained model trade quality for speed: train with stride one, then generate a B-block and keep only the first S rows before advancing, with any 1 <= S <= B.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class GNN(nn.Module):
    """Attentive GRU message passing on an augmented graph for one generation step."""

    def __init__(self, msg_dim, node_dim, edge_feat_dim,
                 num_prop=1, num_layer=7, att_hidden_dim=128):
        super().__init__()
        self.num_prop = num_prop
        self.num_layer = num_layer
        self.edge_feat_dim = edge_feat_dim
        self.update_func = nn.ModuleList([
            nn.GRUCell(msg_dim, node_dim) for _ in range(num_layer)])
        self.msg_func = nn.ModuleList([
            nn.Sequential(
                nn.Linear(node_dim + edge_feat_dim, msg_dim),
                nn.ReLU(),
                nn.Linear(msg_dim, msg_dim))
            for _ in range(num_layer)])
        self.att_head = nn.ModuleList([
            nn.Sequential(
                nn.Linear(node_dim + edge_feat_dim, att_hidden_dim),
                nn.ReLU(),
                nn.Linear(att_hidden_dim, msg_dim),
                nn.Sigmoid())
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


class GRAN(nn.Module):
    """Graph Recurrent Attention Network.

    Block-wise autoregressive graph generator with a GNN-attention per-step
    conditional and a mixture-of-Bernoulli output over the block's edges.
    """

    def __init__(self, max_nodes, hidden_dim=128, embedding_dim=128,
                 num_GNN_layers=7, num_GNN_prop=1, num_mix_component=20,
                 num_canonical_order=1, block_size=1, sample_stride=1,
                 att_edge_dim=64, edge_weight=1.0, **kwargs):
        super().__init__()
        self.max_nodes = max_nodes
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.block_size = block_size
        self.sample_stride = sample_stride
        self.num_mix = num_mix_component
        self.num_canonical_order = num_canonical_order
        self.att_edge_dim = att_edge_dim

        self.decoder_input = nn.Linear(max_nodes, embedding_dim)
        self.decoder = GNN(
            msg_dim=hidden_dim,
            node_dim=hidden_dim,
            edge_feat_dim=2 * att_edge_dim,
            num_prop=num_GNN_prop,
            num_layer=num_GNN_layers)

        self.output_theta = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_mix_component))
        self.output_alpha = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_mix_component))

        pos_weight = torch.ones([1]) * edge_weight
        self.adj_loss_func = nn.BCEWithLogitsLoss(
            pos_weight=pos_weight, reduction='none')
        self.num_nodes_pmf = None

    def _augmented_graph(self, A_prefix, jj, K, device):
        """Existing subgraph plus K new nodes with fully-connected candidate edges."""
        adj = F.pad(
            A_prefix[:jj, :jj], (0, K, 0, K), 'constant', value=1.0)
        adj = torch.tril(adj, diagonal=-1)
        adj = adj + adj.transpose(0, 1)
        edges = adj.to_sparse().coalesce().indices().t()
        att_idx = torch.cat([
            torch.zeros(jj).long(),
            torch.arange(1, K + 1)]).to(device).view(-1, 1)
        ef = torch.zeros(edges.shape[0], 2 * self.att_edge_dim, device=device)
        ef = ef.scatter(1, att_idx[edges[:, 0]], 1.0)
        ef = ef.scatter(1, att_idx[edges[:, 1]] + self.att_edge_dim, 1.0)
        return edges, ef

    def _step_logits(self, node_state, edges, edge_feat, idx_row, idx_col):
        h = self.decoder(node_state, edges, edge_feat)
        diff = h[idx_row, :] - h[idx_col, :]
        return self.output_theta(diff), self.output_alpha(diff)

    def _mixture_bernoulli_logp(self, label, log_theta, log_alpha):
        bce = torch.stack([
            self.adj_loss_func(log_theta[:, k], label)
            for k in range(log_theta.shape[1])], dim=1)
        comp_logp = -bce.sum(dim=0)
        log_alpha = F.log_softmax(log_alpha.mean(dim=0), dim=-1)
        return torch.logsumexp(comp_logp + log_alpha, dim=0)

    def _fit_num_nodes_pmf(self, node_counts, N):
        pmf = torch.zeros(N + 1, device=node_counts.device)
        for c in node_counts.long():
            pmf[c] += 1.0
        self.num_nodes_pmf = pmf / pmf.sum().clamp_min(1.0)

    def training_loss(self, adj, node_counts):
        """Negative log-likelihood over canonical orderings."""
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
                for jj in range(0, n - K + 1):
                    edges, ef = self._augmented_graph(L[b, c], jj, K, device)
                    node_state = torch.zeros(
                        jj + K, self.embedding_dim, device=device)
                    if jj > 0:
                        node_state[:jj] = self.decoder_input(
                            L[b, c, :jj, :N])
                    ir, ic = np.meshgrid(
                        np.arange(jj, jj + K), np.arange(jj + K))
                    idx_row = torch.from_numpy(ir.reshape(-1)).long().to(device)
                    idx_col = torch.from_numpy(ic.reshape(-1)).long().to(device)
                    label = adj[b, c, idx_row, idx_col]
                    lt, la = self._step_logits(
                        node_state, edges, ef, idx_row, idx_col)
                    step_logps.append(
                        self._mixture_bernoulli_logp(label, lt, la))
                if step_logps:
                    order_logps.append(torch.stack(step_logps).sum())
            if order_logps:
                graph_logps.append(
                    torch.logsumexp(torch.stack(order_logps), dim=0))
        return -torch.stack(graph_logps).mean()

    @torch.no_grad()
    def sample(self, n_samples, device):
        """Autoregressive row-block sampling."""
        self.eval()
        K, S, N = self.block_size, self.sample_stride, self.max_nodes
        A = torch.zeros(n_samples, N, N, device=device)
        for ii in range(0, N - K + 1, S):
            jj = ii + K
            A[:, ii:, :] = 0.0
            A = torch.tril(A, diagonal=-1)
            for b in range(n_samples):
                edges, ef = self._augmented_graph(A[b], ii, K, device)
                node_state = torch.zeros(
                    ii + K, self.embedding_dim, device=device)
                if ii > 0:
                    node_state[:ii] = self.decoder_input(A[b, :ii, :N])
                ir, ic = np.meshgrid(np.arange(ii, jj), np.arange(jj))
                idx_row = torch.from_numpy(ir.reshape(-1)).long().to(device)
                idx_col = torch.from_numpy(ic.reshape(-1)).long().to(device)
                keep = idx_row > idx_col
                idx_row, idx_col = idx_row[keep], idx_col[keep]
                if idx_row.numel() == 0:
                    continue
                lt, la = self._step_logits(
                    node_state, edges, ef, idx_row, idx_col)
                k = torch.multinomial(
                    F.softmax(la.mean(dim=0), -1), 1).item()
                A[b, idx_row, idx_col] = torch.bernoulli(torch.sigmoid(lt[:, k]))
        A = torch.tril(A, diagonal=-1)
        A = A + A.transpose(1, 2)
        if self.num_nodes_pmf is not None:
            node_counts = torch.multinomial(
                self.num_nodes_pmf, n_samples, replacement=True)
            node_counts = node_counts.clamp(min=2).to(device)
        else:
            node_counts = (A.sum(dim=-1) > 0).long().sum(dim=-1).clamp(min=2)
        return A, node_counts
```
