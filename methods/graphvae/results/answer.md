# GraphVAE, distilled

GraphVAE is a variational autoencoder that **decodes a continuous latent code into a whole
small graph at once**. Instead of constructing a graph node-by-node (which needs an arbitrary
ordering) or through a string surrogate (which loses structure and emits invalid outputs), the
decoder emits a *probabilistic, fully-connected graph of fixed maximum size `k`*: node and edge
existence as Bernoulli probabilities, node and edge labels as categoricals. Because there is no
canonical node ordering, the reconstruction likelihood is computed only after **aligning** the
`k` predicted slots to the `n <= k` ground-truth nodes via approximate **graph matching**.

## Problem it solves

Unconditional (and conditional) generation of small graphs -- train a model that maps `z ~ N(0,I)`
to graphs resembling a training distribution, end to end by gradient descent. Two graph-specific
obstacles block the naive VAE: graphs are **discrete** (sampling a graph during training is
non-differentiable) and have **no canonical node order** (the adjacency-matrix reconstruction
loss is not permutation-invariant, so it punishes a correct graph produced in a different order).

## Key idea

1. **One-shot probabilistic graph at fixed size `k`.** Restrict to graphs on at most `k` nodes
   (`k` small, ~tens) so a dense representation is tractable, and output the entire graph at once
   as probabilities -- no construction order, and nothing discrete is sampled during training.
   - `A~ in [0,1]^{k x k}`: diagonal `A~_{a,a}` = node-existence prob, off-diagonal `A~_{a,b}` =
     edge-existence prob (sigmoid).
   - `E~ in R^{k x k x d_e}`: edge-class probs (edgewise softmax).
   - `F~ in R^{k x d_n}`: node-class probs (nodewise softmax).
   - Decoder is a deterministic MLP with three heads. At test time, argmax to a discrete graph
     (possibly on fewer than `k` nodes).
2. **Graph matching makes the loss permutation-aware.** Find a strict one-to-one assignment
   `X in {0,1}^{k x n}` between predicted slots and ground-truth nodes by maximizing a pairwise
   similarity `S`, then score the reconstruction in the aligned frame. `X` is treated as a fixed
   constant, so the loss stays differentiable even though the matching solver is not.
3. **VAE wrapper.** Gaussian prior `N(0,I)`, reparameterized Gaussian posterior, closed-form
   Gaussian KL; edge-conditioned graph-convolution encoder with a gated permutation-invariant
   graph-level readout into `(mu, log sigma^2)`.

## Similarity, matching, loss

**Pairwise similarity** (feature compatibility AND existential compatibility -- the latter makes
assignments stable during training):

```
S((i,j),(a,b)) = (E_{i,j,.}^T E~_{a,b,.}) A_{i,j} A~_{a,b} A~_{a,a} A~_{b,b}   [i != j & a != b]
               + (F_{i,.}^T F~_{a,.})    A~_{a,a}                              [i = j & a = b]
```

**Max-pooling matching (Cho et al. 2014)** solves the relaxed QP `x* = argmax x^T S x` by power
iteration `x^{(t+1)} = S x^{(t)} / ||S x^{(t)}||_2`, with the matrix-vector product made robust
by max-pooling over candidate partners instead of summing:

```
x_{ia} <- x_{ia} S_{ia;ia} + sum_{j in N_i} max_{b in N_a} x_{jb} S_{ia;jb}
```

(batch by zero-padding `S` for indices beyond the node count and running a fixed iteration
count). The continuous `X*` is rounded to a hard `X` by the **Hungarian algorithm**
(`linear_sum_assignment`) so each ground-truth node contributes reconstruction error through one
predicted slot instead of being smeared across many slots.

**Reconstruction log-likelihood** (`A' = X A X^T`, `F~' = X^T F~`, `E~'_{.,.,l} = X^T E~_{.,.,l} X`),
averaging nodes and edges separately so edges (`O(k^2)`) don't dominate nodes (`O(k)`):

```
log p(A'|z) = (1/k) sum_a       [A'_aa log A~_aa + (1-A'_aa) log(1-A~_aa)]
            + (1/(k(k-1))) sum_{a!=b} [A'_ab log A~_ab + (1-A'_ab) log(1-A~_ab)]
log p(F|z)  = (1/n)            sum_i     log F_{i,.}^T F~'_{i,.}
log p(E|z)  = (1/(||A||_1 - n)) sum_{i!=j} log E_{i,j,.}^T E~'_{i,j,.}
-log p(G|z) = -lambda_A log p(A'|z) - lambda_F log p(F|z) - lambda_E log p(E|z)
```

**Full objective** (minimize):

```
L(phi, theta; G) = E_{q_phi(z|G)}[ -log p_theta(G|z) ] + KL[ q_phi(z|G) || p(z) ],
KL = -1/2 sum_j (1 + log sigma_j^2 - mu_j^2 - sigma_j^2),   z = mu + sigma (.) eps.
```

## Variants and remedies

- **Conditional / disentangled** (Sohn et al. 2015): condition encoder and decoder on a label `y`
  (decoder fed `[z; y]`; `y` concatenated to node features before pooling). Small latent `c` forces
  the decoder to use the label.
- **Molecule remedies**: predict only the upper triangle (symmetric undirected `A~`, `E~`); at test
  time build a maximum spanning tree over probable nodes `{a : A~_{a,a} >= 0.5}` to force
  connectivity; don't generate hydrogens (added as padding at the validity check).
- **Implicit node probabilities**: tie node existence to edges, `A~_{a,a} = max_b A~_{a,b}`, to
  enforce a connectivity-flavored constraint directly in the probabilistic graph.

## Complexity / scope

Parameters and memory `O(k^2)` (dense output); matching `O(k^4)` (`S` is indexed by pairs of
pairs). A small-graph method -- practical up to a few tens of nodes.

## Working code

(Structure-only core: node/edge existence with degree as the matching feature. Categorical label
terms follow the same alignment step with softmax heads.)

```python
import numpy as np
import scipy.optimize
import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphConv(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(input_dim, output_dim))
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain("relu"))

    def forward(self, x, adj):
        return torch.matmul(torch.matmul(adj, x), self.weight)


class MLP_VAE_plain(nn.Module):
    def __init__(self, h_size, embedding_size, y_size):
        super().__init__()
        self.encode_11 = nn.Linear(h_size, embedding_size)  # mu
        self.encode_12 = nn.Linear(h_size, embedding_size)  # log sigma^2
        self.decode_1 = nn.Linear(embedding_size, embedding_size)
        self.decode_2 = nn.Linear(embedding_size, y_size)

    def forward(self, h):
        z_mu = self.encode_11(h)
        z_lsgms = self.encode_12(h)
        z_sgm = torch.exp(0.5 * z_lsgms)
        z = z_mu + torch.randn_like(z_sgm) * z_sgm
        h_decode = self.decode_2(F.relu(self.decode_1(z)))
        return h_decode, z_mu, z_lsgms


class GraphVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, max_num_nodes,
                 pool="sum", mpm_iters=50):
        super().__init__()
        self.max_num_nodes = max_num_nodes
        self.latent_dim = latent_dim
        self.mpm_iters = mpm_iters
        self.pool = pool

        self.conv1 = GraphConv(input_dim, hidden_dim)
        self.conv2 = GraphConv(hidden_dim, hidden_dim)
        output_dim = max_num_nodes * (max_num_nodes + 1) // 2
        self.vae = MLP_VAE_plain(hidden_dim, latent_dim, output_dim)

    def recover_adj_lower(self, l):
        adj = torch.zeros(self.max_num_nodes, self.max_num_nodes,
                          dtype=l.dtype, device=l.device)
        mask = torch.triu(torch.ones_like(adj, dtype=torch.bool))
        adj[mask] = l
        return adj

    def recover_full_adj_from_lower(self, lower):
        diag = torch.diag(torch.diag(lower))
        return lower + lower.t() - diag

    def edge_similarity_matrix(self, adj, adj_recon, matching_features,
                               matching_features_recon, sim_func):
        k = self.max_num_nodes
        S = torch.zeros(k, k, k, k, dtype=adj_recon.dtype, device=adj_recon.device)
        for i in range(k):
            for j in range(k):
                if i == j:
                    for a in range(k):
                        S[i, i, a, a] = adj[i, i] * adj_recon[a, a] * \
                            sim_func(matching_features[i], matching_features_recon[a])
                else:
                    for a in range(k):
                        for b in range(k):
                            if a == b:
                                continue
                            S[i, j, a, b] = adj[i, j] * adj[i, i] * adj[j, j] * \
                                             adj_recon[a, b] * adj_recon[a, a] * adj_recon[b, b]
        return S

    def mpm(self, x_init, S, max_iters=None):
        x = x_init
        max_iters = self.mpm_iters if max_iters is None else max_iters
        for _ in range(max_iters):
            x_new = torch.zeros_like(x)
            for i in range(self.max_num_nodes):
                for a in range(self.max_num_nodes):
                    x_new[i, a] = x[i, a] * S[i, i, a, a]
                    x_new[i, a] += sum(torch.max(x[j, :] * S[i, j, a, :])
                                       for j in range(self.max_num_nodes) if j != i)
            x = x_new / torch.norm(x_new).clamp_min(1e-12)
        return x

    def deg_feature_similarity(self, f1, f2):
        return 1.0 / (torch.abs(f1 - f2) + 1.0)

    def permute_adj(self, adj, curr_ind, target_ind):
        perm = np.zeros(self.max_num_nodes, dtype=np.int64)
        perm[target_ind] = curr_ind
        perm = torch.as_tensor(perm, dtype=torch.long, device=adj.device)
        return adj.index_select(0, perm).index_select(1, perm)

    def pool_graph(self, x):
        if self.pool == "max":
            return torch.max(x, dim=1).values
        return torch.sum(x, dim=1)

    def _with_node_diagonal(self, adj, node_count):
        A = adj.clone()
        idx = torch.arange(self.max_num_nodes, device=A.device)
        A[idx, idx] = (idx < node_count).to(A.dtype)
        return A

    def adj_recon_loss(self, adj_truth, adj_pred):
        return F.binary_cross_entropy(adj_pred.clamp(1e-6, 1 - 1e-6), adj_truth)

    def forward(self, input_features, adj, node_counts):
        x = F.relu(self.conv1(input_features, adj))
        x = F.relu(self.conv2(x, adj))
        graph_h = self.pool_graph(x)

        h_decode, z_mu, z_lsgms = self.vae(graph_h)
        out = torch.sigmoid(h_decode)

        recon_loss = out.new_tensor(0.0)
        tri_mask = torch.triu(torch.ones(self.max_num_nodes, self.max_num_nodes,
                                         dtype=torch.bool, device=out.device))
        for s in range(adj.size(0)):
            recon_adj_lower = self.recover_adj_lower(out[s])
            recon_adj = self.recover_full_adj_from_lower(recon_adj_lower)

            adj_data = self._with_node_diagonal(adj[s], node_counts[s]).detach()
            adj_features = torch.sum(adj_data, dim=1)
            recon_features = torch.sum(recon_adj.detach(), dim=1)
            S = self.edge_similarity_matrix(adj_data.detach(), recon_adj.detach(),
                                            adj_features, recon_features,
                                            self.deg_feature_similarity)

            init_assignment = torch.ones(self.max_num_nodes, self.max_num_nodes,
                                         device=out.device) / self.max_num_nodes
            assignment = self.mpm(init_assignment, S)
            row_ind, col_ind = scipy.optimize.linear_sum_assignment(
                -assignment.detach().cpu().numpy())

            adj_permuted = self.permute_adj(adj_data, row_ind, col_ind)
            adj_vectorized = adj_permuted[tri_mask]
            recon_loss = recon_loss + self.adj_recon_loss(adj_vectorized, out[s])

        recon_loss = recon_loss / adj.size(0)
        loss_kl = -0.5 * torch.sum(1 + z_lsgms - z_mu.pow(2) - z_lsgms.exp())
        loss_kl = loss_kl / (self.max_num_nodes * self.max_num_nodes)
        return recon_loss + loss_kl
```
