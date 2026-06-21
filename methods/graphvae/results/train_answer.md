The problem is to build a deep generative model that maps a continuous latent code to a graph, the way VAEs and GANs already did for images and text. Graphs resist this direct recipe for two reasons. First, they are discrete: an adjacency matrix is 0/1, labels are categorical, and sampling a discrete graph during training blocks backpropagation. Second, and more fundamentally, a graph has no canonical ordering of its nodes. The same graph can be written as n! different adjacency matrices, so an entrywise reconstruction loss is not permutation-invariant and will punish a correct graph just because it appears in a different node order. Existing workarounds both have serious costs. SMILES-string VAEs and grammar VAEs linearize the molecule through a brittle character or grammar serialization, so the model never sees the actual graph and produces many syntactically or chemically invalid strings. Autoregressive graph construction keeps the object a graph but forces an arbitrary node construction order, which makes the likelihood order-dependent and still leaves the ordering problem unsolved.

I propose GraphVAE, a variational autoencoder that decodes a continuous latent code into a whole small graph at once. The core move is to restrict the domain to graphs on at most k nodes, with k small enough that a dense k-by-k representation is tractable, and to have the decoder emit a probabilistic fully-connected graph on k slots rather than a discrete one. Node existence is encoded on the diagonal of a predicted adjacency matrix A~, edge existence on the off-diagonals, and node or edge labels through separate categorical heads. Because everything is emitted as probabilities, nothing discrete is sampled during training and the whole decoder is differentiable. Because the whole graph is emitted in one shot, there is no construction order to choose. The only remaining obstacle is comparing the k predicted slots to the n ground-truth nodes when the two sets have no shared coordinate frame.

GraphVAE solves this alignment problem with graph matching. It defines a pairwise similarity S between ground-truth node pairs and predicted slot pairs that mixes label compatibility with existential compatibility: a match is good when labels agree, the true edge exists, and the predicted graph believes the relevant nodes and edges exist. This similarity is fed into a max-pooling matching power iteration, which solves the relaxed graph-matching quadratic program by repeatedly updating assignment scores while max-pooling over candidate neighbors to stay robust to noisy early similarities. The continuous assignment is then discretized to a strict one-to-one matching X using the Hungarian algorithm. X is treated as a fixed constant for the loss, so the matching step itself does not need to be differentiable; gradients flow only through the reconstruction terms computed after alignment.

With X in hand, the model maps the ground-truth graph into the predicted frame by A' = X A X^T and maps predicted labels back into the ground-truth frame. The reconstruction loss is then a sum of Bernoulli cross-entropies for node and edge existence and categorical cross-entropies for node and edge labels, with nodes and edges averaged separately so the O(k^2) edge terms do not drown out the O(k) node terms. The encoder is a standard graph convolutional network with a permutation-invariant graph-level readout that emits the VAE mean and log-variance. The full objective is the usual evidence lower bound: the expected reconstruction negative log-likelihood plus the closed-form Gaussian KL between the approximate posterior and the standard normal prior, trained with the reparameterization trick. Molecules can be handled with small adjustments such as predicting only the upper triangle to enforce undirected symmetry and optionally building a maximum spanning tree at test time to encourage connectivity.

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
