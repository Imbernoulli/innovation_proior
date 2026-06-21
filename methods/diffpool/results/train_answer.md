Graph neural networks are built for node-level representations: after several rounds of message passing, every node has a vector summarizing its local neighborhood. Graph classification, however, needs one fixed-length vector for an entire graph of arbitrary size. The usual readouts—summing or averaging all node embeddings, adding a virtual node, or running a set network over the nodes—are flat. They collapse every node in a single step and never represent the intermediate scales that real graphs possess, such as atoms grouping into functional groups or users grouping into communities. The label often lives at one of those intermediate scales, so a flat reduction throws away information before the classifier ever sees it.

The obvious analogue is CNN spatial pooling: interleave local computation with downsampling so deeper layers see coarser, more global structure. But graphs have no grid, no canonical patches, and no node ordering that preserves structure, so the CNN recipe does not transfer directly. Precomputed clustering algorithms can coarsen a graph, yet they are task-agnostic and decoupled from training, so they cannot learn which groupings help the classification objective. What is needed is a differentiable, learned pooling operator that can be stacked with message-passing layers and trained end-to-end.

The method is DiffPool. It treats graph pooling as soft clustering. At each layer, a pooling GNN produces an assignment score for every node over a fixed number of clusters; a row-wise softmax turns these scores into a soft assignment matrix S. Given node embeddings Z and adjacency A, the coarsened graph is computed as X' = Sᵀ Z for the new node features and A' = Sᵀ A S for the new weighted adjacency. Because the assignment is soft, gradients flow through the entire coarsening step. The construction is permutation invariant: relabeling the original nodes by a permutation P sends Z to PZ and S to PS, and both X' and A' are unchanged because PᵀP = I.

DiffPool uses two separate GNNs at each layer. One GNN, the embedding network, produces the content Z that will be carried into the coarser graph. The other, the pooling network, produces the assignment S that decides how nodes are grouped. Keeping these roles separate prevents the gradient for representation quality from fighting the gradient for cluster structure. These blocks are stacked: the output of one embed-pool step becomes the input to the next, coarsening the graph layer by layer. At the final layer, all remaining coarse nodes are pooled into a single graph vector, which is passed to an MLP classifier.

Training the assignment network from classification loss alone is unreliable: soft clustering is a non-convex, factorization-like problem with many poor local minima, and the task gradient must travel through all subsequent layers before reaching the assignment. DiffPool therefore adds two auxiliary losses at every pooling layer. A link-prediction loss, ||A − S Sᵀ||_F, encourages connected nodes to share clusters, since (S Sᵀ)_ij is the soft probability that nodes i and j are co-clustered. An entropy loss, minimized over the rows of S, pushes each node's assignment toward one-hot so clusters are crisp rather than diffuse. The total loss is cross-entropy plus the sum of these auxiliary terms across layers.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

def diff_pool(x, adj, s, normalize=True):
    # x: [B, n, d] embeddings; adj: [B, n, n]; s: [B, n, m] raw assignment scores
    s = torch.softmax(s, dim=-1)                                  # row-stochastic S
    x_next = torch.matmul(s.transpose(1, 2), x)                   # X' = S^T Z  -> [B, m, d]
    adj_next = torch.matmul(torch.matmul(s.transpose(1, 2), adj), s)  # A' = S^T A S
    link_loss = torch.norm(adj - torch.matmul(s, s.transpose(1, 2)), p=2)
    if normalize:
        link_loss = link_loss / adj.numel()
    ent_loss = (-s * torch.log(s + 1e-15)).sum(dim=-1).mean()
    return x_next, adj_next, link_loss, ent_loss

class GNN(nn.Module):
    """Two-layer GCN-style message passing on dense (adj, x)."""
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.w1 = nn.Linear(in_dim, hidden_dim)
        self.w2 = nn.Linear(hidden_dim, out_dim)
        self.bn = nn.BatchNorm1d(hidden_dim)

    def norm_adj(self, adj):
        adj = adj + torch.eye(adj.size(-1), device=adj.device)
        dinv = adj.sum(-1, keepdim=True).clamp(min=1).pow(-0.5)
        return dinv * adj * dinv.transpose(1, 2)

    def forward(self, adj, x):
        a = self.norm_adj(adj)
        h = F.relu(torch.matmul(a, self.w1(x)))
        b, n, c = h.shape
        h = self.bn(h.reshape(b * n, c)).reshape(b, n, c)
        h = torch.matmul(a, self.w2(h))
        return F.normalize(h, p=2, dim=-1)

class DiffPoolNet(nn.Module):
    def __init__(self, in_dim, hidden, num_classes, max_nodes, assign_ratio=0.25, num_pool=1):
        super().__init__()
        self.embed_gnns = nn.ModuleList()
        self.pool_gnns = nn.ModuleList()
        n, d_in = max_nodes, in_dim
        for _ in range(num_pool):
            m = max(int(assign_ratio * n), 1)
            self.embed_gnns.append(GNN(d_in, hidden, hidden))
            self.pool_gnns.append(GNN(d_in, hidden, m))
            n, d_in = m, hidden
        self.final_embed = GNN(d_in, hidden, hidden)
        self.classifier = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, num_classes)
        )

    def forward(self, adj, x):
        link_total = ent_total = 0.0
        for embed_gnn, pool_gnn in zip(self.embed_gnns, self.pool_gnns):
            z = embed_gnn(adj, x)
            s = pool_gnn(adj, x)
            x, adj, lp, ent = diff_pool(z, adj, s)
            link_total += lp
            ent_total += ent
        z = self.final_embed(adj, x)
        graph_vec = z.sum(dim=1)
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

The same link objective has a useful structural bias: dense, small-diameter cliques are collapsed into single hypernodes, which message passing can already summarize well, while sparse path-like or tree-like regions are preserved across several clusters, keeping the structure that message passing struggles to capture over many hops. By learning the hierarchy rather than fixing it in advance, DiffPool adapts the coarsening to both the graph structure and the classification task.