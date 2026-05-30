OK, let me think this through from scratch. I have a message-passing GNN that's good at one thing: given a graph, it produces an embedding *per node*. H^(0) = F, then H^(k) = M(A, H^(k-1); θ^(k)) for K rounds, and out comes Z ∈ R^{n×d}, one row per node, each row summarizing that node's K-hop neighborhood. For node classification that's exactly the unit I want. But the task in front of me is graph *classification* — one label for the whole molecule, the whole protein, the whole community graph — and a bag of n node vectors is not a graph vector. So there's a last mile: collapse n vectors of dimension d into a single vector of dimension D that a classifier can eat. And n varies graph to graph. How do I do that collapse?

The obvious move, the one everyone reaches for, is a global readout in one shot. Sum the rows. Or average them. Or — slightly fancier — run a permutation-invariant set network over the rows, a set2set-style attention readout. All of these are symmetric reductions, so they're permutation invariant, which I do need: the graph vector cannot depend on how I happened to number the nodes. Fine. So why does this nag at me?

Because it's *flat*. Stare at what message passing actually does: it moves information along edges, K hops out, and that's it. Every node ends up with a K-hop-neighborhood summary, and then I dump all n of those summaries into one pot and stir. At no point does the model ever represent anything at a scale *between* "single node's neighborhood" and "the entire graph." And real graphs are emphatically not flat. A molecule is atoms → bonds → functional groups → the whole molecule. A social graph is people → tight communities → the whole network. The label I'm trying to predict often lives at an intermediate scale — "does this molecule contain a benzene ring", "is this a collaboration network of physicists" — and a sum over atom-level embeddings has thrown that scale away before the classifier ever sees it.

Where have I seen this solved? Image CNNs. The thing that makes a deep CNN powerful isn't just convolution — it's convolution *interleaved with pooling*. Conv, then downsample to a coarser grid, then conv on the coarser grid, then downsample again. Each pooling step shrinks the spatial resolution, so deeper layers have larger receptive fields and see more global structure, while the early layers see fine detail. That alternation is what builds a hierarchy. My GNN has the convolution half — message passing *is* graph convolution — but it has no pooling half. So the fix, stated abstractly, is: I need a graph analogue of spatial pooling. A module that takes a graph and produces a *smaller* graph — fewer nodes, a coarser adjacency — so I can stack GNN-then-pool, GNN-then-pool, and let the later GNN operate on a coarsened version that captures bigger structures.

Let me try the naive transcription and watch it break, because that tells me what the real difficulty is. In a CNN, pooling is: take a 2×2 patch, max or average it, output one pixel. The patch is well-defined because the image has spatial locality — every pixel has the same grid of neighbors, and "2×2 block" means the same thing everywhere. So I want to pool "an m×m patch" of the graph. But what is a patch on a graph? There's no grid. Node 5 has three neighbors arranged one way, node 6 has eleven neighbors arranged another way; there's no canonical local window, no notion of "the block to the upper-left." Worse, if I tried to fix this by ordering the nodes into a sequence and then pooling consecutive runs — which is what the linearize-then-1D-CNN approaches do — I'd need a canonical node ordering that preserves structure, and finding that is essentially graph isomorphism. Nodes that are far apart in the graph would end up adjacent in my sequence, and I'd be pooling together things that have nothing to do with each other. Dead end. The locality CNN pooling relies on simply does not exist here.

And there's a second constraint that the CNN analogy hides: graphs in my dataset have different numbers of nodes and edges. My pooling operator can't be tied to a fixed grid size; it has to be one rule that applies uniformly whether the graph has 30 nodes or 400.

So what would a graph pooling operator even *be*, mechanically? Forget patches. The honest abstraction is: I have a graph with n nodes, embeddings Z ∈ R^{n×d}, adjacency A ∈ R^{n×n}, and I want to output a coarser graph with m < n nodes — embeddings Z' ∈ R^{m×d} and an adjacency A' ∈ R^{m×m} between the new nodes. The new "nodes" are *clusters* of old nodes. So pooling on a graph is really: cluster the nodes, treat each cluster as a single coarse node, figure out the embedding of each coarse node and the edges between coarse nodes. That's it. The whole problem reduces to "how do I cluster, and how do I aggregate within and between clusters."

Now, there's an existing answer to "cluster the nodes": run a deterministic graph-clustering algorithm — spectral clustering, some coarsening heuristic — get a partition, then run a GNN on the coarsened graph, and repeat to build the hierarchy. People do this. So why not just use it? Two problems, and they're the same problem really. First, the clustering is fixed *before* any gradient flows. It's computed by a separate subroutine that knows nothing about whether grouping these particular atoms helps predict the molecule's label. The clustering that's best for "minimize spectral cut" is not necessarily the clustering that's best for the classification task. Second, it's computed per-graph as a black-box preprocessing step, so there's no *learned, shared* pooling strategy that transfers across the many graphs in my dataset — and I have a whole dataset of different graphs, so what I really want is a pooling *function* that generalizes, the way the GNN weights generalize. So: I want the clustering itself to be *learned*, end-to-end, jointly with the task, by shared parameters that apply across graphs. That's the real requirement the two-stage methods miss.

Learned clustering, differentiable, end-to-end. The instant I say "differentiable" a problem appears: a clustering is a *hard* assignment — node i goes to cluster j, a discrete choice. Discrete choices have no gradient. I cannot backprop through "argmax which cluster." This is exactly the same wall you hit anywhere you want to learn a discrete structure with SGD, and the standard escape is to *soften* it: instead of assigning each node to one cluster, assign it a *distribution over clusters*. Let me define an assignment matrix S ∈ R^{n×m}: row i is node i, and the entries of row i are the (soft) probabilities that node i belongs to each of the m clusters. To make each row a genuine distribution — nonnegative, summing to one, and differentiable in the underlying scores — I push the row through a softmax. So S = softmax(scores), row-wise. Now the assignment is a smooth function of parameters, and gradients flow.

Given such an S, how do I actually coarsen? Think about what each new cluster-node's embedding should be: the (soft) collection of the embeddings of the nodes assigned to it. Node i contributes to cluster j in proportion to S_{ij}. So the embedding of cluster j is Σ_i S_{ij} Z_i — a weighted sum of node embeddings, weights from column j of S. Stack that over all clusters and it's a single matrix product:

  X' = Sᵀ Z,   X' ∈ R^{m×d}.

Clean. Row j of X' is exactly cluster j's pooled embedding. That's my coarse node features.

Now the coarse adjacency. I need the connectivity strength between cluster i and cluster j. Two clusters should be strongly connected if the original nodes assigned to them are strongly connected. Node-pair (p,q) connectivity is A_{pq}; node p belongs to cluster i with weight S_{pi}, node q to cluster j with weight S_{qj}. So the cluster-i–cluster-j connectivity is Σ_{p,q} S_{pi} A_{pq} S_{qj}, and stacking over all cluster pairs is the bilinear form

  A' = Sᵀ A S,   A' ∈ R^{m×m}.

That falls right out of treating S as a soft node→cluster indicator. Notice A' is real-valued and dense — a fully-connected weighted graph on m cluster-nodes, where A'_{ij} is how tightly clusters i and j are wired together. That's fine; message passing works on weighted adjacencies. So the two equations X' = Sᵀ Z and A' = Sᵀ A S together are my pooling operator. Feed (A', X') into the next GNN, and I've coarsened.

Hold on — I need to check this actually respects permutations, because that was a hard requirement and it'd be embarrassing to break it in the pooling step. Suppose I relabel the original nodes by a permutation matrix P: A ↦ PAPᵀ, and the node features ↦ PX. If my GNNs are permutation-equivariant (relabeling inputs relabels outputs the same way), then Z ↦ PZ and the scores, hence S ↦ PS. Now plug into the pooling: X' = (PS)ᵀ(PZ) = Sᵀ Pᵀ P Z = Sᵀ Z, because P is a permutation matrix so PᵀP = I. And A' = (PS)ᵀ(PAPᵀ)(PS) = Sᵀ (PᵀP) A (PᵀP) S = Sᵀ A S. Both unchanged. So the coarsened graph is *invariant* to how I numbered the original nodes — the relabeling washes out through PᵀP = I. Good, the operator is permutation invariant as long as the component GNNs are equivariant. That's the property I need for a graph-level answer to be well-defined.

Now, where does S come from? S is supposed to encode, for each node, which cluster it belongs to — and that should depend on the node's features and its position in the graph. That's precisely the kind of thing a GNN computes: a function of (A, X) producing a per-node output. So let me *generate S with a GNN*. Run a GNN on the current (A, X), and let its per-node output (of width m, the number of clusters) be the assignment scores; softmax each row:

  S = softmax( GNN_pool(A, X) ),   row-wise softmax, output width m.

The output width m is the number of clusters at the next layer — a hyperparameter I choose.

But wait — I also need the node embeddings Z that I'm going to pool. Where do those come from? Also a GNN on (A, X). So I have two jobs: produce embeddings to pool, and produce the assignment to pool *with*. Should one GNN do both? Let me think about whether to share. The embedding Z and the assignment S have genuinely different jobs. Z is the *content* — the representation I want to carry forward and ultimately classify on; it should encode whatever about a node's neighborhood is predictive of the label. S is the *partition* — it should encode which nodes belong together structurally, regardless of whether that's directly label-predictive. If I force one GNN to emit both, I tie these two objectives through shared weights, and the gradient for "represent the content well" fights the gradient for "group the right nodes." They want different features — e.g. the partition mostly wants connectivity/homophily, while the content wants whatever discriminates the label. So I'll use *two separate GNNs* on the same inputs, with distinct parameters:

  Z^(l) = GNN_{l,embed}(A^(l), X^(l)),
  S^(l) = softmax( GNN_{l,pool}(A^(l), X^(l)) ).

Same inputs, different weights, different roles. The embedding GNN makes the cluster-node features (via X' = Sᵀ Z), the pooling GNN makes the assignment. Each of these GNNs is itself K rounds of message passing — say a couple of GCN layers, H^(k) = ReLU(D̃^{-1/2}ÃD̃^{-1/2} H^(k-1) W^(k-1)) with Ã = A+I — so the assignment for a node is informed by its neighborhood, not just its own features, which is what I want for clustering.

Now stack it. Layer 0: inputs are the original A and node features F. Run embed-GNN and pool-GNN, get S^(0), Z^(0), coarsen to (A^(1), X^(1)) with m_1 < n clusters. Layer 1: run two more GNNs on (A^(1), X^(1)), coarsen to (A^(2), X^(2)) with m_2 < m_1. And so on, coarser and coarser, exactly the conv-then-pool stack I wanted. How does it end? At the final layer I want to land on a *single* graph vector. The cleanest way: at the penultimate layer, set the assignment to map every remaining node into *one* cluster — S = a column vector of all ones. Then X' = Sᵀ Z sums all the remaining cluster embeddings into a single row, the graph embedding. Feed that to a classifier (a small MLP then softmax over classes), cross-entropy loss, and the whole tower — every GNN, every assignment — trains end-to-end by SGD. The hierarchy is *learned*; nothing is precomputed.

How many clusters at each layer — how do I pick m_l? It has to shrink (that's the whole point of pooling), but not so aggressively that I throw away structure in one step. The natural parameterization is a *fraction* of the current node count: m_{l+1} = α · n_l for some reduction ratio α < 1, so the graph coarsens by a roughly constant factor per layer, the same way CNN feature maps halve resolution per pool. I'd expect the exact value not to matter much within a sensible band — pick α somewhere in 0.1–0.5; somewhere around a quarter is a reasonable default. If I set m larger than the number of "real" clusters in a graph, the softmax just leaves some columns near-empty, which is harmless.

I think the architecture is complete. But let me actually try to *train* this in my head, because I have a bad feeling about the pooling GNN. The only signal reaching S^(l) is the gradient of the classification loss, routed back through the coarsening and all the subsequent layers. That's a long, indirect path, and the thing it's trying to learn — a good soft clustering — is a *non-convex* object with lots of spurious local minima. Think about it as the matrix-factorization problem it resembles: a good assignment makes S Sᵀ look like the adjacency A (connected nodes share clusters), and approximating A by a low-rank S Sᵀ is exactly a non-convex factorization with many bad critical points. Early in training, with S near-uniform, the classification gradient through this mess is weak and easily wanders into a degenerate clustering — everything into one blob, or a clustering that ignores the graph entirely — and once there it's stuck. I don't think classification loss alone will reliably push the pooling GNN toward meaningful clusters. I need to *inject* the structural prior directly.

What's the prior? "Nodes that are connected should tend to end up in the same cluster." How do I write that as a differentiable loss on S? Here's the link I was circling: if S is a good clustering, then two nodes in the same cluster have similar rows of S, so S_i · S_j is large exactly when i and j are grouped together. The matrix S Sᵀ has entry (i,j) equal to S_i · S_j — the soft probability that i and j share a cluster. And A has entry (i,j) = 1 when they're actually connected. So "connected nodes get grouped together" is precisely "S Sᵀ should reconstruct A." I can penalize the gap directly:

  L_LP = || A^(l) − S^(l) S^(l)ᵀ ||_F,

the Frobenius norm of the difference, at every layer. This is a link-prediction objective: predict each edge of A from the cluster co-membership S Sᵀ. It's the same low-rank-factorization-of-A idea from before, but now it's an *auxiliary loss* trained jointly with the task — so it both guides the pooling GNN toward sensible clusters *and* lets the task tune them, which should find a task-better local minimum than a frozen two-step "factorize then classify" would. One subtlety to keep straight: at deeper layers A^(l) is itself the coarsened Sᵀ A S from below, so it's not a fixed target — it changes during training as the lower assignments change. That's fine, it's still a valid reconstruction signal at each layer; I just can't think of A^(l) as a constant.

There's a second failure mode the link loss alone doesn't fix. Softmax rows can stay *diffuse* — node i spread 1/m across every cluster. A diffuse S still partially satisfies the link objective but gives mushy clusters: every coarse node is a faint average of the whole graph, and the coarsened graph carries almost no information. I want each node to commit — each row of S close to one-hot, so cluster membership is crisp. "Close to one-hot" is "low entropy." So regularize the entropy of each assignment row downward:

  L_E = (1/n) Σ_i H(S_i),   H the entropy of row i,

averaged over nodes, minimized. Push every node's distribution toward a spike on one cluster (allowing the rare genuine bridge node to stay split). Add L_LP and L_E from every layer to the classification cross-entropy, and train the whole thing. I'd expect this to converge slower than bare classification — there's more to satisfy — but to land on cleaner, more interpretable clusters and better accuracy. The auxiliary terms are doing the early shaping that the weak, indirect classification gradient can't.

Let me sanity-check the whole mechanism on extreme inputs, because I want to know it does something *sensible* structurally, not just that it trains. Take a subgraph that's *dense* — close to a clique, adjacency almost all ones. What does L_LP want there? A^(l) ≈ 11ᵀ (the all-ones matrix). If S puts that whole subgraph into a single cluster — one column of all ones, the rest zero — then S Sᵀ over that block is also all ones, matching A. So the link objective *wants* to collapse a dense clique into one hypernode. Is that the right thing to do? Yes, and here's why it's not just an artifact: a GNN passes messages efficiently across a dense, small-diameter clique — in a clique every node reaches every other in one hop, so there's almost no fine structure for message passing to lose by collapsing it. Pooling a clique to a point throws away nothing the GNN couldn't already see. Now the opposite extreme: a *sparse* subgraph — a path, a cycle, a tree. High diameter, real structure that message passing struggles to capture across many hops. There L_LP is minimized by *spreading* the nodes across several clusters (S Sᵀ has small off-block entries, matching the sparse A), so the assignment keeps higher entropy and splits the sparse region into pieces — preserving exactly the path/cycle/tree structure the GNN would otherwise miss. So the same objective adaptively collapses dense regions (safe) and preserves sparse ones (necessary). I'd predict, then, that the method shines on graphs with clear sparse hierarchical structure and gains least on graphs that are uniformly very dense — there's just nothing to pool there beyond collapsing everything.

One more design choice that I'd otherwise make on autopilot: what features feed the two GNNs. For the *embedding* GNN, the more the merrier — raw node features plus structural descriptors like degree and clustering coefficient, since the label can depend on structure. For the *pooling* GNN, I actually want to *remove* the structural descriptors and lean on connectivity/homophily, because the partition should group by "who's near whom," and feeding raw degree into the assignment biases clusters by node degree rather than community. And dimensions: like CNN channels growing as spatial resolution shrinks, I let the embedding width grow at deeper (coarser) layers while the assignment width m shrinks — coarser graphs have fewer, bigger clusters but richer per-cluster representations. And one stability nicety: stacking many GNN-then-pool layers lets embedding magnitudes drift, so I ℓ2-normalize node embeddings after each graph-conv layer and add batch normalization, which I also expect to curb overfitting on these smallish graph datasets.

Let me also pin the WL connection, because it tells me the assignment GNN is computing the right *kind* of thing. If I strip a GNN down — hash the multiset of neighbor labels and iterate — that's the Weisfeiler–Lehman vertex-refinement test, the classical way to assign structural roles to nodes. My pooling GNN is a trainable, continuous relaxation of exactly that refinement, so it's natural that it can discover structurally coherent groups; the softmax-over-clusters head just turns "structural role" into "cluster membership."

Now let me write the real thing, grounded in how this actually gets implemented as a dense batched operation (graphs padded to a common max node count, adjacency and features as dense tensors). The core is the pooling module — it takes the pooling GNN's raw scores `s`, the embedding GNN's output `x`, and the current adjacency `adj`, and returns the coarsened `(x', adj')` plus the two auxiliary losses.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

def diff_pool(x, adj, s, normalize=True):
    # x:   [B, n, d]      node embeddings from the EMBEDDING gnn  (the Z to pool)
    # adj: [B, n, n]      current adjacency
    # s:   [B, n, m]      raw assignment scores from the POOLING gnn (pre-softmax)

    s = torch.softmax(s, dim=-1)                       # row-wise soft assignment S = softmax(scores)

    out     = torch.matmul(s.transpose(1, 2), x)       # X' = Sᵀ Z   -> [B, m, d]  (pool embeddings into clusters)
    out_adj = torch.matmul(torch.matmul(s.transpose(1, 2), adj), s)  # A' = Sᵀ A S -> [B, m, m] (coarsened adjacency)

    # auxiliary link-prediction loss: cluster co-membership S Sᵀ should reconstruct A
    link_loss = adj - torch.matmul(s, s.transpose(1, 2))            # A - S Sᵀ
    link_loss = torch.norm(link_loss, p=2)                          # || A - S Sᵀ ||_F
    if normalize:
        link_loss = link_loss / adj.numel()                        # normalize by number of adjacency entries

    # entropy regularization: push each assignment row toward one-hot (crisp membership)
    ent_loss = (-s * torch.log(s + 1e-15)).sum(dim=-1).mean()       # mean over nodes of H(S_i)

    return out, out_adj, link_loss, ent_loss
```

And the model wraps two GNNs per pooling step around it, stacks them, and collapses to a graph vector:

```python
class GNN(nn.Module):
    """A few rounds of GCN-style message passing on a dense (adj, x). Permutation-equivariant."""
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.W1 = nn.Linear(in_dim, hidden_dim)
        self.W2 = nn.Linear(hidden_dim, out_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)

    def norm_adj(self, adj):
        adj = adj + torch.eye(adj.size(-1), device=adj.device)      # Ã = A + I
        deg = adj.sum(-1, keepdim=True).clamp(min=1)
        dinv = deg.pow(-0.5)
        return dinv * adj * dinv.transpose(1, 2)                     # D̃^{-1/2} Ã D̃^{-1/2}

    def forward(self, adj, x):
        a = self.norm_adj(adj)
        h = F.relu(torch.matmul(a, self.W1(x)))
        b, n, c = h.shape
        h = self.bn1(h.reshape(b * n, c)).reshape(b, n, c)
        h = torch.matmul(a, self.W2(h))
        h = F.normalize(h, p=2, dim=-1)                              # ℓ2-normalize per node
        return h

class DiffPoolNet(nn.Module):
    def __init__(self, in_dim, hidden, num_classes, max_nodes, assign_ratio=0.25, num_pool=1):
        super().__init__()
        self.embed_gnns, self.pool_gnns = nn.ModuleList(), nn.ModuleList()
        n = max_nodes
        d_in = in_dim
        for _ in range(num_pool):
            m = max(int(assign_ratio * n), 1)                       # m_{l+1} = α · n_l clusters
            self.embed_gnns.append(GNN(d_in, hidden, hidden))       # Z^(l) = GNN_embed(A,X)
            self.pool_gnns.append(GNN(d_in, hidden, m))             # scores -> S^(l) = softmax(GNN_pool(A,X))
            n, d_in = m, hidden
        self.final_embed = GNN(d_in, hidden, hidden)                # last embedding GNN, S = ones -> single cluster
        self.classifier = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, num_classes),
        )

    def forward(self, adj, x):
        link_total = ent_total = 0.0
        for embed_gnn, pool_gnn in zip(self.embed_gnns, self.pool_gnns):
            z = embed_gnn(adj, x)                                   # embeddings to pool
            s = pool_gnn(adj, x)                                    # assignment scores
            x, adj, lp, ent = diff_pool(z, adj, s)                  # coarsen + aux losses
            link_total = link_total + lp
            ent_total  = ent_total + ent
        z = self.final_embed(adj, x)                                # S^(L-1) = 1 vector: sum all clusters
        graph_vec = z.sum(dim=1)                                    # X' = 1ᵀ Z -> one graph vector
        logits = self.classifier(graph_vec)
        return logits, link_total, ent_total

def train_step(model, batch, opt):
    adj, x, y = batch
    logits, link_loss, ent_loss = model(adj, x)
    loss = F.cross_entropy(logits, y) + link_loss + ent_loss        # task + auxiliary objectives
    opt.zero_grad(); loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 2.0)               # gradient clip at norm 2.0
    opt.step()
    return loss.item()
```

So the causal chain, start to finish: I needed a graph-level vector and the only honest version of CNN pooling on a graph is "cluster nodes into a coarser graph." Hard clustering kills the gradient, so I softened it into a row-stochastic assignment matrix S = softmax(scores); that immediately gives the two pooling maps X' = Sᵀ Z and A' = Sᵀ A S, which I checked are permutation invariant through PᵀP = I. I generate the content and the partition with two separate GNNs so their objectives don't fight, stack the embed-pool block to coarsen the graph layer by layer, and collapse the last layer to one cluster for a single graph vector and a softmax classifier. Because the pooling GNN's only signal — task gradient through a non-convex factorization-like clustering — is too weak to find good clusters, I inject the structural prior directly: a link-prediction loss ||A − S Sᵀ||_F that makes connected nodes share clusters, and an entropy loss that makes each assignment commit to one cluster. The same link objective collapses dense cliques (which message passing handles fine) and preserves sparse path/cycle/tree structure (which it doesn't), so the learned hierarchy adapts to the graph — and the whole tower trains end-to-end.
