OK, let me start from the thing that actually hurts. I have a GCN — each layer is x_v^(ℓ+1) = σ(Σ_{u} Â_{v,u} (W^(ℓ))ᵀ x_u^(ℓ)), a normalized average over neighbors, a linear map, a nonlinearity — and I want to train it on a graph with hundreds of thousands of nodes, possibly with several layers. Full-batch training, the way the original spectral/Chebyshev GCNs do it, wants the whole graph and every node's activation in memory at once and takes one giant gradient step. That doesn't fit, and it doesn't parallelize into the SGD-with-minibatches workflow everything else in deep learning uses. So I need minibatches. The question is how to cut a giant graph into minibatches.

The naive answer — pick a handful of nodes, compute their loss — detonates immediately, and I want to be precise about *why*, because the failure mode dictates the whole design. Take one node v and ask what an L-layer GCN needs to produce its output. Its layer-L representation is built from its neighbors' layer-(L−1) representations. Each of *those* needs its own neighbors at layer L−2. Unroll it: to compute one node I need its 1-hop neighbors, their neighbors (2-hop), … all the way out to L hops. On a graph with average degree d that's on the order of d^L support nodes. So a "minibatch" of a few nodes secretly drags in a huge, exponentially-growing chunk of the graph, and the cost per step explodes with depth. This is neighbor explosion, and it's the wall.

How have people climbed it? The whole existing literature shares one move: keep the GCN defined on the full graph, but at *each layer* sample which nodes/edges you actually propagate through, to bound the support. Let me walk these and find where each one still bleeds. GraphSAGE caps the per-node fanout — sample, say, 25 neighbors at layer 1 and 10 at layer 2, uniformly. That bounds each node's branching, but the branching still *multiplies* across layers: budget^L support per output node, so deeper nets still blow up, just with a smaller base. The variance-reduction line (S-GCN) pushes the fanout down to two support nodes per layer by reusing stale historical activations to stand in for the neighbors it didn't sample — clever, but it has to store and maintain a historical activation for every node, and even fanout-2 is costly when the net is deep. FastGCN takes a sharper angle: forget tracking which layer-ℓ node feeds which layer-(ℓ+1) node; instead sample a *constant* number of nodes *independently* at each layer, by importance sampling with probability proportional to ‖Â_{:,u}‖², and reweight. Now the sample size is constant across layers — fanout effectively 1, no depth blow-up. 

But FastGCN trades one disease for another, and watching that trade fail is what points me at the answer. Because the layers are sampled *independently*, there's no guarantee that a node I picked at layer ℓ+1 has *any* of its neighbors picked at layer ℓ. When that happens, that node aggregates over an empty or near-empty neighborhood — its representation is garbage. The minibatch is "too sparse": the inter-layer wiring that the GCN needs to actually propagate is shredded by the independence. Quantify it on a uniform-degree-d graph where each layer keeps an edge with probability p: an input-layer node "survives" — stays connected through all L independent samplers — with probability roughly (1−(1−p)^d)^{L−1}, which decays as the net gets deeper. So FastGCN's minibatches get emptier with depth, and accuracy drops. AS-GCN patches exactly this by learning a sampler that picks each layer *conditioned* on the next layer's chosen nodes, restoring connectivity — but now I'm training an extra sampling network with its own parameters and cost. ClusterGCN sidesteps layer sampling entirely: precompute dense graph clusters and let each minibatch be a few whole clusters, so a minibatch is a self-contained subgraph and there's no explosion — but the clustering is a heuristic, and because clusters are fixed and chosen non-uniformly, the minibatch loss is a *biased* estimate of the true full-batch loss, with nobody correcting it.

Let me stare at what's common to the failures. Layer sampling fights neighbor explosion by thinning the *connections between* layers, and every time it does that it risks severing the very connectivity the GCN needs — that's the FastGCN sparsity problem — or it pays to repair it. And ClusterGCN shows that the moment you sample non-uniformly without correction, you bias the estimator. So I want two things at once that layer sampling can't give together: (a) every minibatch should be *fully connected the way the GCN expects* — if an edge is used at one layer it's available at all layers — and (b) the minibatch should be an *unbiased, low-variance* estimate of the full graph.

Here's the reframe. What if I don't sample the *layers* at all? What if I sample the *graph* first — carve out a small subgraph G_s ⊆ G — and then build a *complete, full* GCN on that subgraph, with all L layers, no layer sampling whatsoever? Look at what that buys. Inside G_s every node sees all its G_s-neighbors at every layer, because the subgraph is fixed before propagation starts — there's no per-layer thinning, so connectivity *never* degrades with depth (if an edge is in the subgraph, it's in the subgraph for all layers). Neighbor explosion is gone too, because the support of every node is confined to G_s, which is small and never expands outside the batch. The GCN of each minibatch is small but *complete*. That single inversion — sample the training graph, not the GCN — fixes both the connectivity problem and the explosion problem at once. Everything after this is making it correct and choosing how to carve.

So the procedure is: each step, sample a subgraph G_s(V_s, E_s) with |V_s| ≪ |V|; build the full GCN on G_s; forward to get an embedding and loss for every v ∈ V_s; backprop; update. What do I want from the sampler? Two intuitions. First, nodes that strongly influence each other should tend to land in the *same* subgraph, so they can support each other's propagation without needing anything outside the batch — that's what makes the small subgraph self-sufficient. Second, every edge should have a non-negligible chance of being sampled, so over many subgraphs the model explores the whole feature/label space and generalizes. I'll define "influence" purely from topology (connectivity), not features — a feature-aware sampler would have to infer relationships between attributes and be far more expensive.

Now the catch, and it's the same catch that bit ClusterGCN. The first requirement — group mutually-influential, i.e. well-connected, nodes together — *guarantees* a non-uniform sampler: high-centrality nodes and the edges among dense regions get sampled far more often than peripheral ones. Non-uniform sampling means a biased estimator: the model would preferentially learn the features of frequently-sampled nodes. I cannot just accept that the way ClusterGCN does. I have to *remove* the bias analytically. Let me do that carefully, because it's the part that makes graph sampling actually correct rather than just convenient.

Analyzing the full multi-layer GCN at once is hard because of the nonlinearities, so I'll do what the layer-sampling analyses do and treat each layer independently — assume each layer learns its embedding on its own. Take a node v that got sampled (v ∈ V_s) at layer ℓ+1. In the full graph its aggregated pre-activation is Σ_{u∈V} Â_{v,u} x̃_u^(ℓ), where I write x̃_u^(ℓ) = (W^(ℓ))ᵀ x_u^(ℓ) for the transformed feature. In the subgraph I only get the neighbors u that happen to be present, so I'd compute Σ_u Â_{v,u} x̃_u^(ℓ) 1_{u|v}, where 1_{u|v} indicates that edge (u,v) made it into the subgraph *given* v is in it. That sum is biased — it's missing terms and weighting present ones by whatever the sampling happened to do. So introduce a per-edge correction constant α_{u,v} and define the estimator

  ζ_v^(ℓ+1) = Σ_{u∈V} (Â_{v,u} / α_{u,v}) x̃_u^(ℓ) 1_{u|v}.

I want to pick α so this is *unbiased* — so E[ζ_v^(ℓ+1)] equals the true full-graph aggregation. Take the expectation. By linearity it's Σ_u (Â_{v,u}/α_{u,v}) x̃_u^(ℓ) E[1_{u|v}]. And E[1_{u|v}] is the probability that edge (u,v) is sampled *given* v is sampled — a conditional probability — which is P((u,v) sampled)/P(v sampled) = p_{u,v}/p_v, writing p_{u,v} for the edge's sampling probability and p_v for the node's. So

  E[ζ_v^(ℓ+1)] = Σ_u (Â_{v,u}/α_{u,v}) x̃_u^(ℓ) · (p_{u,v}/p_v).

For this to collapse to the true Σ_u Â_{v,u} x̃_u^(ℓ), I need the correction to exactly cancel the conditional edge probability: α_{u,v} = p_{u,v}/p_v. That's the *aggregator normalization* — divide each neighbor's contribution by the conditional probability that its edge survived given the node did, and the per-layer aggregation becomes an unbiased estimator of the full aggregation. Clean: it's just inverse-probability weighting, but with the *conditional* edge probability, because we're already conditioning on v being in the batch.

The aggregation is one place bias enters; the *loss* is the other. If I just sum L_v over the sampled nodes, frequently-sampled nodes contribute to the loss more often and dominate the gradient. So reweight each node's loss too. Define the minibatch loss L_batch = Σ_{v∈V_s} L_v/λ_v with a per-node *loss normalization* λ_v. I want E[L_batch] to equal the true average loss (1/|V|)Σ_{v∈V} L_v. Take the expectation: each node v contributes L_v/λ_v exactly when it's sampled, i.e. with probability p_v, so E[L_batch] = Σ_{v∈V} (L_v/λ_v) p_v. For that to equal (1/|V|)Σ_v L_v I need (p_v/λ_v) = 1/|V|, i.e. λ_v = |V|·p_v. So divide each node's loss by |V|·p_v and the minibatch loss is an unbiased estimate of the full-graph loss — frequently-sampled nodes get downweighted by exactly their oversampling, and the model stops preferring them.

Both corrections need the sampling probabilities p_v and p_{u,v}. For dead-simple samplers (uniform random node, uniform random edge) I can write those in closed form. But the interesting samplers — the connectivity-preserving ones I actually want — have no closed-form node/edge probability. So estimate them empirically, once, up front. Before training, run the sampler N times to get a pile of N subgraphs (which I'll *reuse* as minibatches, so this isn't wasted work). Keep a counter C_v for how many of those subgraphs contained node v, and C_{u,v} for how many contained edge (u,v). Then the empirical node probability is C_v/N and the empirical edge probability is C_{u,v}/N, so

  α_{u,v} = p_{u,v}/p_v = (C_{u,v}/N)/(C_v/N) = C_{u,v}/C_v,   λ_v = |V|·p_v = |V|·C_v/N.

(In code these show up as their reciprocals — each edge's aggregator factor stored as C_v/C_{u,v}, each node's loss factor as N/(C_v·|V|) — same thing.) The preprocessing is light: a handful of cheap topology-only samples and two integer counters, and the subgraphs become the training minibatches.

Now the second lever: *which* sampler. Unbiasedness I've handled for any sampler; but among unbiased estimators I want the lowest *variance*, because that's what makes SGD converge fast. So let me actually derive the variance-optimal sampler instead of guessing. To make it tractable, ignore the activations' detailed values and look at the aggregate signal. For an edge e = (u,v), the contribution it carries at layer ℓ is b_e^(ℓ) = Â_{v,u} x̃_u^(ℓ−1) + Â_{u,v} x̃_v^(ℓ−1) (both directions). Stack the unbiased per-node estimators across all layers and nodes into one quantity, reweighting by 1/p_v so the whole thing is unbiased for the sum of all aggregations:

  ζ = Σ_ℓ Σ_{v∈V_s} ζ_v^(ℓ)/p_v = Σ_ℓ Σ_e (b_e^(ℓ)/p_e) 1_e^(ℓ).

Two facts I'll lean on. First, I'll restrict to *independent edge sampling*: decide each edge e ∈ E independently, include it with probability p_e, under a budget Σ_e p_e = m so the expected number of sampled edges is a chosen constant m. Independence makes the covariance across distinct edges vanish. Second, in my scheme once an edge is in the subgraph it's there for *all* layers — there's no per-layer resampling — so 1_e^(ℓ) is the same Bernoulli 1_e for every ℓ. That means the *same* edge's indicator is perfectly correlated across layers: Cov(1_e^(ℓ₁), 1_e^(ℓ₂)) = Var(1_e) = p_e − p_e², while Cov across different edges is 0.

Compute the variance. For one edge, summing its self-terms and its within-edge cross-layer terms:

  Var(ζ) = Σ_e [ Σ_ℓ (b_e^(ℓ))²/p_e² + 2 Σ_{ℓ₁<ℓ₂} b_e^(ℓ₁)b_e^(ℓ₂)/p_e² ] · (p_e − p_e²).

Factor (p_e − p_e²)/p_e² = (1/p_e − 1), and the bracket is exactly (Σ_ℓ b_e^(ℓ))² (square of a sum = sum of squares + twice the cross terms). So

  Var(ζ) = Σ_e (1/p_e − 1)(Σ_ℓ b_e^(ℓ))² = Σ_e (Σ_ℓ b_e^(ℓ))²/p_e − Σ_e (Σ_ℓ b_e^(ℓ))².

The second sum doesn't depend on the p_e at all — it's a constant I can't touch. So minimizing variance means minimizing Σ_e (Σ_ℓ b_e^(ℓ))²/p_e subject to Σ_e p_e = m. This is a textbook constrained minimization, and Cauchy–Schwarz nails it: for any positive p_e,

  [ Σ_e (Σ_ℓ b_e^(ℓ))²/p_e ] · [ Σ_e p_e ] ≥ ( Σ_e |Σ_ℓ b_e^(ℓ)| )²,

with equality exactly when (Σ_ℓ b_e^(ℓ))²/p_e ∝ p_e, i.e. when p_e ∝ |Σ_ℓ b_e^(ℓ)|. Since Σ_e p_e = m, that pins it: p_e = m·|Σ_ℓ b_e^(ℓ)| / Σ_{e'}|Σ_ℓ b_{e'}^(ℓ)|. (For the real vector-valued b_e^(ℓ), minimizing the summed-over-dimensions variance gives the same form with the absolute value replaced by the norm ‖·‖.) So the variance-optimal edge probability is proportional to the magnitude of that edge's total contribution across layers.

That's the right answer but it's not *usable* as stated: Σ_ℓ b_e^(ℓ) depends on the transformed activations x̃, which change every step and are expensive to compute just to decide what to sample. So I simplify — drop the activation dependence and keep only the part of b_e that's pure topology. Strip x̃ out and what's left of b_e is governed by Â_{v,u} + Â_{u,v}, and with the row-normalized Â = D^{-1}A that's exactly 1/deg(u) + 1/deg(v). So set

  p_e ∝ Â_{v,u} + Â_{u,v} = 1/deg(u) + 1/deg(v).

And this *agrees with the influence intuition* I started from: if u and v are connected and each has few neighbors, then 1/deg(u)+1/deg(v) is large, the edge is sampled with high probability — and indeed two low-degree connected nodes are exactly the pair that strongly influence each other and should ride in the same subgraph. The variance math and the hand-wave land on the same place, which is reassuring.

Let me also double-check the "sample graph not layers" claim against this same edge probability, because it's worth seeing the contrast sharply. Suppose I took this very p_e and used it as a *layer* sampler in the FastGCN style — sample each layer's edges independently with this probability. On a uniform-degree-d graph with per-layer keep-probability p, an input node stays connected through all L independent layers with probability (1−(1−p)^d)^{L−1}, which shrinks as L grows: deeper ⇒ sparser minibatches. But in graph sampling the edge, once chosen, lives in every layer, so connectivity *never* drops with depth. Same edge probabilities, opposite behavior — the win is in *when* you sample, not just with what distribution.

So I have a family of samplers. The edge sampler I just derived (sample m edges with probability ∝ 1/deg(u)+1/deg(v)). A node sampler (sample n nodes with probability ∝ ‖Â_{:,v}‖², borrowed from FastGCN's layer sampler but applied to pick subgraph nodes). And random-walk samplers, which are a natural fit for an L-layer GCN: if I ignore activations, L stacked layers act like a single hop matrix B = Â^L, and the same variance logic would want to sample node *pairs* with p_{u,v} ∝ B_{u,v}+B_{v,u} — the probability a length-L walk connects u and v. I can't independently sample arbitrary pairs, but a length-h random walk approximates picking pairs that are mutually reachable, so an h-hop walk sampler (r roots, each walking h uniform-random steps) is a sound proxy for an h-layer GCN. There's also a multi-dimensional / frontier random walk (Ribeiro–Towsley): maintain a frontier of r nodes, repeatedly pick one with probability ∝ its degree, step to a random neighbor, swap it into the frontier, collecting visited nodes. For *all* of these samplers I then take the *node-induced* subgraph — include every edge of G between the sampled nodes, even ones the sampler didn't traverse. The induction adds connections back, which empirically speeds up convergence: more intra-batch edges means each node has more support during propagation.

Let me write the real thing. The bones are: a preprocessing pass that samples a bunch of subgraphs and turns the node/edge counters into the normalization coefficients; a per-step routine that pops a sampled subgraph and builds a complete GCN on it; and the two normalizations wired in (aggregator α on the adjacency, loss λ on the per-node loss). The subgraph adjacency is rescaled by the per-edge aggregator factor once, and the loss is multiplied by the per-node factor.

```python
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- samplers: all return a set of node IDs; the subgraph is node-INDUCED ----
def sampler_edge(adj_train, budget_m, deg):
    # p_e ∝ 1/deg(u) + 1/deg(v)  (variance-optimal, topology-only)
    rows, cols = adj_train.nonzero()
    pe = 1.0 / deg[rows] + 1.0 / deg[cols]
    pe = pe / pe.sum()
    pick = np.random.choice(len(rows), size=budget_m, replace=True, p=pe)   # m edges w/ replacement
    return np.unique(np.concatenate([rows[pick], cols[pick]]))             # endpoints -> induce

def sampler_rw(adj_train, num_roots, walk_len):
    nodes = set(np.random.randint(adj_train.shape[0], size=num_roots))
    for r in list(nodes):
        u = r
        for _ in range(walk_len):
            nbrs = adj_train.indices[adj_train.indptr[u]:adj_train.indptr[u + 1]]
            if len(nbrs) == 0: break
            u = np.random.choice(nbrs)
            nodes.add(u)
    return np.array(sorted(nodes))

def induce(adj_train, sub_nodes):
    return adj_train[sub_nodes][:, sub_nodes].tocsr()      # node-induced subgraph (adds back edges)

# ---- preprocessing: estimate aggregator (alpha) and loss (lambda) normalization ----
def estimate_norms(adj_train, sampler, num_train, sample_coverage=50):
    n = adj_train.shape[0]
    C_v   = np.zeros(n)                                    # counts of node v over subgraphs
    C_uv  = np.zeros(adj_train.nnz)                        # counts of edge (u,v) over subgraphs
    subgraphs, total = [], 0
    while total <= sample_coverage * num_train:            # sample until enough coverage
        sub = sampler(adj_train)
        subgraphs.append(sub); total += len(sub)
        C_v[sub] += 1
        sub_adj = induce(adj_train, sub)                   # edges present this subgraph
        # accumulate edge counts onto the global edge index (omitted index bookkeeping)
    N = len(subgraphs)
    # aggregator normalization stored as reciprocal 1/alpha = C_v / C_uv (per edge (u,v))
    aggr_norm = np.divide(C_v[adj_train.tocoo().row], np.clip(C_uv, 1, None))
    # loss normalization stored as reciprocal 1/lambda = N / (C_v * |V_train|)
    loss_norm = np.where(C_v > 0, N / np.clip(C_v, 1, None) / num_train, 0.1)
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
            h = F.relu(torch.sparse.mm(adj_norm, W(h)))    # σ( Â_α · (W h) ), Â already aggr-normalized
        return self.out(h)

def train(adj_train, features, labels, model, opt, sampler, num_train, steps):
    subgraphs, aggr_norm, loss_norm = estimate_norms(adj_train, sampler, num_train)
    loss_norm = torch.tensor(loss_norm, dtype=torch.float32)
    i = 0
    for _ in range(steps):
        if i >= len(subgraphs):                            # reuse the pre-sampled subgraphs as batches
            subgraphs += [sampler(adj_train) for _ in range(len(subgraphs))]
        sub = subgraphs[i]; i += 1
        sub_adj = induce(adj_train, sub)
        sub_adj = sp.diags(1.0 / np.clip(sub_adj.sum(1).A1, 1, None)).dot(sub_adj)   # row-normalize Â on subgraph
        # aggregator normalization: scale each edge by 1/alpha = C_v / C_uv before propagation
        adj_norm = to_torch_sparse(scale_edges(sub_adj, aggr_norm, sub))
        logits = model(adj_norm, features[sub])
        per_node = F.cross_entropy(logits, labels[sub], reduction='none')
        loss = (per_node * loss_norm[sub]).sum()           # loss normalization: divide each L_v by lambda_v
        opt.zero_grad(); loss.backward(); opt.step()
```

The causal chain, end to end: minibatching a GCN by picking nodes detonates because the support set grows like d^L with depth — neighbor explosion. Every layer-sampling fix thins the inter-layer connections, which either re-introduces the explosion (GraphSAGE/S-GCN) or shreds connectivity into too-sparse minibatches (FastGCN), or pays a learned sampler (AS-GCN) or an uncorrected bias (ClusterGCN) to avoid it. Inverting the order — sample a subgraph first, then build a *complete* GCN on it — kills the explosion (support stays inside the subgraph) and preserves connectivity at every depth (an edge in the subgraph is there for all layers). The price is that a connectivity-preserving sampler is non-uniform and so biased, which I cancel exactly with two inverse-probability corrections: aggregator normalization α_{u,v} = p_{u,v}/p_v makes each layer's aggregation unbiased, and loss normalization λ_v = |V|·p_v makes the minibatch loss unbiased; the unknown probabilities I estimate by counting node/edge appearances over a batch of pre-sampled subgraphs. Then, among unbiased samplers, minimizing the estimator's variance under independent edge sampling gives — via Cauchy–Schwarz — p_e ∝ ‖Σ_ℓ b_e^(ℓ)‖, which I make practical by dropping activations to p_e ∝ 1/deg(u)+1/deg(v), matching the influence intuition that low-degree connected nodes belong together. Node-induced subgraphs add the edges back to densify connectivity, and the pre-sampled subgraphs double as the training minibatches.
