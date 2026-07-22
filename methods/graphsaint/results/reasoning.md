OK, let me start from the thing that actually hurts. I have a GCN — each layer is x_v^(ℓ+1) = σ(Σ_{u} Â_{v,u} (W^(ℓ))ᵀ x_u^(ℓ)), a normalized average over neighbors, a linear map, a nonlinearity — and I want to train it on a graph with hundreds of thousands of nodes, possibly with several layers. Full-batch training, the way the original spectral/Chebyshev GCNs do it, wants the whole graph and every node's activation in memory at once and takes one giant gradient step. That doesn't fit, and it doesn't parallelize into the SGD-with-minibatches workflow everything else in deep learning uses. So I need minibatches. The question is how to cut a giant graph into minibatches.

The naive answer — pick a handful of nodes, compute their loss — detonates immediately, and I want to be precise about *why*, because the failure mode dictates the whole design. Take one node v and ask what an L-layer GCN needs to produce its output. Its layer-L representation is built from its neighbors' layer-(L−1) representations. Each of *those* needs its own neighbors at layer L−2. Unroll it: to compute one node I need its 1-hop neighbors, their neighbors (2-hop), … all the way out to L hops. On a graph with average degree d that's on the order of d^L support nodes. So a "minibatch" of a few nodes secretly drags in a huge, exponentially-growing chunk of the graph, and the cost per step explodes with depth. This is neighbor explosion, and it's the wall.

How have people climbed it? The whole existing literature shares one move: keep the GCN defined on the full graph, but at *each layer* sample which nodes/edges you actually propagate through, to bound the support. Let me walk these and find where each one still bleeds. GraphSAGE caps the per-node fanout — sample, say, 25 neighbors at layer 1 and 10 at layer 2, uniformly. That bounds each node's branching, but the branching still *multiplies* across layers: budget^L support per output node, so deeper nets still blow up, just with a smaller base. The variance-reduction line (S-GCN) pushes the fanout down to two support nodes per layer by reusing stale historical activations to stand in for the neighbors it didn't sample — clever, but it has to store and maintain a historical activation for every node, and even fanout-2 is costly when the net is deep. FastGCN takes a sharper angle: forget tracking which layer-ℓ node feeds which layer-(ℓ+1) node; instead sample a *constant* number of nodes *independently* at each layer, by importance sampling with probability proportional to ‖Â_{:,u}‖², and reweight. Now the sample size is constant across layers — fanout effectively 1, no depth blow-up. 

FastGCN has its own weak spot, though. Because the layers are sampled *independently*, there's no guarantee that a node I picked at layer ℓ+1 has *any* of its neighbors picked at layer ℓ. When that happens, that node aggregates over an empty or near-empty neighborhood — its representation is garbage. The minibatch is "too sparse": the inter-layer wiring that the GCN needs to actually propagate is shredded by the independence. Let me see how bad this actually gets rather than just wave at it. On a uniform-degree-d graph where each layer's sampler independently keeps each edge with probability p, an input-layer node stays connected through all L−1 inter-layer transitions only if, at *each* transition, at least one of its d edges survives — probability (1−(1−p)^d) per transition, so (1−(1−p)^d)^{L−1} overall. Put numbers in (d=5, p=0.3): at L=2 that's 0.83, at L=3 it's 0.69, at L=4 it's 0.58, at L=6 it's 0.40. I simulated the same thing — draw d Bernoulli(p) edges per transition, require ≥1 survivor each of the L−1 transitions — and got 0.832, 0.691, 0.575, 0.398, matching. So this is real: the fraction of usable input nodes roughly *halves* going from a 2-layer to a 6-layer net, and the minibatches get emptier exactly when I want depth. AS-GCN patches this by learning a sampler that picks each layer *conditioned* on the next layer's chosen nodes, restoring connectivity — but now I'm training an extra sampling network with its own parameters and cost. ClusterGCN sidesteps layer sampling entirely: precompute dense graph clusters and let each minibatch be a few whole clusters, so a minibatch is a self-contained subgraph and there's no explosion — but the clustering is a heuristic, and when node or edge inclusion probabilities differ, the minibatch loss has no inverse-probability correction.

Let me stare at what's common to the failures. Layer sampling fights neighbor explosion by thinning the *connections between* layers, and every time it does that it risks severing the very connectivity the GCN needs — that's the FastGCN sparsity problem — or it pays to repair it. And ClusterGCN shows that the moment you sample non-uniformly without correction, you bias the estimator. So I want two things at once that layer sampling can't give together: (a) every minibatch should be *fully connected the way the GCN expects* — if an edge is used at one layer it's available at all layers — and (b) the minibatch should be an *unbiased, low-variance* estimate of the full graph.

The thing every one of these methods has in common is that it samples the *layers*, thinning the connections during propagation, and that thinning is the source of both failure modes. So let me try the opposite order. Suppose I sample the *graph* first — carve out a small subgraph G_s ⊆ G — and then build a *complete, full* GCN on that subgraph, with all L layers, no layer sampling whatsoever. What does that do to the two failure modes? Inside G_s every node sees all its G_s-neighbors at every layer, because the subgraph is fixed before propagation starts — there's no per-layer thinning, so connectivity can't degrade with depth: if an edge is in the subgraph, it's in the subgraph for all layers, the (1−(1−p)^d)^{L−1} decay just doesn't arise. And neighbor explosion is gone too, because the support of every node is confined to G_s, which is small and never expands outside the batch. The GCN of each minibatch is small but *complete*. So sampling the graph rather than the GCN addresses both failure modes at the source. That's promising enough to build on — but it's only *convenient*, not yet *correct*, and the next worry is exactly the one that bit ClusterGCN.

So the procedure is: each step, sample a subgraph G_s(V_s, E_s) with |V_s| ≪ |V|; build the full GCN on G_s; forward to get an embedding and loss for every v ∈ V_s; backprop; update. What do I want from the sampler? Two intuitions. First, nodes that strongly influence each other should tend to land in the *same* subgraph, so they can support each other's propagation without needing anything outside the batch — that's what makes the small subgraph self-sufficient. Second, every edge should have a non-negligible chance of being sampled, so over many subgraphs the model explores the whole feature/label space and generalizes. I'll define "influence" purely from topology (connectivity), not features — a feature-aware sampler would have to infer relationships between attributes and be far more expensive.

Now the catch, and it's the same catch that bit ClusterGCN. The first requirement — group mutually-influential, i.e. well-connected, nodes together — *guarantees* a non-uniform sampler: high-centrality nodes and the edges among dense regions get sampled far more often than peripheral ones. Non-uniform sampling means a biased estimator: the model would preferentially learn the features of frequently-sampled nodes. I cannot just accept that the way ClusterGCN does. I have to *remove* the bias analytically. Let me do that carefully, because it's the part that makes graph sampling actually correct rather than just convenient.

Analyzing the full multi-layer GCN at once is hard because of the nonlinearities, so I'll do what the layer-sampling analyses do and treat each layer independently — assume each layer learns its embedding on its own. Take a node v that got sampled (v ∈ V_s) at layer ℓ+1. In the full graph its aggregated pre-activation is Σ_{u∈V} Â_{v,u} x̃_u^(ℓ), where I write x̃_u^(ℓ) = (W^(ℓ))ᵀ x_u^(ℓ) for the transformed feature. In the subgraph I only get the neighbors u that happen to be present, so I'd compute Σ_u Â_{v,u} x̃_u^(ℓ) 1_{u|v}, where 1_{u|v} indicates that edge (u,v) made it into the subgraph *given* v is in it. That sum is biased — it's missing terms and weighting present ones by whatever the sampling happened to do. So introduce a per-edge correction constant α_{u,v} and define the estimator

  ζ_v^(ℓ+1) = Σ_{u∈V} (Â_{v,u} / α_{u,v}) x̃_u^(ℓ) 1_{u|v}.

I want to pick α so this is *unbiased* — so E[ζ_v^(ℓ+1)] equals the true full-graph aggregation. Take the expectation. By linearity it's Σ_u (Â_{v,u}/α_{u,v}) x̃_u^(ℓ) E[1_{u|v}]. And E[1_{u|v}] is the probability that edge (u,v) is sampled *given* v is sampled — a conditional probability — which is P((u,v) sampled)/P(v sampled) = p_{u,v}/p_v, writing p_{u,v} for the edge's sampling probability and p_v for the node's. So

  E[ζ_v^(ℓ+1)] = Σ_u (Â_{v,u}/α_{u,v}) x̃_u^(ℓ) · (p_{u,v}/p_v).

For this to collapse to the true Σ_u Â_{v,u} x̃_u^(ℓ), I need the correction to exactly cancel the conditional edge probability: α_{u,v} = p_{u,v}/p_v. So divide each neighbor's contribution by the conditional probability that its edge survived given the node did. It's inverse-probability weighting, but with the *conditional* edge probability, because we're already conditioning on v being in the batch.

I don't fully trust an algebra step until I've watched it produce the right number, so let me put a tiny case on paper. One target node v with two neighbors u₁, u₂; row-normalized Â so Â_{v,u₁}=Â_{v,u₂}=1/2; feature values x̃_{u₁}=3.7, x̃_{u₂}=−1.2. The true aggregation is (1/2)(3.7)+(1/2)(−1.2)=1.25. Now a sampler that, given v is in the batch, keeps edge (u₁,v) with conditional probability 0.3 and (u₂,v) with 0.8 — so α_{u₁,v}=0.3, α_{u₂,v}=0.8. The estimator pays out (Â_{v,u₁}/α₁)x̃₁ only when edge 1 survives, etc. Its expectation is (1/2/0.3)(3.7)(0.3) + (1/2/0.8)(−1.2)(0.8) = (1/2)(3.7) + (1/2)(−1.2) = 1.25 — the α and the survival probability cancel term by term, which is the whole point. I also ran it as a Monte-Carlo with two million draws and got 1.255, so it's not an algebra slip. And to see that the correction is doing real work, drop it (α=1): the expectation becomes 0.3·(1/2)(3.7)+0.8·(1/2)(−1.2)=0.075, badly biased toward whichever edge is sampled more. So the aggregator normalization is what makes the per-layer aggregation unbiased; without it the estimate is off by almost 20×.

The aggregation is one place bias enters; the *loss* is the other. If I just sum L_v over the sampled nodes, frequently-sampled nodes contribute to the loss more often and dominate the gradient. So reweight each node's loss too. Define the minibatch loss L_batch = Σ_{v∈V_s} L_v/λ_v with a per-node *loss normalization* λ_v. I want E[L_batch] to equal the true average loss (1/|V|)Σ_{v∈V} L_v. Take the expectation: each node v contributes L_v/λ_v exactly when it's sampled, i.e. with probability p_v, so E[L_batch] = Σ_{v∈V} (L_v/λ_v) p_v. For that to equal (1/|V|)Σ_v L_v I need (p_v/λ_v) = 1/|V|, i.e. λ_v = |V|·p_v. So divide each node's loss by |V|·p_v: frequently-sampled nodes get downweighted by exactly their oversampling.

Same instinct — check the number. Four nodes, sampling probabilities p=(0.2,0.5,0.7,0.9), per-node losses L=(1,2,3,4), so the true average loss is 2.5 and λ_v=4·p_v=(0.8,2.0,2.8,3.6). Drawing each node independently into the batch with its p_v and summing L_v/λ_v over those present, the expectation is Σ_v (L_v/λ_v)p_v = Σ_v L_v/4 = (1+2+3+4)/4 = 2.5 — the p_v in λ_v cancels the p_v sampling rate exactly. A two-million-step simulation gave 2.4999. So loss normalization makes the minibatch loss an unbiased estimate of the full-graph average loss, and the model stops preferring the nodes it happens to see often.

Both corrections need the sampling probabilities p_v and p_{u,v}. For dead-simple samplers (uniform random node, uniform random edge) I can write those in closed form. But the interesting samplers — the connectivity-preserving ones I actually want — have no closed-form node/edge probability. So estimate them empirically, once, up front. Before training, run the sampler N times to get a pile of N subgraphs (which I'll *reuse* as minibatches, so this isn't wasted work). Keep a counter C_v for how many of those subgraphs contained node v, and C_{u,v} for how many contained edge (u,v). Then the empirical node probability is C_v/N and the empirical edge probability is C_{u,v}/N, so

  α_{u,v} = p_{u,v}/p_v = (C_{u,v}/N)/(C_v/N) = C_{u,v}/C_v,   λ_v = |V|·p_v = |V|·C_v/N.

(In code these show up as their reciprocals — each outgoing edge from v stores C_v/C_{u,v}, each training node's loss stores N/(C_v·|V_train|) — same thing. If a count is zero, the implementation uses small finite fallback values or clipping; those are numerical guards, not part of the unbiasedness argument.) The preprocessing is light: a handful of cheap topology-only samples and two integer counters, and the sampled subgraph pool is reused as training minibatches.

Now the second lever: *which* sampler. Unbiasedness I've handled for any sampler; but among unbiased estimators I want the lowest *variance*, because that's what makes SGD converge fast. So let me actually derive the variance-optimal sampler instead of guessing. To make it tractable, ignore the activations' detailed values and look at the aggregate signal. For an edge e = (u,v), the contribution it carries at layer ℓ is b_e^(ℓ) = Â_{v,u} x̃_u^(ℓ−1) + Â_{u,v} x̃_v^(ℓ−1) (both directions). Stack the unbiased per-node estimators across all layers and nodes into one quantity, reweighting by 1/p_v so the whole thing is unbiased for the sum of all aggregations:

  ζ = Σ_ℓ Σ_{v∈V_s} ζ_v^(ℓ)/p_v = Σ_ℓ Σ_e (b_e^(ℓ)/p_e) 1_e^(ℓ).

Two facts I'll lean on. First, I'll restrict to *independent edge sampling*: decide each edge e ∈ E independently, include it with probability p_e, under a budget Σ_e p_e = m so the expected number of sampled edges is a chosen constant m. Independence makes the covariance across distinct edges vanish. Second, in my scheme once an edge is in the subgraph it's there for *all* layers — there's no per-layer resampling — so 1_e^(ℓ) is the same Bernoulli 1_e for every ℓ. That means the *same* edge's indicator is perfectly correlated across layers: Cov(1_e^(ℓ₁), 1_e^(ℓ₂)) = Var(1_e) = p_e − p_e², while Cov across different edges is 0.

Compute the variance. For one edge, summing its self-terms and its within-edge cross-layer terms:

  Var(ζ) = Σ_e [ Σ_ℓ (b_e^(ℓ))²/p_e² + 2 Σ_{ℓ₁<ℓ₂} b_e^(ℓ₁)b_e^(ℓ₂)/p_e² ] · (p_e − p_e²).

Factor (p_e − p_e²)/p_e² = (1/p_e − 1), and the bracket is exactly (Σ_ℓ b_e^(ℓ))² (square of a sum = sum of squares + twice the cross terms). I'll sanity-check that collapse on one edge before trusting it for the whole derivation: take b_e^(ℓ) = (2, −0.5, 1.3) over three layers and p_e = 0.4. The bracket Σ_ℓ (b^ℓ)² + 2Σ_{ℓ₁<ℓ₂} b^{ℓ₁}b^{ℓ₂} = 7.275 + 0.565 = 7.84, and (Σ_ℓ b^ℓ)² = 2.8² = 7.84 — equal, so the cross-layer terms really do reassemble into the squared sum. With (1/0.4 − 1)·7.84 = 1.5·7.84 = 11.76, and directly Var((S/p_e)·1_e) = (2.8/0.4)²(0.4−0.16) = 49·0.24 = 11.76 — and a Monte-Carlo over the single Bernoulli edge gave 11.761. So

  Var(ζ) = Σ_e (1/p_e − 1)(Σ_ℓ b_e^(ℓ))² = Σ_e (Σ_ℓ b_e^(ℓ))²/p_e − Σ_e (Σ_ℓ b_e^(ℓ))².

The second sum doesn't depend on the p_e at all — it's a constant I can't touch. So minimizing variance means minimizing Σ_e (Σ_ℓ b_e^(ℓ))²/p_e subject to Σ_e p_e = m. This is a constrained minimization, and Cauchy–Schwarz bounds it below: for any positive p_e,

  [ Σ_e (Σ_ℓ b_e^(ℓ))²/p_e ] · [ Σ_e p_e ] ≥ ( Σ_e |Σ_ℓ b_e^(ℓ)| )²,

with equality exactly when (Σ_ℓ b_e^(ℓ))²/p_e ∝ p_e, i.e. when p_e ∝ |Σ_ℓ b_e^(ℓ)|. Since Σ_e p_e = m, that pins it: p_e = m·|Σ_ℓ b_e^(ℓ)| / Σ_{e'}|Σ_ℓ b_{e'}^(ℓ)|. Let me confirm it's actually a minimum and not just a stationary point, on two edges with |Σ_ℓ b| = (4, 1) and budget m=2. The proposed optimum is p ∝ (4,1), i.e. p=(1.6,0.4), giving objective 4²/1.6 + 1²/0.4 = 10 + 2.5 = 12.5, which equals the Cauchy–Schwarz lower bound (4+1)²/2 = 12.5 — so this allocation is tight, it can't be beaten. Probing neighbors confirms it: the uniform split (1,1) gives 17, the aggressive (1.9,0.1) gives 18.4, (1.5,0.5) gives 12.67, (1.7,0.3) gives 12.75 — every alternative is strictly worse, and the objective bottoms out right at p ∝ |b|. (For the real vector-valued b_e^(ℓ), minimizing the summed-over-dimensions variance gives the same form with the absolute value replaced by the norm ‖·‖.) So the variance-optimal edge probability is proportional to the magnitude of that edge's total contribution across layers.

That's the right answer but it's not *usable* as stated: Σ_ℓ b_e^(ℓ) depends on the transformed activations x̃, which change every step and are expensive to compute just to decide what to sample. So I simplify — drop the activation dependence and keep only the part of b_e that's pure topology. Strip x̃ out and what's left of b_e is governed by Â_{v,u} + Â_{u,v}. With the row-normalized Â = D^{-1}A, row v is scaled by 1/deg(v), so for an existing edge Â_{v,u} = 1/deg(v) and Â_{u,v} = 1/deg(u); plugging deg(u)=4, deg(v)=7 as a spot check, 1/7 + 1/4 = 0.3929, which is exactly 1/deg(u)+1/deg(v). So set

  p_e ∝ Â_{v,u} + Â_{u,v} = 1/deg(u) + 1/deg(v).

Now I can check this topology-only sampler against the influence intuition I started from, since the intuition was a separate line of argument and ought to either confirm or contradict the variance result. If u and v are connected and each has few neighbors, 1/deg(u)+1/deg(v) is large, so the edge is sampled with high probability — and two low-degree connected nodes are exactly the pair that most strongly influence each other (each is a large fraction of the other's neighborhood) and should ride in the same subgraph. Conversely a hub with degree 1000 connected to another hub gets ≈ 0.002, sampled rarely, which is also right: neither node depends much on that one edge. So the variance derivation and the influence heuristic point at the same distribution rather than fighting, which is some evidence I haven't reasoned myself into a corner.

It's worth seeing the contrast with layer sampling sharply, because the win turns out to be in *when* I sample, not which distribution I use. Suppose I took this very p_e and used it as a *layer* sampler in the FastGCN style — sample each layer's edges independently with this probability. Then connectivity follows exactly the (1−(1−p)^d)^{L−1} decay I computed earlier: 0.83, 0.69, 0.58, 0.40 as L goes 2, 3, 4, 6 — deeper means sparser minibatches. But in graph sampling the edge, once chosen, lives in every layer, so the survival probability through depth is identically 1 — connectivity never drops, regardless of L. Same edge probabilities, opposite behavior; the order of sampling is doing the work, not the distribution.

So I have a family of samplers. The edge sampler I just derived samples m undirected edges with probability ∝ 1/deg(u)+1/deg(v), then takes their endpoints. A node sampler samples n nodes with the FastGCN-inspired probability ∝ ‖Â_{:,v}‖², but uses the sampled nodes to induce one subgraph instead of independently sampling every layer. And random-walk samplers are a natural fit for an L-layer GCN: if I ignore activations, L stacked layers act like a single hop matrix B = Â^L, and the same variance logic would want to sample node *pairs* with p_{u,v} ∝ B_{u,v}+B_{v,u} — the probability a length-L walk connects u and v. I can't independently sample arbitrary pairs, but a length-h random walk approximates picking pairs that are mutually reachable, so an h-hop walk sampler with r roots is a sound proxy for an h-layer GCN. There's also a multi-dimensional / frontier random walk (Ribeiro–Towsley): maintain a frontier of r nodes, repeatedly pick one with probability ∝ its degree, step to a random neighbor, swap it into the frontier, collecting visited nodes. For *all* of these samplers I then take the *node-induced* subgraph — include every edge of G between the sampled nodes, even ones the sampler didn't traverse. The induction adds connections back, which empirically speeds up convergence: more intra-batch edges means each node has more support during propagation.

Let me write the real thing. The bones are: a preprocessing pass that samples a bunch of subgraphs and turns the node/edge counters into the normalization coefficients; a per-step routine that pops a sampled subgraph and builds a complete GCN on it; and the two normalizations wired in (aggregator α on the adjacency, loss λ on the per-node loss). The subgraph adjacency is first rescaled by the reciprocal aggregator factor, then row-normalized by the original training-graph degrees of the subgraph nodes; that preserves the intended full-graph D^{-1}A coefficient rather than replacing it with a subgraph-degree coefficient. The loss is multiplied by the reciprocal per-node loss factor.

```python
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- samplers: all return a set of node IDs; the subgraph is node-INDUCED ----
def sampler_edge(adj_train, budget_m, deg):
    # sample each undirected edge once; p_e ∝ 1/deg(u) + 1/deg(v)
    coo = sp.triu(adj_train, k=1).tocoo()
    rows, cols = coo.row, coo.col
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

def induce_with_edge_ids(adj_train, sub_nodes):
    # node-induced subgraph, plus original CSR positions for each retained directed edge
    sub_nodes = np.array(sorted(np.unique(sub_nodes)))
    pos = {v: i for i, v in enumerate(sub_nodes)}
    indptr, indices, edge_ids = [0], [], []
    for v in sub_nodes:
        start, end = adj_train.indptr[v], adj_train.indptr[v + 1]
        for eid in range(start, end):
            u = adj_train.indices[eid]
            if u in pos:
                indices.append(pos[u]); edge_ids.append(eid)
        indptr.append(len(indices))
    data = np.ones(len(indices), dtype=np.float32)
    return sp.csr_matrix((data, indices, indptr), shape=(len(sub_nodes), len(sub_nodes))), np.array(edge_ids)

# ---- preprocessing: estimate aggregator (alpha) and loss (lambda) normalization ----
def estimate_norms(adj_train, sampler, train_nodes, sample_coverage=50):
    n = adj_train.shape[0]
    num_train = len(train_nodes)
    C_v   = np.zeros(n)                                    # counts of node v over subgraphs
    C_uv  = np.zeros(adj_train.nnz)                        # counts of edge (u,v) over subgraphs
    subgraphs, total = [], 0
    while total <= sample_coverage * num_train:            # sample until enough coverage
        sub = sampler(adj_train)
        subgraphs.append(sub); total += len(sub)
        C_v[sub] += 1
        sub_adj, edge_ids = induce_with_edge_ids(adj_train, sub)
        C_uv[edge_ids] += 1
    N = len(subgraphs)
    # reciprocal 1/alpha for each global CSR edge leaving v: C_v / C_uv, with finite guards
    rows = np.repeat(np.arange(n), np.diff(adj_train.indptr))
    aggr_norm = np.divide(C_v[rows], C_uv, out=np.full_like(C_uv, 0.1), where=C_uv > 0)
    aggr_norm = np.clip(aggr_norm, 0, 1e4)
    # loss normalization stored as reciprocal 1/lambda = N / (C_v * |V_train|)
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
            h = F.relu(torch.sparse.mm(adj_norm, W(h)))    # σ( Â_α · (W h) ), Â already aggr-normalized
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
        if i >= len(subgraphs):                            # reuse the pre-sampled subgraphs as batches
            subgraphs += [sampler(adj_train) for _ in range(len(subgraphs))]
        sub = subgraphs[i]; i += 1
        sub_adj, edge_ids = induce_with_edge_ids(adj_train, sub)
        sub_adj.data = aggr_norm[edge_ids]                  # reciprocal aggregator norm: C_v/C_uv
        sub_adj = sp.diags(1.0 / np.clip(deg_train[sub], 1, None)).dot(sub_adj)
        adj_norm = to_torch_sparse(sub_adj)                 # D_train^{-1}(A_s / alpha)
        logits = model(adj_norm, features[sub])
        per_node = loss_fn(logits, labels[sub], reduction='none')
        loss = (per_node * loss_norm[sub]).sum()           # loss normalization: divide each L_v by lambda_v
        opt.zero_grad(); loss.backward(); opt.step()
```
