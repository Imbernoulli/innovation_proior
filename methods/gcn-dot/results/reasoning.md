Let me start from what actually hurts. I have a graph — say a citation network — handed to me as an adjacency matrix A over N nodes, and usually a feature matrix X where row i is some description of node i, a bag-of-words for a paper. Part of the graph is hidden: somebody pulled out a chunk of the real edges, and I have to score every candidate node pair so the held-out true edges float to the top and the genuinely-unconnected pairs sink. That is link prediction, and it is the same computation behind friend recommendation, paper recommendation, filling in a protein-interaction network. No labels anywhere — this has to be learned from the connectivity I can see, plus the features. So the whole game is: out of A and X, manufacture a vector z_i per node such that some cheap function of (z_i, z_j) tells me whether i and j should be linked.

What do people do today, and why does none of it give me what I want? There's a whole tradition of hand-designed similarity scores. Common neighbors: f(i,j) = |Γ(i) ∩ Γ(j)|, the number of shared neighbors — two papers that cite many of the same papers probably belong together. Adamic-Adar refines it, Σ over shared neighbors z of 1/log|Γ(z)|, because a neighbor that everyone shares is uninformative and should count for less. Katz sums all walks between i and j with longer walks discounted. These are real and they work — when the graph's formation mechanism happens to match the heuristic you picked. But each one is a single fixed lens on the topology. If the true structure isn't "links follow common neighbors" or "links follow degree product," the heuristic is blind to it. And crucially, every one of them reads only the adjacency; none can touch X. If two papers have never shared a citation yet have nearly identical abstracts, common-neighbors says zero and the features that scream "these should link" go unused. I don't want to pre-commit to one structural lens, and I refuse to throw away the features.

So who learns the embedding instead of hand-designing it? Matrix factorization. Approximate the adjacency by a low-rank product, Â_ij = z_i^T z_j, fit the z_i by minimizing reconstruction error Σ_{(i,j)∈Ω} (A_ij − z_i^T z_j)² over the available pair labels Ω, then predict an unseen link by the inner product of the two fitted vectors. I like a lot about this. It is *learned*, not hand-designed: whatever low-rank structure best explains the observed adjacency, the optimization finds it, and a single global objective ties every node's vector to every other node in its component. And the scoring rule that drops out is beautiful in its simplicity — the inner product z_i^T z_j. Let me sit with why that's the natural scorer for a moment, because I think it's more than a convenience. If A is roughly a low-rank PSD matrix, then A ≈ Z Z^T for some Z, and the (i,j) entry of Z Z^T is exactly z_i^T z_j. So "reconstruct the adjacency" and "score a pair by the inner product of their latent vectors" are the *same* statement — the decoder is forced on me by the factorization, it isn't an arbitrary choice. Connected nodes get embeddings that align (large positive inner product); unconnected nodes get embeddings that are orthogonal or anti-aligned. The dot product is, up to norms, a cosine similarity, and homophily — the tendency of linked nodes to be alike — is precisely what a similarity-in-latent-space decoder encodes.

But matrix factorization has the same disease as the heuristics on the feature front: the z_i are fit per node from the adjacency alone. There's nowhere to plug X in. It's transductive — the vector belongs to a node ID, not to a node's content — and it needs a surprisingly large rank to even reproduce a simple heuristic like common-neighbors. DeepWalk is the same story dressed up: sample random walks from each node, treat each walk as a sentence of node IDs, run SkipGram so nodes that co-occur in walks get similar vectors. That captures multi-hop neighborhood proximity nicely, but it's a two-stage pipeline (generate walks, then separately train SkipGram), and once again the model is a lookup table Φ ∈ R^{|V|×d} indexed by node ID, with no port for features. Spectral clustering, embedding by the bottom Laplacian eigenvectors, is yet another flavor of "structure-only, transductive, no learned feature map." The pattern is glaring: the embedding methods that I like for being learned and end-to-end-ish are all structure-only and can't read X.

So here's the tension. The heuristics and embeddings have a clean decoder I love — score a pair by latent similarity, ideally the inner product, which falls out of factorizing the adjacency — but a hopeless *encoder*: a per-node lookup table that ignores features. Meanwhile, there's a piece of machinery sitting in a completely different room that is a fantastic encoder of exactly the thing I'm missing. Graph convolutional networks. A GCN layer is H^{(l+1)} = σ(Â_norm H^{(l)} W^{(l)}) with H^{(0)} = X and Â_norm = D̃^{-1/2} Ã D̃^{-1/2}, Ã = A + I. Read it slowly: it takes node representations, mixes each node with its neighbors through the normalized propagation matrix, multiplies by a learned weight, applies a nonlinearity. Self-loops (the +I) keep a node's own features in the mix; the symmetric normalization weights the edge (i,j) by 1/√(d̃_i d̃_j) instead of plain averaging, so a high-degree neighbor doesn't dominate, and it keeps the operator symmetric with a real spectrum, which is why the renormalization trick — folding the self-loop into Ã so the eigenvalues stay bounded — makes it stable to stack. The output is a per-node vector that *jointly* depends on the node's features and the features of its neighborhood out to the number of layers. That is precisely the encoder the embedding methods lack: a differentiable map f(X, A) → Z that reads features and structure together.

But the GCN as I've seen it used is a *supervised* model. You stack two layers, put a softmax on top, and minimize cross-entropy against given node labels. It produces wonderful per-node representations — but it was trained to classify nodes, with labels I don't have, and it never once scores a pair of nodes for whether an edge exists between them. The unsupervised, relational objective is just not there. So the question crystallizes: can I drive a GCN encoder with an *unsupervised* objective defined over node pairs, instead of a supervised one over node labels?

Now the two rooms snap together. I have a great encoder with no objective, and a great decoder-plus-objective with no encoder. Bolt them: let the GCN be the encoder that produces Z = GCN(X, A), and let the inner-product matrix-factorization rule be the decoder that turns Z back into a predicted adjacency. The encoder maps the graph into a latent space; the decoder reconstructs the graph from the latent space; train the whole thing to make the reconstruction match the observed graph. That is an auto-encoder — but the data being auto-encoded is not an image or a vector, it's the *connectivity* of the graph, and the encoder is graph-aware. The "code" is the embedding matrix Z; the reconstruction is the adjacency.

Let me write it concretely. Encode with a two-layer GCN — first layer ReLU, second layer linear to a latent dimension F — so Z = Â_norm ReLU(Â_norm X W_0) W_1, an N×F matrix of node embeddings. Decode by reconstructing the full adjacency from inner products: the logit for pair (i,j) is z_i^T z_j, and as a matrix the reconstructed adjacency is Â = σ(Z Z^T), with σ the logistic sigmoid squashing each inner product into a (0,1) edge probability. Eq for the whole non-probabilistic model: Z = GCN(X, A), Â = σ(Z Z^T). The decoder has zero parameters — all the model's capacity lives in the encoder's W_0, W_1 — which I find appealing: the inner product is a hard inductive bias (latent similarity = link), and the encoder does all the work of arranging the latent space so that bias is right.

Now the objective. I have predicted edge probabilities σ(z_i^T z_j) and the true binary adjacency A_ij ∈ {0,1}. The honest probabilistic model is: A_ij is Bernoulli with success probability σ(z_i^T z_j). The negative log-likelihood of a Bernoulli is binary cross-entropy, so the loss over all node pairs is

  L = Σ_{i,j} [ −A_ij log Â_ij − (1 − A_ij) log(1 − Â_ij) ],

with Â_ij = σ(z_i^T z_j). Minimizing this pushes the inner product up for true edges and down for non-edges. Why cross-entropy rather than the squared error that vanilla matrix factorization used? Because A is *binary*. Modeling a binary observation as a Gaussian and using MSE is a model mismatch; the Bernoulli likelihood is the right one, and its loss — cross-entropy on the logit z_i^T z_j — is also numerically stable when computed directly from logits rather than from the sigmoided probability.

I have to actually try to optimize this, because something is going to go wrong. The sum is over all N² pairs. In a real citation network, the number of true edges |E| is a tiny fraction of N² — these graphs are extremely sparse, maybe a fraction of a percent of pairs are edges. So in that double sum, the (1 − A_ij) term — the non-edges — outnumbers the A_ij term by hundreds or thousands to one. If I just minimize the raw cross-entropy, the gradient is dominated by "make all these non-edges score low," and the trivial solution "predict almost nothing is an edge" already drives the loss way down. The few true edges drown. The model learns to say "no edge" everywhere and calls it a day. Wall.

The disease is class imbalance, so I rebalance. There are two practical ways to do it, depending on whether I train against the dense adjacency or against sampled pairs. In the dense all-pairs version, I up-weight the positive (A_ij = 1) entries by the ratio of negatives to positives. For the training adjacency before adding the reconstruction self-loops, let ΣA be the number of positive entries; then pos_weight = (N² − ΣA)/ΣA, and a normalization constant norm = N²/((N² − ΣA)·2) keeps the loss on a comparable scale. The flattened dense labels can then be `adj_train + I`, while the constants still come from `adj_train`. This is the weighted cross-entropy used when the whole reconstructed adjacency is materialized. In the sampled-pair version, I don't sum over all N² non-edges at all — for each training step, I subsample a set of non-edges comparable in size to the true edges, and compute the loss only on the true edges plus this sampled negative set. Pick negatives by sampling random node pairs that aren't edges. Then the loss is

  L = − mean over positive edges of log σ(z_i^T z_j) − mean over sampled negative edges of log(1 − σ(z_i^T z_j)),

balanced by construction. Either way the fix is the same idea: the all-pairs cross-entropy is structurally biased toward the majority class, and I either reweight the minority or subsample the majority so the rare edges carry their fair share of the signal. I'll keep the full weighted form when I can afford the dense N² reconstruction on small graphs and use the sampled-pair form when I want a sparse training loop.

That's the whole non-probabilistic model and I could ship it: GCN encoder, inner-product decoder, balanced cross-entropy. Z = GCN(X, A), Â = σ(Z Z^T). Let me call this the graph auto-encoder. Before I commit, let me ask whether I'm leaving anything on the table. The latent space Z is a deterministic point per node. If a graph has community structure, then reconstructing many within-community links rewards the encoder for aligning those nodes, so the geometry of Z can carry useful structure. But a purely deterministic auto-encoder has no pressure toward a regular latent space; it can scatter embeddings however minimizes reconstruction loss, including pathological spread. I would like the latent space to be smooth and well-organized, not just good enough to reconstruct edges.

There's a framework built exactly for "auto-encode, but with a well-behaved latent space" — the variational auto-encoder. Its setup: data x is generated from a latent z by z ∼ p(z), x ∼ p_θ(x|z), with the true posterior p(z|x) intractable; you introduce a recognition model q_φ(z|x) — a probabilistic encoder — to approximate it, and you train by maximizing the evidence lower bound. Let me see whether my graph problem fits that mold. The "data" is the adjacency A. The latent is the embedding matrix Z. I already have a generative story for A given Z — that's exactly my inner-product decoder, p(A_ij = 1 | z_i, z_j) = σ(z_i^T z_j), so p(A | Z) = Π_{i,j} p(A_ij | z_i, z_j). And the encoder, instead of producing a point z_i, should produce a *distribution* over z_i. So make the inference model a GCN that outputs, per node, a mean and a log standard deviation:

  q(Z | X, A) = Π_i q(z_i | X, A),   q(z_i | X, A) = N(z_i | μ_i, diag(σ_i²)),

with μ = GCN_μ(X, A) and log σ = GCN_σ(X, A). I'll have the two GCNs share their first layer W_0 — the first convolution extracts shared features from (X, A) and only the second layer splits into a mean head and a log-standard-deviation head — because there's no reason to learn the feature extraction twice and the parameter economy helps on small graphs. I output log σ rather than σ directly so the network head is unconstrained on the reals and exp(log σ) is automatically positive; when I code it, I can clamp this log value from above for numerical safety.

The objective is the ELBO:

  L = E_{q(Z|X,A)}[ log p(A | Z) ] − KL[ q(Z | X, A) || p(Z) ],

with a standard normal prior p(Z) = Π_i N(z_i | 0, I). The first term is the expected reconstruction log-likelihood — the same balanced cross-entropy as before, now averaged over latent samples — and the second is a KL divergence pulling each node's posterior toward N(0, I). That KL is the regularizer I was reaching for: it stops the encoder from scattering embeddings arbitrarily, anchoring the latent space to a smooth prior and giving exactly the well-organized geometry I wanted.

Two things have to be handled to train this. First, the expectation over q is intractable to differentiate naively — the gradient w.r.t. the encoder parameters of E_{q_φ}[·] has notoriously high variance if I push the gradient through the sampling. The reparameterization trick fixes it: instead of sampling z_i ∼ N(μ_i, σ_i²) opaquely, write z_i = μ_i + σ_i ⊙ ε_i with ε_i ∼ N(0, I), so the stochasticity lives in ε and the gradient flows cleanly through μ_i and σ_i. In matrix form, Z = μ + ε ⊙ exp(log σ), ε standard normal, and at evaluation time I just use the mean, Z = μ. Second, the KL term: since both q(z_i | X, A) = N(μ_i, σ_i²) and the prior N(0, I) are Gaussian, the KL is analytic, no sampling needed. Let me actually derive it for one coordinate so I'm sure of the sign and the factors, because this is exactly the kind of place a stray minus sign hides. For two univariate Gaussians, KL(N(μ, σ²) || N(0, 1)) = ∫ N(μ,σ²) [log N(μ,σ²) − log N(0,1)]. The log densities are log N(x; μ, σ²) = −½ log(2πσ²) − (x−μ)²/(2σ²) and log N(x; 0,1) = −½ log(2π) − x²/2. Subtract: log N(μ,σ²) − log N(0,1) = −½ log σ² − (x−μ)²/(2σ²) + x²/2. Take the expectation under x ∼ N(μ, σ²): E[(x−μ)²] = σ², so the middle term is −σ²/(2σ²) = −½; E[x²] = μ² + σ², so the last term is (μ² + σ²)/2. Collecting,

  KL = −½ log σ² − ½ + (μ² + σ²)/2 = ½( −log σ² − 1 + μ² + σ² ) = −½( 1 + log σ² − μ² − σ² ).

So the per-coordinate KL is −½(1 + log σ² − μ² − σ²), and the total KL sums this over all latent dimensions and all nodes. The reconstruction term wants to *maximize* the ELBO, so in a loss I minimize, the contribution is −ELBO = (balanced cross-entropy reconstruction) + KL, i.e. I add ½ Σ(−1 − log σ² + μ² + σ²). Writing log σ as the network's output, log σ² = 2 log σ, the node-averaged positive KL is −½ mean over nodes of Σ_dim(1 + 2 log σ − μ² − exp(2 log σ)); if the reconstruction loss is averaged over all N² entries, I scale this node-averaged KL by 1/N so it lives on the same per-entry scale. Good — the signs check out and it's the same closed form the VAE uses, just applied per node.

Now I should stress-test this combination rather than declare victory, because there's a tension between the two pieces I just glued. The prior is N(0, I), centered at the origin. But look at what the inner-product decoder *wants*: to make an edge confident, σ(z_i^T z_j) near 1, it needs z_i^T z_j large and positive, which means it wants the embeddings to have *large norm* and point in aligned directions — it pushes embeddings *away* from the zero center. The KL regularizer, meanwhile, pulls them *toward* the zero center and toward unit-ish scale. So the decoder and the prior are fighting: a Gaussian prior centered at zero is arguably a poor match for a decoder that profits from pushing embeddings outward. I can accept that as a modeling tradeoff rather than a bug: the prior gives me regularization and a smoother latent space, while the decoder asks for confident similarity scores. This is the kind of thing the deterministic auto-encoder sidesteps entirely (no prior, no tension) at the cost of an unregularized latent space. So I have a family: the deterministic graph auto-encoder, Z = GCN(X, A), Â = σ(Z Z^T), trained on balanced cross-entropy — clean, parameter-light decoder, no latent regularization; and its variational sibling, which adds the Gaussian posterior, the reparameterized sampling, and the KL regularizer for a smoother latent space at the cost of the prior-decoder tension.

What does this give me over the baselines I was unhappy with? Both members read node features *and* structure through the GCN encoder, so the moment X carries signal that connectivity lacks, the model can use it — exactly what DeepWalk, spectral clustering, and plain matrix factorization could not do. And it's a single end-to-end objective: one cross-entropy (or one ELBO), backpropagated through the encoder, rather than a walk-generation stage followed by a separately optimized SkipGram. For the featureless case — if I genuinely have no X — I can still run the model by setting X to the identity matrix I_N, which makes each node a one-hot indicator and gives a structure-only control against the baselines. The clean test is then paired: run the same encoder-decoder with real features and with identity features, and see how much signal X contributes.

I should be clear-eyed about what the inner-product decoder cannot do, because it bounds the method. It scores a pair purely by symmetric bilinear similarity, so it is a natural fit for undirected homophily/proximity but not a universal edge model. If the graph's link rule needs a richer pair interaction than "large aligned latent vectors mean likely edge," the decoder is the place where capacity would have to increase. But the inner product is the canonical, zero-parameter choice that defines this auto-encoder and that ties it cleanly back to matrix factorization, so that's where I land for the base model.

Defaults follow from the setting and from the encoder I inherited. These citation graphs are small enough to train full-batch, so I do plain full-batch gradient descent with Adam at learning rate 0.01, the same rate that worked for the GCN node-classification setup. A 32-dimensional first hidden layer gives the encoder some room before the final bottleneck, and a 16-dimensional latent space keeps the factorization low-rank and parameter-light. Glorot initialization for the weights, again carried over from the GCN. Train for a couple hundred iterations. None of these are load-bearing for the idea; they're the standard small-graph full-batch recipe.

Let me write the model as the code I'd actually ship, filling the two empty slots in the harness — the unsupervised encoder and the pair-scoring rule with its loss. The encoder is a two-layer GCN; the decoder is the inner product; the loss is the balanced reconstruction objective, with the variational sibling adding the reparameterized sampling and the analytic KL.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


def normalized_adjacency(A):
    """D̃^{-1/2} Ã D̃^{-1/2} with Ã = A + I  (renormalization trick)."""
    A = A + torch.eye(A.size(0), device=A.device)          # add self-loops -> Ã
    d = A.sum(1)                                            # degrees of Ã
    d_inv_sqrt = d.pow(-0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
    return d_inv_sqrt.unsqueeze(1) * A * d_inv_sqrt.unsqueeze(0)


class GraphConv(nn.Module):
    """One GCN layer: H' = Â_norm (H W)."""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, H, A_norm):
        return A_norm @ self.W(H)


def inner_product_scores(z, pair_index, sigmoid=False):
    """Decoder: logit for pair (i,j) is z_i^T z_j  (Â = σ(ZZ^T))."""
    s = (z[pair_index[0]] * z[pair_index[1]]).sum(dim=-1)
    return torch.sigmoid(s) if sigmoid else s


class GAE(nn.Module):
    """Graph Auto-Encoder: GCN encoder -> embeddings; inner-product decoder
    reconstructs the adjacency.  Z = GCN(X, A);  Â = σ(Z Z^T)."""

    def __init__(self, in_dim, hidden_dim=32, emb_dim=16):
        super().__init__()
        self.gc1 = GraphConv(in_dim, hidden_dim)           # first conv (shared)
        self.gc2 = GraphConv(hidden_dim, emb_dim)          # linear conv -> embeddings

    def encode(self, X, A_norm):
        H = F.relu(self.gc1(X, A_norm))                    # ReLU first layer
        return self.gc2(H, A_norm)                         # Z, linear second layer

    def recon_loss(self, z, pos_edge_index, neg_edge_index):
        # balanced binary cross-entropy: pull σ(z_i·z_j) up on true edges, down
        # on sampled non-edges (subsampling cures the N^2 class imbalance)
        eps = 1e-15
        pos = inner_product_scores(z, pos_edge_index, sigmoid=True)
        neg = inner_product_scores(z, neg_edge_index, sigmoid=True)
        return -(torch.log(pos + eps).mean() + torch.log(1 - neg + eps).mean())


class VGAE(GAE):
    """Variational sibling: GCN posterior q(z_i)=N(μ_i, σ_i^2); reparameterized
    sampling; KL(q || N(0,I)) regularizer on top of the reconstruction term."""

    def __init__(self, in_dim, hidden_dim=32, emb_dim=16):
        super().__init__(in_dim, hidden_dim, emb_dim)
        self.gc1 = GraphConv(in_dim, hidden_dim)           # shared first layer W_0
        self.gc_mu = GraphConv(hidden_dim, emb_dim)        # mean head
        self.gc_logstd = GraphConv(hidden_dim, emb_dim)    # log-std head

    def encode(self, X, A_norm):
        H = F.relu(self.gc1(X, A_norm))
        self.mu = self.gc_mu(H, A_norm)
        self.logstd = self.gc_logstd(H, A_norm).clamp(max=10.0)
        if self.training:                                  # reparameterize: z = μ + σ⊙ε
            return self.mu + torch.randn_like(self.logstd) * torch.exp(self.logstd)
        return self.mu                                     # use the mean at eval time

    def kl_loss(self):
        # per-node KL(N(μ,σ^2) || N(0,I)) = -½ Σ (1 + 2logσ − μ² − exp(2logσ))
        return -0.5 * torch.mean(
            torch.sum(1 + 2 * self.logstd - self.mu**2 - torch.exp(2 * self.logstd), dim=1))


def train(model, X, A_norm, pos_edge_index, num_nodes, n_iters=200, lr=0.01, variational=False):
    from torch_geometric.utils import negative_sampling
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(n_iters):
        model.train()
        optimizer.zero_grad()
        z = model.encode(X, A_norm)
        neg_edge_index = negative_sampling(pos_edge_index, z.size(0))
        loss = model.recon_loss(z, pos_edge_index, neg_edge_index)
        if variational:
            loss = loss + (1.0 / num_nodes) * model.kl_loss()   # negative ELBO scaling
        loss.backward()
        optimizer.step()
```

Let me retrace the causal chain. I started needing to score node pairs for missing links from connectivity plus features, with no labels. Hand-designed heuristics gave a fixed structural lens and ignored features; the learned embedding methods — matrix factorization, DeepWalk, spectral clustering — gave a clean similarity decoder (the inner product falls straight out of factorizing the adjacency) but a hopeless encoder, a per-node lookup table blind to features and transductive. Separately, the graph convolutional network was a superb encoder of features-and-structure, but it had only ever been driven by a supervised node-label objective and never scored pairs. Gluing them solved both: let the GCN be the encoder, Z = GCN(X, A), and let the inner-product matrix-factorization rule be the decoder, Â = σ(Z Z^T) — an auto-encoder whose data is the graph's connectivity and whose code is the embedding. Training it as a Bernoulli edge model gives a cross-entropy loss, which on these sparse graphs is dominated by non-edges, so I rebalanced — up-weight the rare positive terms in the dense all-pairs loss or subsample negatives in the sampled-pair loss — to keep the true edges' signal alive. That's the deterministic graph auto-encoder. Wanting a smoother, regularized latent space, I cast the same encoder-decoder as a variational auto-encoder: a GCN posterior outputting per-node mean and log standard deviation (sharing the first layer), the reparameterization trick to get low-variance gradients, and the analytic Gaussian KL toward an N(0, I) prior — deriving the per-coordinate KL = −½(1 + log σ² − μ² − σ²) by hand and noting the genuine tension that the inner-product decoder wants large-norm embeddings while the zero-centered prior pulls them in. Both members read features and train end-to-end, the inner product is the canonical zero-parameter decoder that ties everything back to factorization, and the sparse implementation drops into a full-batch encoder with sampled negatives each step.
