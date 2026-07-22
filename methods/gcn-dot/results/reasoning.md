Let me start from what actually hurts. I have a graph — say a citation network — handed to me as an adjacency matrix A over N nodes, and usually a feature matrix X where row i is some description of node i, a bag-of-words for a paper. Part of the graph is hidden: somebody pulled out a chunk of the real edges, and I have to score every candidate node pair so the held-out true edges float to the top and the genuinely-unconnected pairs sink. That is link prediction, and it is the same computation behind friend recommendation, paper recommendation, filling in a protein-interaction network. No labels anywhere — this has to be learned from the connectivity I can see, plus the features. So the whole game is: out of A and X, manufacture a vector z_i per node such that some cheap function of (z_i, z_j) tells me whether i and j should be linked.

What do people do today, and why does none of it give me what I want? There's a whole tradition of hand-designed similarity scores. Common neighbors: f(i,j) = |Γ(i) ∩ Γ(j)|, the number of shared neighbors — two papers that cite many of the same papers probably belong together. Adamic-Adar refines it, Σ over shared neighbors z of 1/log|Γ(z)|, because a neighbor that everyone shares is uninformative and should count for less. Katz sums all walks between i and j with longer walks discounted. These are real and they work — when the graph's formation mechanism happens to match the heuristic you picked. But each one is a single fixed lens on the topology. If the true structure isn't "links follow common neighbors" or "links follow degree product," the heuristic is blind to it. And crucially, every one of them reads only the adjacency; none can touch X. If two papers have never shared a citation yet have nearly identical abstracts, common-neighbors says zero and the features that scream "these should link" go unused. I don't want to pre-commit to one structural lens, and I refuse to throw away the features.

So who learns the embedding instead of hand-designing it? Matrix factorization. Approximate the adjacency by a low-rank product, Â_ij = z_i^T z_j, fit the z_i by minimizing reconstruction error Σ_{(i,j)∈Ω} (A_ij − z_i^T z_j)² over the available pair labels Ω, then predict an unseen link by the inner product of the two fitted vectors. I like a lot about this. It is *learned*, not hand-designed: whatever low-rank structure best explains the observed adjacency, the optimization finds it, and a single global objective ties every node's vector to every other node in its component. And notice the scoring rule it ships with: the inner product z_i^T z_j. Let me check whether that's a coincidence or whether it's forced, because if it's forced it tells me something about the right decoder. If A is roughly low-rank, I write A ≈ Z Z^T and the predicted entry is (Z Z^T)_{ij}. Is (Z Z^T)_{ij} the same thing as z_i^T z_j? Let me just compute it on a tiny case to be sure I'm not fooling myself. Take three nodes in two dimensions, z_0=(1, 0.5), z_1=(−0.3, 2), z_2=(0.8, 0.8). Stacking these as rows of Z and forming Z Z^T:

  Z Z^T = [[1.25, 0.70, 1.20], [0.70, 4.09, 1.36], [1.20, 1.36, 1.28]].

Take the off-diagonal entry (0,1): it's 0.70. And z_0·z_1 = 1·(−0.3) + 0.5·2 = −0.3 + 1.0 = 0.70. They match, and the same holds entry by entry across the whole matrix (I checked all nine). So it is *not* a coincidence — the (i,j) entry of the reconstruction literally is the inner product of node i's and node j's latent vectors. That means "reconstruct the adjacency by a low-rank product" and "score a pair by the inner product of their latent vectors" are the same operation written two ways; the decoder is forced on me by the factorization, not chosen for taste. Connected nodes get embeddings that align (large positive inner product); unconnected nodes get embeddings that are orthogonal or anti-aligned. The dot product is, up to norms, a cosine similarity, and homophily — the tendency of linked nodes to be alike — is precisely what a similarity-in-latent-space decoder encodes.

But matrix factorization has the same disease as the heuristics on the feature front: the z_i are fit per node from the adjacency alone. There's nowhere to plug X in. It's transductive — the vector belongs to a node ID, not to a node's content — and it needs a surprisingly large rank to even reproduce a simple heuristic like common-neighbors. DeepWalk is the same story dressed up: sample random walks from each node, treat each walk as a sentence of node IDs, run SkipGram so nodes that co-occur in walks get similar vectors. That captures multi-hop neighborhood proximity nicely, but it's a two-stage pipeline (generate walks, then separately train SkipGram), and once again the model is a lookup table Φ ∈ R^{|V|×d} indexed by node ID, with no port for features. Spectral clustering, embedding by the bottom Laplacian eigenvectors, is yet another flavor of "structure-only, transductive, no learned feature map." The pattern is glaring: the embedding methods that I like for being learned and end-to-end-ish are all structure-only and can't read X.

So here's the tension. The heuristics and embeddings have a clean decoder I love — score a pair by latent similarity, ideally the inner product, which falls out of factorizing the adjacency — but a hopeless *encoder*: a per-node lookup table that ignores features. Meanwhile, there's a piece of machinery sitting in a completely different room that is a fantastic encoder of exactly the thing I'm missing. Graph convolutional networks. A GCN layer is H^{(l+1)} = σ(Â_norm H^{(l)} W^{(l)}) with H^{(0)} = X and Â_norm = D̃^{-1/2} Ã D̃^{-1/2}, Ã = A + I. Read it slowly: it takes node representations, mixes each node with its neighbors through the normalized propagation matrix, multiplies by a learned weight, applies a nonlinearity. Self-loops (the +I) keep a node's own features in the mix; the symmetric normalization weights the edge (i,j) by 1/√(d̃_i d̃_j) instead of plain averaging, so a high-degree neighbor doesn't dominate, and it keeps the operator symmetric with a real spectrum, which is why the renormalization trick — folding the self-loop into Ã so the eigenvalues stay bounded — makes it stable to stack. The output is a per-node vector that *jointly* depends on the node's features and the features of its neighborhood out to the number of layers. That is precisely the encoder the embedding methods lack: a differentiable map f(X, A) → Z that reads features and structure together.

But the GCN as I've seen it used is a *supervised* model. You stack two layers, put a softmax on top, and minimize cross-entropy against given node labels. It produces wonderful per-node representations — but it was trained to classify nodes, with labels I don't have, and it never once scores a pair of nodes for whether an edge exists between them. The unsupervised, relational objective is just not there. So the question crystallizes: can I drive a GCN encoder with an *unsupervised* objective defined over node pairs, instead of a supervised one over node labels?

Now the two rooms snap together. I have a great encoder with no objective, and a great decoder-plus-objective with no encoder. Bolt them: let the GCN be the encoder that produces Z = GCN(X, A), and let the inner-product matrix-factorization rule be the decoder that turns Z back into a predicted adjacency. The encoder maps the graph into a latent space; the decoder reconstructs the graph from the latent space; train the whole thing to make the reconstruction match the observed graph. That is an auto-encoder — but the data being auto-encoded is not an image or a vector, it's the *connectivity* of the graph, and the encoder is graph-aware. The "code" is the embedding matrix Z; the reconstruction is the adjacency.

Let me write it concretely. Encode with a two-layer GCN — first layer ReLU, second layer linear to a latent dimension F — so Z = Â_norm ReLU(Â_norm X W_0) W_1, an N×F matrix of node embeddings. Decode by reconstructing the full adjacency from inner products: the logit for pair (i,j) is z_i^T z_j, and as a matrix the reconstructed adjacency is Â = σ(Z Z^T), with σ the logistic sigmoid squashing each inner product into a (0,1) edge probability. Eq for the whole non-probabilistic model: Z = GCN(X, A), Â = σ(Z Z^T). The decoder has zero parameters — all the model's capacity lives in the encoder's W_0, W_1 — which I find appealing: the inner product is a hard inductive bias (latent similarity = link), and the encoder does all the work of arranging the latent space so that bias is right.

Now the objective. I have predicted edge probabilities σ(z_i^T z_j) and the true binary adjacency A_ij ∈ {0,1}. The honest probabilistic model is: A_ij is Bernoulli with success probability σ(z_i^T z_j). The negative log-likelihood of a Bernoulli is binary cross-entropy, so the loss over all node pairs is

  L = Σ_{i,j} [ −A_ij log Â_ij − (1 − A_ij) log(1 − Â_ij) ],

with Â_ij = σ(z_i^T z_j). Does minimizing this actually move the logits the way I want? Let me differentiate one term w.r.t. the logit s = z_i^T z_j. For a true edge the term is −log σ(s), with derivative −(1 − σ(s)), which is negative everywhere (at s=0 it's −0.5, at s=2 it's −0.12); a negative gradient means gradient descent *raises* s. For a non-edge the term is −log(1 − σ(s)), with derivative +σ(s), positive everywhere (at s=0 it's +0.5, at s=2 it's +0.88); a positive gradient means descent *lowers* s. So yes — minimizing this pulls the inner product up on true edges and down on non-edges, and the push is strongest exactly where the prediction is most wrong. Why cross-entropy rather than the squared error that vanilla matrix factorization used? Because A is *binary*. Modeling a binary observation as a Gaussian and using MSE is a model mismatch; the Bernoulli likelihood is the right one, and its loss — cross-entropy on the logit z_i^T z_j — is also numerically stable when computed directly from logits rather than from the sigmoided probability.

I have to actually try to optimize this, because something is going to go wrong. The sum is over all N² pairs. Let me put real numbers on the sparsity instead of hand-waving "a fraction of a percent." Cora has N ≈ 2708 nodes and about 5278 undirected edges, so the symmetric A has 2·5278 = 10556 positive entries out of N² = 7,333,264 — that's 0.144% of pairs, and the negatives outnumber the positives 7,322,708 to 10,556, a ratio of about 694 to 1. Now suppose the model gives up and predicts a tiny constant probability everywhere, say σ(z_i^T z_j) ≈ 10⁻³ for every pair. What's the raw mean cross-entropy? Each non-edge contributes −log(1−10⁻³) ≈ 0.0010, each true edge contributes −log(10⁻³) ≈ 6.9. Averaged over all N² pairs: the non-edge mass is (7,322,708 · 0.0010)/7,333,264 ≈ 0.00100, the true-edge mass is (10,556 · 6.9)/7,333,264 ≈ 0.0099, total ≈ 0.011. So "predict almost nothing is an edge" already sits at a loss of ~0.011, and — this is the damning part — even getting *every single true edge catastrophically wrong* only costs 0.0099 of that, because the 694:1 dilution divides the edge penalty down to nothing in the mean. The gradient is dominated by "make these millions of non-edges score low," the rare edges barely register, and gradient descent will happily walk to the all-zeros predictor. The few true edges drown. Wall.

The disease is class imbalance, so I rebalance. There are two practical ways to do it, depending on whether I train against the dense adjacency or against sampled pairs. In the dense all-pairs version, I up-weight the positive (A_ij = 1) entries by the ratio of negatives to positives. For the training adjacency before adding the reconstruction self-loop labels, let ΣA be the number of positive entries; then pos_weight = (N² − ΣA)/ΣA, and a normalization constant norm = N²/((N² − ΣA)·2) keeps the loss on a comparable scale. The flattened dense labels are the training adjacency with identity added, while the rebalancing constants still come from the training adjacency before those self-loop labels are added. This is the weighted cross-entropy used when the whole reconstructed adjacency is materialized. In the sampled-pair version, I don't sum over all N² non-edges at all — for each training step, I subsample a set of non-edges comparable in size to the true edges, and compute the loss only on the true edges plus this sampled negative set. Pick negatives by sampling random node pairs that aren't edges. Then the loss is

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

So the per-coordinate KL is −½(1 + log σ² − μ² − σ²). Before I trust this I want two checks on it, because a KL has to satisfy hard constraints — it's nonnegative and vanishes iff the two distributions coincide. First the special case: at μ=0, σ=1 the formula gives −½(1 + log 1 − 0 − 1) = −½(1 + 0 − 0 − 1) = 0, exactly, as it must when q equals the prior. Second, a numeric spot-check against the definition: integrating q·log(q/p) directly for, say, (μ,σ) = (2, 0.5) gives 2.318, and the formula gives −½(1 + log 0.25 − 4 − 0.25) = −½(1 − 1.386 − 4 − 0.25) = −½(−4.636) = 2.318 — agreement to three decimals, and likewise for (−1, 2) → 1.307 and (0.5, 1.5) → 0.345, all nonnegative. The formula holds. The total KL sums this over all latent dimensions and all nodes. The reconstruction term wants to *maximize* the ELBO, so in a loss I minimize, the contribution is −ELBO = (balanced cross-entropy reconstruction) + KL, i.e. I add ½ Σ(−1 − log σ² + μ² + σ²). Writing log σ as the network's output, log σ² = 2 log σ, the node-averaged positive KL is −½ mean over nodes of Σ_dim(1 + 2 log σ − μ² − exp(2 log σ)); if the reconstruction loss is averaged over all N² entries, I scale this node-averaged KL by 1/N so it lives on the same per-entry scale. So the KL is computable in closed form, no sampling needed — it's the standard Gaussian-to-Gaussian formula, applied per node.

Now I should stress-test this combination rather than declare victory, because I suspect there's a tension between the two pieces I just glued. Go back to the reconstruction gradient I computed: on a true edge the gradient w.r.t. s = z_i^T z_j is −(1 − σ(s)), so descent keeps pushing s upward, and it never stops wanting more — even at s=2 the pull is still −0.12, and to get σ(s) close to 1 the decoder needs s to keep growing. The only way to make z_i^T z_j large and positive for many edges is to give the embeddings large norm and align their directions; the decoder is, in effect, paying the embeddings to grow outward from the origin. But the KL term I just derived is −½(1 + log σ² − μ² − σ²), and its μ² penalty grows quadratically with the mean's distance from zero — it pulls every embedding *toward* the origin and σ toward 1. So the two objectives pull in opposite directions on the norm: the decoder rewards ‖z‖ large, the prior penalizes it. That's a real conflict, not a cosmetic one — a Gaussian prior centered at zero is arguably a poor match for a decoder that profits from pushing embeddings outward. I can accept that as a modeling tradeoff rather than a bug: the prior gives me regularization and a smoother latent space, while the decoder asks for confident similarity scores. This is the kind of thing the deterministic auto-encoder sidesteps entirely (no prior, no tension) at the cost of an unregularized latent space. So I have a family: the deterministic graph auto-encoder, Z = GCN(X, A), Â = σ(Z Z^T), trained on balanced cross-entropy — clean, parameter-light decoder, no latent regularization; and its variational sibling, which adds the Gaussian posterior, the reparameterized sampling, and the KL regularizer for a smoother latent space at the cost of the prior-decoder tension.

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
        nn.init.xavier_uniform_(self.W.weight)              # Glorot, as in the GCN/GAE code

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
