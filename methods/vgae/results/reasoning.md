Let me start from what actually goes wrong when I try to predict links on one of these citation graphs. I have a single undirected graph: a few thousand documents, citation edges between them, and for each document a sparse bag-of-words feature vector. Someone has hidden a tenth of the edges and I have to rank the hidden pairs so that the real ones come out on top. So I need two things — a way to turn each node into a vector, and a way to turn a pair of vectors into a score — and I have to learn both from the edges that remain, with no link labels beyond "these pairs are edges." That last part matters: this is unsupervised in the link sense. The only signal is the adjacency itself.

What can I reach for? There are two piles of tools and they don't talk to each other. One pile learns a vector per node from the structure: take random walks over the graph, treat each walk as a sentence and each node as a word, run skip-gram on that corpus the way word2vec does on text, and out come embeddings where nodes that co-occur in walks end up close. That's DeepWalk, and its descendants LINE and node2vec just sharpen the walk statistics — node2vec interpolating between breadth-first and depth-first exploration. These work, they're unsupervised, the embeddings really do encode community structure. But stare at what they consume: only the adjacency. The bag-of-words vector on each document — which on a citation graph is enormously informative, two papers about the same topic cite each other — never enters the model at all. And it's a multi-step pipeline: generate walks, then separately optimize a skip-gram loss, two objectives stapled together, nothing end-to-end. The other pile is spectral: embed nodes via the leading eigenvectors of the graph Laplacian L = I − D^{-1/2} A D^{-1/2}, project onto them, score a pair by the inner product of their embeddings, i.e. an entry of Z Zᵀ. Also unsupervised, also a clean global summary of the adjacency — and also structure-only, plus an O(N²) eigendecomposition I'd rather avoid. So both piles throw away the features, and the features are half my evidence.

The obvious question: is there anything that eats structure *and* features at once? There is, and it's recent. Start from the spectral-convolution picture — filtering a node signal x by g_θ ⋆ x = U g_θ(Λ) Uᵀ x in the Laplacian eigenbasis U. That's expensive and dense, but if I approximate the filter by a first-order Chebyshev polynomial in L and tie the coefficients down to a single parameter, the whole Fourier apparatus collapses to a cheap local operator: g_θ ⋆ x ≈ θ (I_N + D^{-1/2} A D^{-1/2}) x. The catch is that I_N + D^{-1/2} A D^{-1/2} has eigenvalues in [0, 2], so if I stack many layers the repeated multiplication blows up or vanishes the signal. The fix is to renormalize by folding the self-loop into the adjacency *before* normalizing: set Ã = A + I_N, let D̃ be its degree matrix, and use Ã_norm = D̃^{-1/2} Ã D̃^{-1/2}, whose spectrum is tamed for deep stacking. A layer is then H^{(l+1)} = σ( Ã_norm H^{(l)} W^{(l)} ) with H^{(0)} = X, so a two-layer network is GCN(X, A) = Ã_norm ReLU(Ã_norm X W₀) W₁. Read what one layer does: each node's new vector is a learned linear map of a normalized average over itself and its neighbors. After two layers a node has aggregated its two-hop neighborhood *and* carried the feature vectors along the way. It costs one sparse-dense matmul per layer, O(|E|), and it's fully differentiable. That's exactly the fusion I wanted — a representation z_i that depends on both A and X. The only problem is how it's been used: bolted to a softmax and a cross-entropy label loss for semi-supervised node classification. I have no node labels here, and even if I did, a representation tuned to a label taxonomy isn't what I want; I want a representation tuned to *explaining the edges*. So I have the right encoder and the wrong objective. I need to give this encoder an unsupervised job: take (X, A), produce node vectors, and be trained to reconstruct the adjacency I do see, so it can score the adjacency I don't.

So let me think about the simplest honest version first, before I complicate anything. Encoder: Z = GCN(X, A), one vector z_i per node. Decoder: I need p(there is an edge between i and j) as a function of z_i and z_j. What's the cheapest scoring rule that has the right geometry? An edge should be likely when the two nodes are "close" in the learned space. The dot product z_iᵀ z_j is exactly an unnormalized closeness — large and positive when the vectors point the same way and are big, negative when they oppose — and squashing it through a logistic sigmoid turns it into a probability: Â_ij = σ(z_iᵀ z_j). In matrix form Â = σ(Z Zᵀ). This is beautiful for several reasons I should make explicit rather than wave at. It has *no parameters of its own* — all the learning is pushed into the encoder, which is where the structure-plus-feature fusion lives. It's O(F) per scored pair, F the embedding width, so scoring is trivially cheap. It's symmetric in i and j, which is exactly right for an undirected graph: σ(z_iᵀ z_j) = σ(z_jᵀ z_i) automatically. And it's permutation-equivariant — relabel the nodes and the scores follow — because it depends only on the vectors, not on node identity. Geometrically it says: arrange the nodes in space so that linked nodes sit where their dot product is large; then a held-out pair scores high precisely when the encoder has placed them close. That is the whole inductive bias, and it's a good one for link prediction. Contrast with an alternative I could have picked — feed [z_i; z_j] into an MLP and let it learn a score. That adds parameters, breaks the clean symmetry unless I symmetrize by hand, and throws away the prior that "near = linked." The bare inner product is the right default; I'll keep it. So a non-probabilistic graph auto-encoder is already in hand: Z = GCN(X, A), Â = σ(Z Zᵀ), train by reconstructing A.

How do I train that reconstruction? Each entry A_ij is a Bernoulli — edge or not — so the natural loss is binary cross-entropy between A_ij and σ(z_iᵀ z_j), summed over pairs. But here I hit the lopsidedness I flagged at the start. The graph is sparse: |E| ≪ N², so the N² − |E| zero entries massively outnumber the ones. If I just average BCE over all pairs, the loss is dominated by the easy "no edge" majority, and the gradient that pushes linked pairs together is a rounding error against the gradient that pushes everything apart. The model's lazy optimum is "predict no edge." I have to rebalance. Two ways: up-weight the positive (A_ij = 1) terms so each rare edge counts as much as the crowd of zeros around it, with a matching rescale so the total stays calibrated; or subsample the zeros so each step sees a balanced mix of edges and a comparable number of non-edges. Either makes the rare positives carry real gradient. I'll use positive re-weighting in the full-batch reconstruction; in a minibatched harness the same effect comes from negative sampling — pair each real edge with a freshly sampled non-edge and apply BCE to both. The point is identical: don't let the N² zeros drown out the |E| ones.

Now, is the deterministic auto-encoder enough, or do I want more? It works, but the embedding is just a per-node point — a fancy lookup table. There's no organization to the space beyond what the reconstruction loss happens to impose, nothing stopping it from scattering points anywhere as long as the dot products come out right, and nothing that makes it a *model* of the graph I could reason about probabilistically. I'd like the latent space to be smooth and regularized — nearby points genuinely meaning similar nodes — and I'd like a principled objective rather than "minimize a reconstruction error and hope." That nudges me toward a latent-variable generative model: treat each z_i as a *random* latent variable with a prior, and ask the model to explain the adjacency as generated from the z's.

The machinery for fitting exactly that kind of model already exists, for i.i.d. data. You have a generative model p(x) = ∫ p(z) p(x|z) dz with a neural-net likelihood, the posterior p(z|x) is intractable, and you approximate it with a recognition network — an encoder — q(z|x). The trick that makes it trainable is the evidence lower bound. For any q,

  log p(x) = D_KL( q(z|x) ‖ p(z|x) ) + L,   L = E_{q(z|x)}[ log p(x|z) ] − D_KL( q(z|x) ‖ p(z) ),

and since the KL to the true posterior is ≥ 0, L lower-bounds log p(x); maximizing L over the encoder and decoder is the surrogate for maximizing the marginal likelihood. Let me make sure I see why that decomposition is the right shape for me. The first term, E_q[log p(x|z)], is an expected *reconstruction*: sample a latent from the encoder, ask the decoder how well it explains the data. The second, D_KL(q(z|x) ‖ p(z)), is a *regularizer*: it pulls the per-datapoint posterior toward a fixed prior. That's precisely the "reconstruct, but keep the latent space organized" objective I just said I wanted, and it falls out of the math rather than being bolted on. So if I cast each node as a datapoint, with latent z_i, prior p(z_i), encoder q(z_i | X, A), and a decoder that reconstructs the adjacency, the ELBO gives me a regularized auto-encoder for free.

But there's a mismatch I have to be honest about before I get excited. That framework is built for *i.i.d.* data — one independent x per datapoint, each with its own private posterior computed from its own x alone. My nodes are emphatically not i.i.d.: a node's latent should depend on its neighbors, that's the entire point of using a graph. And the thing I reconstruct isn't a per-node x; it's the *adjacency*, which couples nodes pairwise. So I can't just drop a VAE in. I have to bend it in two places. First, the encoder: instead of a per-node MLP q(z_i | x_i) that sees only node i's features, I make the encoder a *GCN*, so q(z_i | X, A) sees node i's whole neighborhood and feature context — the amortized posterior for each node is computed by message passing, not in isolation. The factorization q(Z | X, A) = ∏_i q(z_i | X, A) keeps the posterior a product over nodes (one Gaussian per node), but each factor's parameters are produced by a network that has looked at the graph. Second, the decoder: the "data" being reconstructed is A, and the likelihood is the inner-product Bernoulli I already designed, p(A_ij = 1 | z_i, z_j) = σ(z_iᵀ z_j), with p(A | Z) = ∏_i ∏_j p(A_ij | z_i, z_j). With those two substitutions the i.i.d. VAE becomes a generative model of a graph.

Now I need the encoder to output a *distribution* per node, not a point. The standard choice is a diagonal Gaussian: q(z_i | X, A) = N(z_i | μ_i, diag(σ_i²)). So the encoder must emit a mean vector μ_i and a (per-coordinate) variance for each node. How do I produce both from the GCN? Two output heads. Run the shared first GCN layer to get a hidden representation of every node, then split into two second-layer GCNs: one produces μ = GCN_μ(X, A), the other produces the log-scale of the variance, GCN_σ(X, A). Why share the first layer and split only at the top? Because both the mean and the spread of a node's posterior are functions of the same neighborhood evidence — it's wasteful and over-parameterized to learn two entirely separate networks, and tying the first layer forces μ and σ to be read off a common feature of the node's context. And why output the *log* of the scale rather than the scale itself? Because a network output is an unconstrained real number, and a variance must be positive. If I made the head emit σ directly I'd have to clamp or square it; if I emit log σ instead, then σ = exp(log σ) is automatically positive for any real output, and the optimizer never has to fight a positivity constraint. So the heads are μ and log σ.

With a distributional encoder I have to actually sample z_i ~ N(μ_i, diag(σ_i²)) to evaluate the reconstruction term E_q[log p(A|Z)], and then backpropagate through the sampling into μ and σ. Sampling is not differentiable as written. The naive way to get a gradient — the score-function estimator, ∇_φ E_q[f] = E_q[f ∇_φ log q] — exists but has notoriously high variance; for a network it's too noisy to train with. The reparameterization trick removes the variance entirely. The idea is to push the randomness out of the parameters and into a fixed noise source: instead of drawing z from q(z|x) directly, write z as a deterministic differentiable function of parameter-free noise. For a diagonal Gaussian the reparameterization is exactly

  z_i = μ_i + σ_i ⊙ ε,   ε ~ N(0, I),

where ⊙ is elementwise. Check it: if ε ~ N(0, I) then μ_i + σ_i ⊙ ε is Gaussian with mean μ_i and covariance diag(σ_i²) — the right distribution. But now the only randomness is in ε, which carries no parameters, so the expectation E_q[f(z)] = E_{ε~N(0,I)}[f(μ + σ⊙ε)] can be estimated by drawing ε and the gradient flows straight through μ and σ as ordinary deterministic quantities. A single sample (L = 1) per node suffices in practice because the rest of the estimator (especially the KL, see below) is exact. And at *test* time, when I'm scoring links rather than training, I don't want this jitter — I want the encoder's best point estimate — so I drop the noise and use z_i = μ_i directly. (One practical guard: log σ is an unconstrained output, and a runaway positive value makes σ = exp(log σ) and hence the sampled z explode; clamping log σ at a modest ceiling keeps the variance from blowing up in early training without affecting the healthy regime.)

That handles the reconstruction term. Now the regularizer, D_KL( q(z|X,A) ‖ p(z) ). First the prior. Take the simplest isotropic choice, p(z_i) = N(0, I) per node, p(Z) = ∏_i N(z_i | 0, I). With both q and p Gaussian, the KL doesn't need sampling at all — it's a closed form — and I should derive it rather than quote it, because the exact expression is what goes into the code and a sign error here would quietly wreck training. Work per latent dimension j (the diagonal Gaussian factorizes, so the KL is a sum over j of one-dimensional Gaussian KLs). I need −D_KL(q ‖ p) = ∫ q(z) (log p(z) − log q(z)) dz = ∫ q log p − ∫ q log q, with q = N(μ, σ²) and p = N(0, 1) in one dimension. Take the two pieces.

The cross term: ∫ q(z) log p(z) dz with log p(z) = −½ log(2π) − ½ z². So ∫ q log p = −½ log(2π) − ½ E_q[z²]. For z ~ N(μ, σ²), E_q[z²] = Var + mean² = σ² + μ². Hence ∫ q log p = −½ log(2π) − ½ (μ² + σ²). Summing over the J dimensions, ∫ q log p(z) dz = −J/2 log(2π) − ½ Σ_j (μ_j² + σ_j²).

The entropy-like term: ∫ q(z) log q(z) dz, which is the negative differential entropy of N(μ, σ²). For a Gaussian, the entropy is ½ log(2π e σ²) = ½ log(2π) + ½ + ½ log σ². So ∫ q log q = −[½ log(2π) + ½ + ½ log σ²] = −½ log(2π) − ½(1 + log σ²). Let me double-check that by direct integration so I trust it: log q(z) = −½ log(2π) − ½ log σ² − (z−μ)²/(2σ²), and E_q[(z−μ)²] = σ², so E_q[log q] = −½ log(2π) − ½ log σ² − σ²/(2σ²) = −½ log(2π) − ½ log σ² − ½ = −½ log(2π) − ½(1 + log σ²). Good, matches. Summing over J dimensions, ∫ q log q dz = −J/2 log(2π) − ½ Σ_j (1 + log σ_j²).

Subtract: −D_KL = ∫ q log p − ∫ q log q = [ −J/2 log(2π) − ½ Σ_j (μ_j² + σ_j²) ] − [ −J/2 log(2π) − ½ Σ_j (1 + log σ_j²) ]. The −J/2 log(2π) terms cancel exactly, and I'm left with

  −D_KL( q(z|X,A) ‖ p(z) ) = ½ Σ_j ( 1 + log σ_j² − μ_j² − σ_j² ),

per node, then summed (or averaged) over nodes. So the regularizer, written to be *added* to the ELBO (we maximize L), is +½ Σ_j (1 + log(σ_j²) − μ_j² − σ_j²). Sanity: it's maximized (KL = 0) when μ_j = 0 and σ_j² = 1, i.e. when the posterior equals the N(0,1) prior, exactly as a regularizer toward the prior should be. And it's expressed in the encoder's outputs: my head emits log σ, so log σ² = 2 log σ and σ² = exp(2 log σ), giving the equivalent implementation form

  −D_KL = ½ Σ_j ( 1 + 2 log σ_j − μ_j² − exp(2 log σ_j) ),

which is what I'll write in code (no separate squaring of a variance, just the head's own log σ).

So the full training objective per node-pair structure is the ELBO

  L = E_{q(Z|X,A)}[ log p(A | Z) ] − D_KL( q(Z|X,A) ‖ p(Z) ),

the first term estimated by reparameterized sampling (the inner-product Bernoulli reconstruction, with the positive-edge re-weighting so the sparse ones aren't drowned) and the second term the closed form I just derived. I maximize L, equivalently minimize (reconstruction BCE) − (the −D_KL bonus) = recon + D_KL.

There's a balance question I shouldn't gloss over: how heavily does the KL weigh against the reconstruction? The reconstruction term, as I'll actually compute it, is a *mean* binary cross-entropy over the scored pairs — a per-edge quantity. The KL, derived above, is naturally a *sum over latent dimensions* for each node. If I add a per-node sum to a per-edge mean, the scales are mismatched and the regularizer can swamp or be swamped by reconstruction depending on N. The clean fix is to put the KL on the same per-something footing: average the per-node KL over the N nodes, i.e. weight the total KL by 1/N. Then reconstruction is "average badness per scored pair" and the regularizer is "average KL per node," two comparable per-unit quantities, and the trade-off no longer drifts with graph size. So the loss is recon_BCE + (1/N) · KL.

Now a subtlety the prior choice creates, which I want to face rather than ignore. I picked p(z) = N(0, I), which pulls every node's posterior toward the origin. But look at what the inner-product decoder *wants*. To make σ(z_iᵀ z_j) ≈ 1 for a real edge, it needs z_iᵀ z_j to be large and positive, which it achieves by giving the linked nodes embeddings with large norm pointing the same way — it pushes embeddings *away* from the zero-center. The reconstruction force pushes outward; the Gaussian prior pulls inward. So there's a genuine tension between the decoder and the prior: a centered isotropic Gaussian is not perfectly matched to a decoder that rewards large-norm, well-separated embeddings. This isn't fatal — the 1/N weighting keeps the prior gentle enough that reconstruction can still spread the embeddings out where it needs to, and the variational regularization still buys a smoother, less overfit latent space than the bare deterministic auto-encoder — but it's the honest reason the centered Gaussian prior is a default rather than an ideal, and a place a future model could improve.

Let me also notice that the deterministic auto-encoder I built first isn't a separate method — it's the degenerate limit of this one. Take the variational model and remove the variance head: no log σ, so no sampling, z_i = μ_i = GCN(X, A) deterministically, and with no distribution there's no KL term. What's left is exactly Z = GCN(X, A), Â = σ(Z Zᵀ), trained on reconstruction alone — the graph auto-encoder. So I have a clean two-member family: drop the stochastic part and I recover the simple deterministic baseline; keep it and I get the regularized probabilistic model. That's the right kind of generalization — the simpler thing falls out, it isn't a different animal. And the featureless case is another simple specialization: if a graph has no node attributes, set X = I, the identity, so each node's "feature" is its own one-hot identity and the model learns purely from structure — recovering the structure-only regime of the older baselines, but now inside the same end-to-end differentiable encoder.

One more design point worth stating: training is full-batch. These citation graphs have a few thousand to a few tens of thousands of nodes, the sparse adjacency fits in memory, and the GCN's cost is O(|E|) per layer, so I can compute the whole forward pass on the entire graph each step and run plain gradient descent with Adam. No minibatching of nodes is needed at this scale (and minibatching a graph is delicate anyway, because a node's GCN receptive field pulls in its multi-hop neighborhood). Standard fan-in/fan-out weight initialization, a modest hidden width and a small latent dimension, a learning rate around 0.01, a couple hundred iterations — carried straight over from how the GCN encoder is trained for classification.

Let me write it as the predictor the harness expects, filling the empty encode/decode slots, grounded in how the canonical implementation is actually structured. The encoder is shared GCN layers feeding two top heads (μ and log σ); encode reparameterizes at train time and returns μ at eval; decode is the inner product; and the KL has to reach the loss even though the external loop only applies BCE to the returned scores, so I expose it for the loop to add (or, equivalently, fold the per-node KL into the model's output the way the scaffold's training loop consumes it).

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

MAX_LOGSTD = 10  # clamp on log sigma so exp(log sigma) can't blow up early


class LinkPredictor(nn.Module):
    """Variational Graph Auto-Encoder for link prediction.

    GCN encoder -> per-node Gaussian posterior q(z_i | X, A) = N(mu_i, diag(sigma_i^2)),
    with two GCN heads producing mu and log sigma that share their first layer.
    Reparameterized sampling at train time; mu at eval time. Inner-product decoder
    sigma(z_i^T z_j). ELBO = reconstruction (BCE, applied by the loop) - KL(q || p),
    p = N(0, I); the KL is computed here and exposed for the objective.
    """

    def __init__(self, in_channels, hidden_channels, num_layers=2, dropout=0.0):
        super().__init__()
        self.dropout = dropout

        # shared trunk: all but the last layer (one shared GCN layer for the 2-layer case)
        self.shared_convs = nn.ModuleList()
        if num_layers > 1:
            self.shared_convs.append(GCNConv(in_channels, hidden_channels))
            for _ in range(num_layers - 2):
                self.shared_convs.append(GCNConv(hidden_channels, hidden_channels))
            last_in = hidden_channels
        else:
            last_in = in_channels

        # two heads off the shared trunk: mean and log-std (no nonlinearity)
        self.conv_mu = GCNConv(last_in, hidden_channels)       # mu  = GCN_mu(X, A)
        self.conv_logstd = GCNConv(last_in, hidden_channels)   # log sigma = GCN_sigma(X, A)

        self._mu = None
        self._logstd = None

    def encode(self, x, edge_index):
        # shared GCN layers: each mixes a node with its neighbors (renormalized adjacency)
        for conv in self.shared_convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        self._mu = self.conv_mu(x, edge_index)
        self._logstd = self.conv_logstd(x, edge_index).clamp(max=MAX_LOGSTD)

        if self.training:
            # reparameterization: z = mu + sigma * eps,  eps ~ N(0, I)
            std = torch.exp(self._logstd)
            eps = torch.randn_like(std)
            return self._mu + eps * std
        # at test time use the posterior mean (no sampling jitter)
        return self._mu

    def decode(self, z_src, z_dst):
        # inner-product decoder: logit of p(edge) = z_i^T z_j  (sigmoid applied by BCE-with-logits)
        return (z_src * z_dst).sum(dim=-1)

    def kl_loss(self):
        # -D_KL(q||p) = 1/2 sum_j (1 + 2 log sigma - mu^2 - exp(2 log sigma)); return +D_KL
        mu, logstd = self._mu, self._logstd
        return -0.5 * torch.mean(
            torch.sum(1 + 2 * logstd - mu.pow(2) - torch.exp(2 * logstd), dim=-1)
        )

    def forward(self, x, edge_index, edge_label_index):
        z = self.encode(x, edge_index)
        z_src = z[edge_label_index[0]]
        z_dst = z[edge_label_index[1]]
        return self.decode(z_src, z_dst)
```

And the objective the training loop forms around it — reconstruction by binary cross-entropy on the candidate edges (positives plus sampled negatives, which is the negative-sampling form of the positive re-weighting), plus the per-node-averaged KL:

```python
def train_step(model, data, optimizer):
    model.train()
    optimizer.zero_grad()
    logits = model(data.x, data.edge_index, data.edge_label_index)
    recon = F.binary_cross_entropy_with_logits(logits, data.edge_label)  # -E_q[log p(A|Z)]
    loss = recon + (1.0 / data.num_nodes) * model.kl_loss()              # + (1/N) D_KL(q||p)
    loss.backward()
    optimizer.step()
    return float(loss)
```

Let me trace the causal chain back. The pain was that link prediction on one attributed graph needs both topology and node features, but the unsupervised methods on hand — random-walk skip-gram (DeepWalk, node2vec) and spectral embedding — use structure only and the one feature-aware encoder (the GCN) was wired to a supervised label loss I don't have. So I gave the GCN an unsupervised job: encode each node and reconstruct the adjacency. The cheapest scoring rule with the right "near = linked" geometry is the parameter-free inner-product decoder σ(z_iᵀ z_j), symmetric and permutation-equivariant; trained by BCE with the sparse positives re-weighted (or balanced by negative sampling) so they aren't drowned by the N² zeros. That bare deterministic auto-encoder works but gives an unregularized point embedding, so I promoted z_i to a latent variable and fit it with the evidence lower bound, whose reconstruction-minus-KL split is exactly the "explain the edges, keep the latent space organized" objective I wanted. Bending the i.i.d. variational framework to a graph meant two substitutions: the encoder becomes a GCN with two heads (μ and log σ, sharing their trunk, outputting log σ so the variance stays positive), and the likelihood becomes the inner-product Bernoulli over A. The reparameterization z = μ + σ⊙ε made the sampled reconstruction differentiable with low variance; the Gaussian-Gaussian KL came out in closed form, ½ Σ_j(1 + log σ_j² − μ_j² − σ_j²), which I weight by 1/N to match the per-edge reconstruction scale. The centered Gaussian prior is a gentle default in mild tension with a decoder that wants large-norm embeddings, and dropping the stochastic part recovers the deterministic graph auto-encoder as a special case, with X = I covering the featureless regime. The whole thing trains full-batch with Adam on graphs that fit in memory.
