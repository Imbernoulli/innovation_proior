# Context: learning to *decode* graphs from a continuous code (circa 2017-2018)

## Research question

Deep generative modeling had a banner few years for images and text: latent-variable models
and adversarial models could draw a vector from a prior and turn it into a photorealistic
image or a fluent sentence, and the latent space was smooth enough to interpolate and to
search over. None of that machinery had crossed over to graphs. Almost all deep learning on
graphs to that point pointed the *other* way -- it learned to **encode** a graph into a vector
(graph embeddings for classification, regression, retrieval). The inverse map, **decoding** a
graph out of a vector, was essentially open. So the question is concrete: build a model that
maps a point `z` in a continuous code space to a graph `G`, trainable end to end by gradient
descent, such that drawing `z` from a prior produces graphs that look like a training
distribution -- and ideally with a code space smooth enough to interpolate between graphs and
to condition on desired properties. The payoff would be large: *de novo* molecule design
(search a continuous embedding for compounds with target properties), scene-graph synthesis,
network modeling. The crux is what makes this *harder* than images: a graph is a discrete
object with no fixed layout, and gradient-based learning wants something continuous and
canonically ordered.

## Background

**The two obstacles that make graphs resist gradient-based generation.** First, graphs are
**discrete**: an adjacency matrix has 0/1 entries, node and edge labels are categorical.
Discrete decisions are not differentiable, so a decoder that *samples* a graph during
training blocks backpropagation. Image pixels are continuous (or treated as such), so this
never bites for images; for graphs it is the first wall. Second, and deeper, a graph has **no
canonical ordering of its nodes**. Text generation works because language is already a
sequence -- there is one left-to-right order, and you predict token by token. A graph of `n`
nodes can be written down as an adjacency matrix in up to `n!` different ways, all
representing the *same* graph. There is no agreed best way to linearize an arbitrary graph
into a sequence of construction steps; canonical-labeling algorithms exist (nauty, McKay &
Piperno 2014), but it had been observed empirically (Vinyals, Bengio & Kudlur 2015, "Order
matters") that *which* order you impose changes what a model learns when it operates on sets.
The practical bite of this is in the loss: the matrix representation of a graph is **not
invariant to permutation of its nodes**, so comparing a predicted adjacency matrix to a
ground-truth one entrywise is meaningless -- the model could predict exactly the right graph
in a different node order and be punished for it.

**The continuous-latent generative framework that was mature.** Variational autoencoders
(Kingma & Welling 2013) gave a clean, differentiable recipe for a directed latent-variable
generative model. Train by maximizing the evidence lower bound

```
L(phi, theta; x) = E_{q_phi(z|x)}[ -log p_theta(x|z) ] + KL[ q_phi(z|x) || p(z) ],
```

(written here as a loss to minimize) with an isotropic Gaussian prior `p(z)=N(0,I)`, a
Gaussian recognition model `q_phi(z|x)` whose two heads emit `(mu, log sigma^2)`, and the
reparameterization `z = mu + sigma (.) eps`, `eps ~ N(0,I)`, so gradients flow through the
stochastic layer. For Gaussian prior and posterior the KL has the closed form
`KL = -1/2 sum_j (1 + log sigma_j^2 - mu_j^2 - sigma_j^2)`. The reconstruction term
`E_q[-log p_theta(x|z)]` is the only piece that has to be sampled, and its form is dictated by
the data type of `x` (per-pixel Bernoulli cross-entropy for binary data, Gaussian for
continuous). This framework was the standard substrate for "code space -> object" models. The
conditional variant (Sohn, Lee & Yan 2015) showed how to steer generation: condition both
encoder and decoder on a side label `y`, giving `log p(y|x) >= -KL(q(z|x,y)||p(z|x)) +
E_q[log p(y|x,z)]`, so the latent carries only what the label does not.

**The graph-encoding tools that were mature.** Message-passing / graph convolution had become
the standard way to turn a graph into vectors. Edge-conditioned convolution (Simonovsky &
Komodakis 2017) handles *labeled* edges: a filter-generating network maps an edge label to a
weight matrix, so a node aggregates neighbors through edge-type-specific filters,
`X^l(i) = (1/|N(i)|) sum_{j in N(i)} F^l(L(j,i); w^l) X^{l-1}(j) + b^l`. To collapse the
per-node states into a single graph-level vector in a permutation-invariant way, the gated
graph-level readout of Li, Tarlow, Brockschmidt & Zemel (2015) sums node states through a
soft-attention gate, `h_G = tanh( sum_v sigma(i(h_v,x_v)) (.) tanh(j(h_v,x_v)) )`, where
`sigma(i(.))` decides which nodes are relevant. So "graph in -> vector out" was a solved
problem; "vector out -> graph" was not.

**The matching tool that existed in vision.** Aligning two structured objects with no shared
coordinate frame is exactly (second-order) **graph matching**: find a correspondence
`X in {0,1}^{k x n}` between the nodes of two graphs that maximizes a pairwise similarity
`sum S((i,j),(a,b)) X_{a,i} X_{b,j}`. This is an NP-hard integer quadratic program, routinely
relaxed to continuous `X* in [0,1]^{k x n}`. Max-pooling matching (Cho, Sun, Duchenne & Ponce
2014) solves the relaxation by a power iteration `x^{(t+1)} = S x^{(t)} / ||S x^{(t)}||_2`,
where the matrix-vector product is made robust to clutter by **max-pooling** over candidate
neighbors instead of summing: `x_{ia} <- x_{ia} S_{ia;ia} + sum_{j in N_i} max_{b in N_a}
x_{jb} S_{ia;jb}`. A continuous `X*` can be rounded to a strict one-to-one assignment by the
Hungarian algorithm. This was an off-the-shelf vision primitive, not something associated
with generative modeling.

**The molecule-generation prior art, all going through strings.** Because graphs were hard,
cheminformatics had resorted to *textual* surrogates. Molecules were written as SMILES
strings and fed to RNN text generators (Gomez-Bombarelli et al. 2016; Segler et al. 2017).
This inherits text-generation machinery but pays for it: SMILES syntax is brittle, so
character-level samplers emit many *invalid* strings, and a string is a lossy linearization of
the molecular graph. The diagnostic phenomenon that motivated everything downstream is exactly
this fragility -- a single misplaced character breaks a SMILES string, and a single wrong
edge/atom breaks chemical validity, whereas flipping one pixel in an MNIST digit is harmless.

## Baselines

These are the prior generative approaches a graph decoder would be measured against.

**SMILES character VAE (Gomez-Bombarelli, Duvenaud, Hernandez-Lobato et al. 2016,
arXiv:1610.02415).** A standard VAE whose encoder/decoder are RNNs over SMILES *characters*.
It buys a continuous, searchable latent space over molecules, which is the right ambition.
**Limitation:** the decoder generates a character string with no notion of syntax, so a large
fraction of sampled strings are not even valid SMILES, let alone valid molecules; and because
the representation is a string, the model never sees the graph structure it is really trying
to produce -- the atoms-and-bonds object is hidden behind a fragile serialization.

**Grammar VAE (Kusner, Paige & Hernandez-Lobato 2017).** Sharpen the string approach by
encoding/decoding the *parse tree* of a SMILES context-free grammar: the decoder emits a
sequence of grammar production rules, so every output is syntactically valid by construction.
This guarantees syntax at the grammar level. **Limitation:** syntactic validity is not chemical
(semantic) validity -- valence and bonding constraints are not enforced by the grammar -- and
the grammar still narrows generation through a string parse rather than the molecular graph
itself, so it cannot directly predict per-atom and per-bond attributes of the graph.

**Autoregressive graph construction (the node-by-node line).** The alternative to going
through strings is to build the graph *incrementally* -- add a node, then its edges, then the
next node -- predicting each step with a recurrent model. This keeps the object a graph
throughout. **Limitation:** it has to commit to an *order* of construction, and a graph has no
canonical order; the model's likelihood and samples depend on the imposed ordering, and
learning a good order involves discrete, non-differentiable decisions. So the node-ordering
problem is not solved, only relocated into the generation procedure.

**Classical (pre-deep) graph models.** Erdos-Renyi random graphs, Barabasi-Albert preferential
attachment, stochastic blockmodels (Snijders & Nowicki 1997), state-transition-matrix learning
(Gong & Xiang 2003) generate graphs from a handful of structural parameters. **Limitation:**
they model coarse statistics (degree, community structure) by hand-chosen mechanisms; they do
not learn a rich distribution from data and have no continuous, conditionable code space.

## Evaluation settings

The natural yardsticks for an unconditional graph generator, all pre-existing:

- **Datasets.** Small-graph collections that fit a one-shot dense model:
  `community_small` (synthetic two-community graphs, ~12-20 nodes), `ego_small` (ego networks
  extracted from a citation graph, ~4-18 nodes), and `enzymes` (protein tertiary-structure
  graphs from BRENDA, ~10-125 nodes). For molecules specifically, the public organic-chemistry
  sets QM9 (~134k molecules, up to 9 heavy atoms; 4 atom types, 4 bond types) and ZINC
  (~250k drug-like molecules, up to 38 heavy atoms; 9 atom types, 4 bond types).
- **Structural-distribution metrics (lower is better).** Since judging "graph realism" by eye
  is unreliable, the field compares *distributions of graph statistics* between generated and
  reference graphs via Maximum Mean Discrepancy (MMD, a kernel two-sample distance): MMD of the
  degree distribution, MMD of the clustering-coefficient distribution, MMD of the 4-node orbit
  count distribution, and their average.
- **Chemistry-specific metrics (for molecules).** Decode many samples to discrete graphs and
  check, with a cheminformatics toolkit, the fraction that are chemically *valid*, *unique*
  among themselves, and *novel* relative to the training set; ELBO / reconstruction
  log-likelihood as the VAE bookkeeping number.
- **Protocol.** Hold out validation/test splits for model selection; train all methods under a
  shared, fixed schedule (e.g. a fixed number of epochs, batch size 32, single GPU) over
  multiple seeds for reliability; for sampling, draw `z` from the prior and take the discrete
  point estimate of the decoded graph.

## Code framework

A graph generator plugs into a standard dense-graph training harness. The data pipeline yields
batches of padded node-feature tensors `[B, k, d]`, adjacency matrices `[B, k, k]` (binary,
symmetric, zero diagonal, padded to a maximum size `k`), and per-graph node counts `[B]`; an
Adam optimizer and the SGD training loop already exist. The substrate is generic: a model object
receives one dense graph batch and returns a scalar training loss. The single empty slot is the
model and objective.

```python
import torch
import torch.nn as nn


class DenseGraphModel(nn.Module):
    """Generic dense-graph model. It consumes padded node features [B, k, d],
    adjacency matrices [B, k, k], and node counts [B], then returns a scalar loss."""

    def __init__(self, input_dim, hidden_dim, latent_dim, max_nodes, **kwargs):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.max_nodes = max_nodes
        # TODO: define the model components.

    def forward(self, node_features, adj, node_counts):
        # node_features: [B, k, d]; adj: [B, k, k]; node_counts: [B]
        # TODO: compute the training objective.
        pass
```

The training loop calls `forward`, backpropagates the returned loss, and steps the optimizer.
Everything specific to the graph model lives inside the empty slot.
