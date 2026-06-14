## Research question

The goal is to learn a distribution over graphs and sample new ones that are statistically
indistinguishable from a training set — graphs with categorical node attributes (e.g. atom
types) and categorical edge attributes (bond types, with "no edge" as one of the categories).
The applications that drive the problem — molecule design, traffic and social-network modeling,
program/code structure — all want a *one-shot* generator: produce a whole graph at once, rather
than emitting it node-by-node in some chosen order. Two structural facts make this hard and are
the crux of the whole problem.

First, **graphs are unordered**: the same graph has up to `n!` adjacency-matrix
representations, one per node permutation. A learner that treats the adjacency matrix as a fixed
array will waste capacity memorizing arbitrary orderings, and its likelihood — formally the sum
of the likelihoods of all `n!` permutations — is intractable. So a good model must be
*permutation equivariant* (its prediction permutes with its input) and must induce a
permutation-*invariant* training signal, without resorting to data augmentation over random
permutations or to an expensive graph-matching step at training time.

Second, **graphs are sparse and discrete**: a real graph has `O(n)` edges out of `O(n²)`
possible, and node/edge attributes are categories, not real numbers. Whatever generative
mechanism we choose has to respect that the objects are categorical and that the *structure*
(which edges exist, how the cycles and connected components are arranged) is the thing we
actually care about reproducing.

The precise target: a generative model that (1) is permutation equivariant with a permutation
invariant loss; (2) trains within a fixed compute budget and samples valid undirected graphs;
(3) does not collapse to or require an arbitrary node ordering; (4) reproduces the structural
statistics (degree, clustering, orbit/substructure counts) of the data. Each existing family
below achieves some of this; none does all of it cleanly at scale.

## Background

By this time, denoising diffusion models have become the dominant generative paradigm for
images, video and audio, outperforming GANs and autoregressive models on perceptual quality.
The recipe (Sohl-Dickstein et al. 2015; Ho et al. 2020, "DDPM"; Song et al. 2021, score SDEs)
is a fixed *forward* process that gradually corrupts a datapoint `x` into noise through a Markov
chain `q(z^1,…,z^T | x) = q(z^1|x) ∏_t q(z^t|z^{t-1})`, and a learned *reverse* (denoising)
network trained to invert one step at a time; sampling draws from a simple prior and applies
the network repeatedly. The load-bearing technical facts:

- **The denoiser is trained to predict the clean signal, not the previous latent.** Sohl-Dickstein
  et al. and Song et al. observed that `z^{t-1}` is a high-variance target that depends on the
  sampled trajectory, whereas when the posterior `q(z^{t-1}|z^t,x)` is tractable, the clean `x`
  can be used as the regression target and the noisy reverse step recovered analytically from
  it. This removes a large source of label noise. DDPM concretely predicts the clean `x_0` (or
  equivalently the noise `ε`) and reconstructs the reverse mean `μ̃_t(x_t, x_0)` in closed form.
- **Three properties make a diffusion model efficient.** (1) `q(z^t|x)` has a closed form, so
  every timestep can be trained in parallel without unrolling the chain. (2) The posterior
  `q(z^{t-1}|z^t,x)` has a closed form, so `x` can be the network's target. (3) The limit
  `q_∞ = lim_{T→∞} q(z^T|x)` is independent of `x`, so it can serve as the sampling prior. For
  Gaussian noise all three hold: `q(x_t|x_0)=N(√ᾱ_t x_0, (1-ᾱ_t)I)` with
  `ᾱ_t = ∏_{s≤t}(1-β_s)`, and `q(x_{t-1}|x_t,x_0)` is again Gaussian (DDPM Eq. 6-7).
- **The objective is a variational (evidence) lower bound** that decomposes into a prior term
  `KL[q(z^T|x) ‖ q_∞]`, per-step diffusion terms `KL[q(z^{t-1}|z^t,x) ‖ p_θ(z^{t-1}|z^t)]`, and
  a reconstruction term — usable both as a training loss and as a likelihood estimate.
- **Noise schedule.** The cosine schedule (Nichol & Dhariwal 2021) sets the cumulative noise via
  `ᾱ_t = cos²(½π (t/T + s)/(1+s))` (normalized so `ᾱ_0 = 1`) with a small offset `s ≈ 0.008`,
  which corrupts more gently near `t=0` and `t=T` than a linear schedule and improves sample
  quality.

Diffusion is not inherently continuous. A parallel line studies diffusion over **discrete state
spaces** for text, images and audio (Hoogeboom et al. 2021; Austin et al. 2021, "D3PM"; Johnson
et al. 2021; Yang et al. 2022). For a categorical variable with `K` classes, one-hot encoded as
a row vector `x`, the forward step is a *transition matrix*: `[Q^t]_{ij} = q(z^t=j | z^{t-1}=i)`
so that `q(z^t|z^{t-1}) = z^{t-1} Q^t` (a row-vector–matrix product). The chain's `t`-step
marginal is `q(z^t|x) = x Q̄^t` with `Q̄^t = Q^1 Q^2 … Q^t`, and the posterior follows from Bayes'
rule. The simplest, most-studied transition is **uniform**:
`Q^t = α^t I + (1-α^t) 11'/K`, doubly stochastic with strictly positive entries, whose stationary
distribution is uniform over the `K` classes (so it satisfies property 3). D3PM also studies
absorbing-state (`[MASK]`) and structured transitions, and trains a hybrid loss that adds an
auxiliary cross-entropy term predicting `x_0` to the variational bound.

Two facts about the representational power of graph networks are also load-bearing, because the
denoiser will be a graph network. Message-passing networks (MPNNs) and graph transformers are
bounded in expressivity: they are at most as discriminative as the 1-Weisfeiler-Leman test (Xu
et al. 2018; Morris et al. 2019) and, concretely, **cannot count cycles or detect simple
substructures** on their own (Chen et al. 2020). The known remedies are to either use a
strictly more powerful (and far more expensive) architecture, or to **augment the input with
features the network cannot compute itself** — substructure counts (Bouritsas et al. 2022) or
spectral features of the graph Laplacian (Beaini et al. 2021; Chung 1997), which encode
connectivity and global structure. Whether such augmentation is even *possible* depends on the
generative mechanism, because the features must be computed on whatever intermediate object the
model manipulates.

## Baselines

These are the prior one-shot and diffusion-style graph generators a new method would be
measured against and would react to.

**Autoregressive generators (GraphRNN, You et al. 2018; GRAN, Liao et al. 2019).** Emit a graph
node-by-node (or block-by-block), an RNN or graph-attention network predicting each new node's
connections conditioned on the partial graph. They model rich structural dependencies and, with
domain knowledge, reach strong validity on molecules. **Limitation:** they impose an arbitrary
node *ordering* — the likelihood depends on the order, so they are not permutation invariant and
typically train over sampled orderings; generation is inherently sequential.

**VAE generators (GraphVAE, Simonovsky & Komodakis 2018; Set2GraphVAE, Vignac & Frossard 2021;
JT-VAE, Jin et al. 2018).** Encode a graph into a latent vector and decode a probabilistic
adjacency tensor of fixed maximum size in one shot. **Limitation:** matching the decoded graph
to the target requires solving a graph-matching/alignment problem (or an elaborate invariance
construction), which is costly and a recurring source of difficulty; output size is capped.

**Normalizing-flow generators (GraphNVP, Madhawa et al. 2019; MoFlow, Zang & Wang 2020;
categorical flows, Lippe & Gavves 2020; GraphDF, Luo et al. 2021).** Invertible maps from a
simple latent to graphs, exact likelihood. **Limitation:** the architecture is constrained to
stay invertible with tractable Jacobian, which restricts the dependencies between nodes and
edges the model can capture; performance trails autoregressive and motif-based methods.

**Continuous (Gaussian / score-based) graph diffusion (EDP-GNN, Niu et al. 2020; GDSS, Jo et al.
2022).** Port diffusion to graphs by embedding the graph in a *continuous* space: one-hot node
features and the adjacency matrix are treated as real tensors, additive Gaussian noise is
applied, and a network learns the score `∇ log p_t` of the noised joint distribution; a reverse
SDE/Langevin sampler generates. GDSS models nodes and edges jointly through a system of coupled
SDEs. They achieve competitive results and inherit diffusion's training stability.
**Limitation:** Gaussian noise turns the adjacency into a *dense* matrix of continuous values
partway through the forward process. The graph stops being sparse, and the discrete structural
notions the data is made of — whether an edge exists, the number of connected components, cycle
and orbit counts — are no longer defined on the noised object. The denoiser sees a blurry dense
tensor rather than a graph, and there is no sparse intermediate on which to read off graph
descriptors. GDSS's joint SDE noise model is also non-factorized, hence complex. On larger
graphs this continuous route degrades sharply.

The shape of the gap: the diffusion recipe is the strongest generative paradigm available and is
permutation-friendly, but every existing way of applying it to graphs first *continuizes* them
and pays for it in lost sparsity and lost structure; the order-free, one-shot, structure-aware
generator the problem asks for does not yet exist.

## Evaluation settings

The yardsticks already in use for one-shot graph generation, all permutation-blind and
structure-based:

- **Abstract/structural graphs.** Stochastic block model and planar-graph benchmarks (Martinkus
  et al. 2022): ~200 graphs each, up to ~64-200 nodes; community and ego-network datasets
  (Community-small, Ego-small from Citeseer); and protein graphs (ENZYMES / BRENDA, 10-125
  nodes). The metrics are *Maximum Mean Discrepancy* (MMD) between the distribution of a graph
  statistic on generated vs. reference graphs: degree distribution, clustering coefficient, and
  orbit (4-node substructure) counts, plus an average. Some protocols report the MMD as a ratio
  to the train-vs-test MMD. Lower MMD is better. Additional structural checks: fraction of
  generated graphs that are valid/planar/connected and unique and novel (V.U.N.).
- **Molecular graphs.** QM9 (small molecules, ≤9 heavy atoms), and the large MOSES and GuacaMol
  drug-molecule datasets (>1M molecules). Metrics: validity (RDKit sanitization), uniqueness,
  novelty, and, on the large sets, distributional scores (Fréchet ChemNet Distance, KL
  divergence over physicochemical descriptors, scaffold similarity).
- **Protocol.** Train for a fixed schedule (e.g. a fixed number of epochs, batch size 32, single
  GPU) shared across all methods; multiple seeds for reliability; adjacency matrices arrive
  binary, symmetric, zero-diagonal, padded to a maximum node count; the number of nodes per
  graph is itself drawn from the empirical training distribution. Likelihood (NLL / bits-per-
  element), where available, is reported for model comparison.

## Code framework

The generator plugs into a fixed harness: it receives padded dense graph tensors, masks invalid
nodes, takes training steps, and must be able to sample new node and edge categories. Nothing about
the generative mechanism is settled, so the substrate is only the generic machinery that already
exists: a module that owns its parameters and optimizer, a `train_step` that consumes categorical
node and edge tensors and returns a loss, and a `sample` method that returns valid masked graphs.
The empty slot is the model itself and the training and sampling rule it will use.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math


class GraphGenerator(nn.Module):
    """One-shot graph generator. Receives padded node categories
    [B, max_nodes, n_node_types], padded symmetric edge categories
    [B, max_nodes, max_nodes, n_edge_types], optional graph features, and
    a node mask; must train within a fixed budget and sample valid
    undirected categorical graphs.
    The generative mechanism is exactly what we will design."""

    def __init__(self, max_nodes, n_node_types, n_edge_types, **kwargs):
        super().__init__()
        self.max_nodes = max_nodes
        self.n_node_types = n_node_types
        self.n_edge_types = n_edge_types
        # TODO: the model we will design (its parameters), and its optimizer.
        #       Whatever turns clean graph tensors into a training objective
        #       and random draws into samples lives here.
        self.optimizer = None

    def train_step(self, X, E, y=None, node_mask=None):
        """One training step on a batch of categorical graph tensors.
           X: [B, max_nodes, n_node_types] node categories.
           E: [B, max_nodes, max_nodes, n_edge_types] edge categories,
              symmetric with a designated no-edge / diagonal class.
           y: optional graph-level features or labels.
           node_mask: [B, max_nodes], true for real nodes.
           Returns a dict with at least 'loss'."""
        # TODO: build a training target from the clean batch and the model,
        #       compute the loss, backpropagate, step the optimizer.
        raise NotImplementedError

    @torch.no_grad()
    def sample(self, n_samples, device):
        """Generate n_samples graphs.
           Returns (X, E, node_counts):
             X: [n_samples, max_nodes, n_node_types].
             E: [n_samples, max_nodes, max_nodes, n_edge_types],
                symmetric with padded entries masked to the no-edge class.
             node_counts: [n_samples], minimum 2."""
        # TODO: draw node counts from the data; produce categorical graphs
        #       with the generative mechanism we design; return valid masks.
        raise NotImplementedError
```

The harness supplies clean graphs to `train_step` and asks `sample` for new ones; everything
between — the internal representation, the network, the loss, and the sampling loop — is the
contribution to be designed.
