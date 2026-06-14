## Research question

We want to learn a distribution `p(G)` over simple graphs from a dataset of observed graphs,
and then sample new graphs that are statistically indistinguishable from the training set. The
graphs are undirected, unattributed, of *varying* size, and — this is what makes the problem
hard — they carry no canonical node identities: relabelling the nodes gives the same graph.
Concretely, a graph on `N` nodes has up to `N!` adjacency matrices, all representing the same
object, so the model's likelihood of a graph is really a sum over orderings,
`p(G) = sum_pi p(A^pi)`, with `N!` terms.

Three demands collide and no existing method meets all three at once. (1) **Expressiveness:**
real graphs have richly correlated edges — whether two nodes connect depends strongly on the
rest of the structure (shared neighbors, communities, regular lattices), so a model that
generates edges independently produces unrealistic samples. (2) **Permutation handling:** the
training objective and any reconstruction loss must cope with the `N!`-fold ambiguity without
enumerating it. (3) **Scale and speed:** the model must train and sample in feasible time on
graphs well beyond the toy regime — hundreds to thousands of nodes — where the prior deep
models stall, either because they run a heavy network per individual edge decision or because
the number of sequential decisions grows quadratically in `N`. The problem is to build a single
generative model that is expressive about edge correlations, principled about permutations, and
fast enough to scale, simultaneously.

## Background

**The field state.** Generative models of graphs split into two broad families by the
*dependency structure* of their generation decisions.

The first family generates the components of the graph **independently or with weak
dependency**, given a latent code. The variational-autoencoder approaches
(Kipf & Welling 2016; Simonovsky & Komodakis 2018) encode a graph into a continuous latent `z`
and decode a probabilistic adjacency matrix whose entries are independent Bernoulli variables
given `z`. These models are parallelizable and conceptually clean, but the conditional
independence of edges given the latent is a strong assumption that is observed to compromise
sample quality, and they confront the permutation problem head-on at the loss: comparing a
decoded probabilistic graph to a ground-truth graph requires aligning their nodes, i.e. graph
matching.

The second family makes **auto-regressive** decisions: a graph is built as a sequence of
local choices (add a node, add edges to it), and each choice is conditioned on everything
generated so far. By construction this captures complex edge dependencies — each new edge can
depend on the whole partial graph. The cost is sequentiality: many steps, performed in order,
and the difficulty of carrying information across a long generation sequence.

**Load-bearing concepts.**

*Graph neural networks / message passing.* A GNN computes node representations by iterated
local message passing: each node aggregates messages from its neighbors and updates its state,
repeated for several rounds. Crucially the computation is *permutation-equivariant* in the node
representations — relabel the nodes and the per-node outputs relabel with them — and the
parameterization is independent of graph size, so a GNN is invariant to isomorphism in the
right sense. Gated Graph Sequence Neural Networks (Li, Tarlow, Brockschmidt & Zemel 2015)
established a stable form of this update in which the per-node state is updated by a GRU that
takes the aggregated incoming message as input, which keeps multi-round propagation from
vanishing or exploding. This GRU-gated message passing is the standard substrate for
applying GNNs in a multi-step procedure.

*Auto-regressive factorization with teacher forcing.* Any joint distribution over a sequence
of variables factorizes into a product of conditionals, `p(x_1,...,x_T) = prod_t p(x_t | x_{<t})`.
For images, PixelCNN-style models (van den Oord et al. 2016) exploit this to train in parallel:
because the ground-truth prefix is known at training time, every conditional can be evaluated
at once without unrolling a recurrence, even though sampling remains sequential. The same trick
applies whenever the conditioning context at each step is a known prefix of the data.

*The permutation marginal.* Because `p(G) = sum_pi p(A^pi)` has factorially many terms, exact
likelihood is intractable. The available handles are: pick a single canonical ordering and model
`p(A^pi)` for that one ordering (cheap, but commits to one possibly-bad ordering); treat the
ordering as a latent and approximately marginalize; or solve a matching problem at the loss.

**Motivating diagnostic findings about existing systems.** Several facts about the landscape set
up the problem:

- *Edge independence hurts.* When edges are generated independently (given a latent or given the
  existing graph), generated graphs are observed to be of markedly lower quality than when edge
  dependencies are modeled (reported by Li et al. 2018 and You et al. 2018). Realistic graphs
  require *jointly* generated, correlated edges.
- *GNN-in-the-loop is accurate but slow.* Running a full multi-round message-passing network for
  each individual node/edge decision yields strong quality but a cost that, by the standard
  analysis, scales like `O(m * n^2 * diam(G))` for a graph with `m` edges, `n` nodes; in
  practice such models do not exceed a few tens of nodes and fail to run in reasonable time on
  graphs of hundreds of nodes.
- *RNN-only autoregression scales further but has a structural weakness.* Dropping the GNN and
  summarizing the partial graph with a recurrent hidden state lets the generation scale to
  hundreds of nodes, but the best such model still uses `O(N^2)` generation steps, and the
  sequential ordering creates a long-term bottleneck: two nodes that are adjacent in the graph
  can be far apart in the generation sequence, so the dependency between them must survive a
  long RNN path.
- *Matching-based permutation handling does not scale.* Aligning a decoded graph to a target by
  graph matching costs on the order of `O(k^4)` in the graph size `k` and `O(k^2)` parameters,
  confining matching-based models to small graphs (around a few dozen nodes).

## Baselines

**GraphRNN (You, Ying, Ren, Hamilton & Leskovec, ICML 2018).** Represents a graph under a node
ordering `pi` as a sequence of adjacency vectors `S^pi = (S_1^pi, ..., S_n^pi)`, where
`S_i^pi in {0,1}^{i-1}` records the edges from node `i` to the previously generated nodes. It
models `p(S^pi) = prod_{i=1}^{n+1} p(S_i^pi | S_{<i}^pi)` with a hierarchical RNN: a *graph-level*
RNN carries the graph state, `h_i = f_trans(h_{i-1}, S_{i-1}^pi)`, and an *edge-level* component
emits `S_i^pi`. In the simplified variant (GraphRNN-S) the edge-level model is a single MLP that
outputs a multivariate Bernoulli over `S_i^pi` — edges within a step independent; in the full
model an edge-level RNN decomposes `p(S_i^pi | S_{<i}^pi) = prod_{j} p(S_{i,j}^pi | S_{i,<j}^pi,
S_{<i}^pi)`, a dependent Bernoulli sequence, recovering edge correlations one entry at a time.
To tame the `N!` orderings, GraphRNN trains under **breadth-first-search orderings**: many node
permutations map to the same BFS ordering, so it learns over BFS orderings rather than all
orderings, and BFS bounds the adjacency-vector width to the maximum BFS frontier `M`, giving
complexity `O(M * n)`. Evaluated with MMD over graph statistics, it scales to graphs roughly
`50x` larger than earlier deep models. **Gaps it leaves open:** the full model still makes
`O(N^2)` generation decisions (one entry or column at a time); the recurrent summary, with no
GNN in the loop, does not let the next decision depend *directly* on the existing graph's
topology — it must pass through the RNN hidden state, and topologically near / sequentially far
nodes hit the long-term bottleneck; and it commits to a single BFS ordering family, described as
efficient but arguably suboptimal.

**Learning Deep Generative Models of Graphs / DeepGMG (Li, Vinyals, Dyer, Pascanu & Battaglia,
2018).** Builds a graph as a sequence of structured decisions — `f_addnode` (add a node or
stop), `f_addedge` (add an edge to the new node or not), `f_nodes` (which existing node to
connect to) — each parameterized by a graph net. Before every decision it runs `T` rounds of
message passing to refresh node states (`a_v = sum_{u: (u,v) in E} f_e(h_u, h_v, x_{uv})`, then
`h_v' = f_n(a_v, h_v)` with `f_n` a GRU), reads out a graph vector by a gated sum
`h_G = sum_v g_v * h_v`, and maps it through an MLP and softmax/sigmoid. Node states are carried
over across propagation and decision steps, making the whole process recurrent. This is the most
faithful "use the graph structure at every decision" model. **Gap:** running full propagation
per *individual* node/edge decision makes it very slow — by the standard analysis on the order
of `O(m * n^2 * diam(G))` — capping it at a few tens of nodes; and carrying states forward across
steps couples the steps, blocking the parallel-training trick. It supplies the GRU-message-passing
node update but at an unscalable price.

**GraphVAE (Simonovsky & Komodakis, 2018).** A one-shot latent-variable model in the
independent-edge family. The encoder (a graph convolutional / edge-conditioned network) maps a
graph to a Gaussian latent `z`; the decoder is an MLP that outputs a probabilistic
fully-connected graph on a fixed maximum of `k` nodes — an adjacency `Ã in [0,1]^{k x k}` whose
edges are independent Bernoulli variables, plus node/edge attribute distributions. Because
neither the target nor the decoded graph has a fixed node ordering, the reconstruction term
`p(G | G̃)` requires aligning their nodes by approximate graph matching (max-pooling matching
followed by the Hungarian algorithm). **Gap:** the matching costs about `O(k^4)` and the model
about `O(k^2)` parameters, so it is usable only for small graphs (up to `k` of order a few
dozen), and its independent-Bernoulli decoder pays the edge-independence quality penalty noted
above.

**Erdős–Rényi.** The classical random-graph baseline: every pair of nodes is connected
independently with a single probability `p` estimated by maximum likelihood from the training
graphs. It fits one statistic (edge density) and nothing about higher-order structure — no
heavy-tailed degrees, no clustering — so it is the floor a learned model must clear.

## Evaluation settings

The natural yardsticks at the time, following the protocol established by You et al. (2018):

- **Datasets** spanning sizes and domains: synthetic graphs with known structure (e.g. 2D grid
  graphs), real protein/enzyme contact graphs, and small community / ego graphs; for the
  small-graph regime, sets such as a synthetic two-community dataset (12-20 nodes, ~100 graphs),
  ego graphs extracted from a citation network (4-18 nodes, ~200 graphs), and protein-structure
  graphs from an enzyme database (10-125 nodes, ~587 graphs). Larger benchmarks include 2D grids
  (100-400 nodes), protein contact graphs (100-500 nodes), and 3D point-cloud graphs with up to
  several thousand nodes.
- **Splits and sampling protocol:** an 80/20 train/test split per dataset, with 20% of the
  training graphs held out for validation; generate the same number of samples as the test set
  and compare distributions.
- **Metrics — distributional, lower is better.** Because exact likelihood is intractable for
  ordering-dependent models, quality is measured by the Maximum Mean Discrepancy (MMD) between
  the distributions of graph statistics over generated vs. reference graphs: the node-degree
  distribution, the clustering-coefficient distribution, and the count distribution of all
  4-node orbits, plus optionally the spectrum (eigenvalues of the normalized Laplacian) for a
  global view. MMD is computed with a Gaussian kernel; the kernel distance over the
  earth-mover (first Wasserstein) distance is faithful but slow, and a total-variation distance
  can substitute for speed.
- **Fixed training budget (the per-task harness):** a fixed schedule shared across all methods —
  e.g. 500 epochs, batch size 32, single GPU, multiple seeds — so methods are compared under
  equal compute, and a model must train within this budget and sample valid undirected graphs
  without touching the evaluation labels.

## Code framework

A new model plugs into a generic step-wise graph-generation harness. The data pipeline supplies
binary, symmetric, zero-diagonal adjacency matrices padded to a maximum size, plus per-graph
node counts; the optimizer (Adam) is created once and stepped each iteration. What is *not*
settled — and is exactly what must be designed — is how to define the per-step conditional that
turns an already-generated partial graph into a distribution over the next chunk of the
adjacency matrix, and how to train it under the `N!`-ordering ambiguity. The scaffold therefore
exposes one generic empty slot for that per-step conditional, plus the standard training/sample
loop around it.

```python
import torch
import torch.nn as nn


class GraphGenerator(nn.Module):
    """Step-wise generative model of simple undirected graphs.

    The data loader supplies padded lower-triangular adjacency matrices (under one
    or more structural node orderings) plus per-graph node counts. The runner owns
    the optimizer and calls `training_loss` for a scalar loss or `sample` for
    autoregressive draws.
    """

    def __init__(self, max_nodes, **kwargs):
        super().__init__()
        self.max_nodes = max_nodes
        # TODO: the per-step conditional object we will design --- the object that
        #       maps an already-generated partial graph to a distribution over the
        #       next chunk of edges. (Architecture and output distribution TBD.)

    def _conditional_logits(self, prefix):
        """Evaluate the empty per-step slot on a teacher-forced partial graph."""
        # TODO: map an indexed partial graph to logits for its candidate edges.
        raise NotImplementedError

    def training_loss(self, adj, node_counts):
        """Combine edge-slot log-probabilities into per-graph log-probabilities,
        then aggregate over whatever node-ordering scheme we adopt."""
        # TODO: teacher-force each sub-prefix, score its candidate edges via the
        #       conditional slot, and combine the step / graph / ordering terms.
        raise NotImplementedError

    @torch.no_grad()
    def sample(self, n_samples, device):
        """Run the same conditional slot autoregressively and symmetrize output."""
        adj = torch.zeros(n_samples, self.max_nodes, self.max_nodes, device=device)
        # TODO: draw edge chunks step by step until the graph is complete; then
        #       symmetrize and read off node counts.
        node_counts = torch.full((n_samples,), 2, device=device, dtype=torch.long)
        return adj, node_counts
```

A completed generator fills the conditional slot, the probability aggregation in
`training_loss`, and the autoregressive draw in `sample`; the surrounding data
pipeline and runner optimizer remain generic.
