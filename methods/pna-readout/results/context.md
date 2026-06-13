# Context: how much can a single neighbor-aggregator retain, and what does it take to retain more

## Research question

A message-passing GNN updates each node by aggregating a *multiset* of its neighbors' feature vectors
and combining the result with the node's own vector. The aggregator is the heart of the layer: it is a
permutation-invariant function that must compress a variable-size multiset of vectors into one vector.
Almost every architecture picks *one* such aggregator — a mean, a sum, a max, an attention-weighted
average — by intuition, and the field's expressivity theory has so far been written for a *countable*
feature universe (discrete node labels), where a single carefully-chosen sum can be made injective on
multisets.

But the vectors an aggregator actually sees in a deep network are not from a countable alphabet: they
are *continuous* hidden representations in ℝ^d. The precise problem: in the continuous setting, how much
of a neighbor multiset can a single continuous aggregator retain, and if one is not enough, exactly how
many — and which — aggregators does it take to retain "enough"? A solution must (1) say something sharp
about the number of aggregators needed to tell continuous multisets apart, (2) turn that into a concrete,
trainable aggregation operator, and (3) remain robust as a function of node degree, since real graphs mix
low-degree leaves and high-degree hubs and an aggregator that behaves wildly with degree will not train
stably across depth.

## Background

By this point neighborhood-aggregation (message-passing) GNNs are the standard family (Gilmer et al.
2017). One layer has the shape

```
m_{j->i} = M( x_i, e_{j->i}, x_j )                       (message)
x_i'     = U( x_i, AGG_{ j in N(i) } m_{j->i} )          (aggregate + update)
```

with AGG a permutation-invariant reduction over the neighbor multiset. The load-bearing prior results:

- **Expressivity on countable features (Xu, Hu, Leskovec & Jegelka 2019).** When node features come from
  a *countable* universe, a *sum* aggregator with a learned per-element map is injective on bounded
  multisets, which makes a GNN as discriminative as the 1-Weisfeiler-Lehman test — the ceiling for this
  family. The crucial caveat: this argument is built on countability (a place-value encoding of
  multiplicities). It says nothing direct about multisets of *real-valued* vectors.

- **Deep Sets / sum-decomposition (Zaheer et al. 2017).** Any permutation-invariant function of a set
  from a countable universe factors as ρ(Σ_x φ(x)). Again the clean statement is for countable inputs;
  the continuous case is where single-aggregator schemes get shaky.

- **The aggregators in use, and what each discards.** Mean keeps the *distribution* of neighbor features
  but divides out absolute counts (k copies of a multiset have the same mean as one copy). Max keeps the
  *support* — which distinct features are present — and discards counts and proportions. Sum keeps the
  count-weighted total but is degree-coupled: its magnitude scales with the number of neighbors. None of
  these keeps the *spread* of the neighborhood; the standard deviation does, and is itself
  permutation-invariant. The empirical folklore is that different tasks favor different aggregators,
  which is a hint that no one of them is sufficient.

- **Degree and signal scale.** A node's degree varies by orders of magnitude across a graph. A sum
  aggregator's output magnitude grows linearly with degree, so stacking sum layers can amplify signals
  exponentially with depth in dense regions and destabilize training; a mean is degree-blind and cannot
  *use* degree even when degree is informative. There is a tension: we want degree to *modulate*
  aggregation (it is structurally meaningful) without letting it blow up the activations.

- **Borsuk–Ulam (topology).** A classical theorem: any continuous map from the n-sphere to ℝ^n sends some
  antipodal pair to the same point. It is the standard tool for proving that a continuous map *cannot* be
  injective on a space of a given dimension — exactly the shape of argument one would need to lower-bound
  the number of continuous aggregators required to separate multisets.

The diagnostic that motivates everything below: single-aggregator GNNs that are provably maximal on
*countable* features nonetheless fail to distinguish simple *continuous* neighbor multisets, and which
multisets they confuse depends on which single aggregator they chose. That failure is not an optimization
artifact; it is information lost at the aggregation step, and it differs systematically with the
aggregator.

## Baselines

The aggregation schemes a new design is measured against and reacts to.

- **Mean aggregation (GCN, Kipf & Welling 2017).** AGG = average of neighbors. Degree-stable and good
  where the neighborhood's feature *distribution* is the signal. Gap: discards counts, and a single
  continuous mean is not injective on multisets of size > 1.

- **Sum aggregation (GIN, Xu et al. 2019).** AGG = sum of (learned) neighbor features. Injective on
  *countable* multisets and WL-powerful there. Gap: the injectivity argument does not transfer to
  continuous features, and the sum's magnitude is degree-coupled, which can destabilize deep stacks.

- **Max-pooling aggregation (GraphSAGE, Hamilton et al. 2017).** AGG = elementwise max. Robust, picks out
  salient neighbors. Gap: keeps only the support — counts and proportions are gone — and is a single
  continuous aggregator, so it confuses any two multisets with the same coordinatewise maxima.

- **Attention-weighted mean (GAT, Veličković et al. 2018).** AGG = a learned convex combination of
  neighbors. More flexible than a plain mean, but it is still *one* mean-like reduction; it reweights the
  distribution, it does not add a complementary view, and it remains degree-blind in magnitude.

## Evaluation settings

The natural yardsticks are tasks where the aggregator's information content is directly stressed:

- **Multi-task synthetic graph-theory benchmark.** A battery of node- and graph-level targets that demand
  faithful neighborhood aggregation — single-source shortest paths, node eccentricity, graph
  connectivity, diameter, spectral radius, Laplacian features — on random graphs of varied structure.
  Because the targets are exact graph functions, an aggregator that loses neighborhood information shows
  up as error directly; a stringent extra test is *extrapolation* to graphs larger than any seen in
  training, which punishes degree-unstable aggregation.
- **Real-world molecular/chemical and vision graphs** (e.g. ZINC molecular regression, a molecular
  property classification set, superpixel-graph image classification). Structure-heavy chemical graphs
  are where richer aggregation should pay; near-regular grid graphs (vision) are where it should matter
  least, a useful negative control.
- **Protocol.** A fixed message-passing backbone with the aggregation operator swapped; a parameter-
  matched comparison (enlarge baselines so any gain is not just more parameters); accuracy / MAE per
  task, and the extrapolation gap to larger graphs.

## Code framework

The harness exposes a message-passing layer whose only open slot is the aggregation of the neighbor
multiset; messages, the per-node update MLP, and the training loop already exist. We write the scaffold
with the aggregation operator left as the missing piece.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import degree


class AggregationOperator(nn.Module):
    """Reduce a node's neighbor-message multiset into one vector.

    Inputs to forward():
        messages   [E, F]   one message per edge (already computed by M)
        index      [E]      destination node of each edge (for scatter)
        node_degree[N]      degree of each destination node
        num_nodes  int
    Output:
        [N, out_F]   one aggregated vector per node
    """

    def __init__(self, in_dim, out_dim):
        super().__init__()
        # TODO: choose the aggregation. set out_dim to the post-aggregation width.
        self.out_dim = out_dim

    def forward(self, messages, index, node_degree, num_nodes):
        # TODO: permutation-invariant reduction of the neighbor multiset per node.
        pass


class MPNNLayer(nn.Module):
    """message M -> AggregationOperator -> update U. The aggregator is the open slot."""

    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.msg = nn.Linear(2 * in_dim, hidden_dim)
        self.aggr = AggregationOperator(hidden_dim, hidden_dim)
        self.upd = nn.Sequential(
            nn.Linear(in_dim + self.aggr.out_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x, edge_index):
        src, dst = edge_index
        m = self.msg(torch.cat([x[dst], x[src]], dim=-1))
        deg = degree(dst, num_nodes=x.size(0), dtype=x.dtype)
        a = self.aggr(m, dst, deg, x.size(0))
        return self.upd(torch.cat([x, a], dim=-1))
```

The message and update are fixed; the question is only what reduction fills `AggregationOperator`.
