The stripped DiffPool came back saying exactly what I feared when I built it. On PROTEINS it sat at $70.98$ mean accuracy with a seed band so tight — $\{70.98, 71.44, 70.52\}$ — that it was plainly not learning a partition; it collapsed to the same diffuse, mean-like answer every run, the signature of an assignment softmax that never committed to crisp clusters. PROTEINS, the dataset with the larger graphs where a real hierarchy would help most, was its *lowest* of the three. NCI1 landed at an unremarkable $78.25$, and MUTAG gave away the variance story with a six-point spread $\{85.64, 78.71, 79.77\}$, the coin-flip I expected on 188 graphs. So the diagnosis is clean and it is not a tuning problem: the one idea I kept — a learned soft clustering pooled as $S^{\top}X$ — needed the auxiliaries the harness would not let me wire in, and without them it became a *mean* (discarding counts) and, just as bad, made no use of the per-layer node embeddings sitting unused in `layer_outputs`. I threw away two things at once: the *injective* reduction that keeps multiplicities, and the *multi-scale* signal already present in the layer stack.

So I stop trying to learn a hierarchy with no support for it and go back to first principles on the flat reduction, where I actually have a clean theory. The question is narrow: among permutation-invariant reductions over the multiset of node embeddings, which keeps the most information? A mean has a precise defect — take a multiset and an inflated copy where every multiplicity is scaled by the same integer $k$; the mean is identical for both, since $\tfrac{1}{kn}\,k\sum f = \tfrac{1}{n}\sum f$, so it captures only the *distribution* (the proportions) and is blind to absolute counts. A max is worse: $\max_x f(x)$ depends only on which *distinct* elements are present, so it sees only the *support* and loses both counts and proportions. The sum, by contrast, is *injective* on bounded multisets: with a suitable per-element map $f$, a reduction like $\sum_x N^{-Z(x)}$ positionally encodes the exact multiplicity profile in base $N$, so distinct multisets give distinct sums. There is a strict pecking order in discriminative power, $\text{sum} \sqsupset \text{mean} \sqsupset \text{max}$, and DiffPool by drifting to a mean sat below the top. Crucially, the GIN backbone in front of me was *built* so that an injective sum readout makes the whole network as discriminating as the Weisfeiler–Lehman test — the ceiling for any message-passing GNN — so using anything weaker than a sum at the readout wastes the backbone's expressivity. That settles the aggregator.

I propose the **GIN JK-Sum readout**: sum-pool each layer's node embeddings *independently* into a per-layer graph vector, then *concatenate* across layers,

$$h_G = \mathrm{CONCAT}\Big(\ \textstyle\sum_{v\in G} h_v^{(k)}\ :\ k = 1\dots K\ \Big).$$

The reason to read every layer, not just the last, is that a node's embedding after $k$ rounds is a learned summary of its rooted subtree of height $k$ — the height-1 embedding sees immediate neighbors, the height-5 embedding sees five hops out. Deeper features are more global and more discriminative, so I want depth for power; but the deepest features are also the most *specialized*, and on small sets like MUTAG the shallower, more local features often generalize better, while over-smoothing can wash the last layer out. Rather than gamble on one depth — and reading only `x`, the final layer, *is* that gamble — I read all of them. This is the jumping-knowledge idea, and it recovers the multi-scale signal DiffPool was reaching for, except the hierarchy is already present *for free* in the layer stack and I just have to read it. Each per-layer sum is injective on its own multiset; concatenating rather than mixing keeps all $K$ side by side so the downstream classifier can weight depths as it likes. There is even a clean reading of *what* this computes: summing learned height-$k$ subtree embeddings is the continuous analogue of *counting subtrees*, exactly what the WL subtree kernel does by hand — except the subtrees are embedded in a continuous space, so *similar* subtrees land near each other, something one-hot WL labels can never do. The output width is $\text{hidden\_dim}\times\text{num\_layers} = 5\times64 = 320$ with no projection bottleneck, so every depth's full signal reaches the classifier.

There is one wrinkle, and it is the single place this readout departs from the bare textbook JK-sum, so I reason it out rather than copy it. Concatenating per-layer *sum* pools concatenates vectors at very different scales: the deeper GIN layers, after several rounds of neighbor summation plus the $(1+\epsilon)$ self-weighting inside `GINConv`, can have systematically larger magnitudes than the shallow ones, and a sum over a 100-node graph is $\sim$100$\times$ the per-node scale while a sum over a 17-node MUTAG graph is $\sim$17$\times$ — so one concatenated vector mixes wildly different magnitudes across both layers and graph sizes. Fed straight into the classifier, the large-magnitude coordinates dominate the first linear layer's gradient, the small ones are effectively ignored, and some folds simply fail to converge — the optimization stalls on the scale mismatch, not on anything about the graphs. The fix that keeps the injectivity intact is to **batch-normalize each layer's graph-level embedding before concatenating**: a `BatchNorm1d(hidden_dim)` per layer, applied to the $[B, \text{hidden\_dim}]$ pooled vectors. BatchNorm is an affine, invertible (at fixed statistics) rescaling, so it does not collapse the multiset distinctions the sum encoded — it just puts every layer's graph embedding on a common, well-conditioned scale so the classifier sees all five depths on equal footing and every fold trains. This per-layer graph-BN is the one piece I add beyond "sum each layer and concatenate," and it is there for optimization stability, not expressivity.

In the scaffold this is a `nn.ModuleList` of `num_layers` `BatchNorm1d(hidden_dim)` modules; in `forward` I loop over `layer_outputs`, do `global_add_pool(h, batch)` for each, run it through that layer's BN, collect, and return `torch.cat(graph_embs, dim=-1)`, with `output_dim = hidden_dim * num_layers`. I read from `layer_outputs` and ignore `x`, since `x` is just `layer_outputs[-1]`, already included; no `edge_index`, no dense batching, no clusters. The delta from DiffPool is precise and points the opposite way: DiffPool spent its capacity trying to *learn* structure with none of the support that makes it pay and ended up mean-like; this rung spends *no* learned capacity on the pooling itself — it takes the provably-injective sum, applies it at every depth, and lets the downstream classifier, not a fragile assignment softmax, decide how to weigh the scales. The only learned parameters in the readout are the per-layer BNs, and they exist for conditioning, not clustering. The falsifiable claim is the ranking: PROTEINS should clear DiffPool's $70.98$ clearly (if it does not, my "DiffPool became a mean" story is wrong), NCI1 should clear $78.25$ where keeping counts and reading every depth matters most, and MUTAG's seed variance should tighten now that the readout is deterministic.

```python
class GraphReadout(nn.Module):
    """GIN JK-Sum Readout (Xu et al., 2019).

    Concatenates sum-pooled embeddings from all GIN layers
    (Jumping Knowledge). Each layer's graph embedding is batch-normalized
    before concatenation to stabilize training -- this prevents the
    different-scale representations across layers from causing
    optimization issues (some folds failing to converge).

    The output dimension is hidden_dim * num_layers, matching the
    original GIN paper's readout.
    """

    def __init__(self, hidden_dim, num_layers):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        # Full concatenated dimension -- no projection bottleneck
        self.output_dim = hidden_dim * num_layers
        # Per-layer batch normalization on graph-level embeddings
        self.graph_bns = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)
        ])

    def forward(self, x, edge_index, batch, layer_outputs):
        # Sum-pool each layer's node embeddings independently
        graph_embs = []
        for i, h in enumerate(layer_outputs):
            g = global_add_pool(h, batch)
            g = self.graph_bns[i](g)
            graph_embs.append(g)
        # Concatenate all layers (Jumping Knowledge)
        return torch.cat(graph_embs, dim=-1)
```
