SAGPool's numbers split *exactly* along the line I had drawn, which is the most useful thing a result can do — it confirms the failure precisely enough to name the cure. On the motif-driven sets selection paid off and then some: MUTAG jumped from gin-sum's $84.02$ to $90.95$ (seeds $\{86.17, 93.04, 93.63\}$) and PROTEINS from $74.54$ to $77.99$, with concentrating the readout on the decisive nodes undoing the dilution I diagnosed in the uniform sum. But on NCI1 — where I predicted hard selection would destroy a distributed decision — it collapsed: $70.75$ mean, down nearly nine points from gin-sum's $79.52$, with a brutal seed spread $\{66.76, 75.74, 69.76\}$ whose worst seed lost *thirteen* points. That is not noise; it is the irreversible top-k discard doing exactly what I feared. NCI1's compound-activity label depends on the whole molecular context, the score convolution is mis-calibrated on the genuinely distributed cases, and halving the node set twice throws away the very atoms that carried the signal, with no way to recover them. So the strongest baseline is strong *on average* only because a six-point MUTAG windfall masks a nine-point NCI1 catastrophe. The single fact that organizes the whole ladder now: gin-sum won NCI1 because it threw *nothing* away; SAGPool lost NCI1 because it threw *half* away, twice.

So I want the next readout to do what SAGPool reached for — be more expressive than a single uniform sum, recover the per-graph-structure sensitivity that lifted MUTAG — while never committing the destructive act that sank NCI1. Concretely: *keep every node* (non-destructive, like gin-sum) and *keep every layer* (the jumping-knowledge robustness that won NCI1 in the first place), but extract *more* from that full, undiscarded set than a lone sum can. Going back to the aggregator theory that has been the through-line of this climb: I established that among $\{\text{sum},\text{mean},\text{max}\}$ the sum is the *most* expressive single reduction. But "most expressive *single* reduction" hides the real ceiling. *Any* one permutation-invariant aggregator is a lossy summary of a multiset — you cannot in general invert one scalar-per-feature back to a set — and different aggregators lose *different* information: mean keeps the distribution and drops counts, max keeps the support and drops counts and proportions, sum keeps the count-weighted total but blurs whether a large value came from many small contributions or a few big ones, and a standard deviation keeps the *spread* all three throw away entirely. The decisive observation is that these losses are *complementary*: what a mean discards is partly recovered by a max, and the spread neither keeps is held by a std. So a readout that is *one* aggregator — gin-sum's sum, or SAGPool's sum+mean — leaves recoverable information on the table by construction. If I cannot drop nodes and I want more than a sum, I read the full node set through *several complementary aggregators at once* and let the classifier combine them.

There is a second, subtler defect of single-aggregator readouts that becomes acute on exactly the dataset I am rescuing, and naming it gives me the rest of the construction. Sum and mean sit at opposite extremes of *size sensitivity*: a sum scales linearly with node count (a 110-atom molecule produces a readout $\sim$6$\times$ larger in magnitude than an 18-atom one), while a mean is completely size-blind. Neither is right when the *degree structure* of a node should modulate how much it contributes — a high-degree hub in a dense region and a low-degree leaf are weighted identically by both sum and mean, yet their structural roles, and how much their possibly over-smoothed embeddings should be trusted, differ. What I want is a *family* of degree-dependent rescalings: one that amplifies high-degree nodes, one that attenuates them, and the identity in between, so the readout can express "weight this graph's aggregation by its degree profile" rather than committing to a single fixed scaling. The clean, principled form is a *logarithmic degree scaler*,

$$S(d, \alpha) = \left(\frac{\log(d+1)}{\delta}\right)^{\alpha},$$

where $d$ is the node degree, $\delta$ is the average of $\log(d+1)$ over the nodes (a normalizer so the scaler hovers around 1 and needs no per-dataset tuning), and $\alpha \in \{0, +1, -1\}$ gives identity, amplification, and attenuation. The relation to the sum is worth stating precisely so I do not overclaim it: it is the *linear* scaler $S(d)=d$ that turns a mean into a sum (mean times degree is sum), and that linear scaling is exactly the unstable thing I just rejected — it compounds across layers. The *logarithmic* scaler is its bounded-magnitude replacement: since $\log(d+1)\neq d$ it does *not* reproduce gin-sum's plain sum, but it reinjects a controlled, monotone degree dependence the fixed sum (linear in count) and the plain mean (constant) both lack, without the blow-up. The value is not "contains the sum" but "a stable, learnable degree knob the sum could not offer."

I propose a **PNA-style degree-aware multi-aggregation readout**. The two ideas — multiple aggregators, multiple degree scalers — compose the way they actually should, as a *tensor product*: for each scaler I rescale every node's embedding by that factor and then apply *each* aggregator, so the readout is the stack of all (aggregator $\times$ scaler) channels,

$$\bigoplus \;=\; \big[\,1,\; S(\cdot,+1),\; S(\cdot,-1)\,\big]^{\top} \otimes \big[\,\mu,\; \max,\; \sigma\,\big].$$

Each channel is a different lossy view of the same full, undiscarded node set, and together they recover far more of the multiset's information than any single view. This is the principal-neighborhood-aggregation idea — that no single aggregator suffices and the right object is the product of a complementary aggregator set with degree scalers — *adapted* from its original home (per-node *neighborhood* aggregation in message passing) to the place I actually control, the *graph-level* readout, where the "neighborhood" is the whole graph's node set. I keep the adaptation in view rather than pretending it is a literal lift: the canonical operator applies the scaler to a node's *post-aggregation neighborhood* summary with $\delta$ a fixed statistic of the *training set's* degree distribution, whereas here I scale each node's *embedding before* the graph-level pool and take $\delta$ as a *per-graph* mean of $\log(d+1)$ (the harness gives one readout call per batch with no persistent training-degree statistic, and per-graph normalization keeps the scaler near 1 without a global buffer). Same scaler *family*, same {aggregators $\times$ scalers} structure; the application point and the normalizer are the harness-fitted differences.

I also tailor the aggregator set to this harness rather than copying it blind. Canonical PNA uses four aggregators — mean, max, min, std — but my nodes are the output of a fixed GIN backbone and every GIN layer ends in a ReLU, so the embeddings in `layer_outputs` are *non-negative*. A min aggregator over non-negative, ReLU-sparse features is near-constant zero across graphs: it carries almost no discriminative signal here and would only spend budget. So I drop min and keep the three informative aggregators $\{\text{mean}, \max, \text{std}\}$ — three aggregators $\times$ three scalers $=$ nine channels of width $\text{hidden\_dim}$, projected $9H\to H$. This is also what keeps me inside the parameter budget: a $\text{Linear}(9H, H)$ projection (36,928 params at $H=64$) fits comfortably under the Set2Set-sized readout allowance where the full $\text{Linear}(12H, H)$ would not. The std I compute the numerically safe way, $\sigma = \sqrt{\mathrm{ReLU}\!\big(\mathbb{E}[x^2] - \mathbb{E}[x]^2\big) + \epsilon}$, clamping the variance non-negative before the root so floating-point noise cannot produce a NaN.

The last decision is how to fold in the layers, because keeping every layer is half the point of beating SAGPool on NCI1. gin-sum concatenated per-layer sums (width $H\cdot\text{num\_layers}$); doing the full nine-channel product *per layer* would blow the budget. The size-neutral way to keep all depths is to first combine the per-layer node embeddings into one jumping-knowledge node representation by an *element-wise sum across layers*, $h = \sum_k \text{layer\_outputs}[k]$ — width stays $H$, every layer present, no node dropped — and then apply the nine-channel readout to that. So the readout reads every node and every layer (gin-sum's two robustness sources, both preserved) and extracts nine complementary views through the aggregator$\times$scaler product (the expressivity SAGPool reached for), with *zero* hard selection (SAGPool's fatal move, removed). The output width is $\text{hidden\_dim}$ after the projection, so the fixed classifier head consumes it unchanged.

The bar this has to clear, against the strongest baseline's real numbers, is sharpest on NCI1, since there is no leaderboard row for this readout to lean on. Because it drops no nodes and reads every layer, NCI1 should recover gin-sum's $\sim$79.5 regime rather than SAGPool's collapsed $70.75$, and it should do so *without* SAGPool's thirteen-point worst-seed swing — if NCI1 does not climb well above $73$ with a tight band, the "non-destructive readout recovers distributed signal" thesis is wrong and I would conclude the MUTAG gain genuinely *requires* throwing nodes away. On MUTAG and PROTEINS the multi-aggregator-plus-scaler product is strictly more expressive than gin-sum's single sum, so it should clear $84.02$ and $74.54$; whether it matches SAGPool's motif-driven MUTAG peak of $90.95$ is the genuine open question — selection may simply be the better tool when the label *is* a single small motif, in which case this readout's win is SAGPool-on-NCI1 robustness at near-SAGPool MUTAG accuracy, a better *average and a far better worst case*. The crisp success criterion: beat SAGPool's three-dataset mean *and* cut the cross-dataset variance, with the NCI1 recovery as the load-bearing evidence — and the first thing I would validate, before trusting any mean, is the per-seed NCI1 row, because if the degree scalers are doing their job the worst NCI1 seed should never fall the way SAGPool's $66.76$ did, since nothing is ever discarded.

```python
class GraphReadout(nn.Module):
    """PNA-style graph readout (Corso et al., 2020), lifted to the graph level.

    Keeps every node and every layer (no selection, no node dropping).
    Reads the full node-embedding multiset through complementary aggregators
    {mean, max, std} crossed with logarithmic degree scalers
    {identity, amplify (+1), attenuate (-1)} -- a 9-channel tensor product --
    then projects back to hidden_dim. The 'min' aggregator of canonical PNA is
    dropped: post-ReLU GIN features are non-negative, so min is near-constant.
    """

    def __init__(self, hidden_dim, num_layers):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_aggrs = 3    # mean, max, std
        self.num_scalers = 3  # identity, amplification (+1), attenuation (-1)
        self.proj = nn.Linear(
            self.num_aggrs * self.num_scalers * hidden_dim, hidden_dim)
        self.output_dim = hidden_dim
        self.eps = 1e-5

    def _aggregate(self, feat, batch):
        # mean, max, and numerically safe std over the node set per graph.
        # (global_max_pool is not in the scaffold's module imports; pull it
        # from torch_geometric.nn locally, as the other baselines do.)
        from torch_geometric.nn import global_max_pool
        m = global_mean_pool(feat, batch)
        mx = global_max_pool(feat, batch)
        msq = global_mean_pool(feat * feat, batch)
        s = torch.sqrt(F.relu(msq - m * m) + self.eps)
        return [m, mx, s]

    def forward(self, x, edge_index, batch, layer_outputs):
        # Jumping-knowledge node representation: element-wise sum across layers
        # (every layer present, width stays hidden_dim, no node dropped).
        h = torch.stack(layer_outputs, dim=0).sum(dim=0)            # [N, H]

        # Per-node degree and the per-graph log-degree normalizer delta,
        # so the scaler S(d, alpha) = (log(d+1)/delta)^alpha hovers near 1.
        deg = degree(edge_index[1], num_nodes=h.size(0), dtype=h.dtype)
        log_deg = torch.log(deg + 1.0)                              # [N]
        delta = global_mean_pool(log_deg.unsqueeze(-1), batch).clamp(min=self.eps)
        ratio = (log_deg / delta[batch].squeeze(-1)).clamp(min=self.eps)  # [N]

        # Scalers applied to node embeddings before the graph-level pool:
        # identity (raw h), amplification (* ratio), attenuation (/ ratio).
        amp = h * ratio.unsqueeze(-1)                               # S(d, +1)
        att = h / ratio.unsqueeze(-1)                               # S(d, -1)

        channels = (self._aggregate(h, batch)
                    + self._aggregate(amp, batch)
                    + self._aggregate(att, batch))                 # 9 x [B, H]
        return self.proj(torch.cat(channels, dim=-1))              # [B, H]
```
