**Problem (from the SAGPool rung).** Hard top-k selection won the motif-driven sets (MUTAG 84.02→90.95,
PROTEINS 74.54→77.99) but *collapsed* NCI1 (gin-sum 79.52 → 70.75, worst seed 66.76, −13 pts): discarding
half the nodes twice destroys a label that depends on the whole molecular context, and the discard is
irreversible. The strongest baseline is strong on average only because a MUTAG windfall masks an NCI1
catastrophe — a *less robust* readout than gin-sum, which won NCI1 precisely by throwing nothing away.

**Key idea.** Keep gin-sum's two robustness sources — *every node* (non-destructive) and *every layer*
(jumping knowledge) — but extract more than a single sum can. Any one permutation-invariant aggregator is
a lossy multiset summary, and the losses are *complementary*: mean keeps the distribution, max the support,
std the spread. Read the full undiscarded node set through several aggregators at once, and cross them with
*logarithmic degree scalers* `S(d, α) = (log(d+1)/δ)^α` (α ∈ {0,+1,−1}: identity / amplify high-degree /
attenuate), so the readout is size/degree-aware where sum (linear in count) and mean (size-blind) are not.
The operator is the tensor product `[1, S(·,+1), S(·,−1)]ᵀ ⊗ [μ, max, σ]` — Principal Neighbourhood
Aggregation lifted from per-node *neighborhood* aggregation to the *graph-level* readout (the whole graph's
node set as the "neighborhood").

**Why it works.** Several complementary views recover far more of the node-multiset's information than a
lone sum (the expressivity SAGPool reached for), while *nothing is ever discarded* (the destructive move
SAGPool committed, removed). The degree scalers add size/degree sensitivity a fixed sum or mean cannot
express: the *linear* scaler `S(d)=d` would turn a mean into a sum, and the *logarithmic* one used here is
its bounded-magnitude replacement — it does not reproduce a sum (`log(d+1)≠d`) but reinjects a controlled
degree dependence without the sum's blow-up.

**Scaffold edit / hyperparameters.** Harness-specific adaptation of canonical PNA: the GIN node embeddings
are post-ReLU *non-negative*, so the `min` aggregator is near-constant-zero and uninformative — drop it,
keep {mean, max, std}; 3 aggregators × 3 scalers = 9 channels of width `hidden_dim`, projected `9H→H`
(36,928 params at H=64, inside the Set2Set-sized budget where the full 12-channel product would not fit).
Fold layers by an *element-wise sum across* `layer_outputs` into one width-`H` JK node representation (every
depth present, no node dropped), then apply the 9-channel readout. δ is the per-graph mean of `log(d+1)`
(a normalizer so scalers hover near 1, no per-dataset tuning); std computed safely as
`sqrt(ReLU(E[x²]−E[x]²)+ε)`. `output_dim = hidden_dim`.

**Bar to clear (no leaderboard row for this readout).** The load-bearing test is NCI1 robustness: because
no node is dropped and every layer is read, NCI1 should recover gin-sum's ~79.5 regime rather than SAGPool's
70.75, *without* SAGPool's 13-point worst-seed swing — if NCI1 does not climb well above 73 with a tight
band, the "non-destructive readout recovers distributed signal" thesis is wrong. On MUTAG/PROTEINS the
multi-aggregator-plus-scaler product is strictly more expressive than gin-sum's single sum, so it should
clear gin-sum's 84.02 and 74.54; whether it matches SAGPool's motif-driven MUTAG 90.95 is the open question.
Success = beat SAGPool's three-dataset mean *and* cut the cross-dataset variance, with NCI1 recovery the
evidence; validate the per-seed NCI1 row first.

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
