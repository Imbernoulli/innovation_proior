SAGPool's numbers came back and they split *exactly* along the line I drew, which is the most useful
thing a result can do: it confirms the failure precisely enough to name the cure. On the motif-driven
sets, selection paid off and then some — MUTAG jumped from gin-sum's 84.02 to 90.95 (seeds {86.17,
93.04, 93.63}) and PROTEINS from 74.54 to 77.99. Concentrating the readout on the decisive nodes undid
the dilution I diagnosed in gin-sum's uniform sum. But on NCI1 — the dataset where I predicted hard
selection would *destroy* a distributed decision — it collapsed: 70.75 mean, down nearly nine points
from gin-sum's 79.52, and with a brutal seed spread {66.76, 75.74, 69.76} where the worst seed lost
*thirteen* points off gin-sum. That is not noise; that is the irreversible top-k discard doing exactly
what I feared. NCI1's compound-activity label depends on the whole molecular context, the score
convolution is mis-calibrated on the genuinely distributed cases, and halving the node set twice — a
quarter of the atoms surviving to level 2 — throws away the very nodes that carried the signal, with no
way to recover them. So the strongest baseline is strong *on average* only because a six-point MUTAG
windfall masks a nine-point NCI1 catastrophe. It is a *less robust* readout than gin-sum, not a uniformly
better one. The single fact that organizes the whole ladder now: gin-sum won NCI1 because it threw
*nothing* away; SAGPool lost NCI1 because it threw *half* away, twice.

So I want the next readout to do what SAGPool reached for — be more expressive than a single uniform sum,
recover the per-graph-structure sensitivity that lifted MUTAG — while never committing the destructive
act that sank NCI1. Concretely: *keep every node* (non-destructive, like gin-sum) and *keep every layer*
(the JK robustness that won NCI1 in the first place), but extract *more* from that full, undiscarded set
than a lone sum can. The question becomes: given that I refuse to drop nodes, how do I make a flat
readout more discriminating than `Σ_v h_v`?

Go back to the aggregator theory, because it has been the through-line of this whole climb and it has
more to say. I established at the gin-sum rung that among {sum, mean, max} the sum is the *most*
expressive single reduction — it is injective on the multiset, keeping the counts a mean discards and the
multiplicities a max discards. But "most expressive *single* reduction" hides the real ceiling, and the
ceiling is the thing to attack now. A single permutation-invariant aggregator, *whatever* it is, is a
lossy summary of a multiset: there is no one symmetric scalar-per-feature reduction that retains all the
information in a node-feature multiset of unbounded size, because you cannot in general invert one number
back to a set. Different aggregators lose *different* information — mean keeps the distribution and drops
counts; max keeps the support and drops counts and proportions; sum keeps the count-weighted total but
blurs whether a large value came from many small contributions or a few big ones; a standard deviation
keeps the *spread* that all three of those throw away entirely. The decisive observation is that these
losses are *complementary*: the things a mean discards are partly recovered by a max, the spread neither
keeps is held by a std. So a readout that is *one* aggregator — gin-sum's sum, or SAGPool's sum+mean —
is leaving recoverable information on the table by construction. If I cannot drop nodes and I want more
than a sum, the move is to read the full node set through *several complementary aggregators at once* and
let the classifier combine them. This is precisely the diagnosis SAGPool got half-right (it did add a
mean alongside its sum) and then ruined by pairing it with a destructive selection.

But there is a second, subtler failure of single-aggregator readouts that becomes acute exactly on the
dataset I am trying to rescue, and naming it gives me the rest of the construction. Sum and mean sit at
opposite extremes of *size sensitivity*: a sum scales linearly with the number of nodes (a 110-atom NCI1
molecule produces a readout ~6× larger in magnitude than an 18-atom one), while a mean is completely
size-blind (it divides the size out). Neither is right when the *degree structure* of a node should
modulate how much it contributes. A high-degree hub in a dense region and a low-degree leaf are summed
with identical weight by a plain sum, and identical weight by a mean — yet their structural roles, and
how much their over-smoothed embeddings should be trusted, are different. What I want is a *family* of
size/degree-dependent rescalings: one that amplifies the contribution of nodes by a function of their
degree, one that attenuates it, and the identity in between, so the readout can express "weight this
graph's aggregation by its degree profile" rather than committing to the single fixed scaling that sum
(linear in count) or mean (constant) bakes in. The clean, principled form is a *logarithmic degree
scaler*: define a per-node factor `S(d, α) = (log(d+1)/δ)^α`, where `d` is the node degree, `δ` is the
average of `log(d+1)` over the nodes (a normalizer so the scaler hovers around 1 and needs no per-dataset
tuning), and `α ∈ {0, +1, −1}` gives identity, amplification (up-weight high-degree nodes), and
attenuation (up-weight low-degree nodes). The relation to the sum is worth stating precisely so I do not
overclaim it: it is the *linear* scaler `S(d)=d` that turns a mean into a sum (mean times degree is sum),
and that linear scaling is exactly the unstable thing I just rejected — it compounds across layers. The
*logarithmic* scaler I am adopting is its bounded-magnitude replacement: `log(d+1)≠d`, so it does *not*
reproduce gin-sum's plain sum, but it reinjects a controlled, monotone degree dependence — the size-aware
behavior gin-sum's fixed sum (linear in count) and a plain mean (constant) both lack — without the
blow-up. So the value is not "contains the sum" but "a stable, learnable degree knob the sum could not
offer."

Now combine the two ideas — multiple aggregators, multiple degree scalers — the way they actually
compose: take the *tensor product*. For each scaler `S(·, α)` I rescale every node's embedding by that
factor and then apply *each* aggregator; the readout is the stack of all (aggregator × scaler) channels,
`⊕ = [1, S(·,+1), S(·,−1)]ᵀ ⊗ [μ, max, σ]`. Each channel is a different lossy view of the same full,
undiscarded node set, and together they recover far more of the multiset's information than any single
view. This is the principal-neighborhood-aggregation idea — that no single aggregator is enough and the
right object is the product of a complementary aggregator set with degree scalers — *adapted* from its
original home (per-node *neighborhood* aggregation in message passing) to the place I actually control
(the *graph-level* readout, where the "neighborhood" is the whole graph's node set). I should be honest
that this is an adaptation, not a literal lift of the canonical operator. The original applies the degree
scaler to a node's *post-aggregation* neighborhood summary, with `δ` a fixed statistic of the *training
set's* degree distribution; here I scale each node's *embedding before* the graph-level pool, and I take
`δ` as a *per-graph* mean of `log(d+1)` (the harness gives me one readout call per batch with no
persistent training-degree statistic, and per-graph normalization keeps the scaler near 1 without a global
buffer). Same scaler *family* and same {aggregators × scalers} structure; the application point and the
normalizer are the harness-fitted differences, and I keep them in view rather than pretending the
graph-level readout reproduces the per-node operator exactly.

Now bring it into the scaffold, and here is where I tailor the canonical aggregator set to *this* harness
rather than copying it blind. The full method uses four aggregators — mean, max, min, std. But my nodes
are the *output of a fixed GIN backbone*, and every GIN layer ends in a ReLU, so the node embeddings
`layer_outputs` are *non-negative*. A min aggregator over non-negative, ReLU-sparse features is
near-constant zero across graphs — it carries almost no discriminative signal here and would only spend
parameter budget. So I drop min and keep the three informative aggregators {mean, max, std}; three
aggregators × three scalers = nine channels of width `hidden_dim`. This is a deliberate harness-specific
adaptation, not a shortcut: it is also what keeps me inside the parameter budget, since a `Linear(9·H, H)`
projection (36,928 params at H=64) fits comfortably under the Set2Set-sized readout allowance, where the
full `Linear(12·H, H)` would not. The std aggregator I compute the numerically safe way — `σ =
sqrt(ReLU(E[x²] − E[x]²) + ε)` — clamping the variance non-negative before the root so floating-point
noise cannot produce a NaN.

The last decision is how to fold in the layers, because keeping every layer is half the point of beating
SAGPool on NCI1. gin-sum concatenated per-layer sums (width `H·num_layers`); doing the full PNA product
*per layer* would blow the budget (nine channels × five layers). The size-neutral way to keep all depths
is to first combine the per-layer node embeddings into one jumping-knowledge node representation by an
*element-wise sum across layers* — `h = Σ_k layer_outputs[k]`, width stays `H`, every layer present, no
node dropped — and then apply the nine-channel PNA readout to that. So the readout reads every node and
every layer (gin-sum's two robustness sources, both preserved) and extracts nine complementary views
through the aggregator×scaler product (the expressivity SAGPool reached for), with *zero* hard selection
(SAGPool's fatal move, removed). The output width is `hidden_dim` after the projection, so the fixed
classifier head consumes it unchanged. The full scaffold module is in the answer.

Let me state the bar this has to clear and what I would validate, against the strongest baseline's real
numbers, because there is no leaderboard row for this readout to lean on. The headline target is the
aggregate: SAGPool's means are MUTAG 90.95, PROTEINS 77.99, NCI1 70.75. The *robustness* claim — the
whole reason to build this — is the NCI1 number, and it is the sharpest falsifiable test. Because this
readout drops no nodes and reads every layer, it should recover gin-sum's NCI1 regime (~79.5) rather than
SAGPool's collapsed 70.75, and it should do so *without* SAGPool's thirteen-point worst-seed swing — if
NCI1 does not climb back well above 73 with a tight seed band, the "non-destructive readout recovers
distributed signal" thesis is wrong and I would conclude the gain on MUTAG genuinely *requires* throwing
nodes away. On MUTAG and PROTEINS the claim is softer: the multi-aggregator-plus-scaler product is
strictly more expressive than gin-sum's single sum, so I expect it to clear gin-sum's 84.02 and 74.54;
whether it matches SAGPool's motif-driven MUTAG peak of 90.95 is the genuine open question — selection
may simply be the better tool when the label *is* a single small motif, in which case this readout's win
is "SAGPool-on-NCI1 robustness at near-SAGPool MUTAG accuracy," a better *average and a far better worst
case*. The crisp success criterion I would hold it to: beat SAGPool's three-dataset mean *and* cut the
cross-dataset variance, with the NCI1 recovery as the load-bearing evidence. What I would validate first,
before trusting any mean, is the per-seed NCI1 row — if the degree scalers are doing their job, the
worst NCI1 seed should never fall the way SAGPool's 66.76 did, because nothing is ever discarded.

The causal chain in one breath: SAGPool's hard top-k won the motif sets (MUTAG 84→91) but *destroyed*
NCI1's distributed decision (79.5→70.75, worst seed 66.76) by discarding half the nodes twice → so keep
every node and every layer (gin-sum's two robustness sources) but extract more than a single sum can,
since *any* one aggregator is a lossy multiset summary and their losses are complementary → read the full
node set through several complementary aggregators {mean, max, std} crossed with logarithmic degree
scalers {identity, amplify, attenuate}, `[1, S(·,+1), S(·,−1)]ᵀ ⊗ [μ, max, σ]`, the tensor-product
operator that is size/degree-aware (a stable log degree-knob the sum lacks, not a re-derivation of the sum)
where sum and mean are not → adapt to the
harness (drop the min aggregator, useless on post-ReLU GIN features and budget-costly; element-wise-sum
the layers into one JK node rep so all depths stay) → drop it into `GraphReadout`, projecting `9H→H`,
expecting it to recover gin-sum's NCI1 regime without SAGPool's worst-seed collapse while clearing
gin-sum on the motif sets — a more robust readout that gains expressivity without ever throwing a node
away.
