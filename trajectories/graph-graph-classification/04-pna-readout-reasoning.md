SAGPool's numbers came back and they split *exactly* along the line I drew, which is the most useful thing
a result can do: it confirms the failure precisely enough to name the cure. Let me read the split
quantitatively before I react to it. On the motif-driven sets, selection paid off and then some — MUTAG
jumped from gin-sum's 84.02 to 90.95, a lift of 6.93 points, and PROTEINS from 74.54 to 77.99, a lift of
3.45. Concentrating the readout on the decisive nodes undid the dilution I diagnosed in gin-sum's uniform
sum, and it undid it most where the motif is smallest and most local, MUTAG. So far the prediction is
vindicated. But on NCI1 — the dataset where I predicted hard selection would *destroy* a distributed
decision — it collapsed: 70.75 mean, down 8.77 points from gin-sum's 79.52, and with a brutal seed spread
{66.76, 75.74, 69.76} whose worst seed at 66.76 sits nearly thirteen points below gin-sum's 79.52 mean and
whose range of 8.98 is almost four times gin-sum's tight 2.36. That is not noise; that is the irreversible
top-k discard doing exactly what I feared. NCI1's compound-activity label depends on the whole molecular
context, the single-conv score is mis-calibrated on the genuinely distributed cases, and halving the node
set twice — a quarter of the atoms surviving to level 2 — throws away the very atoms that carried the
signal, with no way to recover them.

The macro_f1 column sharpens the autopsy in a way the accuracy mean alone hides. On NCI1, gin-sum's
acc-minus-F1 gap was 0.13 — the two classes balanced, both metrics moving together. SAGPool's NCI1 gap
widens to 70.75 − 69.04 = 1.71, more than a tenfold increase, and on the worst seed it is worse still:
66.76 accuracy against 63.29 macro_f1, a gap of 3.47. That widening is the signature of a discard that is
not class-neutral: when the score drops the atoms that define the minority class, accuracy can coast on the
majority while macro_f1 falls away, and that is precisely what a hard, mis-calibrated selection does on a
distributed decision. And the variance story is not confined to NCI1 either — even where SAGPool *won*, it
won noisily: MUTAG's seed range widened from gin-sum's 3.19 to 7.46, and PROTEINS's from 1.17 to 5.66. So
selection is a high-variance operator everywhere; it just happens to have positive expected value on the
motif sets and catastrophically negative value on the distributed one.

Now zoom out to the ranking arithmetic, because it is more honest than any single dataset. SAGPool's
three-dataset mean is (90.95 + 77.99 + 70.75)/3 = 79.90; gin-sum's is (84.02 + 74.54 + 79.52)/3 = 79.36. So
the strongest baseline beats the robust one by a mere 0.54 points on average — and it buys that half-point
with a nine-point hole in NCI1. Its cross-dataset spread (best dataset mean minus worst) is
90.95 − 70.75 = 20.20 points; gin-sum's is 84.02 − 74.54 = 9.48. SAGPool has more than twice the
cross-dataset spread of gin-sum for barely any gain in the mean. So the strongest baseline is strong *on
average* only because a six-point MUTAG windfall masks a nine-point NCI1 catastrophe. It is a *less robust*
readout than gin-sum, not a uniformly better one. The single fact that organizes the whole ladder now:
gin-sum won NCI1 because it threw *nothing* away; SAGPool lost NCI1 because it threw *half* away, twice.

So I want the next readout to do what SAGPool reached for — be more expressive than a single uniform sum,
recover the per-graph-structure sensitivity that lifted MUTAG — while never committing the destructive act
that sank NCI1. Concretely: *keep every node* (non-destructive, like gin-sum) and *keep every layer* (the
jumping-knowledge robustness that won NCI1 in the first place), but extract *more* from that full,
undiscarded set than a lone sum can. The question becomes sharp: given that I refuse to drop nodes, how do
I make a flat readout more discriminating than Σ_v h_v?

Let me first ask whether there is a non-destructive route I should prefer to the one I am circling toward,
because it would be lazy to reach for a multi-aggregator readout without weighing the obvious alternatives.
The initial framing named two set-pooling readouts that keep every node — an attention-weighted pool and a
Set2Set-style recurrent set network — and the parameter budget is literally sized against the larger of
them, so both are on the table and both are affordable. Walk the attention pool: it learns a scalar weight
per node and returns a weighted sum, non-destructive, no discard. But it is still *one* aggregation — one
weighted average — a single lossy view of the node multiset, and it re-introduces the learned softmax over
nodes that I watched go diffuse in DiffPool; on a distributed decision like NCI1 it has the same freedom to
flatten toward a plain mean and the same fragility. Walk Set2Set: a recurrent attention network that
iterates a query over the node set, more expressive than a plain reduction, but again it produces a *single*
attention-pooled vector per graph, it is sequential and the most parameter-heavy option (it is the budget's
reference size), and it, too, gives one view with no notion of node degree. Both non-destructive
alternatives share the same limitation: they re-weight the nodes into *one* summary. Neither multiplies the
*number of complementary views*, and neither is size- or degree-aware. So they do not attack the actual
ceiling I am now going to name.

Go back to the aggregator theory, because it has been the through-line of this whole climb and it has more
to say. I established at the gin-sum rung that among {sum, mean, max} the sum is the *most* expressive
single reduction — injective on the multiset, keeping the counts a mean discards and the multiplicities a
max discards. But "most expressive *single* reduction" hides the real ceiling, and the ceiling is the thing
to attack now. A single permutation-invariant aggregator, *whatever* it is, is a lossy summary of a
multiset: there is no one symmetric scalar-per-feature reduction that retains all the information in a
node-feature multiset of unbounded size, because you cannot in general invert one number back to a set.
Different aggregators lose *different* information — a mean keeps the distribution and drops absolute counts;
a max keeps the support and drops counts and proportions; a sum keeps the count-weighted total but blurs
whether a large value came from many small contributions or a few big ones; a standard deviation keeps the
*spread* that all three of those throw away entirely. The decisive observation is that these losses are
*complementary*: the things a mean discards are partly recovered by a max, and the spread neither of them
keeps is held by a std. So a readout that is *one* aggregator — gin-sum's sum, or even SAGPool's sum+mean —
is leaving recoverable information on the table by construction. If I cannot drop nodes and I want more than
a sum, the move is to read the full node set through *several complementary aggregators at once* and let the
classifier combine them. This is precisely the diagnosis SAGPool got half-right — it did pair a mean
alongside its sum — and then ruined by welding it to a destructive selection.

But there is a second, subtler failure of single-aggregator readouts that becomes acute exactly on the
dataset I am trying to rescue, and naming it gives me the rest of the construction. Sum and mean sit at
opposite extremes of *size sensitivity*: a sum scales linearly with the number of nodes (a 110-atom NCI1
molecule produces a readout ~6× larger in magnitude than an 18-atom one), while a mean is completely
size-blind (it divides the size out). Neither is right when the *degree structure* of a node should modulate
how much it contributes. A high-degree hub sitting in a dense region and a low-degree leaf are summed with
identical weight by a plain sum, and with identical weight by a mean — yet their structural roles differ,
and how much their possibly over-smoothed embeddings should be trusted differs too (a hub's embedding has
averaged over many neighbors and may be washed out; a leaf's is sharper). What I want is a *family* of
size/degree-dependent rescalings: one that amplifies the contribution of nodes by a function of their
degree, one that attenuates it, and the identity in between, so the readout can express "weight this graph's
aggregation by its degree profile" rather than committing to the single fixed scaling that sum (linear in
count) or mean (constant) bakes in. The clean, principled form is a *logarithmic degree scaler*: define a
per-node factor S(d, α) = (log(d+1)/δ)^α, where d is the node degree, δ is the average of log(d+1) over the
graph's nodes (a normalizer so the scaler hovers around 1 and needs no per-dataset tuning), and
α ∈ {0, +1, −1} gives identity, amplification (up-weight high-degree nodes), and attenuation (up-weight
low-degree nodes).

Let me put a small graph through it to be sure the scaler does something sane rather than something I only
hope for. Take three nodes of degree 1, 2, 3. Then log(d+1) = {0.693, 1.099, 1.386}, the per-graph
normalizer δ is their mean, 1.059, and the ratios log(d+1)/δ come out {0.654, 1.037, 1.309}. Under
amplification (α = +1) the degree-3 hub is scaled by 1.31 and the degree-1 leaf by 0.65 — the hub's
contribution is roughly doubled relative to the leaf's; under attenuation (α = −1) the factors invert to
{1.53, 0.96, 0.76}, up-weighting the leaf and damping the hub. So the identity, amplification, and
attenuation channels genuinely present the *same* node set weighted three different ways by degree, which is
the size/degree sensitivity a fixed sum or mean cannot express. The relation to the sum is worth stating
precisely so I do not overclaim it: it is the *linear* scaler S(d) = d that turns a mean into a sum (mean
times degree is sum), and that linear scaling is exactly the unstable thing I rejected two rungs ago — it
compounds across layers and blows the magnitudes up. The *logarithmic* scaler I am adopting is its
bounded-magnitude replacement: log(d+1) ≠ d, so it does *not* reproduce gin-sum's plain sum; it reinjects a
controlled, monotone degree dependence — the size-aware behavior gin-sum's fixed sum (linear in count) and a
plain mean (constant) both lack — without the blow-up. So the value is not "contains the sum" but "a stable,
learnable degree knob the sum could not offer."

Now combine the two ideas — multiple aggregators, multiple degree scalers — the way they actually compose:
take the *tensor product*. For each scaler S(·, α) I rescale every node's embedding by that factor and then
apply *each* aggregator, so the readout is the stack of all (aggregator × scaler) channels,
⊕ = [1, S(·,+1), S(·,−1)]ᵀ ⊗ [μ, max, σ]. Each channel is a different lossy view of the same full,
undiscarded node set, and together they recover far more of the multiset's information than any single view.
This is the principal-neighborhood-aggregation idea — that no single aggregator is enough and the right
object is the product of a complementary aggregator set with degree scalers — *adapted* from its original
home (per-node *neighborhood* aggregation inside message passing) to the place I actually control here (the
*graph-level* readout, where the "neighborhood" is the whole graph's node set). I should be honest that this
is an adaptation, not a literal lift of the canonical operator. The original applies the degree scaler to a
node's *post-aggregation* neighborhood summary, with δ a fixed statistic of the *training set's* degree
distribution; here I scale each node's *embedding before* the graph-level pool, and I take δ as a *per-graph*
mean of log(d+1) — because the harness gives me one readout call per batch with no persistent training-degree
statistic to reference, and per-graph normalization keeps the scaler near 1 without a global buffer. Same
scaler *family*, same {aggregators × scalers} structure; the application point and the normalizer are the
harness-fitted differences, and I keep them in view rather than pretending the graph-level readout
reproduces the per-node operator exactly.

Now bring it into the scaffold, and here is where I tailor the canonical aggregator set to *this* harness
rather than copying it blind. The full method uses four aggregators — mean, max, min, std. But my nodes are
the *output of a fixed GIN backbone*, and every GIN layer ends in a ReLU, so the node embeddings in
`layer_outputs` are *non-negative*, and the jumping-knowledge representation I am about to form (a sum of
those) is non-negative too. Think about what a min aggregator does to a non-negative, ReLU-sparse feature:
per channel, the minimum over the graph's nodes is zero the moment *any* node has that channel switched off
by its ReLU, and with the sparsity ReLU induces that happens in almost every channel of almost every graph.
So the min channel is near-constant zero across graphs — it carries almost no discriminative signal here and
would only spend parameter budget on a dead input. So I drop min and keep the three informative aggregators
{mean, max, std}; three aggregators × three scalers = nine channels of width hidden_dim.

Dropping min is not only clean, it turns out to be *budget-necessary*, and the arithmetic is worth doing
because it is the difference between an edit the harness accepts and one it rejects. The readout ends in a
projection from the stacked channels down to hidden_dim so the classifier head consumes a hidden_dim-wide
vector. With the four-aggregator product that projection is `Linear(12H, H)` = 12·64·64 + 64 = 49,216
parameters, which *exceeds* the 10·H² + 9·H = 41,536 headroom — the full canonical product would not fit.
Dropping min to three aggregators makes it `Linear(9H, H)` = 9·64·64 + 64 = 36,928, which sits comfortably
under 41,536 with room to spare. So the same observation that makes min uninformative (post-ReLU
non-negativity) is what brings the operator inside budget; the two arguments point the same way. The std
aggregator I compute the numerically safe way — σ = sqrt(ReLU(E[x²] − E[x]²) + ε) — clamping the variance
non-negative before the root so floating-point cancellation in E[x²] − E[x]² cannot hand sqrt a tiny
negative and produce a NaN.

The last decision is how to fold in the layers, because keeping every layer is half the point of beating
SAGPool on NCI1 and I have to do it without blowing the budget I just barely fit under. gin-sum concatenated
per-layer sums, width H·num_layers = 320. If I applied the full nine-channel PNA product *per layer* I would
have 9 × 5 = 45 channels and a projection `Linear(45H, H)` = 45·64·64 + 64 = 184,384 parameters — more than
four times the entire headroom, hopeless. So the per-layer product is out; I need a size-neutral fold. The
one that keeps every depth without widening is to first combine the per-layer node embeddings into one
jumping-knowledge node representation by an *element-wise sum across layers* — h = Σ_k layer_outputs[k], the
width stays H, every layer is present, and no node is dropped — and then apply the nine-channel PNA readout
to that single width-H node representation. So the readout reads every node and every layer (gin-sum's two
robustness sources, both preserved) and extracts nine complementary views through the aggregator × scaler
product (the expressivity SAGPool reached for), with *zero* hard selection (SAGPool's fatal move, removed).
The output width is hidden_dim after the projection, so the fixed classifier head consumes it unchanged.

Let me check the shapes and the invariance once through so I trust the module before I run it. The layer
sum `torch.stack(layer_outputs).sum(0)` maps five `[N, H]` tensors to one `[N, H]`; the per-node degree from
`degree(edge_index[1], num_nodes=N)` is `[N]`, log(d+1) is `[N]`, the per-graph δ from a mean-pool over the
batch is `[B, 1]` gathered back to `[N]` via `delta[batch]`, and the ratio is `[N]`; amplification and
attenuation multiply/divide h by `ratio.unsqueeze(-1)` to give `[N, H]` each; every aggregator maps `[N, H]
→ [B, H]`, nine of them concatenate to `[B, 9H]`, and the projection returns `[B, H]`. Permutation
invariance holds channel by channel: degree is computed from the adjacency and is node-equivariant, so the
per-node scalers relabel with the nodes; each of mean/max/std is a symmetric reduction over the per-graph
node set; and a concatenation of invariants projected linearly is invariant. Nothing in the pipeline can see
node order, which is the property the graph-level answer needs to be well-defined.

Let me state the bar this has to clear and what I would validate, against the strongest baseline's real
numbers, because there is no leaderboard row for this readout to lean on. The headline target is the
aggregate: SAGPool's dataset means are MUTAG 90.95, PROTEINS 77.99, NCI1 70.75, for a three-dataset mean of
79.90 and a cross-dataset spread of 20.20; gin-sum's are 84.02, 74.54, 79.52, for a mean of 79.36 and a
spread of 9.48. So the two baselines are only 0.54 apart on the mean, and the real difference between them
is robustness — which is exactly the axis this readout is built for. The *robustness* claim is the NCI1
number, and it is the sharpest falsifiable test. Because this readout drops no nodes and reads every layer,
it should recover gin-sum's NCI1 regime (~79.5) rather than SAGPool's collapsed 70.75, and it should do so
*without* SAGPool's thirteen-point worst-seed swing — if NCI1 does not climb back well above 73 with a tight
seed band, the "non-destructive readout recovers distributed signal" thesis is wrong and I would conclude the
gain on MUTAG genuinely *requires* throwing nodes away. On MUTAG and PROTEINS the claim is softer: the
multi-aggregator-plus-scaler product is strictly more expressive than gin-sum's single sum, so I expect it
to clear gin-sum's 84.02 and 74.54; whether it matches SAGPool's motif-driven MUTAG peak of 90.95 is the
genuine open question — selection may simply be the better tool when the label *is* a single small motif, in
which case this readout's win is "SAGPool-on-NCI1 robustness at near-SAGPool MUTAG accuracy," a better
*average and a far better worst case*. The crisp success criterion I would hold it to: beat SAGPool's
three-dataset mean of 79.90 *and* cut the cross-dataset spread below its 20.20 — ideally toward gin-sum's
9.48 — with the NCI1 recovery as the load-bearing evidence. What I would validate first, before trusting any
mean, is the per-seed NCI1 row: if the degree scalers are doing their job, the worst NCI1 seed should never
fall the way SAGPool's 66.76 did, because nothing is ever discarded.

The causal chain in one breath: SAGPool's hard top-k won the motif sets (MUTAG 84→91, PROTEINS 74.5→78) but
*destroyed* NCI1's distributed decision (79.5→70.75, worst seed 66.76, range 8.98, macro_f1 gap widened to
1.71) by discarding half the nodes twice, so its 79.90 average beats gin-sum's 79.36 by only 0.54 while
doubling the cross-dataset spread to 20.20 → so keep every node and every layer (gin-sum's two robustness
sources) but extract more than a single sum can, since *any* one aggregator is a lossy multiset summary and
their losses are complementary — and reject the attention-pool and Set2Set alternatives because they still
give only one degree-blind view → read the full node set through several complementary aggregators
{mean, max, std} crossed with logarithmic degree scalers {identity, amplify, attenuate},
[1, S(·,+1), S(·,−1)]ᵀ ⊗ [μ, max, σ], the tensor-product operator that is size/degree-aware (a stable log
degree-knob the sum lacks, not a re-derivation of the sum) where sum and mean are not → adapt to the harness
(drop the min aggregator, uninformative on post-ReLU GIN features and, at 12H→H = 49,216 params, over budget
where 9H→H = 36,928 fits; element-wise-sum the layers into one JK node rep so all depths stay without the
45H→H blow-up) → drop it into `GraphReadout`, projecting 9H→H, expecting it to recover gin-sum's NCI1 regime
without SAGPool's worst-seed collapse while clearing gin-sum on the motif sets — a more robust readout that
gains expressivity without ever throwing a node away.
