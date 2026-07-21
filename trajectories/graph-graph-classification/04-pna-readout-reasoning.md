SAGPool's numbers split *exactly* along the line I drew, which confirms the failure precisely enough to
name the cure. On the motif-driven sets selection paid off and then some — MUTAG jumped from gin-sum's
84.02 to 90.95, a lift of 6.93, and PROTEINS from 74.54 to 77.99, up 3.45 — undoing the dilution I
diagnosed in gin-sum's uniform sum, most where the motif is smallest, MUTAG. But on NCI1, where I
predicted hard selection would destroy a distributed decision, it collapsed: 70.75 mean, down 8.77 from
79.52, with a brutal seed spread {66.76, 75.74, 69.76} whose worst seed sits nearly thirteen points below
gin-sum's mean and whose range of 8.98 is almost four times gin-sum's tight 2.36. That is the irreversible
top-k discard doing exactly what I feared: NCI1's compound-activity label depends on the whole molecular
context, the single-conv score is mis-calibrated on the distributed cases, and keeping only a quarter of
the atoms throws away the very ones that carried the signal.

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

The ranking arithmetic is more honest than any single dataset. SAGPool's three-dataset mean is 79.90;
gin-sum's is 79.36. So the strongest readout beats the robust one by a mere 0.54 points — and buys that
half-point with a nine-point hole in NCI1. Its cross-dataset spread (best minus worst) is 20.20 against
gin-sum's 9.48: more than twice the spread for barely any gain in the mean, strong on average only because
a six-point MUTAG windfall masks a nine-point NCI1 catastrophe. It is *less robust* than gin-sum, not
uniformly better. The single fact that matters now: gin-sum won NCI1 because it threw *nothing* away;
SAGPool lost NCI1 because it threw *half* away, twice.

So I want the next readout to do what SAGPool reached for — be more expressive than a single uniform sum,
recover the per-graph-structure sensitivity that lifted MUTAG — while never committing the destructive act
that sank NCI1. Concretely: *keep every node* (non-destructive, like gin-sum) and *keep every layer* (the
jumping-knowledge robustness that won NCI1 in the first place), but extract *more* from that full,
undiscarded set than a lone sum can. The question becomes sharp: given that I refuse to drop nodes, how do
I make a flat readout more discriminating than Σ_v h_v?

The two non-destructive alternatives the framing named — an attention-weighted pool and a Set2Set
recurrent set network, both affordable within the budget — I can dismiss quickly: each re-weights the
nodes into a *single* summary (and the attention pool re-introduces the learned softmax I watched go
diffuse in DiffPool), giving one lossy, degree-blind view. Neither multiplies the *number of complementary
views*, which is the ceiling I now name.

Go back to the aggregator theory, the through-line of this whole climb. Among {sum, mean, max} the sum is
the most expressive single reduction — injective, keeping the counts a mean discards and the multiplicities
a max discards. But "most expressive *single* reduction" hides the real ceiling. A single
permutation-invariant aggregator, *whatever* it is, is a lossy summary of a
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

Three nodes of degree 1, 2, 3 make it concrete: log(d+1) = {0.693, 1.099, 1.386}, δ their mean 1.059, so
the ratios are {0.654, 1.037, 1.309}. Amplification (α = +1) scales the degree-3 hub by 1.31 and the
degree-1 leaf by 0.65, roughly doubling the hub relative to the leaf; attenuation (α = −1) inverts to
{1.53, 0.96, 0.76}, up-weighting the leaf. So the three channels present the *same* node set weighted
three ways by degree. One thing to state precisely so I do not overclaim: it is the *linear* scaler
S(d) = d that turns a mean into a sum, and that linear scaling is exactly the unstable thing I rejected
two readouts ago — it compounds across layers and blows magnitudes up. The *logarithmic* scaler is its
bounded-magnitude replacement: log(d+1) ≠ d, so it does *not* reproduce a plain sum; it reinjects a
controlled, monotone degree dependence without the blow-up. The value is not "contains the sum" but "a
stable, learnable degree knob the sum could not offer."

Now combine the two ideas — multiple aggregators, multiple degree scalers — the way they actually compose:
take the *tensor product*. For each scaler S(·, α) I rescale every node's embedding by that factor and then
apply *each* aggregator, so the readout is the stack of all (aggregator × scaler) channels,
⊕ = [1, S(·,+1), S(·,−1)]ᵀ ⊗ [μ, max, σ]. Each channel is a different lossy view of the same full,
undiscarded node set, and together they recover far more of the multiset's information than any single view.
This is the principal-neighborhood-aggregation idea — that no single aggregator is enough and the right
object is the product of a complementary aggregator set with degree scalers — *adapted* from its original
home (per-node *neighborhood* aggregation inside message passing) to the *graph-level* readout, where the
"neighborhood" is the whole graph's node set. It is an adaptation, not a literal lift: the original scales
a node's *post-aggregation* neighborhood summary with δ a fixed statistic of the *training set's* degrees;
here I scale each node's *embedding before* the graph-level pool and take δ as a *per-graph* mean of
log(d+1), because I get one readout call per batch with no persistent training-degree statistic and
per-graph normalization keeps the scaler near 1 without a global buffer. Same scaler family, same
{aggregators × scalers} structure; application point and normalizer are the differences I keep in view.

Now I tailor the canonical aggregator set to this harness. The full method uses four aggregators — mean,
max, min, std. But my nodes are the output of a fixed GIN backbone, every layer ends in a ReLU, so the
embeddings in `layer_outputs` (and their jumping-knowledge sum) are *non-negative*. A min aggregator over
a non-negative, ReLU-sparse feature is zero the moment *any* node has that channel switched off, which the
sparsity makes happen in almost every channel of almost every graph — so the min channel is near-constant
zero, discriminatively dead, spending budget on nothing. Drop it, keep {mean, max, std}: three aggregators
× three scalers = nine channels of width hidden_dim.

Dropping min is also *budget-necessary*. The readout ends in a projection to hidden_dim; with four
aggregators that is `Linear(12H, H)` = 49,216 params, which *exceeds* the 10·H² + 9·H = 41,536 headroom.
Three aggregators makes it `Linear(9H, H)` = 36,928, comfortably under. So the same post-ReLU observation
that makes min uninformative is what brings the operator inside budget. The std aggregator I compute the
numerically safe way — σ = sqrt(ReLU(E[x²] − E[x]²) + ε) — clamping the variance
non-negative before the root so floating-point cancellation in E[x²] − E[x]² cannot hand sqrt a tiny
negative and produce a NaN.

The last decision is how to fold in the layers, since keeping every layer is half the point of beating
SAGPool on NCI1. Applying the full nine-channel product *per layer* would give 9 × 5 = 45 channels and a
projection `Linear(45H, H)` = 184,384 params, four times the headroom — hopeless. So I need a size-neutral
fold: combine the per-layer embeddings into one jumping-knowledge node representation by an *element-wise
sum across layers*, h = Σ_k layer_outputs[k] — width stays H, every layer present, no node dropped — then
apply the nine-channel readout to that. So the readout reads every node and every layer (gin-sum's two
robustness sources) and extracts nine complementary views through the aggregator × scaler product (the
expressivity SAGPool reached for), with *zero* hard selection. Output width is hidden_dim after projection.

Permutation invariance holds: degree is computed from the adjacency and is node-equivariant so the
per-node scalers relabel with the nodes, each of mean/max/std is a symmetric per-graph reduction, and a
concatenation of invariants projected linearly is invariant — nothing sees node order.

The bar, against the two baselines' real numbers since there is no leaderboard row to lean on: SAGPool's
79.90 three-dataset mean beats gin-sum's 79.36 by only 0.54, and the real difference is robustness — the
axis this readout is built for. The load-bearing test is NCI1: dropping no nodes and reading every layer,
it should recover gin-sum's ~79.5 regime rather than SAGPool's collapsed 70.75, and *without* SAGPool's
thirteen-point worst-seed swing. If NCI1 does not climb well above 73 with a tight band, the
"non-destructive readout recovers distributed signal" thesis is wrong and the MUTAG gain genuinely
requires throwing nodes away. On MUTAG and PROTEINS the claim is softer — the product is strictly more
expressive than a single sum, so it should clear gin-sum's 84.02 and 74.54, but whether it matches
SAGPool's motif-driven 90.95 is open, since selection may simply be the better tool when the label *is* a
single small motif. Success = beat SAGPool's 79.90 mean *and* cut the cross-dataset spread below its
20.20, with the NCI1 recovery the evidence; I would validate the per-seed NCI1 row first, where nothing
discarded should mean the worst seed never falls the way SAGPool's 66.76 did.
