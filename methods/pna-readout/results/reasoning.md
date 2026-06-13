Let me start from the thing that actually bothers me about message passing. Every layer I write reduces a
node's neighbors to one vector with a single aggregator — I pick a mean, or a sum, or a max — and I pick it
the way everyone does, by intuition or by what worked last time. The expressivity story I trust says a sum
with a learned per-element map is injective on the neighbor multiset and makes the network as strong as the
Weisfeiler-Lehman test, the ceiling for this whole family. So I should just use sum and stop worrying. But
that argument has a foundation I have been quietly ignoring: it is built on the node features coming from a
*countable* universe — discrete labels — and the proof is essentially a place-value code that encodes
integer multiplicities in a base large enough that there are no carries. The vectors my aggregator
actually sees three layers deep are not discrete labels. They are continuous hidden activations in ℝ^d.
Does the "sum is injective, so we're maximal" story survive the move from a countable alphabet to ℝ? I do
not think I have ever checked, and until I do, "use sum" is a guess wearing a theorem's clothes.

So let me make the question precise and adversarial. Forget which aggregator is fashionable; ask the rawest
thing. A single continuous aggregator is a continuous, permutation-invariant map from a multiset of n
real vectors to one real vector. Can *one* such map be injective on all multisets of size n drawn from a
continuous space — i.e., never collapse two genuinely different neighborhoods? If the answer is no, then
"pick the best single aggregator" is the wrong frame entirely; the right question becomes "how many
aggregators, and which, do I need."

Let me try to prove a single one cannot do it, because the structure of the impossibility will tell me what
the fix has to look like. Strip to the simplest nontrivial case: scalar features, so a multiset of n real
numbers, and ask whether one continuous symmetric scalar-valued aggregator can distinguish them all. A
symmetric function of n reals is a continuous function on the quotient of ℝ^n by the symmetric group — and
the obstruction I want is exactly the kind a topological argument catches. Borsuk-Ulam says any continuous
map from the n-sphere into ℝ^n must identify some antipodal pair. The shape generalizes: a continuous map
into a space of *too low dimension* relative to the domain is forced to glue points together — it cannot be
injective. A single aggregator's output lives in too few dimensions to separate all n-element multisets of
reals; there are simply more independent ways an n-element continuous multiset can differ than one
continuous number can record. Pushing the counting through, the conclusion is sharp: to discriminate all
multisets of size n whose elements range over ℝ, I need *at least n* aggregators. One is provably not
enough, and the deficit grows with the neighborhood size. That is a much stronger statement than "sum is a
bit lossy" — it says single-aggregator GNNs are *structurally* incapable of telling apart neighborhoods
that a downstream task may genuinely need separated, and no amount of training fixes a representational
collapse.

This also explains a nagging empirical fact I had filed away as noise: different tasks want different
aggregators, and a model that is provably maximal on countable features still confuses simple continuous
neighborhoods. It was never noise. It is the n-aggregator lower bound showing through. Each single
aggregator is a different *projection* of the multiset, and each loses a different part of it.

Now n aggregators is a lower bound for *exact* injectivity at neighborhood size n, and n is unbounded in a
real graph — I cannot literally instantiate one aggregator per possible degree. So I will not chase exact
injectivity. I will instead ask the engineering version: a small, fixed set of aggregators whose
*information losses are complementary*, so that together they retain far more of the multiset than any one
of them, and let the network's update function recombine them. The lower bound tells me the direction
(more than one, and they must be genuinely different views); pragmatism tells me to pick a handful that
cover the obvious failure modes.

So let me enumerate what each candidate keeps and discards, and choose a covering set. Mean keeps the
*distribution* of neighbor features — the proportions — and throws away absolute count: k copies of a
neighborhood and one copy give the same mean, because (1/(kn))·k·Σ = (1/n)·Σ. Max keeps the *support* —
which feature values are present, coordinatewise — and throws away both counts and proportions; two
neighborhoods with the same coordinatewise maxima are indistinguishable to it. Min is the mirror image,
the other extreme of the order statistics, and on signed continuous features it carries information the max
does not (the smallest value is not recoverable from the largest). Those three — mean, max, min — already
disagree about what matters: the center of mass, the upper envelope, the lower envelope. But all three are
*location* statistics; none of them sees how *spread out* the neighborhood is. Two neighborhoods can share
a mean, a max, and a min and still differ in dispersion. The natural permutation-invariant statistic that
captures spread is the standard deviation, and it is cheap: I already compute the mean, and the variance is
just E[x²] − E[x]² over the neighbors, which is two means. So my covering set is {mean, max, min, std}:
center, upper envelope, lower envelope, spread. Four complementary projections of the multiset. That is the
aggregator half of the operator.

One numerical caution on the std before I forget it: E[x²] − E[x]² is non-negative in exact arithmetic but
floating-point cancellation can drive it slightly negative for a near-constant neighborhood, and then the
square root is a NaN that poisons the whole batch. So I clamp the variance non-negative before the root —
σ = sqrt(ReLU(E[x²] − E[x]²) + ε) — with a small ε so the gradient is finite at zero spread. Cheap
insurance, and it is the kind of thing that silently breaks training if I skip it.

Now the second axis, and it comes from a different defect — not *which* statistic, but how the aggregation
behaves as a function of *degree*. Sum couples its magnitude to degree: a node with 50 neighbors produces a
sum ~50× a node with one, and stacking sum layers makes signal magnitudes grow roughly like a product of
degrees down the depth, which in dense regions explodes and destabilizes training. That is the known reason
people retreat to mean. But mean overcorrects: it divides degree out *entirely*, so it cannot *use* degree
even when degree is structurally informative — and degree usually is, a hub and a leaf play different roles.
What I want is a *controllable* dependence on degree: a knob that lets aggregation be amplified by degree,
left alone, or attenuated by degree, so the network can learn how much degree should matter rather than
having "fully" (sum) or "not at all" (mean) baked in.

So I want a scaler — a positive factor that multiplies the aggregated result — that is a tunable function of
the node's degree d. If I scale linearly in d, I recover the sum's pathology: linear scaling compounds
multiplicatively across layers and blows up exponentially with depth. The fix is to scale *logarithmically*
in degree, because log grows slowly enough that compounding across a handful of layers stays bounded. Define
the scaler

  S(d, α) = ( log(d+1) / δ )^α,

where the log(d+1) is the slow degree dependence (the +1 keeps it finite and positive at d=0), δ is a
normalizing constant — the average of log(d+1) over the training graphs — so the base of the power hovers
around 1 and the scaler needs no per-dataset retuning, and α is the knob. α = 0 gives the identity (no
degree dependence, recover the bare aggregator). α = +1 *amplifies*: up-weight high-degree nodes' aggregate.
α = −1 *attenuates*: up-weight low-degree nodes'. Three settings, {−1, 0, +1}, spanning amplify / neutral /
attenuate.

Here is the part that makes the scalers more than a normalization trick, and it is worth checking rather
than asserting. Start with the *linear* scaler, S(d) = d. A degree-amplifying linear scaler applied to a
*mean* reproduces a *sum*: mean times degree is sum (the sum is mean scaled by d). So with the linear
scaler the family *contains* the sum as a special case — and more, a degree-linear injective scaler
composed with a suitably constructed per-element map makes the mean injective on bounded countable
multisets (this is the spirit of the scaler-injectivity result: it is the constructed element map plus the
cardinality scaler that does it, not a raw mean times any scaler). But the linear scaler is exactly what I
just ruled out: scaling by d itself compounds multiplicatively across layers and blows up. So I do *not*
adopt the linear scaler; the *logarithmic* one, S(d,+1) = log(d+1)/δ, is its bounded-magnitude
*replacement*. It does not reproduce the sum — log(d+1) is not d — but it reinjects a controlled, monotone
dependence on degree that recovers the count-sensitivity a mean throws away, without the unbounded growth a
sum (or a linear scaler) risks across depth. That is the whole point of choosing log over linear: keep
degree as a usable, learnable signal while staying stable as the network deepens.

Now combine the two axes. I have a set of aggregators that disagree about *which statistic*, and a set of
scalers that disagree about *how degree modulates*. The honest way to combine "several of these and several
of those" is the tensor (outer) product: apply every scaler to every aggregator, and concatenate all the
resulting channels. With four aggregators and three scalers that is the operator

  ⊕ = [ I, S(d, +1), S(d, −1) ]ᵀ ⊗ [ μ, σ, max, min ],

twelve channels, each a different (statistic × degree-modulation) view of the same neighbor multiset. Each
channel is individually lossy — that was the whole point of the lower bound, no single one can be otherwise
— but the twelve together retain far more of the multiset than any one, and they are cheap (they reuse the
same neighbor messages; only the scaling and the reduction differ). I concatenate the twelve channels (plus
the node's own current state) and hand them to the per-node update MLP U, which learns whatever task-specific
recombination it needs. The update is where the channels get mixed back down to the layer width; the
aggregator's job was only to *not destroy* the information on the way in.

Let me sanity-check this does the right thing at the extremes, because an operator that is merely "more
channels" is not obviously better — it has to *use* the structure. Two neighborhoods that share a mean but
differ in spread: the μ channels collapse them, the σ channels separate them — caught. Two that share mean,
max, min, std but differ in *count* (a multiset and its k-fold inflation): the identity-scaler aggregators
(all degree-blind statistics) can collapse them, but the amplification channel S(d,+1)·μ scales by degree
and so distinguishes the inflated copy — caught, and caught precisely by the channel built to reinject
count. Two that differ only in their upper envelope: max catches them where mean and min do not. The
channels are pulling in different directions exactly where I designed them to. And the degree stability:
because the scalers are logarithmic, stacking these layers does not compound magnitudes the way stacked sums
do, so the operator should *extrapolate* to graphs with larger degrees than training without the activations
running away — which is the stringent test I most want to pass, since it is the one that exposes
degree-unstable aggregation.

A cost note I should be honest about: twelve channels is 12× the width into the update relative to a single
aggregator, so the projection back down (U's first linear layer) is the dominant new parameter cost. That
is a real budget item; if I ever need to trim, the place to cut is an aggregator whose channel is
near-constant on a given feature distribution (for example, on strictly non-negative features the min
channel carries little, since the elementwise min of non-negatives clusters near zero), and dropping such a
dead channel costs almost no information while saving a quarter of the projection. But the canonical operator
is the full twelve, and I keep all four aggregators and three scalers unless a specific feature distribution
makes one provably dead.

Let me write the causal chain back to myself. I distrusted "sum is maximal" because its proof assumes
countable features and I aggregate continuous ones. I asked whether *one* continuous aggregator can be
injective on continuous multisets and, via a Borsuk-Ulam-shaped dimension-counting argument, found it
cannot — you need at least n aggregators for size-n multisets, so single-aggregator GNNs are structurally
lossy and which neighborhoods they confuse depends on the aggregator chosen (which retro-explains why
different tasks want different aggregators). Unable to instantiate n aggregators, I chose a small set with
*complementary* losses — mean (distribution), max (upper support), min (lower support), std (spread) —
covering center, both envelopes, and dispersion, with a clamped-variance std for numerical safety. Then I
attacked the orthogonal defect, degree behavior: sum is degree-coupled and explodes across depth, mean is
degree-blind, so I built a *logarithmic* degree scaler S(d,α) = (log(d+1)/δ)^α with α ∈ {−1,0,+1} that
makes degree-dependence a learnable knob and is stable across depth (log, not linear) — the bounded
replacement for the *linear* scaler that would turn a mean into a sum but blow up across layers — so it
reinjects count-sensitivity safely without reproducing the sum. The full operator is the tensor product of
the three scalers with the four aggregators — twelve complementary channels — concatenated and fed to the
update MLP. Now the code: I scatter the messages to their destination nodes once, compute the four
aggregators by scatter-reduce, build the per-node degree scalers, take the outer product, and concatenate.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import degree, scatter


class PNAAggregation(nn.Module):
    """Principal Neighbourhood Aggregation: 4 aggregators x 3 degree-scalers.

    The tensor product [I, S(d,+1), S(d,-1)]^T (x) [mean, max, min, std]
    gives 12 complementary views of the neighbor multiset, concatenated.
    """

    def __init__(self, in_dim, avg_deg_log):
        super().__init__()
        # delta = average of log(d+1) over the training set (a fixed normalizer
        # so the degree scalers hover around 1 and need no per-dataset tuning).
        self.register_buffer("delta", torch.tensor(float(avg_deg_log)))
        self.eps = 1e-5
        self.out_dim = in_dim * 12  # 4 aggregators x 3 scalers

    def _aggregate(self, messages, index, num_nodes):
        # Four complementary aggregators over the neighbor multiset per node.
        mean = scatter(messages, index, dim=0, dim_size=num_nodes, reduce="mean")
        mx = scatter(messages, index, dim=0, dim_size=num_nodes, reduce="max")
        mn = scatter(messages, index, dim=0, dim_size=num_nodes, reduce="min")
        mean_sq = scatter(messages * messages, index, dim=0,
                          dim_size=num_nodes, reduce="mean")
        std = torch.sqrt(F.relu(mean_sq - mean * mean) + self.eps)
        return torch.cat([mean, mx, mn, std], dim=-1)        # [N, 4F]

    def forward(self, messages, index, node_degree, num_nodes):
        agg = self._aggregate(messages, index, num_nodes)    # [N, 4F]

        # Logarithmic degree scalers S(d, alpha) = (log(d+1)/delta)^alpha.
        log_deg = torch.log(node_degree + 1.0)               # [N]
        ratio = (log_deg / self.delta.clamp(min=self.eps)).clamp(min=self.eps)
        amp = ratio.unsqueeze(-1)                            # S(d, +1)
        att = ratio.pow(-1.0).unsqueeze(-1)                  # S(d, -1)

        # Tensor product: each scaler times the whole aggregator block.
        return torch.cat([agg, agg * amp, agg * att], dim=-1)  # [N, 12F]
```

The twelve channels are then concatenated with the node's own state and pushed through the update MLP, which
mixes them back to the layer width. The aggregator never had to be injective by itself — the lower bound
says it cannot be — it only had to lose *complementary* things across its channels and stay degree-stable,
so the update has enough surviving information to reconstruct what the task needs.
