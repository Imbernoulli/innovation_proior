Let me start from the thing that actually bothers me about message passing. Every layer I write reduces a
node's neighbors to one vector with a single aggregator — I pick a mean, or a sum, or a max — and I pick it
the way everyone does, by intuition or by what worked last time. The expressivity story I lean on says a sum
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

Let me try to settle whether a single one can do it, because the structure of any impossibility will tell me
what the fix has to look like. Strip to the simplest nontrivial case: scalar features, so a multiset of n
real numbers, and ask whether one continuous symmetric scalar-valued aggregator can distinguish them all.
First I want to know how big the domain actually is. A multiset of n reals, taken up to permutation, is a
point in ℝ^n / S_n — and a clean coordinate system on that quotient is the n *order statistics*, the sorted
values x_(1) ≤ x_(2) ≤ … ≤ x_(n). Those n numbers are free (any nondecreasing tuple is a legal multiset) and
they determine the multiset. So the domain is genuinely n-dimensional; there are n independent directions a
size-n multiset can move in. A single aggregator outputs *one* real. So I am asking for a continuous map from
an n-dimensional space onto ℝ^1 that never identifies two points. For n ≥ 2 that is exactly what invariance
of domain forbids: a continuous injection from an open subset of ℝ^n into ℝ^m needs m ≥ n. One real cannot
hold an injective image of an n-dimensional set when n ≥ 2. The same flavor of obstruction is what
Borsuk-Ulam packages topologically — a continuous map into too few dimensions is forced to glue points — but
I do not even need the antipodal machinery here; the bare dimension count is enough to kill injectivity for a
single scalar aggregator.

Now count what it would take. If I use K aggregators, the joint output lives in ℝ^K, and for the map
multiset ↦ (a_1, …, a_K) to be injective the same dimension argument says I need K ≥ n. Let me tabulate it so
I am not fooling myself: a size-2 multiset has a 2-dim domain → need K ≥ 2; size-3 → K ≥ 3; size-5 → K ≥ 5;
size-n → K ≥ n. So the requirement grows one-for-one with neighborhood size, and a single aggregator is
short by a factor of n. That is a much stronger statement than "sum is a bit lossy" — it says
single-aggregator GNNs are *structurally* incapable of telling apart neighborhoods that a downstream task may
genuinely need separated, and no amount of training fixes a representational collapse, because the collapse
is in the map's image dimension, not in the weights.

This reframes a nagging empirical fact I had filed away as noise: different tasks want different aggregators,
and a model that is maximal on countable features still confuses simple continuous neighborhoods. If a single
aggregator is one fixed *projection* of an n-dimensional multiset down to one number, then which
neighborhoods it cannot tell apart depends entirely on which projection it is — mean confuses one family,
max another. The task-dependence of "best aggregator" is then not capriciousness; it is exactly which slice
of the multiset that task happens to need and that aggregator happens to drop. I had read it as folklore; the
dimension count says it should be expected.

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
*location* statistics, and I should check whether that leaves a gap rather than just suspecting one. Can two
genuinely different neighborhoods share all three? Take A = {0, 2, 4} and B = {0, 1, 2, 3, 4}. Both have mean
2, max 4, min 0 — identical on every location statistic I have so far. But their spreads differ: std(A) =
sqrt((0+4+16)/3 − 4) = sqrt(20/3 − 4) = sqrt(8/3) ≈ 1.633, while std(B) = sqrt((0+1+4+9+16)/5 − 4) = sqrt(6 −
4) = sqrt(2) ≈ 1.414. So mean, max, and min collapse A and B, and a dispersion statistic pulls them apart by
about 0.22. That settles it: the three location statistics share a blind spot, and the cheapest
permutation-invariant thing that sees through it is the standard deviation — and std is cheap, because the
variance is just E[x²] − E[x]² over the neighbors, which is two means I am already in a position to compute.
So my covering set is {mean, max, min, std}: center, upper envelope, lower envelope, spread. That is the
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
the node's degree d. If I scale linearly in d, I recover the sum's pathology, and I should put numbers to
"pathology" before I trust the word. Take a node sitting in a region of degree 50 and stack L of these
layers; a linear scaler multiplies the running magnitude by ~50 each layer, so the factor after L layers is
50^L: L=1 → 50, L=2 → 2.5e3, L=3 → 1.25e5, L=4 → 6.25e6, L=5 → 3.1e8. Eight orders of magnitude across five
layers — that is the explosion, not a figure of speech. Now do the same with a *logarithmic* scaler
log(d+1)/δ. If I normalize δ to the average degree (say 25, so δ = log 26), then for this degree-50 node the
per-layer factor is log(51)/log(26) ≈ 1.207, and across L layers it goes 1.207, 1.456, 1.757, 2.121, 2.559 —
it still *grows*, so degree is still doing real work and hasn't been divided out, but after five layers it is
~2.6 rather than 3e8. That four-orders-of-magnitude-plus gap between 2.6 and 3e8 is the whole reason to scale
by log rather than linearly: log grows slowly enough that compounding across a handful of layers stays
bounded while still carrying a degree signal. Define the scaler

  S(d, α) = ( log(d+1) / δ )^α,

where the log(d+1) is the slow degree dependence (the +1 keeps it finite and positive at d=0), δ is a
normalizing constant — the average of log(d+1) over the training graphs — so the base of the power hovers
around 1 and the scaler needs no per-dataset retuning, and α is the knob. α = 0 gives the identity (no
degree dependence, recover the bare aggregator). α = +1 *amplifies*: up-weight high-degree nodes' aggregate.
α = −1 *attenuates*: up-weight low-degree nodes'. Three settings, {−1, 0, +1}, spanning amplify / neutral /
attenuate.

Here is the part that makes the scalers more than a normalization trick, and it is worth checking rather
than asserting. Start with the *linear* scaler, S(d) = d. A degree-amplifying linear scaler applied to a
*mean* should reproduce a *sum* — let me confirm the identity on a multiset rather than wave at it. Take
{1.5, −0.5, 2.0, 4.0}: d = 4, mean = 7/4 = 1.75, and mean·d = 1.75·4 = 7 = the sum. The algebra is just
(1/d)·Σ · d = Σ, so it holds for any multiset, not only this one. So with the linear scaler the family
*contains* the sum as a special case — and more, a degree-linear injective scaler composed with a suitably
constructed per-element map makes the mean injective on bounded countable multisets (this is the spirit of
the scaler-injectivity result: it is the constructed element map plus the cardinality scaler that does it,
not a raw mean times any scaler). But the linear scaler is exactly the one whose 50^L explosion I just
watched run away across depth. So I do *not*
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

Let me push the hardest case through the actual operator, because an operator that is merely "more channels"
is not obviously better — it has to *use* the structure on a neighborhood that defeats the single
aggregators. The case that most worries me is count: a multiset and its k-fold inflation, which is invisible
to every *location and spread* statistic at once, so all four of my aggregators collapse it and only the
degree scaler can save it. Take C = {1, 3} (degree 2) and its 2-fold inflation D = {1, 1, 3, 3} (degree 4).
Run the four aggregators on both: mean(C)=mean(D)=2, max=3, min=1, and var = E[x²]−E[x]² = (1+9)/2 − 4 = 1 for
C and (1+1+9+9)/4 − 4 = 1 for D, so std=1 for both. Every degree-blind channel returns *the same vector* on C
and D — exactly the collapse I feared. Now bring in the scaler. With δ = mean of log(d+1) over these two
nodes = (log 3 + log 5)/2 ≈ 1.354, the amplification ratio is log(d+1)/δ: for C, log(3)/1.354 ≈ 0.811; for D,
log(5)/1.354 ≈ 1.189. So the amplification channel S(d,+1)·mean gives 2·0.811 = 1.622 on C and 2·1.189 =
2.377 on D — different by 0.76, and the attenuation channel S(d,−1)·mean gives 2/0.811 = 2.465 on C and
2/1.189 = 1.683 on D, different the other way. The degree-scaled channels separate C from D cleanly even
though all four bare aggregators could not. That is the mechanism doing precisely the job it was built for:
reinjecting the count information that every location-and-spread statistic discards. (The spread case from
earlier is the easier one and I already saw it work: shared mean/max/min, σ separates by ~0.22. And differing
upper envelope is caught by max where mean and min agree.) So the channels do pull in different directions
where I designed them to, not just in principle. And the degree stability falls out of the same numbers as
the 50^L check: because the scalers are logarithmic, stacking these layers grows magnitudes like ~2.6 over
five layers rather than ~3e8, so the operator should *extrapolate* to graphs with larger degrees than
training without the activations running away — which is the stringent test I most want to pass, since it is
the one that exposes degree-unstable aggregation.

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
injective on continuous multisets and, by counting the dimension of the domain — a size-n multiset is its n
order statistics, an n-dimensional space, mapped to one real — found it cannot: invariance of domain forces
at least n aggregators for size-n multisets, so single-aggregator GNNs are structurally lossy and which
neighborhoods they confuse depends on the aggregator chosen (which retro-explains why different tasks want
different aggregators). Unable to instantiate n aggregators, I chose a small set with *complementary* losses
— mean (distribution), max (upper support), min (lower support), std (spread) — covering center, both
envelopes, and dispersion, after checking on {0,2,4} vs {0,1,2,3,4} that mean/max/min alone collapse a pair
std splits, with a clamped-variance std for numerical safety. Then I attacked the orthogonal defect, degree
behavior: sum is degree-coupled and explodes across depth (I watched 50^L hit 3e8 by five layers), mean is
degree-blind, so I built a *logarithmic* degree scaler S(d,α) = (log(d+1)/δ)^α with α ∈ {−1,0,+1} that makes
degree-dependence a learnable knob and is stable across depth (~2.6 not 3e8 over five layers) — the bounded
replacement for the *linear* scaler that would turn a mean into a sum (mean·d = Σ, verified) but blow up
across layers — so it reinjects count-sensitivity safely without reproducing the sum, as the C vs D
inflation check confirmed the four bare aggregators cannot. The full operator is the tensor product of
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
