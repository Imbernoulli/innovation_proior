## Research question

To train a model with differential privacy, we have to bound how much any one
contributor can move the model, and the standard lever is a single scalar: the
clipping norm `C`. Each contributor's gradient (or, in the federated setting, each
user's model update) is projected into the `L2` ball of radius `C`, the clipped
contributions are summed, and Gaussian noise whose standard deviation is
*proportional to* `C` is added before the optimizer step. That proportionality is
forced: `C` is exactly the `L2` sensitivity of the sum to one contributor, and the
Gaussian mechanism must add noise scaled to the sensitivity to hide that
contributor. So `C` sits at the center of a bias-variance tradeoff. Set it low and
many contributions are scaled down — their directions are preserved, but their
relative magnitudes are flattened. Set it high and the noise standard deviation `zC`
grows with it.

The distribution of contribution norms depends on the model architecture, the loss,
how much data each contributor holds, the (client) learning rate, and other knobs,
and it shifts over the course of training. The question is how to set the clipping
norm `C` automatically and online — from a stream of contributions whose norm
distribution moves over training — using only information the server is permitted to
see under DP, while spending little extra privacy budget, and without requiring
contributors to reveal anything beyond the minimal, already-clipped contribution
they were going to send (so that the policy remains compatible with secure
aggregation and update compression).

## Background

The field state rests on a few load-bearing pieces.

**Differential privacy and the Gaussian mechanism.** An algorithm `M` is
`(ε,δ)`-differentially private if for all neighboring datasets `D, D'` (differing by
one contributor) and all output sets `S`,
`Pr[M(D)∈S] ≤ e^ε·Pr[M(D')∈S] + δ` (Dwork et al. 2006). For a vector-valued query
`f` with `L2`-sensitivity `Δ₂f = max_{D~D'} ‖f(D)−f(D')‖₂`, the Gaussian mechanism
`f(D) + N(0, σ²I)` is `(ε,δ)`-DP for `σ ≥ sqrt(2 ln(1.25/δ))·Δ₂f/ε`; the ratio
`z = σ/Δ₂f` is the *noise multiplier* and is the knob that trades privacy for
utility (Dwork & Roth; Mironov 2017). Rényi DP (Mironov 2017) and the subsampled-RDP
accountant (Wang et al. 2019) give tight composition across many rounds, which is
what makes thousands of training rounds affordable at a fixed `(ε,δ)`.

**User-level vs example-level neighbors.** Two datasets are example-level neighbors
if they differ by one training example, and user-level neighbors if they differ by
*all* the data of one user (McMahan et al. 2018). User-level is the stronger guarantee
and the natural one when a single user contributes many correlated examples;
example-level is recovered as the special case of one example per user.

**The clipping bias-variance phenomenon.** The clipping norm trades bias against
variance — aggressive clipping discards magnitude information, loose clipping pays in
noise — and this is both empirically observed and theoretically analyzed as an
inherent property of DP learning (McMahan et al. 2018; Amin et al. 2019). It was also
observed that the norms of the updates *change as rounds progress*, and that manually
decreasing the clipping norm after some initial number of rounds can improve a
language model's accuracy (McMahan et al. 2018). It was separately suggested, as a
heuristic, that one might clip to the *median* of the update-norm distribution (Abadi
et al. 2016).

**Online convex optimization.** A convex cost is revealed one round at a time; the
learner commits to a point before seeing the cost and is measured by regret against
the best fixed point. Online gradient descent on a sequence of convex, bounded-
gradient losses attains sublinear regret, and for a convex-but-not-strongly-convex
loss a step size proportional to `1/√t` yields an `O(√T)` regret bound (Zinkevich
2003; Shalev-Shwartz 2012). If a quantity is the minimizer of a convex online loss
whose gradients can be observed, online gradient descent tracks it.

**Quantile regression.** For a scalar random variable `X` and a target level
`γ∈[0,1]`, the asymmetric "pinball" loss of quantile regression — piecewise linear,
weighting over- and under-shoot by `γ` and `1−γ` — is convex and is minimized exactly
at the `γ`-quantile of `X` (Koenker & Bassett 1978). Its derivative is bounded, so it
is a `1`-Lipschitz convex loss.

**Composing several Gaussian queries per round.** A round of private training often
issues more than one Gaussian sum query against the same sampled batch (for instance,
the gradient sum and some scalar summary). A general technique composes `G` such
queries — each with its own clip `S_g` and noise `σ̃_g` — into a *single* Gaussian sum
query for accounting purposes: rescale group `g` by `S_g/σ̃_g`, after which the
concatenation has `L2`-sensitivity `S* = sqrt(Σ_g (S_g/σ̃_g)²)` against a single unit-
variance Gaussian, so the combined mechanism is exactly a single Gaussian sum query
with noise multiplier `z = 1/S* = (Σ_g (S_g/σ̃_g)²)^{-1/2}`, to which the standard
accountant applies (McMahan et al. 2018, "A General Approach to Adding DP to
Iterative Training Procedures").

## Baselines

**DP-SGD (Abadi et al. 2016).** Clip each per-example gradient to `L2` norm `C` via
`g ← g·min(1, C/‖g‖₂)`, sum over a lot, add `N(0, σ²C²I)`, and take a descent step on
the average; calibrate `σ` to the target `(ε,δ)` with the moments accountant, which
tightly tracks composition of the subsampled Gaussian mechanism. `C` is a fixed
hyperparameter set up front and held constant across all of training; the original
DP-SGD work noted in passing that clipping to the median of the gradient-norm
distribution might be better.

**DP-FedAvg / DP-FedAvg-M (McMahan et al. 2018).** Federated Averaging with user-level
DP. On each round a set of `m` users is sampled; each runs local SGD and returns its
model delta `Δ_i`, which is FlatClipped, `π(Δ, C) = Δ·min(1, C/‖Δ‖)`; the server
forms the noised average `(1/m)(Σ_i Δ_i + N(0, σ²I))` with `σ = zC`, optionally
applies server momentum, and steps. The moments accountant bounds total privacy. The
clipping norm is a constant `C` treated as a hyper-parameter and tuned; the same work
reports that manually lowering it partway through training can help.

**Per-round private-median via smooth sensitivity.** One could, each round, estimate
the median *unclipped* update norm directly and clip to it, adding noise calibrated to
the smooth sensitivity of the median (Nissim et al. 2007). This requires each user to
transmit its *unclipped* update (or an extra communication round in which the server
first collects norms), and estimates the median independently each round from that
round's sample.

**Coordinate-wise adaptive clipping (Pichapati et al. 2019).** Adapts a per-coordinate
clip for DP-SGD, introducing its own hyperparameter.

## Evaluation settings

The natural yardsticks are realistic federated learning benchmarks with non-i.i.d.,
user-partitioned data (Reddi et al. 2020): image classification on CIFAR-100 with a
ResNet; character recognition and an autoencoder on EMNIST partitioned by writer; a
character LSTM on Shakespeare partitioned by speaking role; and next-word prediction
and tag prediction on Stack Overflow partitioned by user (with a very large user
population, ~342k, that makes good user-level privacy attainable). The protocol
samples a fixed number of clients per round (e.g. `m = 100`), uses unweighted
federated averaging, and reports the task's evaluation metric (accuracy / recall /
reconstruction loss) on a validation/test split, scanning a small grid of noise
multipliers `z` and of clip settings, with client and server learning rates taken
from the optimized non-private configuration and lightly re-tuned. Privacy is reported
as the `(ε,δ)` attained under RDP composition with subsampling for a stated population
size. The same privacy definition also has an example-level special case: one can
view each sampled "user" as contributing a single example.

## Code framework

The mechanism plugs into an existing private-training harness. The data pipeline, the
model, the loss, the optimizer/training loop, and the per-contributor gradient (or
update) computation already exist and are fixed. What already exists on the privacy
side is the Gaussian sum machinery: a routine that clips a contribution to a given
`L2` norm and a routine that adds Gaussian noise scaled to that norm, plus the
accountant that converts a noise multiplier into an `(ε,δ)` over the run. The policy
that decides the clipping norm `C` is the open slot: at present `C` is a constant
passed in.

```python
import collections


class GaussianSumQuery:
    """Existing primitive: clip each record into the L2 ball of radius
    l2_norm_clip, sum, and add Gaussian noise of the given stddev to the sum."""

    def __init__(self, l2_norm_clip, stddev):
        self.l2_norm_clip = l2_norm_clip
        self.stddev = stddev

    def preprocess_record(self, params, record):
        # Existing implementation projects record into the L2 ball set by params.
        pass

    def get_noised_result(self, sample_state, global_state):
        # Existing implementation adds N(0, stddev^2 I) to the clipped sum.
        pass

    def make_global_state(self, l2_norm_clip, stddev):
        return collections.namedtuple("State", ["l2_norm_clip", "stddev"])(
            l2_norm_clip, stddev)


class ClippedSumMechanism:
    """Existing query-shaped shell for private clipped summation. The Gaussian
    sum already exists; the clipping norm C is currently a constant passed in,
    and the noise stddev is held at noise_multiplier * C so the accountant can
    certify the run. How C should change over time is the open slot."""

    State = collections.namedtuple("State", ["noise_multiplier", "sum_state"])

    def __init__(self, initial_clip, noise_multiplier):
        self.noise_multiplier = noise_multiplier
        self.sum_query = GaussianSumQuery(initial_clip,
                                          noise_multiplier * initial_clip)

    def get_noised_result(self, sample_state, state):
        noised_sum = self.sum_query.get_noised_result(sample_state, state.sum_state)

        # TODO: the clipping-norm policy we will design. Decide the clip C for
        #       the next step using only quantities the server may observe under
        #       DP, account for any privacy it spends, and rebuild the sum state.
        next_clip = state.sum_state.l2_norm_clip            # placeholder: held fixed
        next_sum_state = self.sum_query.make_global_state(
            next_clip, state.noise_multiplier * next_clip)
        next_state = self.State(state.noise_multiplier, next_sum_state)
        return noised_sum, next_state
```

The clipping-and-noising of the contributions is the Gaussian-sum part that already
exists; the `TODO` is the one open slot — the rule that sets the next `C`.
