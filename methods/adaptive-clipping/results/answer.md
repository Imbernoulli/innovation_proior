# Adaptive (Quantile) Clipping for DP learning, distilled

Adaptive clipping removes the differentially-private clipping norm `C` as a tuned
hyperparameter. Instead of a fixed `C`, it clips each contribution to a value that
tracks the `gamma`-quantile of the contribution-norm distribution, estimating that
quantile online and privately via gradient descent on the pinball (quantile-
regression) loss, with a multiplicative (geometric) update. The default is `gamma =
0.5`, i.e. clip to the median. The extra privacy cost is negligible (≈0.5% more noise
on the model updates at typical settings), and contributors send only their already-
clipped update plus a single bit, so the method is compatible with secure aggregation
and update compression.

## Problem it solves

In DP learning, each contributor's update is clipped to `L2` norm `C` and Gaussian
noise of standard deviation `zC` (proportional to the `L2` sensitivity) is added. `C`
sits at a bias-variance tradeoff: too small over-clips (bias), too large adds too
much noise (variance). There is no good a priori `C` — the norm distribution depends
on the model, loss, per-contributor data, and learning rate, and it drifts over
training by orders of magnitude — so a fixed `C` is wrong, and a hand-designed
schedule for `C` requires foreknowledge of the norm evolution. The goal is to set `C`
automatically and online, spending negligible privacy, without revealing raw norms.

## Key idea

Target a *quantile* of the norm distribution rather than an absolute magnitude: clip
so that a fraction `gamma` of contributions are left unclipped. A quantile tracks the
distribution as it slides, so it is scale-following by construction.

Estimate the quantile online with the pinball loss. For a scalar `X` and level
`gamma`,

```
ell_gamma(C; X) = (1 - gamma)(C - X)   if X <= C
               =  gamma  (X - C)        if X >  C
```

is convex with derivative `(1 - gamma)` when `X <= C` and `-gamma` when `X > C`, so

```
E[ ell'_gamma(C; X) ] = (1 - gamma) Pr[X <= C] - gamma Pr[X > C] = Pr[X <= C] - gamma,
```

which is zero exactly at `C* = the gamma-quantile of X` (`Pr[X <= C*] = gamma`). It is
`1`-Lipschitz convex, so online gradient descent converges to the quantile. Over a
round of `m` samples the average gradient is `b_bar - gamma`, where
`b_bar = (1/m) #{x_i <= C}` is the empirical fraction at or below `C` - only a count
is needed, never the magnitudes.

**Geometric update.** The additive step `C <- C - eta_C (b_bar - gamma)` moves `C` by
at most `eta_C` per round (slow when `C` is on the wrong order of magnitude, and can
overshoot negative). The multiplicative step is scale-free, converges from orders-of-
magnitude off, and stays positive:

```
C <- C * exp( -eta_C (b_bar - gamma) ).
```

Its steady-state jitter is proportional to the quantile value (constant relative
accuracy). With `eta_C = 0.2`, `gamma = 0.5`, and every update clipped (`b_bar = 0`),
`C` grows by `exp(0.1)` per round, i.e. a factor of ten every `ln(10)/0.1 ≈ 23`
rounds. Initialize `C` low (e.g. `0.1`); geometric growth catches up quickly, and a
too-high init wastes noise early.

**Privatizing the count.** The fraction `b_bar` leaks information about contributors'
norms, so it is privatized as a Gaussian sum of centered below-threshold bits
`b_i = I[||Delta_i|| <= C]`:

```
b_tilde = clip((1/m) ( sum_i (b_i - 1/2) + N(0, sigma_b^2) ) + 1/2, 0, 1).
```

Shifting the bits to `b_i - 1/2 in {-1/2, +1/2}` makes the count's `L2` sensitivity
`1/2` (a single record moves the sum by at most `1/2`), so its noise multiplier is
`2 sigma_b`. Recommended `sigma_b = m / 20`: the average count's stddev is then
`sigma_b / m = 1/20 = 0.05` regardless of `m`, so the error is `< 0.1` with 95.4%
probability and `< 0.15` with 99.7% probability; even `0.15` perturbs `C` by only
`exp(0.2 * 0.15) ≈ 1.03`.

**Joint privacy accounting.** Each round runs two Gaussian sum queries on the same
batch: the vector sum (clip `C`, noise `z_Delta C`) and the count (clip `1/2`, noise
`sigma_b`). Composing them as a single Gaussian query gives the combined noise
multiplier

```
z = ( z_Delta^{-2} + (2 sigma_b)^{-2} )^{-1/2},
```

so to hit a target combined `z` one sets the vector noise multiplier to

```
z_Delta = ( z^{-2} - (2 sigma_b)^{-2} )^{-1/2}.
```

`z_Delta` is only slightly larger than `z`: with `z = 1`, `m = 100` (`sigma_b = 5`),
`z_Delta = (1 - 0.01)^{-1/2} ≈ 1.005` - about 0.5% extra noise on the updates, and
this surcharge vanishes as `m` grows. The accountant charges the single combined `z`
per round (RDP composition with subsampling).

## Final algorithm (DP-FedAvg-M with adaptive clipping)

```
Inputs: clients/round m, target quantile gamma, clip LR eta_C, server LR eta_s,
        momentum beta, combined noise multiplier z, count stddev sigma_b
Initialize model theta^0, clip C^0 (small, e.g. 0.1), momentum mbar = 0
z_Delta = ( z^{-2} - (2 sigma_b)^{-2} )^{-1/2}
for each round t = 0, 1, 2, ...:
    Q^t  <- sample m users uniformly
    for each user i in Q^t (in parallel):
        Delta_i  <- local SGD from theta^t
        b_i      <- 1 if ||Delta_i|| <= C^t else 0
        Delta_i' <- Delta_i * min(1, C^t / ||Delta_i||)        # FlatClip to C^t
    sigma_Delta  <- z_Delta * C^t
    Delta_tilde  <- (1/m) ( sum_i Delta_i' + N(0, I sigma_Delta^2) )
    mbar         <- beta * mbar + Delta_tilde                   # server momentum
    theta^{t+1}  <- theta^t + eta_s * mbar
    b_tilde      <- clip((1/m) ( sum_i (b_i - 1/2) + N(0, sigma_b^2) ) + 1/2, 0, 1)
    C^{t+1}      <- C^t * exp( -eta_C ( b_tilde - gamma ) )     # geometric clip update
```

Server momentum is a post-processing of the privatized average, so it does not change
the privacy guarantee. (An adaptive server learning rate was tried and found worse
under DP, because the injected noise inflates the second-moment preconditioner
prematurely.) Default constants: `gamma = 0.5`, `eta_C = 0.2`, `sigma_b = m/20`,
`C^0 = 0.1`.

## Working code

Filling the scalar-policy slot in the same two-query shape as the standard privacy
query implementation:

```python
import collections
import tensorflow as tf


def calibrate_update_noise_multiplier(combined_z, count_stddev):
    inv = combined_z ** -2 - (2.0 * count_stddev) ** -2
    if inv <= 0:
        raise ValueError("count_stddev is too small for this combined z")
    return inv ** -0.5


def global_l2_norm(record):
    return tf.linalg.global_norm(tf.nest.flatten(record))


def clip_record(record, l2_norm_clip):
    norm = global_l2_norm(record)
    factor = tf.minimum(1.0, l2_norm_clip / (norm + 1e-12))
    return tf.nest.map_structure(lambda x: x * factor, record), norm


def sum_and_noise(records, stddev):
    summed = tf.nest.map_structure(lambda x: tf.reduce_sum(x, axis=0), records)
    return tf.nest.map_structure(
        lambda x: x + tf.random.normal(tf.shape(x), stddev=stddev), summed)


class QuantileEstimatorQuery:
    """DPQuery-like scalar estimator for the target below-threshold quantile."""

    GlobalState = collections.namedtuple("GlobalState", [
        "current_estimate", "target_quantile", "learning_rate"])
    SampleParams = collections.namedtuple("SampleParams", ["current_estimate"])

    def __init__(self, initial_estimate, target_quantile, learning_rate,
                 below_estimate_stddev, expected_num_records,
                 geometric_update=False):
        self.initial_estimate = float(initial_estimate)
        self.target_quantile = float(target_quantile)
        self.learning_rate = float(learning_rate)
        self.below_estimate_stddev = float(below_estimate_stddev)
        self.expected_num_records = float(expected_num_records)
        self.geometric_update = geometric_update

    def initial_global_state(self):
        return self.GlobalState(self.initial_estimate, self.target_quantile,
                                self.learning_rate)

    def derive_sample_params(self, state):
        return self.SampleParams(state.current_estimate)

    def preprocess_record(self, params, record_norm):
        # Center each 0/1 below-threshold bit so the count query has L2 clip 0.5.
        return tf.cast(record_norm <= params.current_estimate, tf.float32) - 0.5

    def get_noised_result(self, centered_below, state):
        noised_centered_count = (
            tf.reduce_sum(centered_below) +
            tf.random.normal([], stddev=self.below_estimate_stddev))
        below_estimate = noised_centered_count / self.expected_num_records + 0.5
        below_estimate = tf.clip_by_value(below_estimate, 0.0, 1.0)

        loss_grad = below_estimate - state.target_quantile
        update = state.learning_rate * loss_grad
        if self.geometric_update:
            new_estimate = state.current_estimate * tf.exp(-update)
        else:
            new_estimate = state.current_estimate - update
        new_state = state._replace(current_estimate=new_estimate)
        return new_estimate, new_state, below_estimate


class QuantileAdaptiveClipSumQuery:
    """Gaussian sum query whose clip is updated by QuantileEstimatorQuery."""

    GlobalState = collections.namedtuple("GlobalState", [
        "noise_multiplier", "l2_norm_clip", "quantile_estimator_state"])
    SampleParams = collections.namedtuple("SampleParams", [
        "l2_norm_clip", "quantile_estimator_params"])
    SampleState = collections.namedtuple("SampleState", [
        "clipped_records", "centered_below"])

    def __init__(self, initial_l2_norm_clip, noise_multiplier,
                 target_unclipped_quantile, learning_rate,
                 clipped_count_stddev, expected_num_records,
                 geometric_update=True):
        self.initial_l2_norm_clip = float(initial_l2_norm_clip)
        self.noise_multiplier = float(noise_multiplier)  # this is z_Delta
        self.expected_num_records = float(expected_num_records)

        self.quantile_estimator = QuantileEstimatorQuery(
            initial_l2_norm_clip, target_unclipped_quantile, learning_rate,
            clipped_count_stddev, expected_num_records, geometric_update)

    def initial_global_state(self):
        return self.GlobalState(
            self.noise_multiplier, self.initial_l2_norm_clip,
            self.quantile_estimator.initial_global_state())

    def derive_sample_params(self, state):
        return self.SampleParams(
            state.l2_norm_clip,
            self.quantile_estimator.derive_sample_params(
                state.quantile_estimator_state))

    def preprocess_record(self, params, record):
        clipped_record, record_norm = clip_record(record, params.l2_norm_clip)
        centered_below = self.quantile_estimator.preprocess_record(
            params.quantile_estimator_params, record_norm)
        return self.SampleState(clipped_record, centered_below)

    def get_noised_result(self, sample_state, state):
        noised_sum = sum_and_noise(
            sample_state.clipped_records,
            state.noise_multiplier * state.l2_norm_clip)

        new_clip, new_quantile_state, _ = self.quantile_estimator.get_noised_result(
            sample_state.centered_below, state.quantile_estimator_state)
        new_clip = tf.maximum(new_clip, 0.0)
        new_state = self.GlobalState(state.noise_multiplier, new_clip,
                                     new_quantile_state)
        noised_average = tf.nest.map_structure(
            lambda x: x / self.expected_num_records, noised_sum)
        return noised_average, new_state

```
