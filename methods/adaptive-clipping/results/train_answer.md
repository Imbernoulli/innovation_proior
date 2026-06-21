Training with differential privacy forces every update through a single scalar clipping norm C. Each contributor's gradient or model delta is projected into the L2 ball of radius C, the clipped contributions are summed, and Gaussian noise with standard deviation proportional to C is added. Because C is exactly the L2 sensitivity of that sum, the noise scale is not a free choice: the privacy guarantee requires the Gaussian mechanism to scale with C. This makes C the pivot of a bias-variance tradeoff. Set it too low and large contributions are flattened, biasing the aggregate toward small updates. Set it too high and the noise zC swamps the signal. The right value is narrow and unknown in advance, and it drifts over training as norms move by orders of magnitude. A fixed C chosen well for one phase is wrong for another, and a hand-designed schedule requires foreknowledge of that drift that practitioners do not have.

Existing approaches leave the problem unsolved. DP-SGD and DP-FedAvg treat C as a constant hyperparameter; the original DP-SGD paper even suggested clipping to the median gradient norm as a heuristic but gave no private way to estimate it. Coordinate-wise adaptive clipping relocates the tuning burden to a new hyperparameter. Private median estimation via smooth sensitivity would force contributors to reveal raw norms or add an extra communication round, breaking secure aggregation and update compression and giving the server more information than necessary. What is needed is a scale-tracking quantile of the contribution-norm distribution that can be estimated online from information the server is already allowed to see, at negligible privacy cost, and using only the clipped update plus a single bit per contributor.

The method I propose is Adaptive Quantile Clipping. Instead of fixing an absolute clipping norm, it targets a quantile of the current contribution-norm distribution, by default the median with target fraction gamma = 0.5. The intuition is that a quantile is a position in the distribution rather than an absolute magnitude, so when the whole norm distribution slides up or down by orders of magnitude the quantile slides with it. The algorithm clips each contribution so that a fraction gamma of contributions are left untouched; as training progresses, the clipping threshold tracks the distribution automatically.

To estimate the quantile online, the method runs gradient descent on the pinball loss from quantile regression. For a sample norm X and current threshold C, the loss is (1 - gamma)(C - X) when X is at most C and gamma(X - C) when X is larger. Its derivative with respect to C is (1 - gamma) for samples below C and -gamma for samples above C, so the expected derivative is Pr[X <= C] - gamma, which is zero exactly at the gamma-quantile. The loss is convex and 1-Lipschitz, so online gradient descent converges to the quantile. In practice the gradient over a sampled batch reduces to the difference between the empirical fraction of contributions that fall below C and the target gamma, a single scalar that requires only a count, not the actual norms.

A plain additive update C <- C - eta_C(b_bar - gamma) fails because C and eta_C can live on different scales: starting four orders of magnitude too low would take hundreds of rounds to recover, while a step that is too large can overshoot or even drive C negative. The fix is a multiplicative geometric update C <- C * exp(-eta_C(b_bar - gamma)). This makes the step a percentage change rather than an absolute change, so convergence is scale-free. With eta_C = 0.2 and gamma = 0.5, if every contribution is clipped then C grows by a factor of ten roughly every twenty-three rounds. The update also stays positive because it multiplies a positive value by a positive exponential. Initialize C small, around 0.1, and geometric growth brings it to the right scale within the first few dozen rounds of a long training run.

The fraction b_bar is private information about contributors' update norms, so it must be released through the same Gaussian-sum mechanism. Each contributor sends one extra bit indicating whether its unclipped norm is at most C. The server sums these bits, adds Gaussian noise, and divides by the number of sampled contributors. Centering the bits by subtracting 1/2, so each value lies in {-1/2, +1/2}, halves the L2 sensitivity of the count query from 1 to 1/2. Choosing the count noise so that sigma_b = m / 20 fixes the standard deviation of the released fraction at 1/20 regardless of batch size m, which is accurate enough that a three-sigma error perturbs C by only about three percent. The two Gaussian queries per round, the vector update sum and the scalar count sum, are composed into a single Gaussian-sum query for privacy accounting by rescaling each group by its clip divided by its noise standard deviation. The combined noise multiplier is z = (z_Delta^{-2} + (2 sigma_b)^{-2})^{-1/2}, where z_Delta is the vector update multiplier. To hit a target combined z, set z_Delta = (z^{-2} - (2 sigma_b)^{-2})^{-1/2}. At z = 1 and m = 100 this surcharge is about half a percent extra noise on the model updates and vanishes as m grows, so adaptivity is essentially free.

Each round therefore samples m contributors, runs local training, clips each delta to the current C, and returns the clipped delta plus a single below-threshold bit. The server averages the noised clipped deltas, applies server momentum as post-processing, and steps the model. It then computes the noised fraction of below-threshold contributors and updates C geometrically. Server momentum does not affect the privacy guarantee because it acts only on the already-privatized average. The default constants are gamma = 0.5, eta_C = 0.2, sigma_b = m / 20, and C^0 = 0.1.

```python
import collections
import math
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
                 geometric_update=True):
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
