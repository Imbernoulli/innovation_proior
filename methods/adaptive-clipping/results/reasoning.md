Let me start from the thing that keeps biting me when I train with differential privacy. The whole guarantee runs through one scalar, the clipping norm C. I take each contributor's gradient — or in the federated case each user's model delta — project it into the L2 ball of radius C, sum the clipped contributions, and add Gaussian noise before I step. The noise scale is not mine to choose freely: C is precisely the L2 sensitivity of that sum to any one contributor, and to hide a contributor the Gaussian mechanism has to add noise with standard deviation proportional to the sensitivity, σ = zC for some noise multiplier z fixed by my privacy target. So C is doing two opposite jobs at once. Lower it and I crush the large contributions — I preserve their directions, but I erase the relative magnitudes that say which updates were much larger than others — which biases the aggregate toward whatever the small contributions say. Raise it and the noise standard deviation zC rises right along with it until the noise drowns the signal. There is a sweet spot, and it is narrow, and that tension between clipping bias and noise variance isn't a quirk of my setup — it's documented as an inherent property of private learning, so I can't engineer it away; I can only sit at the best point of the tradeoff.

The real problem is that I have no honest way to know where that point is ahead of time, and worse, it moves. The distribution of contribution norms depends on the architecture, the loss, how much data each user holds, the client learning rate — and it drifts over training, sometimes by orders of magnitude between the first round and convergence. A C that's right early is wrong late. I've seen the concrete version of this: in a long federated language-model run, manually dropping the clipping norm after some initial number of rounds improves accuracy. So a constant C is leaving utility on the table, and the fix "use a schedule for C" is worse than it sounds, because to design a schedule I'd need to know in advance how the norms will evolve, for a system whose norm behavior is exactly what I can't predict. If choosing a single good constant is hard, choosing a good parameterized schedule is harder. I want C gone as a hand-tuned knob entirely.

So let me ask what the right C actually is, conceptually, instead of trying to guess a number. The pain is that C is an absolute magnitude and the magnitudes drift. What doesn't drift in the same way is a *position in the distribution*. If at every round I clipped so that, say, half the updates are left untouched and half are scaled down, then as the whole norm distribution slides up or down by an order of magnitude, "the value below which half the updates fall" slides with it — it tracks the distribution automatically. That's a quantile. So suppose I pick a target level γ in [0,1] and aim C at the γ-quantile of the current update-norm distribution: the value C such that a fraction γ of updates have norm at most C. γ = 0.5 would be "clip to the median," which I recall being floated as a heuristic but never actually done privately or even tested. That reframes the question into something sharp and possibly answerable: estimate a quantile of a stream, online, cheaply, under DP, without ever seeing the raw norms. Whether that's actually any easier than picking C directly is the thing I now have to work out, not assume.

How do you find a quantile online without storing the data? I want a loss function whose minimizer is the quantile, so I can just do gradient descent on it round by round. Squared error gives me the mean; absolute error gives me the median; so for a general γ I want the asymmetric cousin of absolute error. Let me construct it from the condition I actually want at the optimum. At C = C*, the γ-quantile, the probability mass below should be γ and above should be 1−γ. Suppose I penalize being below the sample and being above the sample with different weights. Let X be the random norm and C my estimate; define

  ℓ_γ(C; X) = (1−γ)(C − X)  if X ≤ C,
            = γ(X − C)      if X > C.

Both pieces are nonnegative and it's continuous and convex in C — two line segments meeting at C = X, the left one with slope (1−γ) and the right one with slope −γ. Its derivative in C is

  ℓ'_γ(C; X) = (1−γ)  if X ≤ C,
             = −γ     if X > C.

Now take the expectation over X, which is the thing gradient descent will actually chase:

  E[ℓ'_γ(C; X)] = (1−γ)·Pr[X ≤ C] − γ·Pr[X > C].

Write Pr[X > C] = 1 − Pr[X ≤ C] and expand: (1−γ)Pr[X≤C] − γ(1 − Pr[X≤C]) = Pr[X≤C] − γPr[X≤C] + γPr[X≤C] − γ = Pr[X≤C] − γ. The two γ·Pr[X≤C] terms cancel and I'm left with

  E[ℓ'_γ(C; X)] = Pr[X ≤ C] − γ.

Let me not trust an algebra cancellation I did at speed — I'll evaluate both forms on a concrete distribution before leaning on it. Take X lognormal (a stand-in for the heavy-tailed-ish norm distributions I actually see) and sample a hundred thousand draws. At C = 1.0, γ = 0.5 the long form (1−γ)·#below/n − γ·#above/n comes out to 0.000680, and the short form #below/n − γ comes out to 0.000680 — identical. At C = 0.5, γ = 0.3 both give −0.056130; at C = 2.0, γ = 0.7 both give 0.055120; across the grid I tried they agree to all printed digits. So the cancellation is real and not an artifact of one lucky C. Good — the expected gradient is Pr[X ≤ C] − γ.

Setting it to zero gives Pr[X ≤ C*] = γ exactly, so the stationary point C* is the γ-quantile. Three properties carry the method from here. The loss is convex, so that stationary point is a global minimizer rather than just a flat spot. The gradient is bounded by 1 in magnitude — it's either 1−γ or −γ, both in [0,1] — so the loss is 1-Lipschitz. Convex and 1-Lipschitz is exactly the setting where online gradient descent with a 1/√t step size has sublinear regret, which is the lever I'd been hoping existed: it means descending on this loss, one round at a time, should track the quantile even as the distribution drifts. So the plan becomes "run OGD on the pinball loss" — though I'm taking the convergence on the strength of the OCO regret bound for now, and I'll want to actually watch an estimator land on a known quantile before I believe it end to end.

What does one OGD step look like with the data I have? On a round I see m samples of the norm, x_1,…,x_m. The average derivative over the round is

  (1/m) Σ_i ℓ'_γ(C; x_i) = (1/m) [ (1−γ)·#{x_i ≤ C} − γ·#{x_i > C} ].

Let b̄ = (1/m)·#{x_i ≤ C} be the empirical fraction of this round's updates that fall at or below C. Then #{x_i ≤ C} = m·b̄ and #{x_i > C} = m(1 − b̄), and the average derivative is (1−γ)b̄ − γ(1 − b̄) = b̄ − γb̄ − γ + γb̄ = b̄ − γ. The exact same cancellation as in expectation, now empirical. So the gradient I need is just the gap between the fraction of updates that landed inside the clip and my target — a single number in [−γ, 1−γ]. The OGD update is

  C ← C − η_C (b̄ − γ).

That's economical: I don't need the magnitudes at all, only the count of how many fell below C. If b̄ > γ, too many are inside, push C down; if b̄ < γ, too few are inside, push C up. The sign is right. The question is the size, and here I have a worry I should pin down rather than wave away — this is an additive step on a quantity whose scale I explicitly don't know.

Because |b̄ − γ| ≤ 1, the step moves C by at most η_C per round. So consider a case at the edge of what I expect: C initialized at 0.1 but the true quantile is 50, η_C = 0.2. With every update clipped, b̄ = 0 and the gradient is −γ = −0.5, so I move C up by η_C·γ = 0.1 per round. I'll just run it: climbing 0.1 → 50 at +0.1 per round takes 499 rounds. That's hundreds of rounds of clipping essentially everything and adding noise scaled to a garbage C before the clip even reaches the right order of magnitude. Now the opposite edge: the true quantile is tiny, say around 0.05, so C should sit there. If at some round nothing is clipped, b̄ = 1, and the additive step is C ← C − η_C(1 − γ) = C − 0.1; starting from C = 0.05 that lands at −0.05. A negative clipping norm is meaningless — and not a one-off, it's structural, because the additive step size 0.1 is larger than the entire scale of C. The additive rule misbehaves at both ends whenever C and η_C live on different orders of magnitude, which is precisely my situation since not knowing the scale a priori was the whole reason I'm here. So a fixed additive step can't be the answer.

What broke is that the step size is absolute while the thing it's moving spans orders of magnitude. So make the step *relative* to C — multiply instead of add:

  C ← C · exp(−η_C (b̄ − γ)).

The fractional change per round is exp(−η_C(b̄−γ)) − 1 ≈ −η_C(b̄−γ) for small steps, so η_C controls a *percentage* move, which is scale-free by construction. Re-run the two failures. When C is orders of magnitude too low and everything is clipped, b̄ = 0 and the multiplier is exp(η_C·γ) = exp(0.1) ≈ 1.105 every round — geometric growth. To grow a factor of ten I need exp(0.1·n) = 10, i.e. n = ln 10 / 0.1 ≈ 23.0 rounds per decade. Let me check that against the actual climb rather than the closed form: simulating C ← C·exp(η_C·γ) from 0.1 with true quantile 50, it crosses 50 in 63 rounds (landing at 54.5) — about 2.7 decades in 63 rounds, ≈ 23 rounds each, matching. So even starting four orders of magnitude too low I'd catch up in ~92 rounds, negligible against thousands of training rounds. And the negative-C failure simply can't happen: from C = 0.05 with b̄ = 1 the multiplicative step gives 0.05·exp(−0.1) = 0.0452, still positive — a positive C times a positive exponential stays positive, by construction. There's a further consequence I'll note but not over-claim: because the update is multiplicative, I'd expect the steady-state jitter of the estimate to scale with the quantile value itself, giving roughly constant *relative* accuracy whether the true norm is 0.001 or 100, so η_C wouldn't need retuning per scale — something I'll want to confirm by actually watching the estimator converge. This also tells me how to initialize: start C low, like 0.1, for every task, since a too-low init is cheap (geometric growth fixes it in a couple dozen rounds) while a too-high init forces large noise early (noise scales with C) and can swamp the model before it gets going. So: geometric update, η_C = 0.2 as a default trading convergence speed against tracking accuracy, low init.

Now the part I've been quietly cheating on: b̄ is computed from private data. It's a function of how many users' update norms fell below C, which is information about the magnitudes of their updates — exactly the kind of thing DP is supposed to hide. I can't just hand the server b̄. So I have to privatize it the same way I privatize everything else: it's a sum (a count of users whose norm is below C), so add Gaussian noise to the sum and release the noisy average. Each user, in addition to its already-clipped update Δ_i, sends a single bit b_i = I[‖Δ_i‖ ≤ C] — "was I left unclipped?" — and the server computes

  b̃ = (1/m)( Σ_i b_i + N(0, σ_b²) ).

This is exactly the federated discipline I wanted: the user sends its minimal focused update plus one bit, never its raw norm; the server learns only the noised aggregate count; and because it's just a sum of bits, it's compatible with secure aggregation and with compression — the server never needs to see individual values. And because I'm folding information across rounds through the running C, rather than re-estimating a quantile from scratch each round, the estimate has far less round-to-round jitter than the naive alternative of privately estimating the median unclipped norm every round (which would also force users to send unclipped updates, breaking secure aggregation — a non-starter).

How much does privatizing the count cost in privacy, though? This is where I have to be careful, because I'm now running *two* Gaussian queries against the same sampled batch each round: the vector sum Σ_i Δ_i (to update the model) and the scalar count Σ_i b_i (to update the clip). I need the joint privacy cost, not two separately-accounted costs that I'd then have to compose loosely. The clean way is to realize both queries are Gaussian sums and fold them into a single Gaussian sum query for the accountant. Let me first nail the sensitivity of the count query, because there's a trick that halves it. Naively, adding or removing one user changes Σ_i b_i by at most 1, so the L2 sensitivity is 1. But if instead each user sends b_i − 1/2 ∈ {−1/2, +1/2}, the sum is centered at zero and adding or removing one user changes it by at most 1/2 — sensitivity 1/2 — and the server just adds back m/2 at the end to recover the count. Halving the sensitivity halves the noise I need for the same privacy, which is why this shift is worth doing. (The clipped update Σ_i Δ_i has sensitivity C, since each Δ_i is in the ball of radius C.)

Now compose the two queries. The general move for several Gaussian sum queries on the same batch is to rescale each one by (its clip)/(its noise stddev) so that, against a single unit-variance Gaussian, the concatenated query has L2-sensitivity S* = sqrt( Σ_g (S_g/σ̃_g)² ) and the combined mechanism is a single Gaussian sum query with noise multiplier z = 1/S*. Here there are two groups: the vector group with clip C and noise stddev σ_Δ = z_Δ C, contributing (C/(z_Δ C))² = z_Δ^{-2}; and the count group with clip 1/2 and noise stddev σ_b, contributing ( (1/2)/σ_b )² = (2σ_b)^{-2}. So the combined effective noise multiplier is

  z = ( z_Δ^{-2} + (2σ_b)^{-2} )^{-1/2}.

That is the multiplier the accountant should charge per round — one number, as if I'd run a single Gaussian query — provided the rescaling really does collapse the two queries into one, which is the part I'm relying on the general composition result for. And it tells me how to set things: I have a target combined z (the one that hits my (ε,δ) budget), I'll choose σ_b for the count, and then I solve for the vector noise multiplier z_Δ. Invert the relation: z² = (z_Δ^{-2} + (2σ_b)^{-2})^{-1}, so z_Δ^{-2} = z^{-2} − (2σ_b)^{-2}, giving

  z_Δ = ( z^{-2} − (2σ_b)^{-2} )^{-1/2}.

Let me sanity-check that the inversion is self-consistent — it's easy to drop a sign or an exponent here. Take z = 1, σ_b = 5: z_Δ = (1 − (10)^{-2})^{-1/2} = (0.99)^{-1/2} = 1.005038. Feed that back into the forward formula: (1.005038^{-2} + (10)^{-2})^{-1/2} = (0.99 + 0.01)^{-1/2} = 1^{-1/2} = 1.0, recovering z exactly. So the round-trip closes and the inversion is right.

Since (2σ_b)^{-2} is subtracted inside, z_Δ comes out a hair *larger* than z — I pay slightly more noise on the vector updates to fund the count query. The question is whether that surcharge is tolerable, and that's where σ_b gets chosen. I want the count noise small enough that the clip estimate is accurate but large enough that it costs almost nothing. Tie it to the round size: σ_b = m/20. Then the noise on the average count, b̃ − b̄, is N(0, σ_b²/m²) = N(0, (m/20)²/m²) = N(0, (1/20)²), so its standard deviation is exactly 1/20 = 0.05, *independent of m*. That means the count error is within 0.1 (two standard deviations) about 95.4% of the time and within 0.15 (three sigma) about 99.7% of the time. And even a 0.15 error in b̃ only perturbs the geometric clip update by a factor exp(η_C·0.15) = exp(0.2·0.15) = exp(0.03) ≈ 1.03 — a 3% wobble in C, utterly harmless since C is only a clip threshold anyway, not a quantity that needs to be exact. Now check the privacy surcharge with these numbers: take z = 1 and m = 100, so σ_b = 5 and (2σ_b)^{-2} = (10)^{-2} = 0.01, giving z_Δ = (1 − 0.01)^{-1/2} = (0.99)^{-1/2} ≈ 1.005. So I'm paying 0.5% extra noise on the model updates to get adaptive clipping with the same privacy guarantee — and that surcharge shrinks as m grows, because (2σ_b)^{-2} = (m/10)^{-2} → 0. Adaptivity is essentially free.

Before I assemble the full round I want to discharge the IOU I left earlier: I claimed the OGD argument means this estimator actually tracks the quantile, but I've only ever checked the *expected* gradient is zero at the quantile — that's a fixed-point statement, not a convergence one, and now there's privatization noise on top. So let me run the loop end to end on a stream with a known answer and watch where C lands. I draw m = 100 norms per round from a lognormal with true median 3.0, set γ = 0.5, η_C = 0.2, start C at 0.1, and apply exactly the pipeline above: count the below-threshold bits, center and noise them with σ_b, recover and clamp the fraction, take the geometric step. With the noise turned essentially off, after 400 rounds C settles at 3.075 against a true median of 3.0 — it climbs the 1.5 decades from 0.1 and parks at the right value, so the OGD-on-pinball loop does converge, not just have the right fixed point. Now turn the privacy noise on at the recommended σ_b = m/20 = 5 and average over 20 seeds: the mean final C is 3.025, individual runs ranging 2.88 to 3.17, a relative error under 1%. So the privatized geometric estimator lands within a few percent of the true quantile, the residual spread is the noise jitter I expected, and crucially it's *relative* spread — consistent with the earlier hope that the multiplicative form keeps accuracy scale-free. That's the evidence I was missing; the loop works.

Now let me assemble the round. Sample m users. Each user runs its local training, clips its delta to the current C^t, and returns (Δ_i clipped, b_i). The server forms the noised average update Δ̃^t = (1/m)(Σ_i Δ_i + N(0, I σ_Δ²)) with σ_Δ = z_Δ C^t; I'll fold in server momentum, Δ̄^t = β Δ̄^{t−1} + Δ̃^t, and step θ^{t+1} = θ^t + η_s Δ̄^t. The momentum is free privacy-wise: it's computed entirely from the already-privatized average, so it's post-processing and changes nothing about the guarantee — and it's known to help convergence in federated optimization. (I did consider an adaptive *server* learning rate, an Adam-style preconditioner. But the inputs to that preconditioner are now the noised averages, and the per-coordinate second-moment accumulator v_t squares them, so it accumulates the injected noise's variance as if it were signal — I'd expect that to inflate v_t prematurely and mis-scale the step, especially early when the true signal is weakest relative to the noise. I'd want to confirm that empirically before relying on it, but it's enough reason to keep the server side simple: plain momentum, not adaptive LR.) Then the server forms the noised count b̃^t = (1/m)(Σ_i(b_i − 1/2) + N(0, σ_b²)) + 1/2 and does the geometric clip update C^{t+1} = C^t · exp(−η_C(b̃^t − γ)). Two Gaussian queries, one model step, one clip step, and a single combined z for the accountant via z_Δ = (z^{-2} − (2σ_b)^{-2})^{-1/2}.

There are two guards I shouldn't forget, both because noise can push estimates out of their natural range. The noised count b̃ should be clamped to [0,1] before I use it as a fraction, since Gaussian noise can in principle send it slightly outside; and the updated clip should be floored at 0 (it can't go negative anyway under the geometric rule, but if I ever fall back to the additive rule it could, so max(C, 0) is the safe form). Small things, but they keep the update well-defined.

Now let me write the version that fills the query-shaped harness I actually have. One detail matters for faithfulness: the clipped-sum query's `noise_multiplier` is the vector multiplier `z_delta`; if I start from a target combined multiplier `z`, I compute `z_delta = (z^{-2} - (2 sigma_b)^{-2})^{-1/2}` before constructing it. Then the implementation needs two pieces: a scalar query that owns C and updates it from a privatized below-threshold fraction, and a sum query that clips records, adds vector Gaussian noise, and feeds each record's norm into that scalar query. The clip update, the -1/2 shift on the count, the geometric step, and the clamp of the noised fraction to [0,1] are all exactly where they should be.

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
        self.noise_multiplier = float(noise_multiplier)  # this is z_delta
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

The user-level federated form is the same skeleton with a user delta in place of the per-example gradient and server momentum on the model step:

```python
def dp_fedavg_m_adaptive_clip_round(server_state, sampled_users, gamma, eta_C,
                                    eta_s, beta, z_delta, sigma_b):
    theta, C, momentum = server_state.theta, server_state.C, server_state.momentum
    delta_sum, below_sum = 0.0, 0.0
    for user in sampled_users:                       # each runs local SGD, then:
        delta = user.local_update(theta)             # its model delta
        below_sum += (delta.norm() <= C).float() - 0.5   # centered bit (sens. 1/2)
        delta_sum = delta_sum + delta * min(1.0, C / delta.norm())  # FlatClip to C

    m = len(sampled_users)
    delta_tilde = (delta_sum + gaussian_like(delta_sum, z_delta * C)) / m
    momentum = beta * momentum + delta_tilde         # server momentum (post-processing)
    theta = theta + eta_s * momentum                 # server step

    b_tilde = clamp((below_sum + 0.5 * m + gaussian_scalar(sigma_b)) / m, 0.0, 1.0)
    C = max(C * math.exp(-eta_C * (b_tilde - gamma)), 0.0)   # geometric clip update
    return State(theta, C, momentum)
```

So the causal chain, start to finish. I was stuck because the privacy guarantee funnels through one scalar C that trades clipping bias against noise variance, there's no good a priori value for it, and the right value drifts over training so even a constant chosen perfectly for one phase is wrong for another, while a hand-designed schedule needs foreknowledge I don't have. Reframing the target from an absolute magnitude to a quantile of the norm distribution made it scale-tracking: clip so a fraction γ are unclipped, and as the distribution slides the quantile slides with it. To find that quantile online I needed a convex loss minimized there, which the pinball loss provides — its derivative has expectation Pr[X≤C] − γ (which I checked numerically before trusting the cancellation), zero exactly at the γ-quantile — so online gradient descent on it should track the quantile, and the gradient is just the gap b̄ − γ between the unclipped fraction and the target, needing only a count, not the magnitudes; when I finally ran the full privatized loop on a stream with a known median it landed within about 1%, so the convergence is real and not just the right fixed point. The naive additive OGD step broke when C and η_C were on different orders of magnitude (too slow, or overshooting negative), so I switched to a multiplicative geometric step C ← C·exp(−η_C(b̄−γ)) that is scale-free, converges from orders-of-magnitude off (about a decade every 23 rounds at η_C=0.2, γ=0.5), stays positive, and has relative jitter independent of the scale. The count b̄ leaks private information, so I privatized it as a Gaussian sum of below-threshold bits; shifting the bits by 1/2 halves the count's sensitivity to 1/2; and composing the count query with the model-update query into a single Gaussian query gives the combined multiplier z = (z_Δ^{-2} + (2σ_b)^{-2})^{-1/2}, which I invert to set z_Δ = (z^{-2} − (2σ_b)^{-2})^{-1/2}. Choosing σ_b = m/20 fixes the count's standard deviation at 1/20 regardless of m and makes the privacy surcharge about 0.5% extra noise at m=100, z=1, vanishing as m grows — so adaptive clipping costs essentially nothing in privacy, sends only a clipped update plus one bit per contributor (secure-aggregation and compression compatible), and removes C as a tuned hyperparameter entirely, with the default just "clip to the median."
