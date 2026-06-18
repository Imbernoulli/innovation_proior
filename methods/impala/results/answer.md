# IMPALA and V-trace

## Problem

Scale deep reinforcement learning to very high throughput (many machines, GPU
learners) so a single agent can be trained on many tasks at once, without losing
data efficiency or stability. The obstacle that scaling creates: if you decouple
many *actors* (which generate experience) from one or a few *learners* (which do
the SGD), the actor's policy that produced a trajectory lags behind the learner's
current policy by several updates by the time the gradient is computed. The data
is therefore **off-policy**, and a naive on-policy actor-critic update on it is
biased and unstable.

## Architecture

- **Actors** run on (many) machines. At the start of a trajectory an actor copies
  the learner's latest policy into its local **behaviour policy** μ, runs μ for
  `n` steps in its environment, and sends the trajectory
  `x_1, a_1, r_1, …, x_n, a_n, r_n` together with the behaviour probabilities
  `μ(a_t|x_t)` (and the initial recurrent state) to the learner through a queue.
  Actors communicate *experience*, not gradients — cheaper, fault-tolerant,
  decoupled from model size.
- The **learner** consumes batches of these trajectories and does GPU SGD,
  parallelising every time-independent op (fold time into batch for the conv net
  and the output layer; fuse the LSTM). This gives throughput an online agent
  like A3C cannot reach. Multiple learners shard the parameters and update
  synchronously.

Because μ lags the learner's policy π, the learner's update must correct for the
**policy-lag** off-policiness. That correction is V-trace.

## V-trace target

Given a trajectory `(x_t, a_t, r_t)_{t=s}^{s+n}` generated under behaviour μ, with
target policy π and a value estimate `V`, the `n`-step V-trace value target for
`V(x_s)` is

```
v_s = V(x_s) + Σ_{t=s}^{s+n-1} γ^{t-s} ( Π_{i=s}^{t-1} c_i ) δ_t V
```

with the temporal difference and truncated importance weights

```
δ_t V = ρ_t ( r_t + γ V(x_{t+1}) − V(x_t) )
ρ_t   = min( ρ̄ , π(a_t|x_t) / μ(a_t|x_t) )
c_i   = min( c̄ , π(a_i|x_i) / μ(a_i|x_i) )          ( Π_{i=s}^{s-1} c_i = 1 )
```

and the assumption `ρ̄ ≥ c̄`. It computes recursively (this is how code does it):

```
v_s = V(x_s) + δ_s V + γ c_s ( v_{s+1} − V(x_{s+1}) ).
```

**The two clipping constants do different jobs.**

- **ρ̄ sets the fixed point (the bias).** `ρ_t` multiplies the TD residual, so it
  determines *which* value function the update solves for. The fixed point (where
  the expected TD is zero under μ) is `V^{π_ρ̄}`, the value function of

  ```
  π_ρ̄(a|x) = min(ρ̄ μ(a|x), π(a|x)) / Σ_b min(ρ̄ μ(b|x), π(b|x)),
  ```

  a policy *between* μ and π. With `ρ̄ = ∞` (no truncation) `π_ρ̄ = π`, so the
  fixed point is the target's value `V^π`; as `ρ̄ → 0` it tends to `V^μ`. Larger
  ρ̄ means less off-policy bias. The `ρ_t` are **not** multiplied together over
  time, so their variance does not explode with the horizon.

- **c̄ sets the contraction speed (the variance), not the fixed point.** The `c_i`
  are the trace; their product `c_s…c_{t-1}` is how much a TD seen at time `t`
  propagates back to the update at time `s`. That **product** is where variance
  blows up the more off-policy the data is, so truncating it with c̄ is the main
  variance-reduction lever. The fixed point is independent of c̄ — it depends on
  ρ̄ only. c̄ only changes the contraction modulus, i.e. how fast we converge.

**On-policy special case.** If π = μ (and c̄ ≥ 1), all `c_i = 1` and `ρ_t = 1`, and
the target telescopes to the on-policy `n`-step Bellman target

```
v_s = Σ_{t=s}^{s+n-1} γ^{t-s} r_t + γ^n V(x_{s+n}).
```

So the *same* algorithm handles on- and off-policy data (a property Retrace lacks).

## Contraction

Define the operator
`R V(x) = V(x) + E_μ[ Σ_{t≥0} γ^t (c_0…c_{t-1}) ρ_t (r_t + γ V(x_{t+1}) − V(x_t)) ]`.
With `ρ̄ ≥ c̄` and some `β ∈ (0,1]` such that `E_μ ρ_0 ≥ β`, `R` is an
η-contraction in sup-norm with unique fixed point `V^{π_ρ̄}`, where

```
η = γ^{-1} − (γ^{-1} − 1) E_μ[ Σ_{t≥0} γ^t (Π_{i=0}^{t-2} c_i) ρ_{t-1} ]
  ≤ 1 − (1 − γ) β < 1.
```

Smaller `c̄` ⇒ smaller traces ⇒ closer to `1 − (1−γ)β` (slower); the bound shows
convergence holds for *any* truncation level. In the tabular case, online V-trace
updates with Robbins-Monro stepsizes converge a.s. to `V^{π_ρ̄}`.

## V-trace actor-critic update

With value params θ and policy params ω, on a batch of behaviour trajectories:

- **value**: descend the l2 loss to the target — direction
  `(v_s − V_θ(x_s)) ∇_θ V_θ(x_s)`;
- **policy**: ascend the V-trace policy gradient — direction
  `ρ_s ∇_ω log π_ω(a_s|x_s) ( r_s + γ v_{s+1} − V_θ(x_s) )`,
  where `q_s = r_s + γ v_{s+1}` estimates `Q^{π_ρ̄}(x_s,a_s)` and `V_θ(x_s)` is the
  baseline; `q_s` (not `v_s`) is used because under a perfect value function
  `E[q_s | x_s, a_s] = Q^{π_ρ̄}(x_s,a_s)` exactly, whereas `v_s` is not unbiased
  for `Q`;
- **entropy**: add `−∇_ω Σ_a π_ω(a|x_s) log π_ω(a|x_s)` to delay premature
  convergence.

The total update is the weighted sum of the three.

## Code

V-trace target and policy-gradient advantage (`clip_rho_threshold` = ρ̄; the
trace uses `min(1, ρ)`, i.e. c̄ = 1):

```python
import collections
import tensorflow as tf

VTraceReturns = collections.namedtuple('VTraceReturns', 'vs pg_advantages')


def log_probs_from_logits_and_actions(policy_logits, actions):
  # log π(a_t | x_t) for the chosen actions; shape [T, B].
  return -tf.nn.sparse_softmax_cross_entropy_with_logits(
      logits=policy_logits, labels=actions)


def from_importance_weights(log_rhos, discounts, rewards, values,
                            bootstrap_value,
                            clip_rho_threshold=1.0,
                            clip_pg_rho_threshold=1.0):
  """V-trace targets vs and PG advantages from log IS weights.

  log_rhos[t] = log( π(a_t|x_t) / μ(a_t|x_t) ).  discounts[t] = γ·(not done).
  values[t] = V(x_t) under the target policy; bootstrap_value = V(x_T).
  clip_rho_threshold = ρ̄ (fixed point); the trace clips ρ at 1 (c̄ = 1).
  """
  rhos = tf.exp(log_rhos)
  clipped_rhos = tf.minimum(clip_rho_threshold, rhos)        # ρ_t = min(ρ̄, π/μ)
  cs = tf.minimum(1.0, rhos)                                 # c_t = min(c̄, π/μ), c̄=1

  # V(x_{t+1}) with the bootstrap value appended at the tail (n-step truncation).
  values_t_plus_1 = tf.concat([values[1:],
                               tf.expand_dims(bootstrap_value, 0)], axis=0)
  # δ_t V = ρ_t ( r_t + γ V(x_{t+1}) − V(x_t) ).
  deltas = clipped_rhos * (rewards + discounts * values_t_plus_1 - values)

  # Recursion v_s − V(x_s) = δ_s + γ c_s ( v_{s+1} − V(x_{s+1}) ), scanned
  # backwards from the end of the trajectory.
  def scanfunc(acc, seq):
    discount_t, c_t, delta_t = seq
    return delta_t + discount_t * c_t * acc

  vs_minus_v_xs = tf.scan(
      fn=scanfunc, elems=(discounts, cs, deltas),
      initializer=tf.zeros_like(bootstrap_value),
      parallel_iterations=1, back_prop=False, reverse=True)
  vs = vs_minus_v_xs + values                                # v_s

  # PG advantage q_s − V(x_s) = ρ_s ( r_s + γ v_{s+1} − V(x_s) ).
  vs_t_plus_1 = tf.concat([vs[1:],
                           tf.expand_dims(bootstrap_value, 0)], axis=0)
  clipped_pg_rhos = tf.minimum(clip_pg_rho_threshold, rhos)  # ρ_s in the PG
  pg_advantages = clipped_pg_rhos * (rewards + discounts * vs_t_plus_1 - values)

  return VTraceReturns(vs=tf.stop_gradient(vs),
                       pg_advantages=tf.stop_gradient(pg_advantages))


def from_logits(behaviour_policy_logits, target_policy_logits, actions,
                discounts, rewards, values, bootstrap_value,
                clip_rho_threshold=1.0, clip_pg_rho_threshold=1.0):
  # log_rhos = log π(a) − log μ(a), computed in log-space for stability.
  target_log_probs = log_probs_from_logits_and_actions(
      target_policy_logits, actions)
  behaviour_log_probs = log_probs_from_logits_and_actions(
      behaviour_policy_logits, actions)
  log_rhos = target_log_probs - behaviour_log_probs
  return from_importance_weights(
      log_rhos, discounts, rewards, values, bootstrap_value,
      clip_rho_threshold, clip_pg_rho_threshold)

def off_policy_targets(behaviour_policy_logits, target_policy_logits, actions,
                       discounts, rewards, values, bootstrap_value):
  returns = from_logits(behaviour_policy_logits, target_policy_logits, actions,
                        discounts, rewards, values, bootstrap_value)
  return returns.vs, returns.pg_advantages
```

Learner loss (the three weighted terms; cross-entropy carries the `−log π`, so
multiplying by the corrected PG advantage gives the policy-gradient ascent):

```python
def compute_baseline_loss(advantages):              # 0.5 Σ (v_s − V(x_s))^2
  return 0.5 * tf.reduce_sum(tf.square(advantages))

def compute_entropy_loss(logits):                   # −Σ entropy(π)
  policy = tf.nn.softmax(logits)
  log_policy = tf.nn.log_softmax(logits)
  return -tf.reduce_sum(tf.reduce_sum(-policy * log_policy, axis=-1))

def compute_policy_gradient_loss(logits, actions, advantages):
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=actions, logits=logits)               # = −log π(a|x)
  advantages = tf.stop_gradient(advantages)         # = ρ_s (r_s + γ v_{s+1} − V)
  return tf.reduce_sum(cross_entropy * advantages)

def learner_loss(behaviour_logits, target_logits, actions,
                 discounts, rewards, values, bootstrap_value,
                 baseline_cost=0.5, entropy_cost=0.01):
  value_targets, pg_advantages = off_policy_targets(
      behaviour_logits, target_logits, actions, discounts, rewards, values,
      bootstrap_value)
  loss  = compute_policy_gradient_loss(
      target_logits, actions, pg_advantages)
  loss += baseline_cost * compute_baseline_loss(
      value_targets - values)
  loss += entropy_cost * compute_entropy_loss(target_logits)
  return loss
```

Optimised with RMSProp and a linearly-decayed learning rate; typical settings are
`n` (unroll) = 20–100, γ = 0.99, baseline cost 0.5, entropy cost ~0.01,
`ρ̄ = c̄ = 1`.
