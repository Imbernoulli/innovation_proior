The problem I set out to solve is throughput. I have an actor-critic agent — a policy $\pi$ and a value baseline $V^\pi$ — and on a single domain it already burns on the order of a billion frames and several days of wall-clock to get good. I want to train one agent on tens of domains at once, and at that scale a single GPU, or a single machine's worth of actors, cannot feed a deep network fast enough. So the first question is architectural: what is the most experience I can push through this system per second? A3C is the obvious template — many workers, each holding its own copy of the parameters, each stepping its own environment and computing an on-policy $n$-step advantage gradient against its local copy, then shipping the *gradient* back to a parameter server, Hogwild-style. But sending gradients couples bandwidth to model size and only gets worse as nets deepen; each worker does a tiny forward/backward that a GPU hates; and with many asynchronous workers the gradients arriving at the server are stale. Batched A2C, the synchronous alternative, steps a whole batch of environments in lockstep — which batches the tensor ops nicely, but every batch step runs only as fast as its slowest environment, so on expensive, high-variance 3D simulators like DeepMind Lab the tail of the per-step time distribution caps throughput. GA3C decouples acting from learning with dynamic batching, which is the right instinct, but it runs an on-policy update on data the decoupling has made off-policy and patches the resulting instability by adding a small constant to action probabilities — a band-aid over the real wound.

That wound is the crux. The way to get throughput is to flip what gets communicated: let many cheap, fully asynchronous *actors* only generate experience and ship raw trajectories to one GPU *learner* that holds the optimiser, stacks trajectories into one fat batch, folds time into the batch dimension for the conv net and output layer, and does a single accelerated update. But by the time a trajectory reaches the learner, the learner's policy has already moved several updates ahead, so the behaviour policy $\mu$ that generated the data lags the learner's current policy $\pi$ — and that policy-lag grows with exactly the scale I am chasing. The data is off-policy. Plain importance sampling would correct it: reweight a multi-step return by the product of per-step ratios $\prod_t \pi(a_t|x_t)/\mu(a_t|x_t)$, unbiased in principle, but that product explodes or collapses as the horizon grows and as $\mu$ and $\pi$ disagree. Retrace$(\lambda)$ tames the product by clipping each ratio, but it is built to correct a state-action value $Q$ and requires learning $Q$, whereas my actor-critic learns a state-value $V$; and it does not collapse to the ordinary on-policy $n$-step return when $\mu = \pi$. I want one rule that targets $V$, that truncates the explosive trace, and that *is* the on-policy update when there is no lag.

I propose IMPALA, the decoupled actor-learner architecture, with V-trace as its off-policy correction. The defining object is a value *target* written as "current estimate plus correction". For a trajectory $(x_t, a_t, r_t)_{t=s}^{s+n}$ generated under $\mu$, the $n$-step V-trace target for $V(x_s)$ is

$$v_s = V(x_s) + \sum_{t=s}^{s+n-1} \gamma^{t-s} \Big(\prod_{i=s}^{t-1} c_i\Big)\, \delta_t V,\qquad \delta_t V = \rho_t\big(r_t + \gamma V(x_{t+1}) - V(x_t)\big),$$

with the truncated per-step weights

$$\rho_t = \min\!\big(\bar\rho,\, \tfrac{\pi(a_t|x_t)}{\mu(a_t|x_t)}\big),\qquad c_i = \min\!\big(\bar c,\, \tfrac{\pi(a_i|x_i)}{\mu(a_i|x_i)}\big),\qquad \prod_{i=s}^{s-1} c_i = 1,$$

and the assumption $\bar\rho \ge \bar c$. The structural insight is that two different things are wrong off-policy, in two different places, and they need two different knobs. Each TD residual $r_t + \gamma V(x_{t+1}) - V(x_t)$ was produced by an action drawn from $\mu$, but I want it as if drawn from $\pi$ — a per-step distributional mismatch corrected by $\rho_t$, which lives *inside* one TD. Separately, a residual at time $t$ only belongs in the target for $x_s$ if the path $a_s, \dots, a_{t-1}$ was something $\pi$ would have taken — so its *propagation* back to $s$ is corrected by the trace product $\prod_{i=s}^{t-1} c_i$. The trace is the multiplicative object that explodes, so clipping the $c_i$ is non-negotiable; the $\rho_t$ are never multiplied across time, so their variance does not compound with the horizon, which is why the two ceilings can be set independently and why $\bar\rho \ge \bar c$ is natural.

What makes this the right correction is the fixed-point analysis. Define the operator $R V(x) = V(x) + \mathbb{E}_\mu\big[\sum_{t\ge 0}\gamma^t (c_0\cdots c_{t-1})\rho_t(r_t + \gamma V(x_{t+1}) - V(x_t))\big]$. Taking two value functions and subtracting, the rewards and constants cancel and $R$ acts linearly on the difference; reindexing the telescoping sum so every term is in $V_1(x_t) - V_2(x_t)$ gives weights $\alpha_t = \rho_{t-1} - c_{t-1}\rho_t$. These are non-negative in expectation precisely because $\bar c \le \bar\rho$ forces $c_{t-1} \le \rho_{t-1}$, so $\alpha_t \ge c_{t-1}(1 - \rho_t)$, and conditioning on the history through $x_t$ gives $\mathbb{E}_\mu[\rho_t | x_t] \le \sum_a \pi(a|x_t) = 1$. Summing the coefficients yields total weight $\eta = \gamma^{-1} - (\gamma^{-1}-1)\,\mathbb{E}_\mu\big[\sum_{t\ge 0}\gamma^t(\prod_{i=0}^{t-2}c_i)\rho_{t-1}\big] \le 1 - (1-\gamma)\beta < 1$, where $\beta \in (0,1]$ lower-bounds $\mathbb{E}_\mu \rho_0$ (the behaviour policy keeps *some* mass where $\pi$ wants to go). So $R$ is a sup-norm contraction with a unique fixed point reached geometrically; in the tabular case, online V-trace with Robbins-Monro stepsizes converges almost surely to it.

The payoff is *which* value function that fixed point is. Setting the expected correction to zero, the key identity $\mu(a|x)\min(\bar\rho, \pi(a|x)/\mu(a|x)) = \min(\bar\rho\,\mu(a|x), \pi(a|x))$ lets me normalise and read off that the fixed point is $V^{\pi_{\bar\rho}}$, the value of

$$\pi_{\bar\rho}(a|x) = \frac{\min(\bar\rho\,\mu(a|x),\, \pi(a|x))}{\sum_b \min(\bar\rho\,\mu(b|x),\, \pi(b|x))},$$

a policy *between* $\mu$ and $\pi$. With $\bar\rho = \infty$ the min is always $\pi$, so $\pi_{\bar\rho} = \pi$ and the fixed point is $V^\pi$ with no off-policy bias; as $\bar\rho \to 0$ it tends to $V^\mu$. Crucially $\bar c$ appears nowhere in the fixed point — only in $\eta$. So the two knobs really are separated: $\bar\rho$, the clip on the residual ratio, picks *which* policy I evaluate (the bias), while $\bar c$, the clip on the trace, controls only the contraction speed and is the real variance lever. I can crank $\bar c$ down hard to kill variance and still land on the same $V^{\pi_{\bar\rho}}$. And in the on-policy case $\pi = \mu$ (with $\bar c \ge 1$), every $\rho_t = 1$ and $c_i = 1$, and the target telescopes to the ordinary $n$-step Bellman target $\sum_{t} \gamma^{t-s} r_t + \gamma^n V(x_{s+n})$ — one algorithm for both regimes.

For computation I unfold the double sum from the back: peeling the $t=s$ term and factoring $\gamma c_s$ out of the rest gives the backward recursion $v_s = V(x_s) + \delta_s V + \gamma c_s (v_{s+1} - V(x_{s+1}))$, a single cheap vectorised pass scanning from the bootstrap value at the tail. For the policy I importance-sample between the evaluated policy $\pi_{\bar\rho}$ and $\mu$; since $\pi_{\bar\rho}/\mu \propto \min(\bar\rho, \pi/\mu)$, the policy-gradient weight on the score is exactly the same truncated $\rho_s$. The $Q$-estimate matters: $q_s = v_s$ is tempting but biased — unfolding at a perfect value gives the blend $(1-\rho_s)V^{\pi_{\bar\rho}}(x_s) + \rho_s Q^{\pi_{\bar\rho}}(x_s,a_s)$ — whereas $q_s = r_s + \gamma v_{s+1}$ has all future expected TDs vanish at the true value, giving $\mathbb{E}[q_s|x_s,a_s] = Q^{\pi_{\bar\rho}}(x_s,a_s)$ exactly. So the policy gradient ascends $\rho_s \nabla_\omega \log \pi_\omega(a_s|x_s)\,(r_s + \gamma v_{s+1} - V_\theta(x_s))$, with the baseline $V_\theta(x_s)$ subtracted (it does not bias the gradient). The value head descends the $\ell_2$ loss $(v_s - V_\theta(x_s))\nabla_\theta V_\theta(x_s)$, and an entropy bonus $-\nabla_\omega \sum_a \pi_\omega(a|x_s)\log\pi_\omega(a|x_s)$ delays premature collapse. The learner update is the weighted sum of these three terms.

Working in log-space for the ratios and taking $\bar\rho = \bar c = 1$ (the simplest stable choice), the V-trace target and policy-gradient advantage are computed as follows:

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

The learner loss is the three weighted terms; the cross-entropy carries the $-\log\pi$, so multiplying it by the corrected PG advantage gives the policy-gradient ascent:

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

Optimised with RMSProp and a linearly-decayed learning rate; typical settings are $n$ (unroll) $= 20$–$100$, $\gamma = 0.99$, baseline cost $0.5$, entropy cost $\approx 0.01$, and $\bar\rho = \bar c = 1$.
