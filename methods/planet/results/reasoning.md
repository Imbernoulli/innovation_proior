I want an agent that controls from images with very few real environment episodes. If the dynamics were known, I would not need to learn a policy first; I could search over future actions, score predicted rewards, execute the first action, and re-plan after the next observation. So the route is model-based control. The obstacle is that the model has to be useful for planning, not merely good at one-step visual prediction.

The setting is partially observed. A single frame does not tell me velocities, contacts, or hidden object state, so the belief has to be a function of the history. That rules out a latent model that treats each image encoding as a complete Markov state. I need a filtering state, something like $q(s_t\mid o_{\le t},a_{<t})$, because at action time I only have the past. A smoother that looks at future observations could be convenient during training, but it would train the wrong inference object for online planning.

Pixel prediction is too expensive to put inside the planner. The model can still learn from pixels, but the planner should roll forward a compact state. I therefore start with a latent state-space model:
$$
p(s_t\mid s_{t-1},a_{t-1}),\qquad p(o_t\mid s_t),\qquad p(r_t\mid s_t).
$$
The observation likelihood gives dense gradients from the image, and the reward likelihood gives the signal the planner needs. During planning I only need the transition and reward model; generating images would waste computation.

Because the state is latent, I train with a variational bound. For the observation part, with actions suppressed only where the notation gets heavy,
$$
\log p(o_{1:T}\mid a_{1:T})
=\log \mathbb E_q\left[
\prod_{t=1}^T
\frac{p(o_t\mid s_t)p(s_t\mid s_{t-1},a_{t-1})}
{q(s_t\mid o_{\le t},a_{<t})}
\right].
$$
Jensen gives
$$
\log p(o_{1:T}\mid a_{1:T})\ge
\sum_{t=1}^T\left(
\mathbb E_{q(s_t\mid o_{\le t},a_{<t})}[\log p(o_t\mid s_t)]
-\mathbb E_{q(s_{t-1}\mid o_{\le t-1},a_{<t-1})}
\left[\mathrm{KL}\left(q(s_t\mid o_{\le t},a_{<t})\|p(s_t\mid s_{t-1},a_{t-1})\right)\right]\right).
$$
The sign is important: I maximize reconstruction log-likelihood and minimize posterior-to-prior KL. Reward is added by the same likelihood logic, as $\mathbb E_q[\log p(r_t\mid s_t)]$. With unit-variance Gaussian observation and reward distributions, each negative log-likelihood becomes a half squared error plus constants, summed over event dimensions and averaged over sampled time/batch entries in code.

Now I have to choose the latent dynamics. A purely stochastic Markov latent, sampled at every step, can represent uncertainty, but it has a memory problem. Information from an early frame has to survive a fresh sampled bottleneck again and again. It could theoretically learn zero variance on memory dimensions, but relying on that degenerate solution is fragile. A purely deterministic recurrent model has the opposite failure: it remembers well but gives only one future and removes the KL-regularized stochastic state. I need both channels.

So I split the state into a deterministic recurrent belief $h_t$ and a stochastic state $s_t$:
$$
h_t=f(h_{t-1},s_{t-1},a_{t-1}),\qquad
s_t\sim p(s_t\mid h_t),\qquad
o_t\sim p(o_t\mid h_t,s_t),\qquad
r_t\sim p(r_t\mid h_t,s_t).
$$
The posterior uses the same deterministic update and then folds in the current image embedding:
$$
q(s_t\mid h_t,o_t).
$$
This resolves the two extremes: $h_t$ carries memory deterministically, and $s_t$ carries uncertainty. I also have to prevent a shortcut. If the image embedding can flow deterministically into reconstruction, the model can copy the current image without forcing information through the sampled state, and then the prior that must plan without images learns too little. All observation information has to pass through the sampled posterior state before it reaches the decoder.

The feature vector for heads and planning is the concatenation $[s_t;h_t]$. That is not cosmetic. The reward head reads the same features that the transition can roll forward, so an action sequence can be scored by latent rollout alone. The image head is still essential for learning but not for planning.

For action selection, I use model-predictive control with a derivative-free optimizer. At the current time I infer the state belief from the current observation and previous action, then search over an open-loop sequence of future actions. A time-dependent diagonal Gaussian over action sequences starts at zero mean and unit variance. Each iteration samples $J$ candidate sequences, clips them to the action range, rolls the latent model forward for horizon $H$, sums predicted mean rewards, keeps the best $K$, and refits the Gaussian. After $I$ iterations I execute only the first mean action and discard the rest, because the next observation will update the belief. Resetting the action-sequence distribution each environment step avoids anchoring the next search to a stale local optimum.

There is a small but important code-fidelity split here. The primary algorithm writes the CEM refit scale as a mean absolute deviation over the elite actions. The canonical Google code uses `tf.nn.moments` over the elite actions and sets `stddev = sqrt(variance + 1e-6)`. If I am writing the final code artifact to match the canonical code, I should use the variance refit, while still recording the primary algorithm's formula in the math description.

The training loop co-evolves data and model. I seed the replay dataset with random episodes, train the model on uniformly sampled sequence chunks, then periodically collect new episodes with the current planner plus Gaussian action noise. Repeating actions shortens the effective planning horizon and makes the visual dynamics smoother. The primary hyperparameter statement says five seed episodes and collection every $C=100$ update steps; the canonical code expresses this as `collect_every=5000` phase steps while each training operation advances by batch size 50, which is the same collection cadence in update-count terms.

The standard bound still has a gap for planning. The transition only receives one-step KL training: it is asked whether the next prior agrees with the next posterior. In a planner, the model rolls forward several steps without observations. If the model family is restricted and capacity is finite, the one-step optimum need not be the multi-step optimum. I need a way to train multi-step predictions without decoding every imagined image.

Define the $d$-step predictive prior by repeatedly applying the one-step transition and integrating intermediate states:
$$
p(s_t\mid s_{t-d})
=\int \prod_{\tau=t-d+1}^{t}p(s_\tau\mid s_{\tau-1})\,ds_{t-d+1:t-1}
=\mathbb E_{p(s_{t-1}\mid s_{t-d})}[p(s_t\mid s_{t-1})],
$$
with action conditioning understood. Replacing the one-step prior in the latent generative model gives a bound on the $d$-step predictive distribution:
$$
\log p_d(o_{1:T}\mid a_{1:T})
\ge
\sum_t\left(
\mathbb E_{q(s_t\mid o_{\le t},a_{<t})}[\log p(o_t\mid s_t)]
-\mathbb E_{p(s_{t-1}\mid s_{t-d},a_{t-d:t-2})q(s_{t-d}\mid o_{\le t-d},a_{<t-d})}
\left[\mathrm{KL}\left(q(s_t\mid o_{\le t},a_{<t})\|p(s_t\mid s_{t-1},a_{t-1})\right)\right]
\right).
$$
The second inequality in the derivation comes from pushing the log inside the multi-step prior recursion with Jensen. All expectations are outside the KL, so sample averages give an unbiased estimator of this objective. The KL direction remains posterior to one-step prior; what changes is that the conditioning state for that one-step prior comes from an open-loop rollout starting $d$ steps back.

I should not overstate the relation between this $d$-step bound and the ordinary one-step likelihood. The source presents a data-processing argument as a conjecture in expectation: for a Markov latent chain, $I(s_t;s_{t-d})\le I(s_t;s_{t-1})$, leading to an expected log-likelihood ordering. That motivates treating the multi-step objective as consistent with the one-step objective, but it is not a proven pointwise theorem I can lean on. The guaranteed variational lower bound is on the $d$-step predictive distribution itself.

A single distance $d$ is not enough for planning, so I average distances $1,\ldots,D$ and optionally weight each KL by $\beta_d$:
$$
\frac1D\sum_{d=1}^D\log p_d(o_{1:T}\mid a_{1:T})
\ge
\sum_t\left(
\mathbb E_q[\log p(o_t\mid s_t)]
-\frac1D\sum_{d=1}^D\beta_d\,
\mathbb E\left[\mathrm{KL}\left(q(s_t\mid o_{\le t},a_{<t})\|p(s_t\mid s_{t-1},a_{t-1})\right)\right]\right).
$$
For $d>1$ I stop gradients through the posterior targets, so the open-loop predictions move toward informed filtering states rather than dragging those filtering states toward weaker rollouts. With the mixed deterministic-stochastic state, this regularizer can be left off by default and kept as an option; the canonical code default has overshooting scale zero.

Now I translate the result into code while matching the canonical implementation rather than a convenient reimplementation. The state is a dictionary with `mean`, `stddev`, `sample`, `belief`, and `rnn_state`. The posterior calls the transition first, then uses the resulting belief plus the image embedding. Features are `concat(sample, belief)`. Free nats are not implemented by clamping KL upward to 3; the divergence loss is `max(0, KL - 3)`, which grants the first three nats zero gradient. Image and reward heads maximize log-probabilities; in the canonical config the reward loss scale defaults to 10, divergence scale to 1, and overshooting scale to 0. Preprocessing floors 8-bit images to 5-bit bins, divides by 32, adds uniform dequantization noise in `[0,1/32)`, and subtracts 0.5.

```python
def preprocess(image_uint8, bits=5):
    bins = 2 ** bits
    image = image_uint8.astype("float32")
    image = floor(image / (2 ** (8 - bits)))
    image = image / bins
    image = image + uniform_like(image, 0.0, 1.0 / bins)
    return image - 0.5

class RSSM:
    def transition(self, prev, prev_action):
        hidden = dense(concat([prev["sample"], prev_action]), 200, activation=relu)
        belief, rnn_state = gru_block_cell(hidden, prev["rnn_state"], units=200)
        hidden = dense(belief, 200, activation=relu)
        mean = dense(hidden, 30, activation=None)
        stddev = softplus(dense(hidden, 30, activation=None)) + 0.1
        sample = normal_diag(mean, stddev).sample()
        return {
            "mean": mean,
            "stddev": stddev,
            "sample": sample,
            "belief": belief,
            "rnn_state": rnn_state,
        }

    def posterior(self, prev, prev_action, embed):
        prior = self.transition(prev, prev_action)
        hidden = dense(concat([prior["belief"], embed]), 200, activation=relu)
        mean = dense(hidden, 30, activation=None)
        stddev = softplus(dense(hidden, 30, activation=None)) + 0.1
        sample = normal_diag(mean, stddev).sample()
        return prior, {
            "mean": mean,
            "stddev": stddev,
            "sample": sample,
            "belief": prior["belief"],
            "rnn_state": prior["rnn_state"],
        }

    def features(self, state):
        return concat([state["sample"], state["belief"]])

def compute_loss(priors, posteriors, targets, heads, free_nats=3.0):
    features = rssm.features(posteriors)
    image_logprob = heads["image"](features).log_prob(targets["image"])
    reward_logprob = heads["reward"](features).log_prob(targets["reward"])
    kl = kl_divergence(normal(posteriors), normal(priors))
    divergence_loss = relu(kl - free_nats)
    return (
        -1.0 * mean(image_logprob)
        -10.0 * mean(reward_logprob)
        +1.0 * mean(divergence_loss)
    )

def cem_plan(cell, reward_head, current_state, action_shape,
             horizon=12, amount=1000, topk=100, iterations=10):
    mean = zeros([horizon, *action_shape])
    stddev = ones([horizon, *action_shape])
    tiled_state = tile_state(current_state, amount)
    for _ in range(iterations):
        normal = standard_normal([amount, horizon, *action_shape])
        action = clip(normal * stddev[None] + mean[None], -1.0, 1.0)
        state = rollout_prior(cell, tiled_state, action)
        returns = sum_over_time(reward_head(cell.features(state)).mean())
        elite = top_k_indices(returns, topk)
        best = gather(action, elite)
        mean = best.mean(axis=0)
        variance = ((best - mean[None]) ** 2).mean(axis=0)
        stddev = sqrt(variance + 1e-6)
    return mean
```

That gives me the complete causal chain. I learn a filtering latent dynamics model from pixel sequences; the state has deterministic memory and stochastic uncertainty; all image information passes through the sampled latent; the objective is reconstruction and reward likelihood minus posterior-to-prior KL with free nats; optional latent overshooting trains open-loop consistency across distances without image generation; and the policy is CEM/MPC in latent space, using predicted rewards and re-planning after each new observation.
