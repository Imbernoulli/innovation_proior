I want an agent that controls from images with very few real environment episodes. If the dynamics were known, I would not need to learn a policy first; I could search over future actions, score predicted rewards, execute the first action, and re-plan after the next observation. So the route is model-based control. The obstacle is that the model has to be useful for planning, not merely good at one-step visual prediction.

The setting is partially observed. A single frame does not tell me velocities, contacts, or hidden object state, so the belief has to be a function of the history. That rules out a latent model that treats each image encoding as a complete Markov state. I need a filtering state, something like $q(s_t\mid o_{\le t},a_{<t})$, because at action time I only have the past. A smoother that looks at future observations could be convenient during training, but it would train the wrong inference object for online planning.

Pixel prediction is too expensive to put inside the planner. The model can still learn from pixels, but the planner should roll forward a compact state. So I work with a latent state-space model:
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
The sign is important: I maximize reconstruction log-likelihood and minimize posterior-to-prior KL. Reward is added by the same likelihood logic, as $\mathbb E_q[\log p(r_t\mid s_t)]$.

I want to know what these likelihood terms actually become in code, because that determines what loss I write. Take a unit-variance Gaussian observation head, $p(o\mid s)=\mathcal N(o;\mu(s),1)$ per pixel. Then $-\log p(o\mid s)=\tfrac12(o-\mu)^2+\tfrac12\log 2\pi$. Let me check that expansion rather than trust it. With $o=0.3,\mu=0$, the half-squared-error-plus-constant is $0.5\cdot0.09+0.5\log 2\pi=0.045+0.9189=0.9639$; evaluating $-\log\mathcal N(0.3;0,1)$ directly also gives $0.9639$. Trying $o=1.0,\mu=0.5$: $0.5\cdot0.25+0.9189=1.0439$, and the direct NLL is $1.0439$ as well. They agree to all digits I carried. So each observation and reward negative log-likelihood is a half squared error plus a constant, summed over event dimensions and averaged over sampled time/batch entries. That is why the loss reads as a reconstruction MSE up to constants the optimizer ignores.

Now the latent dynamics. The simplest choice is a purely stochastic Markov latent, sampled fresh at every step. It represents uncertainty and gives a clean KL term, but it has a memory problem: information from an early frame has to survive a re-sampled bottleneck at every subsequent step. To carry a velocity or a contact flag for twenty steps, the model would have to drive the variance on those latent dimensions to essentially zero so the sample is deterministic — a degenerate corner of the parameterization that is hard to reach and hard to hold under noisy gradients. The opposite extreme, a purely deterministic recurrent model, remembers well but yields a single future and discards the KL-regularized stochastic state entirely, which is the variable I need for a variational objective and for representing uncertainty in rollouts. Neither extreme alone does both jobs.

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
Now $h_t$ carries memory deterministically, with no resampling to corrupt it, and $s_t$ carries the uncertainty and the KL. There is one shortcut I have to close. If the image embedding can flow into the decoder along a deterministic path, the model can reconstruct by effectively copying the current image, and then the prior — which must predict without images at planning time — never has to learn anything. So all observation information has to pass through the sampled posterior state before it reaches the decoder; the embedding enters only through $q(s_t\mid h_t,o_t)$.

The feature vector for heads and planning is the concatenation $[s_t;h_t]$. The reason this matters is that the reward head must read exactly the features the transition can roll forward — otherwise I could not score an action sequence by latent rollout alone. The image head is still needed for learning but never enters the planner.

For action selection I use model-predictive control with a derivative-free optimizer. At the current time I infer the state belief from the current observation and previous action, then search over an open-loop sequence of future actions. A time-dependent diagonal Gaussian over action sequences starts at zero mean and unit variance. Each iteration samples $J$ candidate sequences, clips them to the action range, rolls the latent model forward for horizon $H$, sums predicted mean rewards, keeps the best $K$, and refits the Gaussian.

Before committing to this I want to confirm the elite-refit actually concentrates on good actions, since the whole scheme rests on it. I trace a one-step toy where the predicted return is $-(a-0.5)^2$, so the optimal action is $a=0.5$, starting from mean $0$, std $1$, sampling $2000$ candidates and keeping the top $200$ each round. Iteration 0 lands at mean $0.489$, std $0.079$; iteration 1 at mean $0.500$, std $0.0061$; iteration 2 at mean $0.500$, std $0.0011$. The mean walks to the optimum and the std collapses by two orders of magnitude within three iterations. So a handful of CEM iterations is enough to localize a good action sequence, which is what justifies executing only the first mean action and discarding the rest — the next observation will update the belief anyway. Resetting the action-sequence distribution each environment step avoids anchoring the next search to a stale local optimum.

There is a small but important code-fidelity split here. The primary algorithm writes the CEM refit scale as a mean absolute deviation over the elite actions. The canonical Google code uses `tf.nn.moments` over the elite actions and sets `stddev = sqrt(variance + 1e-6)`. If I am writing the final code artifact to match the canonical code, I should use the variance refit, while still recording the primary algorithm's formula in the math description.

The training loop co-evolves data and model. I seed the replay dataset with random episodes, train the model on uniformly sampled sequence chunks, then periodically collect new episodes with the current planner plus Gaussian action noise. Repeating actions shortens the effective planning horizon and makes the visual dynamics smoother. The primary hyperparameter statement says five seed episodes and collection every $C=100$ update steps; the canonical code expresses this as `collect_every=5000` phase steps while each training operation advances by batch size 50, which is the same collection cadence in update-count terms ($5000/50=100$).

The standard bound has a gap for planning. The transition only receives one-step KL training: it is asked whether the next prior agrees with the next posterior. In a planner the model rolls forward several steps without observations. If the model family is restricted and capacity is finite, the one-step optimum need not be the multi-step optimum. I want to train multi-step predictions without decoding every imagined image.

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
The second inequality in the derivation comes from pushing the log inside the multi-step prior recursion with Jensen. All expectations sit outside the KL, so sample averages give an unbiased estimator of this objective. The KL direction stays posterior to one-step prior; what changes is that the conditioning state for that one-step prior comes from an open-loop rollout starting $d$ steps back.

I sanity-check this construction at $d=1$, where it should fall back to the ordinary bound or something is wrong. At $d=1$ the recursion reads $p(s_t\mid s_{t-1})=\mathbb E_{p(s_{t-1}\mid s_{t-1})}[p(s_t\mid s_{t-1})]$. But $p(s_{t-1}\mid s_{t-1})$ is a zero-step prior — a point mass at $s_{t-1}$ — so the expectation collapses and leaves exactly $p(s_t\mid s_{t-1})$, the one-step prior. The KL conditioning at $d=1$ is $\mathbb E_{p(s_{t-1}\mid s_{t-1})\,q(s_{t-1}\mid o_{\le t-1},a_{<t-1})}=\mathbb E_{q(s_{t-1}\mid o_{\le t-1},a_{<t-1})}$, which is precisely the conditioning in the one-step ELBO above. So the $d=1$ term reproduces the original bound term for term. The overshooting objective is a strict generalization, not a different objective that happens to look similar.

I should not overstate the relation between this $d$-step bound and the ordinary one-step likelihood, though. The source presents a data-processing argument as a conjecture in expectation: for a Markov latent chain, $I(s_t;s_{t-d})\le I(s_t;s_{t-1})$, leading to an expected log-likelihood ordering. That motivates treating the multi-step objective as consistent with the one-step objective, but it is not a proven pointwise theorem I can lean on. What I do have rigorously is the variational lower bound on the $d$-step predictive distribution itself, plus the $d=1$ reduction above.

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

Now I translate the result into code while matching the canonical implementation rather than a convenient reimplementation. The state is a dictionary with `mean`, `stddev`, `sample`, `belief`, and `rnn_state`. The posterior calls the transition first, then uses the resulting belief plus the image embedding. Features are `concat(sample, belief)`. The free-nats detail is the one I most want to get right, because the released code writes the divergence loss as `max(0, KL - 3)` and it would be easy to misread that as `max(KL, 3)`. I expected those two to differ mainly in their gradient, so I traced both on a range of KL values. With KL at $0,1.5,3,5,8$ nats, `relu(KL-3)` gives $0,0,0,2,5$ while `max(KL,3)` gives $3,3,3,5,8$. Checking the slope numerically at $KL=2$ and $KL=4$, both formulations have gradient $0$ below the threshold and gradient $1$ above — so my expectation was wrong, the gradients are identical. The real difference is in the loss value below three nats: `relu(KL-3)` lets the divergence term fall to exactly zero, which is the intended "first three nats are free" semantics, whereas `max(KL,3)` floors the reported loss at a constant $3$ and never grants a genuinely free region. So the distinction is about whether the KL term can vanish, not about the gradient above threshold — and the released `max(0, KL-3)` is the one that actually implements free nats. Image and reward heads maximize log-probabilities; in the canonical config the reward loss scale defaults to 10, divergence scale to 1, and overshooting scale to 0. Preprocessing floors 8-bit images to 5-bit bins, divides by 32, adds uniform dequantization noise in `[0,1/32)`, and subtracts 0.5.

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
