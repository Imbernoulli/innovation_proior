I want an agent that learns to control from raw $64\times64\times3$ images while spending as few real environment episodes as possible, far fewer than a model-free visual agent needs. The clean way to be sample-efficient is model-based control: if I knew the dynamics I would not need to learn a policy at all, because I could search over future action sequences, score their predicted rewards, execute the first action, and re-plan after the next observation. So the real problem reduces to learning a model that is good enough to plan inside, not merely good at one-step prediction. Two facts make this hard. First, the setting is partially observed: a single frame does not reveal velocities, contact states, or hidden objects, so the agent must act from a belief over the history rather than from any one image. This rules out treating each image encoding as a complete Markov state, and it rules out a smoother that peeks at future frames during training, because at action time I only ever have the past. Second, pixel prediction is far too expensive to put inside a planner that scores thousands of candidate trajectories per step. The prior art each misses one of these: PILCO learns probabilistic dynamics for sample-efficient control but only on low-dimensional Markov states; E2C and RCE embed images but assume local linearity and strong Markov structure; PETS combines probabilistic ensembles with model-predictive control and the cross-entropy method but takes low-dimensional simulator states as input, not pixels; and World Models compresses frames with a VAE and trains a recurrent latent model, but optimizes a small controller separately rather than planning online end to end with the dynamics.

I propose PlaNet, the Deep Planning Network: a latent dynamics model learned from pixel sequences, with online planning carried out entirely in latent space and no learned policy or value network in the core agent. The model is a latent state-space model, $p(s_t\mid s_{t-1},a_{t-1})$, $p(o_t\mid s_t)$, $p(r_t\mid s_t)$, where the observation likelihood supplies dense gradients from the image and the reward likelihood supplies exactly the signal the planner consumes; at plan time I roll forward only the compact transition and reward, never decoding an image. Because the state is latent I train with a variational filtering bound. Writing the observation likelihood as an expectation under the filtering posterior $q(s_t\mid o_{\le t},a_{<t})$ and applying Jensen gives
$$
\log p(o_{1:T}\mid a_{1:T})\ge
\sum_{t=1}^T\left(
\mathbb E_{q(s_t\mid o_{\le t},a_{<t})}[\log p(o_t\mid s_t)]
-\mathbb E_{q(s_{t-1}\mid o_{\le t-1},a_{<t-1})}
\left[\mathrm{KL}\left(q(s_t\mid o_{\le t},a_{<t})\,\|\,p(s_t\mid s_{t-1},a_{t-1})\right)\right]\right),
$$
with the reward contributing the analogous term $\mathbb E_q[\log p(r_t\mid s_t)]$. The sign matters: I maximize reconstruction and reward log-likelihood and minimize the posterior-to-prior KL, and with unit-variance Gaussian heads each negative log-likelihood is just a half squared error plus constants.

The load-bearing design choice is the latent dynamics itself. A purely stochastic Markov latent, sampled at every step, can represent uncertainty but has a memory problem: information from an early frame must survive a freshly sampled bottleneck again and again, and the only way it persists is the degenerate solution of near-zero variance on the memory dimensions, which is fragile. A purely deterministic recurrent latent has the opposite failure: it remembers well but offers a single future and discards the KL-regularized stochastic state entirely. PlaNet keeps both channels by splitting the state into a deterministic recurrent belief $h_t$ and a stochastic state $s_t$,
$$
h_t=f(h_{t-1},s_{t-1},a_{t-1}),\qquad
s_t\sim p(s_t\mid h_t),\qquad
o_t\sim p(o_t\mid h_t,s_t),\qquad
r_t\sim p(r_t\mid h_t,s_t),
$$
with the posterior reusing the same deterministic update and then folding in the current image embedding through $q(s_t\mid h_t,o_t)$. Here $h_t$ carries memory deterministically while $s_t$ carries uncertainty under KL regularization. A second, easy-to-miss choice protects this structure: all observation information must enter through the sampled posterior state before it reaches the decoder. If the image embedding were allowed to flow deterministically into reconstruction, the model would simply copy the current frame and the prior, which has to plan without images, would learn almost nothing. The feature vector consumed by both heads and by the planner is the concatenation $[s_t;h_t]$, which is not cosmetic: it means the reward head reads exactly the features that the transition can roll forward, so any action sequence can be scored by latent rollout alone, and the image head, though essential for learning, is never needed at plan time.

The standard filtering bound still has a gap for planning, because the transition is only ever trained with one-step KL, asked whether the next prior agrees with the next posterior; but a planner rolls the model forward many steps with no intervening observations, and for a finite-capacity model family the one-step optimum need not be the multi-step optimum. I close this with latent overshooting. Define the $d$-step predictive prior by repeatedly applying the one-step transition and integrating out the intermediate states,
$$
p(s_t\mid s_{t-d})=\int\prod_{\tau=t-d+1}^{t}p(s_\tau\mid s_{\tau-1})\,ds_{t-d+1:t-1}
=\mathbb E_{p(s_{t-1}\mid s_{t-d})}[p(s_t\mid s_{t-1})],
$$
with action conditioning understood. Substituting this $d$-step prior into the generative model and again pushing the log inside the recursion with Jensen yields a genuine lower bound on the $d$-step predictive distribution, in which the KL still runs from the informed posterior at $t$ to the one-step prior, but the conditioning state for that prior now comes from an open-loop rollout that started $d$ steps back. Since every expectation sits outside the KL, sample averages give an unbiased estimator. A single distance is not enough for a planner, so I average over $d=1,\dots,D$ with optional per-distance weights $\beta_d$,
$$
\frac1D\sum_{d=1}^D\log p_d(o_{1:T}\mid a_{1:T})\ge
\sum_t\left(
\mathbb E_q[\log p(o_t\mid s_t)]
-\frac1D\sum_{d=1}^D\beta_d\,
\mathbb E\left[\mathrm{KL}\left(q(s_t\mid o_{\le t},a_{<t})\,\|\,p(s_t\mid s_{t-1},a_{t-1})\right)\right]\right),
$$
and for $d>1$ I stop gradients through the posterior targets so that the open-loop predictions move toward the informed filtering states rather than dragging those states down toward weaker rollouts. I am careful not to overstate the connection to ordinary likelihood: the data-processing relation $I(s_t;s_{t-d})\le I(s_t;s_{t-1})$ that would order the objectives is only a conjecture in expectation, not a pointwise theorem, so the guaranteed object is the lower bound on the $d$-step predictive distribution itself. With the mixed deterministic-stochastic state this regularizer is strong enough to be optional, and the released default leaves the overshooting scale at zero.

For action selection I use model-predictive control with a derivative-free population optimizer in latent space. At each environment step I infer the current belief from the current image and previous action, then run the cross-entropy method over open-loop action sequences: I initialize a time-dependent diagonal Gaussian over $H=12$ future actions at zero mean and unit variance, sample $J=1000$ candidate sequences, clip them to $[-1,1]$, roll the RSSM prior forward in latent space, score each sequence by the sum of predicted mean rewards, keep the best $K=100$, and refit the Gaussian, repeating for $I=10$ iterations. I then execute only the first mean action and discard the rest, because the next observation will update the belief, and I reset to a fresh zero-mean, unit-variance search distribution each step so the next search is not anchored to a stale local optimum. The training loop co-evolves data and model: I seed the replay buffer with five random episodes, train on uniformly sampled $50\times50$ sequence chunks with Adam, and periodically collect new episodes with the current planner plus $\mathcal N(0,0.3)$ action noise, repeating actions per task to shorten the effective horizon and smooth the visual dynamics. The primary algorithm writes the CEM scale refit as a mean absolute deviation $\frac{1}{K-1}\sum_{k\in\mathcal K}|a^{(k)}-\mu|$ over the elite actions, but to match the canonical implementation byte-for-byte the code below uses the elite-action variance, `stddev = sqrt(variance + 1e-6)`. A few other canonical details are load-bearing for a faithful run: free nats are implemented as `relu(KL - 3)` so the first three nats carry zero gradient rather than clamping the KL upward; the posterior computes the prior transition first and then combines its `belief` with the image embedding; the default loss scales are image 1, reward 10, divergence 1, and overshooting 0; and preprocessing floors 8-bit pixels to 5-bit bins, divides by 32, adds uniform dequantization noise in $[0,1/32)$, and subtracts $0.5$.

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
        hidden = dense(concat([prev["sample"], prev_action]), 200, relu)
        belief, rnn_state = gru_block_cell(hidden, prev["rnn_state"], 200)
        hidden = dense(belief, 200, relu)
        mean = dense(hidden, 30, None)
        stddev = softplus(dense(hidden, 30, None)) + 0.1
        sample = normal_diag(mean, stddev).sample()
        return dict(mean=mean, stddev=stddev, sample=sample,
                    belief=belief, rnn_state=rnn_state)

    def posterior(self, prev, prev_action, embed):
        prior = self.transition(prev, prev_action)
        hidden = dense(concat([prior["belief"], embed]), 200, relu)
        mean = dense(hidden, 30, None)
        stddev = softplus(dense(hidden, 30, None)) + 0.1
        sample = normal_diag(mean, stddev).sample()
        post = dict(mean=mean, stddev=stddev, sample=sample,
                    belief=prior["belief"], rnn_state=prior["rnn_state"])
        return prior, post

    def features(self, state):
        return concat([state["sample"], state["belief"]])

def model_loss(prior, post, targets, heads, free_nats=3.0):
    features = rssm.features(post)
    image_lp = heads["image"](features).log_prob(targets["image"])
    reward_lp = heads["reward"](features).log_prob(targets["reward"])
    divergence = kl_divergence(normal(post), normal(prior))
    div_loss = relu(divergence - free_nats)
    return -mean(image_lp) - 10.0 * mean(reward_lp) + mean(div_loss)

def cem_plan(current_state, cell, reward_head, horizon=12,
             amount=1000, topk=100, iterations=10):
    mean = zeros([horizon, action_dim])
    stddev = ones([horizon, action_dim])
    tiled = tile_state(current_state, amount)
    for _ in range(iterations):
        action = standard_normal([amount, horizon, action_dim])
        action = clip(action * stddev[None] + mean[None], -1.0, 1.0)
        states = rollout_prior(cell, tiled, action)
        returns = sum_over_time(reward_head(cell.features(states)).mean())
        best = gather(action, top_k_indices(returns, topk))
        mean = best.mean(axis=0)
        stddev = sqrt(((best - mean[None]) ** 2).mean(axis=0) + 1e-6)
    return mean[0]
```
