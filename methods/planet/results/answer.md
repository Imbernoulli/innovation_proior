# PlaNet (Deep Planning Network)

## Problem

Control from $64\times64\times3$ images in a POMDP while using far fewer real
environment episodes than model-free visual-control agents. The policy is online
planning in a learned latent model; there is no learned policy network or value
network in the core agent.

## Method

The latent dynamics model is a recurrent state-space model with deterministic
memory $h_t$ and stochastic state $s_t$:
$$
h_t=f(h_{t-1},s_{t-1},a_{t-1}),\qquad
s_t\sim p(s_t\mid h_t),
$$
$$
o_t\sim p(o_t\mid h_t,s_t),\qquad
r_t\sim p(r_t\mid h_t,s_t),\qquad
q(s_t\mid h_t,o_t).
$$
The feature passed to image and reward heads is $[s_t;h_t]$. The deterministic
path preserves memory under partial observability; the stochastic path represents
uncertainty and supplies the KL-regularized latent variable. Observation
information enters the model through the sampled posterior state, avoiding a
deterministic reconstruction shortcut.

## Objective

For the observation likelihood, the standard filtering ELBO is
$$
\log p(o_{1:T}\mid a_{1:T})\ge
\sum_{t=1}^T\left(
\mathbb E_{q(s_t\mid o_{\le t},a_{<t})}[\log p(o_t\mid s_t)]
-\mathbb E_{q(s_{t-1}\mid o_{\le t-1},a_{<t-1})}
\left[\mathrm{KL}\left(q(s_t\mid o_{\le t},a_{<t})\|p(s_t\mid s_{t-1},a_{t-1})\right)\right]\right).
$$
Reward adds the analogous term $\mathbb E_q[\log p(r_t\mid h_t,s_t)]$. Under
unit-variance Gaussian heads, negative log-likelihoods are half squared errors
plus constants.

Latent overshooting defines the $d$-step prior
$$
p(s_t\mid s_{t-d})=
\int\prod_{\tau=t-d+1}^{t}p(s_\tau\mid s_{\tau-1})\,ds_{t-d+1:t-1},
$$
and applies a KL from the informed posterior at $t$ to the one-step prior whose
conditioning state came from a $d$-step open-loop rollout. Averaging
$d=1,\ldots,D$ gives
$$
\frac1D\sum_{d=1}^D\log p_d(o_{1:T}\mid a_{1:T})
\ge
\sum_t\left(
\mathbb E_q[\log p(o_t\mid s_t)]
-\frac1D\sum_{d=1}^D\beta_d\,
\mathbb E\left[\mathrm{KL}\left(q(s_t\mid o_{\le t},a_{<t})\|p(s_t\mid s_{t-1},a_{t-1})\right)\right]\right).
$$
The data-processing relation between $p_d$ and the one-step predictive
distribution is a conjectured expectation-level motivation, not a pointwise
guarantee. In practice, posterior targets for $d>1$ are stop-gradient targets.

## Planning

At each environment step, infer the current state from the current image and
previous action, then run CEM over action sequences:

- initialize a diagonal Gaussian over $H=12$ future actions with zero mean and
  unit variance;
- sample $J=1000$ candidate sequences, clip actions to $[-1,1]$, and roll the
  RSSM prior forward in latent space;
- score each sequence by the sum of predicted mean rewards;
- keep the best $K=100$ sequences and refit the Gaussian;
- repeat for $I=10$ iterations, execute the first mean action, then re-plan from
  a fresh zero-mean/unit-variance search distribution after the next observation.

The primary algorithm writes the CEM scale refit as
$\frac{1}{K-1}\sum_{k\in\mathcal K}|a^{(k)}-\mu|$. The canonical Google code
uses elite-action variance via moments and sets
`stddev = sqrt(variance + 1e-6)`.

## Constants And Code Checks

Primary-source hyperparameters: 30-dimensional diagonal Gaussian latent state,
200-unit GRU belief, image encoder/decoder from World Models, Adam learning rate
$10^{-3}$ with $\epsilon=10^{-4}$, gradient clipping norm 1000, batch shape
$50\times50$, 3 free nats, five random seed episodes, action noise
$\mathcal N(0,0.3)$, and action repeats cart-pole 8, reacher 4, cheetah 4,
finger 2, cup 4, walker 2.

Canonical code differences that matter for faithful implementation:

- image preprocessing floors 8-bit pixels to 5-bit bins, divides by 32, adds
  uniform dequantization noise, then subtracts 0.5;
- RSSM state is `mean`, `stddev`, `sample`, `belief`, `rnn_state`;
- posterior first computes the prior transition, then combines the resulting
  `belief` with the current image embedding;
- features are `concat(sample, belief)`;
- free nats are `max(0, KL - 3)`, not `max(KL, 3)`;
- default loss scales in the released code are image 1, reward 10, divergence 1,
  overshooting 0;
- released config defaults use ReLU activation, one RSSM model layer of width
  200, and a reward head network of 3 layers with 300 units unless overridden.

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
