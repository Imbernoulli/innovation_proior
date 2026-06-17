# VariBAD, distilled

VariBAD (variational Bayes-Adaptive Deep RL) meta-learns approximately Bayes-optimal behaviour for
fast adaptation to unseen tasks. It represents an unknown task by a small stochastic latent `m`,
infers an online posterior `q_phi(m | tau_{:t})` over `m` with a recurrent amortised encoder, trains
that encoder with a VAE-style ELBO whose decoder reconstructs the reward and transition of the
*entire* trajectory (future included), and conditions the policy on the posterior *distribution*
(mean and variance) so it can trade off exploration and exploitation as a function of its own task
uncertainty. The whole thing is trained end-to-end by RL plus the ELBO; at test time it adapts with
forward passes only — no gradient steps, no privileged task labels, no belief-space planning.

## Problem it solves

Meta-RL: given a distribution `p(M)` over related MDPs (shared `S, A, gamma, H`, task-specific
reward `R_i` and transition `T_i`), produce an agent that maximises **online return during learning**
on held-out tasks. The optimal behaviour is the Bayes-optimal policy of the Bayes-Adaptive MDP
(BAMDP): condition on the belief `b_t = p(R,T|tau_{:t})` over the task and explore only insofar as it
raises expected return within the horizon. Solving the BAMDP exactly is intractable in three places —
the parameterisation of `R,T`, the belief update, and belief-space planning — and prior methods either
don't scale, don't represent uncertainty, or need privileged task labels.

## Key idea

1. **Latent task variable.** Replace the belief over `(R,T)` (millions of params) with a belief over a
   low-dim stochastic latent `m`: `R_i(.) ~= R(.; m_i)`, `T_i(.) ~= T(.; m_i)` with `R, T` shared
   across tasks. Belief over the task collapses to a Gaussian over `m`.
2. **Amortised online inference.** A recurrent (GRU) encoder maps the past trajectory `tau_{:t}` to a
   diagonal-Gaussian posterior `q_phi(m|tau_{:t}) = N(mu_t, sigma_t^2)`, updated one transition per
   step. A single forward pass yields the belief — fast enough to run online at every timestep.
3. **Encode the past, decode past AND future.** The decoder reconstructs the reward and transition of
   the *whole* trajectory `tau_{:H^+}` from an `m` inferred off the prefix `tau_{:t}` — forcing `m` to
   capture the task (so it generalises to unseen states), not merely compress the seen history. This
   is a Jaderberg-style auxiliary loss on top of RL.
4. **ELBO at every context length, prior = previous posterior.** Train the ELBO summed over all `t`,
   so inference is online and the belief sharpens with data; set each KL's prior to the previous
   posterior `q_phi(m|tau_{:t-1})` (initial prior `N(0,I)`), giving a Bayesian-filter chain that
   penalises changing one's mind without evidence.
5. **Policy conditions on the posterior, not a sample.** `pi(a_t | s_t, q(m|tau_{:t}))` — feed both
   `mu_t` and `log sigma_t^2`. Acting on the *distribution* gives Bayes-optimal-style behaviour;
   feeding a sample would reduce to inefficient posterior sampling. This is the BAMDP hyper-state with
   the learned latent belief as `b_t`.
6. **Don't backprop RL through the encoder.** Train the VAE (encoder + decoders) and the policy with
   separate optimisers, learning rates, and data buffers; detach the latent before the policy. Cheaper
   (no embedding recompute per on-policy minibatch) and avoids RL/VAE gradient interference.

Removing the decoder and the ELBO recovers **RL²** (a recurrent policy fed `(s,a,r)`); the remaining
differences — the stochastic latent (represents uncertainty) and the future-reconstruction auxiliary
loss — are the inductive biases VariBAD adds.

## Objective

The model-learning objective `max_theta,phi E_rho[ log p_theta(tau_{:H^+}) ]` is intractable, so
maximise, for each context length `t`, the ELBO

```
log p_theta(tau_{:H^+}) >= E_{q_phi(m|tau_{:t})}[ log p_theta(tau_{:H^+} | m) ] - KL( q_phi(m|tau_{:t}) || p_theta(m) ) = ELBO_t,
```

derived from the log-marginal by Jensen and `p(tau,m) = p(tau|m) p(m)`. The reconstruction factorises
over time:

```
log p(tau_{:H^+} | m, a) = log p(s_0|m) + sum_i [ log p(s_{i+1}|s_i,a_i,m) + log p(r_{i+1}|s_i,a_i,s_{i+1},m) ].
```

The overall objective combines RL return and the ELBO summed over all `t` (weight `lambda`; encoder
params `phi` are shared):

```
L(phi, theta, psi) = E_{p(M)}[ J(psi, phi) + lambda * sum_{t=0}^{H^+} ELBO_t(phi, theta) ].
```

With diagonal Gaussians and prior = previous posterior, each chained KL term is the full
Gaussian-to-Gaussian divergence

```
KL( N(mu_t,S_t) || N(mu_{t-1},S_{t-1}) ) = 0.5 [ log|S_{t-1}|/|S_t| - K + tr(S_{t-1}^{-1} S_t) + (mu_{t-1}-mu_t)^T S_{t-1}^{-1} (mu_{t-1}-mu_t) ];
```

The first empty-history posterior is regularised against the unit Gaussian `N(0,I)`. If all KL terms
are instead taken to that fixed unit prior, the formula reduces to
`KL(q||N(0,I)) = -0.5 sum_k (1 + log sigma_k^2 - mu_k^2 - sigma_k^2)`.

## Defaults (MuJoCo, from the canonical setup)

RL algorithm PPO (A2C in the gridworld); latent dim 5; KL weight in the ELBO `lambda_KL = 0.1`;
policy LR 7e-4, VAE LR 1e-3; GRU hidden size 128; encoder embeds states/actions/rewards (32/16/16)
then GRU then a 5-d output; reward decoder 2 hidden layers, MSE (deterministic) reconstruction.
Reward-only decoder is used on the continuous-control families (even where dynamics change).

## Working code

Faithful to the canonical implementation (RNN encoder + reward/transition decoders + ELBO + policy on
the posterior). Encoder reads out a per-step Gaussian posterior; a full-trajectory pass gives
`q(m|tau_{:t})` for every prefix `t` with the prior prepended:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class RNNEncoder(nn.Module):
    """q_phi(m | tau_:t): embed (a,s,r), run a GRU online, read out (mu, logvar)."""

    def __init__(self, latent_dim=5, hidden_size=128, action_dim=2, state_dim=2,
                 a_embed=16, s_embed=32, r_embed=16):
        super().__init__()
        self.latent_dim, self.hidden_size = latent_dim, hidden_size
        self.action_encoder = nn.Linear(action_dim, a_embed)
        self.state_encoder = nn.Linear(state_dim, s_embed)
        self.reward_encoder = nn.Linear(1, r_embed)
        self.gru = nn.GRU(a_embed + s_embed + r_embed, hidden_size)
        self.fc_mu = nn.Linear(hidden_size, latent_dim)
        self.fc_logvar = nn.Linear(hidden_size, latent_dim)

    def reparameterise(self, mu, logvar):
        return mu + torch.exp(0.5 * logvar) * torch.randn_like(logvar)

    def prior(self, batch_size):
        h = torch.zeros(1, batch_size, self.hidden_size, device=self.fc_mu.weight.device)
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        return self.reparameterise(mu, logvar), mu, logvar, h

    def forward(self, actions, states, rewards, hidden_state=None, return_prior=True):
        if return_prior:
            prior_sample, prior_mu, prior_logvar, hidden_state = self.prior(actions.shape[1])
        h = torch.cat((F.relu(self.action_encoder(actions)),
                       F.relu(self.state_encoder(states)),
                       F.relu(self.reward_encoder(rewards))), dim=-1)
        out, _ = self.gru(h, hidden_state)
        mu, logvar = self.fc_mu(out), self.fc_logvar(out)
        sample = self.reparameterise(mu, logvar)
        recurrent_state = out
        if return_prior:
            sample = torch.cat((prior_sample, sample))
            mu = torch.cat((prior_mu, mu))
            logvar = torch.cat((prior_logvar, logvar))
            recurrent_state = torch.cat((hidden_state, recurrent_state))
        return sample, mu, logvar, recurrent_state


class RewardDecoder(nn.Module):
    """R'(r | s, a, s'; m): the auxiliary signal that forces m to encode the task."""

    def __init__(self, latent_dim, state_dim, action_dim, hidden=64, s_embed=32, a_embed=16):
        super().__init__()
        self.state_encoder = nn.Linear(state_dim, s_embed)
        self.action_encoder = nn.Linear(action_dim, a_embed)
        self.net = nn.Sequential(
            nn.Linear(latent_dim + 2 * s_embed + a_embed, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, 1))

    def forward(self, latent, prev_state, next_state, action):
        h = torch.cat((latent,
                       F.relu(self.state_encoder(next_state)),
                       F.relu(self.state_encoder(prev_state)),
                       F.relu(self.action_encoder(action))), dim=-1)
        return self.net(h)


class StateTransitionDecoder(nn.Module):
    """T'(s' | s, a; m)."""

    def __init__(self, latent_dim, state_dim, action_dim, hidden=64, s_embed=32, a_embed=16):
        super().__init__()
        self.state_encoder = nn.Linear(state_dim, s_embed)
        self.action_encoder = nn.Linear(action_dim, a_embed)
        self.net = nn.Sequential(
            nn.Linear(latent_dim + s_embed + a_embed, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, state_dim))

    def forward(self, latent, state, action):
        h = torch.cat((latent, F.relu(self.state_encoder(state)),
                       F.relu(self.action_encoder(action))), dim=-1)
        return self.net(h)
```

The negative-ELBO loss: encode once (posterior at every prefix `t`), decode the whole trajectory under
each `q(m|tau_{:t})`, sum reconstruction over decode steps and over `t`, add the chained KL:

```python
def kl_chained_to_previous(mu, logvar):
    """KL terms for t=0..T. mu/logvar are [T+1, batch, latent]:
    index 0 is the empty-history posterior, regularised to N(0,I);
    later indices are regularised to the previous posterior."""
    unit_mu = torch.zeros_like(mu[:1])
    unit_logvar = torch.zeros_like(logvar[:1])
    all_mu = torch.cat((unit_mu, mu), dim=0)
    all_logvar = torch.cat((unit_logvar, logvar), dim=0)
    mu_t, logvar_t = all_mu[1:], all_logvar[1:]
    mu_p, logvar_p = all_mu[:-1], all_logvar[:-1]
    var_t, var_p = logvar_t.exp(), logvar_p.exp()
    return 0.5 * ((logvar_p - logvar_t)
                  + (var_t + (mu_p - mu_t).pow(2)) / var_p - 1.0).sum(dim=-1)


def compute_elbo_loss(encoder, reward_decoder, state_decoder,
                      prev_obs, next_obs, actions, rewards,
                      kl_weight=0.1, rew_coeff=1.0, state_coeff=1.0, decode_state=True):
    _, mu, logvar, _ = encoder(actions, next_obs, rewards, hidden_state=None, return_prior=True)
    samples = encoder.reparameterise(mu, logvar)
    T = next_obs.shape[0]
    rew_loss = state_loss = 0.0
    for t in range(samples.shape[0]):                 # sum_t ELBO_t
        m_t = samples[t].unsqueeze(0).expand(T, -1, -1)   # decode whole trajectory incl. future
        rp = reward_decoder(m_t, prev_obs, next_obs, actions)
        rew_loss = rew_loss + (rp - rewards).pow(2).mean(-1).sum(0).mean()
        if decode_state:
            sp = state_decoder(m_t, prev_obs, actions)
            state_loss = state_loss + (sp - next_obs).pow(2).mean(-1).sum(0).mean()
    kl_loss = kl_chained_to_previous(mu, logvar).sum(0).mean()
    return rew_coeff * rew_loss + state_coeff * state_loss + kl_weight * kl_loss
```

The agent conditions the policy on the detached posterior distribution, and meta-training keeps the
RL and VAE objectives on separate optimisers/buffers:

```python
class VariBADAgent(nn.Module):
    def __init__(self, state_dim, action_dim, latent_dim=5, hidden=128):
        super().__init__()
        self.encoder = RNNEncoder(latent_dim, hidden, action_dim, state_dim)
        self.reward_decoder = RewardDecoder(latent_dim, state_dim, action_dim)
        self.state_decoder = StateTransitionDecoder(latent_dim, state_dim, action_dim)
        self.policy = build_policy(state_dim, 2 * latent_dim, action_dim, hidden)  # state + (mu,logvar)
        self.reset_belief(1)

    def reset_belief(self, batch_size):
        _, self.mu, self.logvar, self.hidden = self.encoder.prior(batch_size)

    def update_belief(self, action, next_state, reward):
        _, mu, logvar, hidden = self.encoder(action[None], next_state[None], reward[None],
                                             hidden_state=self.hidden, return_prior=False)
        self.mu, self.logvar = mu[-1], logvar[-1]
        self.hidden = hidden[-1:].detach()

    def act(self, state, deterministic=False):
        belief = torch.cat((self.mu, self.logvar), dim=-1).detach()   # posterior, detached from RL
        return self.policy(torch.cat((state, belief), dim=-1), deterministic=deterministic)


def meta_train(agent, task_distribution, rl_algorithm, num_iters, lr_vae=1e-3, kl_weight=0.1):
    vae_params = (list(agent.encoder.parameters())
                  + list(agent.reward_decoder.parameters())
                  + list(agent.state_decoder.parameters()))
    vae_optimiser = torch.optim.Adam(vae_params, lr=lr_vae)
    policy_buffer, vae_buffer = RecentBuffer(), TrajectoryBuffer()
    for _ in range(num_iters):
        trajectories = rollout(agent, task_distribution.sample_batch())   # belief updated online
        policy_buffer.add(trajectories); vae_buffer.add(trajectories)
        rl_algorithm.update(agent.policy, policy_buffer.recent())         # no RL grad to encoder
        prev_obs, next_obs, actions, rewards = vae_buffer.sample()
        vae_optimiser.zero_grad()
        loss = compute_elbo_loss(agent.encoder, agent.reward_decoder, agent.state_decoder,
                                 prev_obs, next_obs, actions, rewards, kl_weight=kl_weight)
        loss.backward()
        vae_optimiser.step()
```

At meta-test, roll out on held-out tasks with forward passes only: update the belief online per step,
act on the posterior, let the posterior contract — no gradient adaptation, no task labels, no
planning.
