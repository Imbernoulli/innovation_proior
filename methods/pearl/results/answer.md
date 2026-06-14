# PEARL, distilled

PEARL (Probabilistic Embeddings for Actor-critic RL) is an off-policy meta-RL algorithm that
disentangles task inference from control. It infers a probabilistic latent context variable `z`
from collected experience with a permutation-invariant variational encoder, conditions a
soft-actor-critic policy and critic on `z`, and meta-trains the actor-critic fully off-policy from a
replay buffer while training the encoder on recent, near-on-policy context. Because `z` is
probabilistic, test-time adaptation is posterior sampling: commit to a task hypothesis for an
episode (temporally extended exploration), update the belief, repeat.

## Problem it solves

Meta-RL with both meta-training and adaptation sample-efficiency: reuse off-policy data so
meta-training is cheap, and explore in a structured way so adaptation is fast even in sparse-reward
tasks. The obstacle is that meta-learning assumes the adaptation-data distribution matches between
meta-train and meta-test; test-time adaptation is on-policy, which naively forces on-policy
(inefficient) meta-training.

## Key ideas

- **Disentangle inference from control.** Make the task belief an explicit latent `z`; condition
  `π(a|s,z)` and `Q(s,a,z)` on it. The matching principle constrains only the encoder's input data,
  not the control updates. So train the actor-critic off-policy on the whole buffer; train the
  encoder on recent context drawn by a *separate* sampler.

- **Amortized variational inference with an information bottleneck.** Train `q_φ(z|c)` to a
  variational lower bound `E_T[ E_{z∼q}[R(T,z)] − β·KL(q_φ(z|c) ‖ p(z)) ]`, prior `p(z)=N(0,I)`.
  The KL is a variational bound on `I(Z;C)`: it squeezes `z` to a minimal task-relevant statistic,
  mitigating overfitting to training tasks. Reparameterize `z = μ + σ⊙ε`, `ε∼N(0,I)`.

- **Permutation-invariant probabilistic encoder = product of Gaussians.** The context is an
  unordered set of transitions, so each transition `c_n` emits a Gaussian factor
  `Ψ_φ(z|c_n)=N(f_φ^μ(c_n), f_φ^σ(c_n))` and the belief is their product
  `q_φ(z|c_{1:N}) ∝ ∏_n Ψ_φ(z|c_n)`. Closed form (diagonal, per coordinate):
  `σ² = 1/Σ_n(1/σ_n²)`, `μ = σ²·Σ_n(μ_n/σ_n²)` — precisions add, so the belief sharpens with more
  evidence (Bayesian filtering), and the product is permutation-invariant.

- **Train the encoder from the critic.** Empirically, optimizing `z` to recover the state-action
  value (Bellman/critic gradients) beats optimizing for actor returns or for state/reward
  reconstruction. So `z` gets gradients only from the Q-loss and the KL; it is detached in the value
  and policy losses.

- **Posterior sampling for exploration.** Sample `z` (from the prior, then the posterior), hold it
  fixed for an episode, act optimally for that sampled task hypothesis — temporally extended (deep)
  exploration — then update `q_φ(z|c)` and repeat. A deterministic `z` collapses this and fails on
  sparse-reward tasks.

- **Decoupled samplers.** Actor/critic: decorrelated transitions from the *entire* buffer
  (off-policy). Encoder context `S_c`: *recently* collected data, distinct from the RL batch. Fully
  off-policy context reintroduces train/test mismatch and hurts; sampling correlated trajectories for
  the RL batch hurts.

- **SAC backbone.** Maximum-entropy off-policy actor-critic: two Q-nets and their min (anti-
  overestimation), a value net with a slow target, a reparameterized squashed-Gaussian actor,
  reward scaled by the inverse entropy temperature.

## Losses (z appended to the state; `z̄` = detached)

```
L_critic = E_{(s,a,r,s')~B, z~q_φ(z|c)} [ ( Q(s,a,z) − ( r + V̄(s', z̄) ) )² ]   # encoder gets grads here
L_actor  = E_{s~B, a~π, z~q_φ(z|c)} [ D_KL( π(a|s, z̄) ‖ exp(Q(s,a,z̄))/𝒵(s) ) ]
L_KL     = β · KL( q_φ(z|c) ‖ N(0, I) )                                          # information bottleneck
```

## Meta-training (Algorithm)

```
Init replay buffers B^i per training task
while not done:
    # collect (separate context vs. RL data)
    for each sampled task T_i:
        clear T_i's encoder buffer            # keep context recent
        z ~ prior;       gather data, add to B^i and encoder buffer
        z ~ q_φ(z|c^i);  gather data, add to B^i (and encoder buffer)
        z ~ q_φ(z|c^i);  gather extra RL-only data, add to B^i (not encoder buffer)
    # meta-gradient steps
    for each training step:
        sample task minibatch; for each task:
            context c^i ~ S_c(B^i)            # recent, distinct from RL batch
            RL batch b^i ~ B^i                # decorrelated, whole buffer
            z ~ q_φ(z|c^i)  (reparameterized)
        L_KL.backward (retain graph) -> encoder
        L_critic.backward -> Q-nets; step Q1,Q2 and encoder optimizer
        L_vf (z detached) -> step V; soft-update target V̄
        L_actor (z detached) + mean/std reg -> step policy
```

## Meta-testing (posterior sampling)

```
c = {}
for k = 1..K:
    z ~ q_φ(z|c)                 # k=1: prior
    roll out π(a|s,z) for an episode (z fixed)  -> trajectory D_k
    c = c ∪ D_k                  # accumulate, posterior narrows
report return after ~2 trajectories aggregated, acting with deterministic z
```

## Defaults

`latent_dim=5`; encoder 3-layer MLP, hidden 200, out `2·latent_dim` (mean + pre-softplus var);
actor/critic net size 300; two Q-nets + V + slow target V, `τ=5e-3`; Adam, lr `3e-4`; `γ=0.99`;
reward scale `5.0` (= 1/entropy-temperature); `β=kl_lambda=0.1`; meta-batch 16; RL batch 256;
context/embedding batch 64; policy mean/std regularizer weights `1e-3`.

## Working code

Fills the `context_module` (a permutation-invariant product-of-Gaussians encoder) and the
meta-gradient step in the off-policy actor-critic harness.

```python
import copy
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import rlkit.torch.pytorch_util as ptu


def _product_of_gaussians(mus, sigmas_squared):
    """Fuse per-transition Gaussian factors: precisions add, natural params add."""
    sigmas_squared = torch.clamp(sigmas_squared, min=1e-7)
    sigma_squared = 1. / torch.sum(torch.reciprocal(sigmas_squared), dim=0)
    mu = sigma_squared * torch.sum(mus / sigmas_squared, dim=0)
    return mu, sigma_squared


class PEARLAgent(nn.Module):
    """Probabilistic permutation-invariant context encoder + z-conditioned SAC."""

    def __init__(self, obs_dim, action_dim, latent_dim=5, net_size=300,
                 reward_dim=1, use_next_obs_in_context=False, **kwargs):
        super().__init__()
        self.obs_dim, self.action_dim, self.latent_dim = obs_dim, action_dim, latent_dim
        self.use_next_obs_in_context = use_next_obs_in_context
        self.sparse_rewards = kwargs.get('sparse_rewards', False)

        context_input_dim = obs_dim + action_dim + reward_dim
        if use_next_obs_in_context:
            context_input_dim += obs_dim
        context_output_dim = latent_dim * 2          # per-coord mean + pre-softplus var
        self.context_encoder = build_mlp(context_input_dim, context_output_dim,
                                         hidden_dim=200, n_layers=3)
        self.context_encoder_output_size = context_output_dim

        self.policy = build_policy(obs_dim, action_dim, latent_dim, net_size)
        self.qf1 = build_qf(obs_dim, action_dim, latent_dim, net_size)
        self.qf2 = build_qf(obs_dim, action_dim, latent_dim, net_size)
        self.vf = build_vf(obs_dim, latent_dim, net_size)
        self.target_vf = copy.deepcopy(self.vf)

        self.register_buffer('z', torch.zeros(1, latent_dim))
        self.register_buffer('z_means', torch.zeros(1, latent_dim))
        self.register_buffer('z_vars', torch.ones(1, latent_dim))
        self._context = None

    def clear_context(self, num_tasks=1):
        self.z_means = ptu.zeros(num_tasks, self.latent_dim)   # reset belief to prior N(0,I)
        self.z_vars = ptu.ones(num_tasks, self.latent_dim)
        self.sample_z()
        self._context = None

    def clear_z(self, num_tasks=1):
        self.clear_context(num_tasks)

    @property
    def context(self):
        return self._context

    def update_context(self, inputs):
        o, a, r, no, d, info = inputs
        if self.sparse_rewards:
            r = info.get('sparse_reward', r)
        o = ptu.from_numpy(o[None, None, ...]); a = ptu.from_numpy(a[None, None, ...])
        r = ptu.from_numpy(np.array([r])[None, None, ...]); no = ptu.from_numpy(no[None, None, ...])
        data = torch.cat([o, a, r, no], dim=2) if self.use_next_obs_in_context \
            else torch.cat([o, a, r], dim=2)
        self._context = data if self._context is None \
            else torch.cat([self._context, data], dim=1)

    def infer_posterior(self, context):
        params = self.context_encoder(context)
        params = params.view(context.size(0), -1, self.context_encoder_output_size)
        mu = params[..., :self.latent_dim]
        sigma_squared = F.softplus(params[..., self.latent_dim:])
        z_params = [_product_of_gaussians(m, s)
                    for m, s in zip(torch.unbind(mu), torch.unbind(sigma_squared))]
        self.z_means = torch.stack([p[0] for p in z_params])
        self.z_vars = torch.stack([p[1] for p in z_params])
        self.sample_z()

    def sample_z(self):
        posteriors = [torch.distributions.Normal(m, torch.sqrt(s))
                      for m, s in zip(torch.unbind(self.z_means), torch.unbind(self.z_vars))]
        self.z = torch.stack([d.rsample() for d in posteriors])    # reparameterized

    def adapt(self):
        if self._context is not None:
            self.infer_posterior(self._context)

    def compute_kl_div(self):
        prior = torch.distributions.Normal(ptu.zeros(self.latent_dim), ptu.ones(self.latent_dim))
        posteriors = [torch.distributions.Normal(mu, torch.sqrt(var))
                      for mu, var in zip(torch.unbind(self.z_means), torch.unbind(self.z_vars))]
        kls = [torch.distributions.kl.kl_divergence(post, prior) for post in posteriors]
        return torch.sum(torch.stack(kls))

    def get_action(self, obs, deterministic=False):
        in_ = torch.cat([ptu.from_numpy(obs[None]), self.z], dim=1)
        return self.policy.get_action(in_, deterministic=deterministic)

    def forward(self, obs, context):
        self.infer_posterior(context); self.sample_z()
        task_z = self.z
        t, b, _ = obs.size()
        obs = obs.view(t * b, -1)
        task_z = torch.cat([z.repeat(b, 1) for z in task_z], dim=0)
        in_ = torch.cat([obs, task_z.detach()], dim=1)
        policy_outputs = self.policy(in_, reparameterize=True, return_log_prob=True)
        return policy_outputs, task_z

    def detach_z(self):
        self.z = self.z.detach()

    @property
    def networks(self):
        return [self.policy, self.qf1, self.qf2, self.vf, self.target_vf]


class PEARLSoftActorCritic:
    """Decoupled-sampler off-policy meta-training."""

    def __init__(self, agent, env, train_tasks, eval_tasks,
                 replay_buffer, enc_replay_buffer, config):
        self.agent, self.env = agent, env
        self.train_tasks, self.eval_tasks = train_tasks, eval_tasks
        self.replay_buffer, self.enc_replay_buffer = replay_buffer, enc_replay_buffer
        self.config = config
        self.sampler = InPlacePathSampler(env=env, policy=agent,
                                          max_path_length=config['max_path_length'])
        self.batch_size = config.get('batch_size', 256)
        self.meta_batch = config.get('meta_batch', 16)
        self.discount = config.get('discount', 0.99)
        self.reward_scale = config.get('reward_scale', 5.0)
        self.kl_lambda = config.get('kl_lambda', 0.1)
        self.soft_target_tau = config.get('soft_target_tau', 0.005)
        self.sparse_rewards = config.get('sparse_rewards', False)
        self.use_next_obs_in_context = config.get('use_next_obs_in_context', False)
        self.embedding_batch_size = config.get('embedding_batch_size', 64)
        self.embedding_mini_batch_size = config.get('embedding_mini_batch_size', 64)
        self.num_tasks_sample = config.get('num_tasks_sample', 5)
        self.num_steps_prior = config.get('num_steps_prior', 400)
        self.num_steps_posterior = config.get('num_steps_posterior', 0)
        self.num_extra_rl_steps_posterior = config.get('num_extra_rl_steps_posterior', 400)
        self.num_train_steps_per_itr = config.get('num_train_steps_per_itr', 2000)
        self.update_post_train = config.get('update_post_train', 1)

        lr = 3e-4
        self.policy_optimizer = optim.Adam(agent.policy.parameters(), lr=lr)
        self.qf1_optimizer = optim.Adam(agent.qf1.parameters(), lr=lr)
        self.qf2_optimizer = optim.Adam(agent.qf2.parameters(), lr=lr)
        self.vf_optimizer = optim.Adam(agent.vf.parameters(), lr=lr)
        self.context_optimizer = optim.Adam(agent.context_encoder.parameters(), lr=lr)

    def collect_initial_data(self):
        n = self.config.get('num_initial_steps', 200)
        for idx in self.train_tasks:
            self.env.reset_task(idx)
            collect_data(self.agent, self.env, self.sampler,
                         self.replay_buffer, self.enc_replay_buffer,
                         idx, n, 1, np.inf, add_to_enc_buffer=True, config=self.config)

    def train_iteration(self, iteration_idx):
        for _ in range(self.num_tasks_sample):
            idx = np.random.randint(len(self.train_tasks))
            self.env.reset_task(idx)
            self.enc_replay_buffer.task_buffers[idx].clear()
            if self.num_steps_prior > 0:
                collect_data(self.agent, self.env, self.sampler, self.replay_buffer,
                             self.enc_replay_buffer, idx, self.num_steps_prior, 1, np.inf,
                             config=self.config)
            if self.num_steps_posterior > 0:
                collect_data(self.agent, self.env, self.sampler, self.replay_buffer,
                             self.enc_replay_buffer, idx, self.num_steps_posterior, 1,
                             self.update_post_train, config=self.config)
            if self.num_extra_rl_steps_posterior > 0:
                collect_data(self.agent, self.env, self.sampler, self.replay_buffer,
                             self.enc_replay_buffer, idx, self.num_extra_rl_steps_posterior, 1,
                             self.update_post_train, add_to_enc_buffer=False, config=self.config)
        for _ in range(self.num_train_steps_per_itr):
            indices = np.random.choice(self.train_tasks, self.meta_batch)
            self._take_step(indices)
        return {}

    def _take_step(self, indices):
        mb = self.embedding_mini_batch_size
        num_updates = self.embedding_batch_size // mb
        context_batch = sample_context_from_buffer(
            self.enc_replay_buffer, indices, self.embedding_batch_size,
            sparse_rewards=self.sparse_rewards,
            use_next_obs_in_context=self.use_next_obs_in_context)
        self.agent.clear_z(num_tasks=len(indices))
        for i in range(num_updates):
            context = context_batch[:, i * mb: i * mb + mb, :]
            self._update(indices, context)
            self.agent.detach_z()

    def _update(self, indices, context):
        num_tasks = len(indices)
        obs, actions, rewards, next_obs, terms = sample_sac_batch(
            self.replay_buffer, indices, self.batch_size)
        policy_outputs, task_z = self.agent(obs, context)
        new_actions, policy_mean, policy_log_std, log_pi = policy_outputs[:4]

        t, b, _ = obs.size()
        obs_flat = obs.view(t * b, -1)
        actions_flat = actions.view(t * b, -1)
        next_obs_flat = next_obs.view(t * b, -1)

        q1_pred = self.agent.qf1(obs_flat, actions_flat, task_z)
        q2_pred = self.agent.qf2(obs_flat, actions_flat, task_z)
        v_pred = self.agent.vf(obs_flat, task_z.detach())
        with torch.no_grad():
            target_v = self.agent.target_vf(next_obs_flat, task_z)

        # KL (information bottleneck) -> encoder
        self.context_optimizer.zero_grad()
        kl_loss = self.kl_lambda * self.agent.compute_kl_div()
        kl_loss.backward(retain_graph=True)

        # critic (soft Bellman) -> Q-nets AND encoder
        self.qf1_optimizer.zero_grad(); self.qf2_optimizer.zero_grad()
        rewards_flat = rewards.view(self.batch_size * num_tasks, -1) * self.reward_scale
        terms_flat = terms.view(self.batch_size * num_tasks, -1)
        q_target = rewards_flat + (1. - terms_flat) * self.discount * target_v
        qf_loss = torch.mean((q1_pred - q_target) ** 2) + torch.mean((q2_pred - q_target) ** 2)
        qf_loss.backward()
        self.qf1_optimizer.step(); self.qf2_optimizer.step()
        self.context_optimizer.step()

        # value (z detached)
        min_q = torch.min(self.agent.qf1(obs_flat, new_actions, task_z.detach()),
                          self.agent.qf2(obs_flat, new_actions, task_z.detach()))
        v_target = min_q - log_pi
        vf_loss = F.mse_loss(v_pred, v_target.detach())
        self.vf_optimizer.zero_grad(); vf_loss.backward(); self.vf_optimizer.step()
        ptu.soft_update_from_to(self.agent.vf, self.agent.target_vf, self.soft_target_tau)

        # policy (z detached) + SAC mean/std regularizers
        policy_loss = (log_pi - min_q).mean()
        policy_loss = policy_loss + 1e-3 * (policy_mean ** 2).mean() \
                                  + 1e-3 * (policy_log_std ** 2).mean()
        self.policy_optimizer.zero_grad(); policy_loss.backward(); self.policy_optimizer.step()

    @property
    def networks(self):
        return self.agent.networks
```
