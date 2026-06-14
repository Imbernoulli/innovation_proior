**Problem.** The recurrent reward-decoder encoder sharpened *what* `z` encodes but gave the agent no
*exploration* mechanism: the single sampled `z` plus per-step action noise is undirected, so on
`sparse-point-robot` the agent only occasionally stumbles onto the rare +1 (mean 1.99, best seed 3.39). The
deeper need is a latent whose *uncertainty* the agent can act on — a belief it can sample a hypothesis from
and pursue coherently for a whole episode.

**Key idea.** Make the task belief a genuine **probabilistic posterior** `q(z|c)` and enable
**posterior-sampling exploration**: sample a task hypothesis, hold `z` fixed for an episode, act optimally
for it (temporally-extended, committed exploration), observe, update the belief, repeat. The context is an
unordered *set* of independent transitions, so each transition emits a Gaussian factor `N(μ_n, σ_n²)` and the
belief is their **product of Gaussians** — closed form, precisions add (`1/σ²=Σ1/σ_n²`,
`μ=σ²·Σμ_n/σ_n²`), so it is permutation-invariant *and* sharpens with evidence (Bayesian filtering). Train
the encoder with an **information-bottleneck KL** to `N(0,I)` (a variational bound on `I(z;c)` that squeezes
`z` to the minimal task-relevant statistic — crucial under the short budget, and the prior is the
exploration prior) plus the **critic's Bellman gradient** (so `z` is shaped to make the value accurate).

**Why (and why this is a faithful fill).** Unlike the previous two rungs, none of the method's machinery is
dropped — the probabilistic belief, the bottleneck, the posterior-sampling exploration, and the **decoupled
samplers** all fit the harness directly. The decoupling is the load-bearing piece: the matching principle
binds only the encoder's *input*, so the actor/critic learn off-policy from decorrelated minibatches over the
*entire* buffer while the encoder's context is drawn by a *separate* sampler from *recently collected* data
(the separate encoder buffer, cleared per task), distinct from the RL batch. Collection mixes prior- and
posterior-conditioned exploration. `z` carries gradients into the encoder through the Q-loss and the KL, and
is detached in the value and policy losses. The reward decoder is gone; the GRU is gone (a probabilistic
belief wants independent evidence *fused*, not sequentially summarized).

**Hyperparameters.** `latent_dim=5`; encoder 3-layer MLP hidden 200 → `2·latent_dim` (mean + pre-softplus
var); actor/critic net size 300, twin Q + V + slow target, `τ=5e-3`; Adam lr `3e-4`; per-env
`discount`/`reward_scale`; `kl_lambda=β=0.1` (1.0 on sparse via config); meta-batch 16; RL batch 256;
embedding/context batch from the env config; policy mean/std reg `1e-3`.

**What to watch.** Sharp prediction on the sparse task: posterior sampling should roughly double
`sparse-point-robot` above VariBAD's 1.99 (best seed pushing well past 3.39) — the signature of an agent
that commits to sweeping the half-circle. The bottlenecked critic-trained `z` should *improve* `cheetah-vel`
below −83.9 (look for the low-to-mid −60s). Explicit bet: PEARL may *regress* on the easy dense `point-robot`
(VariBAD's recurrent encoder won it at −10.1; product-of-Gaussians discards the sequence) — losing a little
on the easiest family to win decisively on the two that exercise meta-RL.

```python
# EDITABLE region of custom_meta_rl.py — step 3: PEARL (product-of-Gaussians belief + posterior sampling, off-policy SAC)
from rlkit.torch.distributions import TanhNormal


def _product_of_gaussians(mus, sigmas_squared):
    sigmas_squared = torch.clamp(sigmas_squared, min=1e-7)
    sigma_squared = 1. / torch.sum(torch.reciprocal(sigmas_squared), dim=0)
    mu = sigma_squared * torch.sum(mus / sigmas_squared, dim=0)
    return mu, sigma_squared


class CustomMetaRLAgent(nn.Module):
    """PEARL agent with product-of-Gaussians context encoder."""

    def __init__(self, obs_dim, action_dim, latent_dim=5, net_size=300,
                 reward_dim=1, use_next_obs_in_context=False, **kwargs):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.use_next_obs_in_context = use_next_obs_in_context
        self.sparse_rewards = kwargs.get('sparse_rewards', False)

        # Context encoder input: (obs, action, reward [, next_obs])
        context_input_dim = obs_dim + action_dim + reward_dim
        if use_next_obs_in_context:
            context_input_dim += obs_dim
        context_output_dim = latent_dim * 2  # mean + logvar for IB

        # Context encoder: 3-layer MLP
        self.context_encoder = build_mlp(
            context_input_dim, context_output_dim,
            hidden_dim=200, n_layers=3,
        )
        self.context_encoder_output_size = context_output_dim

        # Policy, Q-functions, V-function (z-conditioned)
        self.policy = build_policy(obs_dim, action_dim, latent_dim, net_size)
        self.qf1 = build_qf(obs_dim, action_dim, latent_dim, net_size)
        self.qf2 = build_qf(obs_dim, action_dim, latent_dim, net_size)
        self.vf = build_vf(obs_dim, latent_dim, net_size)
        self.target_vf = copy.deepcopy(self.vf)

        # z distribution
        self.register_buffer('z', torch.zeros(1, latent_dim))
        self.register_buffer('z_means', torch.zeros(1, latent_dim))
        self.register_buffer('z_vars', torch.ones(1, latent_dim))
        self._context = None

    def clear_context(self, num_tasks=1):
        mu = ptu.zeros(num_tasks, self.latent_dim)
        var = ptu.ones(num_tasks, self.latent_dim)
        self.z_means = mu
        self.z_vars = var
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
        o = ptu.from_numpy(o[None, None, ...])
        a = ptu.from_numpy(a[None, None, ...])
        r = ptu.from_numpy(np.array([r])[None, None, ...])
        no = ptu.from_numpy(no[None, None, ...])
        if self.use_next_obs_in_context:
            data = torch.cat([o, a, r, no], dim=2)
        else:
            data = torch.cat([o, a, r], dim=2)
        if self._context is None:
            self._context = data
        else:
            self._context = torch.cat([self._context, data], dim=1)

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
                      for m, s in zip(torch.unbind(self.z_means),
                                      torch.unbind(self.z_vars))]
        z = [d.rsample() for d in posteriors]
        self.z = torch.stack(z)

    def adapt(self):
        if self._context is not None:
            self.infer_posterior(self._context)

    def compute_kl_div(self):
        prior = torch.distributions.Normal(
            ptu.zeros(self.latent_dim), ptu.ones(self.latent_dim))
        posteriors = [torch.distributions.Normal(mu, torch.sqrt(var))
                      for mu, var in zip(torch.unbind(self.z_means),
                                         torch.unbind(self.z_vars))]
        kl_divs = [torch.distributions.kl.kl_divergence(post, prior)
                   for post in posteriors]
        return torch.sum(torch.stack(kl_divs))

    def get_action(self, obs, deterministic=False):
        z = self.z
        obs_t = ptu.from_numpy(obs[None])
        in_ = torch.cat([obs_t, z], dim=1)
        return self.policy.get_action(in_, deterministic=deterministic)

    def forward(self, obs, context):
        self.infer_posterior(context)
        self.sample_z()
        task_z = self.z
        t, b, _ = obs.size()
        obs = obs.view(t * b, -1)
        task_z = [z.repeat(b, 1) for z in task_z]
        task_z = torch.cat(task_z, dim=0)
        in_ = torch.cat([obs, task_z.detach()], dim=1)
        policy_outputs = self.policy(
            in_, reparameterize=True, return_log_prob=True)
        return policy_outputs, task_z

    def set_num_steps_total(self, n):
        self.policy.set_num_steps_total(n)

    def detach_z(self):
        self.z = self.z.detach()

    @property
    def networks(self):
        return [self.policy, self.qf1, self.qf2,
                self.vf, self.target_vf]


class CustomMetaRLAlgorithm:
    """PEARL SAC meta-training algorithm."""

    def __init__(self, agent, env, train_tasks, eval_tasks,
                 replay_buffer, enc_replay_buffer, config):
        self.agent = agent
        self.env = env
        self.train_tasks = train_tasks
        self.eval_tasks = eval_tasks
        self.replay_buffer = replay_buffer
        self.enc_replay_buffer = enc_replay_buffer
        self.config = config

        self.sampler = InPlacePathSampler(
            env=env, policy=agent,
            max_path_length=config['max_path_length'],
        )

        # Hyperparameters
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

        # Optimizers
        lr = 3e-4
        self.policy_optimizer = optim.Adam(agent.policy.parameters(), lr=lr)
        self.qf1_optimizer = optim.Adam(agent.qf1.parameters(), lr=lr)
        self.qf2_optimizer = optim.Adam(agent.qf2.parameters(), lr=lr)
        self.vf_optimizer = optim.Adam(agent.vf.parameters(), lr=lr)
        self.context_optimizer = optim.Adam(
            agent.context_encoder.parameters(), lr=lr)

    def collect_initial_data(self):
        num_initial_steps = self.config.get('num_initial_steps', 200)
        for idx in self.train_tasks:
            self.env.reset_task(idx)
            collect_data(
                self.agent, self.env, self.sampler,
                self.replay_buffer, self.enc_replay_buffer,
                idx, num_initial_steps, 1, np.inf,
                add_to_enc_buffer=True, config=self.config,
            )

    def train_iteration(self, iteration_idx):
        # --- Collect data from sampled tasks ---
        for i in range(self.num_tasks_sample):
            idx = np.random.randint(len(self.train_tasks))
            self.env.reset_task(idx)
            self.enc_replay_buffer.task_buffers[idx].clear()
            if self.num_steps_prior > 0:
                collect_data(
                    self.agent, self.env, self.sampler,
                    self.replay_buffer, self.enc_replay_buffer,
                    idx, self.num_steps_prior, 1, np.inf,
                    config=self.config,
                )
            if self.num_steps_posterior > 0:
                collect_data(
                    self.agent, self.env, self.sampler,
                    self.replay_buffer, self.enc_replay_buffer,
                    idx, self.num_steps_posterior, 1, self.update_post_train,
                    config=self.config,
                )
            if self.num_extra_rl_steps_posterior > 0:
                collect_data(
                    self.agent, self.env, self.sampler,
                    self.replay_buffer, self.enc_replay_buffer,
                    idx, self.num_extra_rl_steps_posterior, 1,
                    self.update_post_train,
                    add_to_enc_buffer=False, config=self.config,
                )

        # --- Meta-gradient updates ---
        for _ in range(self.num_train_steps_per_itr):
            indices = np.random.choice(
                self.train_tasks, self.meta_batch)
            self._take_step(indices)
        return {}

    def _take_step(self, indices):
        mb_size = self.embedding_mini_batch_size
        num_updates = self.embedding_batch_size // mb_size

        context_batch = sample_context_from_buffer(
            self.enc_replay_buffer, indices, self.embedding_batch_size,
            sparse_rewards=self.sparse_rewards,
            use_next_obs_in_context=self.use_next_obs_in_context,
        )
        self.agent.clear_z(num_tasks=len(indices))

        for i in range(num_updates):
            context = context_batch[:, i * mb_size: i * mb_size + mb_size, :]
            self._update(indices, context)
            self.agent.detach_z()

    def _update(self, indices, context):
        num_tasks = len(indices)
        obs, actions, rewards, next_obs, terms = sample_sac_batch(
            self.replay_buffer, indices, self.batch_size)

        # Forward pass through agent
        policy_outputs, task_z = self.agent(obs, context)
        new_actions, policy_mean, policy_log_std, log_pi = policy_outputs[:4]

        t, b, _ = obs.size()
        obs_flat = obs.view(t * b, -1)
        actions_flat = actions.view(t * b, -1)
        next_obs_flat = next_obs.view(t * b, -1)

        # Q and V predictions
        q1_pred = self.agent.qf1(obs_flat, actions_flat, task_z)
        q2_pred = self.agent.qf2(obs_flat, actions_flat, task_z)
        v_pred = self.agent.vf(obs_flat, task_z.detach())
        with torch.no_grad():
            target_v = self.agent.target_vf(next_obs_flat, task_z)

        # KL loss
        self.context_optimizer.zero_grad()
        kl_div = self.agent.compute_kl_div()
        kl_loss = self.kl_lambda * kl_div
        kl_loss.backward(retain_graph=True)

        # Q-function loss
        self.qf1_optimizer.zero_grad()
        self.qf2_optimizer.zero_grad()
        rewards_flat = rewards.view(self.batch_size * num_tasks, -1)
        rewards_flat = rewards_flat * self.reward_scale
        terms_flat = terms.view(self.batch_size * num_tasks, -1)
        q_target = rewards_flat + (1. - terms_flat) * self.discount * target_v
        qf_loss = (torch.mean((q1_pred - q_target) ** 2) +
                   torch.mean((q2_pred - q_target) ** 2))
        qf_loss.backward()
        self.qf1_optimizer.step()
        self.qf2_optimizer.step()
        self.context_optimizer.step()

        # V-function loss
        min_q = torch.min(
            self.agent.qf1(obs_flat, new_actions, task_z.detach()),
            self.agent.qf2(obs_flat, new_actions, task_z.detach()),
        )
        v_target = min_q - log_pi
        vf_loss = F.mse_loss(v_pred, v_target.detach())
        self.vf_optimizer.zero_grad()
        vf_loss.backward()
        self.vf_optimizer.step()
        ptu.soft_update_from_to(
            self.agent.vf, self.agent.target_vf, self.soft_target_tau)

        # Policy loss
        policy_loss = (log_pi - min_q).mean()
        mean_reg = 1e-3 * (policy_mean ** 2).mean()
        std_reg = 1e-3 * (policy_log_std ** 2).mean()
        policy_loss = policy_loss + mean_reg + std_reg
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

    @property
    def networks(self):
        return self.agent.networks
```
