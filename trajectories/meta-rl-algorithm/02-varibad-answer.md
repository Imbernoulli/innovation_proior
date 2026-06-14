**Problem.** The deterministic mean encoder is reliable only on the dense low-dim `point-robot`; it is
unreliable on `cheetah-vel` (the target velocity is a *trend* across transitions a bag-of-transitions mean
cannot read) and fails on `sparse-point-robot` (almost all context rewards are 0, so the contrastive loss
has nothing to separate tasks by). I need an encoder that respects the *order* of experience and a training
signal that forces `z` to encode the task's *reward structure*, not just task identity geometry.

**Key idea.** Replace the permutation-invariant mean with a **recurrent (GRU) encoder**: embed each
transition, run the GRU one step per transition (one belief update per step), and read out a **stochastic
Gaussian latent** `(μ, logσ²)` from the hidden state. Shape it with a **reward decoder** — a small net
predicting reward from `(s, a, z)` — so reconstructing rewards forces `z` to encode the target velocity
(cheetah) and the goal location (sparse), the very quantities the previous encoder missed. The encoder loss
is an ELBO-flavored pair: `L_enc = λ·KL(q(z|c) ‖ N(0,I)) + reward_pred_loss`, the KL acting as an
information bottleneck under the short budget.

**Why (and what this harness reduces).** The clean formulation reconstructs the *whole trajectory (future
included)* from a latent inferred off each *prefix*, sums the ELBO over all context lengths with each KL's
prior set to the *previous posterior* (a Bayes-filter chain), uses a transition decoder too, and conditions
the policy on the *distribution* with the latent detached from RL. This harness cannot carry that and the
per-prefix partial-belief ELBO does not converge on `cheetah-vel` (a `z` from the first few transitions
cannot predict the velocity reward). So the fill reduces to: a **single posterior `z` from the last GRU
step** (not per-prefix), **KL to the fixed `N(0,I)`** (not chained), a **reward decoder only** (no
transition head) predicting the **SAC batch's** rewards, the policy conditioned on a **sampled `z`** (the
scaffold's SAC-on-`z` interface), and the encoder optimizer stepped **with the critic** so `z` also carries
Bellman gradients (detached only in the value/policy losses). Context is capped at `max_path_length` so the
GRU sees one coherent trajectory and its hidden state is not polluted across episode boundaries. What is
kept from the clean idea: the recurrent ordered encoder, the stochastic latent + KL bottleneck, and a
reward-reconstruction auxiliary loss.

**Hyperparameters.** `latent_dim=5`; GRU encoder hidden 200 with a 2-layer ReLU pre-MLP, linear `μ`/`logvar`
heads; reward decoder 200-hidden 3-layer MLP; actor/critic net size 300, twin Q + V + slow target, `τ=5e-3`;
Adam lr `3e-4` (encoder + decoder share one optimizer); `kl_lambda=0.1`, `reward_pred_weight=1.0`; per-env
`discount`/`reward_scale`; meta-batch 16; RL batch 256.

**What to watch.** Expect `cheetah-vel` to firm up — the −108 seed should vanish and the spread tighten as
reward reconstruction forces a consistent target-velocity `z`. Expect `sparse-point-robot` to lift well
above FOCAL's 0.30 with the dead 0.0 seeds rescued (success if mean clears ~1.5). Watch `point-robot` (was
−13.9) for overfitting under the heavier encoder. Open risk: there is still no *exploration* mechanism, so
sparse may sharpen inference without solving "you must reach the goal once" — pointing to a probabilistic
belief the agent can act on next.

```python
# EDITABLE region of custom_meta_rl.py — step 2: VariBAD (GRU encoder + reward-decoder ELBO aux, online SAC)
class GRUContextEncoder(nn.Module):
    """GRU-based context encoder for sequential processing."""

    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        # Pre-process transitions with MLP
        self.fc_pre = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        # Output mean and logvar for latent
        self.fc_mean = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        self.register_buffer('hidden', torch.zeros(1, 1, hidden_dim))

    def reset(self, num_tasks=1):
        self.hidden = self.hidden.new_zeros(1, num_tasks, self.hidden_dim)

    def forward(self, context, return_sequence=False):
        """context: (num_tasks, seq_len, feat_dim)."""
        t, s, f = context.size()
        x = context.reshape(t * s, f)
        x = self.fc_pre(x)
        x = x.view(t, s, -1)
        out, hn = self.gru(x, self.hidden)
        self.hidden = hn.detach()
        mean = self.fc_mean(out)
        logvar = self.fc_logvar(out)
        if return_sequence:
            return mean, logvar
        return mean[:, -1, :], logvar[:, -1, :]


class RewardDecoder(nn.Module):
    """Predict reward from (state, action, belief/z)."""

    def __init__(self, obs_dim, action_dim, latent_dim, hidden_dim=200):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + action_dim + latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs, action, z):
        return self.net(torch.cat([obs, action, z], dim=-1))


class CustomMetaRLAgent(nn.Module):
    """VariBAD agent: GRU encoder + reward decoder."""

    def __init__(self, obs_dim, action_dim, latent_dim=5, net_size=300,
                 reward_dim=1, use_next_obs_in_context=False, **kwargs):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.use_next_obs_in_context = use_next_obs_in_context
        self.sparse_rewards = kwargs.get('sparse_rewards', False)

        context_input_dim = obs_dim + action_dim + reward_dim
        if use_next_obs_in_context:
            context_input_dim += obs_dim

        self.encoder = GRUContextEncoder(
            context_input_dim, hidden_dim=200, latent_dim=latent_dim)
        self.reward_decoder = RewardDecoder(
            obs_dim, action_dim, latent_dim, hidden_dim=200)

        self.policy = build_policy(obs_dim, action_dim, latent_dim, net_size)
        self.qf1 = build_qf(obs_dim, action_dim, latent_dim, net_size)
        self.qf2 = build_qf(obs_dim, action_dim, latent_dim, net_size)
        self.vf = build_vf(obs_dim, latent_dim, net_size)
        self.target_vf = copy.deepcopy(self.vf)

        self.register_buffer('z', torch.zeros(1, latent_dim))
        self.register_buffer('z_means', torch.zeros(1, latent_dim))
        self.register_buffer('z_logvars', torch.zeros(1, latent_dim))
        self._context = None

    def clear_context(self, num_tasks=1):
        self.z = ptu.zeros(num_tasks, self.latent_dim)
        self.z_means = ptu.zeros(num_tasks, self.latent_dim)
        self.z_logvars = ptu.zeros(num_tasks, self.latent_dim)
        self.encoder.reset(num_tasks)
        self._context = None

    def clear_z(self, num_tasks=1):
        self.clear_context(num_tasks)

    def sample_z(self):
        std = torch.exp(0.5 * self.z_logvars)
        eps = torch.randn_like(std)
        self.z = self.z_means + eps * std

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
        self.encoder.reset(context.size(0))
        mean, logvar = self.encoder(context)
        self.z_means = mean
        self.z_logvars = logvar
        self.sample_z()

    def adapt(self):
        if self._context is not None:
            self.infer_posterior(self._context)

    def compute_kl_div(self):
        """KL(q(z|c) || N(0, I))."""
        kl = -0.5 * torch.sum(
            1 + self.z_logvars - self.z_means.pow(2) - self.z_logvars.exp())
        return kl

    def get_action(self, obs, deterministic=False):
        z = self.z
        obs_t = ptu.from_numpy(obs[None])
        in_ = torch.cat([obs_t, z], dim=1)
        return self.policy.get_action(in_, deterministic=deterministic)

    def forward(self, obs, context):
        self.infer_posterior(context)
        task_z = self.z
        t, b, _ = obs.size()
        obs = obs.view(t * b, -1)
        task_z_rep = [z.repeat(b, 1) for z in task_z]
        task_z_rep = torch.cat(task_z_rep, dim=0)
        in_ = torch.cat([obs, task_z_rep.detach()], dim=1)
        policy_outputs = self.policy(
            in_, reparameterize=True, return_log_prob=True)
        return policy_outputs, task_z_rep

    def set_num_steps_total(self, n):
        self.policy.set_num_steps_total(n)

    def detach_z(self):
        self.z = self.z.detach()

    @property
    def networks(self):
        return [self.policy, self.qf1, self.qf2,
                self.vf, self.target_vf, self.reward_decoder]


class CustomMetaRLAlgorithm:
    """VariBAD: SAC + ELBO (reward prediction + KL) loss."""

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

        self.batch_size = config.get('batch_size', 256)
        self.meta_batch = config.get('meta_batch', 16)
        self.discount = config.get('discount', 0.99)
        self.reward_scale = config.get('reward_scale', 5.0)
        self.kl_lambda = config.get('kl_lambda', 0.1)
        self.reward_pred_weight = 1.0
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
        encoder_params = list(agent.encoder.parameters()) + list(agent.reward_decoder.parameters())
        self.policy_optimizer = optim.Adam(agent.policy.parameters(), lr=lr)
        self.qf1_optimizer = optim.Adam(agent.qf1.parameters(), lr=lr)
        self.qf2_optimizer = optim.Adam(agent.qf2.parameters(), lr=lr)
        self.vf_optimizer = optim.Adam(agent.vf.parameters(), lr=lr)
        self.encoder_optimizer = optim.Adam(encoder_params, lr=lr)

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

        for _ in range(self.num_train_steps_per_itr):
            indices = np.random.choice(
                self.train_tasks, self.meta_batch)
            self._take_step(indices)
        return {}

    def _sample_ordered_context_from_buffer(self, indices, batch_size):
        """Sample chronological trajectory context for VariBAD.

        Cap batch_size at max_path_length so each task's context comes
        from a SINGLE trajectory. oyster's random_sequence concatenates
        independent trajectories without resetting the GRU hidden state
        at episode boundaries, polluting the per-step posterior used by
        the ELBO reward decoder.
        """
        if not hasattr(indices, '__iter__'):
            indices = [indices]
        max_path = self.config.get('max_path_length', batch_size)
        bsz = min(batch_size, max_path)
        batches = [ptu.np_to_pytorch_batch(
            self.enc_replay_buffer.random_batch(
                idx, batch_size=bsz, sequence=True))
            for idx in indices]
        context = [unpack_batch(batch, sparse_reward=self.sparse_rewards)
                   for batch in batches]
        context = [[x[i] for x in context] for i in range(len(context[0]))]
        context = [torch.cat(x, dim=0) for x in context]
        if self.use_next_obs_in_context:
            return torch.cat(context[:-1], dim=2)
        return torch.cat(context[:-2], dim=2)

    def _take_step(self, indices):
        mb_size = self.embedding_mini_batch_size
        num_updates = self.embedding_batch_size // mb_size

        context_batch = self._sample_ordered_context_from_buffer(
            indices, self.embedding_batch_size)
        self.agent.clear_z(num_tasks=len(indices))

        for i in range(num_updates):
            context = context_batch[:, i * mb_size: i * mb_size + mb_size, :]
            self._update(indices, context)
            self.agent.detach_z()

    def _sample_sac_batch_with_sparse(self, indices):
        """Sample RL batch; also return sparse rewards when needed."""
        batches = [ptu.np_to_pytorch_batch(
            self.replay_buffer.random_batch(idx, batch_size=self.batch_size))
            for idx in indices]
        unpacked = [unpack_batch(b) for b in batches]
        unpacked = [[x[i] for x in unpacked] for i in range(len(unpacked[0]))]
        unpacked = [torch.cat(x, dim=0) for x in unpacked]
        obs, actions, rewards, next_obs, terms = unpacked
        if self.sparse_rewards:
            sparse_r = torch.cat(
                [b['sparse_rewards'][None, ...] for b in batches], dim=0)
        else:
            sparse_r = rewards
        return obs, actions, rewards, next_obs, terms, sparse_r

    def _update(self, indices, context):
        num_tasks = len(indices)
        obs, actions, rewards, next_obs, terms, _ = \
            self._sample_sac_batch_with_sparse(indices)

        # Forward: encode context and get policy outputs
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

        # --- Encoder loss: KL + reward prediction ---
        # Per-step ELBO with z_seq forces the GRU to predict rewards
        # from partial-belief z_seq[k] (built from only first k context
        # transitions) on cheetah-vel where target velocity is barely
        # observable from a single (s,a) — encoder fails to converge.
        # Fall back to OLD's design: single-posterior z (last step) +
        # decoder predicts the SAC batch's rewards. Ordered context
        # still helps the GRU build a coherent posterior.
        self.encoder_optimizer.zero_grad()

        kl_div = -0.5 * torch.sum(
            1 + self.agent.z_logvars - self.agent.z_means.pow(2)
            - self.agent.z_logvars.exp())

        z_for_pred = self.agent.z
        z_rep = z_for_pred.unsqueeze(1).expand(-1, b, -1).reshape(t * b, -1)
        rewards_flat = rewards.view(t * b, -1)
        reward_pred = self.agent.reward_decoder(
            obs_flat, actions_flat, z_rep)
        reward_pred_loss = F.mse_loss(reward_pred, rewards_flat)

        encoder_loss = (self.kl_lambda * kl_div +
                        self.reward_pred_weight * reward_pred_loss)
        encoder_loss.backward(retain_graph=True)

        # Q-function loss
        self.qf1_optimizer.zero_grad()
        self.qf2_optimizer.zero_grad()
        rewards_scaled = rewards_flat * self.reward_scale
        terms_flat = terms.view(self.batch_size * num_tasks, -1)
        q_target = rewards_scaled + (1. - terms_flat) * self.discount * target_v
        qf_loss = (torch.mean((q1_pred - q_target) ** 2) +
                   torch.mean((q2_pred - q_target) ** 2))
        qf_loss.backward()
        self.qf1_optimizer.step()
        self.qf2_optimizer.step()
        self.encoder_optimizer.step()

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
