**Problem.** The scaffold conditions a SAC actor-critic on a task variable `z`, but the default `z` is a
dummy zero and the training step is a no-op, so the policy is identical on every task. The first thing to
design is the context encoder — the map from collected transitions to a `z` that identifies the task —
plus the meta-gradient step that trains it. I want the thinnest encoder that respects the one hard
constraint and whose training signal is cleanly decoupled from the value learning, to set a floor.

**Key idea.** Under deterministic dynamics with task-transition correspondence, `(P, R)` identifies the
task and a single transition in principle reveals it, so the encoder can be **deterministic and
permutation-invariant**: embed each transition with a shared MLP and take the **mean** as `z` (no
probabilistic belief, no information bottleneck). A continuity argument forces the geometry: distinct
tasks with close `z` make their conditioned value functions unrepresentable, so distinct tasks must be
pushed apart. The plain contrastive loss degenerates — its squared-distance attractor is exactly variance
maximization (`Σ(x_i−x_j)² = 2N²·Var`), a global statistic satisfied by a degenerate distribution that
merges tasks. Replace the margin repulsion with a **negative power of distance**, a Coulomb-like potential
`β/(‖q_i−q_j‖ⁿ + ε)` that is strongest at short range and per-pair, so it cannot be gamed by variance and
forces *every* pair of distinct task clusters apart, settling them to the corners of the tanh-bounded
latent cube. `n = 2` is the inverse-square (Cauchy) form.

**Why (and what this harness omits / changes).** The method's offline-setting **behavior-regularization
machinery (dual-form-KL BRAC, adaptive `α`, value penalty) is dropped here**: this harness is *online* —
`collect_data` interacts every iteration, the buffer is on-policy, and the out-of-distribution-value
problem BRAC fights does not bite — so the fill is a deterministic-encoder + DML loss on the scaffold's
plain SAC. The contrastive loss is computed by splitting each task's context block in half, mean-encoding
each half into two same-label embeddings, and applying squared distance to same-task pairs and
inverse-power to different-task pairs. Note one further deviation from the clean offline version: this fill
does **not** isolate the encoder at the gradient level. The contrastive loss is backpropped into the
encoder, but the encoder optimizer steps together with the critic and the `z` fed to the two Q-heads is
**not** detached, so the Bellman gradient also reaches the encoder (it is detached only in the value and
policy losses). The encoder is therefore shaped by the DML loss *plus* the critic, PEARL-style.

**Hyperparameters.** `latent_dim=5`; encoder width-200 depth-3 MLP → `latent_dim` (no IB head); actor/
critic net size 300, twin Q + V + slow target, `τ=5e-3`; Adam lr `3e-4`; per-env `discount` (0.90–0.99)
and `reward_scale` (5–100); DML `β=1.0`, `ε=1e-3`, power `n=2.0`, contrastive weight `1.0`; meta-batch 16;
RL batch 256; collection budgets from the env config.

**What to watch.** The "one transition reveals the task" premise is weakest where it matters: on
`cheetah-vel` a single `(s,a,r)` barely shows the target velocity and the mean discards temporal
structure, so the encoding should be mediocre; on `sparse-point-robot` almost every context transition has
reward 0 and there is no exploration mechanism to *find* the goal, so a deterministic `z` should fail
hardest (returns near zero). Only the dense low-dim `point-robot` should look healthy. That split forces a
sequential, eventually uncertainty-carrying encoder next.

```python
# EDITABLE region of custom_meta_rl.py — step 1: FOCAL (deterministic mean encoder + inverse-power DML, online SAC)
class CustomMetaRLAgent(nn.Module):
    """FOCAL agent: deterministic encoder with mean aggregation."""

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

        # Deterministic encoder: output is latent_dim (no IB)
        self.context_encoder = build_mlp(
            context_input_dim, latent_dim,
            hidden_dim=200, n_layers=3,
        )

        # Policy, Q-functions, V-function (z-conditioned)
        self.policy = build_policy(obs_dim, action_dim, latent_dim, net_size)
        self.qf1 = build_qf(obs_dim, action_dim, latent_dim, net_size)
        self.qf2 = build_qf(obs_dim, action_dim, latent_dim, net_size)
        self.vf = build_vf(obs_dim, latent_dim, net_size)
        self.target_vf = copy.deepcopy(self.vf)

        self.register_buffer('z', torch.zeros(1, latent_dim))
        self._context = None

    def clear_context(self, num_tasks=1):
        self.z = ptu.zeros(num_tasks, self.latent_dim)
        self._context = None

    def clear_z(self, num_tasks=1):
        self.clear_context(num_tasks)

    def sample_z(self):
        pass  # z is deterministic

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
        """Encode context and take mean over transitions."""
        embeddings = self.context_encoder(context)
        # embeddings: (num_tasks, seq_len, latent_dim)
        embeddings = embeddings.view(context.size(0), -1, self.latent_dim)
        self.z = torch.mean(embeddings, dim=1)  # (num_tasks, latent_dim)

    def adapt(self):
        if self._context is not None:
            self.infer_posterior(self._context)

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
    """FOCAL SAC with deep metric encoder loss."""

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
        self.contrastive_weight = 1.0
        self.dml_beta = config.get('dml_beta', 1.0)
        self.dml_epsilon = config.get('dml_epsilon', 1e-3)
        self.dml_power = config.get('dml_power', 2.0)

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

    def _encode_task_context(self, context):
        embeddings = self.agent.context_encoder(context)
        embeddings = embeddings.view(context.size(0), -1, self.agent.latent_dim)
        return torch.mean(embeddings, dim=1)

    def _compute_contrastive_loss(self, indices, context):
        """FOCAL Eq. 13 deep metric learning loss."""
        half = context.size(1) // 2
        labels_base = np.asarray(indices)
        if labels_base.ndim == 0:
            labels_base = labels_base[None]
        if half == 0:
            z = self._encode_task_context(context)
            labels_np = labels_base
        else:
            ctx_a = context[:, :half, :]
            ctx_b = context[:, half:2*half, :]
            z = torch.cat([
                self._encode_task_context(ctx_a),
                self._encode_task_context(ctx_b),
            ], dim=0)
            labels_np = np.concatenate([labels_base, labels_base])

        labels = torch.as_tensor(labels_np, device=context.device)
        same_task = labels[:, None].eq(labels[None, :]).float()
        dist_sq = torch.sum((z[:, None, :] - z[None, :, :]) ** 2, dim=-1)
        positive_loss = same_task * dist_sq
        dist = torch.sqrt(dist_sq + 1e-12)
        negative_loss = (1.0 - same_task) * (
            self.dml_beta /
            (torch.pow(dist, self.dml_power) + self.dml_epsilon)
        )
        return (positive_loss + negative_loss).mean()

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

        # Eq. 13 deep metric encoder loss
        self.context_optimizer.zero_grad()
        contrastive_loss = self._compute_contrastive_loss(indices, context)
        encoder_loss = self.contrastive_weight * contrastive_loss
        encoder_loss.backward(retain_graph=True)

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
