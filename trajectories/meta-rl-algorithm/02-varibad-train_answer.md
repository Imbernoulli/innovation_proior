The deterministic mean encoder landed exactly where its premise predicted. On the dense low-dimensional `point-robot` it is fine — mean $-13.9$, tight across seeds — because there the goal really is readable from a handful of dense transitions. But `cheetah-vel` is the worst baseline so far at $-84.7$, and the per-seed spread is the tell: $-64.2$, $-81.6$, and a $-108.4$ seed more than a third worse than the best — not noise around a good solution but an encoder that sometimes locks onto a usable target-velocity representation and sometimes does not. And `sparse-point-robot` is the sharpest failure: $0.30$ mean, two of three seeds at *exactly* $0.0$ (the convention's signal that the goal was never reached) and the third at $0.90$. Both failures point at the same two holes. The mean over per-transition embeddings throws away *order*, and on cheetah the target velocity is the *trend* of reward across a sequence, which a permutation-invariant bag cannot read. And the contrastive loss needs *contrast*, but on sparse almost every context transition has reward $0$ and is identical across tasks, so the distance-metric loss has nothing to separate tasks by. I need an encoder that respects the *sequence* of experience and a training signal richer than "same/different task" — one that forces $z$ to encode the task's actual *reward structure*.

I propose VariBAD: a recurrent encoder that reads out a stochastic latent, shaped by a reward-decoder reconstruction loss together with a KL bottleneck. Take the sequence problem first, because it changes the primitive. I argued for permutation-invariance from "each transition independently reveals the task," but the cheetah failure shows that premise is too strong here — when the task is only weakly visible per transition, *what I should do next* depends on the whole sequence so far, not on an order-free set. The right primitive for online, ordered inference is recurrence. So I replace the mean with a GRU encoder: embed each transition $(o,a,r)$ with a small 2-layer ReLU pre-MLP, run a GRU whose hidden state carries the running summary, and read the task latent off the hidden state. One GRU step is one belief update folding in one transition, which is exactly the online structure the rollout protocol needs — `update_context` appends a transition, `adapt` runs the GRU over the accumulated context and reads $z$ from the final hidden step, and `infer_posterior` does the same on the sampled context block. One harness-specific care the mean never needed: I keep the context the GRU sees *chronological and within a single trajectory*, capping context length at the path length, because stitching independent trajectories without resetting the hidden state at episode boundaries pollutes the running summary with discontinuities. A mean does not care about order; a GRU does.

Now the richer training signal, and this is the move that actually forces $z$ to carry reward structure. The previous distance-metric loss was pure geometry — "keep tasks apart" — which remains necessary (the continuity argument from the last rung still holds: distinct tasks must be separable for the value functions to exist) but is never sufficient, because geometry alone never says *what* about the task $z$ should encode. What I want is a *generative* signal: make $z$ good at predicting the thing that differs across tasks. In these families the reward function is what differs, so I attach a **reward decoder** — a 200-hidden 3-layer MLP that, given $(s,a)$ and the task latent $z$, predicts the reward — and train the encoder so the latent it produces lets the decoder reconstruct rewards accurately. This is aimed straight at both failures: on `cheetah-vel`, reconstructing the velocity-matching reward *forces* $z$ to encode the target velocity, the very quantity the mean could not read off the sequence; on `sparse-point-robot`, predicting the reward forces $z$ to encode where the $+1$ lives, exactly the task-distinguishing bit the contrast had nothing to grab.

I make the latent stochastic, because the recurrence readout naturally produces a distribution and I want the regularization that comes with it. The GRU reads out a Gaussian posterior over $z$ — a mean $\mu$ and a log-variance — and I sample $z$ by reparameterization so gradients flow. The stochasticity buys an information bottleneck in the form of a KL to a unit-Gaussian prior,
$$\mathrm{KL}\!\left(q(z\mid c)\,\Vert\,\mathcal{N}(0,I)\right) = -\tfrac{1}{2}\sum\left(1+\log\sigma^2-\mu^2-\sigma^2\right),$$
which squeezes $z$ to the minimal reward-relevant content and discourages it from memorizing training-task idiosyncrasies under the short budget. So the encoder's loss is an ELBO-flavored pair, the reward-prediction reconstruction term plus the KL,
$$L_{\text{enc}} = \lambda\cdot\mathrm{KL}\!\left(q(z\mid c)\,\Vert\,\mathcal{N}(0,I)\right) + \text{reward\_pred\_loss},$$
with $\lambda=0.1$ and reward-prediction weight $1.0$. The encoder and decoder train together under this loss; the decoder shapes $z$ at training time and is dropped at evaluation, where `adapt` just runs the GRU and the policy acts on the inferred belief.

I have to be careful which version of this reconstruction I actually implement, because the clean formulation is more elaborate than this harness can carry, and I should match the harness rather than import machinery it cannot support. The clean version reconstructs the *whole* trajectory — past and future — from a latent inferred off each *prefix*, summing the ELBO over every context length with each KL's prior set to the *previous* posterior, so inference becomes an online Bayes filter. That per-prefix, partial-belief version does not converge on `cheetah-vel` here: a $z$ inferred from only the first few context transitions, where the target velocity is barely observable from a single $(s,a)$, cannot predict rewards well enough for the per-step ELBO to give a usable gradient, and the encoder stalls. So I fall back to the design that actually trains under this budget — a *single* posterior $z$ read off the last GRU step (not a sum over prefixes), the KL taken to the fixed unit Gaussian (not chained to the previous posterior), and the reward decoder asked to predict the rewards of the *SAC training batch* under that single $z$ (not a held-out future of the encoder's own trajectory). I also drop the transition decoder entirely — reward only — because here the reward carries the task identity and transition reconstruction is the expensive, less-informative half. What I keep from the clean idea is the recurrent ordered encoder, the stochastic latent with a KL bottleneck, and a reward-reconstruction auxiliary; what I drop, because the harness and budget cannot carry it, is the per-prefix summed ELBO, the future reconstruction, the belief-chaining prior, and the transition head.

One structural point stays the same as the previous rung: how the encoder relates to the value gradients. The clean formulation detaches the latent from the RL loss, trains the VAE with its own optimizer and buffer, and conditions the policy on the *distribution* rather than a sample. This harness does neither, and I follow it: the policy conditions on a *sampled* $z$ (the scaffold's SAC-on-$z$ interface), and the encoder optimizer is stepped together with the critic, so the encoder receives the Bellman gradient in addition to its own reconstruction-plus-KL loss (the $z$ fed to the Q-heads is not detached; it is detached only in the value and policy losses). The encoder here is thus shaped by three signals at once — reward reconstruction, the KL bottleneck, and the critic — which is more entangled than the clean version, but it is a strict enrichment over the previous rung's geometry-plus-critic signal: I have *added* the reward-prediction term and swapped the bag-of-transitions encoder for a sequential one.

Against the FOCAL numbers, I expect `cheetah-vel` to *firm up*: the $-108$ seed should disappear and the spread tighten, because reconstructing the velocity reward forces a consistent target representation regardless of seed, moving the mean toward the low-to-mid $-80$s with much smaller variance. I expect the reward decoder to rescue the dead sparse seeds — forcing $z$ to predict where the $+1$ lives should turn the two $0.0$ seeds nonzero and lift the mean well above $0.30$, a success only if the mean clears $\sim 1.5$ with no seed stuck at exactly zero. The risk I carry is `point-robot`, already healthy at $-13.9$: a heavier recurrent three-signal encoder could overfit under the tiny budget, though its strictly more expressive sequential readout might also *help*. And the honest open question is sparse in absolute terms: even with a reward decoder this rung still has no *exploration* mechanism — nothing that makes the agent commit to going somewhere to find the rare reward — so a single stochastic $z$ may sharpen task inference without fixing the deeper "you must reach the goal once to learn anything" problem. If sparse improves but stays low, the next rung is already written: a latent whose *uncertainty* the agent can act on, a probabilistic belief that supports committing to a task hypothesis for a whole episode.

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
