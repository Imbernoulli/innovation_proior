The recurrent reward-decoder encoder did exactly the two things I built it to do. On `point-robot` it improved on FOCAL's $-13.9$ to $-10.1$, tight across seeds, because the sequential readout is strictly more expressive and the dense low-dim task rewards that. On `cheetah-vel` it firmed up the spread — $-76.6$, $-89.7$, $-85.4$, no $-108$ outlier — though the mean ($-83.9$) barely moved from FOCAL's $-84.7$ because the budget caps how good any single-$z$ encoder gets. And on `sparse-point-robot` it rescued the dead seeds as predicted, lifting the mean from $0.30$ to $1.99$. But look harder at sparse, because that is where the ceiling is. $1.99$ mean with a best seed of only $3.39$ is still low in absolute terms — the agent reaches the goal occasionally, not reliably, and the spread ($1.17$ to $3.39$) says it is luck-of-the-rollout whether a seed's exploration happens to stumble onto the reward enough times to learn from it. The single stochastic $z$ *sharpened task inference* but gave the agent no *exploration* mechanism: the latent is sampled, fed to the policy, and the only within-episode randomness left is per-step action noise, which is time-invariant and undirected, so the policy jitters around and never *commits* to going somewhere to check whether the goal is there. On a sparse task you must reach the goal at least once to get any signal, and undirected jitter across a half-circle of goals will not reliably do that in this budget. I need a latent whose *uncertainty* the agent can act on — a belief it can sample a hypothesis from and pursue coherently for a whole episode.

I propose PEARL: rebuild the inference side around uncertainty as a first-class object — a genuine probabilistic posterior $q(z\mid c)$ assembled as a product of Gaussians, paired with posterior-sampling exploration. The exploration mechanism is classical Bayesian RL's trick: keep a posterior over which MDP you are in, sample one MDP-hypothesis, act optimally for it for a whole episode (temporally-extended, coherent exploration, not per-step jitter), observe, update the posterior, sample again. As the belief narrows, behavior shifts from exploring to exploiting on its own. That is precisely the "commit to a hypothesis and pursue it" sparse needs, and it only works if $z$ is *probabilistic* and held *fixed for the episode*; a point estimate has no hypothesis to commit to.

For that, three things must line up: the posterior must be a real probabilistic object, it must be permutation-invariant over the context, and it must sharpen with evidence the way a Bayesian belief does. By the Markov property each transition $c_n$ is an independent sample of the same reward and dynamics, so the context is an unordered *set* and the belief given all of them is proportional to the *product* of per-transition factors. I drop the GRU here, and that is a deliberate reversal: I added recurrence at the last rung to read a *trend*, but a probabilistic belief that sharpens correctly wants independent evidence *fused*, not sequentially summarized, and the product fusion is exactly that. Let each transition emit a Gaussian factor $\mathcal{N}(\mu_n,\sigma_n^2)$ — its own vote on the task with its own spread — and fuse them by the product of Gaussians, which is itself Gaussian with a closed form:
$$\frac{1}{\sigma^2}=\sum_n\frac{1}{\sigma_n^2},\qquad \mu=\sigma^2\sum_n\frac{\mu_n}{\sigma_n^2}.$$
Read what that means and it is exactly the belief I want: confident factors (small $\sigma_n^2$) pull harder, and every additional transition *increases* the precision, i.e. *shrinks* the variance — the belief sharpens with evidence, Bayesian filtering falling straight out of the product. And the product is symmetric in its factors, so the encoder is permutation-invariant by construction. The encoder MLP therefore outputs, per transition, a mean and a pre-softplus variance — $2\cdot$`latent_dim` outputs — and the agent fuses them into `(z_means, z_vars)` and samples $z$ by reparameterization.

Now the training signal, the second reversal from the last rung. I trained the GRU with reward reconstruction; that gave $z$ content. But for a *posterior* I want a different regularizer — the information bottleneck. I put a KL from the belief to a unit-Gaussian prior, $\beta\cdot\mathrm{KL}\!\left(q(z\mid c)\,\Vert\,\mathcal{N}(0,I)\right)$, into the objective. It does double duty: it is the VAE-style regularizer that keeps the belief near the prior, *and*, read as a variational bound on the mutual information $I(z;c)$, it is an information bottleneck that squeezes $z$ to the minimal task-relevant statistic — which matters enormously under this tiny budget, where a fat unconstrained $z$ would memorize training-task idiosyncrasies and not transfer to held-out tasks. The prior $\mathcal{N}(0,I)$ is also what I sample from at the very start of an episode when I have no context — pure prior-conditioned exploration. So the KL gives me the bottleneck *and* the exploration prior in one term. And what makes $z$ *good for control* is that I train the encoder from the *critic*: the Bellman gradient of the Q-loss flows into the encoder, so $z$ is shaped to be exactly the task summary that makes the value function accurate — which is what the policy actually needs, more directly than reward reconstruction (which spends capacity predicting reward magnitudes the controller does not consume). So $z$ carries gradients into the encoder through the Q-loss and the KL, and is detached in the value and policy losses. The reward decoder is gone entirely; so is the GRU.

There is one more piece that earns the "off-policy meta-RL" name and that I now make load-bearing: *which data trains which part*. Meta-learning needs the adaptation-data distribution to match between meta-train and meta-test — at test time the encoder infers from on-policy exploration data, so during training I must not feed it ancient off-policy transitions or I train it on the wrong distribution. But that matching constraint binds *only* on the encoder's input, not on the control updates: the actor and critic just need good value estimates and do not care whether the transitions came from a stale policy. So I decouple the samplers. The actor and critic learn from decorrelated minibatches drawn from the *entire* replay buffer, where all the sample-efficiency lives, while the encoder's context is drawn by a *separate* sampler from *recently collected* data — the harness's separate encoder buffer, cleared per task each iteration so the context stays recent — and from a batch distinct from the RL batch. Collection itself mixes prior-conditioned exploration (sample $z$ from $\mathcal{N}(0,I)$, gather data before the agent knows the task) with posterior-conditioned exploration (re-infer $q(z\mid c)$ after some context accumulates), so the encoder sees both, exactly the distribution it faces at test. This decoupling was implicitly carrying the previous rungs; here it is the operational form of "the encoder's data $\neq$ the policy's data," and it is what lets off-policy efficiency coexist with the distribution match the encoder needs.

Putting it together on this edit surface: the agent holds the permutation-invariant product-of-Gaussians encoder (MLP $\to 2\cdot$`latent_dim`, softplus on the variance, fuse, reparameterized sample) and the scaffold's $z$-conditioned SAC heads; `infer_posterior` fuses the context into `(z_means, z_vars)` and samples, `adapt` runs it on accumulated context, `compute_kl_div` is the KL to $\mathcal{N}(0,I)$. The meta-gradient step samples context from the recent encoder buffer (distinct from the off-policy RL batch from the full buffer), backprops the KL into the encoder, backprops the Q-loss into the Q-nets *and* the encoder and steps them together, then the value loss and policy loss on detached $z$, with a soft target-value update at $\tau=5\times10^{-3}$ and Adam at $3\times10^{-4}$. Test-time evaluation is the posterior-sampling loop the protocol already implements: clear context, sample $z$ from the prior, roll out, accumulate, re-infer, repeat, then act deterministically with the inferred $z$. Unlike the previous two rungs, none of the method's machinery is dropped — the probabilistic belief, the bottleneck, the posterior-sampling exploration, and the decoupled samplers all fit the harness directly.

Against the VariBAD numbers, the sharp prediction is the sparse task: posterior sampling gives temporally-extended, hypothesis-committed exploration that undirected jitter never could, so `sparse-point-robot` should jump clearly above $1.99$ — a real success only if the mean roughly doubles into the mid-single-digits, with a best seed pushing well past $3.39$ (a seed near or above $\sim 9$ would be the signature of an agent that actually sweeps the half-circle). The information-bottleneck KL should also help generalization under the tiny budget, so I expect `cheetah-vel` to *improve*, not just firm up: dropping reconstruction for a critic-trained, bottlenecked $z$ should land in the low-to-mid $-60$s, a clear step below $-83.9$. The honest risk is `point-robot`: VariBAD's recurrent encoder won it at $-10.1$, and a permutation-invariant product-of-Gaussians belief discards the sequential structure that helped there, so I would not be surprised by a regression into the $-14$ to $-16$ range. That trade is the explicit bet — give up a little on the easiest low-dimensional family, where any encoder suffices, to win decisively on the two families that actually exercise meta-RL, high-dimensional encoding and sparse-reward exploration, because the probabilistic belief and the posterior-sampling it enables are aimed precisely at those.

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
