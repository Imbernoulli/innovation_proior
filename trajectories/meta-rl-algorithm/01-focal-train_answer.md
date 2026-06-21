The scaffold hands me a SAC actor-critic that already conditions on a task variable $z$, but nothing that makes $z$ mean anything: the default $z$ is a dummy zero, `infer_posterior` and `adapt` are no-ops, and `train_iteration` takes no gradients, so the policy is identical on every task and there is no meta-learning at all. The first object I have to design is therefore the context encoder — the map from collected transitions to a $z$ that tells the policy which task it is in — together with the meta-gradient step that trains it. I want the *thinnest* encoder that could possibly work, with a training signal cleanly decoupled from the value learning, so whatever I build next has an honest floor to beat.

I propose FOCAL: a deterministic, mean-aggregating context encoder shaped by an inverse-power distance-metric loss, dropped onto plain online SAC. The starting point is to ask what the latent even *is*. These continuous-control families share state and action space and differ only in their reward (and sometimes transition) functions; take the clean case of deterministic dynamics and assume task-transition correspondence — two tasks agree on transition and reward everywhere iff they are the same task. Then the pair $(P,R)$ *identifies* the task, each $(s,a)$ has a unique outcome $(s',r)$, and every task is a function $f_T(s,a)=(s',r)$ living on the transition space $S\times A\times S\times\mathbb{R}$ — which is exactly the context space the harness samples. The encoder is not inferring a fuzzy belief about a hidden quantity; it is *embedding the function* $f_T$ from samples of it. The consequence that legitimizes a simple design: because $(P,R)$ pin the task pointwise, a *single* transition tuple already constrains which task I am in, so I do not need to integrate evidence across many transitions to slowly grow confident. Two properties fall straight out. The encoder should be **permutation-invariant** — order cannot matter if each transition independently reveals the task — and **deterministic** — there is no irreducible uncertainty to represent, because the task is recoverable, not guessed. No probabilistic posterior, no information bottleneck. The cheapest permutation-invariant aggregator is the one prototypical networks (Snell et al. 2017) use for class prototypes: embed every transition with a shared MLP and take the **mean** of the per-transition embeddings. So the encoder is a width-200 depth-3 MLP from one transition $(o,a,r)$ to a `latent_dim`-vector, and $z$ for a task is the mean over its context rows. That is `infer_posterior`; `adapt` calls it on the accumulated online context.

"Mean encoder" only says how I *combine* embeddings, not what makes an embedding good — and here the geometry of $z$ turns out to be load-bearing rather than cosmetic. The value function $Q(s,a,z)$ is a continuous network: for close $z_1,z_2$ it is *forced* to output close Q-values. Now take two genuinely different tasks whose embeddings happen to land near each other. The network must give them nearly equal Q-values, but their *true* Q-values, built from different rewards and dynamics, are far apart — a single continuous approximator simply cannot output two well-separated values at two nearly-identical inputs. So if the encoder lets distinct tasks' embeddings sit close, the conditioned value functions for those tasks become *unrepresentable* and control fails downstream no matter how good the actor-critic is. Keeping distinct tasks far apart in latent space is therefore a *precondition* for the value functions to exist, and I need an objective that lives on the embedding space itself and is decoupled from the value learning so I can trust it.

The textbook objective for "cluster same, separate different" is the contrastive loss — a quadratic attractor $\mathbb{1}\{\text{same}\}\cdot\lVert q_i-q_j\rVert^2$ and a margin hinge $\mathbb{1}\{\text{diff}\}\cdot\max(0,m-\lVert q_i-q_j\rVert)^2$ — but it degenerates here into blobs that hold transitions from several tasks at once, and *why* it degenerates dictates the fix. Two failures compound. The margin hinge is a spring that acts only within radius $m$ and is weakest exactly where I need it most: at initialization all embeddings bunch near the origin (the latent is tanh-bounded, the encoder random), every pairwise distance is tiny, and the hinge gradient $-2(m-d)$ is bounded with no special urgency at small $d$. Deeper still, the attractive squared-distance term is algebraically *variance maximization* — $\sum_{i,j}\lVert q_i-q_j\rVert^2 = 2N^2\cdot\mathrm{Var}(X)$ — and variance is a single global scalar that many distributions share, including the degenerate one that piles half the mass at $+1$ and half at $-1$ on each axis. That distribution has huge variance while cramming several tasks into each pile, so the optimizer happily spreads mass to the extremes and merges tasks, because the global statistic says nothing about whether *every* pair of distinct clusters is separated.

So I know exactly what I need: a repulsion between different-task pairs that is *strong at short range and fades with distance* — the opposite profile of the margin spring, and one no global statistic can satisfy. The function with that profile is a *negative power of distance*, $\beta/(\lVert q_i-q_j\rVert^n+\varepsilon)$, which blows up as the distance goes to zero and relaxes once the pair is well separated. Replacing the margin term gives the deep-metric-learning loss
$$L = \mathbb{1}\{\text{same}\}\cdot\lVert q_i-q_j\rVert^2 \;+\; \mathbb{1}\{\text{diff}\}\cdot\frac{\beta}{\lVert q_i-q_j\rVert^n+\varepsilon}.$$
Now the different-task term is a genuine repulsive *potential*, not a capped spring: two distinct embeddings sitting on top of each other feel an enormous push apart, and the variance trick cannot game it, because the offending pairs *inside* a merged pile are at small distance and the $1/d^n$ term is screaming at them — every pair of distinct clusters is forced apart. Physically it is a system of like charges in a bounded box: with $n=1$ it is literally the Coulomb potential, the charges migrate to the boundary and pile at the corners, and the tanh-bounded latent cube $(-1,1)^l$ is the conducting box that lets the repulsion settle into a well-separated equilibrium instead of flinging everything to infinity. The same-task quadratic still pulls each task into a tight cluster, so I get tight clusters scattered to the extremes — exactly the geometry the continuity argument demanded. I take $n=2$ (the inverse-square Cauchy form, sharper than Coulomb, concentrating effort on the closest offending pairs and preserving local topology better than the quadratic), $\beta=1$ to match the attractive and repulsive scales, $\varepsilon=10^{-3}$ to floor the denominator.

One design choice I want to be explicit about is how the encoder relates to the value-learning gradients, because here this fill deviates from the method's original *offline* setting. That setting also carries behavior-regularization machinery — a dual-form-KL discriminator estimating a divergence between the learned policy and the data's behavior policy, folded into the critic target and actor — because offline bootstrapping over out-of-distribution actions diverges without a tether to the data support. But this harness is **online**: `collect_data` interacts every iteration, so the replay buffer is on-policy and the out-of-distribution-value problem that regularizer fights does not bite. I therefore drop it entirely — no discriminator, no value penalty, no adaptive $\alpha$ — and keep only the two load-bearing pieces: the deterministic mean encoder and the inverse-power DML loss. The harness makes the contrastive loss convenient: `sample_context_from_buffer` returns a per-task context block, so I split each task's context in half, mean-encode each half into two embeddings sharing the task's label, and apply the squared distance to same-task pairs and the inverse-power to different-task pairs across the meta-batch. I also follow the scaffold's PEARL-shaped update rather than the offline version's full gradient isolation: the contrastive loss is backpropped into the encoder, but the encoder optimizer is stepped *together with* the critic and the $z$ feeding the two Q-heads is **not** detached, so the Bellman gradient also reaches the encoder (it is detached only in the value and policy losses). The encoder is therefore shaped by the DML loss *plus* the critic. The SAC update underneath is the standard twin-Q / value / squashed-Gaussian-actor step — reward scaled, the soft target value net slowly tracked with $\tau=5\times10^{-3}$, Adam at $3\times10^{-4}$.

I expect this floor to crack where its premise is thinnest. The whole construction rests on "a single transition reveals the task." On `cheetah-vel` the target velocity is only weakly visible in any one $(s,a,r)$ — it is the *trend* across a sequence that pins it down — and a permutation-invariant mean throws that structure away, so the cheetah encoding should be mediocre and unreliable. On `sparse-point-robot` the reward is $+1$ only near the goal and $0$ everywhere else, so almost every context transition carries reward $0$ and is identical across tasks: the contrastive signal has almost nothing to separate tasks *by*, and there is no exploration mechanism that commits to going somewhere to *find* the goal — so I expect sparse to fail hardest, returns near zero. Only the dense low-dimensional `point-robot`, where the goal is readable from a few dense transitions, should look healthy. If that split is what I measure, the diagnosis writes the next rung: an encoder that respects the *sequence* of experience, and eventually a representation that carries *uncertainty* the agent can act on.

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
