We want an agent that, after training across a distribution of tasks $p(\mathcal{T})$ — each one a Markov decision process with its own reward or dynamics — can walk into a brand-new task drawn from that distribution and be good at it after a handful of trajectories. Two costs pull in opposite directions, and the usual methods pay both. There is the cost at *test* time, how many interactions before the agent does well on the new task, and there is the cost at *meta-training* time, how many environment samples are burned while learning to adapt in the first place. The first is advertised everywhere; the second is brutal and rarely discussed: meta-training runs across many tasks and many trials each, so if data cannot be reused, the total sample count is millions per task times a large task set. Every efficient meta-RL method we have is on-policy — RL$^2$ runs a recurrent net over the on-policy stream and trains it with policy gradients; MAML and ProMP take inner and outer policy-gradient steps on on-policy trajectories; MAESN adapts a latent by gradient descent on on-policy data — and they all spend on the order of $10^8$ timesteps meta-training. The lever that makes single-task RL cheap is off-policy learning: keep a replay buffer and learn from transitions collected by old policies, exactly what soft actor-critic does. So the obvious move is to condition the best off-policy actor-critic we have on a summary of the task and meta-train it.

The wall is principled, not an engineering nuisance. Meta-learning rests on the data distribution used to adapt at *test* time matching the distribution the system was meta-trained to adapt from. At test time an RL agent adapts from the experience it gathers by *exploring* the new task, and that experience is on-policy by construction. The matching principle then says meta-training should also adapt from on-policy data — but off-policy replay data, old transitions from stale policies, is systematically unlike the on-policy data the adapted agent will face. Naively meta-training the adaptation mechanism on replayed transitions trains it on a distribution it never sees at test, and it breaks; grafting a recurrent context encoder onto an off-policy value learner simply does not optimize on these continuous-control families. We have a contradiction to resolve, not a knob to turn: off-policy efficiency wants to train on the whole buffer, the matching principle wants the adaptation data on-policy.

I propose PEARL (Probabilistic Embeddings for Actor-critic RL). The resolution is to notice that "the algorithm" is not indivisible. At test time the agent does two distinct jobs: *inferring the task* from collected experience, and *acting* given the task. The matching principle only bites on the input to the inference — the experience the agent infers from must resemble what it will infer from at test. It says nothing about how the *control* part, the policy and value function, is trained: the policy just needs to be good for whatever task it is told it is in, and it does not care whether the transitions teaching it $Q$-values came from the current policy or a stale one — that is the ordinary off-policy claim. So I make the task belief an explicit latent context variable $z$, condition the policy $\pi(a|s,z)$ and the critic $Q(s,a,z)$ on it as if $z$ were part of the state, and then train the actor-critic *fully off-policy* on the entire replay buffer while training the *encoder* — the map from experience to $z$ — on recent, near-on-policy data. The data that trains the encoder need not be the data that trains the policy, and that single decoupling kills the contradiction.

To infer $z$, I use amortized variational inference: a recognition network $q_\phi(z|c)$ approximating the posterior over $z$ given the context $c$ (the collected transitions), with reparameterized samples $z = \mu_\phi(c) + \sigma_\phi(c)\odot\varepsilon$, $\varepsilon\sim\mathcal{N}(0,I)$, so gradients flow through the sample. The training objective is a variational lower bound,
$$\mathbb{E}_{\mathcal{T}}\Big[\,\mathbb{E}_{z\sim q_\phi(z|c^{\mathcal{T}})}\big[R(\mathcal{T},z)\big] - \beta\,\mathrm{KL}\big(q_\phi(z|c^{\mathcal{T}})\,\|\,p(z)\big)\Big],$$
with a unit-Gaussian prior $p(z)=\mathcal{N}(0,I)$. The KL term earns its place twice. It is the VAE regularizer, but read the other way it is a variational upper bound on the mutual information $I(Z;C)$, hence an information bottleneck: penalizing it squeezes $z$ down to only the bits of context needed to act well. That matters because training happens on a finite set of tasks, and a fat unconstrained $z$ would memorize idiosyncrasies and overfit; the bottleneck forces $z$ toward a minimal sufficient statistic of the task, which is exactly what generalizes to held-out tasks. So $\beta$ is not a nuisance constant — it is the dial controlling how much the task belief may remember, and therefore overfitting.

What should $z$ be good *for*? The generative reflex is to reconstruct the MDP — predict reward and next state from $(s,a,z)$ — but that is wasteful: when dynamics are shared and only reward varies, an encoder straining to reconstruct transitions spends capacity on what is identical across tasks. Maximizing the actor's returns directly couples the encoder to a high-variance objective. The thing the agent actually needs in order to act is the *value*. So $R(\mathcal{T},z)$ is the negative critic loss: the encoder receives its gradients from the Bellman update for $Q$, which aims the task representation squarely at "what do I need to know to value actions correctly," the control-relevant summary and nothing more.

The architecture of $q_\phi(z|c)$ follows from a structural fact: to identify an MDP, the *order* of the transitions does not matter. The Markov model is made of local transition and reward laws, and permuting the observed tuples does not change the information they carry about those laws — the context is a *set*, not a sequence. An RNN imposes an order that is not there, wastes capacity learning to be order-invariant, optimizes slowly over long horizons, and can latch onto spurious temporal patterns. So I want a permutation-invariant encoder, and I want $z$ probabilistic with aggregation that behaves like *evidence accumulation* — more transitions, more certainty. A plain mean of point embeddings carries no uncertainty and does not sharpen with $N$. Instead each transition $c_n$ produces a Gaussian *factor* over $z$, $\Psi_\phi(z|c_n)=\mathcal{N}\big(f_\phi^\mu(c_n), f_\phi^\sigma(c_n)\big)$ — its own vote on what the task is, with its own mean and spread — and the belief is their product,
$$q_\phi(z|c_{1:N}) \propto \prod_{n=1}^N \Psi_\phi(z|c_n).$$
A product is symmetric in its factors, so this is automatically permutation-invariant. A product of Gaussians is itself Gaussian with a closed form: multiplying diagonal factors $\mathcal{N}(\mu_n,\sigma_n^2)$ and collecting the quadratic and linear terms in $z$ in the exponent, the precision of the product is the *sum* of the precisions and the natural parameter $\mu/\sigma^2$ is the *sum* of the per-factor natural parameters,
$$\frac{1}{\sigma^2} = \sum_n \frac{1}{\sigma_n^2}, \qquad \mu = \sigma^2\sum_n \frac{\mu_n}{\sigma_n^2}.$$
This says exactly what I want: the posterior mean is a precision-weighted average of the per-transition means, so confident factors (small $\sigma_n^2$) pull harder; and the posterior precision is the sum of precisions, so every additional transition strictly increases precision and shrinks the variance — the more experience, the narrower the belief. Empty context is the unit-Gaussian prior. This is not arbitrary pooling; it is a Bayesian filtering update implemented by a learned inference network.

The probabilistic belief also gives the only mechanism that lets sparse-reward tasks be cracked at all. On a sparse task there is no signal until the goal is touched, so per-timestep action noise is useless — it jitters without ever committing to go anywhere and test a hypothesis. What is needed is posterior sampling: keep a posterior over MDPs, sample one, and act optimally for it for a whole episode, giving temporally extended ("deep") exploration. Having a posterior over $z$, acting optimally for a sampled $z$ *is* acting optimally for a sampled MDP-hypothesis. So at test time I sample $z$, hold it fixed for the whole episode for a committed, coherent attempt at one hypothesis, then update $q_\phi(z|c)$ from what was observed and sample again. As the belief narrows, behavior shifts from exploring to exploiting on its own. This is meta-learned posterior sampling, and it works only because $z$ is probabilistic: collapse $z$ to a point estimate and the only randomness left is the policy's action noise, which is not a persistent hypothesis — so a deterministic-$z$ variant should simply fail on sparse navigation.

The control side is soft actor-critic with $z$ appended to the state everywhere. The critic regresses the soft Bellman target,
$$L_{\text{critic}} = \mathbb{E}_{(s,a,r,s',d)\sim B,\;z\sim q_\phi(z|c)}\Big[\big(Q(s,a,z) - (\eta r + (1-d)\gamma\bar V(s',\bar z))\big)^2\Big],$$
where $\eta$ is the reward scale, $\gamma$ the discount, $d$ the terminal flag, $\bar V$ the slowly-tracked target value net, and $\bar z$ denotes that no gradient flows into the encoder through the target — the target is a fixed regression target, the usual target-network logic. The actor loss is SAC's information projection with $z$ detached,
$$L_{\text{actor}} = \mathbb{E}_{s\sim B,\,a\sim\pi,\,z\sim q_\phi(z|c)}\Big[D_{\mathrm{KL}}\big(\pi(a|s,\bar z)\,\big\|\,\exp(Q(s,a,\bar z))/\mathcal{Z}(s)\big)\Big],$$
and the encoder's regularizer is $L_{\text{KL}} = \beta\,\mathrm{KL}(q_\phi(z|c)\,\|\,p(z))$. The subtle, load-bearing detail is *which gradients touch the encoder*: $z$ carries gradients into the encoder only through the $Q$-loss and the KL, and is detached in the value loss and the policy loss. Letting the policy and value gradients also flow in would pull the task representation in three directions and muddy the inference signal; keeping it driven by the Bellman/critic objective is what keeps $z$ a clean, control-relevant summary. Concretely the $Q$-optimizer step and the context-encoder-optimizer step happen together on the same backward pass, with $z$ live for the $Q$'s but detached for $V$ and $\pi$.

The decoupling is made operational by *what data each part sees*. The actor and critic are off-policy: minibatches of *decorrelated transitions from the entire replay buffer* — that is where the sample-efficiency comes from, and decorrelation matters because sampling whole correlated trajectories for the RL batch destabilizes the value updates. The encoder must respect the matching principle, so its context is drawn by a *separate* sampler from *recently collected*, near-on-policy data, re-collected periodically, and from a batch *distinct* from the RL batch. If the context were also drawn from the whole buffer I would reintroduce exactly the train/test mismatch I set out to kill: at test the encoder sees on-policy exploration data, so feeding it ancient off-policy transitions during meta-training trains it on the wrong distribution. It need not be strictly on-policy — "recent" keeps it close enough while still reusing — but it cannot be the whole buffer. That separate context sampler is the form of "the encoder's data $\neq$ the policy's data," and it is the thing that makes off-policy meta-RL actually train.

So the meta-training inner step, in gradient order: sample a minibatch of tasks; for each, draw a context batch from its recent data, infer $q_\phi(z|c)$ and reparameterize a sample $z$, and draw a separate off-policy RL batch from the full buffer. Backward the KL loss into the encoder (retain the graph, since $z$ is shared); compute the $Q$-loss with live $z$, backward it, and step the $Q$-optimizers and the encoder optimizer together; compute the value loss against $\min(Q_1,Q_2)$ on fresh policy actions with $z$ detached and step $V$, then soft-update $\bar V$; compute the policy loss with $z$ detached plus SAC's mean/std regularizers and step the actor; detach $z$ between mini-batches so backprop does not run away through the shared latent. For data collection each iteration, for a few sampled tasks, clear that task's encoder buffer so context stays recent, gather some steps with $z$ from the *prior* (pure exploration before the task is known) and some with $z$ from the posterior after context has accumulated, so the encoder sees both prior- and posterior-conditioned data; initial data for all tasks is collected from the prior to seed the buffers. Meta-testing is the posterior-sampling loop: start with empty context, sample $z$ from the prior, roll out a trajectory acting optimally for that fixed $z$, accumulate only that task's trajectory into the context, re-infer $q_\phi(z|c)$, sample again, repeat — committing to a coherent hypothesis each episode and sharpening the belief as evidence arrives.

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
        n = self.config.get('num_initial_steps', 2000)
        for idx in self.train_tasks:
            self.env.reset_task(idx)
            collect_data(self.agent, self.env, self.sampler,
                         self.replay_buffer, self.enc_replay_buffer,
                         idx, n, 1, np.inf, add_to_enc_buffer=True, config=self.config)

    def train_iteration(self, iteration_idx):
        for _ in range(self.num_tasks_sample):
            idx = np.random.choice(self.train_tasks)
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
