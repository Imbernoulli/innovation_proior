The problem is offline meta-reinforcement learning: learn from fixed, pre-collected transition logs across a distribution of tasks, then adapt to a new task at test time from only a small batch of its logged transitions, with no environment interaction at any point. Two difficulties are locked together. First, offline value-based RL suffers from bootstrapping error: a value estimate at an out-of-distribution state-action pair is never corrected by fresh data, so over-optimistic estimates feed back and can diverge. Second, task inference must work without exploration: the agent cannot act to disambiguate which task it is in, so it must extract task identity directly from the handful of logged transitions and condition its policy on that identity robustly.

The standard approaches each fail for this setting in a predictable way. PEARL and its variants build a probabilistic belief over tasks with an information bottleneck and use posterior-sampling exploration at test time, but that machinery exists to support exploration, which is forbidden here. Moreover, PEARL's encoder is trained by Bellman gradients, so when offline value learning becomes unstable the task representation is corrupted along with it. A contextual BCQ-style approach constrains the policy to the behavior support, but still ties task inference to the value-learning signal and does not directly encourage distinct tasks to occupy distinct points in embedding space. The continuity argument is decisive: a value network is continuous, so if two different tasks have nearly identical embeddings it is forced to assign them nearly identical Q-values, yet their true Q-values may differ sharply. Unless the encoder keeps different tasks far apart, the conditioned value functions are unrepresentable.

The method I propose is FOCAL, Fully-Offline Context-based Actor-critic meta-RL. It pairs a deterministic, permutation-invariant context encoder with a behavior-regularized SAC actor-critic, and crucially it decouples encoder training from value learning at the gradient level.

FOCAL's encoder maps each logged transition through a shared MLP and takes the mean of the per-transition embeddings as the task variable z. There is no information bottleneck and no product-of-Gaussians posterior: under deterministic dynamics with task-transition correspondence, the pair of transition function and reward function identifies the task pointwise, so a transition in principle reveals task identity and there is no irreducible uncertainty to model. Mean aggregation makes the encoder permutation-invariant, which is appropriate because the logged transitions are simply a set of samples from the task-specific map.

To train this encoder FOCAL uses a negative-power distance metric learning loss instead of the usual contrastive loss. The contrastive loss degenerates because its squared-distance attractive term is exactly variance maximization: summing squared pairwise distances equals twice N squared times the variance, a global statistic that can be large while several distinct tasks are merged into the same cluster. The margin-based repulsive term is also weak: it is a bounded spring that becomes zero beyond the margin and has only finite force at short range. FOCAL replaces that repulsion with an epsilon-capped negative power of distance, beta divided by the distance raised to the power n plus epsilon. For n greater than zero this is a short-range penalty that is largest when distinct-task embeddings are close, so it directly penalizes every merged pair rather than merely encouraging large average distance. The inverse-square case, n equals 2, is analogous to Cauchy graph embedding and works well in practice; the latent space is bounded by tanh to keep the repulsion from diverging.

The actor-critic side is a SAC-style twin-Q plus V architecture conditioned on z, with a behavior regularizer drawn from the BRAC framework. Because the behavior policy is unknown, the KL divergence is estimated via its dual Fenchel form using a learned discriminator, avoiding the need to fit a cloned behavior density. The divergence can enter either the critic target as a value penalty or the actor objective as a policy regularizer, and the regularization strength is adapted against a target divergence. The max-entropy term is retained even offline because it helps when multiple actions produce the same next state and reward. At test time the agent computes z as the mean encoding of the provided context transitions and rolls out the policy deterministically conditioned on z, with no exploration.

Gradient-level decoupling is the design choice that makes this combination stable. Offline value functions can grow to enormous magnitudes; if the encoder received Bellman gradients they would swamp the distance-metric signal and the embedding would collapse. Therefore z is detached when it feeds the actor and critic, and the encoder is updated only by the distance metric loss. This protects task inference from the instability of offline bootstrapping.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class FOCALAgent(nn.Module):
    """Deterministic mean-aggregating encoder with z-conditioned SAC heads."""

    def __init__(self, obs_dim, action_dim, latent_dim=5, net_size=300,
                 reward_dim=1, use_next_obs_in_context=False, **kwargs):
        super().__init__()
        self.latent_dim = latent_dim
        self.use_next_obs_in_context = use_next_obs_in_context

        context_input_dim = obs_dim + action_dim + reward_dim
        if use_next_obs_in_context:
            context_input_dim += obs_dim

        # Deterministic tanh-bounded encoder; output_dim == latent_dim
        self.context_encoder = build_mlp(
            context_input_dim, latent_dim,
            hidden_dim=200, n_layers=3,
        )

        self.policy = build_policy(obs_dim, action_dim, latent_dim, net_size)
        self.qf1 = build_qf(obs_dim, action_dim, latent_dim, net_size)
        self.qf2 = build_qf(obs_dim, action_dim, latent_dim, net_size)
        self.vf = build_vf(obs_dim, latent_dim, net_size)
        self.target_vf = copy.deepcopy(self.vf)

        self.register_buffer('z', torch.zeros(1, latent_dim))
        self._context = None

    def update_context(self, transition_tuple):
        o, a, r, no, d, info = transition_tuple
        o = torch.as_tensor(o)[None, None, ...]
        a = torch.as_tensor(a)[None, None, ...]
        r = torch.as_tensor([[r]], dtype=torch.float32)[None, ...]
        no = torch.as_tensor(no)[None, None, ...]
        if self.use_next_obs_in_context:
            data = torch.cat([o, a, r, no], dim=2)
        else:
            data = torch.cat([o, a, r], dim=2)
        self._context = data if self._context is None else torch.cat([self._context, data], dim=1)

    def infer_posterior(self, context):
        embeddings = self.context_encoder(context)
        embeddings = embeddings.view(context.size(0), -1, self.latent_dim)
        self.z = torch.mean(embeddings, dim=1)

    def adapt(self):
        if self._context is not None:
            self.infer_posterior(self._context)

    def get_action(self, obs, deterministic=False):
        obs_t = torch.as_tensor(obs)[None]
        in_ = torch.cat([obs_t, self.z], dim=1)
        return self.policy.get_action(in_, deterministic=deterministic)

    def forward(self, obs, context):
        self.infer_posterior(context)
        t, b, _ = obs.size()
        obs_flat = obs.view(t * b, -1)
        task_z = torch.cat([z.repeat(b, 1) for z in self.z], dim=0)
        in_ = torch.cat([obs_flat, task_z], dim=1)
        policy_outputs = self.policy(in_, reparameterize=True, return_log_prob=True)
        return policy_outputs, task_z


def focal_z_loss(task_z, task_indices, epsilon=1e-3):
    """Inverse-square distance metric loss on mean task embeddings."""
    pos_loss, neg_loss = 0.0, 0.0
    pos_cnt, neg_cnt = 0, 0
    for i in range(len(task_indices)):
        zi = task_z[i]
        for j in range(i + 1, len(task_indices)):
            zj = task_z[j]
            d_sq = torch.mean((zi - zj) ** 2)
            if task_indices[i] == task_indices[j]:
                pos_loss += torch.sqrt(d_sq + epsilon)
                pos_cnt += 1
            else:
                neg_loss += 1.0 / (d_sq + epsilon * 100)
                neg_cnt += 1
    return pos_loss / (pos_cnt + epsilon) + neg_loss / (neg_cnt + epsilon)


class FOCALAlgorithm:
    """FOCAL training: encoder trained only by DML; SAC+BRAC on detached z."""

    def __init__(self, agent, env, train_tasks, replay_buffer, enc_replay_buffer,
                 qf1, qf2, vf, c_network, divergence, config):
        self.agent = agent
        self.qf1, self.qf2, self.vf = qf1, qf2, vf
        self.target_vf = copy.deepcopy(vf)
        self.train_tasks = train_tasks
        self.replay_buffer = replay_buffer
        self.enc_replay_buffer = enc_replay_buffer
        self.c = c_network
        self.divergence = divergence

        self.batch_size = config.get('batch_size', 256)
        self.discount = config.get('discount', 0.99)
        self.reward_scale = config.get('reward_scale', 5.0)
        self.soft_target_tau = config.get('soft_target_tau', 0.005)
        self.z_loss_weight = config.get('z_loss_weight', 10.0)
        self.use_value_penalty = config.get('use_value_penalty', False)
        self.max_entropy = config.get('max_entropy', True)

        self._alpha = torch.tensor(config.get('alpha_init', 500.0), requires_grad=True)
        self.alpha_max = config.get('alpha_max', 2000.0)
        self.alpha_lr = config.get('alpha_lr', 1.0)
        self.target_divergence = config.get('target_divergence', 0.05)

        lr = 3e-4
        self.context_optimizer = optim.Adam(agent.context_encoder.parameters(), lr=lr)
        self.policy_optimizer = optim.Adam(agent.policy.parameters(), lr=lr)
        self.qf1_optimizer = optim.Adam(qf1.parameters(), lr=lr)
        self.qf2_optimizer = optim.Adam(qf2.parameters(), lr=lr)
        self.vf_optimizer = optim.Adam(vf.parameters(), lr=lr)
        self.c_optimizer = optim.Adam(c_network.parameters(), lr=1e-4)

    def _take_step(self, indices, context):
        num_tasks = len(indices)
        obs, actions, rewards, next_obs, terms = sample_sac_batch(
            self.replay_buffer, indices, self.batch_size)

        policy_outputs, task_z = self.agent(obs, context)
        new_actions, policy_mean, policy_log_std, log_pi = policy_outputs[:4]

        t, b, _ = obs.size()
        obs_f = obs.view(t * b, -1)
        act_f = actions.view(t * b, -1)
        next_f = next_obs.view(t * b, -1)

        # Dual-form KL discriminator update
        div_estimate = self.divergence.dual_estimate(obs_f, new_actions, act_f, task_z)
        c_loss = self.divergence.dual_critic_loss(obs_f, new_actions, act_f, task_z)
        self.c_optimizer.zero_grad()
        c_loss.backward(retain_graph=True)
        self.c_optimizer.step()

        # Encoder update: distance metric loss only
        self.context_optimizer.zero_grad()
        z_loss = self.z_loss_weight * focal_z_loss(task_z, indices)
        z_loss.backward(retain_graph=True)
        self.context_optimizer.step()

        # Critic update on detached z with optional value penalty
        z_for_q = task_z.detach()
        q1 = self.qf1(obs_f, act_f, z_for_q)
        q2 = self.qf2(obs_f, act_f, z_for_q)
        v_pred = self.vf(obs_f, z_for_q)
        with torch.no_grad():
            target_v = self.target_vf(next_f, task_z)
            alpha = torch.clamp(self._alpha, 0.0, self.alpha_max)
            if self.use_value_penalty:
                target_v = target_v - alpha * div_estimate
        rewards_f = rewards.view(self.batch_size * num_tasks, -1) * self.reward_scale
        terms_f = terms.view(self.batch_size * num_tasks, -1)
        q_target = rewards_f + (1.0 - terms_f) * self.discount * target_v
        qf_loss = torch.mean((q1 - q_target) ** 2) + torch.mean((q2 - q_target) ** 2)
        self.qf1_optimizer.zero_grad()
        self.qf2_optimizer.zero_grad()
        qf_loss.backward()
        self.qf1_optimizer.step()
        self.qf2_optimizer.step()

        # Value update
        min_q = torch.min(
            self.qf1(obs_f, new_actions, z_for_q),
            self.qf2(obs_f, new_actions, z_for_q),
        )
        v_target = min_q - log_pi if self.max_entropy else min_q
        vf_loss = F.mse_loss(v_pred, v_target.detach())
        self.vf_optimizer.zero_grad()
        vf_loss.backward()
        self.vf_optimizer.step()
        soft_update(self.vf, self.target_vf, self.soft_target_tau)

        # Policy update with behavior regularization
        if self.max_entropy:
            policy_loss = (log_pi - min_q + alpha.detach() * div_estimate).mean()
        else:
            policy_loss = (-min_q + alpha.detach() * div_estimate).mean()
        policy_loss = policy_loss + 1e-3 * (policy_mean ** 2).mean() + 1e-3 * (policy_log_std ** 2).mean()
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # Adaptive alpha
        alpha_loss = -(self._alpha * (div_estimate - self.target_divergence).detach()).mean()
        alpha_loss.backward()
        with torch.no_grad():
            self._alpha -= self.alpha_lr * self._alpha.grad
            self._alpha.grad.zero_()
```
