I want a continuous-control algorithm that is both sample-efficient and stable. The existing families each fail on one side. On-policy policy gradients such as PPO are reliable, but they discard each batch of trajectories after only a few updates; under a fixed frame budget this throughput is prohibitive, especially for high-dimensional humanoid locomotion where long-horizon gait coordination needs many passes over the data. Off-policy deterministic actor-critics such as DDPG and TD3 reuse experience through a replay buffer, but they are brittle: a deterministic actor is trained to maximize the learned critic, and bootstrapped critics systematically overestimate values, so the actor chases its own optimistic errors. A deterministic policy also has no exploration of its own, forcing us to inject and schedule external noise, which is another fragile per-task knob.

The way out is to make exploration a property of the objective rather than an add-on, and to keep the value estimator honest. I propose FastSAC, a maximum-entropy off-policy actor-critic that marries the Soft Actor-Critic framework with a categorical distributional critic. Instead of maximizing reward alone, it maximizes reward plus the entropy of the policy. The entropy term is not merely added to the actor loss; it is pushed inside the soft value function V(s) = E_{a~pi}[Q(s,a) - alpha log pi(a|s)], so the future entropy the policy will collect appears in the Bellman target and shapes long-horizon behavior. A bonus only on the actor would shape the immediate action but would never propagate through the bootstrap; putting it inside V makes the critic aware of the exploration it is paying for. This gives directed, objective-driven exploration without a hand-tuned noise schedule.

FastSAC instantiates this idea on a categorical distributional substrate. The critic predicts a probability distribution over a fixed grid of return atoms rather than a single scalar, which keeps distinct return modes separated and stabilizes value learning when experience is reused heavily. To fold entropy into the distributional backup, the entropy bonus is subtracted from the reward before the categorical projection: the target distribution is projected from reward - alpha log pi(a'|s') plus discounted bootstrap. Two independent critics are maintained, and the target distribution is taken from whichever critic has the lower expected value, inheriting TD3's clipped-double-Q protection against overestimation. The actor is a tanh-squashed Gaussian trained by the reparameterized gradient E[alpha log pi(a|s) - min(Q1, Q2)], so the stochastic policy both explores and improves with a low-variance pathwise gradient.

The temperature alpha is not hand-set. It is treated as the dual variable of an expected-entropy constraint E[-log pi] >= H_bar, optimized by gradient descent on log alpha. If the policy is too deterministic, alpha rises and forces more entropy; if too random, alpha falls. The target H_bar = -dim(A) scales automatically with the action dimension. The architecture uses LayerNorm and SiLU in both actor and critic to control feature magnitudes under heavy off-policy reuse, AdamW with weight decay, cosine learning-rate annealing, and a fast Polyak target update tau = 0.1 with two gradient steps per environment step. At evaluation time act with the deterministic mean action tanh(mu(s)). The result is an off-policy algorithm that revisits every transition many times, explores through entropy rather than external noise, and self-tunes its stochasticity.

```python
# FastSAC: max-entropy off-policy actor-critic with a categorical distributional critic
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

LOG_STD_MIN, LOG_STD_MAX = -5, 2


class Actor(nn.Module):
    """Tanh-squashed Gaussian actor with LayerNorm + SiLU body."""
    def __init__(self, n_obs, n_act, hidden_dim=512, device="cpu"):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_obs, hidden_dim, device=device), nn.LayerNorm(hidden_dim, device=device), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2, device=device), nn.LayerNorm(hidden_dim // 2, device=device), nn.SiLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4, device=device), nn.LayerNorm(hidden_dim // 4, device=device), nn.SiLU(),
        )
        self.fc_mean = nn.Linear(hidden_dim // 4, n_act, device=device)
        self.fc_logstd = nn.Linear(hidden_dim // 4, n_act, device=device)

    def forward(self, obs):
        x = self.net(obs)
        return self.fc_mean(x), torch.clamp(self.fc_logstd(x), LOG_STD_MIN, LOG_STD_MAX)

    def get_action(self, obs):
        mean, log_std = self.forward(obs)
        normal = torch.distributions.Normal(mean, log_std.exp())
        u = normal.rsample()
        action = torch.tanh(u)
        log_prob = normal.log_prob(u) - torch.log(1 - action.pow(2) + 1e-6)
        return action, log_prob.sum(-1)

    def deterministic_action(self, obs):
        mean, _ = self.forward(obs)
        return torch.tanh(mean)


class DistributionalQNetwork(nn.Module):
    """Categorical critic with LayerNorm + SiLU."""
    def __init__(self, n_obs, n_act, num_atoms, v_min, v_max, hidden_dim, device="cpu"):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_obs + n_act, hidden_dim, device=device), nn.LayerNorm(hidden_dim, device=device), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2, device=device), nn.LayerNorm(hidden_dim // 2, device=device), nn.SiLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4, device=device), nn.LayerNorm(hidden_dim // 4, device=device), nn.SiLU(),
            nn.Linear(hidden_dim // 4, num_atoms, device=device),
        )
        self.v_min, self.v_max, self.num_atoms = v_min, v_max, num_atoms

    def forward(self, obs, actions):
        return self.net(torch.cat([obs, actions], 1))

    def projection(self, obs, actions, rewards, bootstrap, discount, q_support, device):
        delta_z = (self.v_max - self.v_min) / (self.num_atoms - 1)
        batch_size = rewards.shape[0]
        target_z = (rewards.unsqueeze(1) + bootstrap.unsqueeze(1) * discount.unsqueeze(1) * q_support).clamp(self.v_min, self.v_max)
        b = (target_z - self.v_min) / delta_z
        l = torch.floor(b).long(); u = torch.ceil(b).long()
        is_int = (l == u); l_mask = is_int & (l > 0); u_mask = is_int & (l == 0)
        l = torch.where(l_mask, l - 1, l); u = torch.where(u_mask, u + 1, u)
        next_dist = F.softmax(self.forward(obs, actions), dim=1)
        proj_dist = torch.zeros_like(next_dist)
        offset = torch.linspace(0, (batch_size - 1) * self.num_atoms, batch_size, device=device).unsqueeze(1).expand(batch_size, self.num_atoms).long()
        proj_dist.view(-1).index_add_(0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1))
        proj_dist.view(-1).index_add_(0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1))
        return proj_dist


class Critic(nn.Module):
    def __init__(self, n_obs, n_act, num_atoms, v_min, v_max, hidden_dim, device="cpu"):
        super().__init__()
        self.qnet1 = DistributionalQNetwork(n_obs, n_act, num_atoms, v_min, v_max, hidden_dim, device)
        self.qnet2 = DistributionalQNetwork(n_obs, n_act, num_atoms, v_min, v_max, hidden_dim, device)
        self.register_buffer("q_support", torch.linspace(v_min, v_max, num_atoms, device=device))
        self.device = device

    def forward(self, obs, actions):
        return self.qnet1(obs, actions), self.qnet2(obs, actions)

    def projection(self, obs, actions, rewards, bootstrap, discount):
        q1 = self.qnet1.projection(obs, actions, rewards, bootstrap, discount, self.q_support, self.q_support.device)
        q2 = self.qnet2.projection(obs, actions, rewards, bootstrap, discount, self.q_support, self.q_support.device)
        return q1, q2

    def get_value(self, probs):
        return torch.sum(probs * self.q_support, dim=1)


def build_algorithm(n_obs, n_act, device, num_atoms=101, v_min=-250, v_max=250,
                    actor_hidden=512, critic_hidden=1024,
                    actor_lr=3e-4, critic_lr=3e-4, weight_decay=0.1):
    actor = Actor(n_obs, n_act, actor_hidden, device)
    critic = Critic(n_obs, n_act, num_atoms, v_min, v_max, critic_hidden, device)
    critic_target = Critic(n_obs, n_act, num_atoms, v_min, v_max, critic_hidden, device)
    critic_target.load_state_dict(critic.state_dict())
    log_alpha = torch.tensor(np.log(0.2), device=device, requires_grad=True)
    return {
        "actor": actor.to(device), "critic": critic.to(device), "critic_target": critic_target.to(device),
        "log_alpha": log_alpha, "target_entropy": -float(n_act),
        "actor_opt": torch.optim.AdamW(actor.parameters(), lr=actor_lr, weight_decay=weight_decay),
        "critic_opt": torch.optim.AdamW(critic.parameters(), lr=critic_lr, weight_decay=weight_decay),
        "alpha_opt": torch.optim.Adam([log_alpha], lr=actor_lr),
    }


def update_critic(batch, components, gamma):
    actor, critic, critic_target = components["actor"], components["critic"], components["critic_target"]
    alpha = components["log_alpha"].exp().detach()
    obs, action, reward, next_obs, done = batch
    bootstrap = (1.0 - done).float()
    discount = torch.full_like(reward, gamma)

    with torch.no_grad():
        next_action, next_log_prob = actor.get_action(next_obs)
        qf1_next_proj, qf2_next_proj = critic_target.projection(
            next_obs, next_action, reward - alpha * next_log_prob, bootstrap, discount)
        qf1_next_val = critic_target.get_value(qf1_next_proj)
        qf2_next_val = critic_target.get_value(qf2_next_proj)
        qf_next_dist = torch.where(qf1_next_val.unsqueeze(1) < qf2_next_val.unsqueeze(1), qf1_next_proj, qf2_next_proj)

    qf1, qf2 = critic(obs, action)
    qf1_loss = -torch.sum(qf_next_dist * F.log_softmax(qf1, dim=1), dim=1).mean()
    qf2_loss = -torch.sum(qf_next_dist * F.log_softmax(qf2, dim=1), dim=1).mean()
    qf_loss = qf1_loss + qf2_loss
    components["critic_opt"].zero_grad(); qf_loss.backward(); components["critic_opt"].step()
    return qf_loss


def update_actor(batch, components):
    actor, critic = components["actor"], components["critic"]
    alpha = components["log_alpha"].exp().detach()
    obs = batch[0]

    new_action, log_prob = actor.get_action(obs)
    qf1_a, qf2_a = critic(obs, new_action)
    qf1_v = critic.get_value(F.softmax(qf1_a, dim=1))
    qf2_v = critic.get_value(F.softmax(qf2_a, dim=1))
    actor_loss = (alpha * log_prob - torch.minimum(qf1_v, qf2_v)).mean()
    components["actor_opt"].zero_grad(); actor_loss.backward(); components["actor_opt"].step()

    alpha_loss = -(components["log_alpha"].exp() * (log_prob.detach() + components["target_entropy"])).mean()
    components["alpha_opt"].zero_grad(); alpha_loss.backward(); components["alpha_opt"].step()
    return actor_loss, alpha_loss


@torch.no_grad()
def soft_update(src, tgt, tau):
    for p, p_t in zip(src.parameters(), tgt.parameters()):
        p_t.mul_(1.0 - tau).add_(p, alpha=tau)
```
