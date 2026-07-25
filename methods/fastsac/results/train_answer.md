I want a single model-free algorithm for continuous control that is both sample-efficient and stable, and the two established families each fail on a different axis. On-policy policy gradients such as TRPO, PPO, and A3C are reliable, but every gradient step needs fresh trajectories, so a batch is thrown away after a handful of updates; on the hardest task in the suite, the 21-dimensional Humanoid, that sample bill becomes enormous. The off-policy alternative, DDPG, reuses a replay buffer and is far more data-efficient, but it is brittle in a specific, traceable way. DDPG drives a deterministic actor $\mu_\phi(s)$ uphill on a critic $Q_\theta(s,a)$ by the deterministic policy gradient $\nabla_\phi J = \mathbb{E}[\nabla_a Q_\theta(s,a)|_{a=\mu_\phi(s)}\,\nabla_\phi \mu_\phi(s)]$; the critic is trained on bootstrapped Bellman targets, and bootstrapped value estimates carry a well-documented upward bias — a max-like operator over a noisy estimate is a systematic overestimate, and the Bellman backup propagates that bias forward. An actor whose only instruction is "maximize the current critic" walks straight into wherever the critic is most overestimated, so actor and critic end up chasing each other's errors. Worse, because the actor is deterministic it has no exploration of its own, so an external noise process has to be injected and scheduled by hand — one more fragile, per-task knob. TD3 patches the value side by keeping two independently trained critics and bootstrapping from their minimum: a value that is overestimated in one network is unlikely to also be the smaller of the two, so the min systematically favors underestimation, and underestimation, unlike overestimation, does not get amplified through the backup. But TD3's policy is still deterministic and its exploration is still an externally bolted-on noise process; the objective the policy actually optimizes still has no reason to explore.

The way out is to make exploration a property of the objective itself rather than an external process, while keeping the value estimator honest. I propose Soft Actor-Critic (SAC), an off-policy actor-critic that augments the ordinary return with the entropy of the policy,
$$ J(\pi) = \sum_t \mathbb{E}_{(s_t,a_t)\sim\rho_\pi}\Big[\, r(s_t,a_t) + \alpha\,\mathcal{H}\big(\pi(\cdot\,|\,s_t)\big) \,\Big], $$
with a temperature $\alpha$ trading off reward against randomness, and $\alpha\to 0$ recovering ordinary RL. The load-bearing move is not to bolt an entropy bonus onto the actor loss alone but to push it inside the value function itself, so the entropy the policy will collect in the future appears in the bootstrap target and shapes long-horizon behavior rather than just the immediate action:
$$ V(s) = \mathbb{E}_{a\sim\pi}\big[\,Q(s,a) - \alpha\log\pi(a|s)\,\big], \qquad \mathcal{T}^\pi Q(s,a) = r(s,a) + \gamma\,\mathbb{E}_{s'}\big[V(s')\big]. $$
Concretely, the critic is a pair of twin Q-networks $Q_{\theta_1}, Q_{\theta_2}$ — guarding against overestimation exactly as in TD3 — trained on the soft Bellman residual, where the bootstrap target evaluates the *current* policy at the next state, folds in its entropy bonus, and takes the minimum of the two *target* critics:
$$ a' \sim \pi_\phi(\cdot\,|\,s'), \qquad y = r + \gamma\Big(\min_i Q_{\bar\theta_i}(s',a') - \alpha\log\pi_\phi(a'|s')\Big), \qquad \mathcal{L}_Q = \mathrm{MSE}\big(Q_{\theta_1}(s,a), y\big) + \mathrm{MSE}\big(Q_{\theta_2}(s,a), y\big). $$
The actor is a Gaussian whose mean and log-std are produced by a shared two-layer ReLU trunk; the log-std is passed through $\tanh$ and re-mapped into $[-5, 2]$ for numerical stability. An action is drawn by the reparameterization trick, $u = \mu(s) + \sigma(s)\odot\varepsilon$ with $\varepsilon\sim\mathcal{N}(0,I)$, then squashed by $\tanh$ and rescaled into the action box, $a = \tanh(u)\cdot\text{scale} + \text{bias}$. Because $a$ is a deterministic, differentiable function of $\varepsilon$, the actor loss $\mathbb{E}[\alpha\log\pi(a|s) - \min_i Q_{\theta_i}(s,a)]$ can be minimized by a single low-variance pathwise gradient in place of a high-variance likelihood-ratio estimator — DDPG's deterministic policy gradient, extended to a stochastic policy. Squashing through $\tanh$ changes the density, so the exact log-probability needs the change-of-variables correction $\log\pi(a|s) = \log\mu(u|s) - \sum_i \log\big(\text{scale}_i\,(1-\tanh^2(u_i))\big)$, with a $10^{-6}$ term guarding the boundary where the Jacobian vanishes. Rather than hand-set $\alpha$, I treat it as the Lagrange multiplier of a constraint that holds the policy's expected entropy at a target, $\mathbb{E}[-\log\pi] \ge \bar{\mathcal{H}}$, with $\bar{\mathcal{H}} = -\dim(A)$ so the target scales with the action dimension; descending the gradient of $J(\alpha) = \mathbb{E}\big[-\alpha\big(\log\pi(a|s) + \bar{\mathcal{H}}\big)\big]$ on $\log\alpha$ (so $\alpha$ stays positive) gives a self-correcting thermostat — when the policy is too deterministic $\alpha$ rises and forces more entropy, when it is too random $\alpha$ falls and the policy is allowed to commit, and the sign works out the same regardless of whether the differential-entropy target itself is negative. The alpha step uses its own freshly drawn, detached sample of the log-probability rather than reusing the one from the actor step, since the two updates are adjusting different things — the multiplier versus the policy. Target critics move only by Polyak averaging, $\bar\theta_i \leftarrow \tau\theta_i + (1-\tau)\bar\theta_i$ with a small $\tau$ so the bootstrap target moves slowly; the actor and temperature are updated on a delay relative to the critic, but with a compensating number of repeated updates so the long-run ratio of actor to critic steps stays fixed. Hyperparameters throughout: Adam for all three optimizers, policy learning rate $3\times10^{-4}$, Q and temperature learning rate $10^{-3}$, discount $\gamma = 0.99$, a replay buffer of $10^6$ transitions, two hidden layers of 256 units with ReLU, batch size 256, $\tau = 0.005$, policy frequency 2 with two compensated policy updates, target update interval 1. At evaluation the exploration is turned off and I act with the $\tanh$-squashed mean action rather than a sample.

Assembled into modules, this is the twin-critic, squashed-Gaussian actor, and the three update rules that plug into the off-policy training harness:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

LOG_STD_MIN, LOG_STD_MAX = -5, 2


class QNetwork(nn.Module):
    def __init__(self, n_obs, n_act, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(n_obs + n_act, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, obs, action):
        x = torch.cat([obs, action], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class Actor(nn.Module):
    def __init__(self, n_obs, n_act, action_low=None, action_high=None, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(n_obs, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mean = nn.Linear(hidden_dim, n_act)
        self.fc_logstd = nn.Linear(hidden_dim, n_act)

        if action_low is None or action_high is None:
            action_scale = torch.ones(n_act)
            action_bias = torch.zeros(n_act)
        else:
            low = torch.as_tensor(action_low, dtype=torch.float32)
            high = torch.as_tensor(action_high, dtype=torch.float32)
            action_scale = (high - low) / 2.0
            action_bias = (high + low) / 2.0
        self.register_buffer("action_scale", action_scale)
        self.register_buffer("action_bias", action_bias)

    def forward(self, obs):
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = torch.tanh(self.fc_logstd(x))
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1.0)
        return mean, log_std

    def get_action(self, obs):
        mean, log_std = self.forward(obs)
        normal = torch.distributions.Normal(mean, log_std.exp())
        u = normal.rsample()
        y = torch.tanh(u)
        action = y * self.action_scale + self.action_bias
        log_prob = normal.log_prob(u)
        log_prob -= torch.log(self.action_scale * (1 - y.pow(2)) + 1e-6)
        log_prob = log_prob.sum(-1, keepdim=True)
        mean_action = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean_action


def build_algorithm(n_obs, n_act, device, action_low=None, action_high=None,
                    policy_lr=3e-4, q_lr=1e-3):
    actor = Actor(n_obs, n_act, action_low, action_high).to(device)
    qf1, qf2 = QNetwork(n_obs, n_act).to(device), QNetwork(n_obs, n_act).to(device)
    qf1_target = QNetwork(n_obs, n_act).to(device)
    qf2_target = QNetwork(n_obs, n_act).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    log_alpha = torch.zeros(1, requires_grad=True, device=device)
    return {
        "actor": actor, "qf1": qf1, "qf2": qf2,
        "qf1_target": qf1_target, "qf2_target": qf2_target,
        "log_alpha": log_alpha, "target_entropy": -float(n_act),
        "actor_opt": torch.optim.Adam(actor.parameters(), lr=policy_lr),
        "q_opt": torch.optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=q_lr),
        "alpha_opt": torch.optim.Adam([log_alpha], lr=q_lr),
    }


def update_critic(batch, c, gamma):
    actor = c["actor"]
    qf1, qf2 = c["qf1"], c["qf2"]
    qf1_target, qf2_target = c["qf1_target"], c["qf2_target"]
    alpha = c["log_alpha"].exp().detach()
    obs, action, reward, next_obs, done = batch
    with torch.no_grad():
        next_action, next_logp, _ = actor.get_action(next_obs)
        qf1_next = qf1_target(next_obs, next_action)
        qf2_next = qf2_target(next_obs, next_action)
        min_q_next = torch.min(qf1_next, qf2_next) - alpha * next_logp
        next_q = reward + (1.0 - done) * gamma * min_q_next
    qf1_a = qf1(obs, action)
    qf2_a = qf2(obs, action)
    q_loss = F.mse_loss(qf1_a, next_q) + F.mse_loss(qf2_a, next_q)
    c["q_opt"].zero_grad(); q_loss.backward(); c["q_opt"].step()
    return q_loss


def update_actor(batch, c):
    actor, qf1, qf2 = c["actor"], c["qf1"], c["qf2"]
    alpha = c["log_alpha"].exp().detach()
    obs = batch[0]
    pi, logp, _ = actor.get_action(obs)
    min_q_pi = torch.min(qf1(obs, pi), qf2(obs, pi))
    actor_loss = (alpha * logp - min_q_pi).mean()
    c["actor_opt"].zero_grad(); actor_loss.backward(); c["actor_opt"].step()
    with torch.no_grad():
        _, logp_alpha, _ = actor.get_action(obs)
    alpha_loss = (-c["log_alpha"].exp() * (logp_alpha + c["target_entropy"])).mean()
    c["alpha_opt"].zero_grad(); alpha_loss.backward(); c["alpha_opt"].step()
    return actor_loss, alpha_loss


@torch.no_grad()
def soft_update(src, tgt, tau):
    for p, p_t in zip(src.parameters(), tgt.parameters()):
        p_t.mul_(1.0 - tau).add_(p, alpha=tau)


def train_step(batch, c, gamma, tau, global_step,
               policy_frequency=2, target_network_frequency=1):
    q_loss = update_critic(batch, c, gamma)
    actor_loss = alpha_loss = None
    if global_step % policy_frequency == 0:
        for _ in range(policy_frequency):
            actor_loss, alpha_loss = update_actor(batch, c)
    if global_step % target_network_frequency == 0:
        soft_update(c["qf1"], c["qf1_target"], tau)
        soft_update(c["qf2"], c["qf2_target"], tau)
    return q_loss, actor_loss, alpha_loss
```
