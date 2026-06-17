# Soft Actor-Critic (SAC), distilled

SAC is an off-policy actor-critic for continuous control built on the maximum-entropy RL
objective. It maximizes expected return *plus* policy entropy, so exploration is something the
policy optimizes rather than an injected noise process. A stochastic, tanh-squashed Gaussian
actor is trained against a soft Q-critic by a reparameterized (pathwise) gradient; twin critics
with a min curb value overestimation; and the entropy temperature is tuned automatically as a
Lagrange dual variable that holds the policy's expected entropy at a target.

## Problem it solves

Model-free deep RL for continuous control was either sample-efficient but brittle (DDPG-style
off-policy, deterministic actor chasing an overestimated critic, with hand-scheduled
exploration noise) or stable but sample-hungry (on-policy TRPO/PPO/A3C). SAC is designed to
combine off-policy data reuse with the stability and exploration of a max-entropy stochastic
policy, while removing the hand-tuned exploration and temperature knobs.

## Key idea

Optimize the maximum-entropy objective

```
J(pi) = sum_t E_{(s,a)~rho_pi} [ r(s,a) + alpha * H(pi(.|s)) ],
```

and push the entropy *inside* the value function so it propagates through the bootstrap:

```
soft value:   V(s)      = E_{a~pi}[ Q(s,a) - alpha * log pi(a|s) ]
soft backup:  T^pi Q    = r(s,a) + gamma * E_{s'}[ V(s') ]
```

Derived from soft policy iteration (provably convergent in the tabular case), the practical
algorithm is:

- **Critic** minimizes the soft Bellman residual; the bootstrap target evaluates the *current*
  policy at the next state and includes the entropy bonus, using the min of twin target critics
  (TD3-style, against overestimation):
  ```
  a' ~ pi(.|s');   y = r + gamma * ( min_i Q_targ_i(s', a') - alpha * log pi(a'|s') )
  L_Q = MSE(Q_1(s,a), y) + MSE(Q_2(s,a), y)
  ```
- **Actor** minimizes the KL projection onto exp(Q/alpha), which (dropping the log-partition
  constant) is `E[ alpha log pi - Q ]`, estimated with a reparameterized action a = f_phi(eps;s)
  for a low-variance pathwise gradient:
  ```
  a, log pi ~ pi_phi(.|s);   L_pi = E[ alpha * log pi(a|s) - min_i Q_i(s, a) ]
  ```
- **Temperature** alpha is a learned dual variable enforcing an expected-entropy constraint
  E[-log pi] >= H_bar, target entropy H_bar = -dim(A):
  ```
  L_alpha = E[ -alpha * ( log pi(a|s) + H_bar ) ]   (one dual-gradient step; optimize log alpha)
  ```
- **Targets** updated by Polyak averaging: theta_bar <- tau*theta + (1-tau)*theta_bar.

## Why each piece

- **Entropy in the objective, inside V:** exploration the policy chooses; propagated through the
  Bellman recursion so it shapes long-horizon behavior, unlike a bare entropy bonus on the actor.
- **Stochastic vs deterministic actor:** stochasticity is the exploration, so no external noise
  schedule has to be tuned.
- **Reparameterization (vs likelihood-ratio):** Q is differentiable in a, so the pathwise
  gradient is available and far lower variance — this is DDPG's policy gradient extended to a
  stochastic policy.
- **min of twin Q (TD3):** bootstrapped Q can overestimate regardless of entropy; the min
  trades some underestimation for protection against actor updates chasing positive error spikes.
- **No separate V network:** V is determined by Q and pi and estimated unbiasedly from one
  action sample, so it is folded into the Q target.
- **Automatic temperature:** the optimal alpha varies across tasks and across training; the dual
  formulation makes alpha self-adjust to hold expected entropy at H_bar, removing the only
  brittle knob (reward scale).
- **tanh squashing + change of variables:** bounds actions with `a = scale*tanh(u)+bias` and
  exact log-prob
  `log pi(a|s) = log mu(u|s) - sum_i log(scale_i * (1 - tanh^2(u_i)))`, with +1e-6 guarding
  the boundary; log_std is mapped to [-5, 2] for numerical stability.
- **Polyak target, small tau (~0.005):** slow target reduces TD-error variance; large tau
  destabilizes, tiny tau slows learning.

## Final algorithm

```
Initialize theta_1, theta_2 (critics), phi (actor), log alpha; theta_bar_i <- theta_i; D <- {}
for each iteration:
    for each env step:
        a ~ pi_phi(.|s);  s' ~ p(.|s,a);  D <- D u {(s, a, r, s')}
    for each gradient step:
        sample minibatch from D
        a' ~ pi_phi(.|s');  y = r + gamma * ( min_i Q_{theta_bar_i}(s', a') - alpha * log pi_phi(a'|s') )
        theta_i <- theta_i - lr_Q * grad ( MSE(Q_{theta_i}(s,a), y) )       for i in {1,2}
        on policy-update steps:
            a, log pi ~ pi_phi(.|s);  phi <- phi - lr_pi * grad ( alpha*log pi - min_i Q_{theta_i}(s,a) )
            log alpha <- log alpha - lr_alpha * grad ( -alpha * (log pi + H_bar) )
        on target-update steps:
            theta_bar_i <- tau*theta_i + (1-tau)*theta_bar_i                for i in {1,2}
```

Hyperparameters (canonical continuous-control defaults): optimizer Adam, policy learning rate
3e-4, Q/temperature learning rate 1e-3 in CleanRL, discount gamma 0.99, replay buffer 1e6,
two hidden layers of 256 units, ReLU, batch 256, tau 0.005, target update interval 1, policy
frequency 2 with two compensated policy updates, entropy target -dim(A).

## Working code

Standard continuous-control SAC with automatic temperature tuning (tanh-squashed Gaussian
actor, twin soft critics, reparameterized actor loss). This fills the actor / critic / update
slots of the off-policy actor-critic harness.

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
