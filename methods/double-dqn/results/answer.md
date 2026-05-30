# Double DQN

## Problem

Q-learning, and its deep-network instantiation DQN, learns the optimal action
value by regressing $Q(S_t,A_t)$ toward a bootstrap target that contains a
maximization over estimated next-state values,
$$
Y^{\text{DQN}}_t = R_{t+1} + \gamma \max_a Q(S_{t+1}, a; \theta^-).
$$
Because the same estimates are used both to *select* the greedy next action and
to *evaluate* it, and because $\max$ is convex, the target is biased *upward*
whenever the estimates carry error: by Jensen,
$\mathbb{E}[\max_a Q(s',a)] \ge \max_a \mathbb{E}[Q(s',a)]$. The bias is
non-uniform across states, so it distorts the relative value ordering the greedy
policy reads off, and bootstrapping propagates it. Empirically the predicted
value of the greedy policy runs far above its realized return, and on some tasks
diverges while the score collapses.

## Key idea

The overestimation comes from coupling selection and evaluation. **Decouple
them**: let one estimator pick the greedy action and a different one score it.
Reuse the target network — already present in DQN — as the second estimator, so
nothing new is added. The online network $\theta$ selects; the target network
$\theta^-$ evaluates:
$$
Y^{\text{DoubleDQN}}_t = R_{t+1} + \gamma\, Q\big(S_{t+1},\ \arg\max_a Q(S_{t+1}, a; \theta),\ \theta^-\big).
$$
This is the minimal change to DQN: everything else (network, replay, $\epsilon$-greedy,
periodic target copy) is untouched. It reduces but does not perfectly remove the
bias, because $\theta^-$ is a stale copy of $\theta$ (so their errors are
correlated, and right after a target refresh $\theta^-=\theta$ and the target
reverts to plain Q-learning). A longer target-update period $\tau$ decouples them
further.

## Why it works

In a state where all true optimal values are equal, $Q_*(s,a)=V_*(s)$, with
estimates that are balanced ($\sum_a(Q_t(s,a)-V_*(s))=0$) and have mean-squared
error $C>0$ over $m\ge2$ actions, the single-estimator max overshoots by at least
$$
\max_a Q_t(s,a) \ge V_*(s) + \sqrt{\tfrac{C}{m-1}},
$$
and this is tight (attained by $\epsilon_a=\sqrt{C/(m-1)}$ for $m-1$ actions and
$\epsilon_m=-\sqrt{(m-1)C}$); no independence assumption is needed. Under the
decoupled estimate the lower bound on the absolute error is $0$ — an independent
evaluator can be exactly right on the selected action. (Typical-case, with errors
i.i.d. uniform on $[-1,1]$, $\mathbb{E}[\max_a\epsilon_a]=\frac{m-1}{m+1}$, which
*increases* with the number of actions.)

## Algorithm

Identical to DQN except for the target.

1. Act $\epsilon$-greedily w.r.t. $Q(\cdot;\theta)$; store
   $(S_t,A_t,R_{t+1},S_{t+1},\text{done})$ in a uniform replay buffer.
2. Every few steps, sample a minibatch. For each transition compute
   $a^\star = \arg\max_a Q(S_{t+1},a;\theta)$ (online selection),
   $y = R_{t+1} + \gamma\,(1-\text{done})\,Q(S_{t+1},a^\star;\theta^-)$ (target evaluation).
3. Take a gradient step on $\big(y - Q(S_t,A_t;\theta)\big)^2$ w.r.t. $\theta$.
4. Every $\tau$ steps copy $\theta^- \leftarrow \theta$.

## Code

Grounded in a standard DQN-Atari training loop; only the target computation
differs from DQN (one line: the argmax uses the online network).

```python
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim


class QNetwork(nn.Module):
    """Stacked-frame state -> vector of action values (online and target share this)."""
    def __init__(self, num_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512), nn.ReLU(),
            nn.Linear(512, num_actions),
        )

    def forward(self, x):
        return self.net(x / 255.0)


def double_dqn_loss(batch, gamma, online_net, target_net):
    obs, actions, rewards, next_obs, dones = batch
    with torch.no_grad():
        # online net SELECTS the greedy next action (vs DQN: target_net.argmax)
        next_actions = online_net(next_obs).argmax(dim=1, keepdim=True)
        # target net EVALUATES the selected action
        next_q = target_net(next_obs).gather(1, next_actions).squeeze(1)
        td_target = rewards + gamma * next_q * (1.0 - dones)
    q_sa = online_net(obs).gather(1, actions).squeeze(1)   # Q(S_t, A_t; theta)
    return F.mse_loss(td_target, q_sa)


def train(envs, num_actions, total_timesteps=10_000_000, gamma=0.99,
          batch_size=32, lr=1e-4, target_update=10_000, train_freq=4,
          learning_starts=80_000, buffer=None, device="cpu"):
    online_net = QNetwork(num_actions).to(device)
    target_net = QNetwork(num_actions).to(device)
    target_net.load_state_dict(online_net.state_dict())
    optimizer = optim.Adam(online_net.parameters(), lr=lr)

    obs = envs.reset()
    for step in range(total_timesteps):
        epsilon = linear_schedule(1.0, 0.01, 0.1 * total_timesteps, step)
        if random.random() < epsilon:
            actions = envs.sample_actions()
        else:
            with torch.no_grad():
                q = online_net(torch.as_tensor(obs, device=device))
            actions = q.argmax(dim=1).cpu().numpy()

        next_obs, rewards, dones, _ = envs.step(actions)
        buffer.add(obs, actions, rewards, next_obs, dones)
        obs = next_obs

        if step > learning_starts and step % train_freq == 0:
            batch = buffer.sample(batch_size)
            loss = double_dqn_loss(batch, gamma, online_net, target_net)
            optimizer.zero_grad(); loss.backward(); optimizer.step()

        if step % target_update == 0:
            target_net.load_state_dict(online_net.state_dict())
```

Standard Atari settings: $84\times84$ gray-scale, 4-frame stack, action repeat 4,
rewards clipped to $[-1,1]$, replay buffer $10^6$, batch 32, $\gamma=0.99$,
$\tau=10{,}000$ target updates (raise to $30{,}000$ for the tuned variant),
$\epsilon$ decaying $1\to0.1$. The only difference from DQN is the
`online_net(next_obs).argmax(...)` selection inside `double_dqn_loss`.
