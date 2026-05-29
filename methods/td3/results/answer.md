# TD3 — Twin Delayed Deep Deterministic policy gradient

## Problem

Continuous-control actor-critic methods that pair a deterministic actor with a learned critic (the deterministic-policy-gradient family, of which DDPG is the standard) suffer from function-approximation error in the critic. Because the actor ascends the critic's gradient, any region the critic overrates attracts the policy, which generates more data there and inflates the estimate further — a self-reinforcing overestimation loop. Bootstrapping then accumulates this error as variance, and deterministic targets read the critic at single sharp points that function approximation makes unreliable. The result is value estimates that drift far above achievable returns, brittle and seed-sensitive learning, and occasional divergence.

## Key idea

Attack function-approximation error from three directions, each a drop-in modification to a deterministic actor-critic:

1. **Clipped Double Q-learning.** Keep two critics and form the TD target from the *minimum* of the two. If the second critic is above the first, the update reduces to the standard single-critic target; if it is below the first, the target is pulled down in the Double-Q direction. The residual bias is therefore pushed toward *under*estimation, which is safer because the actor tends not to select and amplify low-valued actions, unlike overestimated actions that it actively chases.

2. **Delayed policy (and target) updates.** Update the actor and the target networks only once every $d$ critic updates, with slow soft target updates. This lets the critic's error settle before the policy moves against it, aiming at lower-variance policy gradients through a two-timescale scheme: critic fast, actor slow.

3. **Target policy smoothing.** Add small clipped Gaussian noise to the target action before evaluating the critic, so the target reflects the value of a neighborhood of the action rather than one knife-edge point. This is a SARSA-style regularizer that smooths the value surface across similar actions and prevents the deterministic actor from overfitting narrow, spurious value peaks.

## Final algorithm

Maintain twin critics $Q_{\theta_1},Q_{\theta_2}$, a single actor $\pi_\phi$, and their target copies $\theta'_1,\theta'_2,\phi'$. For each environment step, sample a minibatch and:

Form the smoothed, clipped target action and the clipped double-Q target:
$$\tilde a = \mathrm{clip}\big(\pi_{\phi'}(s') + \epsilon,\,-a_{\max},\,a_{\max}\big),\qquad \epsilon\sim\mathrm{clip}(\mathcal N(0,\tilde\sigma),\,-c,\,c),$$
$$y = r + \gamma\,\min_{i=1,2} Q_{\theta'_i}(s',\tilde a).$$
Take a gradient step on both critics using the shared target:
$$L(\theta_1,\theta_2)=N^{-1}\sum\big(y - Q_{\theta_1}(s,a)\big)^2 + N^{-1}\sum\big(y - Q_{\theta_2}(s,a)\big)^2.$$
Every $d$ steps, update the actor by the deterministic policy gradient through $Q_{\theta_1}$ and soft-update the targets:
$$\nabla_\phi J(\phi) = N^{-1}\sum \nabla_a Q_{\theta_1}(s,a)\big|_{a=\pi_\phi(s)}\,\nabla_\phi\pi_\phi(s),$$
$$\theta'_i \leftarrow \tau\theta_i + (1-\tau)\theta'_i,\qquad \phi'\leftarrow \tau\phi + (1-\tau)\phi'.$$

In the tabular finite-MDP case, the clipped double-Q update converges to $Q^*$. Write the stochastic-approximation term as $F_t=F^Q_t+c_t$, where $F^Q_t$ is the standard Q-learning contraction and $c_t=\gamma(\min(Q^A_t(s',a^*),Q^B_t(s',a^*))-Q^A_t(s',a^*))$. Feeding both tables the same min-based target makes $\Delta^{BA}_t=Q^B_t-Q^A_t$ contract as $(1-\alpha_t)$ per visit, so $c_t\to0$ and the standard Q-learning convergence argument applies.

Default hyperparameters in the code below: discount $\gamma=0.99$; soft-update $\tau=0.005$; smoothing noise $\tilde\sigma=0.2$ clipped to $\pm0.5$ for normalized action ranges; policy delay $d=2$; Adam at $3\times10^{-4}$; minibatches of 256; exploration by uncorrelated Gaussian noise $\mathcal N(0,0.1)$. For non-unit action ranges, pass `policy_noise` and `noise_clip` already scaled by the action magnitude.

## Working code

```python
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super().__init__()
        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)
        self.max_action = max_action

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        # Q1
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)
        # Q2
        self.l4 = nn.Linear(state_dim + action_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        q1 = F.relu(self.l1(sa)); q1 = F.relu(self.l2(q1)); q1 = self.l3(q1)
        q2 = F.relu(self.l4(sa)); q2 = F.relu(self.l5(q2)); q2 = self.l6(q2)
        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)
        q1 = F.relu(self.l1(sa)); q1 = F.relu(self.l2(q1)); q1 = self.l3(q1)
        return q1


class TD3(object):
    def __init__(self, state_dim, action_dim, max_action,
                 discount=0.99, tau=0.005,
                 policy_noise=0.2, noise_clip=0.5, policy_freq=2):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise   # sigma for target-action smoothing
        self.noise_clip = noise_clip       # the [-c, c] clip on the smoothing noise
        self.policy_freq = policy_freq      # delay d between policy/target updates
        self.total_it = 0

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self, replay_buffer, batch_size=256):
        self.total_it += 1
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)
        state, action, next_state, reward, not_done = (
            state.to(device), action.to(device), next_state.to(device),
            reward.to(device), not_done.to(device)
        )

        with torch.no_grad():
            # Target policy smoothing: add clipped noise to the target action
            noise = (torch.randn_like(action) * self.policy_noise
                     ).clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(next_state) + noise
                           ).clamp(-self.max_action, self.max_action)

            # Clipped Double Q-learning: minimum of the two target critics
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + not_done * self.discount * target_Q

        # Both critics regress to the same target
        current_Q1, current_Q2 = self.critic(state, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:
            # Deterministic policy gradient through Q1
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Soft-update target networks
            for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
                tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)
            for p, tp in zip(self.actor.parameters(), self.actor_target.parameters()):
                tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)
```

The training loop (collect a transition with exploration noise, store it, sample a minibatch, call `train`) and a uniform replay buffer complete the system; exploration noise is uncorrelated Gaussian, and target smoothing noise should be passed in action units when actions are not in $[-1,1]$.
