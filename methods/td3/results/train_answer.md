We want an agent that learns reward-maximizing behavior in continuous action spaces — robot-like control where an action is a vector of torques rather than one of a handful of discrete moves — off-policy and sample-efficiently, directly from interaction. The standard recipe couples a learned critic $Q_\theta(s,a)$ with a deterministic actor $\pi_\phi$ that ascends the critic's gradient, the whole deep-Q apparatus of experience replay and slow target networks bolted on. It half-works: a good seed learns a decent gait, a neighboring seed falls over and never recovers, and whenever it does learn, the critic's reported value keeps climbing past anything the policy could actually collect. That last symptom is the diagnostic thread. The critic calls a state worth a thousand; roll the policy out from that state and the discounted return is four hundred. The estimate is not noisy-around-the-truth, it is biased upward, and it worsens over time. The disease is function-approximation error in the critic, and the question is why a gradient-driven actor turns that error into a systematic, self-reinforcing inflation.

The discrete-action world already explains where such a bias comes from. Q-learning builds its target by maximizing, $y = r + \gamma\max_{a'}Q(s',a')$, and if the next-state estimate carries even zero-mean error, $Q(s',a') = Q_{\text{true}}(s',a') + \epsilon_{a'}$ with $\mathbb E[\epsilon_{a'}]=0$, the max overshoots: $\mathbb E_\epsilon[\max_{a'}(Q_{\text{true}}+\epsilon)] \ge \max_{a'}\mathbb E_\epsilon[Q_{\text{true}}+\epsilon] = \max_{a'}Q_{\text{true}}(s',a')$, because the max is convex and, more sharply, because the maximizing action is disproportionately the one whose error happened to be large and positive — the max selects for upward error. Bootstrapping then bakes that inflated target into $(s,a)$, which becomes an earlier state's bootstrap, so the bias propagates backward through the whole reachable set. The prior options for my setting all fail to remove it. Double Q-learning decouples selection from evaluation with two genuinely independent estimators, but independence is the entire premise of its unbiasedness. Double DQN's cheap version reuses the target network as the second estimator — but transcribed to actor-critic the "selector" is the slow policy, so $\pi_\phi(s')$ and $\pi_{\phi'}(s')$ pick almost the same action and the two estimates are far too similar to decouple anything; the very slowness I want for stability kills the correction. And plain DDPG carries a single critic with no mechanism against overestimation at all, nothing against the variance that bootstrapping accumulates, and deterministic targets that read the critic at single sharp points. I first verified the bias even exists for a gradient-driven actor: comparing the policy after a real (approximate-critic) gradient step against the policy after a true-critic step, two facts — each step climbs its own objective — chain with the bridge $\mathbb E[Q_\theta(s,\pi_{\text{true}}(s))] \ge \mathbb E[Q^\pi(s,\pi_{\text{true}}(s))]$ to give $\mathbb E[Q_\theta(s,\pi_{\text{approx}}(s))] \ge \mathbb E[Q^\pi(s,\pi_{\text{approx}}(s))]$ for small enough step size. Belief exceeds the truth of the policy actually deployed; the discrete intuition survives, with no $\max$ anywhere. And because the actor *ascends* the inflated critic, an upward bump does not merely mislead a readout, it attracts the policy, which generates more data in that pocket and lets the critic inflate it further — a self-reinforcing loop.

I propose TD3 — Twin Delayed Deep Deterministic policy gradient — which attacks function-approximation error from three coordinated directions, each a drop-in modification to the deterministic actor-critic, none disturbing its off-policy sample efficiency. The first is clipped Double Q-learning. Real two-of-everything double Q already pulls the overestimation curve down, but the residual is informative: the two critics share a replay buffer and each one's target is built from the other, so their errors correlate and the cross-estimator $Q_{\theta_2}$ sometimes overshoots its partner — using a number *above* an already-inflated estimate, exactly in the pockets the policy is drawn to. But the trouble is only ever that $Q_{\theta_2}$ comes out *higher* than $Q_{\theta_1}$; when it is lower it is doing its job. So I do not need $Q_{\theta_2}$ unbiased — I need it to never make things worse than $Q_{\theta_1}$, which is simply to take the smaller of the two:
$$y = r + \gamma\,\min_{i=1,2} Q_{\theta'_i}(s',\tilde a).$$
If $Q_{\theta_2}\ge Q_{\theta_1}$ the min picks $Q_{\theta_1}$ — the standard single-critic target, so the min can never add bias beyond plain Q-learning, a bounded worst case; if $Q_{\theta_2}<Q_{\theta_1}$ the min takes the Double-Q correction downward. The price is possible underestimation, but the two biases are not symmetric in this system: an overestimated action is selected by the actor and amplified through every later policy update, while an underestimated action is simply avoided and never explicitly chased. Leaning toward underestimation is the safe direction. The min carries a bonus too — the expected minimum of estimates falls as their variance rises, so it quietly penalizes high-variance states and steers the actor toward steady, reliable regions. In the tabular finite-MDP core this update converges to $Q^*$: write the stochastic-approximation term $F_t = F^Q_t + c_t$, where $F^Q_t$ is the ordinary Q-learning contraction with $\|\mathbb E[F^Q_t\mid P_t]\| \le \gamma\|\Delta_t\|$ and $c_t = \gamma(\min(Q^A_t(s',a^*),Q^B_t(s',a^*)) - Q^A_t(s',a^*))$ is the leftover; feeding *both* tables the *same* min-based target makes their gap contract, $\Delta^{BA}_{t+1}(s_t,a_t) = (1-\alpha_t)\Delta^{BA}_t(s_t,a_t)$ — the shared target $y$ cancels — so $c_t\to0$ and the standard convergence argument applies. That same proof licenses the practical shortcut: one actor and one shared target, not two of each.

The second direction handles accumulation, which is a variance story. Function approximation never satisfies Bellman exactly; each fit leaves a residual TD-error $\delta(s,a)$, so $Q_\theta(s,a) = r + \gamma\mathbb E[Q_\theta(s',a')] - \delta(s,a)$, and unrolling to the horizon gives
$$Q_\theta(s_t,a_t) = \mathbb E_{s_i\sim p_\pi,\,a_i\sim\pi}\Big[\sum_{i=t}^{T}\gamma^{i-t}\big(r_i - \delta_i\big)\Big],$$
the expected return *minus* the discounted sum of all future TD-errors. With $\gamma$ near one those weights barely decay, so far-future errors pile in at nearly full strength and the estimate's variance can balloon unless per-step error is held down — and a single minibatch step shrinks error only where it looked. A frozen, soft-updated target $\theta'\leftarrow\tau\theta+(1-\tau)\theta'$ lets the critic fit a stationary objective over many steps so the residual does not snowball; watching the value under different $\tau$ confirms that with a *fixed* policy every rate converges, but with a *learning* policy a fast target ($\tau=1$) diverges — divergence is the interaction of a high-variance estimate with a policy updating against it. So I do not move the actor at the critic's cadence: update the actor and the target networks only once every $d$ critic updates, the two-timescale structure the convergence results ask for — critic fast, actor slow — with $d=2$ keeping the actor moving while each of its steps sees a fresher critic. The third direction is peculiar to deterministic policies: the target action is a single point $\pi_{\phi'}(s')$, and a deterministic actor will find and sit on a narrow spurious peak in the bumpy critic, reading the target off a knife-edge. So fit the value of a *neighborhood* — a SARSA-style "safer" target — by perturbing the target action with clipped noise:
$$\tilde a = \mathrm{clip}\big(\pi_{\phi'}(s') + \epsilon,\,-a_{\max},\,a_{\max}\big),\qquad \epsilon\sim\mathrm{clip}(\mathcal N(0,\tilde\sigma),\,-c,\,c).$$
The noise is clipped to $[-c,c]$ so the perturbed action stays a genuine neighbor, and re-clipped into the valid action range so the critic is never read at an unexecutable action; minibatch averaging supplies the smoothing, denying the actor sharp peaks and giving policies robust to action perturbations. All three stack onto the same base: form $\tilde a$ and the min target $y$, regress both critics to it by minimizing $N^{-1}\sum(y-Q_{\theta_1}(s,a))^2 + N^{-1}\sum(y-Q_{\theta_2}(s,a))^2$ every step, and only every $d$ steps update the actor by the deterministic policy gradient through the first critic, $\nabla_\phi J = N^{-1}\sum\nabla_a Q_{\theta_1}(s,a)|_{a=\pi_\phi(s)}\nabla_\phi\pi_\phi(s)$, and soft-update $\theta'_i\leftarrow\tau\theta_i+(1-\tau)\theta'_i$, $\phi'\leftarrow\tau\phi+(1-\tau)\phi'$. Defaults: discount $\gamma=0.99$, soft-update $\tau=0.005$, smoothing noise $\tilde\sigma=0.2$ clipped to $\pm0.5$ for normalized action ranges, delay $d=2$, Adam at $3\times10^{-4}$, minibatches of 256, exploration by uncorrelated Gaussian noise $\mathcal N(0,0.1)$.

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
