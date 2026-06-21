The problem I am trying to settle is a tension that continuous-control deep RL has never resolved cleanly: I want one model-free algorithm that is both sample-efficient and stable. The algorithms I actually trust to be stable are on-policy — policy gradients, TRPO, PPO, A3C — but they use each batch of experience for a single gradient step and then discard it, because the gradient is only valid for the policy that generated the data. That costs millions of environment steps even on modest tasks, and it gets worse as the task scales. Off-policy methods can reuse a replay buffer, which is where sample efficiency lives, and the cleanest off-policy story for continuous actions is DDPG. But DDPG is brittle: nudge the learning rate or the exploration noise and it swings from "solves the task" to "flatlines," and on the genuinely hard high-dimensional tasks — Ant, the 21-DoF Humanoid — it frequently makes no progress at all while a slower on-policy method still crawls forward. The brittleness is structural, not bad luck. DDPG is really Q-learning wearing an actor: because $\max_a Q(s,a)$ is intractable in continuous spaces, it trains a deterministic policy $\mu_\theta(s)$ to approximate $\arg\max_a Q(s,a)$ and pushes it uphill through $\nabla_\theta Q(s,\mu_\theta(s)) = \nabla_a Q\,\nabla_\theta \mu_\theta$. A deterministic actor explores nothing on its own, so exploration noise must be bolted on and tuned; the actor collapses onto whatever ridge of $Q$ looks tallest right now; $Q$ overestimates because it is being maximized through that actor; and the coupled system has a tiny basin of stable hyperparameters. The other candidate, energy-based soft Q-learning, embraces stochasticity but solves directly for the optimal soft $Q^*$ and recovers the policy as the Boltzmann distribution $\pi \propto \exp(Q/\alpha)$ — which in continuous action spaces cannot be sampled in closed form, so it drags in a separate sampling network trained by amortized Stein variational gradient descent, an inference procedure whose quality bounds convergence and that is itself a source of instability. It is also not a true actor-critic: its $Q$ targets the optimal $Q^*$ and its "actor" is only an approximate sampler.

I propose Soft Actor-Critic (SAC): the maximum-entropy objective, solved by a genuine policy-iteration-style actor-critic with a stochastic, cheaply-sampled actor. Instead of maximizing $\sum_t \mathbb{E}[r(s_t,a_t)]$, I maximize reward plus the policy's entropy at every step,
$$J(\pi) = \sum_t \mathbb{E}_{(s_t,a_t)\sim\rho_\pi}\big[\, r(s_t,a_t) + \alpha\, \mathcal{H}(\pi(\cdot|s_t)) \,\big],$$
where $\mathcal{H}(\pi(\cdot|s)) = \mathbb{E}_{a\sim\pi}[-\log\pi(a|s)]$ and the temperature $\alpha$ trades reward against randomness; as $\alpha\to 0$ ordinary RL returns, so this generalizes the objective rather than abandoning it. This is the right generalization because an entropy-rewarded policy explores widely while still giving up on clearly bad actions, holds multiple modes when several actions are comparably good, and — by Ziebart's analysis — is robust to model and estimation error, which is exactly the fragility I am fighting. A bookkeeping fact I exploit throughout: $\alpha$ can be folded away, since the objective is $\alpha$ times the same problem with reward $r/\alpha$ and entropy coefficient $1$. So I scale rewards by $c = 1/\alpha$, drop $\alpha$ from the equations, and remember that the reward scale is the inverse temperature.

What makes SAC work is deriving the actor-critic from soft policy iteration and proving each step. For evaluation I want the soft value of a fixed policy $\pi$, defining $V(s) = \mathbb{E}_{a\sim\pi}[Q(s,a) - \log\pi(a|s)]$ and the soft backup $\mathcal{T}^\pi Q(s,a) = r(s,a) + \gamma\,\mathbb{E}_{s'}[V(s')]$. Substituting $V$ and pulling the entropy into the reward gives $\mathcal{T}^\pi Q(s,a) = r_\pi(s,a) + \gamma\,\mathbb{E}_{s',a'}[Q(s',a')]$ with $r_\pi(s,a) = r(s,a) + \gamma\,\mathbb{E}_{s',a'}[-\log\pi(a'|s')]$ absorbing the entropy. This is an ordinary policy-evaluation backup with reward $r_\pi$, hence a $\gamma$-contraction in sup norm — subtract two iterates and the entropy term cancels, leaving $|\mathcal{T}^\pi Q_1 - \mathcal{T}^\pi Q_2| \le \gamma\|Q_1 - Q_2\|_\infty$ — so it has a unique fixed point $Q^\pi$. That is the critic, and it converges.

The improvement step is where I had to be cleverer than soft Q-learning. The soft theory points toward making the new policy the energy-based $\pi \propto \exp(Q^{\pi_{\text{old}}})$, but that is exactly what I refuse to sample from. So instead of becoming that distribution I project onto it: keep the policy inside a tractable family $\Pi$ I can sample and evaluate, and within $\Pi$ move it as close as possible to the exponentiated $Q$ in KL divergence,
$$\pi_{\text{new}}(\cdot|s) = \arg\min_{\pi'\in\Pi} D_{\mathrm{KL}}\!\Big(\pi'(\cdot|s)\,\Big\|\,\frac{\exp(Q^{\pi_{\text{old}}}(s,\cdot))}{Z^{\pi_{\text{old}}}(s)}\Big).$$
The partition function $Z$ depends only on $s$, not on $\pi'$, so it is an additive constant in the objective and drops from the argmin and every policy gradient — the very thing that forced soft Q-learning into SVGD simply vanishes because I fit a tractable $\pi$ rather than sample from $\exp(Q)$. This projected step is still a genuine improvement. Writing the objective $J_{\pi_{\text{old}}}(\pi') = \mathbb{E}_{a\sim\pi'}[\log\pi'(a|s) - Q^{\pi_{\text{old}}}(s,a) + \log Z^{\pi_{\text{old}}}(s)]$ and noting $\pi_{\text{old}}\in\Pi$ is feasible, the minimizer can do no worse: $J_{\pi_{\text{old}}}(\pi_{\text{new}}) \le J_{\pi_{\text{old}}}(\pi_{\text{old}})$. Expanding both sides (the $\log Z(s)$ cancels) and using $V^{\pi_{\text{old}}}(s) = \mathbb{E}_{a\sim\pi_{\text{old}}}[Q^{\pi_{\text{old}}} - \log\pi_{\text{old}}]$ yields
$$\mathbb{E}_{a\sim\pi_{\text{new}}}\big[Q^{\pi_{\text{old}}}(s,a) - \log\pi_{\text{new}}(a|s)\big] \ge V^{\pi_{\text{old}}}(s).$$
Feeding this into the soft Bellman equation $Q^{\pi_{\text{old}}}(s,a) = r(s,a) + \gamma\,\mathbb{E}_{s'}[V^{\pi_{\text{old}}}(s')]$ and bootstrapping — each expansion replaces a $V^{\pi_{\text{old}}}$ by the larger right-hand side and never reverses, converging to $Q^{\pi_{\text{new}}}$ — gives $Q^{\pi_{\text{old}}}(s,a) \le Q^{\pi_{\text{new}}}(s,a)$ everywhere. The "fit instead of become" approximation costs nothing in monotonicity. Alternating the two steps produces a monotone, bounded sequence converging to the optimal policy within $\Pi$, and crucially the guarantee holds for any parameterization of $\Pi$ — exactly what soft Q-learning could not promise.

To run this at scale I approximate the tabular operations with networks and stochastic gradient steps: a Q-network, a policy network, and — although $V$ is in principle determined by $Q$ and $\pi$ — a separate value network, because the soft Q-target needs $V$ at the next state, and bootstrapping $Q$ off a quantity computed from that same $Q$ is the unstable self-reference target networks exist to tame. So I keep three networks plus a slow target copy of the value net. The value loss is the definition of $V$ turned into a regression, with states from the replay buffer $D$ but the action drawn fresh from the current $\pi_\phi$:
$$J_V(\psi) = \mathbb{E}_{s\sim D}\Big[\tfrac{1}{2}\big(V_\psi(s) - \mathbb{E}_{a\sim\pi_\phi}[\,\min_i Q_{\theta_i}(s,a) - \log\pi_\phi(a|s)\,]\big)^2\Big].$$
The Q loss is the soft Bellman residual, evaluated against the slow target value net $V_{\bar\psi}$, with both $s$ and $a$ taken straight from the buffer so the term is genuinely off-policy:
$$J_Q(\theta) = \mathbb{E}_{(s,a)\sim D}\Big[\tfrac{1}{2}\big(Q_\theta(s,a) - (c\,r + \gamma(1-d)V_{\bar\psi}(s'))\big)^2\Big],\qquad \bar\psi \leftarrow \tau\psi + (1-\tau)\bar\psi.$$

The actor objective is the KL projection, $J_\pi(\phi) = \mathbb{E}_{s\sim D, a\sim\pi_\phi}[\log\pi_\phi(a|s) - Q_\theta(s,a)]$ after $\log Z$ drops. Differentiating it is the subtle part, because $\phi$ appears both in the integrand and in the sampling distribution. The score-function estimator would avoid differentiating through the action but treats $Q_\theta$ as a black box and pays in variance — yet $Q_\theta$ is a network I can backpropagate through, and I want $\nabla_a Q$. So I use the reparameterization trick: write the action as a deterministic, differentiable transform of fixed noise, $a = f_\phi(\varepsilon; s)$ with $\varepsilon \sim \mathcal{N}(0,I)$, so sampling $a$ is sampling $\varepsilon$ (free of $\phi$) and the expectation moves over the fixed noise. The pathwise gradient is then
$$\nabla_\phi J_\pi(\phi) = \nabla_\phi \log\pi_\phi(a|s) + \big(\nabla_a \log\pi_\phi(a|s) - \nabla_a Q_\theta(s,a)\big)\nabla_\phi f_\phi(\varepsilon;s),\quad a = f_\phi(\varepsilon;s).$$
The $-\nabla_a Q_\theta\,\nabla_\phi f$ term is precisely the DDPG deterministic policy gradient — $\nabla_a Q$ pushed back through the action into the policy parameters — except it now flows through a stochastic reparameterized action and carries an entropy term that pulls the policy toward spreading out. The tension I began with dissolves: off-policy critic, low-variance reparameterized actor, intrinsic stochasticity.

For the tractable family I take a Gaussian, the obvious reparameterizable choice, $a = \mu_\phi(s) + \sigma_\phi(s)\odot\varepsilon$. But physical actions live in a box $[-1,1]$ while Gaussians have unbounded support; clipping kills the boundary gradient and corrupts the log-prob, so instead I squash: sample $u$ from the Gaussian and output $a = \tanh(u)$, which stays differentiable and bounded. Squashing changes the density, so with the elementwise map and diagonal Jacobian $\mathrm{diag}(1 - \tanh^2(u_i))$ the log-prob correction is a clean sum,
$$\log\pi(a|s) = \log\mu(u|s) - \sum_{i=1}^D \log\big(1 - \tanh^2(u_i)\big).$$
Because $1 - \tanh^2(u)$ underflows for large $|u|$, I compute the correction in the equivalent stable form $2(\log 2 - u - \mathrm{softplus}(-2u))$. Two more pieces come from hard experience with value-based methods. Any maximization through a noisy $Q$ biases it upward, and both the value and policy updates lean on $Q$, so I train two independent Q-functions $Q_{\theta_1}, Q_{\theta_2}$ on the same Bellman residual and use their minimum wherever $Q$ feeds the value or policy update — deliberate pessimism, because positive bias is the more dangerous error when the policy chases $Q$ peaks. The target value net tracks $\psi$ by an exponential moving average with a small $\tau \approx 0.005$, so the bootstrapped target moves on a slower timescale than the regressor chasing it. That leaves the one knob I cannot dodge: the reward scale $c = 1/\alpha$, the inverse temperature. Too small and the entropy term dominates — the policy goes near-uniform and ignores reward; too large and entropy becomes negligible — the policy collapses to near-deterministic early and gets stuck in poor optima. It is the single hyperparameter that needs per-task tuning. The loop is then an off-policy actor-critic: at each environment step sample $a\sim\pi_\phi(\cdot|s)$, step, and store $(s,a,r,s')$; then on a minibatch from $D$ descend $J_Q$ on each $\theta_i$, descend $J_V$ on $\psi$, descend $J_\pi$ on $\phi$ (with $Q$ frozen so gradients flow through $Q$ to the action but do not update the critics), and slide $\bar\psi \leftarrow \tau\psi + (1-\tau)\bar\psi$. Defaults are Adam at $3\times10^{-4}$, $\gamma=0.99$, replay buffer $10^6$, batch $256$, two hidden layers of $256$ ReLU units, $\tau=0.005$, one gradient step per environment step; at evaluation I read out the mean action.

```python
import math
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
from copy import deepcopy
from torch.distributions import Normal

LOG_STD_MIN, LOG_STD_MAX = -20, 2

def mlp(sizes, activation=nn.ReLU, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)

class SquashedGaussianActor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden, act_limit):
        super().__init__()
        self.net = mlp([obs_dim] + list(hidden), nn.ReLU, nn.ReLU)
        self.mu = nn.Linear(hidden[-1], act_dim)
        self.log_std = nn.Linear(hidden[-1], act_dim)
        self.register_buffer("act_limit", torch.as_tensor(act_limit, dtype=torch.float32))
    def forward(self, obs, deterministic=False, with_logprob=True):
        h = self.net(obs)
        mu = self.mu(h)
        std = torch.exp(torch.clamp(self.log_std(h), LOG_STD_MIN, LOG_STD_MAX))
        dist = Normal(mu, std)
        u = mu if deterministic else dist.rsample()          # reparameterized sample
        logp = None
        if with_logprob:
            logp = dist.log_prob(u).sum(-1)
            correction = 2 * (math.log(2.0) - u - F.softplus(-2 * u))
            logp = logp - correction.sum(-1)                 # tanh correction
        return torch.tanh(u) * self.act_limit, logp

class QFunction(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden):
        super().__init__()
        self.q = mlp([obs_dim + act_dim] + list(hidden) + [1])
    def forward(self, o, a): return self.q(torch.cat([o, a], -1)).squeeze(-1)

class ValueFunction(nn.Module):
    def __init__(self, obs_dim, hidden):
        super().__init__()
        self.v = mlp([obs_dim] + list(hidden) + [1])
    def forward(self, o): return self.v(o).squeeze(-1)

class ReplayBuffer:
    def __init__(self, obs_dim, act_dim, size):
        self.o  = np.zeros((size, obs_dim), np.float32)
        self.o2 = np.zeros((size, obs_dim), np.float32)
        self.a  = np.zeros((size, act_dim), np.float32)
        self.r  = np.zeros(size, np.float32)
        self.d  = np.zeros(size, np.float32)
        self.ptr, self.size, self.max = 0, 0, size
    def store(self, o, a, r, o2, d):
        i = self.ptr
        self.o[i], self.a[i], self.r[i], self.o2[i], self.d[i] = o, a, r, o2, d
        self.ptr = (i + 1) % self.max; self.size = min(self.size + 1, self.max)
    def sample(self, bs):
        idx = np.random.randint(0, self.size, bs)
        t = lambda x: torch.as_tensor(x, dtype=torch.float32)
        return dict(obs=t(self.o[idx]), act=t(self.a[idx]), rew=t(self.r[idx]),
                    obs2=t(self.o2[idx]), done=t(self.d[idx]))

class SAC:
    def __init__(self, obs_dim, act_dim, act_limit, hidden=(256, 256),
                 gamma=0.99, tau=0.005, lr=3e-4, reward_scale=5.0):
        self.actor   = SquashedGaussianActor(obs_dim, act_dim, hidden, act_limit)
        self.q1      = QFunction(obs_dim, act_dim, hidden)
        self.q2      = QFunction(obs_dim, act_dim, hidden)
        self.vf      = ValueFunction(obs_dim, hidden)
        self.vf_targ = deepcopy(self.vf)
        for p in self.vf_targ.parameters(): p.requires_grad = False
        self.opt_pi = torch.optim.Adam(self.actor.parameters(), lr)
        self.opt_q  = torch.optim.Adam(list(self.q1.parameters()) +
                                       list(self.q2.parameters()), lr)
        self.opt_v  = torch.optim.Adam(self.vf.parameters(), lr)
        self.gamma, self.tau, self.reward_scale = gamma, tau, reward_scale

    def _set_q_requires_grad(self, requires_grad):
        for p in list(self.q1.parameters()) + list(self.q2.parameters()):
            p.requires_grad = requires_grad

    def update(self, data):
        o, a, r, o2, d = (data['obs'], data['act'], data['rew'],
                          data['obs2'], data['done'])
        # --- Q update: soft Bellman residual toward reward_scale * r + gamma * V_targ(s') ---
        with torch.no_grad():
            q_target = self.reward_scale * r + self.gamma * (1 - d) * self.vf_targ(o2)
        loss_q = 0.5 * ((self.q1(o, a) - q_target) ** 2).mean() + \
                 0.5 * ((self.q2(o, a) - q_target) ** 2).mean()
        self.opt_q.zero_grad(); loss_q.backward(); self.opt_q.step()

        # fresh action from current policy (on-policy term inside the off-policy loop)
        # --- V update: regress onto E_{a~pi}[min Q - log pi] ---
        with torch.no_grad():
            a_pi, logp = self.actor(o)
            q_pi = torch.min(self.q1(o, a_pi), self.q2(o, a_pi))
            v_target = q_pi - logp
        loss_v = 0.5 * ((self.vf(o) - v_target) ** 2).mean()
        self.opt_v.zero_grad(); loss_v.backward(); self.opt_v.step()

        # --- policy update: minimize KL == E[log pi - Q] via reparameterization ---
        # Freeze Q weights, but keep the Q(a) computation differentiable with respect to a.
        self._set_q_requires_grad(False)
        try:
            a_pi, logp = self.actor(o)
            q_pi = torch.min(self.q1(o, a_pi), self.q2(o, a_pi))
            loss_pi = (logp - q_pi).mean()
            self.opt_pi.zero_grad(); loss_pi.backward(); self.opt_pi.step()
        finally:
            self._set_q_requires_grad(True)

        # --- slow-tracking target value network ---
        with torch.no_grad():
            for p, pt in zip(self.vf.parameters(), self.vf_targ.parameters()):
                pt.data.mul_(1 - self.tau); pt.data.add_(self.tau * p.data)

    @torch.no_grad()
    def act(self, obs, deterministic=False):
        a, _ = self.actor(torch.as_tensor(obs, dtype=torch.float32),
                          deterministic, with_logprob=False)
        return a.cpu().numpy()

def reset_env(env):
    out = env.reset()
    return out[0] if isinstance(out, tuple) else out

def step_env(env, action):
    out = env.step(action)
    if len(out) == 5:
        obs2, reward, terminated, truncated, info = out
        return obs2, reward, terminated or truncated, info
    obs2, reward, done, info = out
    return obs2, reward, done, info

def train(env, agent, steps=int(1e6), batch_size=256, start_steps=10000):
    buf = ReplayBuffer(env.observation_space.shape[0],
                       env.action_space.shape[0], int(1e6))
    o = reset_env(env)
    for t in range(steps):
        a = env.action_space.sample() if t < start_steps else agent.act(o)
        o2, r, done, _ = step_env(env, a)
        buf.store(o, a, r, o2, float(done))
        o = reset_env(env) if done else o2
        if t >= start_steps and buf.size >= batch_size:
            agent.update(buf.sample(batch_size))
```
