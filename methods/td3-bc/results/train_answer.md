We want a continuous-control policy out of a fixed dataset of transitions $D = \{(s, a, r, s')\}$ collected by some other process, with no further environment interaction — the offline setting that matters because in robotics, healthcare, and similar domains fresh data is expensive or unsafe while logged data already exists. The natural move is to take the strongest off-policy actor-critic on hand, TD3, and train it on $D$ as if $D$ were its replay buffer. It already learns off-policy from a buffer; the buffer is simply frozen now. But it collapses offline, and the mechanism is precise. TD3's critic is fit by TD regression to a bootstrap target $y = r + \gamma \min_i Q_{\theta'_i}(s', \tilde a)$, and its actor is moved by the deterministic policy gradient $\nabla_\phi J = \mathbb{E}_s[\nabla_a Q(s,a)|_{a=\pi(s)}\,\nabla_\phi \pi(s)]$, climbing the critic's value surface. The actor therefore ends up wherever the critic reports high value — including at actions that appear nowhere in $D$, where the critic, a neural net, is purely extrapolating and in practice over-rates, precisely because the actor has been seeking out wherever the value looks large. Online this self-heals: the over-valued action gets taken, the environment returns the true low reward, the next update corrects the critic. Offline nothing refutes it, and worse, the target $y$ bootstraps at the target policy's own drifting next action, so the inflation backs up through the Bellman recursion and the actor-critic loop blows up. The accepted cure is to keep the actor's actions near the data so the critic is only queried where it has support. Everyone implements this by fitting a second model of the data distribution and constraining the policy toward it: BCQ samples from a conditional VAE and lets the critic pick among perturbed candidates; BEAR and BRAC keep an explicit $\hat\pi_\beta$ fit by maximum likelihood and penalize a divergence to it (BRAC a KL $D_{KL}(\pi_\theta, \hat\pi_\beta)$, BEAR an MMD support constraint); CQL instead regularizes the critic toward pessimism on out-of-distribution actions via a logsumexp over many sampled actions; Fisher-BRC reparameterizes the critic as a behavior model's log-density plus a learned offset with a Fisher-divergence gradient penalty and a reward bonus.

Every one of these works, but the accounting is uniquely bad offline. Each requires fitting a second model whose quality caps the constraint, or extra per-step compute, and each drags in a stack of unannounced implementation changes — actor pre-training, max-over-sampled-actions evaluation, entropy removal, modified architectures and learning rates, reward bonuses — that, when stripped back to the base algorithm, turn out to be carrying much of the performance, and that more than double the wall-clock time. The deeper problem is that offline I cannot interact, so I cannot validate any of it: every added knob must be set blind. The cost of complexity is therefore not linear offline; it is punishing. That reframes the goal. I do not want the cleverest constraint; I want the fewest moving parts that still keeps the actor inside the data — ideally zero extra models, near-zero extra compute, one hyperparameter — so that whatever performance I get is attributable and I have almost nothing to tune blind.

I propose TD3+BC: take TD3 unchanged and make exactly two additions. The first is the cheapest possible way to say "stay near the data." Every method above answers this with a learned $\hat\pi_\beta$ and a divergence to it, but evaluating a KL needs a *distribution* $\hat\pi_\beta$, which is the whole reason a generative model gets dragged in — and there is no fundamental argument that KL is the right divergence; it is a choice, and the choice is what forces the behavior model. My actor is *deterministic*: $\pi_\phi(s)$ outputs a single action vector, and the dataset hands me, for each state $s$, the single action $a$ actually taken there. So "stay near the data" at a sampled $(s,a)$ is not a divergence between distributions at all — it is a distance between two points in action space, and the simplest such distance is squared Euclidean $(\pi_\phi(s) - a)^2$, which is exactly the behavior-cloning regression loss, fit on the same $(s,a)$ pairs already sampled for the critic, with no second model and one line of code. But BC alone is the other failure mode — it imitates the dataset wholesale and can never exceed the behavior that generated it. So rather than replace the value objective with imitation, I add the imitation term as a regularizer onto TD3's value-maximizing actor objective:

$$\pi = \arg\max_\pi \; \mathbb{E}_{(s,a)\sim D}\big[\,\lambda\, Q(s, \pi(s)) - (\pi(s) - a)^2\,\big].$$

The $Q$ term pulls the policy toward high value; the BC term tethers it to the action actually seen at that state. Where they agree the policy happily takes the dataset action; where the critic wants to wander to an unsupported action the quadratic penalty grows and pulls it back. The actor can still improve on the data within the neighborhood the critic can be trusted, but it cannot run off to fantasy actions — so the critic, trained on dataset $(s,a)$ pairs, is only ever asked about actions near those pairs, and the extrapolation that caused the collapse is starved at its source.

The trade-off between the two terms is where a naive equal weighting bites. With actions normalized to $[-1,1]$, the mean-squared BC term is bounded — each coordinate differs by at most $2$, so the averaged squared error is at most $4$, a fixed small range regardless of task. The $Q$ term has no such bound: $Q(s,a)$ estimates discounted return, which scales directly with reward magnitude, and that differs by orders of magnitude across tasks and data qualities — $Q$ might sit near $10$ on one task and near $1000$ on another while the BC term stays pinned below $4$ everywhere. A single fixed $\lambda$ therefore cannot transfer: equal weighting leaves the $Q$ term swamping BC on high-reward tasks (no effective constraint, back to the blow-up) and BC swamping $Q$ on low-reward tasks (glorified imitation), forcing a per-dataset re-tune — exactly the blind-knob cost I am eliminating. The fix is to *measure* the scale of the $Q$ term and divide it out, with a cheap robust scalar — the mean absolute value over the same actor-update batch of critic values. So instead of a fixed weight,

$$\lambda = \frac{\alpha}{\frac{1}{N}\sum_i \big| Q_1(s_i, \pi(s_i)) \big|},$$

which makes $\lambda Q$ have magnitude roughly $\alpha \cdot |Q| / \mathrm{mean}|Q| \approx \alpha$, pinned regardless of whether $Q$ lives near $10$ or $1000$. Since the BC term is already $O(1)$ to $O(4)$, the dimensionless $\alpha$ directly sets the exploitation-to-imitation ratio and the *same* $\alpha$ is meaningful across every task — one scale-free number replacing a per-task $\lambda$. One detail matters: $\lambda$ contains $Q$, and it is a scalar scaling factor on the loss, not a contributor to the gradient direction; if gradients flowed through the $Q$ inside $\lambda$ I would be differentiating the normalizer and adding a spurious term. So $\lambda$ is *detached* — computed as a number from the current batch and held constant for the backward pass — and only $Q(s,\pi(s))$ and $(\pi(s)-a)^2$ carry gradient. There is a free second benefit confirming this is the natural object: the deterministic policy gradient is $\nabla_a Q \cdot \nabla_\phi \pi$, and the magnitude of $\nabla_a Q$ also scales with the reward scale, so dividing the value term by $\mathrm{mean}|Q|$ normalizes the actor's effective learning rate across tasks too — one scalar, two problems solved.

The second addition exploits that the dataset is fixed, which means exact statistics can be computed once, up front, for free. Raw observation features — a joint angle, an angular velocity, a contact force — live on very different scales, so an MLP fed raw inputs wastes capacity undoing them. Normalize each state feature to zero mean and unit variance using the dataset statistics,

$$s_i \leftarrow \frac{s_i - \mu_i}{\sigma_i + \epsilon}, \qquad \epsilon = 10^{-3},$$

with $\mu_i, \sigma_i$ the per-feature mean and standard deviation over $D$ and the small $\epsilon$ guarding a near-constant feature from exploding. This is applied to both $s$ and $s'$ so the critic's target sees the same inputs, and $\mu, \sigma$ are kept to normalize live observations at evaluation. It is particularly apt offline, where the statistics are exact and stationary rather than drifting; it is a near-free stabilizer, not the core change.

Everything below the actor objective is untouched TD3, because the entire value of the approach is that performance is attributable. The critic still fits twin $Q_{\theta_1}, Q_{\theta_2}$ by MSE to the clipped-double-Q target $y = r + \gamma(1-d)\min_i Q_{\theta'_i}(s', \tilde a)$ with target policy smoothing $\tilde a = \mathrm{clip}(\pi_{\phi'}(s') + \mathrm{clip}(\mathcal{N}(0,\sigma), -c, c), -a_{\max}, a_{\max})$, $\sigma = 0.2\,a_{\max}$, $c = 0.5\,a_{\max}$; the actor and soft target updates ($\tau = 5\cdot10^{-3}$) fire only every $\mathrm{policy\_freq}=2$ critic steps; $\gamma = 0.99$, Adam at $3\cdot10^{-4}$, batch $256$, $256\times256$ ReLU MLPs. No generative model, no entropy term, no architecture surgery, no actor pre-training, no reward bonus, no extra forward passes. The one hyperparameter is $\alpha$: small $\alpha$ shrinks $\lambda Q \approx \alpha$ relative to the $O(1)$ BC term and collapses toward pure imitation ceilinged by the data, while large $\alpha$ lets the value term dominate and the BC tether go slack, drifting back toward the offline blow-up. A value around $\alpha = 2.5$ sits in the middle band — value-led but firmly tethered — and because the normalization already made $\alpha$ task-invariant, one such value holds across the whole benchmark.

```python
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def normalize_states(states, next_states, eps=1e-3):
    """Per-feature zero-mean/unit-variance over the fixed dataset; eps guards near-constant features.
    Return (mean, std) to normalize live observations the same way at evaluation time."""
    mean = states.mean(0, keepdims=True)
    std = states.std(0, keepdims=True) + eps
    return (states - mean) / std, (next_states - mean) / std, mean, std


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
        q1 = self.l3(F.relu(self.l2(F.relu(self.l1(sa)))))
        q2 = self.l6(F.relu(self.l5(F.relu(self.l4(sa)))))
        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)
        return self.l3(F.relu(self.l2(F.relu(self.l1(sa)))))


class TD3_BC(object):
    def __init__(self, state_dim, action_dim, max_action,
                 discount=0.99, tau=0.005,
                 policy_noise=0.2, noise_clip=0.5, policy_freq=2, alpha=2.5):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise * max_action     # smoothing noise scaled by action range
        self.noise_clip = noise_clip * max_action
        self.policy_freq = policy_freq
        self.alpha = alpha                                 # the single hyperparameter (RL/BC ratio)
        self.total_it = 0

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self, replay_buffer, batch_size=256):
        self.total_it += 1
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        # ---- critic: untouched TD3 (clipped double-Q + target policy smoothing) ----
        with torch.no_grad():
            noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)     # min over twins: no extra overestimation
            target_Q = reward + not_done * self.discount * target_Q

        current_Q1, current_Q2 = self.critic(state, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ---- delayed actor update: the one algorithmic change ----
        if self.total_it % self.policy_freq == 0:
            pi = self.actor(state)
            Q = self.critic.Q1(state, pi)
            lmbda = self.alpha / Q.abs().mean().detach()   # scalar normalizer, detached -> no gradient
            # minimize -(lambda*Q) + (pi - a)^2  <=>  maximize  lambda*Q(s, pi(s)) - (pi(s) - a)^2
            actor_loss = -lmbda * Q.mean() + F.mse_loss(pi, action)
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # soft target updates
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
```
