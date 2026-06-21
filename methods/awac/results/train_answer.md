The problem is to learn a policy from a fixed offline dataset and then keep improving it once a small amount of online interaction becomes available. The dataset can be anything: expert demonstrations, suboptimal rollouts, or random exploration. The algorithm has to be efficient enough to make use of every offline transition during the online phase, stable enough to train on a static set of actions without diverging, and flexible enough to improve once fresh data arrives. Standard off-policy actor-critic methods such as SAC satisfy the efficiency requirement because they bootstrap a Q-function and reuse data from a replay buffer, but they fail the stability requirement. When the critic target is evaluated at an action proposed by the current policy, that action is often outside the data distribution, the Q-value is an unconstrained extrapolation, and the maximizing actor is pulled toward an overestimated value. On a fixed offline dataset there is no new data to correct the error, so the estimates collapse. The usual offline-RL fix is to add an explicit constraint that keeps the policy close to the behavior policy, for example by fitting a parametric density model and penalizing divergence from it. That prevents collapse offline, but during online fine-tuning the behavior model must track a streaming, multi-modal mixture of old and new data, which is hard. As the density estimate degrades, the constraint becomes over-conservative and online improvement stalls. Monte-Carlo advantage-weighted methods avoid the behavior model, but they estimate the value of the behavior policy rather than bootstrapping the current policy, so they are far less sample-efficient and cannot improve far beyond the data in one step.

The method that resolves this tension is AWAC, the Advantage-Weighted Actor-Critic. The core design is to keep an off-policy bootstrapped critic for efficiency, but to make the policy-improvement step stay inside the data distribution without ever fitting a behavior model. Start from the KL-constrained policy-improvement problem: maximize the expected advantage over the new policy subject to a KL divergence from the behavior policy. Solving this exactly gives a closed-form non-parametric optimum in which the new policy is the behavior policy reweighted by the exponentiated advantage. The crucial step is how this optimum is projected onto the parametric policy. Projecting by the forward KL, minimizing the expected negative log-likelihood under the optimal policy, lets the importance-sampling ratio cancel the behavior-policy density entirely. What remains is a simple supervised-learning update: maximize the log-likelihood of the actions that are already in the replay buffer, weighted by the exponentiated advantage of each action. Because the update only sees buffer actions, it cannot place mass on an action the dataset never contained, which enforces the stay-near-the-data constraint implicitly. At the same time, it never queries the critic at a policy-proposed out-of-distribution action during improvement, which removes the main source of offline bootstrap error. The same update is used unchanged during offline pretraining and online fine-tuning, so there is no schedule or transition rule to mis-tune.

The critic in AWAC is a standard twin-Q function trained by temporal-difference learning. The target uses the minimum of the two target critics evaluated at the next action sampled from the current policy, with a Polyak-averaged target network to control overestimation. The value V(s) is estimated as the minimum Q-value at an action sampled from the current policy, and the advantage is A(s,a) = Q(s,a) − V(s). The actor loss is then the negative weighted log-likelihood of the buffer actions, with weights proportional to exp(A(s,a)/λ). The per-state normalizer Z(s) that appears in the exact derivation is dropped in practice; estimating it from samples was found to hurt performance, and it only reweights states rather than actions, so a simple batch-normalization of the weights is enough. The temperature λ corresponds to the Lagrange multiplier on the KL constraint: a small λ produces aggressive, greedy improvement toward the highest-advantage actions, while a large λ flattens the weights and makes the update closer to behavioral cloning. The base implementation uses no entropy bonus, so the stochastic actor is driven mainly by the advantage-weighted likelihood. Typical settings use Adam with learning rate 3e-4 for both actor and critic, actor weight decay 1e-4, discount 0.99, Polyak coefficient 5e-3, batch size 1024, and λ around 0.3 for manipulation tasks or 1.0 for MuJoCo locomotion.

```python
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


def mlp(sizes, act=nn.ReLU):
    layers = []
    for i in range(len(sizes) - 1):
        layers += [nn.Linear(sizes[i], sizes[i + 1])]
        if i < len(sizes) - 2:
            layers += [act()]
    return nn.Sequential(*layers)


class Critic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=(256, 256, 256, 256)):
        super().__init__()
        self.net = mlp([obs_dim + act_dim, *hidden, 1])

    def forward(self, s, a):
        return self.net(torch.cat([s, a], -1)).squeeze(-1)


class TwinCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=(256, 256, 256, 256)):
        super().__init__()
        self.q1 = Critic(obs_dim, act_dim, hidden)
        self.q2 = Critic(obs_dim, act_dim, hidden)

    def forward(self, s, a):
        return self.q1(s, a), self.q2(s, a)


class GaussianPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=(256, 256, 256, 256)):
        super().__init__()
        self.trunk = mlp([obs_dim, *hidden])
        self.mu = nn.Linear(hidden[-1], act_dim)
        self.log_std = nn.Linear(hidden[-1], act_dim)

    def dist(self, s):
        h = self.trunk(s)
        return Normal(torch.tanh(self.mu(h)), self.log_std(h).clamp(-6, 0).exp())

    def log_prob(self, s, a):
        return self.dist(s).log_prob(a).sum(-1)

    def sample_and_log_prob(self, s):
        dist = self.dist(s)
        a = dist.rsample()
        return a, dist.log_prob(a).sum(-1)

    def sample(self, s):
        return self.sample_and_log_prob(s)[0]


def critic_td_loss(batch, critic, target_critic, policy, discount, alpha=0.0):
    s, a, r, s2, done = (
        batch["obs"], batch["act"], batch["rew"],
        batch["obs2"], batch["done"]
    )
    with torch.no_grad():
        a2, logp2 = policy.sample_and_log_prob(s2)
        tq1, tq2 = target_critic(s2, a2)
        target_v = torch.min(tq1, tq2) - alpha * logp2
        y = r + discount * (1.0 - done) * target_v
    q1, q2 = critic(s, a)
    return F.mse_loss(q1, y) + F.mse_loss(q2, y)


def actor_awac_loss(batch, critic, policy, lam):
    s, a = batch["obs"], batch["act"]
    with torch.no_grad():
        q = torch.min(*critic(s, a))
        v = torch.min(*critic(s, policy.sample(s)))
        adv = q - v
        weight = torch.softmax(adv / lam, dim=0) * adv.numel()
    logp = policy.log_prob(s, a)
    return -(weight * logp).mean()


def polyak(critic, target_critic, tau):
    for p, tp in zip(critic.parameters(), target_critic.parameters()):
        tp.data.mul_(1 - tau).add_(tau * p.data)


def update(batch, critic, target_critic, policy, opts, hp):
    q_loss = critic_td_loss(
        batch, critic, target_critic, policy,
        hp["discount"], hp.get("alpha", 0.0)
    )
    opts["q"].zero_grad()
    q_loss.backward()
    opts["q"].step()

    pi_loss = actor_awac_loss(batch, critic, policy, hp["lam"])
    opts["pi"].zero_grad()
    pi_loss.backward()
    opts["pi"].step()

    polyak(critic, target_critic, hp["tau"])


def train_awac(env, buffer, critic, policy, hp,
               pretrain_steps=25000, online_steps=int(1e6)):
    target_critic = copy.deepcopy(critic)
    opts = {
        "q": torch.optim.Adam(critic.parameters(), lr=3e-4),
        "pi": torch.optim.Adam(policy.parameters(), lr=3e-4, weight_decay=1e-4),
    }
    for _ in range(pretrain_steps):
        update(buffer.sample(hp["batch_size"]), critic,
               target_critic, policy, opts, hp)

    o = env.reset()
    for _ in range(online_steps):
        a = policy.sample(torch.as_tensor(o, dtype=torch.float32)).detach().numpy()
        o2, r, done, _ = env.step(a)
        buffer.add(o, a, r, o2, done)
        o = env.reset() if done else o2
        update(buffer.sample(hp["batch_size"]), critic,
               target_critic, policy, opts, hp)
```
