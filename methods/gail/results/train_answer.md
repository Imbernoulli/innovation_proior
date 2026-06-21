We need to learn a policy from nothing but a fixed batch of expert trajectories. There is no reward signal, we cannot query the expert during training, and we only see what the expert did in the states it visited. The standard answers both fail in different ways. Behavioral cloning turns the problem into supervised learning on the expert's state-action pairs, but it optimizes a single-timestep loss under the expert's state distribution. Once the learned policy makes a small error and drifts off the expert's path, it enters states it was never trained on, errors compound, and performance degrades quadratically with horizon. Inverse reinforcement learning avoids this by scoring whole trajectories and recovering a cost function under which the expert is optimal, but it requires solving a reinforcement-learning problem inside its inner loop, which is expensive, and it ultimately returns a cost rather than the policy we wanted.

The right fix is to match the expert's full trajectory distribution without ever running an inner RL loop and without learning an explicit cost. The key observation is that policies can be represented by their occupancy measures, the discounted distribution of state-action pairs they visit. A classical duality result shows that running regularized maximum-causal-entropy IRL followed by RL is exactly equivalent to minimizing a discrepancy between the learner's occupancy measure and the expert's, plus a causal-entropy regularizer. The cost function was just the dual variable enforcing the occupancy match, so it can be eliminated entirely. This reduces imitation to distribution matching in the space of visited transitions.

The method is Generative Adversarial Imitation Learning (GAIL). It chooses a particular convex regularizer so that the induced occupancy discrepancy becomes, up to a constant, the Jensen-Shannon divergence between the learner's and expert's state-action distributions. That divergence is evaluated by a binary classifier, the discriminator, which is trained to distinguish transitions sampled from the current policy from transitions sampled from the expert. The policy then plays the role of the generator: it is updated to make its transitions look indistinguishable from expert transitions, i.e., to fool the discriminator. Because the discriminator is refit each iteration, it acts as an adaptive cost signal, and because the match is at the level of full transition distributions rather than per-step decisions, GAIL avoids behavioral cloning's compounding-error problem.

In practice, the discriminator is a neural network that takes a state-action pair and outputs a logit. It is trained with binary cross-entropy, labeling policy samples as negative and expert samples as positive. The reward handed to the policy optimizer is derived from the discriminator's probability that a transition is expert-like, so moving toward expert-like regions increases return. The policy update is a trust-region step, using a value baseline and generalized advantage estimation to keep variance low and prevent the policy from lurching away from the current behavior. A causal-entropy bonus can be folded into the same policy-gradient machinery to encourage exploration. The result is a scalable imitation algorithm that directly produces a policy while neither requiring a reward function nor solving RL in an inner loop.

```python
import torch, torch.nn as nn, numpy as np
import torch.nn.functional as F

class LearningSignal(nn.Module):
    """Binary transition classifier; score > 0 means expert-like."""
    def __init__(self, obs_dim, act_dim, hidden=100):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1))  # logit / score

    def score(self, obs, act):
        return self.net(torch.cat([obs, act], dim=1)).squeeze(1)

    def reward(self, obs, act, favor_zero_expert=False):
        with torch.no_grad():
            s = self.score(obs, act)
            if favor_zero_expert:
                return F.logsigmoid(s)  # alternate shaping capped at 0
            return F.softplus(s)        # -log(1 - sigmoid(s))

    def fit(self, opt, pi_obs, pi_act, ex_obs, ex_act, steps=1, ent_reg=1e-3):
        obs = torch.cat([pi_obs, ex_obs])
        act = torch.cat([pi_act, ex_act])
        B = len(pi_obs)
        Ball = len(obs)
        labels = torch.zeros(Ball, device=obs.device)
        labels[B:] = 1.0
        weights = torch.empty(Ball, device=obs.device)
        weights[:B] = 1.0 / B
        weights[B:] = 1.0 / (Ball - B)
        for _ in range(steps):
            s = self.score(obs, act)
            bce = F.binary_cross_entropy_with_logits(s, labels, reduction='none')
            ent = F.softplus(-s) + (1.0 - torch.sigmoid(s)) * s
            loss = ((bce - ent_reg * ent) * weights).sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
        return loss.item()

def imitation_loop(env, expert_obs, expert_act, policy, value_fn,
                   obs_dim, act_dim, iters=500, gamma=0.995,
                   gae_lam=0.97, lam_ent=0.0, max_kl=0.01):
    signal = LearningSignal(obs_dim, act_dim)
    signal_opt = torch.optim.Adam(signal.parameters(), lr=1e-2)
    for _ in range(iters):
        obs, act, ep_lens = sample_trajectories(env, policy)
        rew = signal.reward(obs, act)
        if lam_ent:
            rew = rew + lam_ent * (-policy.log_prob(obs, act).detach())
        adv = gae(rew, value_fn(obs).detach(), ep_lens, gamma, gae_lam)
        trpo_step(policy, obs, act, adv, max_kl=max_kl)
        idx = np.random.choice(len(expert_obs), size=len(obs))
        signal.fit(signal_opt, obs, act, expert_obs[idx], expert_act[idx])
        rew_for_value = signal.reward(obs, act)
        if lam_ent:
            rew_for_value = rew_for_value + lam_ent * (-policy.log_prob(obs, act).detach())
        value_fn.fit(obs, returns_from(rew_for_value, ep_lens, gamma))
    return policy
```
