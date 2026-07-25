The return in reinforcement learning is a random variable, not a single number. Standard DQN learns only the expected return, so two state-action pairs with very different risk profiles can look identical once they share the same expectation. A state whose return is either a large win or a large loss, and a state whose return is always average, are indistinguishable to a mean-only value function even though their outcomes and their interaction with exploration are completely different. Earlier distributional attempts tried Gaussian return models, but a Gaussian is unimodal and cannot represent the natural win-versus-lose branching that stochastic rewards and transitions create. A quantile-based alternative learns adaptive locations with fixed probability mass, which spreads resolution evenly across the probability axis and leaves rare but important tail events, such as crashes, under-resolved. What is needed is a representation that can be multimodal, is cheap for a neural network to output, and can be trained by ordinary SGD from sampled transitions.

The right object to learn is the full distribution of returns. The return obeys a distributional Bellman equation: the law of Z(x,a) is the same as the law of R(x,a) plus gamma times the law of Z(X',A'), where the next state and action are sampled from the environment dynamics and the policy. The corresponding distributional Bellman operator contracts in the Wasserstein metric, because Wasserstein sees the horizontal shrinkage caused by multiplying by gamma, whereas total variation, KL divergence, and Kolmogorov distance compare probability mass at matched locations and miss that contraction. Unfortunately the sampled Wasserstein loss has a biased gradient, so it cannot be minimized directly from single transitions. The way out is to give up on adapting both locations and masses: fix the support to a grid of return values and learn only the probability mass at each grid point.

The method is C51, short for Categorical DQN with 51 atoms. The network outputs a categorical distribution over a fixed grid of atoms z_i = V_min + i * Delta z for i = 0,...,N-1, with N = 51 and Delta z = (V_max - V_min)/(N-1). For each action the head produces N logits that are softmaxed into probabilities p_i(x,a), so the predicted return distribution is the weighted sum of point masses at the atoms. Action selection stays simple: act greedily with respect to the mean Q(x,a) = sum_i z_i p_i(x,a), which makes C51 a drop-in replacement for DQN's scalar head.

The Bellman update shifts each atom to r + gamma z_j, and these shifted locations almost never land exactly on the grid. C51 projects each shifted atom's mass onto its two nearest grid neighbors by linear interpolation, clamping anything outside the support to the endpoints. If b_j is the fractional grid position of the shifted atom, the lower neighbor receives weight (u - b_j) and the upper neighbor receives (b_j - l), where l is the floor and u is the ceiling of b_j; when b_j is exactly an integer the full mass goes to that single atom. The resulting target distribution m is then regressed with a cross-entropy loss, negative sum_i m_i log p_i(x,a), which is the same as the KL term from target to prediction. This loss has an unbiased sample gradient, which is why the projection step is essential: it moves the Bellman target back onto the fixed support so that cross-entropy is well defined.

Control with the full return distribution is more delicate than policy evaluation. The optimality operator is not a contraction in any distribution metric, because a tiny change in the mean can flip the greedy action and swap in a very different bootstrap distribution. In practice the smoothing effect of gradient descent, replay sampling, and target-network stabilization integrates away this chattering, and the mean still contracts to the optimal Q star. C51 therefore pairs the distributional head with the standard DQN training scaffold: an experience replay buffer, a linearly-annealed epsilon-greedy behavior policy, a periodically copied target network, and Adam.

```python
import os, random, time
import gymnasium as gym
import numpy as np
import torch, torch.nn as nn, torch.optim as optim

N_ATOMS = 51
V_MIN, V_MAX = -10.0, 10.0


class CategoricalQNetwork(nn.Module):
    """DQN conv torso; head outputs N atom-logits per action, softmaxed into p_i(x,a)."""
    def __init__(self, n_actions, n_atoms=N_ATOMS, v_min=V_MIN, v_max=V_MAX):
        super().__init__()
        self.n_actions, self.n_atoms = n_actions, n_atoms
        self.register_buffer("atoms", torch.linspace(v_min, v_max, steps=n_atoms))  # z_i grid
        self.network = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512), nn.ReLU(),
            nn.Linear(512, n_actions * n_atoms),
        )

    def get_action(self, x, action=None):
        logits = self.network(x / 255.0).view(len(x), self.n_actions, self.n_atoms)
        pmfs = torch.softmax(logits, dim=2)                  # categorical p_i(x,a)
        q_values = (pmfs * self.atoms).sum(2)                # Q(x,a) = sum_i z_i p_i(x,a)
        if action is None:
            action = torch.argmax(q_values, 1)               # greedy on the MEAN
        return action, pmfs[torch.arange(len(x)), action]


def categorical_projection(target_net, rewards, next_obs, dones, gamma,
                           v_min=V_MIN, v_max=V_MAX, n_atoms=N_ATOMS):
    """Phi T-hat Z_{theta~}: shift+scale the target atoms, then project onto the grid."""
    with torch.no_grad():
        _, next_pmfs = target_net.get_action(next_obs)               # a* greedy on next mean
        atoms = target_net.atoms
        delta_z = atoms[1] - atoms[0]
        tz = (rewards + gamma * atoms * (1.0 - dones)).clamp(v_min, v_max)  # T-hat z_j, clamped
        b = (tz - v_min) / delta_z                                   # fractional position
        l = b.floor().clamp(0, n_atoms - 1)
        u = b.ceil().clamp(0, n_atoms - 1)
        # split mass: lower atom gets (u - b), upper gets (b - l);
        # (l == u) sends full mass to the single atom when b_j is integer.
        d_m_l = (u + (l == u).float() - b) * next_pmfs
        d_m_u = (b - l) * next_pmfs
        target_pmfs = torch.zeros_like(next_pmfs)
        for i in range(target_pmfs.size(0)):
            target_pmfs[i].index_add_(0, l[i].long(), d_m_l[i])
            target_pmfs[i].index_add_(0, u[i].long(), d_m_u[i])
    return target_pmfs


def c51_loss(online_net, obs, actions, target_pmfs):
    _, pred_pmfs = online_net.get_action(obs, actions.flatten())
    # cross-entropy term of KL(Phi T-hat Z_{theta~} || Z_theta)
    return (-(target_pmfs * pred_pmfs.clamp(min=1e-5, max=1 - 1e-5).log()).sum(-1)).mean()


def linear_schedule(start_e, end_e, duration, t):
    return max((end_e - start_e) / duration * t + start_e, end_e)


# ---- training loop (off-policy, replay buffer, periodic target sync) ----
# online_net  = CategoricalQNetwork(n_actions)
# target_net  = CategoricalQNetwork(n_actions); target_net.load_state_dict(online_net.state_dict())
# optimizer   = optim.Adam(online_net.parameters(), lr=2.5e-4, eps=0.01 / batch_size)
#
# for step in range(total_timesteps):
#     eps = linear_schedule(1.0, 0.01, 0.10 * total_timesteps, step)
#     if random.random() < eps:
#         action = env.action_space.sample()
#     else:
#         action, _ = online_net.get_action(torch.as_tensor(obs)[None])
#     next_obs, reward, term, trunc, _ = env.step(action)
#     replay.add(obs, action, reward, next_obs, term); obs = next_obs
#
#     if step > learning_starts and step % train_frequency == 0:
#         d = replay.sample(batch_size)
#         m = categorical_projection(target_net, d.rewards, d.next_obs, d.dones, gamma=0.99)
#         loss = c51_loss(online_net, d.obs, d.actions, m)
#         optimizer.zero_grad(); loss.backward(); optimizer.step()
#     if step % target_network_frequency == 0:
#         target_net.load_state_dict(online_net.state_dict())
```

By fixing the return grid and learning a categorical distribution over it, C51 turns the distributional Bellman update into multiclass classification on top of the standard DQN torso. It captures multimodality and tail risk that scalar methods miss, while remaining trainable with the same off-policy machinery.
