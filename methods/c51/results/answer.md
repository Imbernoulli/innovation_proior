# C51

## Problem

Value-based RL learns only the *expected* return $Q(x,a)=\mathbb{E}[Z(x,a)]$, discarding the
shape of the return distribution (multimodality, skew, variance) that arises from stochastic
rewards, stochastic transitions, and a nonstationary policy. C51 learns the **full distribution
of the random return** $Z(x,a)$ with a Bellman-based, bootstrapped algorithm trainable by SGD,
as a drop-in replacement for DQN's scalar head and squared loss.

## Key idea

The return obeys a **distributional Bellman equation** (equality in distribution)
$$Z(x,a)\overset{D}{=}R(x,a)+\gamma\,Z(X',A'),\qquad X'\sim P(\cdot|x,a),\,A'\sim\pi(\cdot|X').$$
With operators $P^\pi Z(x,a)\overset{D}{:=}Z(X',A')$ and
$\mathcal{T}^\pi Z(x,a)\overset{D}{:=}R(x,a)+\gamma P^\pi Z(x,a)$.

- **Policy evaluation contracts in Wasserstein.** Using the maximal Wasserstein metric
  $\bar d_p(Z_1,Z_2)=\sup_{x,a}d_p(Z_1(x,a),Z_2(x,a))$, the operator $\mathcal{T}^\pi$ is a
  $\gamma$-contraction:
  $$\bar d_p(\mathcal{T}^\pi Z_1,\mathcal{T}^\pi Z_2)\le\gamma\,\bar d_p(Z_1,Z_2),$$
  via $d_p(R+\gamma P^\pi Z_1,R+\gamma P^\pi Z_2)\le\gamma\,d_p(P^\pi Z_1,P^\pi Z_2)$ (shift
  property P2 cancels the common $R$; scaling property P1 pulls out $\gamma$). Banach gives a
  unique fixed point $Z^\pi$, with all moments converging geometrically (once the means have
  settled, the residual second-order spread shrinks at rate $\gamma^2$). **Metric subtlety:** $\mathcal{T}^\pi$ is **not** a contraction in total variation,
  KL divergence, or Kolmogorov distance — these overlap/vertical distances are blind to the
  horizontal $\gamma$-shrink of the support; only Wasserstein has the scaling property P1.

- **Control is unstable.** The optimality operator $\mathcal{T}Z=\mathcal{T}^\pi Z$ ($\pi$ greedy
  on the mean) keeps the mean well-behaved ($\|\mathbb{E}\mathcal{T}Z_1-\mathbb{E}\mathcal{T}Z_2\|_\infty\le\gamma\|\mathbb{E}Z_1-\mathbb{E}Z_2\|_\infty$, so $\mathbb{E}Z_k\to Q^*$), but is **not** a
  contraction in any distribution metric, may have no fixed point, and at best converges to the
  set of *nonstationary* optimal value distributions $\mathcal{Z}^{**}$. This motivates a smooth,
  gradient-based (averaging) update rather than a hard greedy one.

- **Categorical representation.** Fixed support of $N$ atoms
  $z_i=V_{\min}+i\Delta z$, $\Delta z=\frac{V_{\max}-V_{\min}}{N-1}$, with softmax probabilities
  $Z_\theta(x,a)=\sum_i p_i(x,a)\delta_{z_i}$, $p_i=\mathrm{softmax}(\theta_i(x,a))$. C51 uses
  $N=51$, $V_{\max}=-V_{\min}=10$.

- **Why not minimize Wasserstein directly?** Wasserstein is the right metric for the operator,
  but its loss cannot be minimized from sampled transitions with SGD: for a mixture target
  $P=P_I$, $d_p(P,Q)\le\mathbb{E}_I d_p(P_I,Q)$ with generally strict inequality, so
  $\nabla_Q d_p(P_I,Q)\ne\mathbb{E}_I\nabla_Q d_p(P_i,Q)$ — the sampled gradient is biased.

- **Categorical projection $\Phi$ + cross-entropy.** The Bellman-updated atoms
  $\hat{\mathcal{T}}z_j=r+\gamma z_j$ fall off the grid, so project their mass onto the two
  nearest atoms (linear interpolation, clamped to $[V_{\min},V_{\max}]$):
  $$\big(\Phi\hat{\mathcal{T}}Z_\theta(x,a)\big)_i=\sum_{j=0}^{N-1}\Big[\,1-\frac{\big|[\hat{\mathcal{T}}z_j]_{V_{\min}}^{V_{\max}}-z_i\big|}{\Delta z}\,\Big]_0^1\,p_j(x',\pi(x')).$$
  Train with the cross-entropy term of $D_{\mathrm{KL}}(\Phi\hat{\mathcal{T}}Z_{\tilde\theta}(x,a)\,\|\,Z_\theta(x,a))$:
  $$\mathcal{L}_{x,a}(\theta)=-\sum_i m_i\log p_i(x,a),\quad m=\Phi\hat{\mathcal{T}}Z_{\tilde\theta}(x,a).$$
  The distributional Bellman update reduces to multiclass classification over the atoms.

## Algorithm (categorical projection, $O(N)$ per atom)

Given transition $(x_t,a_t,r_t,x_{t+1})$, discount $\gamma_t$ ($=0$ at terminal):
1. $Q(x_{t+1},a)=\sum_i z_i\,p_i(x_{t+1},a)$; $\quad a^*=\arg\max_a Q(x_{t+1},a)$ (greedy on mean).
2. $m_i=0$ for all $i$.
3. For each atom $j$: $\hat{\mathcal{T}}z_j=[r_t+\gamma_t z_j]_{V_{\min}}^{V_{\max}}$;
   $\ b_j=(\hat{\mathcal{T}}z_j-V_{\min})/\Delta z$; $\ l=\lfloor b_j\rfloor,\ u=\lceil b_j\rceil$;
   $\ m_l\mathrel{+}=p_j(x_{t+1},a^*)(u-b_j)$; $\ m_u\mathrel{+}=p_j(x_{t+1},a^*)(b_j-l)$.
4. Loss $=-\sum_i m_i\log p_i(x_t,a_t)$.

Architecture = DQN's conv torso with an $N$-atom softmax head per action; target network for
$\tilde\theta$; $\epsilon$-greedy on the mean $Q$; Adam ($\epsilon_{\text{adam}}=0.01/\text{batch}$,
$\alpha=2.5\times10^{-4}$); $\gamma=0.99$; standard Atari preprocessing.

## Code

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

Grounded in the CleanRL `c51_atari.py` implementation. The
single-parameter **Bernoulli** variant ($N=2$) places the interpolation weight by the mean,
$\Phi\hat{\mathcal{T}}Z:=[(\mathbb{E}[\hat{\mathcal{T}}Z]-V_{\min})/\Delta z]_0^1$, but discards
distributional structure; more atoms resolve more of the distribution's shape, hence
$N=51$ (C51).
