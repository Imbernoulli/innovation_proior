# C51 — distributional RL (categorical return distribution)

**Problem.** Every rung so far learns a scalar $Q(x,a)=\mathbb{E}[Z(x,a)]$ — the *mean* of the return. But
$Z(x,a)$ is a random variable (stochastic rewards/transitions, a moving policy): multimodal, skewed,
heavy-tailed. Collapsing it to its mean throws away its shape and gives a thin training signal — one number
per update.

**Key idea.** Learn the whole return distribution via the distributional Bellman equation (equality in
distribution) $Z(x,a)\overset{D}{=}R+\gamma Z(X',A')$. Policy evaluation $\mathcal T^\pi$ is a
$\gamma$-contraction in the **maximal Wasserstein** metric (shift cancels $R$, scaling pulls out $\gamma$)
— but *not* in KL/TV/Kolmogorov (overlap metrics blind to the horizontal $\gamma$-shrink).

**The trainability wall.** Wasserstein is the right metric but *cannot* be minimized from sampled
transitions: for a mixture target $P=\mathbb{E}_I[P_I]$, $d_p(P,Q)\le\mathbb{E}_I[d_p(P_I,Q)]$ (strict), so
the sampled-Wasserstein gradient is biased. So choose a representation + a sample-minimizable loss instead.

**Categorical representation + projection + cross-entropy.** Fixed grid of $N$ atoms
$z_i=V_{\min}+i\Delta z$; head emits a softmax $p_i(x,a)$ per action. Bellman-update the target atoms
$\hat{\mathcal T}z_j=r+\gamma z_j$ (off-grid), then **project** their mass back to the two nearest grid
atoms by linear interpolation ($\Phi$), and train with the cross-entropy term of
$D_{\mathrm{KL}}(\Phi\hat{\mathcal T}Z_{\tilde\theta}\,\|\,Z_\theta)=-\sum_i m_i\log p_i(x,a)$ — multiclass
classification over atoms. A deliberate metric trade (KL-after-projection, minimizable) for the
unminimizable Wasserstein.

**Acting / settings.** Mean-greedy (same risk-neutral policy): $Q(x,a)=\sum_i z_i p_i(x,a)$, $\arg\max$.
$N=51$ atoms on $[-10,10]$. Conv torso, replay, $\epsilon$-noise, target sync unchanged.

**Bar.** Replaces *what is learned*, giving a denser per-update signal on *every* game — expect the
strongest single component, clearing 151% decisively. Caveat: the fixed grid ($V_{\min},V_{\max}$, $N$) is
hand-set.

```python
# C51: categorical N-atom softmax head; Bellman shift+scale, project to grid, cross-entropy loss.
# Code home: vwxyzjn/cleanrl (cleanrl/c51_atari.py); excerpted from methods/c51/results/answer.md.
import torch
import torch.nn as nn

N_ATOMS = 51
V_MIN, V_MAX = -10.0, 10.0


class CategoricalQNetwork(nn.Module):
    """Conv torso; head outputs N atom-logits per action, softmaxed into p_i(x,a)."""
    def __init__(self, n_actions, n_atoms=N_ATOMS, v_min=V_MIN, v_max=V_MAX):
        super().__init__()
        self.n_actions, self.n_atoms = n_actions, n_atoms
        self.register_buffer("atoms", torch.linspace(v_min, v_max, steps=n_atoms))   # z_i grid
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
            action = torch.argmax(q_values, 1)               # greedy on the MEAN (risk-neutral)
        return action, pmfs[torch.arange(len(x)), action]


def categorical_projection(target_net, rewards, next_obs, dones, gamma,
                           v_min=V_MIN, v_max=V_MAX, n_atoms=N_ATOMS):
    """Phi T-hat Z: shift+scale the target atoms, then project onto the fixed grid."""
    with torch.no_grad():
        _, next_pmfs = target_net.get_action(next_obs)       # a* greedy on next mean
        atoms = target_net.atoms
        delta_z = atoms[1] - atoms[0]
        tz = (rewards + gamma * atoms * (1.0 - dones)).clamp(v_min, v_max)   # T-hat z_j, clamped
        b = (tz - v_min) / delta_z
        l = b.floor().clamp(0, n_atoms - 1)
        u = b.ceil().clamp(0, n_atoms - 1)
        d_m_l = (u + (l == u).float() - b) * next_pmfs       # mass split by linear interpolation
        d_m_u = (b - l) * next_pmfs
        target_pmfs = torch.zeros_like(next_pmfs)
        for i in range(target_pmfs.size(0)):
            target_pmfs[i].index_add_(0, l[i].long(), d_m_l[i])
            target_pmfs[i].index_add_(0, u[i].long(), d_m_u[i])
    return target_pmfs


def c51_loss(online_net, obs, actions, target_pmfs):
    _, pred_pmfs = online_net.get_action(obs, actions.flatten())
    # cross-entropy term of KL(Phi T-hat Z_{theta~} || Z_theta)
    return (-(target_pmfs * pred_pmfs.clamp(1e-5, 1 - 1e-5).log()).sum(-1)).mean()
```
