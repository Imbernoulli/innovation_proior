Dueling reached 151%, and I have now improved every axis of the recipe except the one I deliberately kept for last because it is the deepest: the value *object* itself. Through five rungs the thing I learn has stayed a single scalar per action — the decoupled target made that scalar less biased, the dueling head learned its state-value part from more data, but it is still one number, the *mean* of the return. Every rung so far has been a better way to estimate a mean, and the question I have been postponing is whether estimating the mean is the right problem at all. I learn $Q(x,a)=\mathbb{E}[Z(x,a)]$ because I train to maximize *expected* return — but $Z(x,a)$, the actual discounted return from taking $a$ in $x$ and following the policy, is a *random variable*, and on Atari a wildly non-trivial one: stochastic in rewards, in transitions, and in a policy that is itself moving during training, so $Z$ is typically multimodal, skewed, heavy-tailed. Collapsing all of that into its mean throws away its shape, and a single scalar target with no notion of its own spread must absorb all that stochasticity into one number — exactly the wobble the scalar critic produces. The reason to care even though my policy stays mean-greedy is that a *distribution* is a far richer training signal than a mean: if I learn the whole law of $Z$, every transition tells the network not just "the return here is about $v$" but "the return here is *this shape*," many more constraints per update, a denser supervisory signal that shapes the representation better and tends to help even when I only ever read off the mean to act.

So I propose **C51**: stop learning the mean of the return and learn its whole distribution as a categorical over a fixed grid. The first question is whether I can even bootstrap a distribution the way I bootstrap a mean. The mean version rests on $Q=\mathbb{E}[R+\gamma Q']$; the distributional analogue is an equality *in distribution*,
$$Z(x,a)\overset{D}{=}R(x,a)+\gamma\,Z(X',A'),\qquad X'\sim P(\cdot\mid x,a),\;A'\sim\pi(\cdot\mid X'),$$
the return random variable equals (in law) the immediate reward plus $\gamma$ times the next-state return. Writing the operator $\mathcal T^\pi Z(x,a)\overset{D}{:=}R+\gamma Z(X',A')$, for *policy evaluation* it is a $\gamma$-contraction — but only in the right metric. In the maximal Wasserstein metric $\bar d_p(Z_1,Z_2)=\sup_{x,a}d_p(Z_1(x,a),Z_2(x,a))$, the constant shift by $R$ cancels (Wasserstein is shift-covariant) and the scaling by $\gamma$ pulls a factor $\gamma$ out front (Wasserstein is scale-covariant by $|\gamma|$), so $\bar d_p(\mathcal T^\pi Z_1,\mathcal T^\pi Z_2)\le\gamma\,\bar d_p(Z_1,Z_2)$ and Banach gives a unique fixed point $Z^\pi$. The metric subtlety matters: $\mathcal T^\pi$ is *not* a contraction in KL, total variation, or Kolmogorov distance — those are "vertical" overlap distances, blind to the horizontal $\gamma$-shrink of the support; only Wasserstein has the scaling property. (For *control*, with $\pi$ greedy on the mean, the operator keeps the *mean* contracting toward $Q^*$ but is not a contraction in any distribution metric — a reason to use a smooth, averaging gradient update rather than a hard greedy one, not a reason to abandon the approach.)

Wasserstein is the metric the operator contracts in, so the natural idea is to *train* by minimizing Wasserstein distance to the bootstrapped target — and here is the real wall: the Wasserstein loss *cannot be minimized from sampled transitions by SGD*. The bootstrapped target is a mixture over the sampled next states, and for a mixture $P=\mathbb{E}_I[P_I]$ one has $d_p(P,Q)\le\mathbb{E}_I[d_p(P_I,Q)]$ with the inequality generally *strict*, so the gradient of the sampled Wasserstein distance is a *biased* estimate of the gradient of the true one. I cannot minimize the right metric with the only thing I have. So I need a *representation* of the distribution and a *loss* I *can* minimize from samples, accepting that the loss will not be Wasserstein. I represent $Z(x,a)$ as a discrete distribution on a *fixed* grid of $N$ atoms $z_i=V_{\min}+i\,\Delta z$, $\Delta z=\frac{V_{\max}-V_{\min}}{N-1}$, with the network emitting a softmax over the atoms per action, $Z_\theta(x,a)=\sum_i p_i(x,a)\,\delta_{z_i}$, $p_i=\operatorname{softmax}(\theta_i(x,a))$. The head, instead of one scalar per action, emits $N$ logits per action that softmax into a categorical over a shared value grid. I have to *set* $V_{\min},V_{\max}$ up front — the cost of a fixed grid — and with reward clipping and $\gamma=0.99$ on Atari, $[-10,10]$ comfortably covers the clipped discounted return.

The loss is the move that makes it trainable. I apply the distributional Bellman update to the target network's distribution by shifting and scaling each target atom, $\hat{\mathcal T}z_j=r+\gamma z_j$ — but the shifted atoms no longer land on my fixed grid. So I *project* them back: each shifted atom's mass $p_j(x',\pi(x'))$ is split between the two nearest grid atoms by linear interpolation (and clamped to $[V_{\min},V_{\max}]$ at the ends),
$$\big(\Phi\hat{\mathcal T}Z\big)_i=\sum_j\Big[1-\frac{\big|[r+\gamma z_j]_{V_{\min}}^{V_{\max}}-z_i\big|}{\Delta z}\Big]_0^1 p_j(x',\pi(x')).$$
This $\Phi$ produces a target distribution *on my grid*, which I can compare to my prediction with the cross-entropy — the comparison that *is* minimizable from samples. I train with the cross-entropy term of $D_{\mathrm{KL}}\big(\Phi\hat{\mathcal T}Z_{\tilde\theta}(x,a)\,\big\|\,Z_\theta(x,a)\big)$, i.e. $\mathcal L=-\sum_i m_i\log p_i(x,a)$ with $m=\Phi\hat{\mathcal T}Z_{\tilde\theta}(x,a)$ the projected target probabilities — exactly *multiclass classification over the atoms*, a well-behaved, SGD-friendly, sample-unbiased loss. I have traded the metric the operator contracts in (Wasserstein, unminimizable from samples) for one I can actually optimize (KL after projection), eyes open; the projection $\Phi$ is what keeps the targets on the grid so the KL is well-defined, and it works because the projection keeps the representation closed under the Bellman update.

Everything else stays minimal so this is a clean single-axis change over dueling. I act mean-greedily as before — read the mean of each action's distribution, $Q(x,a)=\sum_i z_i\,p_i(x,a)$, and take its $\arg\max$ — keeping the *same risk-neutral policy* so any change in median HNS is attributable to the richer training signal and not a changed objective. The bootstrap action $a^\star$ is greedy on the target net's mean; the categorical head replaces the scalar head but the conv torso, replay buffer, $\epsilon$-noise, and periodic target sync are exactly as the previous rungs left them. $N=51$ atoms on $[-10,10]$ is the working resolution — enough to resolve the shape without an unwieldy head (this is the "C51" of the name) — and the per-atom projection is $O(N)$ and cheap.

The bar: this is the most fundamental change in the ladder, replacing *what the agent learns* rather than how it learns a mean, and the reason to expect a large, broad gain is the denser supervisory signal — *every* game's return is a non-trivial distribution, so on every game the network now gets many constraints per update where before it got one, shaping a better representation across the whole suite, not a minority. So I expect this to clear 151% decisively and to be the single strongest component the ladder has found from changing *one* thing about the floor. The honest caveat is the fixed grid: $V_{\min},V_{\max}$ and $N=51$ are hand-set, and a game whose returns run outside the support or have fine structure between atoms is served coarsely — a real limit, traded against a much richer learning signal everywhere else. If this is the strongest single component, the last question is whether the six improvements attack *independent* enough weaknesses that combining all of them compounds rather than collides — the finale.

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
