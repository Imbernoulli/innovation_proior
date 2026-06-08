# NGU (Never Give Up)

**Problem.** On hard-exploration sparse-reward games (Montezuma's Revenge, Pitfall!, Private Eye)
every standard novelty bonus *vanishes* as the agent gains experience: it pushes the frontier once
but gives no drive to re-traverse cleared regions, which is exactly what reaching the next
undiscovered area requires. The exploration signal must instead be *persistent*.

**Key idea — novelty on two timescales, combined multiplicatively.** Maintain a *within-episode*
novelty that resets every episode (so the world looks fresh each attempt and the agent keeps
re-exploring) and a slow *lifelong* novelty that removes extra amplification from regions seen across
many episodes.
$$i_t = r^{\text{episodic}}_t \cdot \min\!\big\{\max\{\alpha_t,1\},\,L\big\},\qquad L=5.$$
The episodic term is the driver; the lifelong factor is a bounded modulator. Floored at 1 it can
only *amplify*, never kill, the episodic drive (never give up); capped at $L$ a spurious spike
cannot explode the bonus. As everything is globally mastered $\alpha_t\to1$ and $i_t\to
r^{\text{episodic}}_t$.

**Episodic novelty — kNN pseudo-count over a controllable-state memory.** A per-episode slot memory
$M=\{f(x_0),\dots,f(x_{t-1})\}$, emptied each episode. The bonus is a count bonus
$1/\sqrt{n}$ (Strehl & Littman 2008) with the count approximated by a kernel sum over the $k$
nearest neighbors:
$$r^{\text{episodic}}_t=\frac{1}{\sqrt{\sum_{f_i\in N_k}K(f(x_t),f_i)}+c},\qquad
K(x,y)=\frac{\epsilon}{\dfrac{d^2(x,y)}{d_m^2}+\epsilon}.$$
$d$ is Euclidean distance; $d_m^2$ is a running average of the kNN squared distances, updated from
the full list $d_k$ (the mean of the $k$ neighbor squared distances, not only the $k$-th one), making
the kernel scale-free per game. Computation (Alg. 1): get the $k$ NN squared distances $d_k$; update
$d_m^2$ with that list; normalize $d_n=d_k/d_m^2$; **cluster** $d_n\leftarrow\max(d_n-\xi,0)$ so
near-duplicate frames count as the same state; $K_v=\epsilon/(d_n+\epsilon)$;
$s=\sqrt{\sum K_v}+c$; if $s>s_m$ the state is saturated this episode and
$r^{\text{episodic}}_t=0$, else $1/s$.
Constants: $p=32$, $k=10$, $\epsilon=10^{-3}$, $c=10^{-3}$, $\xi=0.008$, $s_m=8$.

**Controllable-state embedding $f$ (inverse dynamics).** $f$ is trained jointly with a one-hidden-
layer softmax classifier $h$ to recover the action from consecutive observations,
$p(a\mid x_t,x_{t+1})=h(f(x_t),f(x_{t+1}))$, by maximum likelihood (Pathak et al. 2017). $f$ thus
encodes only action-relevant content and is blind to uncontrollable variation (TV static, the
disco-maze color churn), so flicker cannot manufacture episodic novelty.

**Lifelong novelty — RND modulator.** Frozen random target $g$, trained predictor $\hat g$
minimizing $\mathrm{err}(x_t)=\|\hat g(x_t)-g(x_t)\|^2$; normalize to
$\alpha_t=1+(\mathrm{err}(x_t)-\mu_e)/\sigma_e$ with running mean/std $\mu_e,\sigma_e$, clipped into
$[1,L]$. A slow, global signal that decreases as a region is seen across training.

**Agent — UVFA family + recurrent distributed value learning.** Because the bonus never vanishes, a
single mixed-reward value function would permanently distort the policy. Train a UVFA family
$Q(x,a,\beta_i)$ (Schaul et al. 2015) over $r^{\beta_i}_t=e_t+\beta_i i_t$ with $N=32$ mixtures,
$\beta_0=0$ (pure exploit, greedily retrievable at evaluation), $\beta_{N-1}=\beta=0.3$ (max
explore), sigmoid-spaced. Each $\beta_i$ has its own discount, log-spaced from $\gamma_0=0.997$
(exploit, near-undiscounted) to $\gamma_{N-1}=0.99$ (explore; intrinsic reward is dense and
small-ranged). Shared weights make the exploratory members auxiliary tasks that improve the
representation even with no extrinsic reward. The base is R2D2 (Kapturowski et al. 2019): LSTM
state, prioritized replay, off-policy Retrace ($\lambda=0.95$) double-Q learning, many parallel
actors. The intrinsic reward, $\beta_i$, the previous action and previous extrinsic reward are fed
as network inputs so the non-stationary augmented reward stays Markov. Performance is always
reported on the extrinsic reward only.

```python
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# provided by the fixed loop: layer_init, RunningMeanStd, RewardForwardFilter, last_frame, Args

P, K = 32, 10
EPS, XI, C, S_M, L = 1e-3, 0.008, 1e-3, 8.0, 5.0
N_BETAS, BETA_MAX = 32, 0.3
GAMMA_MAX, GAMMA_MIN = 0.997, 0.99


def beta_schedule(n=N_BETAS, beta=BETA_MAX):
    values = np.zeros(n, dtype=np.float32)
    values[-1] = beta
    for i in range(1, n - 1):
        values[i] = beta / (1.0 + np.exp(-10.0 * (2 * i - (n - 2)) / (n - 2)))
    return values


def gamma_schedule(n=N_BETAS, gamma_max=GAMMA_MAX, gamma_min=GAMMA_MIN):
    i = np.arange(n, dtype=np.float64)
    log_gap = ((n - 1 - i) * np.log(1.0 - gamma_max) + i * np.log(1.0 - gamma_min)) / (n - 1)
    return (1.0 - np.exp(log_gap)).astype(np.float32)


class EpisodicMemory:
    """Per-env within-episode memory of controllable states; reset each episode (Alg. 1)."""
    def __init__(self, n_envs, device):
        self.device = device
        self.slots = [[] for _ in range(n_envs)]
        self.d2_m = np.ones(n_envs, dtype=np.float64)        # running mean of the kNN squared distances

    def reset(self, i):
        self.slots[i] = []

    def episodic_reward(self, i, emb):
        M = self.slots[i]
        if not M:
            r_epi = float(1.0 / C)                           # empty sum: 1 / (sqrt(0) + c)
        else:
            d2 = ((torch.stack(M) - emb) ** 2).sum(dim=1)
            k = min(K, d2.numel())
            d2_k = torch.topk(d2, k, largest=False, sorted=True)[0].cpu().numpy().astype(np.float64)
            self.d2_m[i] = 0.99 * self.d2_m[i] + 0.01 * d2_k.mean()   # Alg.1: update with full d_k list
            d_n = d2_k / max(self.d2_m[i], 1e-12)             # Alg.1: d_n = d_k / d_m^2
            d_n = np.maximum(d_n - XI, 0.0)                  # cluster
            k_v = EPS / (d_n + EPS)
            s = np.sqrt(k_v.sum()) + C
            r_epi = 0.0 if s > S_M else float(1.0 / s)
        M.append(emb.detach())
        return r_epi


class IntrinsicBonusModule(nn.Module):
    """NGU: episodic kNN-count novelty over an inverse-dynamics controllable state,
    multiplicatively modulated by a normalized, clipped RND lifelong factor."""
    def __init__(self, action_dim, device, args):
        super().__init__()
        self.action_dim, self.device, self.args = action_dim, device, args
        self.obs_rms = RunningMeanStd(shape=(1, 1, 84, 84))
        self.reward_rms = RunningMeanStd()
        self.err_rms = RunningMeanStd()
        self.discounted_reward = RewardForwardFilter(args.int_gamma)
        self.memory = EpisodicMemory(args.num_envs, device)
        fo = 7 * 7 * 64
        self.encoder = nn.Sequential(
            layer_init(nn.Conv2d(1, 32, 8, stride=4)), nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)), nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)), nn.ReLU(),
            nn.Flatten(), layer_init(nn.Linear(fo, P)))
        self.inverse_model = nn.Sequential(
            layer_init(nn.Linear(2 * P, 256)), nn.ReLU(),
            layer_init(nn.Linear(256, action_dim), std=0.01))
        self.predictor = nn.Sequential(
            layer_init(nn.Conv2d(1, 32, 8, stride=4)), nn.LeakyReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)), nn.LeakyReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)), nn.LeakyReLU(),
            nn.Flatten(), layer_init(nn.Linear(fo, 512)), nn.ReLU(),
            layer_init(nn.Linear(512, 512)), nn.ReLU(), layer_init(nn.Linear(512, 512)))
        self.target = nn.Sequential(
            layer_init(nn.Conv2d(1, 32, 8, stride=4)), nn.LeakyReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)), nn.LeakyReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)), nn.LeakyReLU(),
            nn.Flatten(), layer_init(nn.Linear(fo, 512)))
        for p in self.target.parameters():
            p.requires_grad = False

    def initialize(self, envs):
        buf, total = [], self.args.num_steps * self.args.num_iterations_obs_norm_init
        for _ in range(total):
            a = np.random.randint(0, envs.single_action_space.n, size=(self.args.num_envs,))
            o = envs.step(a)[0]; buf.append(o[:, 3:4, :, :])
            if len(buf) >= self.args.num_steps:
                self.obs_rms.update(np.concatenate(buf, axis=0)); buf.clear()

    def trainable_parameters(self):
        return list(self.encoder.parameters()) + list(self.inverse_model.parameters()) \
            + list(self.predictor.parameters())

    def _normalize_obs(self, obs):
        mean = torch.from_numpy(self.obs_rms.mean).to(self.device)
        var = torch.from_numpy(self.obs_rms.var).to(self.device)
        return ((last_frame(obs) - mean) / torch.sqrt(var)).clip(-5, 5).float()

    def update_batch_stats(self, batch_obs, batch_next_obs):
        self.obs_rms.update(last_frame(batch_next_obs).cpu().numpy())

    def reset_memory(self, i):
        self.memory.reset(i)

    @torch.no_grad()
    def compute_bonus(self, obs, next_obs, actions):
        emb = self.encoder(last_frame(next_obs).float())
        r_epi = torch.tensor([self.memory.episodic_reward(i, emb[i]) for i in range(emb.shape[0])],
                             device=self.device, dtype=torch.float32)
        nn_obs = self._normalize_obs(next_obs)
        err = (self.predictor(nn_obs) - self.target(nn_obs)).pow(2).sum(1)
        self.err_rms.update(err.cpu().numpy())
        alpha = (1.0 + (err - float(self.err_rms.mean)) /
                 float(np.sqrt(self.err_rms.var + 1e-8))).clamp(1.0, L)
        return (r_epi * alpha).detach()

    def normalize_rollout_rewards(self, rollout_intrinsic):
        disc = np.stack([self.discounted_reward.update(r)
                         for r in rollout_intrinsic.cpu().numpy()], axis=0).reshape(-1)
        self.reward_rms.update_from_moments(float(disc.mean()), float(disc.var()), int(disc.size))
        return rollout_intrinsic / float(np.sqrt(self.reward_rms.var + 1e-8))

    def loss(self, batch_obs, batch_next_obs, batch_actions):
        f_t = self.encoder(last_frame(batch_obs).float())
        f_tp1 = self.encoder(last_frame(batch_next_obs).float())
        inv = F.cross_entropy(self.inverse_model(torch.cat([f_t, f_tp1], 1)), batch_actions.long())
        nn_obs = self._normalize_obs(batch_next_obs)
        rnd = F.mse_loss(self.predictor(nn_obs), self.target(nn_obs).detach(),
                         reduction="none").sum(-1)
        mask = (torch.rand(len(rnd), device=self.device) < self.args.update_proportion).float()
        return inv + (rnd * mask).sum() / torch.clamp(mask.sum(), min=1.0)


def mix_advantages(ext_advantages, int_advantages, args):
    beta_i = getattr(args, "beta_i", getattr(args, "int_coef", BETA_MAX))
    return getattr(args, "ext_coef", 1.0) * ext_advantages + beta_i * int_advantages
```
