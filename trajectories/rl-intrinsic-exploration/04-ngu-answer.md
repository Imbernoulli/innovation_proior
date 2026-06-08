**Problem (from step 3).** RND broke Private Eye open (one seed at 2756) but is fragile — negative
`auc`, a dead seed — because its novelty is purely *lifelong* and monotonically decreasing: once a
region's global novelty wears off, nothing re-pays the agent to walk back through cleared rooms,
episode after episode, which is exactly what the game needs.

**Key idea.** Split novelty into two timescales. A reset-able *within-episode* kNN pseudo-count
$r^{\text{epi}}_t=1/(\sqrt{\sum_{N_k}K(f(x_t),f_i)}+c)$ over an episodic memory of
controllable-state embeddings (inverse-dynamics $f$, reused from step 2), with a scale-free inverse
kernel normalized by a running $d_m^2$ updated from the full list of $k$ NN squared distances (mean
of the list, not only the $k$-th distance), a cluster floor $\xi$ for near-duplicates, and a saturation
cap $s_m$. Modulate it by RND's *lifelong* error as
$\alpha_t=1+(\text{err}-\mu_e)/\sigma_e$, combined **multiplicatively with a floor at 1**:
$i_t=r^{\text{epi}}_t\cdot\operatorname{clip}(\alpha_t,1,L)$.

**Why it works.** The floor lets the lifelong factor only *amplify* globally-novel regions, never kill
the episodic drive — so even on cleared rooms the agent keeps re-exploring (it never gives up); where
the episodic term saturates, the product is zero, closing the camping exploit. In the limit
$\alpha\to1$ it reduces to pure episodic novelty, the part that must not decay.

**Scaffold edit / hyperparameters.** $P=32$, $k=10$, $\epsilon=10^{-3}$, $\xi=0.008$, $c=10^{-3}$,
$s_m=8$, $L=5$.
Encoder + inverse model (controllable state), frozen-target + deeper-predictor RND (lifelong), a
per-env episodic memory reset at each episode boundary. The natural form also holds a UVFA family
$Q(x,a,\beta_i)$ from $\beta_0=0$ (pure exploit) to $\beta=0.3$ with discounts $0.997\to0.99$ and a
recurrent learner — the part the fixed `Agent` here does not expose; within this task the lever is the
two-timescale bonus, mixed by the loop's two value heads.

**The bar to beat.** No leaderboard NGU row exists; the strongest baseline is RND (Private Eye
`eval_return` 952 / `auc` −84.2 / `nonzero_rate` 0.667). The falsifiable claims are on the *worst* seed
and on `auc`: `nonzero_rate` should move off 0.667 toward 1.0 and `auc` toward positive *before* the
peak eval, with Tutankham/Frostbite held.

```python
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# provided by the fixed loop: layer_init, RunningMeanStd, RewardForwardFilter, last_frame, Args

P = 32      # controllable-state (embedding) dimension
K = 10      # nearest neighbors for the episodic pseudo-count
EPS = 1e-3  # kernel epsilon
XI = 0.008  # cluster floor: distances below this (typical-neighbor units) -> same state
C = 1e-3    # pseudo-count constant (denominator floor)
S_M = 8.0   # max similarity: above this the state is saturated this episode -> zero bonus
L = 5.0     # cap on the lifelong modulator
N_BETAS = 32
BETA_MAX = 0.3
GAMMA_MAX = 0.997
GAMMA_MIN = 0.99


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
    """Per-env within-episode memory of controllable-state embeddings, reset each episode.
    Holds a running average d_m^2 of the kNN squared distances for scale-free kernels."""
    def __init__(self, n_envs, device):
        self.device = device
        self.slots = [[] for _ in range(n_envs)]          # M = {f(x_0),...,f(x_{t-1})} per env
        self.d2_m = np.ones(n_envs, dtype=np.float64)     # running mean of the kNN squared distances

    def reset(self, env_idx):                              # called at every episode boundary
        self.slots[env_idx] = []

    def episodic_reward(self, env_idx, emb):              # emb: f(x_t), shape (P,)
        M = self.slots[env_idx]
        if len(M) == 0:
            r_epi = float(1.0 / C)                         # empty sum: 1 / (sqrt(0) + c)
        else:
            d2 = ((torch.stack(M) - emb) ** 2).sum(dim=1)  # squared distances to memory
            k = min(K, d2.numel())
            d2_k, _ = torch.topk(d2, k, largest=False, sorted=True)
            d2_k = d2_k.cpu().numpy().astype(np.float64)
            self.d2_m[env_idx] = 0.99 * self.d2_m[env_idx] + 0.01 * d2_k.mean()  # Alg.1: update with full d_k list
            d_n = d2_k / max(self.d2_m[env_idx], 1e-12)    # Alg.1: d_n = d_k / d_m^2
            d_n = np.maximum(d_n - XI, 0.0)                # cluster: tiny distances -> 0
            k_v = EPS / (d_n + EPS)                        # inverse kernel
            s = np.sqrt(k_v.sum()) + C                     # sqrt(sum_{N_k} K) + c
            r_epi = 0.0 if s > S_M else float(1.0 / s)     # saturated -> 0, else 1/s
        M.append(emb.detach())                             # append after reward; M holds past states
        return r_epi


class IntrinsicBonusModule(nn.Module):
    """Episodic kNN-count novelty over an inverse-dynamics controllable state,
    multiplicatively modulated by a normalized, clipped RND lifelong-novelty factor."""

    def __init__(self, action_dim, device, args):
        super().__init__()
        self.action_dim = action_dim; self.device = device; self.args = args
        self.obs_rms = RunningMeanStd(shape=(1, 1, 84, 84))
        self.reward_rms = RunningMeanStd()
        self.discounted_reward = RewardForwardFilter(args.int_gamma)
        self.err_rms = RunningMeanStd()                   # mu_e, sigma_e for the RND modulator
        self.memory = EpisodicMemory(args.num_envs, device)
        feature_output = 7 * 7 * 64
        self.encoder = nn.Sequential(                     # controllable state f, inverse-dynamics trained
            layer_init(nn.Conv2d(1, 32, 8, stride=4)), nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)), nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)), nn.ReLU(),
            nn.Flatten(), layer_init(nn.Linear(feature_output, P)))
        self.inverse_model = nn.Sequential(               # predict a_t from (f(x_t), f(x_{t+1}))
            layer_init(nn.Linear(2 * P, 256)), nn.ReLU(),
            layer_init(nn.Linear(256, action_dim), std=0.01))
        self.predictor = nn.Sequential(                   # lifelong RND predictor (deeper, trainable)
            layer_init(nn.Conv2d(1, 32, 8, stride=4)), nn.LeakyReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)), nn.LeakyReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)), nn.LeakyReLU(),
            nn.Flatten(), layer_init(nn.Linear(feature_output, 512)), nn.ReLU(),
            layer_init(nn.Linear(512, 512)), nn.ReLU(), layer_init(nn.Linear(512, 512)))
        self.target = nn.Sequential(                      # lifelong RND target (frozen random)
            layer_init(nn.Conv2d(1, 32, 8, stride=4)), nn.LeakyReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)), nn.LeakyReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)), nn.LeakyReLU(),
            nn.Flatten(), layer_init(nn.Linear(feature_output, 512)))
        for p in self.target.parameters():
            p.requires_grad = False

    def initialize(self, envs):                           # warm obs-norm stats (frozen target can't adapt scale)
        bootstrap, total = [], self.args.num_steps * self.args.num_iterations_obs_norm_init
        for _ in range(total):
            a = np.random.randint(0, envs.single_action_space.n, size=(self.args.num_envs,))
            bootstrap.append(envs.step(a)[0][:, 3:4, :, :])
            if len(bootstrap) >= self.args.num_steps:
                self.obs_rms.update(np.concatenate(bootstrap, axis=0)); bootstrap.clear()

    def trainable_parameters(self):
        return list(self.encoder.parameters()) + list(self.inverse_model.parameters()) \
            + list(self.predictor.parameters())           # target excluded (frozen)

    def _normalize_obs(self, obs):                        # predictor/target only (NOT the policy)
        mean = torch.from_numpy(self.obs_rms.mean).to(self.device)
        var = torch.from_numpy(self.obs_rms.var).to(self.device)
        return ((last_frame(obs) - mean) / torch.sqrt(var)).clip(-5, 5).float()

    def update_batch_stats(self, batch_obs, batch_next_obs):
        self.obs_rms.update(last_frame(batch_next_obs).cpu().numpy())

    @torch.no_grad()
    def compute_bonus(self, obs, next_obs, actions):
        emb = self.encoder(last_frame(next_obs).float())  # controllable state
        r_epi = torch.tensor(
            [self.memory.episodic_reward(i, emb[i]) for i in range(emb.shape[0])],
            device=self.device, dtype=torch.float32)
        norm_next = self._normalize_obs(next_obs)
        err = (self.predictor(norm_next) - self.target(norm_next)).pow(2).sum(1)
        self.err_rms.update(err.cpu().numpy())
        sigma = float(np.sqrt(self.err_rms.var + 1e-8))
        alpha = (1.0 + (err - float(self.err_rms.mean)) / sigma).clamp(min=1.0, max=L)
        return (r_epi * alpha).detach()                   # i_t = r_episodic * clip(alpha, 1, L)

    def reset_memory(self, env_idx):
        self.memory.reset(env_idx)                        # call on each episode boundary

    def normalize_rollout_rewards(self, rollout_intrinsic):
        discounted = np.stack(
            [self.discounted_reward.update(r) for r in rollout_intrinsic.cpu().numpy()], axis=0)
        flat = discounted.reshape(-1)
        self.reward_rms.update_from_moments(float(flat.mean()), float(flat.var()), int(flat.size))
        return rollout_intrinsic / float(np.sqrt(self.reward_rms.var + 1e-8))

    def loss(self, batch_obs, batch_next_obs, batch_actions):
        f_t = self.encoder(last_frame(batch_obs).float())      # inverse dynamics: anchors f
        f_tp1 = self.encoder(last_frame(batch_next_obs).float())
        logits = self.inverse_model(torch.cat([f_t, f_tp1], dim=1))
        inverse_loss = F.cross_entropy(logits, batch_actions.long())
        norm_next = self._normalize_obs(batch_next_obs)        # RND distillation: lifelong novelty
        rnd_loss = F.mse_loss(self.predictor(norm_next),
                              self.target(norm_next).detach(), reduction="none").sum(-1)
        mask = (torch.rand(len(rnd_loss), device=self.device) < self.args.update_proportion).float()
        rnd_loss = (rnd_loss * mask).sum() / torch.clamp(mask.sum(), min=1.0)
        return inverse_loss + rnd_loss


def mix_advantages(ext_advantages, int_advantages, args):
    beta_i = getattr(args, "beta_i", getattr(args, "int_coef", BETA_MAX))   # one member Q(x,a,beta_i)
    return getattr(args, "ext_coef", 1.0) * ext_advantages + beta_i * int_advantages
```
