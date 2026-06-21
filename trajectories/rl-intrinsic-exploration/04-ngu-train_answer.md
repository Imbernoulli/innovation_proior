RND's numbers are the breakthrough I was watching for, and they are fragile in a way that names the next move exactly. On Private Eye, RND is the only baseline to register a real return — per-seed $\{0, 100, \mathbf{2756}\}$, mean 952, where ICM was a flat zero on every seed. The global, slowly-decaying novelty did what the local forward error could not: it kept marking far-flung states as unfamiliar, so the agent had a reason to keep moving past the first mastered region and across the long reward-free gap. But the rest of the row is the tell: the `auc` mean is still *negative* ($-84.2$), seed 42 is a flat 0.0 with the worst auc of the run ($-260.65$) — it never found a single reward and accumulated penalties — and `nonzero_rate` is 0.667. RND can *reach* Private Eye, but only sometimes and not stably. Tutankham held (mean 100) and Frostbite came in tight and all-nonzero $\{216, 250, 286\}$, more dependable than ICM's jackpot, just without the peak.

The failure to fix is the one that *survives every cure so far*. Every novelty bonus on the ladder — ICM's forward-prediction error, RND's distillation error — shares one property: the bonus *vanishes*. As the agent gathers experience near a region, the predictor learns it and the bonus there decays toward zero. That is exactly right for pushing the frontier outward *once*, and exactly wrong for what Private Eye requires, which is to keep walking *back through* already-seen rooms, episode after episode, because the only path to the next undiscovered room runs through the old ones. Once the bonus on the early rooms has decayed, nothing pulls the agent through them — which is why RND's lucky seed found 2756 but seed 42 stalled at zero with negative auc. And this is *not* ICM's "too local" problem, which RND already fixed by going global; RND's problem is the opposite end, a novelty that is purely *lifelong*, a global quantity that only ever decreases over training, with no notion of "this episode" at all. The natural patch — tune the decay so the bonus does not die before learning catches up — is a calibration knife-edge that depends on the game; I want a bonus whose *structure* makes persistent exploration the default.

So I propose NGU, Never Give Up, and the load-bearing idea is to split novelty into two timescales. Watch a good human explorer: within a single attempt they avoid backtracking onto squares already stood on this run, but they happily re-enter, on a fresh attempt, a room cleared a hundred attempts ago, because clearing it again is the price of getting deeper. *Within* an episode novelty should be sharp and reset-able — a state visited this episode is stale, the judgment snapping back to "everything fresh" the instant a new episode starts. *Across* episodes novelty should be slow — a room cleared ten thousand times across training really is mastered. RND-style distillation error is purely the second kind; none of the prior bonuses has the first kind at all, and that is the gap.

Take the within-episode part first, because it is the new thing. I want, at each step, a number that says how much of *this episode's* trajectory I have already spent near the current state. The cleanest version is a count, following the count-based tradition where $1/\sqrt{n}$ has actual regret guarantees (Strehl & Littman 2008), $r^{\text{episodic}}_t=1/\sqrt{n(s_t)}$: at episode start $n=0$ everywhere and the whole world looks fresh; as the agent re-treads ground this episode the relevant counts climb and those squares stop paying; next episode the counts wipe. But these are image states and the agent will essentially never stand on the *exact* same pixel-state twice, so a literal count is always 1 and the bonus is constant. The count has to be generalized so that *close* states share count mass — a pseudo-count, whose substrate is an episodic memory $M$ holding the embeddings of every state seen so far this episode, $M=\{f(x_0),\dots,f(x_{t-1})\}$, emptied at every episode boundary. Reading the memory the way a differentiable episodic memory is read (Pritzel et al. 2017), I find the $k$ nearest neighbors of $f(x_t)$ and sum a smooth kernel similarity over them as the pseudo-count,

$$r^{\text{episodic}}_t=\frac{1}{\sqrt{\sum_{f_i\in N_k}K\big(f(x_t),f_i\big)}+c},$$

with $c=10^{-3}$ flooring the denominator so the first memory-empty step is finite. A Dirac kernel would recover exact counts but generalize nothing; I want partial count mass to *nearby* states, which is what makes this work where exact repeats never happen.

The kernel must be near 1 when two embeddings nearly coincide, decay as they separate, and be *scale-free* so it behaves the same whether a game's embeddings live at distance 0.1 or 100 apart:

$$K(x,y)=\frac{\epsilon}{\dfrac{d^2(x,y)}{d_m^2}+\epsilon},$$

with $d$ Euclidean, $\epsilon=10^{-3}$, and — the scale-free part — $d_m^2$ a *running average of the squared distances to the $k$ nearest neighbors*. Dividing by $d_m^2$ normalizes distances into typical-neighbor units, so $d^2\ll d_m^2$ gives $K\approx1$ (same cluster, full count) and $d^2\gg d_m^2$ gives $K\approx0$ (different state, no count), with no per-game retuning as embedding magnitudes drift. Two pathologies an adversarial agent would otherwise farm: near-duplicate frames — consecutive Atari frames are almost identical, so standing still or oscillating piles up count mass from the agent's own immediate past — are handled by *clustering* the normalized neighbor distances, $d_n\leftarrow\max(d_n-\xi,0)$ with $\xi=0.008$, snapping anything closer than $\xi$ to distance 0 (counted as the same state, not fresh novelty); and saturation — if the summed similarity $s$ exceeds a ceiling $s_m=8$, the episodic bonus is set to exactly zero, because past a point "very visited this episode" should mean *no* bonus, not a small chase-able positive.

What is $f$, the embedding the memory stores? The noisy-TV lesson comes straight back. If $f$ is raw pixels — or anything encoding uncontrollable variation — the episodic count is wrecked by exactly the flicker that wrecked forward-prediction curiosity: every observation is far from everything in memory, the count sits at its floor, and the agent collects a fat "novelty" bonus for standing still while the world churns. So $f$ must be a *controllable state*, and the construction already exists from step 2 — the inverse-dynamics embedding, trained with a small classifier to recover $a_t$ from $(f(x_t),f(x_{t+1}))$, which forces $f$ to encode what the action changes, ignore what it cannot, and forbids collapse. I borrow only ICM's *representation* learner, not its forward-error reward — the part that decayed locally; here the signal is the episodic kNN count, and step 2's machinery turns out to be the right embedding for step 4's memory.

Now the lifelong half, RND, which I am *not* throwing away — resetting every episode has its own blind spot, since a state visited ten thousand times across training looks just as fresh at a new episode's start as one never seen. Early in training that is what I want; late in training I still want the reset-able drive but I do not want globally-familiar rooms amplified like globally-unfamiliar ones. RND's distillation error $\mathrm{err}(x_t)=\|\hat g(x_t)-g(x_t)\|^2$ is the slow global signal for exactly that, used not as a standalone bonus but as a *modulator*, normalized to hover around 1, $\alpha_t=1+(\mathrm{err}(x_t)-\mu_e)/\sigma_e$, with $\mu_e,\sigma_e$ a running mean and std of the error.

How to combine the two is the crux. The instinct is to *add*, $i_t=r^{\text{episodic}}_t+\alpha_t$ — and that is wrong. With addition, once a region is globally mastered ($\alpha_t\to$ floor) the lifelong term still contributes a constant baseline, and worse, the additive structure lets the lifelong term *carry* the bonus even where the episodic term has correctly gone to zero, so the agent can still be paid for camping in a globally-novel-but-locally-stale spot — exactly the penalty-adjacent behavior that left RND's seed 42 stuck. What I want is for the episodic count, the one with the reset and the persistence, to be the *driver*, and for the lifelong signal only to *scale* it — amplify where the region is globally new, leave alone where it is globally familiar, and never kill the episodic drive. That is multiplicative with a floor at 1:

$$i_t=r^{\text{episodic}}_t\cdot\min\!\big\{\max\{\alpha_t,1\},\,L\big\}.$$

The floor means the lifelong factor can only *boost*, never shrink below the episodic value, so even after global novelty has completely vanished — RND's exact failure on cleared Private Eye rooms — the agent still gets the full within-episode signal and keeps re-exploring; it never gives up. The cap $L=5$ stops a single anomalous RND spike from yanking the policy toward a fluke. The limit is right: as everything is globally mastered, $\alpha_t\to1$, the modulator $\to1$, and $i_t\to r^{\text{episodic}}_t$ — pure episodic novelty, the part that should never decay; and where the episodic bonus is itself zero (saturated this episode) the product is zero regardless of $\alpha_t$, so camping stays closed.

That bonus is the part of full NGU that this task's fixed harness lets me build: the editable surface is `IntrinsicBonusModule` + `mix_advantages`, and the fixed loop already carries two value heads with separate discounts ($\gamma=0.999$ extrinsic, `int_gamma`$=0.99$ intrinsic) and the mixing coefficients. Full NGU goes one step further — because a bonus that deliberately never vanishes bakes the exploratory drive permanently into the value function, it holds an exploratory and an exploitative policy at once via a UVFA family $Q(x,a,\beta_i)$ over a ladder of intrinsic weights from $\beta_0=0$ (pure exploit) to $\beta=0.3$, each with its own discount ($0.997\to0.99$), fed with $\beta_i$ into a *recurrent* learner so the memory-dependent reward stays Markov from the agent's view. The fixed `Agent` here is neither $\beta$-conditioned nor recurrent, so within this task the lever I truly control is the two-timescale bonus; I keep the $\beta$/$\gamma$ schedule helpers in place (the natural form of the method) and mix with a single intrinsic coefficient, accepting that the exploit/explore separation is the part the harness does not expose. The hyperparameters are $P=32$ (embedding dimension), $k=10$, $\epsilon=10^{-3}$, $\xi=0.008$, $c=10^{-3}$, $s_m=8$, $L=5$, with a per-env episodic memory reset at each episode boundary.

This task's leaderboard has no NGU row, so the falsifiable claims are on the *worst* seed and on `auc`, not the headline eval: if the episodic reset is the right fix, Private Eye's `nonzero_rate` should move off 0.667 toward 1.0 (seed 42 stops finding nothing) and its `auc` should lift from $-84$ toward positive (less time in the red) *before* the peak eval moves, while Tutankham ($\sim$100) and Frostbite ($\sim$250) hold, since the multiplicative floor reduces the bonus to pure episodic novelty where there is nothing global left to chase. What the trajectory ends on is the construction whose two timescales make persistent re-traversal the default rather than a tuned accident.

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
