On the genuinely hard-exploration games — Montezuma's Revenge, Pitfall!, Private Eye — the extrinsic reward is so sparse that a plain agent maximizing expected return almost never stumbles onto a positive reward, and its advantage signal is essentially always zero. The dominant remedy adds an intrinsic novelty bonus to the reward stream, $r_t = e_t + \beta i_t$, with $i_t$ large in novel states. Every novelty bonus in use measures novelty differently — forward-prediction error in an inverse-dynamics feature space, distillation error against a frozen random network, a density-model pseudo-count — but they share one structural property, and it is precisely the property that defeats them: the bonus is *transient* and *vanishes* as the agent gains experience. As the predictor learns a region, or its visitation count grows, $i_t$ there decays toward zero. That is exactly right for pushing the frontier outward *once*; it is exactly wrong for what these games actually demand, which is to keep walking *back through* already-cleared rooms attempt after attempt, because the only path to the next undiscovered room runs through the old ones. Once the bonus on the early rooms has decayed, nothing pulls the agent through them. One can try to *tune the decay* so the bonus dies neither too early nor too late, but that is a brittle, per-game knife-edge: too slow and the agent wastes its life re-exploring solved regions, too fast and it gives up before reaching the frontier. We do not want to balance a decay rate; we want a bonus whose structure makes persistent exploration the default. And whatever we build must not be fooled, as the noisy-TV literature warns, into paying novelty for variation the agent cannot control, nor distort the policy on dense-reward games where there is nothing left to explore.

I propose NGU (Never Give Up). The central observation is that "persistent exploration" lives on two completely different timescales that existing bonuses conflate or ignore. A good human explorer, *within* a single attempt, avoids backtracking onto squares already stood on this run — that judgment is sharp and must snap back to "everything is fresh" the instant a new episode starts. *Across* attempts, the same explorer happily re-enters a room cleared a thousand times, because clearing it again is the price of getting deeper — yet a room mastered across all of training really is mastered and should not keep paying a frontier-sized bonus forever. So NGU maintains a *within-episode* novelty that resets every episode and a slow *lifelong* novelty, and — this is the crux — combines them not additively but multiplicatively:

$$i_t = r^{\text{episodic}}_t \cdot \min\!\big\{\max\{\alpha_t, 1\},\, L\big\}, \qquad L = 5.$$

The episodic term is the *driver* of exploration: it carries the reset that makes the drive persistent. The lifelong factor is only a bounded *modulator*. Adding the two would be wrong, and seeing why is the whole point. With addition, once a region is globally mastered the lifelong term still contributes a constant baseline; worse, it can *carry* the bonus where the episodic term has correctly gone to zero, so the agent gets paid for camping in a globally-novel-but-locally-stale spot — the very camping pathology we must close. Multiplication with a floor at 1 means the lifelong factor can only *amplify* the episodic bonus, never shrink it below its episodic value (never give up): even after global novelty has completely vanished and $\alpha_t$ would dip below 1, the agent still receives the full within-episode signal and keeps re-exploring. The cap at $L$ keeps a single anomalous RND spike from multiplying the bonus by a huge factor and yanking the policy toward a fluke. And the limiting behavior is exactly right: as everything is globally mastered, $\alpha_t \to 1$, the modulator $\to 1$, and $i_t \to r^{\text{episodic}}_t$ — the method gracefully reduces to pure episodic novelty, the part that must never decay; while where the episodic term is itself zero (a state already saturated this episode) the product is zero regardless of $\alpha_t$, keeping the camping exploit shut.

The episodic term is a within-episode count bonus. Mechanically we want, at each step, a number that says how much of *this episode's* trajectory has already been spent near the current state. In a tabular world that is a per-episode visitation count $n(s_t)$, emptied at every episode boundary, with the count-based bonus $1/\sqrt{n(s_t)}$ (Strehl & Littman 2008, which carries actual regret guarantees). But these are image states; we will essentially never stand on the exact same pixel-state twice, so a literal count is always 1 and the bonus is constant. We must generalize so that *nearby* states share count mass — a pseudo-count, with an episodic memory as its substrate. Keep a per-environment slot memory $M = \{f(x_0), \dots, f(x_{t-1})\}$ of the embeddings seen so far this episode, emptied each episode, and read it as a differentiable episodic memory is read (Blundell et al. 2016; Pritzel et al. 2017): find the $k$ nearest neighbors $N_k$ of the current embedding $f(x_t)$ and sum a smooth kernel over them as the pseudo-count,

$$r^{\text{episodic}}_t = \frac{1}{\sqrt{\sum_{f_i \in N_k} K\!\big(f(x_t), f_i\big)} + c}, \qquad K(x,y) = \frac{\epsilon}{\dfrac{d^2(x,y)}{d_m^2} + \epsilon}.$$

The constant $c$ ($10^{-3}$) floors the denominator so the very first, memory-empty step gives a finite bonus rather than dividing by zero. The kernel is the inverse kernel from the episodic-memory line because it is both peaked near 1 when two embeddings nearly coincide and *scale-free*: $d$ is Euclidean distance and $d_m^2$ is a running average of the $k$-nearest-neighbor squared distances, so dividing $d^2$ by $d_m^2$ measures distance in "typical-neighbor units." When $d^2 \ll d_m^2$ the kernel is $\approx 1$ (same cluster, full count); when $d^2 \gg d_m^2$ it is $\approx 0$ (different state, no count); and the bonus needs no per-game retuning as embedding magnitudes drift. A Dirac-delta kernel would recover exact counts but generalize nothing, which is useless where exact repeats never happen. Two pathologies of this count have to be defused or an adversarial agent will farm them. First, near-duplicate frames: in Atari consecutive frames are almost identical, and if two essentially-identical embeddings each contribute a fat kernel value, then standing still or oscillating piles up count mass from the agent's own immediate past (or, flipped, tiny jitter registers as new and pays for not moving). So after normalizing to neighbor units we *cluster*: $d_n \leftarrow \max(d_n - \xi,\, 0)$ with floor $\xi = 0.008$, snapping any near-duplicate to distance 0 so it counts as the same state. Second, saturation: if the summed similarity $s = \sqrt{\sum K_v} + c$ exceeds a ceiling $s_m = 8$, the state has been stood at many times this episode and the episodic bonus is set to exactly zero — past a point, "very visited this episode" should mean *no* bonus, not a tiny positive one to chase by camping.

Everything hinges on what the embedding $f$ is, and here the noisy-TV lesson returns directly. If $f$ encodes uncontrollable variation — TV static, moving leaves, the disco-maze whose wall colors are re-randomized every step — then every observation is far from everything in memory, the episodic count stays at its floor forever, and the agent collects a fat "novelty" bonus for standing perfectly still while the world flickers. So $f$ must be a *controllable state*: it must represent what the agent's actions can change and be blind to what they cannot. The construction is the inverse-dynamics embedding (Pathak et al. 2017): train $f$ jointly with a small one-hidden-layer softmax classifier $h$ to recover the action from two consecutive observations, $p(a \mid x_t, x_{t+1}) = h(f(x_t), f(x_{t+1}))$, by maximum likelihood. To predict the action, $f$ *must* encode whatever moved because of the action and has no reason to encode wall colors that carry no action information; nor can it collapse to a constant, since then $h$ could not recover the action at all. We borrow only this representation learner and discard the forward-model-error reward; here the novelty signal is the episodic kNN count, not the forward error.

The lifelong half closes the episodic count's own blind spot: resetting every episode means a state visited ten thousand times across training looks just as fresh at the start of a new episode as one never seen. Early in training that is what we want; late in training we still want the reset-able drive but do not want globally familiar rooms amplified like globally unfamiliar ones. RND-style distillation error supplies exactly the slow global signal needed (Burda et al. 2018): a frozen random target $g$ and a trained predictor $\hat g$ minimizing $\mathrm{err}(x_t) = \|\hat g(x_t) - g(x_t)\|^2$, which is high on globally-unfamiliar states and decreases slowly by gradient descent as a region is seen across training; it is cheap and parallelizes trivially. We use it not as a standalone bonus but as a modulator, normalized to a unitless factor hovering around 1,

$$\alpha_t = 1 + \frac{\mathrm{err}(x_t) - \mu_e}{\sigma_e},$$

with $\mu_e, \sigma_e$ a running mean and standard deviation of the error, then clipped into $[1, L]$ before it multiplies the episodic term.

Because this bonus deliberately *does not* vanish, the persistence we worked for would permanently distort the policy if a single value function were trained on $e_t + \beta i_t$: the agent would keep sacrificing extrinsic return to go sightseeing even on a dense-reward game with nothing to explore, and annealing $\beta$ would just throw the persistence away. The fix is to make the value function a *family* indexed by the intrinsic weight itself — a universal value function approximator (Schaul et al. 2015), one network $Q(x, a, \beta_i)$ that conditions on $\beta_i$ and approximates the optimal value for $r^{\beta_i}_t = e_t + \beta_i i_t$ over a discrete set $\{\beta_i\}_{i=0}^{N-1}$ with $N = 32$. The endpoints are pinned: $\beta_0 = 0$ is a pure exploitative value function that never sees the intrinsic reward, and $\beta_{N-1} = \beta = 0.3$ is the most exploratory. Exploitation is then free: at evaluation, act greedily with respect to $Q(x, a, 0)$ and the exploratory bias is simply gone — no annealing, no switch, it was a separate family member all along, and performance is always reported on $e_t$ alone. Because all $\beta_i$ share weights, the strongly-exploratory members act as auxiliary tasks (Jaderberg et al. 2016) that keep the shared representation and skills improving even before any extrinsic reward is seen, which the $\beta_0$ head exploits the instant a reward appears. We use a whole spread rather than just $\{0, \beta\}$ because the pure-exploit and pure-explore policies can be very different and a network asked to represent both extremes with nothing between has to make a hard jump; a smoothly-spaced ladder of intermediate trade-offs is easier to fit, packed (via a sigmoid schedule) toward the two extremes we care most about. Each $\beta_i$ is paired with its own discount: the exploitative end takes the largest discount $\gamma_0 = 0.997$ to be near-undiscounted on the far-apart extrinsic rewards, the exploratory end a smaller $\gamma_{N-1} = 0.99$ since the intrinsic reward is dense and small-ranged so a short horizon suffices, log-spaced between. Finally, the intrinsic reward depends on the within-episode memory contents, so adding it naively turns the MDP into a POMDP; two things keep it Markov from the agent's view — a recurrent agent that summarizes the within-episode history in its state, and feeding the intrinsic reward, $\beta_i$, the previous action and the previous extrinsic reward in *as inputs*. The natural base is R2D2 (Kapturowski et al. 2019): LSTM state, prioritized replay, off-policy Retrace double-Q learning, many parallel actors — and the bonus is cheap and parallel by construction, so it rides that scale.

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
