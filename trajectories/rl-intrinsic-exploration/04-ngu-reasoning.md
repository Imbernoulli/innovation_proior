RND's numbers are the breakthrough I was watching for — and they are fragile in a way that names the next move exactly. On Private Eye, RND is the only baseline to register a real return: per-seed {0, 100, **2756**}, mean 952, where ICM was a flat zero on every seed. The global, slowly-decaying novelty did what the local forward error could not — it kept marking far-flung states as unfamiliar, so the agent had a reason to keep moving past the first mastered region and across the long reward-free gap. That is the single most important result on the ladder. But read the rest of the Private Eye row: the `auc` mean is still **negative** (−84.2), seed 42 is a flat 0.0 with the *worst* auc of the run (−260.65) — it never found a single reward and accumulated penalties along the way — and `nonzero_rate` is 0.667. So the 952 mean is one strong seed, one modest, one dead. RND can *reach* Private Eye, but only sometimes, and not stably. Tutankham held (mean 100, close to ICM's 109) and Frostbite came in tight and all-nonzero {216, 250, 286} — more dependable than ICM's jackpot, just without the peak.

So the failure to fix is precise, and it is *not* "the signal is too local" — that was ICM's problem, which RND already cured by going global. RND's problem is the opposite end: its novelty is purely *lifelong*. The distillation error on a region only ever decreases across training, and it has no notion of "this episode" at all. So once a region's global novelty has worn off, nothing pulls the agent back through it — and at the start of every fresh episode a globally-mastered room looks exactly as stale as it did at the end of the last one. Stare at what Private Eye demands and the gap is obvious: to extend its reach the agent has to re-traverse cleared ground, episode after episode, because the only path to the next undiscovered room runs through the old ones. A lifelong-only bonus has, by construction, stopped paying for that traversal — which is exactly why the seed that found 2756 can't *reliably* reproduce the journey, and why seed 42 stalls at zero. Every novelty bonus on the ladder so far — ICM's forward error, RND's distillation error — is a one-shot frontier-pusher that vanishes, and vanishing is precisely wrong for a game that needs *persistent, repeated* exploration.

A natural patch is to *tune the decay* — slow the bonus so it doesn't die before the agent gets through. But that's a calibration knife-edge: too slow and the agent wastes its life re-exploring solved regions, too fast and it gives up before the frontier, and the right rate depends on the game. I don't want to balance a decay rate against Private Eye's particular geometry. I want a bonus whose *structure* makes persistent exploration the default.

So what does "persistent exploration" mean mechanically? Watch a human explorer: within a single attempt they avoid backtracking onto squares they've already stood on this run; but they happily re-enter, on a fresh attempt, a room they cleared a hundred attempts ago, because clearing it again is the price of getting deeper. Two completely different timescales. *Within* an episode, novelty should be sharp and reset-able: a state visited this episode is stale, one not visited is fresh, and that judgment snaps back to "everything fresh" the instant a new episode begins. *Across* episodes, novelty should be slow: a room cleared ten thousand times across all of training really is mastered. RND is purely the second kind — global, monotonically decreasing, no notion of "this episode." ICM was also the second kind. *Neither has the first kind at all.* That missing within-episode timescale is the whole fix: keep RND's lifelong signal, add a reset-able episodic one, and combine them so the agent never gives up re-exploring.

Take the within-episode part first, since it's the new thing. I want, each step, a number for "how much of *this episode's* trajectory have I already spent near the current state." The cleanest version is a count: how many times have I been at (something like) this state since the episode began? Following the count-based tradition where $1/\sqrt{n}$ has real regret guarantees (Strehl & Littman 2008),
$$r^{\text{episodic}}_t=\frac{1}{\sqrt{n(s_t)}}.$$
At episode start $n=0$ everywhere, the whole world is fresh; as the agent re-treads ground this episode the relevant counts climb and those squares stop paying; next episode the counts wipe and the world is fresh again — exactly the reset-able sharp novelty RND lacks. But these are image states; the agent will essentially never stand on the *exact* same pixel-state twice, so a literal count is always 1. I have to generalize: nearby states should share count mass.

That's a pseudo-count, and its substrate is an episodic memory $M$ holding the embeddings of every state seen *so far this episode*, emptied at each episode boundary. Read it the way differentiable episodic memories are read (Pritzel et al. 2017): find the $k$ nearest neighbors of $f(x_t)$ and sum a smooth kernel over them as the pseudo-count,
$$r^{\text{episodic}}_t=\frac{1}{\sqrt{\sum_{f_i\in N_k}K\big(f(x_t),f_i\big)}+c}.$$
The constant $c$ (small, $10^{-3}$) floors the denominator so the first memory-empty step is finite, not undefined. For the kernel I want similarity $\approx 1$ when two embeddings nearly coincide, decaying as they separate, and *scale-free* so it behaves the same whatever a game's embedding magnitudes happen to be:
$$K(x,y)=\frac{\epsilon}{\dfrac{d^2(x,y)}{d_m^2}+\epsilon},$$
with $d$ Euclidean and $d_m^2$ a *running average of the squared distance to the $k$-th nearest neighbor* — dividing by it normalizes distances into "typical-neighbor units," so the kernel needs no per-game retuning.

Two pathologies of this count an adversarial agent would farm. First, near-duplicate frames: consecutive Atari frames are almost identical, so standing still would pile up count mass from the agent's own immediate past (or tiny jitter would read as "new"). So after normalizing, *cluster*: $d_n\leftarrow\max(d_n-\xi,0)$, snapping anything closer than $\xi$ to distance 0 (the same state, maximal kernel — not fresh novelty). Second, saturation: if the summed similarity $s$ exceeds a ceiling $s_m$, set the episodic bonus to exactly zero — past a point, "very visited this episode" should mean *no* bonus, not a small chase-able positive one.

What is $f$, the embedding the memory stores? This is where the noisy-TV lesson comes straight back. If $f$ is raw pixels — or anything encoding uncontrollable variation — the episodic count is wrecked by exactly the flicker that wrecked forward-prediction curiosity: every frame is far from everything in memory, the count stays at its floor, and the agent collects a fat "novelty" bonus for standing still while the world churns. So $f$ must be a *controllable state*. The construction already exists and I already used it at step 2: the inverse-dynamics embedding (Pathak et al. 2017), trained with a small classifier $h$ to recover the action from $(f(x_t),f(x_{t+1}))$, which forces $f$ to encode what the action changes and ignore what it can't, and forbids collapse. So I borrow only ICM's *representation* learner — not its forward-error reward — because here the signal is the episodic kNN count, not the forward error. Step 2's machinery wasn't wasted; it's the right embedding for step 4's memory.

Now bring back the lifelong half — RND, which I am *not* throwing away, because resetting every episode has its own blind spot: a state visited ten thousand times across training looks just as fresh at a new episode's start as one never seen. Early in training that's fine (re-explore everything); late in training I still want the reset-able drive but I don't want globally-familiar rooms amplified like globally-unfamiliar ones. RND's distillation error $\mathrm{err}(x_t)=\|\hat g(x_t)-g(x_t)\|^2$ is exactly the slow, global signal for that — high on globally-unfamiliar states, decreasing slowly as a region is seen across training, and cheap. Use it as a *modulator*, normalized to hover around 1:
$$\alpha_t=1+\frac{\mathrm{err}(x_t)-\mu_e}{\sigma_e}.$$

How to combine? My first instinct is to add, $i_t=r^{\text{episodic}}_t+\alpha_t$ — but that's wrong, and seeing why is the crux. Addition lets the lifelong term *carry* the bonus even where the episodic term has correctly gone to zero, so the agent could still be paid for camping in a globally-novel-but-locally-stale spot — reintroducing exactly the camping pathology, and exactly the kind of fragile behavior that left RND's seed 42 stuck. What I want is for the episodic count — the one with the reset, the one that gives persistence — to be the *driver*, and for the lifelong signal only to *scale* it: amplify where the region is globally new, leave alone where it's globally familiar, and *never* kill the episodic drive. That's multiplicative with a floor at 1:
$$i_t=r^{\text{episodic}}_t\cdot\min\!\big\{\max\{\alpha_t,1\},\,L\big\}.$$
The floor means the lifelong factor can only *boost*, never shrink below the episodic value — so even after global novelty has completely vanished (RND's exact failure on cleared Private Eye rooms) the agent still gets the full within-episode signal and keeps re-exploring; it never gives up. The cap $L=5$ stops a single anomalous RND spike from yanking the policy toward a fluke. And the limit is right: as everything is globally mastered, $\alpha_t\to1$, the modulator $\to1$, and $i_t\to r^{\text{episodic}}_t$ — it reduces to pure episodic novelty, the part that should never decay. Where the episodic bonus is itself zero (saturated this episode), the product is zero regardless of $\alpha_t$, so camping stays closed.

The bonus settled, this construction creates a *second* problem I must handle or the easy games regress. By design the bonus *does not vanish* — the episodic term pays forever. That's the feature that fixes Private Eye, but it means a value function trained on $e_t+\beta i_t$ has the exploratory drive *permanently* baked in; on a game where there's nothing left to explore the policy will keep sacrificing return to sightsee — and I won't fix that by annealing $\beta$, which would throw away the persistence I just built. I need to hold an exploratory and an exploitative policy *at once*. Use a universal value function (Schaul et al. 2015): one network $Q(x,a,\beta_i)$ conditioned on the intrinsic weight, approximating the optimal value for $r^{\beta_i}_t=e_t+\beta_i i_t$ over a discrete set $\{\beta_i\}_{i=0}^{N-1}$ with pinned endpoints — $\beta_0=0$ a *pure exploit* head (it never sees the bonus) and $\beta_{N-1}=\beta$ the most exploratory. Exploitation is then free: act greedily w.r.t. $Q(x,a,0)$ and the exploratory bias is simply gone, no switch to flip. The strongly-exploratory members act as auxiliary tasks that keep the shared representation improving even before any extrinsic reward is seen — which the $\beta_0$ head exploits the instant a reward appears. A spread of $N$ (not just $\{0,\beta\}$) makes the family easier to fit, packed toward the extremes, each $\beta_i$ paired with its own discount: $\gamma_0=0.997$ at the exploit end (extrinsic rewards are far apart, want a long horizon) down to $\gamma_{N-1}=0.99$ at the explore end (the bonus is dense and short-ranged).

One last subtlety: the intrinsic reward depends on the episodic memory's contents, i.e. the whole within-episode history, so from a memoryless state the reward looks unpredictable — adding it naively turns the MDP into a POMDP. Keep it Markov-from-the-agent's-view by using the recurrent value learner the fixed loop already provides and feeding the intrinsic reward and $\beta_i$ *as inputs*, so the value function can see the very quantity that shifted the reward. At the bonus-module boundary the loop fixes — a per-transition bonus, a rollout normalizer, a training loss, and an advantage-mixing function — this is:

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
    Holds a running average d_m^2 of the k-th-NN squared distance for scale-free kernels."""
    def __init__(self, n_envs, device):
        self.device = device
        self.slots = [[] for _ in range(n_envs)]          # M = {f(x_0),...,f(x_{t-1})} per env
        self.d2_m = np.ones(n_envs, dtype=np.float64)     # running mean of k-th-NN squared distance

    def reset(self, env_idx):                              # called at every episode boundary
        self.slots[env_idx] = []

    def episodic_reward(self, env_idx, emb):              # emb: f(x_t), shape (P,)
        M = self.slots[env_idx]
        self.slots[env_idx] = M + [emb]                   # append AFTER reading
        if len(M) == 0:
            return float(1.0 / C)                         # empty sum: 1 / (sqrt(0) + c)
        d2 = ((torch.stack(M) - emb) ** 2).sum(dim=1)     # squared distances to memory
        k = min(K, d2.numel())
        d2_k, _ = torch.topk(d2, k, largest=False, sorted=True)
        d2_k = d2_k.cpu().numpy().astype(np.float64)
        self.d2_m[env_idx] = 0.99 * self.d2_m[env_idx] + 0.01 * d2_k[-1]  # k-th NN distance
        d_n = d2_k / max(self.d2_m[env_idx], 1e-12)        # normalize to typical-neighbor units
        d_n = np.maximum(d_n - XI, 0.0)                    # cluster: tiny distances -> 0
        k_v = EPS / (d_n + EPS)                            # inverse kernel
        s = np.sqrt(k_v.sum()) + C                         # sqrt(sum_{N_k} K) + c
        if s > S_M:                                        # saturated this episode -> no bonus
            return 0.0
        return float(1.0 / s)                              # 1/sqrt(n)


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

So the delta from step 3: keep RND verbatim as the lifelong factor, add a within-episode kNN pseudo-count over the inverse-dynamics controllable-state memory (reusing step 2's encoder idea), combine them *multiplicatively with a floor at 1* so the episodic drive can never be killed, and train the UVFA family so exploitation stays clean despite a bonus that deliberately never vanishes. Reading RND's shape, the falsifiable claims are on the *worst* seed and on `auc`, not the headline eval: if the episodic reset is the right fix, Private Eye's `nonzero_rate` should move off 0.667 toward 1.0 (seed 42 stops finding nothing) and its `auc` should lift from −84 toward positive (less time in the red) *before* the peak eval moves — and Tutankham (~100) and Frostbite (~250) should hold, since the $\beta_0=0$ exploit head exists precisely so a non-vanishing bonus doesn't drag exploitation. This task's leaderboard has no NGU row to confirm it; what the trajectory ends on is the construction whose two timescales make persistent re-traversal the default rather than a tuned accident.

The causal chain in one breath: RND broke Private Eye open (one seed at 2756) but its purely *lifelong*, monotonically-decaying novelty can't re-pay the traversal of cleared rooms, so it's fragile — negative auc, a dead seed → split novelty into two timescales: a reset-able *within-episode* kNN pseudo-count $1/(\sqrt{\sum_{N_k}K}+c)$ over an episodic memory, with the scale-free inverse kernel ($d_m^2$ normalization), cluster floor $\xi$, and saturation cap $s_m$ → store *controllable-state* embeddings (reuse ICM's inverse-dynamics encoder) so uncontrollable flicker can't manufacture novelty → modulate by RND's lifelong error through $\mathrm{clip}(\alpha,1,L)$ *multiplicatively*, so global novelty amplifies but never kills the episodic drive ($i_t=r^{\text{episodic}}_t\,\mathrm{clip}(\alpha_t,1,L)$, reducing to pure episodic novelty in the limit) → since that drive never vanishes, keep exploitation clean with the UVFA family $Q(x,a,\beta_i)$ from $\beta_0=0$ to $0.3$, discounts $0.997\!\to\!0.99$, feeding $i_t$ and $\beta_i$ into the recurrent learner — the never-give-up structure RND's numbers said was missing.
