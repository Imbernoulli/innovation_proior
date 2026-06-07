Let me start from the failure that survives every cure so far. On the genuinely hard games — Montezuma's Revenge, Pitfall!, Private Eye — the extrinsic reward is so sparse that a plain agent never sees one, and the standard fix is a novelty bonus added to the reward, $r_t=e_t+\beta i_t$. Pathak et al. (2017) make $i_t$ the forward-prediction error in an inverse-dynamics feature space; Burda et al. (2018) make it the distillation error against a frozen random network; Bellemare et al. (2016) make it a density-model pseudo-count. They differ in how they measure novelty, but they share one property, and it's the property that's biting me: the bonus *vanishes*. As the agent gathers experience near a region, the predictor learns it, the count grows, and $i_t$ on that region decays toward zero. That is exactly right for pushing the frontier outward *once*. It is exactly wrong for what these games actually require, which is to keep walking *back through* already-seen rooms, episode after episode, because the only path to the next undiscovered room runs through the old ones. Once the bonus on the early rooms has decayed, nothing pulls the agent through them anymore. The exploration signal is a one-shot frontier-pusher, and these games need a *persistent* one.

A natural patch is to *tune the decay*: slow the bonus down so it doesn't die before learning catches up (Ostrovski et al. 2017; Ecoffet et al. 2019). But that's a calibration knife-edge — too slow and the agent wastes its life re-exploring solved regions, too fast and it gives up before reaching the frontier — and it depends on the game. I don't want to balance a decay rate. I want a bonus whose structure makes persistent exploration the default.

So let me ask what "persistent exploration" even means, mechanically. Watch a good human explorer: within a single attempt they avoid backtracking onto squares they've already stood on this run; but they happily re-enter, on a fresh attempt, a room they cleared a hundred attempts ago, because clearing it again is the price of getting deeper. There are two completely different timescales here. *Within* an episode, novelty should be sharp and reset-able: a state I've visited this episode is stale, a state I haven't is fresh, and that judgment should snap back to "everything is fresh" the instant a new episode starts. *Across* episodes, novelty should be slow: a room I've now cleared ten thousand times across all of training really is mastered and shouldn't keep paying a frontier-sized bonus forever. RND-style distillation error is purely the second kind — it's a global quantity that only ever decreases over training and has no notion of "this episode." A density-model count is also the second kind. None of them has the first kind at all. That's the gap. I need a *within-episode* novelty that resets every episode, and I need to keep a *lifelong* novelty too, and I need to combine them so the agent never gives up re-exploring.

Take the within-episode part first, because it's the new thing. I want, at each step, a number that says "how much of *this episode's* trajectory have I already spent near the current state." The cleanest version of that is a count: how many times have I been at (something like) this state since the episode began? In a tabular world I'd keep a per-episode visitation count $n(s)$ and, following the count-based exploration tradition (Strehl & Littman 2008, where $1/\sqrt{n}$ has actual regret guarantees), set the episodic bonus to
$$r^{\text{episodic}}_t=\frac{1}{\sqrt{n(s_t)}}.$$
At the start of the episode $n=0$ for everything, the whole world looks fresh; as the agent re-treads ground this episode the relevant counts climb and those squares stop paying; next episode the counts are wiped and the world is fresh again. That is precisely the reset-able sharp novelty I described, and it's a count so it's principled. The problem, of course, is that these are image states and I will essentially never stand on the *exact* same pixel-state twice — so a literal count is always 1 and the bonus is constant. I have to generalize the count: states that are *close* should share count mass.

That's a pseudo-count, and the substrate for it is an episodic memory. Keep a slot-based memory $M$ that, at time $t$, holds the embeddings of every state seen *so far this episode*, $M=\{f(x_0),\dots,f(x_{t-1})\}$, and is emptied at every episode boundary. To turn "have I been near here this episode" into a count, read the memory the way differentiable episodic memories are read (Blundell et al. 2016; Pritzel et al. 2017, Neural Episodic Control): find the $k$ nearest neighbors $N_k=\{f_i\}$ of the current embedding $f(x_t)$ and sum a smooth kernel similarity over them, treating that sum as the pseudo-count,
$$n(f(x_t))\approx\sum_{f_i\in N_k}K\big(f(x_t),f_i\big),\qquad r^{\text{episodic}}_t=\frac{1}{\sqrt{\sum_{f_i\in N_k}K(f(x_t),f_i)}+c}.$$
The constant $c$ (small, say $10^{-3}$) is a floor on the denominator; on the first memory-empty step the kernel sum is zero, so the bonus is finite rather than undefined. If the kernel were a Dirac delta the sum would recover exact counts but generalize nothing; I want a kernel that gives partial count mass to *nearby* states, which is what makes this work in a space where exact repeats never happen.

Which kernel? I want similarity near 1 when two embeddings nearly coincide and decaying as they separate, and I want it *scale-free* so it behaves the same whether a particular game's embeddings happen to live at distance 0.1 or distance 100 from each other. The inverse kernel from the episodic-memory line fits both:
$$K(x,y)=\frac{\epsilon}{\dfrac{d^2(x,y)}{d_m^2}+\epsilon},$$
with $d$ the Euclidean distance, $\epsilon$ a small constant, and — this is the scale-free part — $d_m^2$ a *running average of the squared distance to the $k$-th nearest neighbor*. Dividing by $d_m^2$ normalizes distances into "typical-neighbor units," so the kernel doesn't need re-tuning per game as the embedding magnitudes drift; when $d^2\ll d_m^2$ the kernel is $\approx1$ (same cluster, full count), when $d^2\gg d_m^2$ it's $\approx0$ (different state, no count). Good.

Now I have to be careful about two pathologies of this episodic count, both of which an adversarial agent would otherwise farm. First, near-duplicate frames. In Atari, consecutive frames are almost identical; if I let two essentially-identical embeddings each contribute a fat kernel similarity, then standing still or oscillating between two adjacent positions would pile up count mass *from the agent's own immediate past* and crush the bonus, when really the agent hasn't gone anywhere new — or, flipped around, tiny jitter would register as "new" and pay a bonus for not moving. I want the count to ignore distances that are so small they represent the same place. So after normalizing the neighbor distances by $d_m^2$, *cluster* them: subtract a small floor $\xi$ and clamp at zero,
$$d_n\leftarrow\max\!\big(d_n-\xi,\;0\big),$$
so any neighbor closer than $\xi$ (in typical-neighbor units) snaps to distance 0 and contributes the *maximal* kernel value — it's counted as the same state, not as fresh novelty. Second, saturation. If the current state is genuinely one the agent has stood at many times this episode, the summed similarity $s$ gets large and $1/s$ gets small, which is correct — but I'll also impose a hard ceiling: if $s$ exceeds a maximum similarity $s_m$, set the episodic bonus to exactly zero. Past a point, "very visited this episode" should mean *no* bonus, not a tiny positive one the agent could still chase by camping.

Stepping back: what is $f$, the embedding the memory stores and the kernel compares? This is where the noisy-TV lesson from prediction-error curiosity comes straight back in. If $f$ is raw pixels, or any representation that encodes uncontrollable variation, then the episodic count is wrecked by exactly the thing that wrecks forward-prediction curiosity. Picture an environment full of agent-independent motion — pedestrians, traffic, or the gridworld where the wall colors are re-randomized every single step. The pixel-state changes every step no matter what the agent does, so *every* observation is far from everything in the memory, the episodic count stays at its floor forever, and the agent collects a fat "novelty" bonus for standing perfectly still while the world flickers around it. That is not exploration. I need $f$ to be a *controllable state*: it must represent what the agent's actions can change and be blind to what they can't. The construction for that already exists — the inverse-dynamics embedding (Pathak et al. 2017): train $f$ together with a small classifier $h$ so that from two consecutive observations it recovers the action taken, $p(a\mid x_t,x_{t+1})=h(f(x_t),f(x_{t+1}))$, by maximum likelihood. To predict which action was taken, $f$ *must* encode whatever moved because of the action and has no reason to encode the wall colors, which carry no information about the action; and it can't collapse to a constant, because then $h$ couldn't recover the action at all. So I borrow only the inverse model — the *representation* learner — and throw away the forward-model-error reward; here the forward error isn't the signal, the episodic kNN count is. With this $f$, two states that differ only in uncontrollable flicker map to nearby embeddings, the kernel calls them the same, the count rises, and the bonus correctly says "you haven't gone anywhere." In the disco maze, the signal I want is position coverage within the current episode; a random projection or a raw RND error has no reason to separate that from the color churn, while inverse dynamics puts pressure on exactly the action-relevant part.

That's the episodic half. Now bring back the lifelong half, because the episodic count alone has a blind spot of its own. Resetting every episode means a state that's been visited ten thousand times across all of training looks *just as fresh* at the start of a new episode as a state never seen — the episodic count, by construction, has no memory across episodes. Early in training that's what I want (re-explore everything). Late in training I still want the reset-able drive, but I don't want globally familiar rooms to receive the same extra emphasis as globally unfamiliar ones. So I want a slow, global novelty signal layered on top that *removes extra amplification* from regions seen across many episodes — and that is exactly what RND-style distillation error is good at (Burda et al. 2018): a frozen random target $g$ and a predictor $\hat g$ trained on all observations, with error $\mathrm{err}(x_t)=\|\hat g(x_t)-g(x_t)\|^2$ that is high on globally-unfamiliar states and decreases slowly, by gradient descent, as a region is seen across training. It's cheap and parallelizes trivially. So I'll use it not as a standalone bonus but as a *modulator*. Normalize the error to a unitless factor that hovers around 1,
$$\alpha_t=1+\frac{\mathrm{err}(x_t)-\mu_e}{\sigma_e},$$
with $\mu_e,\sigma_e$ a running mean and std of the error — so $\alpha_t\approx1$ for typically-familiar states, $\alpha_t>1$ for globally-novel ones.

How do I combine the two? My instinct is to add them, $i_t=r^{\text{episodic}}_t+\alpha_t$, but that's wrong here, and seeing why is the crux. If I add, then once a region is globally mastered ($\alpha_t\to$ its floor) the lifelong term still contributes a constant baseline, and worse, the additive structure lets the lifelong term *carry* the bonus even where the episodic term has correctly gone to zero — so the agent can still be paid for camping in a globally-novel-but-locally-stale spot, which reintroduces the camping pathology I just removed. What I actually want is for the episodic count to be the *driver* of exploration — it's the one with the reset that gives persistence — and for the lifelong signal to only *scale* that drive: amplify it where the region is globally new, leave it alone where the region is globally familiar, and never, ever let it kill the episodic drive entirely. That's a *multiplicative* modulation with a floor at 1:
$$i_t=r^{\text{episodic}}_t\cdot\min\!\Big\{\max\{\alpha_t,1\},\,L\Big\}.$$
Floor the modulator at 1: the lifelong factor can only *boost* the episodic bonus, never shrink it below the episodic value, so even after global novelty has completely vanished ($\alpha_t$ would dip below 1) the agent still gets the full within-episode signal and keeps re-exploring — it never gives up. Cap it at $L$ (I'll take $L=5$): a single anomalous RND spike — a one-off weird frame — shouldn't be able to multiply the bonus by a huge factor and yank the policy toward a fluke. And notice the limiting behavior is exactly right: as the agent masters everything globally, $\alpha_t$ saturates at the floor, the modulator $\to1$, and $i_t\to r^{\text{episodic}}_t$ — the method gracefully reduces to pure episodic novelty, which is the part that should never decay. Where the episodic bonus has *itself* gone to zero (state already saturated this episode), the product is zero regardless of $\alpha_t$, so the camping exploit stays closed.

So the bonus is settled: a within-episode kNN pseudo-count over an inverse-dynamics controllable-state memory, multiplicatively modulated by a clipped, normalized RND lifelong-novelty factor. But there's a second problem this whole construction creates, separate from *computing* the bonus, and I have to handle it or the gains evaporate on the easy games. By design this bonus *does not vanish* — the episodic term keeps paying out forever. That's the feature. But it means a value function trained on $e_t+\beta i_t$ has the exploratory drive *permanently* baked in; the policy will keep sacrificing extrinsic return to go sightseeing even on a dense-reward game where there's nothing to explore and it should just exploit. A vanishing bonus would have quietly switched itself off; mine won't. I can't fix this by annealing $\beta$ — that would throw away the persistence I worked for. I need an architecture that holds an exploratory policy and an exploitative policy *at the same time* and lets me act with either.

I can get that separation if the value function is not one object but a *family*, indexed by the intrinsic weight itself. Use a universal value function approximator (Schaul et al. 2015): one network $Q(x,a,\beta_i)$ that conditions on $\beta_i$ and approximates the optimal value for the augmented reward $r^{\beta_i}_t=e_t+\beta_i i_t$, for a discrete set $\{\beta_i\}_{i=0}^{N-1}$. Pin the endpoints: $\beta_0=0$ is a *pure exploitative* value function (it never sees the intrinsic reward at all), and $\beta_{N-1}=\beta$ is the most exploratory. Now exploitation is free: at evaluation, act greedily with respect to $Q(x,a,0)$ and the exploratory bias is simply *gone* — no annealing, no switch to flip, it was a separate member of the family all along. And because all the $\beta_i$ share the network's weights, the strongly-exploratory members act as auxiliary tasks (Jaderberg et al. 2016) that keep the shared representation and skills improving *even before any extrinsic reward is ever seen*, which the $\beta_0$ head can then exploit the instant a reward appears. Why a whole spread of $N$ values and not just $\{0,\beta\}$? Because the pure-exploit and pure-explore policies can be very different in behavior, and a network asked to represent both extremes and nothing in between has to make a hard jump; a smoothly-spaced ladder of intermediate trade-offs makes the family easier to fit. I'll pack the spacing toward the two extremes (a sigmoid schedule in $i$) since those are the members I care most about, and pair each $\beta_i$ with its own discount $\gamma_i$: the exploitative end wants the *largest* discount, $\gamma_0=0.997$, to be as close to optimizing undiscounted extrinsic return as possible (rewards are far apart), while the exploratory end can use a *smaller* discount, $\gamma_{N-1}=0.99$, because the intrinsic reward is dense and small-ranged so a short horizon suffices — log-spaced in between.

One last subtlety from feeding a non-stationary reward into a value-based agent. The intrinsic reward depends on the episodic memory's contents, which depend on the whole within-episode history — so from the perspective of a memoryless state the reward looks unpredictable, and adding it naively turns the MDP into a POMDP. Two things keep it Markov-from-the-agent's-view: use a recurrent agent that summarizes the within-episode history in its state, and feed the intrinsic reward (and $\beta_i$, the previous action, the previous extrinsic reward) *as inputs* to the network so the value function can see the very quantity that shifted the reward. The natural base is a distributed recurrent replay value-learner (Kapturowski et al. 2019, R2D2): LSTM state, prioritized replay, off-policy Retrace value learning, many parallel actors — and the bonus is cheap and parallel by construction, so it rides that scale. The embedding network and the RND predictor are trained on a handful of frames from each sampled replay sequence; the RL loss uses all timesteps.

I have to make the construction concrete at the bonus-module boundary, because the policy loop is fixed: the module computes a per-transition bonus, normalizes the rollout's intrinsic stream, and exposes its own training loss; a separate mixing function combines the extrinsic and intrinsic advantage streams. The episodic-reward routine below is exactly the kNN-count-with-clustering procedure I derived; the controllable-state embedding is the inverse-dynamics model; the lifelong factor is normalized clipped RND; the bonus is their product.

```python
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# provided by the fixed loop: layer_init, RunningMeanStd, RewardForwardFilter, last_frame, Args

P = 32      # controllable-state (embedding) dimension
K = 10      # nearest neighbors for the episodic pseudo-count
EPS = 1e-3  # kernel epsilon
XI = 0.008  # cluster floor: distances below this (in typical-neighbor units) -> same state
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
    """Per-environment within-episode memory of controllable-state embeddings, reset each episode.
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
        d2_k, _ = torch.topk(d2, k, largest=False, sorted=True)  # k nearest distances
        d2_k = d2_k.cpu().numpy().astype(np.float64)
        self.d2_m[env_idx] = 0.99 * self.d2_m[env_idx] + 0.01 * d2_k[-1]  # k-th NN distance
        d_n = d2_k / max(self.d2_m[env_idx], 1e-12)        # normalize to typical-neighbor units
        d_n = np.maximum(d_n - XI, 0.0)                    # cluster: tiny distances -> 0 (same state)
        k_v = EPS / (d_n + EPS)                            # inverse kernel
        s = np.sqrt(k_v.sum()) + C                         # sqrt(sum_{N_k} K) + c
        if s > S_M:                                        # saturated this episode -> no bonus
            return 0.0
        return float(1.0 / s)                              # 1/sqrt(n): episodic novelty


class IntrinsicBonusModule(nn.Module):
    """Episodic kNN-count novelty over an inverse-dynamics controllable state,
    multiplicatively modulated by a normalized, clipped RND lifelong-novelty factor."""

    def __init__(self, action_dim: int, device: torch.device, args: "Args"):
        super().__init__()
        self.action_dim = action_dim
        self.device = device
        self.args = args
        self.obs_rms = RunningMeanStd(shape=(1, 1, 84, 84))
        self.reward_rms = RunningMeanStd()
        self.discounted_reward = RewardForwardFilter(args.int_gamma)
        self.err_rms = RunningMeanStd()                   # mu_e, sigma_e for the RND modulator
        self.memory = EpisodicMemory(args.num_envs, device)

        feature_output = 7 * 7 * 64
        # controllable-state embedding f, trained by inverse dynamics (only action-relevant content)
        self.encoder = nn.Sequential(
            layer_init(nn.Conv2d(1, 32, 8, stride=4)), nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)), nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)), nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(feature_output, P)),     # f(x) in R^P
        )
        # inverse model h: predict a_t from (f(x_t), f(x_{t+1})) -> anchors f
        self.inverse_model = nn.Sequential(
            layer_init(nn.Linear(2 * P, 256)), nn.ReLU(),
            layer_init(nn.Linear(256, action_dim), std=0.01),
        )
        # lifelong RND: frozen random target g + trained predictor g_hat
        self.predictor = nn.Sequential(
            layer_init(nn.Conv2d(1, 32, 8, stride=4)), nn.LeakyReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)), nn.LeakyReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)), nn.LeakyReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(feature_output, 512)), nn.ReLU(),
            layer_init(nn.Linear(512, 512)), nn.ReLU(),
            layer_init(nn.Linear(512, 512)),
        )
        self.target = nn.Sequential(
            layer_init(nn.Conv2d(1, 32, 8, stride=4)), nn.LeakyReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)), nn.LeakyReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)), nn.LeakyReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(feature_output, 512)),
        )
        for p in self.target.parameters():
            p.requires_grad = False                       # target frozen at random init

    def initialize(self, envs) -> None:
        # warm the observation-normalization stats with a random rollout (RND target can't adapt scale)
        bootstrap = []
        total = self.args.num_steps * self.args.num_iterations_obs_norm_init
        for _ in range(total):
            a = np.random.randint(0, envs.single_action_space.n, size=(self.args.num_envs,))
            sampled_obs = envs.step(a)[0]
            bootstrap.append(sampled_obs[:, 3:4, :, :])
            if len(bootstrap) >= self.args.num_steps:
                self.obs_rms.update(np.concatenate(bootstrap, axis=0)); bootstrap.clear()

    def trainable_parameters(self):
        return list(self.encoder.parameters()) + list(self.inverse_model.parameters()) \
            + list(self.predictor.parameters())           # target excluded (frozen)

    def _normalize_obs(self, obs):                        # for predictor/target (NOT the policy)
        mean = torch.from_numpy(self.obs_rms.mean).to(self.device)
        var = torch.from_numpy(self.obs_rms.var).to(self.device)
        return ((last_frame(obs) - mean) / torch.sqrt(var)).clip(-5, 5).float()

    def update_batch_stats(self, batch_obs, batch_next_obs) -> None:
        self.obs_rms.update(last_frame(batch_next_obs).cpu().numpy())

    @torch.no_grad()
    def compute_bonus(self, obs, next_obs, actions) -> torch.Tensor:
        # episodic novelty: kNN pseudo-count over the controllable-state memory (per env)
        emb = self.encoder(last_frame(next_obs).float())  # f(x_t): controllable state
        r_epi = torch.tensor(
            [self.memory.episodic_reward(i, emb[i]) for i in range(emb.shape[0])],
            device=self.device, dtype=torch.float32)
        # lifelong RND modulator: alpha = 1 + (err - mu_e)/sigma_e, clipped to [1, L]
        norm_next = self._normalize_obs(next_obs)
        err = (self.predictor(norm_next) - self.target(norm_next)).pow(2).sum(1)
        self.err_rms.update(err.cpu().numpy())
        sigma = float(np.sqrt(self.err_rms.var + 1e-8))
        alpha = 1.0 + (err - float(self.err_rms.mean)) / sigma
        alpha = alpha.clamp(min=1.0, max=L)
        return (r_epi * alpha).detach()                   # i_t = r_episodic * clip(alpha, 1, L)

    def reset_memory(self, env_idx) -> None:
        self.memory.reset(env_idx)                        # call on each episode boundary

    def normalize_rollout_rewards(self, rollout_intrinsic) -> torch.Tensor:
        discounted = np.stack(
            [self.discounted_reward.update(r) for r in rollout_intrinsic.cpu().numpy()], axis=0)
        flat = discounted.reshape(-1)
        self.reward_rms.update_from_moments(float(flat.mean()), float(flat.var()), int(flat.size))
        return rollout_intrinsic / float(np.sqrt(self.reward_rms.var + 1e-8))

    def loss(self, batch_obs, batch_next_obs, batch_actions) -> torch.Tensor:
        # inverse-dynamics loss: train f so (f(x_t), f(x_{t+1})) predicts a_t  (controllable state)
        f_t = self.encoder(last_frame(batch_obs).float())
        f_tp1 = self.encoder(last_frame(batch_next_obs).float())
        logits = self.inverse_model(torch.cat([f_t, f_tp1], dim=1))
        inverse_loss = F.cross_entropy(logits, batch_actions.long())
        # RND distillation loss: predictor toward frozen target (lifelong novelty)
        norm_next = self._normalize_obs(batch_next_obs)
        rnd_loss = F.mse_loss(self.predictor(norm_next),
                              self.target(norm_next).detach(), reduction="none").sum(-1)
        mask = (torch.rand(len(rnd_loss), device=self.device) < self.args.update_proportion).float()
        rnd_loss = (rnd_loss * mask).sum() / torch.clamp(mask.sum(), min=1.0)
        return inverse_loss + rnd_loss


def mix_advantages(ext_advantages, int_advantages, args: "Args") -> torch.Tensor:
    # one member of the family Q(x,a,beta_i): r^{beta_i} = e + beta_i * i
    beta_i = getattr(args, "beta_i", getattr(args, "int_coef", BETA_MAX))
    return getattr(args, "ext_coef", 1.0) * ext_advantages + beta_i * int_advantages
```

I end up with the pressure I wanted. A vanishing bonus pushes the frontier once but does not keep walking the agent back through cleared rooms, so I split novelty into a reset-able within-episode count and a slow lifelong modulator. The within-episode term is $1/(\sqrt{\sum_{N_k}K}+c)$ over an episodic memory, with the inverse kernel normalized by the running $k$-th-nearest-neighbor distance, the cluster floor $\xi$ for near-duplicates, and the saturation cap $s_m$ for states already saturated this episode. The embedding is an inverse-dynamics controllable state, so uncontrollable flicker cannot manufacture novelty. RND supplies $\alpha_t=1+(\mathrm{err}-\mu_e)/\sigma_e$, but only through $\mathrm{clip}(\alpha_t,1,L)$, so it can amplify globally novel regions and cannot kill the episodic drive; the product is $i_t=r^{\text{episodic}}_t\mathrm{clip}(\alpha_t,1,L)$. Since that drive deliberately persists, I keep exploitation clean by training the UVFA family $Q(x,a,\beta_i)$ from $\beta_0=0$ to $\beta=0.3$, with discounts from $\gamma_{\max}=0.997$ down to $\gamma_{\min}=0.99$, and I feed the intrinsic reward and $\beta_i$ into the recurrent value learner so the augmented reward remains visible to the agent.
