ICM's numbers point straight at its successor. Curiosity did exactly what I hoped on two games: Tutankham went from PPO's two dead seeds to all three at $\sim$106–113 (mean 109, every seed nonzero), the variance collapsed, and exploration stopped being a coin flip. Frostbite's mean jumped to 1156, but the per-seed eval $\{232, \mathbf{3066}, 170\}$ is a jackpot, not a dependable lift — one seed found a deep policy and the other two landed near where PPO already was. And on Private Eye, the game I flagged, ICM scored a flat 0.0 on *every* seed, `nonzero_rate` 0.0, never one reward. One good thing did happen there: the `auc` is now $\sim-29$ and *bounded*, where PPO had a $-535$ seed, so curiosity stopped the destructive wandering into penalties — it explored its controllable surroundings rather than blunder — but it could not cross the gap.

The reason is the one I suspected when I built ICM: the forward-prediction error is large only while the controllable dynamics near the agent are unmastered, and it *decays toward zero* the moment the forward model has learned those local transitions. On Private Eye the starting dynamics are quickly learnable, so the bonus on that region collapses before the agent crosses the long reward-free gap — the drive runs out at the first mastered region. Two things are wrong and they share a root. The novelty signal decays too fast and too *locally* — it tracks "have I learned the dynamics right here," when a long-horizon game needs "is this state still unfamiliar *overall*." And ICM does a lot of work to get there: an encoder, an inverse model, and a forward model, three coupled networks whose interaction is exactly what makes Frostbite swing from 170 to 3066 across seeds. I want a *global* and a *more stable* novelty detector, and a far simpler one so there is less to go unstable.

I stay with prediction error — it is cheap, a single forward pass, and the loop runs many parallel envs — but I am precise about *why* ICM's forward model decays locally. A prediction error can come from four sources: too little training data near this input, which is *epistemic* uncertainty, exactly the novelty I want; a genuinely *stochastic* target, the *aleatoric* noisy-TV source; the model lacking the inputs or capacity to represent the target; and the optimizer failing to fit a representable target. ICM spends its inverse-dynamics machinery beating down the stochastic source — and it works, which is why Tutankham stabilized — but its forward error still decays to zero as soon as the local epistemic gap is resolved, because once the model has the local dynamics there is nothing left to predict wrong. I want a bonus driven by epistemic uncertainty *only*, and one whose epistemic content is about global familiarity, not whether one transition's dynamics are locally solved.

So I propose RND, Random Network Distillation, and the load-bearing move is the choice of prediction problem: pick one whose answer is *deterministic* and *inside the predictor's own model class*. Then the aleatoric and misspecification sources are gone by construction and the only thing that can keep the error high is too little data near this state. The deterministic function I use has nothing to do with dynamics and everything to do with "have I seen states like this": take a second neural network, initialize it randomly, and *freeze* it — the target $f:\mathcal{O}\to\mathbb{R}^k$, a fixed arbitrary embedding. Train a *predictor* $\hat f$ by gradient descent on the agent's observations to mimic it,

$$\min_{\theta_{\hat f}}\;\mathbb{E}_x\big\|\hat f(x;\theta_{\hat f})-f(x)\big\|^2,$$

and let the leftover error be the bonus,

$$i_t=\big\|\hat f(s_{t+1})-f(s_{t+1})\big\|^2.$$

On observations the predictor has trained on (and ones near them) gradient descent has pulled $\hat f$ onto $f$, so the error is small; on *globally novel* observations it has not, so the error is high. The error *is* a global-novelty signal driven only by the amount of relevant data seen — no inverse model, no forward model, no encoder to co-train. That answers both of ICM's problems at once: it is global and slowly-decaying rather than local-and-quick-to-die, and dramatically simpler, so far less to make it seed-fragile the way ICM's three-network coupling was.

The obvious objection is whether a powerful predictor could mimic $f$ *everywhere*, including unseen states, collapsing the error globally — in the limit a perfect mimic exists, since $f$ predicts $f$. The question is whether SGD overgeneralizes like that, and it does not: a predictor only becomes accurate on a region once it has seen enough examples *from that region*, so the error stays high on the genuinely unseen, exactly what a novelty detector needs. There is a clean reading of *what* the error estimates — Osband's randomized-prior view says distilling a fixed random function is one ensemble member fit to the constant-zero target, so the predictor–target MSE estimates the *predictive variance*, an uncertainty estimate. That tells me to give the predictor *extra* trainable layers beyond the target's structure, so it can fit the target on visited data while still failing on the unseen.

Two normalizations make or break this, and the second is specific to a *frozen random* target. First, the intrinsic reward's raw scale drifts — an MSE of arbitrary embeddings that shrinks as learning proceeds — so I divide it by a running std of the intrinsic *returns* (the loop's `RewardForwardFilter` plus a `RunningMeanStd`) to keep the bonus scale consistent. Second, because the target is frozen at random init it *cannot adapt to the input scale*; if observations arrive at an arbitrary scale the random target squashes them into a near-constant output and the error becomes meaningless. So I whiten the *observation* into predictor and target — subtract a running mean, divide by a running std, clip $\pm5$ — initialized from a short random rollout before training; this whitening goes only to predictor/target, while the policy net keeps its own $/255$ scaling. The module exposes the distillation error as the bonus and its MSE as the loss, trained on a random `update_proportion` of the batch so the predictor does not overfit a single rollout. The fixed loop's two value heads and discounts ($0.999$ extrinsic for the far-apart reward, $0.99$ intrinsic for the transient bonus) and the `mix_advantages` line ($\texttt{ext\_coef}\cdot A_E+\texttt{int\_coef}\cdot A_I$) are unchanged from ICM — the bonus is the only thing different, which is the point.

Reading ICM's shape, Private Eye is the game I am watching: the global, slowly-decaying signal should finally give the agent a reason to *keep* moving past the first mastered region, and if even one seed crosses the gap and registers a real Private Eye return, that is the breakthrough ICM could not reach. Tutankham and Frostbite I expect to hold roughly where curiosity put them, perhaps trading ICM's high-variance Frostbite peak for more stability.

```python
class IntrinsicBonusModule(nn.Module):
    """Random Network Distillation intrinsic bonus."""

    def __init__(self, action_dim: int, device: torch.device, args: Args):
        super().__init__()
        self.action_dim = action_dim
        self.device = device
        self.args = args
        self.obs_rms = RunningMeanStd(shape=(1, 1, 84, 84))
        self.reward_rms = RunningMeanStd()
        self.discounted_reward = RewardForwardFilter(args.int_gamma)

        feature_output = 7 * 7 * 64
        self.predictor = nn.Sequential(                          # trained; DEEPER than the target
            layer_init(nn.Conv2d(1, 32, 8, stride=4)), nn.LeakyReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)), nn.LeakyReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)), nn.LeakyReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(feature_output, 512)), nn.ReLU(),
            layer_init(nn.Linear(512, 512)), nn.ReLU(),
            layer_init(nn.Linear(512, 512)),
        )
        self.target = nn.Sequential(                             # frozen random target
            layer_init(nn.Conv2d(1, 32, 8, stride=4)), nn.LeakyReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)), nn.LeakyReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)), nn.LeakyReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(feature_output, 512)),
        )
        for param in self.target.parameters():
            param.requires_grad = False

    def initialize(self, envs) -> None:
        bootstrap = []                                           # warm obs-norm (frozen target can't adapt scale)
        total_steps = self.args.num_steps * self.args.num_iterations_obs_norm_init
        for _ in range(total_steps):
            random_actions = np.random.randint(0, envs.single_action_space.n, size=(self.args.num_envs,))
            sampled_obs, _, _, _ = envs.step(random_actions)
            bootstrap.append(sampled_obs[:, 3:4, :, :])
            if len(bootstrap) >= self.args.num_steps:
                self.obs_rms.update(np.concatenate(bootstrap, axis=0)); bootstrap.clear()

    def trainable_parameters(self):
        return list(self.predictor.parameters())                # target excluded (frozen)

    def _normalize_obs(self, obs: torch.Tensor) -> torch.Tensor:
        mean = torch.from_numpy(self.obs_rms.mean).to(self.device)
        var = torch.from_numpy(self.obs_rms.var).to(self.device)
        return ((last_frame(obs) - mean) / torch.sqrt(var)).clip(-5, 5).float()

    def update_batch_stats(self, batch_obs, batch_next_obs) -> None:
        self.obs_rms.update(last_frame(batch_next_obs).cpu().numpy())

    def compute_bonus(self, obs, next_obs, actions) -> torch.Tensor:
        norm_next = self._normalize_obs(next_obs)
        target_feature = self.target(norm_next)
        predict_feature = self.predictor(norm_next)
        return ((target_feature - predict_feature).pow(2).sum(1) / 2).detach()     # i_t: global novelty

    def normalize_rollout_rewards(self, rollout_intrinsic) -> torch.Tensor:
        discounted = np.stack(
            [self.discounted_reward.update(r) for r in rollout_intrinsic.cpu().numpy()], axis=0)
        flat = discounted.reshape(-1)
        self.reward_rms.update_from_moments(float(flat.mean()), float(flat.var()), int(flat.size))
        return rollout_intrinsic / float(np.sqrt(self.reward_rms.var + 1e-8))

    def loss(self, batch_obs, batch_next_obs, batch_actions) -> torch.Tensor:
        norm_next = self._normalize_obs(batch_next_obs)
        predict_feature = self.predictor(norm_next)
        target_feature = self.target(norm_next).detach()
        forward_loss = F.mse_loss(predict_feature, target_feature, reduction="none").mean(-1)
        mask = (torch.rand(len(forward_loss), device=self.device) < self.args.update_proportion).float()
        return (forward_loss * mask).sum() / torch.clamp(mask.sum(), min=1.0)      # distill on a subset


def mix_advantages(ext_advantages, int_advantages, args: Args) -> torch.Tensor:
    return args.ext_coef * ext_advantages + args.int_coef * int_advantages
```
