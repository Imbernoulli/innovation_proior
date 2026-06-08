# Step 3 — RND (random network distillation), distilled

**Problem (from step 2).** ICM stabilized the reachable games but its forward error *decays locally* —
it died before crossing Private Eye's long gap (flat zero) — and its three coupled networks made
Frostbite a one-seed jackpot. Wanted: a *global*, slowly-decaying, far *simpler* novelty signal.

**Key idea.** Pick a prediction target that is *deterministic* and *inside the predictor's model class*,
so the only thing that keeps its error high is too little nearby data — pure novelty. Freeze a random
target network $f$; train a predictor $\hat f$ to distill it on visited observations; the bonus is the
leftover error $i_t=\tfrac12\lVert\hat f(s_{t+1})-f(s_{t+1})\rVert^2$. Make the predictor *deeper*
than the target (the error is an uncertainty estimate, à la randomized priors).

**Why it works.** Deterministic in-class target ⇒ no aleatoric/misspecification error; SGD only fits
the target where it has seen data, so the error stays high on globally-novel states and decays slowly
as a region is seen across training — exactly what Private Eye's persistence needs, with far less
coupled machinery to swing seed-to-seed.

**Scaffold edit / hyperparameters.** Predictor and frozen target are Atari conv stacks → 512-d; only
the predictor is trainable. Whiten the observation into both (a frozen random net can't adapt to input
scale), normalize the bonus by a running intrinsic-return std, train the predictor on a random
`update_proportion` of the batch. The two-value-head loop and `mix_advantages` are unchanged from ICM
— only the bonus differs.

**What to watch.** Private Eye is the test: a non-decaying global bonus should get at least one seed
across the gap; Tutankham/Frostbite should hold, trading ICM's variance for stability.

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
