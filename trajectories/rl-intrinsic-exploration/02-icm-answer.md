**Problem (from step 1).** With no bonus, sparse-reward games where the first reward is out of reach of
random noise barely move. A reward must be manufactured from the agent's own experience — but
pixel-prediction-error curiosity is captured forever by uncontrollable flicker (the noisy-TV trap).

**Key idea.** Reward forward-prediction error in a *learned feature space* $\phi(s)$ that keeps only
action-relevant content. Anchor $\phi$ with an *inverse-dynamics* task (recover $a_t$ from
$\phi(s_t),\phi(s_{t+1})$), which forces $\phi$ to encode what the action changes, ignore what it
can't, and not collapse. The bonus is the forward error in that space,
$r^i_t=\tfrac12\lVert\hat\phi(s_{t+1})-\phi(s_{t+1})\rVert^2$, detached; the module trains on
$L_I+0.2\,L_F$ (inverse dominates so the forward task can't collapse $\phi$).

**Why it works.** $\phi$ has no incentive to encode irrelevant variation, so noise produces no bonus;
the reward is high on unmastered controllable transitions and decays as they're learned, pushing the
agent onward.

**Scaffold edit / hyperparameters.** Encoder = the Atari conv stack on `last_frame`, flattened to a
256-d $\phi$; inverse = Linear(512,256)→Linear(256,#act) (cross-entropy); forward =
Linear(256+#act,256)→Linear(256,256). Whiten the frame (running mean/std, clip $\pm5$), normalize the
rollout's intrinsic stream by a running return-std; `mix_advantages` turns the intrinsic stream back on
($\texttt{ext\_coef}\,A_E+\texttt{int\_coef}\,A_I$).

**What to watch.** Tutankham should stabilize and Frostbite may jump; Private Eye is the risk, because
the forward error decays as local dynamics are mastered and may die before the long gap is crossed.

```python
class IntrinsicBonusModule(nn.Module):
    """Intrinsic Curiosity Module baseline."""

    def __init__(self, action_dim: int, device: torch.device, args: Args):
        super().__init__()
        self.action_dim = action_dim
        self.device = device
        self.args = args
        self.obs_rms = RunningMeanStd(shape=(1, 1, 84, 84))
        self.reward_rms = RunningMeanStd()
        self.discounted_reward = RewardForwardFilter(args.int_gamma)

        feature_output = 7 * 7 * 64
        self.encoder = nn.Sequential(
            layer_init(nn.Conv2d(1, 32, 8, stride=4)), nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)), nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)), nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(feature_output, 256)), nn.ReLU(),
        )
        self.inverse_model = nn.Sequential(                      # [phi(s_t), phi(s_{t+1})] -> action
            layer_init(nn.Linear(512, 256)), nn.ReLU(),
            layer_init(nn.Linear(256, action_dim), std=0.01),
        )
        self.forward_model = nn.Sequential(                      # [phi(s_t), a_t] -> phi_hat(s_{t+1})
            layer_init(nn.Linear(256 + action_dim, 256)), nn.ReLU(),
            layer_init(nn.Linear(256, 256)),
        )

    def initialize(self, envs) -> None:
        bootstrap = []                                           # warm obs-norm with a random rollout
        total_steps = self.args.num_steps * self.args.num_iterations_obs_norm_init
        for _ in range(total_steps):
            random_actions = np.random.randint(0, envs.single_action_space.n, size=(self.args.num_envs,))
            sampled_obs, _, _, _ = envs.step(random_actions)
            bootstrap.append(sampled_obs[:, 3:4, :, :])
            if len(bootstrap) >= self.args.num_steps:
                self.obs_rms.update(np.concatenate(bootstrap, axis=0)); bootstrap.clear()

    def trainable_parameters(self):
        return list(self.parameters())

    def _normalize_obs(self, obs: torch.Tensor) -> torch.Tensor:
        mean = torch.from_numpy(self.obs_rms.mean).to(self.device)
        var = torch.from_numpy(self.obs_rms.var).to(self.device)
        return ((last_frame(obs) - mean) / torch.sqrt(var)).clip(-5, 5).float()

    def _one_hot(self, actions: torch.Tensor) -> torch.Tensor:
        return F.one_hot(actions.long(), num_classes=self.action_dim).float()

    def update_batch_stats(self, batch_obs, batch_next_obs) -> None:
        self.obs_rms.update(last_frame(batch_next_obs).cpu().numpy())

    def compute_bonus(self, obs, next_obs, actions) -> torch.Tensor:
        obs_feat = self.encoder(self._normalize_obs(obs))
        next_feat = self.encoder(self._normalize_obs(next_obs))
        pred_next_feat = self.forward_model(torch.cat([obs_feat, self._one_hot(actions)], dim=1))
        return 0.5 * (pred_next_feat - next_feat).pow(2).mean(dim=1).detach()      # r^i, a signal

    def normalize_rollout_rewards(self, rollout_intrinsic) -> torch.Tensor:
        discounted = np.stack(
            [self.discounted_reward.update(r) for r in rollout_intrinsic.cpu().numpy()], axis=0)
        flat = discounted.reshape(-1)
        self.reward_rms.update_from_moments(float(flat.mean()), float(flat.var()), int(flat.size))
        return rollout_intrinsic / float(np.sqrt(self.reward_rms.var + 1e-8))

    def loss(self, batch_obs, batch_next_obs, batch_actions) -> torch.Tensor:
        obs_feat = self.encoder(self._normalize_obs(batch_obs))
        next_feat = self.encoder(self._normalize_obs(batch_next_obs))
        pred_next_feat = self.forward_model(torch.cat([obs_feat, self._one_hot(batch_actions)], dim=1))
        pred_action = self.inverse_model(torch.cat([obs_feat, next_feat], dim=1))
        inverse_loss = F.cross_entropy(pred_action, batch_actions.long())          # anchors phi
        forward_loss = 0.5 * (pred_next_feat - next_feat.detach()).pow(2).mean()
        return inverse_loss + 0.2 * forward_loss                                   # inverse dominates


def mix_advantages(ext_advantages, int_advantages, args: Args) -> torch.Tensor:
    return args.ext_coef * ext_advantages + args.int_coef * int_advantages
```
