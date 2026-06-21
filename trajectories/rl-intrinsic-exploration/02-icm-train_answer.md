The no-bonus run told me what is missing, and it told me in numbers. On Frostbite all three seeds found reward (mean 215, a tight 96–280) — there the first payoff is reachable by entropy jitter alone, so vanilla PPO is fine. But Tutankham had two of three seeds flatline at *exactly* zero on every metric (the mean of 36.5 is really one lucky seed plus two that never saw a reward), and Private Eye returned $\{0, 100, -1000\}$ for a *negative* mean of $-300$ — the $-1000$ is the tell, the agent wandering into the game's penalties as readily as its rewards because nothing directs it. The failure that bites is sharp and it is a *signal* problem, not a learning-rate or credit-assignment one: where the extrinsic reward is sparse, the policy gradient $\mathbb{E}[\nabla\log\pi\,(\sum_t r_t)]$ is zero almost always, so the agent only learns if it *happens* to reach the goal. I have to manufacture a reward $r^i$ the policy maximizes alongside the mostly-zero extrinsic $r^e$, total $r_t=r_t^i+r_t^e$, large in states the agent has not mastered, so "go somewhere new" becomes something the gradient can ascend.

The natural first instinct is prediction error: build a forward model that predicts the next observation from $(s_t,a_t)$ and reward the agent in proportion to how wrong it is. But a bad bonus is worse than none — PPO's $-1000$ seed already shows the wrong drive is actively harmful — so I check it before committing. Picture an Atari screen with a patch flickering in a way the agent cannot control: a scrolling background, an animated sprite, the boundary churn sticky actions induce. The pixel content there is essentially unpredictable, so pixel-prediction error stays high *forever*, the bonus stays high forever, and the agent is permanently paid to gawk at the flicker. Any inherently unpredictable but inconsequential distractor — a television showing static, shadows from moving objects — becomes an inexhaustible curiosity reward, and novelty-counting has the identical disease since every noisy frame looks novel. On Private Eye that is precisely how you end up camping next to a penalty instead of crossing the gap.

The remedy "reward only what is hard-but-learnable" is the right intuition but has no feasible estimator, so I diagnose *what kind* of variation I want the reward to respond to instead. Everything that can change the agent's observation is one of three things: stuff the agent controls (1), stuff it cannot control but that affects it — a hazard bearing down (2), and stuff out of its control and not affecting it — the flicker (3). Curiosity should care about (1) and (2) and be completely blind to (3). The trouble with pixels is that they mix all three, and (3) dominates the error budget precisely because it is unpredictable. So the real question is not "forward model or not" but "forward model in *what space*": if I had a feature map $\phi(s)$ encoding only action-relevant content and throwing away (3), forward-predicting $\phi(s_{t+1})$ and rewarding *that* error would respond only to what I care about, and the flicker — absent from $\phi$ — could not generate reward. But I cannot anchor $\phi$ with the forward task itself: an encoder trained jointly to make forward prediction easy discovers that the cheapest way to shrink the error is to make $\phi$ *constant*, perfectly predictable and useless. A feature space optimized to be predictable collapses.

So I propose ICM, the Intrinsic Curiosity Module, and the load-bearing idea is the anchor that pins $\phi$. Instead of predicting the future from the action, the *inverse* model predicts the *action* from the present and future: given $\phi(s_t)$ and $\phi(s_{t+1})$, recover $a_t$. This is self-supervised — the tuples $(s_t,a_t,s_{t+1})$ come for free from acting — and it forces $\phi$ exactly the way I want. To recover the action that took $s_t$ to $s_{t+1}$, $\phi$ *must* encode whatever changed because of the action (category 1, and category 2 where it distinguishes actions), but it has no incentive to encode (3) because the flicker does not help predict which action was taken, and it cannot collapse to a constant because that destroys the information the inverse loss needs. The inverse task makes $\phi$ both non-degenerate and noise-blind. Concretely, the inverse model $g$ outputs a softmax over actions, $\hat a_t=g(\phi(s_t),\phi(s_{t+1}))$, trained with cross-entropy (maximum likelihood under a multinomial). Then, *in the feature space $g$ has carved out*, the forward model $f$ predicts the next features from the current features and the action,

$$\hat\phi(s_{t+1})=f\big(\phi(s_t),a_t\big),\qquad L_F=\tfrac12\big\|\hat\phi(s_{t+1})-\phi(s_{t+1})\big\|^2,$$

and the intrinsic reward is precisely that forward error,

$$r_t^i=\tfrac12\big\|\hat\phi(s_{t+1})-\phi(s_{t+1})\big\|^2.$$

This reward is large exactly when the agent meets a transition whose *action-relevant consequence* it cannot yet predict — a genuinely novel piece of controllable dynamics — and it decays as the forward model learns that transition, pushing the agent onward to the next unmastered piece of controllable structure. The flicker contributes nothing because it is not in $\phi$; the noisy-TV cure falls out of the choice of feature space, not from any extra mechanism.

One thing has to be exactly right: the forward error feeds the policy reward, but the policy gradient must *not* push on the ICM's parameters. If it could, the system would cheat — make $\phi$ unpredictable on purpose to inflate the reward. So the bonus is computed under `detach()` (a signal, not a path), the ICM trains *only* on its own losses, and the policy trains *only* to maximize the reward; they share the reward, not the gradient through it. The module loss is $L_I+0.2\,L_F$, with the inverse loss dominating *on purpose* — $\phi$ must be shaped by the inverse task and never by the forward task that could collapse it.

Fitting this into the task's fixed edit surface, the loop already values two reward streams (extrinsic $\gamma=0.999$, intrinsic `int_gamma`$=0.99$) with per-stream advantages, so my job is only the bonus. The encoder is the standard Atari conv stack on the single most recent frame (via `last_frame`), flattened to a 256-d $\phi$; the inverse model is $\text{Linear}(512,256)\to\text{Linear}(256,\#\text{act})$ on $[\phi(s_t),\phi(s_{t+1})]$ with cross-entropy; the forward model is $\text{Linear}(256+\#\text{act},256)\to\text{Linear}(256,256)$ on $[\phi(s_t),\text{onehot}(a_t)]$. I whiten the frame into the encoder with a running mean/std clipped at $\pm5$, initialized from a short random rollout, and normalize the rollout's intrinsic stream by a running return-std so the bonus scale stays consistent. The one line `mix_advantages` controls turns the intrinsic stream back on: $A=\texttt{ext\_coef}\cdot A_E+\texttt{int\_coef}\cdot A_I$.

Reading PPO's shape, I expect Tutankham to *stabilize* — every seed should at least find the reward now that "explore the controllable world" is rewarded, so the two dead seeds come alive — and Frostbite may jump as curiosity keeps pushing past the first ice-floe reward. Private Eye is the open question, and I can feel the risk in the construction itself: the bonus is a forward error that *decays as the forward model masters local dynamics*, and Private Eye's payoff sits past a long stretch whose local dynamics are quickly learned, so curiosity may run out of drive at the first mastered region before crossing the gap. If that happens, the diagnosis for the next step is already written: I would need a novelty signal that does not decay the moment local dynamics are learned.

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
