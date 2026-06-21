Sparse-reward reinforcement learning needs directed exploration, not just dithering. A policy trained only on environment reward can sit in an all-zero learning signal for a very long time when meaningful rewards are separated by long action sequences and risky intermediate states, and on a benchmark like Montezuma's Revenge plain stochastic exploration almost never stumbles onto sustained positive reward. What we want is an intrinsic bonus that is high when the agent reaches a state unlike its past experience and low when the state is familiar — and that can be computed by batched neural-network inference over high-dimensional image observations, scaled across many parallel actors and billions of frames, and dropped into an ordinary policy-gradient optimizer. Tabular count bonuses like $1/N(s)$ or $1/\sqrt{N(s)}$ do not survive raw pixels, where almost every frame is unique; pseudo-count methods recover the needed generalization through a learned density model over observations, but that density model is a heavy extra system that is awkward to scale. Prediction-error curiosity replaces density estimation with a cheap supervised task — a forward model predicts the next observation or its features and large error marks a novel transition — but it confuses the error we want with the error we do not. The target of forward prediction is the next observation, and the next observation is exactly where environment randomness enters: random screen static, or sticky actions making a doorway crossing uncertain, produce a prediction error that no amount of data removes, and the agent learns to harvest that permanent error. This is the noisy-TV trap. Learning-progress methods reward improvement in the predictor rather than raw error to suppress that permanent stochastic component, but measuring improvement is more expensive and less clean inside a large distributed rollout system.

The failure is therefore one of choosing the wrong prediction problem. Prediction error has four causes: too little nearby training data, stochastic targets, misspecification when the inputs do not determine the answer, and optimizer dynamics. Only the first is epistemic novelty; the rest are traps. The right move is to *choose* the prediction problem rather than inherit it from the environment, and to choose it so that the only reason error should remain high is lack of nearby data. I propose Random Network Distillation. Fix a randomly initialized target network $f:\mathcal O\to\mathbb R^k$ once and freeze it, and train a predictor $\hat f_\theta:\mathcal O\to\mathbb R^k$ to imitate it on the observations the agent actually visits, by mean-squared distillation $$L_{\text{pred}}(\theta)=\mathbb E_{x\sim D}\!\left[\frac{1}{k}\sum_{j=1}^k\big(\hat f_\theta(x)_j-\operatorname{sg}[f(x)_j]\big)^2\right],$$ where $\operatorname{sg}[\cdot]$ stops gradients through the target. The intrinsic reward for a transition is the same per-feature error evaluated on the next observation, $$i_t=\frac{1}{k}\sum_{j=1}^k\big(\hat f_\theta(s_{t+1})_j-f(s_{t+1})_j\big)^2.$$ Every design choice here is in service of killing a bad error source. The target is deterministic, so stochastic environment transitions cannot inject irreducible target noise — the noisy-TV trap is closed by construction. The target depends only on the single observation being fed in, with no next-state and no action history, so there is no hidden transition variable and no missing-input misspecification. The target is rich because a random convolutional network defines arbitrary but fixed features that distinguish different images, so distinct states do not collapse to the same target. And the predictor is made strictly more flexible than the target — the same convolutional torso shape plus extra fully connected ReLU layers against a torso-plus-one-linear-layer target — so the residual is not an impossible fit. The argument is not that SGD globally imitates the target everywhere; I do not want that. I want *local* fitting: visited-like states lose error as the predictor trains, while unseen regions stay high until the agent gathers similar observations. This is also why it connects to randomized prior functions — a frozen random function plus a trainable function that cancels it on observed data leaves an ensemble-like spread that is low where the data have constrained the predictor and high where they have not — the mean squared mismatch is playing the role of a posterior uncertainty for a zero target. One implementation subtlety: the bonus can be written as a squared norm $\|\hat f-f\|^2$ or as a mean over the $k=512$ feature dimensions, and these differ only by the constant $k$, which intrinsic-return normalization largely absorbs; the faithful code-level choice mirrors the reference `reduce_mean` and uses the per-feature mean.

Three pieces make this robust enough to actually leave the first room. The first is episode semantics. Intrinsic reward should not reset its horizon at game over: if a dangerous jump might reveal a new room, truncating all future intrinsic return at death makes the agent conservative about exactly the risky attempts exploration needs, since the real cost of dying for curiosity is replaying familiar states, not losing all future novelty. So intrinsic returns are computed non-episodically, ignoring done flags. Extrinsic reward cannot share that rule — make environment reward non-episodic and an agent will grab an early reward, die on purpose, and repeat — so extrinsic returns stay episodic. The linearity of returns lets us keep both at once with separate streams, separate value heads $V_I$ and $V_E$, and separate GAE: $$\delta^I_t=\tilde i_t+\gamma_I V_I(s_{t+1})-V_I(s_t),\qquad \delta^E_t=e_t+\gamma_E\,(1-d_{t+1})\,V_E(s_{t+1})-V_E(s_t),$$ where the intrinsic delta carries no done mask and the extrinsic delta does. The policy is then updated on a weighted sum of *advantages*, not raw rewards, $A_t=c_I A^I_t+c_E A^E_t$, with the tuned Atari constants $c_I=1$, $c_E=2$, $\gamma_I=0.99$, $\gamma_E=0.999$, $\lambda_{\text{GAE}}=0.95$, PPO clip $0.1$, entropy coefficient $0.001$, Adam learning rate $10^{-4}$, rollout length $128$, $4$ minibatches, $4$ epochs. The second piece is input scale. A trained network adapts its first layer to the observation scale, but the frozen target cannot; if the pixel scale drives its random activations into a nearly constant range, the feature target stops carrying image information. So the target and predictor do not receive the policy's four-frame $x/255$ stack — they receive a single grayscale frame whitened by running per-pixel mean and standard deviation and clipped, $$x'=\operatorname{clip}\!\left(\frac{x-\mu}{\sigma},-5,5\right),$$ with the statistics seeded by a random-agent warmup so the first target features are not degenerate, while the policy/value network keeps its ordinary four stacked frames at $x/255$. The third piece is reward-scale drift: because the predictor is learning the target, the raw bonus magnitude moves across training and across games, so a fixed bonus coefficient would mean different things at different times. Rather than min-max scaling or extrinsic statistics, we filter intrinsic rewards into discounted intrinsic returns with $z_t=\gamma_I z_{t-1}+i_t$, track the running variance of $z_t$, and normalize $\tilde i_t=i_t/\sqrt{\operatorname{Var}(z)+\epsilon}$; extrinsic rewards are only sign-clipped by Atari preprocessing and otherwise left alone. Finally there is the predictor update rate: more parallel actors give the policy more data per update, and if the predictor also trains on every extra observation the intrinsic reward decays faster and the policy can miss transient stepping stones. So when scaling to 128 environments we keep only a random quarter of the rollout examples for the distillation loss — implemented as a Bernoulli mask on per-example losses, normalized by the number kept and clamped to at least one — which holds the predictor's effective batch size, and therefore the decay rate of novelty, near the smaller-scale setup.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvTorso(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 8, stride=4), nn.LeakyReLU(inplace=True),
            nn.Conv2d(32, 64, 4, stride=2), nn.LeakyReLU(inplace=True),
            nn.Conv2d(64, 64, 3, stride=1), nn.LeakyReLU(inplace=True),
            nn.Flatten(),
        )

    def forward(self, x):
        return self.net(x)


class RNDBonus(nn.Module):
    def __init__(self, feature_dim=512):
        super().__init__()
        self.target = nn.Sequential(ConvTorso(), nn.Linear(3136, feature_dim))
        self.predictor = nn.Sequential(
            ConvTorso(),
            nn.Linear(3136, feature_dim), nn.ReLU(inplace=True),
            nn.Linear(feature_dim, feature_dim), nn.ReLU(inplace=True),
            nn.Linear(feature_dim, feature_dim),
        )
        for p in self.target.parameters():
            p.requires_grad_(False)

    def forward(self, normalized_frame):
        pred = self.predictor(normalized_frame)
        with torch.no_grad():
            target = self.target(normalized_frame)
        return pred, target

    @torch.no_grad()
    def intrinsic_reward(self, normalized_next_frame):
        pred, target = self.forward(normalized_next_frame)
        return F.mse_loss(pred, target, reduction="none").mean(dim=1)

    def distillation_loss(self, normalized_frames, keep_probability=0.25):
        pred, target = self.forward(normalized_frames)
        per_example = F.mse_loss(pred, target.detach(), reduction="none").mean(dim=1)
        keep = (torch.rand_like(per_example) < keep_probability).float()
        return (per_example * keep).sum() / keep.sum().clamp_min(1.0)


class ReturnStdNormalizer:
    def __init__(self, gamma=0.99, eps=1e-8):
        self.gamma = gamma
        self.eps = eps
        self.forward_return = None
        self.rms = RunningMeanStd(shape=())

    def normalize(self, intrinsic_reward_batch):
        if self.forward_return is None:
            self.forward_return = intrinsic_reward_batch.detach()
        else:
            self.forward_return = self.gamma * self.forward_return + intrinsic_reward_batch.detach()
        self.rms.update(self.forward_return.reshape(-1).cpu())
        return intrinsic_reward_batch / (self.rms.var.sqrt().to(intrinsic_reward_batch.device) + self.eps)
```
