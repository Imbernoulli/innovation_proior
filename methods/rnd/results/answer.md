# Random Network Distillation

## Method

Define a fixed random target network \(f:\mathcal O \to R^k\) and a trainable predictor
\(\hat f_\theta:\mathcal O \to R^k\). The target is initialized once and then frozen. The predictor
is trained on the agent's visited observations by mean-squared distillation:

\[
L_{\text{pred}}(\theta)=
E_{x\sim D}\left[\frac{1}{k}\sum_{j=1}^k
(\hat f_\theta(x)_j-\operatorname{sg}[f(x)_j])^2\right].
\]

The intrinsic reward for a transition is the same error on the next observation:

\[
i_t=\frac{1}{k}\sum_{j=1}^k
(\hat f_\theta(s_{t+1})_j-f(s_{t+1})_j)^2.
\]

The bonus is often written as \(\|\hat f(s_{t+1})-f(s_{t+1})\|^2\). The reference OpenAI code
uses `reduce_mean` over feature dimensions, so the faithful implementation is MSE. A summed squared
norm differs by the constant factor \(k=512\), which is largely absorbed by intrinsic-return
normalization, but the code-level artifact should use the mean.

The prediction problem is chosen to remove the bad sources of prediction error: the target is
deterministic, so stochastic environment transitions cannot create irreducible target noise; the
target depends only on the observation being predicted, so no hidden transition variable is needed;
and the predictor is made more flexible than the target in the reference architecture.

## PPO Combination

Use separate reward streams and value heads:

\[
R^I_t=\sum_{\ell\ge 0}\gamma_I^\ell \tilde i_{t+\ell},
\qquad
R^E_t=\sum_{\ell\ge 0}\gamma_E^\ell
\left(\prod_{m=1}^{\ell}(1-d_{t+m})\right)e_{t+\ell}.
\]

Intrinsic returns are non-episodic: done masks are ignored. Extrinsic returns are episodic: done
masks truncate bootstrapping. GAE is computed separately:

\[
\delta^I_t=\tilde i_t+\gamma_I V_I(s_{t+1})-V_I(s_t),
\qquad
\delta^E_t=e_t+\gamma_E(1-d_{t+1})V_E(s_{t+1})-V_E(s_t).
\]

The policy advantage is the weighted sum:

\[
A_t=c_I A^I_t+c_E A^E_t,\qquad c_I=1,\quad c_E=2.
\]

The value loss trains \(V_I\) toward \(R^I\) and \(V_E\) toward \(R^E\) as separate heads. Default
Atari constants from the reference code are \(\gamma_I=0.99\), \(\gamma_E=0.999\),
\(\lambda_{\text{GAE}}=0.95\), PPO clip range \(0.1\) (ratio clipped to \([0.9,1.1]\)), entropy
coefficient \(0.001\), Adam learning rate \(10^{-4}\), rollout length 128, 4 minibatches, and 4
optimization epochs.

## Normalization And Update Rate

The predictor and target receive one grayscale frame, not the policy's four-frame stack. Their input
is normalized as

\[
x'=\operatorname{clip}\left(\frac{x-\mu}{\sigma},-5,5\right),
\]

with running per-pixel mean and standard deviation initialized by a random-agent rollout. The policy
and value network still use four frames with \(x/255\) scaling.

The raw intrinsic reward is normalized by a running estimate of the standard deviation of discounted
intrinsic returns, not by min-max scaling and not by extrinsic reward statistics. In the reference
code this is implemented with a reward forward filter
\(z_t=\gamma_I z_{t-1}+i_t\), a running variance estimate over \(z_t\), and
\(\tilde i_t=i_t/\sqrt{\operatorname{Var}(z)+\epsilon}\).

When scaling from 32 to 128 parallel environments, the predictor is trained on a random 25 percent of
the rollout examples. This keeps the predictor's effective batch size, and therefore the decay rate
of intrinsic rewards, closer to the smaller-scale setup.

## Canonical Core

This PyTorch core mirrors the OpenAI TensorFlow implementation: DQN-style convolutional torso,
512-dimensional target, a predictor with two extra fully connected ReLU layers, feature-MSE reward,
frozen target, and masked predictor updates.

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

Implementation-faithfulness checks:

- Compute \(i_t\) from \(s_{t+1}\), using the final single frame after predictor/target observation
  normalization.
- Use feature-wise MSE mean, not a batch scalar and not a summed norm, for per-state rewards.
- Stop gradients through the target for both reward and predictor loss.
- Train the predictor with Adam through the same optimizer step as PPO's auxiliary loss, but mask
  examples so only the configured proportion contributes.
- Ignore done flags for intrinsic GAE unless explicitly running the episodic-intrinsic ablation.
- Use done flags for extrinsic GAE.
- Do not normalize extrinsic rewards beyond Atari sign clipping.
- Use two value heads and separate value losses; combine advantages, not raw rewards, with
  coefficients \(c_I=1,c_E=2\).
