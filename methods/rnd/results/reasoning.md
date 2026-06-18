I need a novelty signal, not a surprise signal. A forward model is tempting because its error is
cheap: train on the stream of transitions, reward states or transitions where the model is wrong.
But the target of forward prediction is the next observation, and the next observation is exactly
where environment randomness enters. If a screen region contains random static, or if sticky actions
make a doorway crossing uncertain, then no amount of data removes the prediction error. The agent can
learn to harvest that permanent error. So raw forward error confuses the error I want, "I have not
seen enough data here," with errors I do not want, "the answer is random" or "the inputs do not
determine the answer."

Let me separate the causes. Prediction error can come from lack of nearby training data, stochastic
targets, misspecification, or optimizer dynamics. The first one is novelty. The second is the
noisy-TV trap. The third appears whenever I ask for a prediction that is not determined by the
inputs. The fourth never fully disappears, but I can at least avoid making it the main signal. The
right move is therefore to choose the prediction problem, rather than inherit it from the environment.

The cleanest possible target is deterministic. Take the current observation and pass it through a
function that never changes. If the target has no randomness, aleatoric error is gone. If the target
depends only on the observation being fed to the predictor, there is no hidden transition variable or
missing action history. The target also has to be rich enough that different images do not all look
the same. A randomly initialized convolutional network gives me that: arbitrary but fixed features of
the observation.

So I freeze a random target network \(f:\mathcal O \to R^k\), train a predictor
\(\hat f_\theta:\mathcal O \to R^k\) on observations the agent has actually visited, and use the
prediction error on the next observation as intrinsic reward:
\[
\ell_{\text{pred}}(x;\theta)=\frac{1}{k}\sum_{j=1}^k
(\hat f_\theta(x)_j-\operatorname{sg}[f(x)_j])^2,\qquad
i_t=\ell_{\text{pred}}(s_{t+1};\theta).
\]
I can write the same error either as a squared norm or as a mean over feature dimensions. That
distinction is only a constant factor before reward normalization, but if the code uses a
per-feature MSE then the artifact should use the mean.

Now I need to be honest about misspecification. The ideal argument is that I can choose the target to
be inside the predictor class, so the residual error is not caused by an impossible prediction task.
The practical implementation follows that spirit by making the target simple -- convolutional torso
plus one linear layer -- and the predictor at least as flexible -- the same torso shape plus extra
fully connected layers. That does not prove SGD finds a global imitation of the target everywhere.
In fact I do not want global imitation from data in one region. What I need is local fitting:
visited-like states should lose error as the predictor trains, while unseen regions remain high
until the agent gathers similar observations. A small supervised check with class-imbalanced images
is the sanity test: error on a rare class should fall as examples of that class are added, not vanish
just because the predictor saw many other images.

This also explains the relation to randomized priors. If a random function \(f_{\theta^*}\) is added
as a prior and a trainable function learns to cancel it on observed data, then the remaining ensemble
spread is an uncertainty estimate for a zero target. Here the output coordinates share parameters,
so it is not a literal independent ensemble, but the mean squared mismatch plays the same role:
where the data have constrained the predictor, uncertainty is low; where they have not, it is high.

The next issue is the reward stream. Intrinsic reward should not reset its horizon at game over.
Suppose a dangerous jump might reveal a new room. If death truncates all future intrinsic return, the
agent becomes conservative about exactly the risky attempts that exploration needs. The real cost of
death for curiosity is replaying familiar states, not losing all future novelty. So I should compute
intrinsic returns as non-episodic: ignore done flags in the intrinsic GAE recursion.

Extrinsic reward cannot use the same rule. If environment reward is made non-episodic, an agent can
take an early reward, die on purpose, and repeat. So extrinsic returns must remain episodic. The
linear return identity lets me keep both: estimate separate returns \(R_I\) and \(R_E\), train
separate heads \(V_I\) and \(V_E\), and add the resulting advantages with coefficients. The
implementation computes
\[
\delta^I_t=\tilde i_t+\gamma_I V_I(s_{t+1})-V_I(s_t)
\]
with no done mask, and
\[
\delta^E_t=e_t+\gamma_E(1-d_{t+1})V_E(s_{t+1})-V_E(s_t)
\]
with the done mask. Then PPO receives \(A_t=c_I A^I_t+c_E A^E_t\). The tuned constants are
\(c_I=1\), \(c_E=2\), \(\gamma_I=0.99\), \(\gamma_E=0.999\), and \(\lambda=0.95\).

The frozen random target makes input scale unusually important. A trained network can adapt its
first layer to the observation scale, but a frozen target cannot. If the pixel scale drives the
random activations into a nearly constant range, the feature target stops carrying useful image
information. Therefore the target and predictor should not receive the policy's four-frame
\(x/255\) stack. They receive a single grayscale frame, whitened by running mean and standard
deviation, then clipped to \([-5,5]\). Those statistics need a random-agent warmup before training
so the first target features are not degenerate. The policy/value network keeps the ordinary four
stacked frames scaled by \(1/255\).

The intrinsic reward scale also moves during training because the predictor is learning the target.
A fixed bonus coefficient would mean different things across games and across training time. The
implementation handles this by filtering intrinsic rewards into discounted intrinsic returns,
tracking the running variance of those returns, and dividing the raw intrinsic rewards by the return
standard deviation. That is not min-max scaling and not extrinsic reward normalization. Extrinsic
rewards are clipped by sign in Atari preprocessing and then left unnormalized.

The last scaling trap is the predictor update rate. If I increase the number of parallel actors, the
policy gets more data per update. If I also train the predictor on every extra observation, the
intrinsic reward decays faster and the policy may miss the transient stepping stones. So when using
128 environments, keep only a random quarter of examples for the predictor loss; with more actors,
drop more. The code implements this as a Bernoulli mask on per-example prediction losses and divides
by the number of kept examples, clamped to at least one.

Putting the pieces together, the method is not "reward what is hard to predict" in general. It is
"reward error on a deterministic synthetic target whose only reason to remain high should be lack of
nearby data," then value that stream with the right episode semantics. The target network is frozen
and random; the predictor distills it only on visited observations; the bonus is the mean squared
feature error on \(s_{t+1}\); the policy update sees a weighted sum of intrinsic and extrinsic
advantages while the two value heads are trained on their own returns.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


K = 512


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
    def __init__(self, feature_dim=K):
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

    def forward(self, normalized_single_frame):
        pred = self.predictor(normalized_single_frame)
        with torch.no_grad():
            target = self.target(normalized_single_frame)
        return pred, target

    @torch.no_grad()
    def reward(self, normalized_next_frame):
        pred, target = self.forward(normalized_next_frame)
        return F.mse_loss(pred, target, reduction="none").mean(dim=1)

    def loss(self, normalized_frames, keep_probability=0.25):
        pred, target = self.forward(normalized_frames)
        per_example = F.mse_loss(pred, target.detach(), reduction="none").mean(dim=1)
        mask = (torch.rand_like(per_example) < keep_probability).float()
        return (per_example * mask).sum() / mask.sum().clamp_min(1.0)
```
