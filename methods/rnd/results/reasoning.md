I need a novelty signal, not a surprise signal. A forward model is tempting because its error is
cheap: train on the stream of transitions, reward states or transitions where the model is wrong.
But the target of forward prediction is the next observation, and the next observation is exactly
where environment randomness enters. Let me make that worry concrete before I trust it. Suppose some
patch of the screen shows static that is genuinely random -- a "noisy TV" -- so the relevant target
coordinate is \(y\in\{-1,+1\}\), each with probability one half, independent of anything the agent
can feed the model. The best the predictor can ever do is output the mean, and its residual error
floors at the variance: with the optimal constant prediction \(\hat y=E[y]=0\), the MSE is
\(E[(\hat y-y)^2]=\operatorname{Var}(y)=1\). I checked this on two million samples and the empirical
floor is \(1.0000\); no amount of data drives it down because the input does not determine the answer.
A curiosity agent rewarded by this error would park itself in front of the static forever. So raw
forward error confuses the error I want, "I have not seen enough data here," with an error I do not
want, "the answer is random."

That tells me the predicament is in the choice of target, not in the idea of prediction error. Let me
separate the causes a prediction error can have: lack of nearby training data, stochastic targets,
misspecification, or optimizer dynamics. The first one is the novelty I want. The second is the
noisy-TV trap I just measured. The third appears whenever I ask for a prediction that is not
determined by the inputs -- a missing action history, say. The fourth never fully disappears, but I
can at least avoid making it the main signal. The forward model inherits its target from the
environment and so inherits all four. What if I do not inherit the target at all, but manufacture it?

Then I get to dictate which of those four sources can occur. If I want the stochastic term gone, the
target should be a deterministic function of the observation I feed the predictor -- run the current
observation through some fixed function. Redo the floor calculation under that choice: with
\(y=f(x)\) and the same \(x\) always mapping to the same \(y\), once the predictor has fit that
\(x\) its residual is \((\,\hat y - f(x)\,)^2 = 0\). I confirmed this is exactly \(0.0\), not merely
small: there is no conditional variance left to floor against. And if the target depends only on the
observation being fed to the predictor, there is no hidden transition variable and no missing action
history, so the misspecification source is closed off too. That leaves "not enough nearby data" as
the dominant reason the error stays high -- which is precisely the signal I was after.

The function still has to be rich enough that different images do not collapse to the same target,
or the bonus would be blind to most of what changes on screen. A randomly initialized convolutional
network gives me arbitrary but fixed features of the observation, with the discriminating power of a
deep conv stack and none of the cost of training it. So I freeze a random target network
\(f:\mathcal O \to R^k\), train a predictor \(\hat f_\theta:\mathcal O \to R^k\) on observations the
agent has actually visited, and use the prediction error on the next observation as intrinsic reward:
\[
\ell_{\text{pred}}(x;\theta)=\frac{1}{k}\sum_{j=1}^k
(\hat f_\theta(x)_j-\operatorname{sg}[f(x)_j])^2,\qquad
i_t=\ell_{\text{pred}}(s_{t+1};\theta).
\]
I can write the same error either as a squared norm or as a mean over feature dimensions. To be sure
that distinction is only a scale factor and not a real choice, I generated random feature vectors and
compared \(\sum_j(\cdot)^2\) against \(\frac1k\sum_j(\cdot)^2\): the ratio is \(512.0\) on every row,
i.e. exactly \(k\). So it is a constant factor, largely absorbed by intrinsic-return normalization,
but if the code uses a per-feature MSE then the artifact should use the mean.

I should not wave away the misspecification source as fully closed. The clean argument is that I can
choose the target to lie inside the predictor's function class, so the residual is never caused by an
impossible task. The implementation follows that spirit by making the target simple -- convolutional
torso plus one linear layer -- and the predictor at least as flexible -- the same torso shape plus
extra fully connected layers. That does not prove SGD finds a global imitation of the target
everywhere, and I should not want it to. What I need is local fitting: visited-like states should
lose error as the predictor trains, while unseen regions stay high until the agent gathers similar
observations.

Is that actually the behavior of "regress a trainable function onto a frozen random one," or am I
hoping? Let me strip it to a case I can solve in closed form. Take a linear target \(f(x)=w^\top x\)
with random frozen \(w\) in five dimensions, and a linear predictor fit by least squares. Crucially,
let the observed data live only in a three-dimensional subspace (the last two coordinates are always
zero in training). Solving the regression, the fitted \(\hat\theta\) matches \(w\) on the seen
directions but comes out exactly \(0\) on the two coordinates the data never excited -- I printed
\(\hat\theta_{4:5}=[0,0]\) while the true \(w_{4:5}=[-1.07, 0.87]\). Now read the bonus
\((\hat\theta^\top x^* - w^\top x^*)^2\) at test points: on a point inside the seen subspace it is
\(0.0\) to machine precision, and on a point pointing along the unobserved directions it is positive,
\(0.043\). Pushing further out along those directions raises it monotonically -- \(0.011, 0.043,
0.172, 0.388\) as I scale the unseen component by \(0.5, 1, 2, 3\). So the residual on a frozen
random target really is small where data have pinned the predictor and grows with distance into
regions the data never constrained. That is an uncertainty estimate for a zero-mean quantity, which
is also why this sits next to randomized-prior methods: a fixed random function plus a trainable
function that cancels it on observed data leaves an ensemble-like spread that is large off the data.
Here the output coordinates share parameters rather than forming a literal independent ensemble, but
the mean squared mismatch plays the same role. The same toy also flags the honest limitation: the
predictor never learns the off-subspace structure of \(w\), so those directions stay novel -- which
is the intended behavior, not a bug, but it means the bonus measures distance-from-data, not
distance-from-truth.

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
by the number of kept examples, clamped to at least one. I traced that arithmetic on a tiny batch to
be sure the masking gives an honest mean and not a shrunk one: per-example losses
\([4,9,1,16]\) with a mask keeping examples \(0,2,3\) give \((4+1+16)/3=7.0\), and the code returns
\(7.0\) -- the division by `mask.sum()` keeps it an average over kept examples, so dropping examples
changes the variance of the estimate but not its target.

So the method is not "reward what is hard to predict" in general -- I ruled that out the moment the
noisy-TV floor came out at the full variance. It is "reward error on a deterministic synthetic target
whose only reason to remain high should be lack of nearby data," then value that stream with the
right episode semantics. The target network is frozen and random; the predictor distills it only on
visited observations; the bonus is the mean squared feature error on \(s_{t+1}\); the policy update
sees a weighted sum of intrinsic and extrinsic advantages while the two value heads are trained on
their own returns.

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
