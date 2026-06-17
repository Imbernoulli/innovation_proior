# Context: offline decision-making from fixed trajectory datasets (circa 2021-2022)

## Research question

We are handed a fixed dataset of trajectories collected by some unknown behavior policy on a
Markov decision process — each trajectory a sequence of states, actions, rewards, and each
labeled with the return it achieved — and we may never interact with the environment again.
The trajectories are overwhelmingly sub-optimal. The goal is to recover a return-maximizing
policy from this static log. This matters because in the real world (robotics, health,
driving) online exploration is expensive or unsafe, so the only thing we can count on having
is a pile of previously logged behavior, much of it mediocre.

The hard part is *stitching*: the optimal trajectory may never appear in the dataset, yet it
can be assembled from fragments of several sub-optimal ones (a good A→B segment from one
trajectory, a good B→C segment from another). A method that merely imitates the data cannot
produce a behavior better than the data; a method that can recombine fragments can. So the
precise objective is a procedure that (1) extracts high-return behavior from a dataset of
mostly low-return behavior, (2) recombines fragments across trajectories rather than copying
whole ones, (3) does not drift to actions or states unsupported by the data (where any model
is guessing), and (4) does so without a brittle training pipeline that needs heavy per-task
tuning. Each existing route below buys some of these and pays for it elsewhere.

## Background

The prevailing route to offline decision-making is reinforcement learning built on temporal
difference (TD) learning. One estimates the optimal action-value
`Q*(s,a) = E_{tau~pi*}[R(tau) | s_0=s, a_0=a]` by minimizing the TD/Bellman residual

```
L_TD(theta) = E_{(s,a,r,s')~D} [ ( r + gamma·max_{a'} Q_theta(s',a') - Q_theta(s,a) )^2 ],
```

and, for continuous actions, a parametric policy `pi_phi(a|s)` is trained to maximize
`J(phi) = E_{s~D, a~pi_phi}[Q(s,a)]`. The max/argmax over actions is what implements
dynamic programming — it propagates value backward and is exactly what lets a learner stitch
sub-optimal pieces into something better than any single logged trajectory. But this estimator
combines function approximation, bootstrapping (the target contains `Q_theta` itself), and
off-policy data — the *deadly triad* (Sutton & Barto), which makes the iteration prone to
divergence and over-estimation. The instability is sharper offline: as `pi_phi` shifts the
induced state-visitation `d^{pi_phi}` away from the data distribution `d^{mu}`, the policy
queries `Q_theta` at actions never seen in `D`, where the network freely extrapolates large
values, and there is no environment feedback to correct the error. Offline RL therefore adds
an explicit in-distribution constraint of the form `D(d^{pi_phi} || d^{mu})` (a divergence
penalty / conservative regularizer / behavior constraint) folded into the TD objective. That
turns training into a constrained optimization whose balance must be re-tuned per task, with
implementation heuristics, to get reasonable performance.

A separate body of work — generative modeling — had just shown that diffusion probabilistic
models (Sohl-Dickstein et al. 2015; Ho et al. 2020, DDPM) can learn extremely expressive data
distributions. A diffusion model fixes a forward noising process that gradually corrupts a
clean datum `x_0` into Gaussian noise,
`q(x_{k+1}|x_k) = N(x_{k+1}; sqrt(alpha_k)·x_k, (1-alpha_k) I)`, and learns a reverse
denoising process `p_theta(x_{k-1}|x_k) = N(mu_theta(x_k,k), Sigma_k)`. Rather than the full
variational bound, DDPM trains a noise predictor with the simplified objective

```
L_denoise(theta) = E_{k, x_0~q, eps~N(0,I)} [ || eps - eps_theta(x_k, k) ||^2 ],
```

where `x_k = sqrt(bar_alpha_k)·x_0 + sqrt(1-bar_alpha_k)·eps`. The denoiser's output is, up to
a known scale, the score of the noised data distribution: `eps_theta(x_k,k) ∝ -∇_{x_k} log
p(x_k)` (Song & Ermon; Song et al. 2020). This score view is what makes *guidance* possible —
steering generation toward samples that carry a desired attribute `y` by adding a term to the
score. Two guidance mechanisms existed:

- *Classifier guidance* (Dhariwal & Nichol 2021): separately train a classifier
  `p_phi(y | x_k)` on noised data and sample with the perturbed noise
  `eps_theta(x_k,k) - omega·sqrt(1-bar_alpha_k)·∇_{x_k} log p(y | x_k)`, where `omega` is the
  guidance scale. It works but requires an extra network trained on noisy inputs, and the
  sampling direction is literally a classifier gradient.
- *Classifier-free guidance* (Ho & Salimans 2022): train one network jointly as a conditional
  `eps_theta(x_k, y, k)` and an unconditional `eps_theta(x_k, ∅, k)` model (by replacing `y`
  with a null token `∅` with some probability during training), then sample with
  `eps_hat = eps_theta(x_k, ∅, k) + omega·(eps_theta(x_k, y, k) - eps_theta(x_k, ∅, k))`.
  The difference of conditional and unconditional scores is the gradient of an *implicit*
  classifier `∇log p(y|x) = ∇log p(x|y) - ∇log p(x)`, so no separate classifier is trained.
  Larger `omega` sharpens toward class-typical samples, trading diversity for fidelity.

There is also a known compositional property of score models (Liu et al. 2022; energy-based
composition): if attributes `{y^i}` are conditionally independent given the data, the score of
the joint conditional factorizes additively, `∇log q(x | {y^i}) = ∇log q(x) + sum_i (∇log
q(x|y^i) - ∇log q(x))`, so several attributes can be combined at sampling time even if no
single training example satisfied them jointly.

In these control datasets, states are continuous and state sequences along a trajectory are
relatively smooth, whereas actions — frequently joint torques — are higher-frequency, less
smooth, and sometimes discrete or sharply varied; high-frequency targets are harder for a
denoiser to fit than smooth ones. The datasets are also mixtures: a conditional model trained
naively on `(trajectory, return)` pairs inherits the sub-optimal behavior that dominates the
data.

## Baselines

**TD-based offline RL — CQL (Kumar et al. 2020), IQL (Kostrikov et al. 2021).** Learn a
conservative or expectile-regressed `Q` plus a constrained policy; the `max_{a'} Q(s',a')`
backup performs the stitching, and a behavior/conservatism term keeps the policy in
distribution. *Limitation:* the value estimate sits on the deadly triad and, offline,
over-estimates `Q` on out-of-distribution actions with no environment to correct it; the
in-distribution constraint is an extra knob that demands per-task tuning and heuristics, and
the whole pipeline is fragile and hard to scale.

**Diffuser (Janner et al. 2022).** The first to bring diffusion to planning: learn an
*unconditional* diffusion model over full state-action trajectories `x_k(tau) = (s_t, a_t,
..., s_{t+H-1}, a_{t+H-1})_k` with a temporal (1D-convolutional) U-Net, treating the
trajectory as a 1D image (state-action dimension × horizon). To plan for high return, train a
*separate* return/value predictor `J_phi(x_k)` on noised trajectories and guide the reverse
process by its gradient: `eps_hat = eps_theta(x_k,k) - omega·sqrt(1-bar_alpha_k)·∇ J_phi(x_k)`
— i.e. classifier guidance with the value function as the classifier. *Limitations:* (i) it
still trains a value-style function and uses its gradient, so the value-estimation machinery
(and its instability) is not removed, only relocated to a guidance term; (ii) the sampling
procedure differs from the training procedure — training fits an unconditional model, sampling
bolts on a separately learned objective — so what is sampled is not what was modeled; (iii) it
diffuses over actions jointly with states, and the action channel is the high-frequency, hard
part of the trajectory.

**Return-conditioned sequence models — Decision Transformer (Chen et al. 2021), RvS (Emmons
et al. 2021).** Cast policy learning as conditional behavioral cloning: a transformer (DT) or
a carefully tuned MLP (RvS) models `p(a_t | return-to-go, past states/actions)` and, at test,
is conditioned on a high target return. Competitive with offline RL without any value
function. *Limitations:* it is an autoregressive likelihood model of the next action, so it
inherits no score-composition — it cannot combine several conditioning attributes that were
never seen together — and its stitching ability is limited by how well return-conditioning
alone reorganizes the data.

## Evaluation settings

The natural yardstick is the D4RL benchmark (Fu et al. 2020) for offline RL. Locomotion:
MuJoCo `hopper`, `walker2d`, `halfcheetah` at the `medium`, `medium-replay`, and
`medium-expert` data qualities; the agent is trained purely offline on `env.get_dataset()`
and then rolled out in the environment. The metric is the D4RL normalized score (0 = random
policy, 100 = expert), reported per environment, averaged over evaluation episodes and seeds.
Harder long-horizon credit-assignment settings (D4RL Kitchen) and manipulation/skill settings
(block stacking, quadruped gaits) exist as additional protocols, with success rate or
return-based metrics. Standard practice: normalize returns into `[0,1]` per dataset; fix the
diffusion-step count, planning horizon, and optimizer; and compare against established
offline-control and trajectory-model baselines. Comparators on the leaderboard include CQL,
IQL, the trajectory transformer, MoReL, Decision Transformer, and Diffuser.

## Code framework

The substrate is a generic conditional-trajectory planning harness: an offline dataloader
that yields trajectory arrays with labels, a temporal network interface, standard DDPM
noising and reverse-posterior utilities, and a receding-horizon rollout loop that asks the
model for one action at each environment step. Before the design exists, the unresolved
object is only a trajectory-model slot: it receives offline batches during training and an
observation plus label during control.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


def cosine_beta_schedule(K):
    # standard DDPM noise schedule -> betas, alphas, bar_alpha (cumulative product)
    ...


class OfflineTrajectoryPlanner(nn.Module):
    """Generic offline trajectory-planning shell with standard DDPM utilities."""

    def __init__(self, trajectory_model, horizon, obs_dim, act_dim, K=100):
        super().__init__()
        self.trajectory_model = trajectory_model  # TODO: fill the trajectory model.
        self.horizon, self.obs_dim, self.act_dim, self.K = horizon, obs_dim, act_dim, K
        self.register_buffer_schedule(cosine_beta_schedule(K))

    def q_sample(self, x0, k, eps):
        # Standard DDPM forward: x_k = sqrt(bar_alpha_k) x0
        #                         + sqrt(1 - bar_alpha_k) eps.
        return self.sqrt_bar_alpha[k] * x0 + self.sqrt_one_minus_bar_alpha[k] * eps

    def q_posterior(self, x0_pred, x_k, k):
        # Standard DDPM posterior q(x_{k-1} | x_k, x0_pred).
        beta_k = self.beta[k]
        alpha_k = self.alpha[k]
        bar_k = self.bar_alpha[k]
        bar_prev = self.bar_alpha_prev[k]
        mean = (beta_k * bar_prev.sqrt() / (1 - bar_k)) * x0_pred
        mean = mean + ((1 - bar_prev) * alpha_k.sqrt() / (1 - bar_k)) * x_k
        var = beta_k * (1 - bar_prev) / (1 - bar_k)
        return mean, var

    def loss(self, trajectory, label):
        # TODO: define the training objective for the trajectory model.
        return self.trajectory_model.training_loss(trajectory, label, self.q_sample)

    @torch.no_grad()
    def reverse_step(self, sample, k, label):
        # TODO: define one model-specific reverse step.
        return self.trajectory_model.reverse_step(sample, k, label, self.q_posterior)

    @torch.no_grad()
    def plan_action(self, obs, label):
        # TODO: define how the trajectory model returns one executable action.
        return self.trajectory_model.plan_action(obs, label, self.reverse_step)


# existing offline training / rollout harness the method plugs into
def train(model, dataloader, optimizer):
    for trajectory, label in dataloader:       # offline (trajectory, return/attribute)
        optimizer.zero_grad()
        loss = model.loss(trajectory, label)   # the slot above
        loss.backward()
        optimizer.step()


def evaluate(model, env, target_label):
    obs, done = env.reset(), False
    while not done:
        act = model.plan_action(obs, target_label)
        obs, _, done, _ = env.step(act)
```

The harness supplies the DDPM machinery, the offline dataloader, and the rollout loop; the
single unresolved object supplies the trajectory-model behavior.
