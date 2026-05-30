# Context

## Research question

Offline reinforcement learning asks for an effective policy learned **entirely from a fixed dataset**
`D` of transitions collected by some behavior policy `π_β`, with no further environment interaction.
This is exactly what one wants in robotics, logistics, or healthcare, where running an untrained
policy is costly or dangerous but logged data is plentiful.

The central difficulty is a tension that has no analogue in online RL. To do better than `π_β`, the
agent must estimate the value of actions *other* than those it has seen — but a value function is
only trustworthy on the state-action distribution it was trained on. The moment the agent queries
`Q(s, a)` at an action `a` far from the data, the estimate can be wildly wrong, and standard
Q-learning makes this catastrophic: its bootstrap target `max_{a'} Q(s', a')` deliberately searches
over *all* actions, finds whichever one the function approximator happens to over-value, and backs
that error up — so the value function inflates without bound and the policy chases the inflation.
Every prior remedy buys safety with a knob that trades policy improvement against robustness to this
distributional shift. The precise question: **can we build an offline RL method that never queries the
value of an unseen action during value training at all, yet still performs genuine multi-step dynamic
programming** — so it can chain together pieces of suboptimal trajectories ("stitching") and improve
substantially over the best behavior in the data?

## Background

**Approximate dynamic programming with TD error.** Most value-based RL minimizes a temporal-difference
loss `L_TD(θ) = E_{(s,a,s')~D}[(r(s,a) + γ max_{a'} Q_θ̂(s',a') − Q_θ(s,a))²]`, with a slowly-tracked
target network `θ̂` (Polyak averaging), and defines the policy as `π(s) = argmax_a Q_θ(s,a)`. The
`max_{a'}` queries actions the dataset may never contain; on those out-of-distribution (OOD) actions
`Q_θ̂` extrapolates, typically *upward*, and because the policy maximizes the estimate the error is
self-reinforcing.

**SARSA-style fitted policy evaluation.** If instead the bootstrap action `a'` is taken from the
dataset — `L(θ) = E_{(s,a,s',a')~D}[(r + γ Q_θ̂(s',a') − Q_θ(s,a))²]` — no OOD action is ever queried.
Because this is a mean-squared error, the optimum fits `Q_θ(s,a)` to the *mean* of the TD targets, so
it learns `Q^{π_β}`, the value of the behavior policy. Safe, but it only *evaluates* `π_β`; it does no
improvement.

**Expectile regression.** A τ-expectile (Newey & Powell 1987) of a random variable `X`, for
`τ ∈ (0,1)`, is the minimizer `m_τ = argmin_m E[L_2^τ(x − m)]` of the asymmetric squared loss
`L_2^τ(u) = |τ − 1(u<0)|·u²`. At `τ = 0.5` this is ordinary MSE and `m_τ` is the mean; for `τ > 0.5`
positive residuals are upweighted, pulling `m_τ` toward the upper part of the distribution, and as
`τ → 1` the expectile approaches the supremum of the support. The conditional form
`argmin_{m(·)} E_{(x,y)~D}[L_2^τ(y − m(x))]` is trained by ordinary SGD and fits a network to the
state-conditional expectile. It is the asymmetric-L2 cousin of quantile regression (Koenker 2001,
asymmetric L1), which distributional RL (Dabney et al. 2018) uses to estimate the *return*
distribution under stochastic dynamics — a different object from the per-state distribution over
*actions* considered here.

**Advantage-weighted regression.** A line of work (REPS, Peters et al. 2010; AWR, Peng et al. 2019;
AWAC, Nair et al. 2020) extracts an improved policy from a value function by solving the
KL-constrained improvement problem `max_π E_{a~π}[A(s,a)]` s.t. `KL(π‖π_β) ≤ ε`. Its closed-form
solution is `π*(a|s) ∝ π_β(a|s)·exp(A(s,a)/λ)`, and projecting that onto a parametric policy by
weighted maximum likelihood gives `L_π(φ) = E_{(s,a)~D}[exp(β·A(s,a))·log π_φ(a|s)]` with `A = Q − V`
and an inverse temperature `β`. Crucially this objective only ever evaluates *dataset* actions — it
reweights observed `(s,a)` pairs — so it queries no OOD action. Small `β` makes it behavioral cloning;
large `β` makes it greedy. It also implicitly enforces a stay-close-to-`π_β` constraint.

**Overestimation control and targets.** Clipped double Q-learning (Fujimoto et al. 2018) maintains two
Q-networks and uses their minimum to counter the upward bias of bootstrapping; a Polyak-averaged
target network stabilizes the bootstrap.

## Baselines

**Policy-constraint methods — BCQ, BEAR, BRAC, TD3+BC, AWAC.** These keep the learned policy close to
`π_β`. BCQ (Fujimoto et al. 2019) fits a generative model `μ(·|s)` of dataset actions, samples `N`
candidates from it, and takes `argmax` of `Q` over those candidates — a soft support constraint, but
the generative model can still emit OOD actions, and `N` plays a role analogous to a tail parameter.
TD3+BC (Fujimoto et al. 2021) adds a behavioral-cloning term to the actor loss; AWAC (Nair et al.
2020) enforces an implicit KL constraint via advantage weighting. Gap: all still bootstrap or query a
*learned* Q at policy actions during improvement, so they retain the improvement-vs-shift tradeoff.

**Value-regularization methods — CQL, Fisher-BRC.** CQL (Kumar et al. 2020) adds a regularizer that
pushes `Q` down on OOD actions and up on dataset actions, yielding a conservative lower bound. Strong
on the benchmark, but it still evaluates `Q` on OOD actions (to push them down), needs a tuned
regularization weight, and is comparatively slow.

**One-step / single-step methods — Onestep RL, AWR, Decision Transformer.** Onestep RL
(Brandfonbrener et al. 2021) and AWR (Peng et al. 2019) fit the value function of `π_β` with a
SARSA-style objective (no OOD actions) and then do a *single* step of policy improvement (greedy or
advantage-weighted); Decision Transformer (Chen et al. 2021) drops value functions entirely for a
return-conditioned BC sequence model. All are safe and simple and do well on near-optimal locomotion
datasets. Gap: they perform no iterated dynamic programming, so they cannot propagate value across a
path that no single dataset trajectory traverses — they fail badly on datasets that require stitching
suboptimal trajectories (e.g. medium/large ant mazes).

## Evaluation settings

The standard benchmark is D4RL (Fu et al. 2020): fixed datasets across MuJoCo Gym locomotion
(halfcheetah, hopper, walker2d, in medium / medium-replay / medium-expert variants — many of which
contain near-optimal trajectories), the Ant Maze navigation tasks (umaze / medium / large, with play
and diverse data — containing few or no near-optimal trajectories, so success requires stitching),
and the Adroit dexterous-hand and Franka Kitchen manipulation suites. Performance is the normalized
score (0 = random, 100 = expert), averaged over evaluation trajectories and several seeds. Standard
preprocessing matters: locomotion rewards are scaled by the spread of dataset trajectory returns, and
ant-maze rewards are shifted by a constant. Online fine-tuning is also assessed: initialize from the
offline solution, then run a fixed budget of online interaction and report final performance. A
controlled diagnostic is a small stochastic u-shaped maze with one optimal trajectory among many
random ones, where the learned value function can be compared against the exact optimal `V*`.

## Code framework

The primitives that already exist: an MLP builder, a state-action critic and a state-value network, a
Gaussian policy, Adam, a Polyak target-update, clipped double-Q, and an offline loop that samples
minibatches from a static dataset. What does not yet exist is any value-training objective that does
multi-step backups *without* a `max` over actions, nor a policy-extraction step that avoids querying
unseen actions. Those are the stubs.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


def mlp(sizes, activation=nn.ReLU):
    layers = []
    for i in range(len(sizes) - 1):
        layers += [nn.Linear(sizes[i], sizes[i + 1])]
        if i < len(sizes) - 2:
            layers += [activation()]
    return nn.Sequential(*layers)


class Critic(nn.Module):
    """Q(s, a)."""
    def __init__(self, obs_dim, act_dim, hidden=(256, 256)):
        super().__init__()
        self.net = mlp([obs_dim + act_dim, *hidden, 1])
    def forward(self, s, a):
        return self.net(torch.cat([s, a], dim=-1)).squeeze(-1)


class DoubleCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=(256, 256)):
        super().__init__()
        self.q1 = Critic(obs_dim, act_dim, hidden)
        self.q2 = Critic(obs_dim, act_dim, hidden)
    def forward(self, s, a):
        return self.q1(s, a), self.q2(s, a)


class ValueNet(nn.Module):
    """V(s)."""
    def __init__(self, obs_dim, hidden=(256, 256)):
        super().__init__()
        self.net = mlp([obs_dim, *hidden, 1])
    def forward(self, s):
        return self.net(s).squeeze(-1)


class GaussianPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=(256, 256)):
        super().__init__()
        # TODO: mean network + (state-independent) log-std
        pass
    def forward(self, s):
        # TODO: return an action distribution
        pass


def value_objective(q_target, v_pred):
    # TODO: a regression target for V that estimates the value of the *best
    #       in-data action* at each state, WITHOUT a max over actions
    pass


def q_objective(reward, mask, v_next, q_pred, discount):
    # TODO: a bootstrap target for Q that uses only dataset transitions
    pass


def policy_extraction_objective(q, v, log_prob):
    # TODO: turn the converged value functions into a policy WITHOUT
    #       querying Q at the policy's (possibly unseen) actions
    pass


def update(batch, critic, target_critic, value, policy, opts, hp):
    # TODO: order the value / policy / Q updates and the target sync
    pass
```
