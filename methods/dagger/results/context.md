## Research question

We want to program a controller by demonstration: an expert policy `π*` shows good behavior,
and we want a learned policy `π` in some class `Π` that reproduces it. The standard move is to
treat this as supervised learning — collect `(state, expert action)` pairs from the expert's
trajectories and fit a classifier/regressor. This is a *sequential* prediction problem rather
than an i.i.d. one: the controller's own predictions (actions) determine the future states it
will see, so once `π` acts it changes the distribution of inputs it is later evaluated on,
unlike the fixed-distribution setting supervised learning is built around.

Concretely, write `d^t_π` for the distribution of states at time `t` when `π` is executed
from step `1`, and `d_π = (1/T) Σ_{t=1}^T d^t_π` for the average state distribution over a
horizon `T`. With task cost `C(s,a) ∈ [0,1]`, `C_π(s) = E_{a∼π(s)}[C(s,a)]`, the total
cost-to-go is `J(π) = Σ_{t=1}^T E_{s∼d^t_π}[C_π(s)] = T·E_{s∼d_π}[C_π(s)]`. In imitation
learning we usually do not observe `C`; instead we observe the expert and minimize a surrogate
loss `ℓ(s,π)` — for instance the expected 0-1 disagreement with `π*` at state `s`, or a
squared/hinge loss against `π*(s)`. The object of interest is the loss *under the policy's own
induced distribution*:

```
π̂ = argmin_{π ∈ Π} E_{s ∼ d_π}[ ℓ(s,π) ].
```

Here `d_π` depends on `π`, the dynamics are unknown and complex so we can only *sample* `d_π`
by executing `π`, and the dependence of the input distribution on the optimizee makes the
objective non-convex even when `ℓ(s,·)` is convex in `π` for every `s`. The question is how to
drive the loss under a policy's *own* state distribution to a small value over a horizon `T`,
given interactive access to the expert and a per-state error `ε`.

## Background

By this time, learning controllers from demonstration has produced state-of-the-art results
across robotics and games — outdoor navigation (Silver et al. 2008), legged locomotion
(Ratliff et al. 2006), manipulation (Schaal 1999), and game playing — and the dominant recipe
is the supervised one above. The field also has a mature **online learning / no-regret**
theory: an algorithm commits to a hypothesis `π_n` each round, then suffers a loss `ℓ_n(π_n)`
chosen possibly adversarially, and is measured by its *average regret* against the best fixed
hypothesis in hindsight,

```
(1/N) Σ_{i=1}^N ℓ_i(π_i)  −  min_{π ∈ Π} (1/N) Σ_{i=1}^N ℓ_i(π)  ≤  γ_N,
```

a *no-regret* algorithm being one with `γ_N → 0`. For strongly convex losses, classical
results (Hazan, Kalai, Kale & Agarwal 2006; Kakade & Shalev-Shwartz 2008; Kakade & Tewari
2009) give `γ_N = Õ(1/N)`, and the simplest such algorithm — *Follow-The-Leader*, which at
round `n` plays the hypothesis minimizing the cumulative loss seen so far — is no-regret on
strongly convex losses. There is also a well-developed **online-to-batch** machinery
(Cesa-Bianchi, Conconi & Gentile 2004) for converting per-round online guarantees into
high-probability bounds on the loss of one of the produced hypotheses under a fresh sample,
and a **reductions** viewpoint (Beygelzimer et al. 2005) that relates a hard learning problem
to a sequence of simpler classification/optimization calls and tracks how their errors
compose.

A recurring empirical observation in demonstration-trained controllers is the
**compounding-error phenomenon**. Pomerleau's ALVINN road-follower noted that a network "when
driving for itself … may occasionally stray from the center of road and so must be prepared to
recover by steering the vehicle back," yet recovery behavior is rare in a good human driver's
demonstrations and so is sparsely represented in the training data. When such a controller
drifts, it can find itself in a state unlike those it was trained on, and subsequent errors
accumulate over the horizon. This pattern has a quantitative signature (below).

A second background fact, from the policy-iteration side of reinforcement learning, is that
**changing the executed policy slowly** is a useful tactic. Conservative Policy Iteration
(Kakade & Langford 2002) improves a policy by interpolating only a little of a new greedy
policy into the current one each step, so the state distribution does not lurch and the
improvement estimates stay valid. The same instinct — keep the data-collecting policy close to
something trustworthy and let it drift toward the learner gradually — recurs in the
structured-prediction and imitation methods below.

## Baselines

**Supervised imitation / behavior cloning.** Ignore the distribution change and fit a policy
that does well under the *expert's* states:

```
π̂_sup = argmin_{π ∈ Π} E_{s ∼ d_{π*}}[ ℓ(s,π) ].
```

This is any off-the-shelf supervised learner. Its guarantee carries the compounding-error
signature: if `ℓ` upper-bounds the 0-1 loss and `E_{s∼d_{π*}}[ℓ] = ε`, then (Ross & Bagnell
2010)

```
J(π̂_sup) ≤ J(π*) + T² ε.
```

The bound is quadratic in `T`: the first time the learner errs (probability `~ε`) it can fall
into states never visited by `π*` and pay maximal cost `1` for the remaining steps, summed over
the horizon. The bound is tight — Kääriäinen (2006) exhibits a sequence-prediction problem
where predicting the next output from the previous *correct* output with per-step error `ε`
yields `T/2 − (1−(1−2ε)^{T+1})/(4ε) + 1/2` expected mistakes, which is `Θ(T²ε)` for small `ε`;
Ross & Bagnell (2010) give an imitation example with cost `(1−εT)J(π*) + T²ε`. The learner is
trained on `d_{π*}` and evaluated on `d_{π̂}`, two different distributions.

**Forward training (Ross & Bagnell 2010).** Train a *non-stationary* policy, one classifier
`π_t` per time step, sequentially from `t = 1` to `T`; at step `t`, `π_t` is fit to mimic `π*`
on the state distribution `d^t` actually induced by the already-trained `π_1,…,π_{t-1}`. Then
each `π_t` is trained on exactly the states it will face at deployment. The performance bound
improves to linear in `T` when a *cost-to-go gap* `u` is small: if
`Q^{π*}_{T-t+1}(s,a) − Q^{π*}_{T-t+1}(s,π*) ≤ u` for all `a,t` (the extra cost-to-go from one
off-policy action, then reverting to `π*`), and `E_{s∼d_π}[ℓ] = ε`, then
`J(π) ≤ J(π*) + uTε`, with `u ≤ 1` when the cost *is* the 0-1 imitation loss and `u = O(1)`
whenever `π*` recovers quickly from disturbances (e.g. a rapidly-mixing chain). It trains and
stores `T` distinct policies in sequence, running from `t = 1` to `T`.

**Stochastic Mixing Iterative Learning, SMILe (Ross & Bagnell 2010), and SEARN (Daumé,
Langford & Marcu 2009).** Get back to a horizon-independent, *stationary* policy by building a
stochastic *mixture*. SMILe starts from `π_0` that always queries and executes the expert;
at iteration `n` it trains `π̂_n` to mimic the expert under the trajectories `π_{n-1}` induces,
then sets `π_n = π_{n-1} + α(1−α)^{n-1}(π̂_n − π_0)`, which adds probability `α(1−α)^{n-1}` of
executing `π̂_n` at any step and removes that much probability of executing the queried expert;
after `n` iterations the probability of still deferring to the expert is `(1−α)^n`, and one
stops at `N` and returns the renormalized mixture `π̃_N = (π_N − (1−α)^N π_0)/(1−(1−α)^N)`.
With `α = O(1/T²)` and `N = O(T² log T)` this attains near-linear regret in `T` and `ε`. SEARN
(treating structured prediction as a degenerate imitation problem with trivial dynamics) and
Conservative Policy Iteration (Kakade & Langford 2002) are the same family — incrementally
fold a freshly trained policy into a growing mixture. The result is a *stochastic* controller,
a distribution over many policies, with an iteration scale of `O(T² log T)` in the standard
guarantee.

## Evaluation settings

The natural yardsticks already in use for imitation learning and structured prediction:

- **Super Tux Kart** (3D racing). Steer a kart at fixed speed from current image features; a
  human expert supplies the analog steering value in `[-1,1]`. Base learner a linear controller
  updated at 5 Hz via ridge regression on LAB pixel features of a downscaled image
  (`L(w,b) = (1/n) Σ (wᵀx_i + b − y_i)² + (λ/2) wᵀw`, `λ = 10^{-3}`). Metric: average number of
  falls off the track per lap (the track floats in space; the kart is reset to center on a
  fall). Roughly one lap (~1000 datapoints) of fresh data per iteration, ~20 iterations.
- **Super Mario Bros.** (platformer, from a Mario AI competition simulator). Play from image
  features; the expert is a near-optimal planner with full access to game state. Four binary
  actions {left, right, jump, speed}, four independent linear SVMs trained by SGD on a very
  sparse `27152`-dim binary feature vector (`λ = 10^{-4}`). Metric: average distance travelled
  per stage before dying / timing out / finishing (range ~`[0, 4300]`), on difficulty-1 stages
  with a 60 s limit. ~5000 datapoints per iteration, ~20 iterations.
- **Handwriting recognition / OCR** (Taskar et al. 2003 dataset, ~6600 words, ~52000
  characters, 10 folds). Treated as degenerate imitation learning (structured prediction):
  predict each character left-to-right, feeding the previously predicted character as input,
  with a linear multiclass SVM (all-pairs reduction). Large-dataset protocol: train on 9 folds,
  test on 1, repeat; metric is per-character accuracy on the test fold. Greedy, single-pass
  decoding.
- Protocol throughout: iterative training, fresh trajectories collected each iteration,
  performance plotted as a function of total training data; the existing iterative baselines
  (SMILe, SEARN) and the supervised baseline are the comparison points; for mixing-style
  methods the interpolation parameter is swept over a grid.

## Code framework

A controller-learning procedure can use the interactive imitation-learning harness that already
exists: we can roll out a policy in the system to collect states, we can query the expert `π*`
for its action at any state, and we have an off-the-shelf supervised learner that fits a policy
to a labeled dataset of `(state, action)` pairs. The substrate provides these generic pieces,
with one empty slot for the outer interaction loop that turns this interaction into a
controller.

```python
import numpy as np


def rollout(policy, env, horizon):
    """Execute `policy` in the system for `horizon` steps; return the visited states."""
    states, s = [], env.reset()
    for _ in range(horizon):
        states.append(s)
        s = env.step(policy.act(s))
    return states


def query_expert(expert, states):
    """Ask the expert for its action at each visited state (interactive access to π*)."""
    return [expert.act(s) for s in states]


def fit_policy(dataset):
    """Off-the-shelf supervised learner: fit a policy minimizing surrogate loss on a
    labeled (state, action) dataset. Any classifier/regressor/SVM/ridge can go here."""
    states = np.stack([s for s, a in dataset])
    actions = np.stack([a for s, a in dataset])
    policy = SupervisedPolicy()
    policy.train(states, actions)          # minimize ℓ on the given (s, a) pairs
    return policy


def imitate(expert, env, horizon, n_iters):
    """Turn interactive access to an expert into a deployable controller.

    The pieces above already exist: roll out a policy, query the expert at visited
    states, and refit a supervised policy. The outer interaction logic remains open."""
    # TODO: the interaction loop.
    pass
```

The inner supervised learner (`fit_policy`), the rollout, and the expert query are all given;
the open part lives entirely in the body of `imitate`.
