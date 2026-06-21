# Context: derivative-free optimization of a noisy black-box objective

## Research question

I have a real-valued objective `S(x)` over a vector `x` and I want its maximizer
`x* = argmax_x S(x)` (equivalently, a cost `C(x)` to minimize, with `S = -C`). I am *not* given a
gradient `∇S` — `S` is a black box I can only *query*: hand it an `x`, get back a number. The query
may itself be expensive (each evaluation is a simulation or a forward rollout of a model) and it may
be **noisy**, so two queries of the same `x` need not agree. The landscape is **multi-extremal**:
`S` can have many local optima. And `x` can be moderately high-dimensional (tens of coordinates),
too large to grid-search.

The question is how to optimize such an `S` using function *values* alone: how to spend a limited
query budget to locate the high-performing region of a multi-extremal surface, and what rule should
carry the search from "where I have been sampling" to "where I should sample next."

A concrete instance. Suppose the "objective" is the negative cost of a *plan*: `x` is a sequence of
actions `(a_1, ..., a_H)` over a horizon `H`, and `S(x)` is minus the cost incurred by rolling those
actions forward through a learned dynamics model and scoring the resulting trajectory against a goal.
Here the model rollout *is* the black box; the cost surface over action sequences is bumpy and
multi-modal (walls, detours, narrow doorways), and we want a planner that simply asks the model
"what does this action sequence cost?" and improves. The dimensionality is `H × action_dim`, large
enough that blind search is expensive but small enough for batched sampling.

## Background

The simplest gradient-free idea is **pure random search**: fix a sampling distribution over `x`,
draw a batch, evaluate `S` on each, keep the best so far, repeat. It needs only function values, it
explores globally, and it does not get stuck in a local optimum the way a hill-climber does. The
distribution it samples from on iteration 100 is the same one it used on iteration 1.

There is a separate body of theory about **estimating the probability of a rare event** by
simulation. Suppose `X ~ f(·; u)` for some baseline density `f(·; u)`, and I want
`ℓ = P_u(S(X) ≥ γ) = E_u[ I{S(X) ≥ γ} ]` for a threshold `γ` so high that the event is very
unlikely. Crude Monte-Carlo — draw `N` samples, count how many clear `γ` — needs an astronomically
large `N` when `ℓ` is tiny, since almost every indicator is zero. The standard tool is **importance
sampling (IS)**: instead of sampling from `f(·; u)`, sample from a *different* density `g` chosen to
make the rare event common, and correct for the change of measure with the likelihood ratio
`W = f(·; u)/g`:

```
ℓ = E_g[ I{S(X) ≥ γ} · f(X; u)/g(X) ],   estimated by  (1/N) Σ_k I{S(X_k) ≥ γ} · f(X_k; u)/g(X_k).
```

There is a known *variance-minimizing* IS density — the one that drives the estimator's variance to
zero. It is the baseline density **restricted to the rare event and renormalized**:

```
g*(x) = I{S(x) ≥ γ} · f(x; u) / ℓ .
```

Under `g*` the estimator is exactly `ℓ` for every sample (zero variance), because the
likelihood-ratio summand becomes constant on the event. `g*` contains the unknown `ℓ` and is not in
general a convenient density to sample from. The rare-event toolkit treats the choice of simulation
density as a parameter-selection problem rather than a fixed Monte-Carlo counting problem. It also
includes the **Kullback--Leibler divergence** (relative entropy),

```
D(g, h) = E_g[ ln(g(X)/h(X)) ] = ∫ g(x) ln g(x) dx − ∫ g(x) ln h(x) dx ,
```

an asymmetric measure that is zero iff `g = h`. Two further standing facts: densities in common
exponential families (Gaussian, Bernoulli, categorical, ...) have fast random generators; and
ordinary parameter-estimation problems for many of these families have closed forms, often in terms
of empirical frequencies or moments.

For the global, multi-extremal side, the prevailing gradient-free heuristics are metaheuristics:
**simulated annealing** (Kirkpatrick, Gelatt & Vecchi 1983), which does a biased random walk with a
temperature that is slowly lowered so it escapes local optima early and settles late; and **genetic
algorithms** (Holland 1975; Goldberg 1989), which maintain a *population* of candidate solutions and
evolve it by selection, crossover and mutation. Both are population- or sample-based and
derivative-free, and both encode the same instinct — keep the good candidates, perturb, repeat —
with update rules cast as biological/physical analogies.

## Baselines

These are the prior methods a new gradient-free optimizer would be measured against and would react
to.

**Pure / adaptive random shooting.** Draw `N` action sequences (or candidate vectors) from a fixed
distribution, evaluate each by rollout, execute the best. In model-predictive control this is
"random shooting." It is trivially parallel, gradient-free, and samples the space uniformly within
the fixed proposal.

**Simulated annealing (Kirkpatrick, Gelatt & Vecchi 1983).** A single-state stochastic search:
propose a local perturbation, accept it with probability `min(1, exp(ΔS / T))`, and lower the
temperature `T` over time so that early on almost anything is accepted (global exploration) and
later only improvements are (local refinement). It carries one state and is governed by a cooling
schedule and a proposal step size.

**Genetic / evolutionary algorithms (Holland 1975; Goldberg 1989).** Maintain a population, select
the fitter members, recombine and mutate them to form the next generation. A population gives
parallel exploration of several basins, and selection concentrates effort on good candidates. The
operators (crossover, mutation) and knobs (population size, crossover and mutation rates, encoding)
are specified by analogy.

**Sampling-based receding-horizon random shooting.** In an optimal-control or planning harness,
sample many action sequences around a nominal plan, roll each out through the simulator or model, and
execute the best first action before replanning. The controller uses a simulator as a black-box cost
oracle and avoids differentiating through the rollout.

## Evaluation settings

The natural yardsticks for a gradient-free planner of this kind:

- **Goal-conditioned navigation in a gridworld with obstacles.** A "two rooms" 2-D environment — a
  bounded grid (e.g. 65×65) split by a wall with a single doorway — where an agent at a random start
  must reach a random goal. Action is a 2-D displacement per step; an episode runs up to a fixed
  number of steps (e.g. 200); a run counts as a success if the agent ends within a small Euclidean
  radius of the goal (e.g. < 4.5). The straight line to the goal is blocked, so the planner must
  discover the detour through the doorway.
- **Planning horizons.** The same environment is run at several look-ahead horizons (e.g. 30, 60, 90
  steps) to probe short-, medium-, and long-range planning separately.
- **Replanning protocol.** Receding-horizon / model-predictive control: at each environment step the
  planner optimizes an action sequence of up to `plan_length`, executes (typically) the first action,
  observes, and replans.
- **The model as oracle.** A *fixed*, pre-trained latent dynamics model is provided; the planner does
  not train it. It exposes a `rollout(initial_state, actions)` that forward-simulates a batch of
  action sequences and an `objective(predicted_states)` that returns a scalar cost per sequence
  (lower is better). The planner's only interface to the world is querying this cost.
- **Metric.** Success rate (fraction of episodes that reach the goal) per horizon; higher is better.
- **Compute budget.** Per plan, a fixed number of sampled sequences per iteration and a fixed number
  of iterations — the optimizer must do well within that query budget.

## Code framework

The planner plugs into a fixed model-predictive-control harness. The dynamics model and the cost are
*given* (a fixed checkpoint); the open part is the **search procedure** that turns "I can query the
cost of any action sequence" into "here is a good action sequence." The substrate below is only the
generic machinery that already exists: a batched rollout-and-cost oracle, the action-sequence tensor
shapes, and the abstract planner interface with one empty slot where the optimization loop will go.

```python
import torch


class Planner:
    """Abstract planner over a fixed world model. Subclasses implement `plan`."""

    def __init__(self, unroll):
        # unroll(obs_init, actions): forward-simulate a BATCH of action sequences through the
        #   fixed world model.
        #     obs_init : [1, C, 1, H, W]   initial latent observation
        #     actions  : [B, A, T]         B action sequences, action_dim A, horizon T
        #   returns    : [B, D, T+1, H, W] predicted latent states
        self.unroll = unroll

    def objective(self, encodings):
        # cost of predicted states, [B, D, T, H, W] -> [B] (lower is better). Given/fixed.
        raise NotImplementedError

    def cost_function(self, actions, obs_init):
        # convenience: roll out then score. actions [B, A, T] -> cost [B].
        return self.objective(self.unroll(obs_init, actions))

    def plan(self, obs_init, steps_left=None, eval_mode=True, t0=False, plan_vis_path=None):
        raise NotImplementedError


class SamplingPlanner(Planner):
    """A gradient-free planner over action sequences. The cost is a black box queried via
    cost_function: given a batch of action sequences it returns one scalar per sequence
    (lower is better). No gradient of the cost is available. The search procedure that
    proposes sequences and improves them from past evaluations is what we will design."""

    def __init__(self, unroll, action_dim=2, plan_length=15,
                 num_samples=300, n_iters=30, max_norms=None, **kwargs):
        super().__init__(unroll)
        self.action_dim = action_dim
        self.plan_length = plan_length
        self.num_samples = num_samples      # query budget per iteration
        self.n_iters = n_iters              # number of optimization iterations
        self.max_norms = max_norms          # optional action-norm constraints supplied by the harness
        # TODO: any state the search procedure we design needs

    @torch.no_grad()
    def plan(self, obs_init, steps_left=None, eval_mode=True,
             t0=False, plan_vis_path=None):
        horizon = min(self.plan_length, steps_left) if steps_left else self.plan_length

        # TODO: the gradient-free optimization procedure we will design.
        #   We may query self.cost_function(actions, obs_init) -> [B] for any batch of
        #   B candidate action sequences (actions: [horizon, B, action_dim] rearranged to
        #   [B, action_dim, horizon]), as many times as the budget allows. Using only those
        #   returned costs, produce a good action sequence to execute.
        pass
```

The harness supplies the batched cost oracle and the tensor shapes; the body of `plan` is where the
search procedure will live.
