# Context: derivative-free optimization of a noisy black-box objective

## Research question

I have a real-valued objective `S(x)` over a vector `x` and I want its maximizer
`x* = argmax_x S(x)` (equivalently, a cost `C(x)` to minimize, with `S = -C`). The hard part is
what I am *not* given. There is no gradient `∇S` — `S` is a black box I can only *query*: hand it an
`x`, get back a number. The query may itself be expensive (each evaluation is a simulation or a
forward rollout of a model) and it may be **noisy**, so two queries of the same `x` need not agree.
The landscape is **multi-extremal**: `S` can have many local optima, so any method that just walks
uphill from one starting point will lock onto whichever basin it happens to fall into. And `x` can
be moderately high-dimensional (tens of coordinates), so I cannot grid-search the space.

A method that solves this would have to: explore globally enough to find the right basin among many;
need only function *values*, never derivatives; tolerate noise in those values; spend its limited
query budget where it counts, concentrating effort on promising regions instead of sampling
uniformly forever; and come with some principle for *how* to move from "where I've been sampling" to
"where I should sample next" — not an ad-hoc nudge, but a rule I can justify. Closing that gap — a
gradient-free, noise-tolerant, globally-minded optimizer with a principled update — is the problem.

A concrete instance that stresses every one of these. Suppose the "objective" is the negative cost
of a *plan*: `x` is a sequence of actions `(a_1, ..., a_H)` over a horizon `H`, and `S(x)` is minus
the cost incurred by rolling those actions forward through a learned dynamics model and scoring the
resulting trajectory against a goal. Here the model rollout *is* the black box; it is differentiable
in principle but the cost surface over action sequences is bumpy and multi-modal (walls, detours,
narrow doorways), gradients through a long rollout are unreliable, and we want a planner that simply
asks the model "what does this action sequence cost?" and improves. The dimensionality is `H ×
action_dim`, which is enough to make blind search expensive but still small enough for batched
sampling to be plausible.

## Background

The simplest gradient-free idea is **pure random search**: fix a sampling distribution over `x`,
draw a batch, evaluate `S` on each, keep the best so far, repeat. It needs only function values, it
explores globally, and it cannot get stuck in a local optimum the way a hill-climber does. But it
never *learns*: the distribution it samples from on iteration 100 is the same one it used on
iteration 1, so it keeps spending queries on regions it has already learned are bad. All the
structure revealed by past evaluations is thrown away. The obvious wish is to let the sampling
distribution *adapt* toward where the good points have been — but "toward the good points" has to be
turned into a precise update rule, and that is exactly where naive averaging schemes get shaky.

The field also has a body of theory about a superficially different problem: **estimating the
probability of a rare event** by simulation. Suppose `X ~ f(·; u)` for some baseline density
`f(·; u)`, and I want `ℓ = P_u(S(X) ≥ γ) = E_u[ I{S(X) ≥ γ} ]` for a threshold `γ` so high that the
event is very unlikely. Crude Monte-Carlo — draw `N` samples, count how many clear `γ` — is hopeless
when `ℓ` is tiny: almost every indicator is zero, so you need an astronomically large `N` for a
single hit, let alone a low-variance estimate. The standard cure is **importance sampling (IS)**:
instead of sampling from `f(·; u)`, sample from a *different* density `g` chosen to make the rare
event common, and correct for the change of measure with the likelihood ratio `W = f(·; u)/g`:

```
ℓ = E_g[ I{S(X) ≥ γ} · f(X; u)/g(X) ],   estimated by  (1/N) Σ_k I{S(X_k) ≥ γ} · f(X_k; u)/g(X_k).
```

There is a known *best possible* IS density — the one that drives the estimator's variance to zero.
It is the baseline density **restricted to the rare event and renormalized**:

```
g*(x) = I{S(x) ≥ γ} · f(x; u) / ℓ .
```

Under `g*` the estimator is exactly `ℓ` for every sample (zero variance), because the likelihood-ratio
summand becomes constant on the event. The catch is that `g*` is unusable directly: it contains the
unknown `ℓ`, and it is not in general a convenient density to sample from. The rare-event simulation
toolkit nevertheless makes the choice of simulation density a parameter-selection problem rather than
a fixed Monte-Carlo counting problem. The older toolkit also includes the **Kullback--Leibler
divergence** (relative entropy),

```
D(g, h) = E_g[ ln(g(X)/h(X)) ] = ∫ g(x) ln g(x) dx − ∫ g(x) ln h(x) dx ,
```

an asymmetric measure that is zero iff `g = h`. Two more pre-existing facts are available without
settling how to use them. First, densities in common exponential families (Gaussian, Bernoulli,
categorical, ...) have fast random generators. Second, ordinary parameter-estimation problems for
many of these families have closed forms, often in terms of empirical frequencies or moments. Those
are just raw statistical primitives at this point; they do not yet say which samples should be
treated as data, how a threshold should be chosen, or what update an optimizer should perform.

For the global, multi-extremal side of the picture, the prevailing gradient-free heuristics of the
time are metaheuristics: **simulated annealing** (Kirkpatrick, Gelatt & Vecchi 1983), which does a
biased random walk with a temperature that is slowly lowered so it escapes local optima early and
settles late; and **genetic algorithms** (Holland 1975; Goldberg 1989), which maintain a *population*
of candidate solutions and evolve it by selection, crossover and mutation. Both are population- or
sample-based and derivative-free, and both encode the same instinct — keep the good candidates,
perturb, repeat — but their update rules are biological/physical analogies rather than statements
about an underlying distribution, so there is no clean answer to "what distribution am I converging
to, and why this update and not another."

The unresolved bridge is now precise but still empty. Rare-event simulation has tools for choosing a
simulation law once a difficult event has been specified, and black-box optimization has scored
samples but no principled way to turn them into the next proposal. Nothing in the background yet says
how to choose the useful event level from the current batch, which samples should count as evidence,
or how the next proposal should be fit.

## Baselines

These are the prior methods a new gradient-free optimizer would be measured against and would react
to.

**Pure / adaptive random shooting.** Draw `N` action sequences (or candidate vectors) from a fixed
distribution, evaluate each by rollout, execute the best. In model-predictive control this is
"random shooting." Core idea: trivially parallel, gradient-free, unbiased exploration. **Limitation:**
the proposal never improves — every iteration re-samples from the same spread, so under a fixed
query budget it wastes most evaluations on regions already shown to be poor, and its hit rate on a
narrow high-performing region degrades badly as dimension grows.

**Simulated annealing (Kirkpatrick, Gelatt & Vecchi 1983).** A single-state stochastic search:
propose a local perturbation, accept it with probability `min(1, exp(ΔS / T))`, and lower the
temperature `T` over time so that early on almost anything is accepted (global exploration) and
later only improvements are (local refinement). **Limitation:** it carries one state, not a
population, so it gleans no *collective* picture of where the good region is; performance is very
sensitive to the cooling schedule and the proposal step size; and on a noisy `S` the accept/reject
test is corrupted by the evaluation noise.

**Genetic / evolutionary algorithms (Holland 1975; Goldberg 1989).** Maintain a population, select
the fitter members, recombine and mutate them to form the next generation. Core idea: a population
gives parallel exploration of several basins, and selection concentrates effort on good candidates.
**Limitation:** crossover and mutation are heuristic operators with many knobs (population size,
crossover and mutation rates, encoding) and no underlying statement of *what distribution* the
population represents or is converging to; the update is justified by analogy, not by an
optimality criterion, so there is no principled answer to "given my current good candidates, where
exactly should I sample next."

**Sampling-based receding-horizon random shooting.** In an optimal-control or planning harness,
sample many action sequences around a nominal plan, roll each out through the simulator or model, and
execute the best first action before replanning. Core idea: the controller can use a simulator as a
black-box cost oracle and avoid differentiating through the rollout. **Limitation:** with a fixed or
hand-tuned sampling law, every planning call spends the same kind of exploration budget again; if
only the single best sequence is retained, the rest of the population gives little statistical
picture of the promising region.

## Evaluation settings

The natural yardstick for a gradient-free planner of this kind, all of which predate any particular
update rule:

- **Goal-conditioned navigation in a gridworld with obstacles.** A "two rooms" 2-D environment — a
  bounded grid (e.g. 65×65) split by a wall with a single doorway — where an agent at a random start
  must reach a random goal. Action is a 2-D displacement per step; an episode runs up to a fixed
  number of steps (e.g. 200); a run counts as a success if the agent ends within a small Euclidean
  radius of the goal (e.g. < 4.5). This stresses *long-range, multi-modal* planning: the straight
  line to the goal is blocked, so the planner must discover the detour through the doorway.
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
