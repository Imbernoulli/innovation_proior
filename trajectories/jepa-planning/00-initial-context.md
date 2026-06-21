## Research question

A JEPA world model is already trained: given an encoded observation and an action sequence, it
forward-simulates the *latent* encodings the agent would see, and a fixed objective scores how close
the predicted final latent is to a goal latent. The checkpoint is frozen — I am not allowed to retrain
it. The single thing being designed is the **planner**: the rule that, from the current observation,
searches over action sequences and returns the one to execute. The environment is Two Rooms — a 65×65
grid split by a wall with a single doorway — and the agent must navigate from a random start to a
random goal, which forces the plan to route *around* the wall and *through* the door rather than head
straight at the goal. Everything else (the world model, the cost, the receding-horizon control loop)
is fixed. Can a planner beat the standard derivative-free optimizers — CEM, MPPI — by exploiting the
structure of the learned model more carefully?

## Prior art / Background / Baselines

The planners used in this setting are zeroth-order trajectory optimizers inside a model-predictive
loop: sample candidate action sequences, roll each through the model, score it, keep the best, act,
and re-plan.

- **Random shooting.** Sample action sequences from a fixed distribution, evaluate them all, and
  return the lowest-cost one.
- **Cross-Entropy Method (CEM).** Refit a Gaussian over action sequences to the top-`k` lowest-cost
  "elites" each iteration, so the search distribution shifts toward better regions and auto-sizes
  its spread.
- **Model Predictive Path Integral (MPPI).** Refit the Gaussian to an exponentially cost-weighted
  average of the samples, so a markedly better rollout pulls the mean and tightens the variance more
  than a marginal one.
- **iCEM.** CEM augmented with temporally-correlated colored action noise, elite reuse across
  iterations and across env steps, and population decay to squeeze more refinement from a fixed
  budget.

## Fixed substrate / Code framework

A pre-trained JEPA world model and a receding-horizon control loop are frozen and must not be touched.
The loop, at each control step, calls the planner for an action sequence, executes its first action,
re-observes, and re-plans, until the episode ends (max 200 steps) or the agent is within Euclidean
distance 4.5 of the goal. The world model exposes, through the `Planner` base class the planner
inherits:

- `self.unroll(obs_init, actions)` — forward-simulate a batch of action sequences through the world
  model. `obs_init` is `[1, C, 1, H, W]`; `actions` is `[B, A, T]`; returns predicted state encodings
  `[B, D, T+1, H, W]`.
- `self.objective(encodings)` — cost of predicted encodings `[B, D, T, H, W]`, returns `[B]` (lower is
  better).
- `self.cost_function(actions, obs_init)` — convenience: `unroll` then `objective`, returns `[B]` cost
  per sample. Both `unroll` and `objective` are differentiable in the world model, so a cost can be
  backpropagated to the actions.

The action space is 2-D (x/y movement) with a per-step norm limit of `max_norm = 2.45`. The plan
horizon is capped at `plan_length` (and at `steps_left` when fewer steps remain).

## Editable interface

Exactly one region is editable — the `CustomPlanner` class in `eb_jepa/custom_planner.py` (the
`plan()` method and the constructor). `CustomPlanner` extends `Planner` and must implement:

```python
def plan(self, obs_init, steps_left=None, eval_mode=True,
         t0=False, plan_vis_path=None) -> PlanningResult:
```

`t0=True` marks the first call of an episode (the place to reset any state carried across env steps).
The method returns a `PlanningResult(actions=Tensor[T, A], ...)`; the loop executes the first action.
Every method fills this one slot. The starting point is the scaffold default: a single-pass **random
search** — sample action sequences once, clip them to the feasible ball, return the lowest-cost one.

```python
# EDITABLE region of eb_jepa/custom_planner.py — default fill (single-pass random search)
class CustomPlanner(Planner):
    """Random Search planner (lower bound baseline).

    Samples random action sequences and returns the one with lowest cost.
    No iterative refinement -- purely single-pass random sampling.
    """

    def __init__(self, unroll, action_dim=2, plan_length=15,
                 num_samples=200, n_iters=20, **kwargs):
        super().__init__(unroll)
        self.action_dim = action_dim
        self.plan_length = plan_length
        self.num_samples = num_samples
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @torch.no_grad()
    def plan(self, obs_init, steps_left=None, eval_mode=True,
             t0=False, plan_vis_path=None):
        from einops import rearrange

        plan_length = min(self.plan_length, steps_left) if steps_left else self.plan_length

        # Sample random actions
        actions = torch.randn(
            plan_length, self.num_samples, self.action_dim, device=self.device
        )

        # Clip action norms
        max_norm = 2.45
        norms = actions.norm(dim=-1, keepdim=True)
        actions = actions * (max_norm / norms.clamp(min=1e-6)).clamp(max=1.0)

        # Evaluate all samples and pick the best
        cost = self.cost_function(
            rearrange(actions, "t b a -> b a t"), obs_init
        )
        best_idx = cost.argmin()

        return PlanningResult(actions=actions[:, best_idx])
```

## Evaluation settings

- Environment: Two Rooms, 65×65 grid with a wall and a single door.
- Episodes: 20 per benchmark, random start and goal controlled by the MLS-Bench seed (seed 42).
- Max steps per episode: 200; success when Euclidean distance to goal `< 4.5`.
- Three planning-horizon benchmarks — 30, 60, 90 steps — probing short, medium, and long-range
  planning.
- Metric: `success_rate` per horizon (fraction of the 20 episodes solved), higher is better. The
  leaderboard also reports `mean_dist` (final distance to goal, lower better) and
  `mean_steps_to_success` per horizon.
