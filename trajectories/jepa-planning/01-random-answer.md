**Problem.** From a frozen JEPA world model I can only *query* a black-box cost over action sequences
(roll through `unroll`, score with `objective`); no trusted gradient, a multi-modal maze cost (two
routes through the door), and each rollout is expensive. The reference planner must establish the floor
that every cleverer planner is measured against — no iteration, no learning from the scores.

**Key idea (single-pass random shooting).** Draw `num_samples` action sequences once from an isotropic
zero-mean Gaussian (no prior on direction; per-timestep actions independent — white noise), project
each per-step action into the feasible ball of radius `max_norm = 2.45`, roll all of them through the
model, and return the single lowest-cost sequence. It cannot lock into one basin the way a hill-climber
would, and it has no data-dependent state that can mistune — which is exactly what makes it the honest
yardstick.

**Why it is the floor.** It spends the same per-batch budget (`num_samples = 200`) the iterative
methods spend *per iteration*, but only once and without ever folding the scores back into the sampling
distribution (`n_iters` is ignored). So any later win is attributable to the iteration and the refit,
not to a bigger first batch. Its structural weakness is built in: white actions integrate to a Brownian
walk whose net displacement grows only like √steps, so most sequences never reach the door — depressed
success and large `mean_steps_to_success`, worst at the long horizons.

**Hyperparameters.** `num_samples = 200` (single batch), `action_dim = 2`, `plan_length` capped at
`steps_left`, `max_norm = 2.45`. No iterations, no persisted state, runs under `no_grad`.

```python
# EDITABLE region of eb_jepa/custom_planner.py — step 1: single-pass random search
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
