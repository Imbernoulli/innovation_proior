I am handed a frozen JEPA world model and a fixed receding-horizon loop, and the only thing left to design is the planner: from the current encoded observation, produce an action sequence whose first action the loop executes before re-observing and asking me again. The model gives me one black-box operation — roll a batch of action sequences forward (`unroll`) and score the predicted latents (`objective`), composed into `cost_function`, which returns one scalar cost per candidate, lower is better. The actions are 2-D with a per-step norm cap of $\text{max\_norm} = 2.45$. Before I reach for anything clever I want a *reference rung*: the planner whose behavior I understand completely, that says plainly what "doing nothing clever" scores, so every later method has an honest yardstick to beat. The whole design space is just three decisions — which candidate sequences to try, how to score them, which one to return — and the floor should make the crudest defensible choice at each.

I propose single-pass random shooting (in the lineage of Rao 2009 and Richards 2005). The cost is a black box over action sequences, the surface is plainly multi-modal — a walls-and-doorway maze admits at least two qualitatively different routes to most goals, one through the door from each side — and every evaluation is a full horizon-length neural rollout, so queries are not free. The only operation I have is *querying the cost at points I choose*, and the floor's answer is to choose them once, at random, from a fixed distribution, and keep the best. No refit, no iteration that learns from the scores, no gradient. Its single defining virtue is that nothing in it can go subtly wrong in a data-dependent way: there is no distribution being updated from possibly-misleading early scores, no step size to mistune, no smoothness assumption. Whatever a cleverer planner buys, it buys *on top of* this and has to beat this curve to justify its machinery.

The sampling distribution follows from having no prior. Before I have looked at any cost I have no reason to prefer moving in any direction at any timestep, so the mean is zero in every action coordinate at every step and the coordinates are independent and identically spread — an isotropic standard normal over the whole $[\text{plan\_length}, \text{num\_samples}, \text{action\_dim}]$ tensor. Drawing each timestep *independently* makes a sampled sequence a white-noise action signal, and I want to be honest that this is the weakest part of the design: white actions integrate to a Brownian walk in position, whose expected squared displacement grows only *linearly* in the number of steps because the independent increments cancel, so a fixed budget of action energy buys only a small net excursion. In a maze where the goal sits on the far side of a wall most of these jittery sequences never reach the door at all. But that is precisely the failure I want the reference rung to *exhibit* rather than hide — it is the gap a smarter noise source or an iterative refit will later have to close, and leaving the noise white and single-pass keeps that gap legible.

Two refusals define the floor, and the refusals are the point. The first is the gradient: the pipeline — `unroll` then `objective` — is differentiable end to end, so I could in principle backpropagate the cost to the actions and descend it. I deliberately do not, because I have not earned the right to trust that the learned model's cost surface over action sequences is smooth enough that its gradient points anywhere useful; a deep composition of nonlinear predictor steps can have sharp, locally-misleading gradient directions that drive a descent straight into the wall. The second refusal is *adaptation* — folding the scores back into where I sample next. That is exactly what CEM will do, and the cleanest way to attribute CEM's eventual win to its refit rather than to some incidental advantage is to make this rung spend the same per-batch budget but refuse to learn from it. So the floor is two abstentions: no gradient, no adaptation.

The continuous action space forces one elementwise fix. A standard normal has unbounded support, so some per-step actions exceed the cap, and I do not want to hand the cost a candidate the agent could never execute — that would score a fantasy. I project each sampled 2-D action into the feasible ball before scoring: multiply by $(\text{max\_norm}/\|a\|)$ clamped to at most $1$ (with a tiny $\varepsilon$ in the denominator so a zero action does not divide by zero). Actions already inside the ball get a factor of $1$ and are untouched; actions outside are pulled exactly to the boundary. Feasibility is enforced before the black box ever sees a candidate, at the cost of one operation.

The budget choices keep the comparison fair. The harness hands the constructor $\text{num\_samples} = 200$ and $\text{n\_iters} = 20$, the shared defaults every ladder planner receives, and the floor deliberately *ignores* `n_iters` — it does one pass — while spending the $200$-sample budget as a single batch. That is the whole point: it spends the same per-batch budget the iterative methods will spend *per iteration*, but only once and without ever updating the sampling distribution. So if CEM, which reuses essentially this batch twenty times while marching toward the elites, beats the floor, the win is attributable cleanly to the iteration and the refit, not to a larger first pass. The selection rule is then forced: roll all $200$ candidates through the model, read the $200$ costs, return the single sequence with the lowest cost via `argmin`. The whole sequence goes back (the loop takes its first action); nothing persists across env steps, so the `t0` flag is irrelevant here; the plan length is $\min(\text{plan\_length}, \text{steps\_left})$ so the planner never proposes past the episode's end; and everything runs under `torch.no_grad()` because the floor only ever reads function values — the defining limitation of the whole zeroth-order family this rung opens. I expect a modest, nonzero success rate at the short horizon where lucky drifts and frequent re-planning can stumble to the goal, success capped well below the iterative methods at every horizon, and a large `mean_steps_to_success` — when random shooting succeeds, it succeeds slowly, because it wanders. That large step count is exactly what the next rung is on the hook to fix, by refitting the sampling distribution toward the low-cost sequences instead of re-drawing white noise.

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
