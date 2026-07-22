Let me start from what actually hurts. I have an agent navigating a cluttered environment, and I want two things at once that pull against each other: collect reward by reaching goals, and almost never touch the hazards. The hazards come with their own signal — a per-step indicator cost c_t that is 1 when the agent does the unsafe thing and 0 otherwise — and it is genuinely separate from the reward, not some term I get to define inside r_t. So the thing I want, written honestly, is: maximize E[Σ r_t] while keeping E[Σ c_t] under some budget d that a human picks because they know how much risk is acceptable. d = 25 cost-events per episode, say.

The reflex is to mash the two signals into one scalar and hand it to whatever policy optimizer I already trust — optimize E[Σ (r_t − β c_t)] for some penalty weight β. And every time I reach for this I hit the same wall. What is β? There is no formula that takes my budget d and returns the β that achieves it; the map from "penalty coefficient" to "resulting cost return" depends on the whole environment and the policy class in a way I can't invert ahead of time. If I guess β too small, at convergence the agent decides a few hazards are worth the reward and sits comfortably above the budget. If I guess too big, it becomes so frightened of cost that it never learns the task. And even in the lucky case where some β lands a safe policy at the optimum, that β tells me nothing about the trajectory of cost *during* training — the agent could spend the whole run wildly unsafe and only become safe at the end. So a fixed penalty is structurally the wrong object: it bakes a trade-off I don't know how to set into a constant I have to set before I see anything. And there's a worse failure even than mis-tuning: the one β has to do two incompatible jobs across the run — stay small early so the still-incompetent policy can learn the task at all, then push hard once the policy has found rewarding-but-risky shortcuts. No single number does both.

So I should stop trying to fold safety into the objective and instead keep it as what it is — a constraint. The clean statement is the constrained problem

  max_θ J_r(θ)  s.t.  J_c(θ) ≤ d,

where J_r(θ) = E_{τ~π_θ}[Σ r_t] and J_c(θ) = E_{τ~π_θ}[Σ c_t] are the expected returns of reward and of cost, defined identically except for which signal they sum. This is exactly a constrained MDP in Altman's sense: an MDP with an extra cost function and a threshold, whose feasible set is the policies sitting under the budget. The reason I like the constraint framing over the penalty framing isn't aesthetic — a constraint with a threshold encodes "get cost under d and then stop caring about it; spend everything else on reward," which is the actual shape of a safety requirement, whereas a penalty keeps trading reward against cost forever even far inside the safe region. The threshold has a saturation built in; the penalty doesn't.

Now, how do I actually optimize a constrained objective when J_r and J_c are both some horrible nonconvex function of a few hundred thousand neural-net weights? I can't project onto the feasible set — I don't even have a closed form for it. The standard tool for "maximize subject to an inequality" is the Lagrangian. Write the constraint in standard form g(θ) = J_c(θ) − d ≤ 0, introduce a multiplier λ ≥ 0 for it, and form

  L(θ, λ) = J_r(θ) − λ · g(θ) = J_r(θ) − λ (J_c(θ) − d).

The classical result is that the constrained problem equals the max-min of this Lagrangian: I want

  max_θ min_{λ≥0} L(θ, λ).

Let me make sure I believe the inner min before I build a whole algorithm on it. Fix θ. If the constraint is *violated*, g(θ) > 0, then min over λ ≥ 0 of −λ·(positive) drives λ → ∞ and L → −∞ — so the max over θ will never pick such a θ; infeasible policies are punished to −∞. If the constraint is *satisfied with slack*, g(θ) < 0, then −λ·(negative) is increasing in λ, so the min picks λ = 0 and L = J_r(θ) — the penalty switches off and I get pure reward. On the boundary g = 0 the λ term vanishes for every λ. So the inner min reconstructs exactly "reward if feasible, −∞ if not," which is the constrained objective. Good — the Lagrangian isn't a heuristic stand-in; its max-min *is* the constrained problem, at least at the level of values, and at the saddle the multiplier obeys complementary slackness, λ·g = 0: λ is positive only when the constraint binds.

But that's the textbook story for convex problems, and J_r, J_c are nonconvex in θ. For a generic nonconvex program there's a duality gap — the max-min through the Lagrangian can fall strictly short of the true constrained optimum, and then dual methods chase the wrong target. So I should not just assume duality works here; I need a reason it does for *this* problem. And there is one, and it's specific to RL. The nonconvexity is in the parameterization θ ↦ π_θ, not in the problem itself: viewed in the space of state-action *occupancy measures*, the return and the cost return are both *linear* in the occupancy measure, and the set of valid occupancy measures is convex (it's cut out by the Bellman flow equations). A linear objective with linear constraints over a convex set is a linear program, and an LP has no duality gap. So the *underlying* constrained-RL problem, expressed over occupancy measures, has zero duality gap under the usual feasibility assumptions — this is Altman's LP view of CMDPs, and the gap-closing fact carries to the policy formulation too. I want to be careful here, though: this clean duality lives in occupancy-measure space, while I will actually do gradient steps on the neural parameters θ, where the objective looks nonconvex and each policy update is only an approximate primal step. So the right reading is: dual descent is aimed at the correct saddle-point structure of the true problem; my neural implementation is an approximation of it, not an exact solver. That's enough license to build the algorithm — I'm not chasing a phantom.

So now I have a saddle-point problem, max over θ and min over λ ≥ 0 of L(θ, λ), and the natural algorithm for a saddle point is to alternate: ascend θ to push L up, descend λ to push L down, and let them chase each other to the saddle. This is primal-dual / dual descent, the Arrow–Hurwicz idea. Let me work out each piece.

The λ update first, because it's a single scalar and it's where the adaptivity lives. I want to move λ to *decrease* L = J_r − λ·g(θ), i.e. gradient descent on λ. The gradient of L with respect to λ is just −g(θ) = −(J_c(θ) − d). Descent means stepping in the negative gradient direction:

  λ ← λ − η · ∂L/∂λ = λ − η · ( −(J_c − d) ) = λ + η · (J_c − d).

And I have to keep λ ≥ 0, so project after the step:

  λ ← [ λ + η · (J_c(θ) − d) ]_+ .

Let me re-check the sign by behavior, not just algebra, because a flipped sign here would silently invert the whole safety mechanism. If the current policy is *over budget*, J_c > d so g > 0: the bracket increases, λ climbs, the next policy update weighs cost more heavily and gets pushed toward safety. Correct. If the policy is *under budget*, J_c < d so g < 0: the bracket decreases, λ falls toward zero, the policy is freed to chase reward with the slack it has. Also correct. So λ is doing the trade-off tuning *for* me, reading the actual measured violation each round, which is precisely the adaptive coefficient the fixed-penalty baseline could never produce. Structurally λ is the integral of the constraint-violation signal over time — it accumulates the running sum of (J_c − d), so it settles to whatever penalty weight is needed to drive the average violation to zero. The projection isn't cosmetic: an inequality multiplier must be nonnegative (that's the KKT condition, and it's the complementary-slackness story again — the multiplier measures how much reward I'd gain by loosening a binding budget, which can't be negative), and a negative λ would mean *paying* the agent to incur cost, exactly backwards.

One subtlety on what J_c is here. The constraint is on the *expected episodic* cost return, not on any single step. So I shouldn't update λ from a per-timestep cost; I should update it from an estimate of J_c over the epoch — the mean episodic cost across the trajectories I just collected. So once per update epoch I read the logged mean episodic cost, call it Jc, and do one λ step from it. I should also assert Jc isn't NaN before trusting it, because if the cost statistic ever goes bad I'd corrupt λ silently and the whole safety mechanism would quietly die.

Now the θ update — the policy. I'm ascending θ on L(θ, λ) = J_r(θ) − λ J_c(θ) (the constant −λ·(−d) = λd doesn't depend on θ, drop it). The gradient is ∇_θ J_r − λ ∇_θ J_c. I already have a machine that ascends an expected return given an advantage estimate: the policy-gradient / PPO surrogate, where the gradient of J is E[∇log π · A] with A the advantage. So ∇_θ J_r corresponds to the reward advantage A_r and ∇_θ J_c to the cost advantage A_c — I run GAE twice, once on a reward value function and once on a cost value function, and get A_r and A_c per state-action. The combined gradient ∇_θ(J_r − λ J_c) then corresponds to a single combined advantage

  A = A_r − λ A_c,

and I feed *that* into exactly the policy optimizer I already trust. Concretely PPO: form the importance ratio r_t = π_θ(a|s)/π_θ_old(a|s), and maximize the clipped surrogate min(r_t A, clip(r_t, 1−ε, 1+ε) A) on this combined A. The clip does its usual job — keeps each update inside a trust region around the old policy so a big advantage can't yank the policy too far in one step — and it's agnostic to what A means, reward or reward-minus-cost. So almost nothing about the base optimizer changes; I've only redefined the advantage it consumes. That's the whole appeal: a safe-RL algorithm that is PPO plus one scalar and one line that mixes two advantages.

Let me try shipping it just like that — A = A_r − λ A_c — and see where it breaks. Picture training where the agent keeps blowing the budget, so the λ update keeps incrementing λ epoch after epoch. λ doesn't stay small; it can climb to 5, 10, 50, whatever it takes to make the agent feel the cost. Now look at the magnitude of A = A_r − λ A_c when λ = 50: the cost term dominates and the *scale* of the combined advantage is roughly 50× the scale of a normal advantage. PPO's clip range, its learning rate, its whole step-size calibration were tuned assuming the advantage has roughly its usual magnitude. A combined advantage whose magnitude grows linearly in λ silently inflates the effective policy-gradient step size — the policy starts taking enormous, badly-scaled steps exactly when λ is large, which is exactly when I most need the update to be careful and stable. And the clip won't save me, because it clips the probability *ratio* r_t, not the advantage; a giant A still scales the per-sample gradient that the ratio multiplies. So λ has quietly leaked into two places at once — the relative weight of reward versus cost, which I *want*, and the numerical scale of the gradient samples, which I *don't*. Wall.

I don't want to change λ's *meaning* to fix this — the ratio of reward weight to cost weight, 1 : λ, is the thing the dual is carefully tuning, and I must preserve it. What I want is to peel apart those two effects: keep the *direction* of the combined advantage but normalize its *scale* so PPO sees a roughly unit-scale advantage regardless of how big λ has grown. The cleanest normalization that's monotone in λ and equals 1 at λ = 0 is dividing by (1 + λ):

  A = (A_r − λ A_c) / (1 + λ).

Check it does what I want. Rewrite by setting u = λ/(1+λ). Then 1/(1+λ) = 1 − u and λ/(1+λ) = u, so

  A = (1 − u) A_r − u A_c,  u = λ/(1+λ) ∈ [0, 1).

Now it reads as a *convex blend* of the reward advantage and the (negated) cost advantage. At λ = 0: u = 0, A = A_r — pure reward, the unconstrained PPO update, which is right because λ = 0 means the constraint is slack and I shouldn't be paying any safety tax. As λ → ∞: u → 1, A → −A_c — pure cost-avoidance, the agent does nothing but flee hazards, which is right because a huge λ means the constraint is being badly violated and reward should be sacrificed. And in between, since the two weights 1−u and u are both in [0,1] and sum to 1, |A| stays on the order of the individual advantages no matter how large λ gets — so PPO's tuned step size and clip behave the same throughout training. I want to be precise about what this does and does not change: the *ratio* of the reward weight to the cost weight is (1−u):u = 1:λ, identical to the un-normalized version, so the per-update gradient direction in policy space is unchanged and the stationary point of the primal-dual dynamics is unchanged. The (1+λ) factor is a learning-rate normalization that keeps the first-order update numerically sane; it is not a claim that the convex-blend form defines a new or better optimum. I've rescaled the step, not moved the target. That's the fix, and it's forced by the scale problem, not decorative.

There's a small implementation choice in how I take the λ step that I want to get right. Plain projected gradient ascent on λ is λ ← [λ + η(J_c − d)]_+. There are two honest ways to realize this in code. One is to parameterize the multiplier through a positive transform — store an unconstrained scalar and pass it through softplus so λ is nonnegative by construction — and train that underlying parameter. The other is to store λ directly as a parameter and enforce nonnegativity by projection. I'll take the second: keep λ as a learnable scalar, and to get its ascent step as a clean optimizer call, define a scalar loss whose gradient is exactly the *descent* direction I want,

  loss_λ = −λ · (J_c − d).

Then ∂loss_λ/∂λ = −(J_c − d), so a gradient-descent step λ ← λ − η·∂loss_λ/∂λ = λ + η(J_c − d) — exactly the dual ascent on λ I derived. Wrapping this in an optimizer (rather than a bare fixed-step update) also smooths the trajectory of the single dual scalar against the epoch-to-epoch noise in the Monte Carlo estimate of J_c. After the optimizer step I clamp λ.data to the nonnegative range, which is the projection. I use the direct-clamped multiplier rather than the softplus parameterization because the policy update reads λ directly through the advantage hook, and a directly-stored, directly-clamped scalar is the simplest thing that hook can consume.

Now the order of operations within an epoch matters a little. Each epoch: collect rollouts; read the mean episodic cost J_c from the logger (it's already averaged across parallel workers); update λ from J_c first; *then* run the PPO actor/critic update using the freshly-updated λ inside the combined advantage. Updating λ before the policy means the policy step already reflects the latest read on how unsafe we are. Then log λ so I can watch it climb when the agent is unsafe and relax when it's safe — that trajectory is the clearest diagnostic of whether the dual loop is doing its job.

I should name the one weakness of this loop, because it's real and it's intrinsic to the integral-only character of λ. λ is the *integral* of the violation: it only responds to *accumulated* over-budget, so it lags. When the agent suddenly starts violating, λ has to integrate up from wherever it was, which takes several epochs; by the time λ is large enough to rein the agent in, it has overshot — and then once cost drops below budget, λ keeps being large for a while (it integrates down equally slowly), over-correcting toward timidity, so cost undershoots, so λ falls, so cost rises again. The result is oscillation of cost around the budget rather than a clean settle. This is the generic behavior of a pure-integral controller on a system with delay, and it's the flip side of the appeal: it is a tiny change to a strong first-order optimizer, but because it only reacts after observed cost it can lag and oscillate. The integral term is exactly what makes λ converge to the right steady-state penalty with *zero steady-state violation*, which is why I want it — so I keep it as the canonical method and note that adding proportional and derivative terms on the violation is the obvious next refinement if the oscillation costs too much. It is also, deliberately, *not* a per-update feasibility solve in the trust-region style: it makes no promise that every intermediate policy is feasible, only that the dual loop drives the average toward the budget.

Let me write the whole thing as the subclass it actually is, filling exactly the two slots the harness left open — the per-epoch state-and-update, and the advantage combination — on top of the PPO base that already owns the clipped surrogate, the reward and cost value functions, GAE, and the rollout loop.

```python
import numpy as np
import torch


class Lagrange:
    """The dual variable and its update — the integral controller on the constraint."""

    def __init__(self, cost_limit, lagrangian_multiplier_init, lambda_lr, lambda_optimizer,
                 lagrangian_upper_bound=None):
        self.cost_limit = cost_limit                      # the budget d
        self.lambda_lr = lambda_lr                        # dual step size eta
        self.lagrangian_upper_bound = lagrangian_upper_bound  # optional cap on lambda
        # lambda as a learnable nonnegative scalar; start at the given init, floored at 0
        init_value = max(lagrangian_multiplier_init, 0.0)
        self.lagrangian_multiplier = torch.nn.Parameter(
            torch.as_tensor(init_value), requires_grad=True,
        )
        # run an optimizer on the single dual scalar to smooth its noisy trajectory
        torch_opt = getattr(torch.optim, lambda_optimizer)
        self.lambda_optimizer = torch_opt([self.lagrangian_multiplier], lr=lambda_lr)

    def compute_lambda_loss(self, mean_ep_cost):
        # loss whose gradient wrt lambda is -(Jc - d): descending it ascends lambda when over budget
        return -self.lagrangian_multiplier * (mean_ep_cost - self.cost_limit)

    def update_lagrange_multiplier(self, Jc):
        # one dual-ascent step then projection: lambda <- [lambda + eta (Jc - d)]_+
        self.lambda_optimizer.zero_grad()
        lambda_loss = self.compute_lambda_loss(Jc)
        lambda_loss.backward()
        self.lambda_optimizer.step()
        self.lagrangian_multiplier.data.clamp_(0.0, self.lagrangian_upper_bound)  # project to [0, inf)


class PPOLag(PPO):
    """PPO inner optimizer + a Lagrange-multiplier dual loop on the cost constraint."""

    def _init(self):
        super()._init()                                   # PPO: clipped surrogate, r/c critics, GAE
        self._lagrange = Lagrange(**self._cfgs.lagrange_cfgs)

    def _init_log(self):
        super()._init_log()
        self._logger.register_key('Metrics/LagrangeMultiplier', min_and_max=True)  # watch lambda

    def _update(self):
        # mean episodic cost Jc for this epoch (already averaged across workers)
        Jc = self._logger.get_stats('Metrics/EpCost')[0]
        assert not np.isnan(Jc), 'cost for updating lagrange multiplier is nan'
        self._lagrange.update_lagrange_multiplier(Jc)     # dual step first, from measured violation
        super()._update()                                 # then PPO actor/critic on combined advantage
        self._logger.store({'Metrics/LagrangeMultiplier': self._lagrange.lagrangian_multiplier})

    def _compute_adv_surrogate(self, adv_r, adv_c):
        # combined advantage = gradient of (J_r - lambda J_c), rescaled by 1/(1+lambda)
        # to keep PPO's effective step size stable as lambda grows: (1-u) A_r - u A_c, u = lambda/(1+lambda)
        penalty = self._lagrange.lagrangian_multiplier.item()
        return (adv_r - penalty * adv_c) / (1 + penalty)
```
