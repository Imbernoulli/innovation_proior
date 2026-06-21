I have an agent navigating a cluttered environment and I want two things that pull against each other: collect reward by reaching goals, and almost never touch the hazards. The hazards carry their own per-step indicator cost $c_t$ — $1$ when the agent does the unsafe thing, $0$ otherwise — that is genuinely separate from the reward, not a term I get to fold into $r_t$. So written honestly the specification is to maximize $\mathbb{E}[\sum_t r_t]$ while keeping $\mathbb{E}[\sum_t c_t]$ under a budget $d$ that a human picks because they know how much risk is acceptable. The reflex is to mash the two signals into one scalar and hand it to whatever policy optimizer I already trust — optimize $\mathbb{E}[\sum_t (r_t - \beta c_t)]$ for some fixed penalty $\beta$ — and every time I reach for it I hit the same wall. There is no formula taking my budget $d$ to the $\beta$ that achieves it; the map from penalty coefficient to resulting cost return depends on the whole environment and policy class in a way I cannot invert ahead of time. Guess $\beta$ too small and the agent decides a few hazards are worth the reward and sits above budget; guess too big and it becomes so frightened of cost it never learns the task. Even when some $\beta$ lands a safe optimum, it says nothing about cost *during* training, and worst of all that one number has to do two incompatible jobs across a run — stay small early so the still-incompetent policy can learn the task at all, then push hard once the policy finds rewarding-but-risky shortcuts. The constrained alternatives are heavier than I want: unconstrained TRPO/PPO ignore the budget by construction, and CPO holds the constraint per update but needs second-order Fisher-vector products, conjugate gradient, and a line search every step, with guarantees that rest on a trust-region linearization that can fail on hard high-dimensional tasks.

So I stop folding safety into the objective and keep it as what it is — a constraint — which gives the constrained MDP $\max_\theta J_r(\theta)$ s.t. $J_c(\theta) \le d$, with $J_r(\theta) = \mathbb{E}_{\tau\sim\pi_\theta}[\sum_t r_t]$ and $J_c(\theta) = \mathbb{E}_{\tau\sim\pi_\theta}[\sum_t c_t]$ defined identically except for which signal they sum. The constraint framing beats the penalty framing not aesthetically but structurally: a threshold encodes "get cost under $d$ and then stop caring about it; spend everything else on reward," which is the actual shape of a safety requirement, whereas a penalty keeps trading reward against cost forever even deep inside the safe region. The threshold has a saturation built in; the penalty does not. I propose PPO-Lagrangian: keep the entire PPO machinery — clipped surrogate, separate reward and cost critics, GAE run twice, the rollout loop — and wrap it in a single-scalar dual loop that adapts the cost penalty from the measured constraint violation.

The way to optimize a constrained objective when $J_r$ and $J_c$ are horrible nonconvex functions of a few hundred thousand weights is the Lagrangian. Writing the constraint in standard form $g(\theta) = J_c(\theta) - d \le 0$ and introducing a multiplier $\lambda \ge 0$, $$L(\theta, \lambda) = J_r(\theta) - \lambda\,(J_c(\theta) - d),$$ and the constrained problem equals $\max_\theta \min_{\lambda \ge 0} L(\theta, \lambda)$. I checked the inner min by behavior rather than trusting the formula: fix $\theta$; if the constraint is violated, $g > 0$, then $\min$ over $\lambda \ge 0$ of $-\lambda \cdot (\text{positive})$ drives $\lambda \to \infty$ and $L \to -\infty$, so infeasible policies are punished to $-\infty$; if it is satisfied with slack, $g < 0$, then $-\lambda \cdot (\text{negative})$ is increasing in $\lambda$, the min picks $\lambda = 0$ and $L = J_r(\theta)$, pure reward; on the boundary $g = 0$ the $\lambda$ term vanishes for every $\lambda$. So the inner min reconstructs exactly "reward if feasible, $-\infty$ if not," and at the saddle the multiplier obeys complementary slackness $\lambda\,g = 0$. The one worry is that this is the textbook convex story while $J_r, J_c$ are nonconvex in $\theta$, where a generic program has a duality gap that would aim dual methods at the wrong target. The rescue is specific to RL: the nonconvexity lives in the parameterization $\theta \mapsto \pi_\theta$, not in the problem, because in the space of state-action occupancy measures both returns are linear and the valid occupancy measures form a convex set cut out by the Bellman flow equations — a linear program, which has no duality gap. So the underlying constrained-RL problem has zero duality gap under the usual feasibility assumptions; dual descent is aimed at the correct saddle, and my neural update is an approximation of it, not a chase after a phantom.

The natural algorithm for a saddle point is primal-dual alternation — ascend $\theta$, descend $\lambda$, let them chase each other. The $\lambda$ update is where the adaptivity lives. I want to move $\lambda$ to decrease $L = J_r - \lambda\,g$, i.e. gradient descent on $\lambda$, whose gradient is $-g(\theta) = -(J_c - d)$; descending and projecting to keep $\lambda \ge 0$ gives $$\lambda \leftarrow \big[\,\lambda + \eta\,(J_c(\theta) - d)\,\big]_+.$$ Reading it by behavior rather than algebra, because a flipped sign would silently invert the whole safety mechanism: if the policy is over budget, $J_c > d$, the bracket increases, $\lambda$ climbs, the next update weighs cost more heavily and is pushed toward safety; if under budget, $J_c < d$, $\lambda$ falls toward zero and the policy is freed to chase reward with its slack. So $\lambda$ does the trade-off tuning for me, reading the actual measured violation each round — exactly the adaptive coefficient the fixed penalty could never produce. Structurally $\lambda$ is the *integral* of the violation: it accumulates the running sum of $(J_c - d)$ and settles at whatever penalty drives the average violation to zero. The projection is not cosmetic — an inequality multiplier must be nonnegative by KKT, and a negative $\lambda$ would mean paying the agent to incur cost, exactly backwards. One subtlety: the constraint is on the expected *episodic* cost return, not on any single step, so I update $\lambda$ once per epoch from the logged mean episodic cost across the trajectories just collected, and assert it is not NaN first so a bad cost statistic cannot silently corrupt $\lambda$.

For the policy I ascend $\theta$ on $L(\theta, \lambda) = J_r(\theta) - \lambda J_c(\theta)$ (the $\lambda d$ term is constant in $\theta$), whose gradient is $\nabla_\theta J_r - \lambda \nabla_\theta J_c$. PPO already ascends an expected return given an advantage, with $\nabla J = \mathbb{E}[\nabla \log\pi \cdot A]$, so $\nabla_\theta J_r$ corresponds to the reward advantage $A_r$ and $\nabla_\theta J_c$ to the cost advantage $A_c$ — I run GAE twice, on a reward value function and a cost value function. The combined gradient $\nabla_\theta(J_r - \lambda J_c)$ then corresponds to a single combined advantage $A = A_r - \lambda A_c$, fed straight into the PPO clipped surrogate $\min(r_t A,\, \mathrm{clip}(r_t, 1-\varepsilon, 1+\varepsilon)\,A)$ with importance ratio $r_t = \pi_\theta(a|s)/\pi_{\theta_{\text{old}}}(a|s)$. The clip does its usual trust-region job and is agnostic to what $A$ means.

Shipping $A = A_r - \lambda A_c$ raw breaks, though, and the failure is forced not decorative. When the agent keeps blowing the budget, $\lambda$ keeps climbing — to $5$, $10$, $50$ — and at $\lambda = 50$ the cost term dominates and the *scale* of $A$ is roughly $50\times$ a normal advantage. PPO's clip range, learning rate, and whole step-size calibration assume the advantage has its usual magnitude; a combined advantage whose magnitude grows linearly in $\lambda$ silently inflates the effective gradient step exactly when $\lambda$ is large, which is exactly when the update most needs to be careful. The clip does not save me because it clips the *ratio* $r_t$, not the advantage, and a giant $A$ still scales the per-sample gradient the ratio multiplies. So $\lambda$ has leaked into two places: the relative weight of reward versus cost, which I want, and the numerical scale of the gradient, which I do not. I refuse to change $\lambda$'s meaning, since the $1:\lambda$ reward-to-cost ratio is what the dual is carefully tuning; instead I peel the two effects apart by keeping the direction of the combined advantage but normalizing its scale. The cleanest normalization monotone in $\lambda$ and equal to $1$ at $\lambda = 0$ is dividing by $(1+\lambda)$: $$A_{\text{lag}} = \frac{A_r - \lambda A_c}{1 + \lambda}.$$ Setting $u = \lambda/(1+\lambda) \in [0,1)$, so $1/(1+\lambda) = 1 - u$, rewrites it as a convex blend $$A_{\text{lag}} = (1-u)\,A_r - u\,A_c.$$ At $\lambda = 0$, $u = 0$, $A_{\text{lag}} = A_r$ — pure reward, the unconstrained PPO update, right because the constraint is slack and no safety tax is owed. As $\lambda \to \infty$, $u \to 1$, $A_{\text{lag}} \to -A_c$ — pure hazard-avoidance, right because a huge $\lambda$ means the budget is badly violated and reward should be sacrificed. In between, the weights $1-u$ and $u$ lie in $[0,1]$ and sum to $1$, so $|A_{\text{lag}}|$ stays on the order of the individual advantages no matter how large $\lambda$ grows, and PPO's tuned step size and clip behave the same throughout training. Crucially the ratio of reward weight to cost weight is $(1-u):u = 1:\lambda$, identical to the un-normalized form, so the gradient *direction* in policy space and the stationary point of the primal-dual dynamics are unchanged — the $(1+\lambda)$ factor is a learning-rate normalization that keeps the first-order step numerically sane, not a claim of a new optimum. I rescaled the step, not moved the target.

A few implementation choices close it out. I realize the $\lambda$ ascent as a clean optimizer call by storing $\lambda$ as a learnable scalar and defining a loss whose gradient is the descent direction, $\text{loss}_\lambda = -\lambda\,(J_c - d)$, since $\partial\text{loss}_\lambda/\partial\lambda = -(J_c - d)$ makes a gradient-descent step $\lambda \leftarrow \lambda + \eta(J_c - d)$, exactly the dual ascent; wrapping it in an optimizer also smooths the single dual scalar against the epoch-to-epoch noise in the Monte Carlo estimate of $J_c$, and I clamp $\lambda$ to the nonnegative range afterward as the projection. I use this directly-clamped multiplier rather than a softplus parameterization because the advantage hook reads $\lambda$ directly. Within each epoch the order matters: collect rollouts, read mean episodic cost, update $\lambda$ first, then run the PPO actor/critic update with the freshly updated $\lambda$ inside the combined advantage, and log $\lambda$ so its rise and fall is the clearest diagnostic of whether the dual loop is working. The one intrinsic weakness, which I keep rather than hide, is that $\lambda$ is integral-only and therefore lags: it has to integrate up from wherever it was when violation starts, overshoots, then integrates down equally slowly and over-corrects toward timidity, so cost oscillates around the budget before settling — the generic behavior of a pure-integral controller on a delayed system. The integral term is exactly what gives zero steady-state violation, which is why I keep it canonical; proportional and derivative terms on the violation are the obvious next refinement. And deliberately this makes no per-update feasibility promise like CPO — only that the dual loop drives the average toward the budget.

```python
import numpy as np
import torch


class Lagrange:
    def __init__(
        self,
        cost_limit: float,
        lagrangian_multiplier_init: float,
        lambda_lr: float,
        lambda_optimizer: str,
        lagrangian_upper_bound: float | None = None,
    ) -> None:
        self.cost_limit = cost_limit
        self.lagrangian_upper_bound = lagrangian_upper_bound
        init_value = max(lagrangian_multiplier_init, 0.0)
        self.lagrangian_multiplier = torch.nn.Parameter(
            torch.as_tensor(init_value),
            requires_grad=True,
        )
        torch_opt = getattr(torch.optim, lambda_optimizer)
        self.lambda_optimizer = torch_opt([self.lagrangian_multiplier], lr=lambda_lr)

    def compute_lambda_loss(self, mean_ep_cost: float) -> torch.Tensor:
        return -self.lagrangian_multiplier * (mean_ep_cost - self.cost_limit)

    def update_lagrange_multiplier(self, Jc: float) -> None:
        self.lambda_optimizer.zero_grad()
        lambda_loss = self.compute_lambda_loss(Jc)
        lambda_loss.backward()
        self.lambda_optimizer.step()
        self.lagrangian_multiplier.data.clamp_(0.0, self.lagrangian_upper_bound)


class PPOLag(PPO):
    def _init(self) -> None:
        super()._init()
        self._lagrange = Lagrange(**self._cfgs.lagrange_cfgs)

    def _init_log(self) -> None:
        super()._init_log()
        self._logger.register_key("Metrics/LagrangeMultiplier", min_and_max=True)

    def _update(self) -> None:
        Jc = self._logger.get_stats("Metrics/EpCost")[0]
        assert not np.isnan(Jc), "cost for updating lagrange multiplier is nan"
        self._lagrange.update_lagrange_multiplier(Jc)
        super()._update()
        self._logger.store(
            {"Metrics/LagrangeMultiplier": self._lagrange.lagrangian_multiplier},
        )

    def _compute_adv_surrogate(
        self,
        adv_r: torch.Tensor,
        adv_c: torch.Tensor,
    ) -> torch.Tensor:
        penalty = self._lagrange.lagrangian_multiplier.item()
        return (adv_r - penalty * adv_c) / (1 + penalty)
```
