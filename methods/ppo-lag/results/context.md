# Context

## Research question

A reinforcement-learning agent is being trained in an environment where task success and unsafe behavior are measured by different signals. The reward `r_t` measures progress on the task. A separate cost `c_t` records unsafe events such as entering a hazardous region, colliding, or disturbing protected objects. The desired specification is not "maximize a hand-shaped reward" but

```text
maximize    J_r(pi) = E_pi[sum_t r_t]
subject to  J_c(pi) = E_pi[sum_t c_t] <= d,
```

where `d` is a human-chosen cost budget. A useful optimizer must adjust the reward-safety tradeoff from data, because the correct tradeoff coefficient is not known before training and may need to change as the policy improves.

## Background

The standard formalism is the constrained Markov decision process (CMDP), introduced by Altman (1999, *Constrained Markov Decision Processes*). A CMDP augments an ordinary MDP with one or more cost functions `c_1,...,c_k`, each defining a constraint function `J_ci(pi)` — an expected cumulative cost, defined exactly like an expected return but on the cost rather than the reward — and a threshold `d_i`. The feasible set is the set of policies whose cost returns sit under their thresholds, `Pi_C = { pi : J_ci(pi) <= d_i }`, and an optimal constrained policy is `pi* = argmax_{pi in Pi_C} J_r(pi)`. This keeps "what the agent should do" and "what the agent must avoid" as distinct design objects, and the framework is expressive: chance constraints, conditional-value-at-risk constraints, and per-state constraints can all be cast this way, and a single aggregate indicator cost can fold many distinct hazards into one constraint.

A key fact about CMDPs concerns their behavior under duality. Although the return `J_r(pi)` and the cost return `J_c(pi)` are nonconvex functions of the policy parameters, the constrained-RL problem, viewed in the space of state-action *occupancy measures* (equivalently, over the convex set of stationary policies, as in Altman's linear-programming treatment), is linear in the occupancy measure over a convex feasible set defined by the Bellman flow constraints — a linear program. Linear programs have no duality gap. So the underlying constrained-RL problem has zero duality gap under the usual feasibility assumptions (Paternain et al. 2019, "Constrained Reinforcement Learning Has Zero Duality Gap"), which is the theoretical reason Lagrangian and dual methods are natural here. The caveat is that this clean duality lives in occupancy-measure space, while practical neural policy optimization works on nonconvex parameters and remains an approximate procedure.

On the optimization side, the field has a mature pair of on-policy policy-gradient methods for unconstrained continuous control. *Trust Region Policy Optimization* (Schulman et al. 2015) maximizes a surrogate advantage objective subject to a hard KL-divergence trust region between old and new policy, with a second-order conjugate-gradient update. *Proximal Policy Optimization* (Schulman et al. 2017) replaces the hard KL constraint with a clipped importance-sampling surrogate,

```text
L^CLIP(theta) = E_t[ min( r_t(theta) A_t,  clip(r_t(theta), 1-eps, 1+eps) A_t ) ],
                r_t(theta) = pi_theta(a_t|s_t) / pi_theta_old(a_t|s_t),
```

achieving most of TRPO's stability with plain first-order minibatch SGD. Both rely on an advantage estimate `A_t`; *generalized advantage estimation* (Schulman et al. 2016) supplies the bias/variance-tunable estimator used in practice, and it can be run independently on a reward value function and a cost value function to produce separate reward and cost advantages.

The motivating empirical picture, on high-dimensional navigation tasks with separate reward and cost signals, is that reward and cost trade off meaningfully: an unconstrained agent scores high return precisely by taking unsafe actions, and denser hazard layouts produce systematically higher episodic cost.

## Baselines

**Fixed-penalty scalarization.** The simplest constraint-aware approach folds a penalty into the reward: optimize `E[sum_t (r_t - beta c_t)]` with a fixed coefficient `beta`. Its core idea is to reduce the constrained problem to an ordinary unconstrained one.

**Unconstrained TRPO / PPO.** Run the policy optimizer on the reward only, ignoring cost. These set an upper reference on achievable return and confirm the environments are not trivially safe.

**Constrained Policy Optimization (CPO; Achiam et al. 2017).** CPO works in TRPO's trust-region framework but adds the cost constraint directly to each update: it solves a small constrained quadratic program in the trust region, linearizing both the reward surrogate and the cost surrogate and computing analytically, from scratch, the update (and an effective penalty coefficient) that maximizes reward improvement while keeping the linearized cost within budget; when the current policy is infeasible it takes a recovery step toward feasibility. Its core idea is per-update, theoretically grounded near-constraint-satisfaction, and it requires second-order machinery (Fisher-vector products, conjugate gradient, a backtracking line search) every update.

## Evaluation settings

The natural yardstick is a suite of high-dimensional continuous-control safe-exploration tasks built on the MuJoCo physics simulator and the Gym API: a robot (a planar Point robot, a wheeled Car, a quadruped) must accomplish a task (reach goal positions; press the correct goal button; push a box to a goal) while interacting safely with hazard elements, with the layout randomized at the start of every episode so the agent must generalize, and difficulty levels varying the density of unsafe elements. Each environment exposes a separate reward signal and a separate aggregate indicator cost. Representative environments include `SafetyPointGoal1`, `SafetyCarGoal1`, and `SafetyPointButton1`. The optimization problem is finite-horizon and undiscounted for reporting:

```text
maximize    E_pi[sum_{t=0}^T r_t]
subject to  E_pi[sum_{t=0}^T c_t] <= d.
```

The main metrics are average episodic return `J_r`, average episodic cost `J_c`, and cost rate over the whole training run. A common benchmark-style setup uses cost limit `d = 25`, episode length `T_ep = 1000`, and therefore a training cost-rate target `d/T_ep = 0.025`. The comparison rule prioritizes satisfying the constraint first; among constraint-satisfying agents, higher return and lower training cost rate are better.

## Code framework

The existing optimizer already collects on-policy rollouts, computes reward and cost advantages, maintains reward and cost value functions, and performs PPO's clipped policy update. The extension point is deliberately small:

```python
class CustomConstraint(PPO):
    def _init(self) -> None:
        super()._init()
        # TODO: initialize constraint-handling state.

    def _update(self) -> None:
        Jc = self._logger.get_stats('Metrics/EpCost')[0]
        # TODO: update constraint-handling state from measured episodic cost.
        super()._update()

    def _compute_adv_surrogate(self, adv_r, adv_c):
        # TODO: combine reward and cost advantages into one clipped-surrogate advantage.
        raise NotImplementedError
```

The expected solution should fill those two slots: update a small amount of per-epoch constraint state from the measured mean episodic cost, then turn `adv_r` and `adv_c` into one PPO-compatible advantage tensor.
