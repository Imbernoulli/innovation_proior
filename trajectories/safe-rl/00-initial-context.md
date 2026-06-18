## Research question

Safe reinforcement learning on Safety-Gymnasium navigation: an agent must maximize task reward while
holding a separate, long-run **cost** — the count of safety violations per episode, things like
entering a hazard or hitting an obstacle — below a fixed budget. The cost is its own signal, not a
term inside the reward; the specification is "earn reward, but keep expected episodic cost under
`d = 25`." The single thing being designed is the **constraint-handling mechanism**: how a scalar
multiplier `lambda` is updated from the measured cost violation, and how that multiplier blends the
reward advantage and the cost advantage into the one advantage PPO ascends. Everything else about the
agent — the rollout loop, the two value functions, the optimizer, the environment interface — is
fixed.

## Prior art before the first rung (constrained-RL lineage)

The framing the first rung sits inside — a constrained MDP solved by a dual multiplier — is itself the
resolution of a line of work on optimizing under a cost budget. These are the ancestors the ladder
reacts to; the fixed substrate below is what they converged to.

- **Constrained MDPs (Altman 1999).** Formalizes "maximize reward subject to a cost return staying
  under a threshold" as an MDP augmented with a cost function and a budget `d`, with the feasible set
  the policies sitting under the budget. The linear-programming view over state-action occupancy
  measures shows reward and cost returns are both *linear* in the occupancy measure and the feasible
  set is convex, so the underlying problem has zero duality gap. Gap: the LP view is exact only in the
  tabular/occupancy formulation; with a neural policy the objective is nonconvex in the parameters and
  one needs an iterative primal-dual scheme that only approximates the saddle point.
- **Penalty / reward-shaping baselines.** Fold safety into the reward as `r - beta*c` for a fixed
  penalty weight `beta`, then run any unconstrained RL method. Simple, but `beta` cannot be set from
  the budget `d` (the map from penalty weight to resulting cost return is environment- and
  policy-dependent and not invertible ahead of time), and one constant cannot be right both early
  (when a weak policy needs slack to learn the task) and late (when it has found rewarding-but-risky
  shortcuts). Gap: the trade-off is baked into a number that must be guessed before any data is seen.
- **Lagrangian / primal-dual methods (Arrow–Hurwicz; Platt & Barr 1987).** Dualize the constraint with
  a nonnegative multiplier `lambda`, form `L(theta, lambda) = J_r - lambda*(J_c - d)`, and solve the
  saddle point by alternating ascent on the policy and descent on `lambda`, growing `lambda` from the
  running violation `J_c - d`. Platt & Barr's "basic differential multiplier method" is exactly this
  in continuous time, `lambda_dot = alpha*(J_c - d)`. Gap: the multiplier is a pure *integral* of the
  violation, so it reacts only to accumulated over-budget, lags, and makes cost oscillate around the
  limit during training rather than settling — fine at convergence, but safe RL needs the budget
  respected *while learning*.
- **PPO (Schulman et al. 2017).** The inner unconstrained optimizer the substrate fixes: a clipped
  surrogate `min(r_t*A, clip(r_t, 1-eps, 1+eps)*A)` with `r_t = pi_theta/pi_old`, multi-epoch reuse of
  each batch inside a first-order trust region, GAE advantages, a value loss and entropy bonus. It is
  agnostic to what the advantage `A` means — reward, or reward-minus-cost — which is exactly the hook a
  constraint mechanism plugs into.

## The fixed substrate

A constrained-PPO loop is frozen and must not be touched. It owns: the Safety-Gymnasium environment
interface and vectorized rollout; **two** value functions, one for reward and one for cost, each with
its own GAE so the loop produces a reward advantage `adv_r` and a cost advantage `adv_c` per
transition; the PPO actor/critic update (clipped surrogate, value losses, Adam); the
`Metrics/EpCost` and `Metrics/EpRet` logging; and the registration plumbing. The agent's file is a
subclass of PPO registered as `CustomLag`. The base class already exposes the two slots the mechanism
fills and supplies the rest: `super()._update()` runs the PPO actor/critic step on whatever advantage
`_compute_adv_surrogate` returns; `self._logger.get_stats('Metrics/EpCost')[0]` returns the current
mean episodic cost (already averaged across workers); `self._cfgs.lagrange_cfgs.cost_limit` is the
budget `d = 25.0`; `self._cfgs.lagrange_cfgs.lambda_lr` is the dual step size (default `0.035`).

## The editable interface

Exactly one region is editable — the constraint-handling methods of the `CustomLag` subclass in
`custom_lag.py`: the imports line, and the block of methods `_init`, `_init_log`, `_update`, and
`_compute_adv_surrogate`. Every method on the ladder is a fill of this same contract. `_init` sets up
the multiplier state (after `super()._init()`); `_init_log` registers the `LagrangeMultiplier` key
(after `super()._init_log()`); `_update` reads the measured mean episodic cost `Jc`, updates the
multiplier from it, then calls `super()._update()` to run the PPO step and logs the multiplier; and
`_compute_adv_surrogate(adv_r, adv_c)` returns the single advantage PPO ascends, the place the
multiplier blends reward against cost.

The starting point is the scaffold default: **no constraint handling**. The multiplier stays at zero
and `_compute_adv_surrogate` returns `adv_r` unchanged — pure reward-maximizing PPO that ignores
safety. Each method replaces exactly these definitions and nothing else.

```python
# EDITABLE region of custom_lag.py -- default fill (no constraint handling)

# (no extra imports)

@registry.register
class CustomLag(PPO):
    """Custom Lagrangian-based safe RL algorithm."""

    def _init(self) -> None:
        super()._init()
        self._cost_limit: float = self._cfgs.lagrange_cfgs.cost_limit
        self._lagrangian_multiplier: float = 0.0

    def _init_log(self) -> None:
        super()._init_log()
        self._logger.register_key('Metrics/LagrangeMultiplier', min_and_max=True)

    def _update(self) -> None:
        Jc = self._logger.get_stats('Metrics/EpCost')[0]
        assert not np.isnan(Jc), 'cost is nan'
        # Default: no multiplier update -- the mechanism is what gets designed
        super()._update()
        self._logger.store({'Metrics/LagrangeMultiplier': self._lagrangian_multiplier})

    def _compute_adv_surrogate(self, adv_r: torch.Tensor, adv_c: torch.Tensor) -> torch.Tensor:
        """Default: only use reward advantage (ignores safety constraints entirely)."""
        return adv_r
```

## Evaluation settings

Three Safety-Gymnasium navigation environments spanning the difficulty range —
**SafetyPointGoal1-v0** (point robot reaching goals while avoiding hazards), **SafetyCarGoal1-v0**
(a non-holonomic car robot, same goal structure, harder dynamics), and **SafetyPointButton1-v0**
(point robot pressing goal buttons amid hazards) — each over three seeds {42, 123, 456}, each trained
for the benchmark's fixed step budget. Two metrics per environment: episode return (`reward`, higher
is better) and episode cost (`cost`, lower is better, with a target threshold of `25.0` per the
Safety-Gymnasium convention). A method is good only when it earns return *and* holds cost at or below
the budget across all three environments — high return with an uncontrolled cost is a failure of the
actual objective, not a success.
