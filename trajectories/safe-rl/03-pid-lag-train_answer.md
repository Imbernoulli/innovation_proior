The dual loop did the thing it was supposed to, and it did it incompletely, and the numbers say exactly where it fell short. Cost came down everywhere relative to naive — PointGoal $51 \to 45.6$, CarGoal $61 \to 46.7$, PointButton $152 \to 56.2$ — so the sign is right, the blend is right, $\lambda$ is reading the violation and paying for it. The price was steep on return, as expected: $25.5 \to 15.1$, $32.8 \to 18.6$, $19.7 \to 4.0$. But look at the cost against the budget: $45.6$, $46.7$, $56.2$ — still nearly twice over $25$ on every environment and seed. It paid heavily in reward and *still did not satisfy the constraint*: the worst of both worlds, the reward of a safe method without the safety. This is exactly the integral-only lag I flagged when I built the dual loop, now visible in the data. As the integral of the violation, $\lambda$ climbs only at its accumulation rate, and on these runs it simply has not integrated up to the value that would push cost to $25$ before training ends. I will not fix it by cranking `lambda_lr`, because a faster integrator on a delayed plant does not track better — it oscillates harder. The right move is not to tune the one knob but to ask what *kind* of update rule $\lambda$ should obey.

That question opens once I read the whole training loop as the feedback control system it actually is. The cost limit $d = 25$ is the *setpoint*. The measured mean episodic cost $J_c$ — the $45.6/46.7/56.2$ off the logger — is the *output* I want held at the setpoint. The multiplier $\lambda$ is the *control input* I get to set. And the policy-optimization step — PPO grinding the policy forward under the current $\lambda$ — is the *plant*: a nonlinear, frankly unknown map from $\lambda$ to next epoch's cost. The dual update I shipped, $\lambda \leftarrow [\lambda + K_I (J_c - d)]_+$, is one impoverished choice: *integral-only* control. Integral control has memory but no reflexes — it reacts only to *accumulated* error, so $\lambda$ rises only after violation has been piling up, and the response is intrinsically late. Platt & Barr's continuous-time form $\dot\lambda = \alpha\,(J_c - d)$ makes it precise: the multiplier is literally the time-integral of the constraint error, and a pure integrator shifts a sinusoid at its input by ninety degrees at its output — so when the loop cycles, violation and $\lambda$ are a quarter cycle apart, the textbook fingerprint. My runs are stuck in the slow approach before even reaching that limit cycle.

I propose the **PID Lagrangian** (CPPOPID): replace the integral-only dual update with a full proportional–integral–derivative controller on the cost-constraint error. Let me bring in the two new terms and check, in this same dynamical-systems picture, that each does what I hope.

Proportional first: add to $\lambda$ a term on the *current* error, $K_P (J_c - d)$, reacting instantly with no wait for accumulation. In continuous time it enters as $\dot\lambda = \alpha g + \beta \dot g$ with $g = J_c - d$. Redoing Platt & Barr's collapse of the coupled first-order system into one second-order equation per coordinate, the integral-only case gives a damped oscillator $\ddot x + A\dot x + \alpha g\,\nabla g = 0$; the $\beta\dot g$ piece changes only the damping matrix, which gains $\beta\,(\nabla g)(\nabla g)^\top$. That outer product is positive semidefinite by construction — $v^\top (\nabla g\,\nabla g^\top) v = (\nabla g \cdot v)^2 \ge 0$ — so it can only *add* nonnegative damping in the $\nabla g$ direction; the restoring force $\alpha g\,\nabla g$ is untouched, so the equilibrium does not move. Proportional is pure damping, exactly what kills an oscillation. Satisfyingly, this $\beta\,(\nabla g)(\nabla g)^\top$ is the same damping the quadratic-penalty method produces from adding $(c/2)g^2$ — but the penalty pays for it twice: it must modify the *primal* update (the penalty force lives in the policy dynamics), and its damping comes bundled with a $c\,g\,\nabla^2 g$ term carrying the constraint Hessian, which need not be positive semidefinite. Proportional control gives the clean positive-semidefinite damping touching only the $\lambda$ update — and it addresses my plateau directly, reacting to the current $45$-vs-$25$ gap immediately instead of waiting for the integrator.

Derivative next: proportional reacts to the present, the derivative term lets me *anticipate* — if cost is rising fast toward the limit, raise $\lambda$ now, before cost crosses $d$. A term on the trend enters one derivative higher, $\dot\lambda = \alpha g + \gamma \ddot g$. Grinding out $\ddot g$ it contains $\ddot x$, so acceleration appears on both sides; collecting, the $\ddot x$ terms couple through $B = I + \gamma\,(\nabla g)(\nabla g)^\top$ — identity plus a positive-semidefinite outer product, hence positive definite and invertible. Left-multiplying by $B^{-1}$ to decouple, two effects appear: $B^{-1}$ has eigenvalues no greater than one (strictly smaller along $\nabla g$), reshaping the damping operator to $B^{-1}A$; and a velocity-quadratic force modulated by the constraint curvature along the direction of motion multiplies $B^{-1}\nabla g$, whose sign is such that when $g > 0$ and the motion is into a region where $g$ curves upward — violation about to worsen — this anticipatory force adds to the restoring force and brakes harder before overshooting. The derivative is also the most delicate term, prone to amplifying noise because second differences of a noisy signal are wild. And for a one-sided (inequality) constraint I bake in an asymmetry: derivative control should *brake* cost increases (the dangerous direction) but stay silent on decreases (a falling cost is good), so I rectify it to the positive part of the cost-increase rate.

The full controller is the combination: proportional damps, derivative anticipates, integral does the one thing neither can. I keep integral because at convergence I need *zero* steady-state violation, and only the integral term supplies the standing $\lambda$ that holds cost exactly at $25$. When the system settles, $J_c - d \approx 0$, so the proportional term vanishes and the derivative term (cost not changing) vanishes too; if those were my only terms $\lambda$ would collapse and cost drift back up. The integral has *remembered* the accumulated history and holds a nonzero $\lambda$ even at zero instantaneous error. Integral eliminates steady-state offset; proportional and derivative shape the *transient* — the very thing the dual-loop plateau is failing at. And setting $K_P = K_D = 0$ recovers the integral-only dual update exactly, so this is a strict generalization: I am widening the update rule from a one-knob family to a three-knob one, with the prior baseline at the origin of the two new axes.

The discrete per-epoch rule I ship reads $J_c$ from the same logger and forms the error $\delta = J_c - d$. The integral is the running accumulation with its gain folded in, projected nonnegative for anti-windup — I do not want it banking a negative reservoir during a feasible stretch, which would delay its reaction, the exact disease I am curing: $I \leftarrow \max(0,\ I + K_I\,\delta)$. The proportional input is the error smoothed against the noisy minibatch estimate (the dual-loop PointButton cost ranged $51$–$60$) via a long-window EMA, $\delta_p \leftarrow 0.95\,\delta_p + 0.05\,\delta$, so one bad epoch cannot jerk the controller. The derivative works off a smoothed cost, $\text{cost}_d \leftarrow 0.95\,\text{cost}_d + 0.05\,J_c$, and rather than a jittery one-step difference takes a *delayed* finite difference over a short queue, rectified: $\text{pid}_d = \max(0,\ \text{cost}_d - \text{cost}_{ds}[\text{oldest}])$. The output is $\text{pid}_o = K_P\,\delta_p + I + K_D\,\text{pid}_d$ (with $I$ already carrying its gain), and $\lambda = \max(0,\ \text{pid}_o)$ for the KKT nonnegativity, then I push $\text{cost}_d$ onto the queue to advance the delay. The CPPOPID configuration fixes the gains at $K_P = 0.1$, $K_I = 0.01$, $K_D = 0.01$, the EMA coefficient at $0.95$, the delay at $10$ epochs, and seeds the integral at $0.0$ so $\lambda$ starts at zero like the dual loop. The gains are deliberately small: the integral is the slow memory that fixes steady-state, while proportional and derivative give fast *shaped* responses on top, so I get responsiveness without pushing $K_I$ into the regime that ruins reward. I deliberately do not implement the general controller's sum-norm / diff-norm modes or penalty cap — this harness exposes none of them.

The advantage blend is unchanged from the dual loop, and deliberately so: the large-$\lambda$ step-size blowup is the same problem regardless of how $\lambda$ is computed, and $\arg\max (J_r - \lambda J_c) = \arg\max (J_r - \lambda J_c)/(1 + \lambda)$ since dividing by the positive constant does not move the argmax but normalizes the gradient magnitude. So `_compute_adv_surrogate` returns $(\text{adv}_r - \lambda\,\text{adv}_c)/(1 + \lambda)$ exactly as before — using the identical blend is what makes this a clean comparison: the only thing that changed is the controller on $\lambda$, so any difference is attributable to PID versus integral-only. The substrate's two separate value functions are even more important here than before: because $\lambda$ now changes *rapidly* under the proportional and derivative terms, a single critic trained on a moving $r + \lambda c$ target would be perpetually stale, whereas the fixed-target reward and cost critics stay valid and are blended only at the policy-gradient stage by the current $\lambda$.

My falsifiable expectation against the dual-loop numbers: the PID controller should *actually reach the budget*. I expect cost at or below $25$ on all three environments — PointButton from $56$ into the mid-$20$s or below (the case where the slow integrator was most visibly stuck), the others from the mid-$40$s under $25$ — because proportional and derivative react to the current gap and the rising trend immediately while integral still guarantees no overshoot into permanent timidity. Return is the honest price of finally satisfying the constraint: it need not recover toward naive and may drop below the dual-loop $15/19/4$, because holding cost at $25$ (rather than hovering at $46$) forces detours around hazards the agent was cutting through. If cost lands controlled at the budget across all three environments — the thing neither naive nor the integral-only loop achieved — then on the actual safe-RL objective, high return *only when* the constraint is controlled, this is the first method on the ladder that succeeds.

```python
# EDITABLE region of custom_lag.py -- step 3: PID Lagrangian (CPPOPID)

from collections import deque

@registry.register
class CustomLag(PPO):
    """PID Lagrangian: a PID controller on the dual multiplier (CPPOPID defaults)."""

    def _init(self) -> None:
        super()._init()
        self._cost_limit: float = self._cfgs.lagrange_cfgs.cost_limit
        # PID controller gains (CPPOPID defaults)
        self._pid_kp: float = 0.1
        self._pid_ki: float = 0.01
        self._pid_kd: float = 0.01
        # PID state
        self._pid_i: float = 0.0
        self._delta_p: float = 0.0
        self._cost_d: float = 0.0
        self._cost_ds: deque = deque(maxlen=10)
        self._cost_ds.append(0.0)
        self._lagrangian_multiplier: float = 0.0

    def _init_log(self) -> None:
        super()._init_log()
        self._logger.register_key('Metrics/LagrangeMultiplier', min_and_max=True)

    def _update(self) -> None:
        Jc = self._logger.get_stats('Metrics/EpCost')[0]
        assert not np.isnan(Jc), 'cost is nan'
        # PID update
        delta = float(Jc - self._cost_limit)
        self._pid_i = max(0.0, self._pid_i + delta * self._pid_ki)
        self._delta_p = 0.95 * self._delta_p + 0.05 * delta
        self._cost_d = 0.95 * self._cost_d + 0.05 * float(Jc)
        pid_d = max(0.0, self._cost_d - self._cost_ds[0])
        pid_o = self._pid_kp * self._delta_p + self._pid_i + self._pid_kd * pid_d
        self._lagrangian_multiplier = max(0.0, pid_o)
        self._cost_ds.append(self._cost_d)
        super()._update()
        self._logger.store({'Metrics/LagrangeMultiplier': self._lagrangian_multiplier})

    def _compute_adv_surrogate(self, adv_r: torch.Tensor, adv_c: torch.Tensor) -> torch.Tensor:
        """PID Lagrangian: combine advantages using the PID-controlled multiplier."""
        penalty = self._lagrangian_multiplier
        return (adv_r - penalty * adv_c) / (1 + penalty)
```
