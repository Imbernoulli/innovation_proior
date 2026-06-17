Let me start from what actually goes wrong when I train a safe agent. I have a constrained MDP: maximize reward `J(pi)` subject to a cost staying under a limit, `J_C(pi) <= d`. The standard move is to dualize — introduce a nonnegative multiplier `lambda`, optimize `J - lambda(J_C - d)`, and grow `lambda` from the running violation `J_C - d`. The equation is clean, and at convergence it gives a feasible policy. But I don't care only about convergence. In safe RL the agent has to respect the budget *while it learns*, and when I watch a real run the cost doesn't sit politely at the limit — it swings above and below `d`, sometimes wildly, for the whole run. So the dual solver is "correct" and still useless for my actual requirement, which is constraint satisfaction at every iterate, not just in the limit.

So I need to understand *why* it oscillates, not just damp it by fiddling with the learning rate. Let me look at the exact update the Lagrangian method uses for an inequality constraint. It is a projected step on the measured violation:

  lambda_{k+1} = ( lambda_k + K_I (J_C - d) )_+.

Stare at that. `lambda` is a running accumulation of the violation `J_C - d`: every iteration I add the current error to a stored quantity. The continuous-time version Platt and Barr wrote down makes this even starker — `lambda_dot = alpha g(x)`, with `g` the constraint. The multiplier is literally the time-integral of the constraint error. And I notice the empirical signature in my runs: the cost curve and the `lambda` curve oscillate, but offset — when cost peaks, `lambda` is still rising; `lambda` peaks about a quarter cycle after cost does. A quarter-cycle lag between a system's input and its response.

That lag is the clue, and it's pointing somewhere specific. A pure integrator turns a sinusoid at its input into a sinusoid at its output shifted by exactly ninety degrees — that's just what integration does to `sin` (it gives `-cos`). So the quarter-cycle offset I'm seeing between violation and `lambda` is not some quirk of deep RL; it is the textbook fingerprint of an integral controller, observed in the wild. Which reframes the whole picture: the safe-RL training loop *is* a feedback control system. The cost limit `d` is a setpoint. The measured episodic cost `J_C` is the output I want to hold at the setpoint. The multiplier `lambda` is the control input I get to set. And the policy-optimization step — PPO grinding `theta` forward to chase reward — is the plant, some nonlinear, frankly unknown map from `lambda` to next iteration's cost. Write it as a discrete system: `theta_{k+1} = F(theta_k, lambda_k)`, `y_k = J_C(pi_{theta_k})`, `lambda_k = h(y_0, ..., y_k, d)`. The traditional method is one specific, impoverished choice of the control rule `h`: integral-only control. Once I see it that way, the question stops being "how do I tune the Lagrangian update" and becomes "what control rule `h` should I use," and a century of control theory has opinions about that.

Before I reach for richer controllers, I want to be sure I understand mechanically why integral-only control oscillates and overshoots, because the cure has to attack the cause. Integral control has memory but no reflexes. It only reacts to *accumulated* error — `lambda` rises only after violation has been piling up for a while. So the response is intrinsically late (there's the ninety-degree lag again). By the time `lambda` has integrated up to a value large enough to push cost back down, cost has already shot well past `d`. Now cost is falling, but `lambda` is still being fed a positive error for a while (cost is still above `d`), so `lambda` keeps climbing, overcorrects, drives cost below `d`, then the error goes negative, `lambda` finally comes down, cost rebounds — a limit cycle. And there's a structural lever Platt and Barr already identified: turning their first-order coupled system into one second-order equation per coordinate,

  x_ddot + A x_dot + alpha g(x) grad g = 0,

a damped oscillator with restoring force `alpha g grad g` toward the constraint surface and damping matrix `A = grad^2 f + lambda grad^2 g`. They proved convergence via the energy `E = (1/2) sum x_dot^2 + (alpha/2) g^2`, whose derivative is `E_dot = - x_dot^T A x_dot`, dissipative exactly when `A` is positive definite. And they noted: crank the gain `alpha` up and the oscillation frequency rises and the settling time gets *worse*. So in this picture my only knob, `K_I` (their `alpha`), controls the integral gain, and pushing it doesn't buy me clean tracking — it buys me faster, longer-ringing oscillation, with reward degrading as a side effect. One knob, bad trade. That matches the runs.

What does a controls person reach for when an integral loop rings? The other two terms of the most-used controller there is: proportional and derivative. Let me bring them in one at a time and check, in this same dynamical-systems picture, whether they actually do what I hope rather than just sounding good.

Proportional first. The idea is to add to `lambda` a term proportional to the *current* error, `K_P (J_C - d)` — an instantaneous reaction, no waiting for accumulation. To see its effect on the dynamics I have to express it in continuous time. The integral term was `lambda_dot = alpha g`. The current value of the constraint `g` is what gets multiplied by `K_P` in the proportional term — but in the differential equation for `lambda`, "a term proportional to `g`" has to enter as a contribution to `lambda_dot` that *tracks* `g`, and the natural way for `lambda` to instantaneously follow `g` is for its rate to carry `g`'s rate. So the proportional term appears in `lambda_dot` as the time-derivative of the constraint:

  lambda_dot = alpha g(x) + beta g_dot(x) = alpha g(x) + beta sum_j (dg/dx_j) x_dot_j.

Now I redo Platt and Barr's collapse with this extra piece and see what it does to the second-order equation. Differentiating `x_i_dot = -df/dx_i - lambda dg/dx_i` in time, I get `x_i_ddot = -d/dt(df/dx_i) - lambda_dot (dg/dx_i) - lambda d/dt(dg/dx_i)`. The chain rule gives `d/dt(df/dx_i) = sum_j (d^2 f / dx_i dx_j) x_dot_j` and likewise for `g`. Substituting `lambda_dot = alpha g + beta sum_j (dg/dx_j) x_dot_j`, the new beta piece contributes `- beta (sum_j (dg/dx_j) x_dot_j)(dg/dx_i)`, and gathering terms:

  x_i_ddot = - sum_j ( d^2f/dx_i dx_j + lambda d^2g/dx_i dx_j + beta (dg/dx_i)(dg/dx_j) ) x_dot_j - alpha g (dg/dx_i).

In vector form,

  x_ddot + ( A + beta grad g grad^T g ) x_dot + alpha g(x) grad g = 0.

The restoring force `alpha g grad g` is untouched — so the proportional term doesn't move the equilibrium, the solution set is still `x_dot = 0, g = 0`. What changed is the damping matrix: it gained `beta grad g grad^T g`. That term is the outer product of the vector `grad g` with itself, which is positive semidefinite by construction — `v^T (grad g grad^T g) v = (grad g . v)^2 >= 0` for any `v`. So it can only add nonnegative damping in the `grad g` direction. More damping is exactly what kills an oscillation. The proportional term provides damping. That's not an analogy; it falls straight out of the same energy argument: now the dissipation term is `-x_dot^T (A + beta grad g grad^T g) x_dot`, so the new piece subtracts `beta (grad g . x_dot)^2` from the energy derivative.

And there's something satisfying here, because I've seen this exact `beta grad g grad^T g` term before — it's what the quadratic penalty method produces. Adding `(c/2) g^2` to the objective injects a force `-c g grad g` into the primal dynamics and, when you redo the collapse, hands you a damping matrix `A + c grad g grad^T g + c g grad^2 g`. So the penalty method's damping benefit is real and it's the *same* outer-product term. But the penalty pays for it twice over. First, it had to modify the primal update `x_dot` itself — the penalty force lives in the `x` dynamics, an extra thing to implement and a change to the objective the policy is optimizing. My proportional term does nothing to `x_dot` or to the Lagrangian; it only touches the `lambda` update. Second, the penalty's damping comes bundled with that extra `c g grad^2 g` term, which carries a Hessian of the constraint `grad^2 g` that need not be positive semidefinite — so the penalty can simultaneously help (via the outer product) and hurt (via the indefinite Hessian), and both share the single coefficient `c`, so I can't dial in the good without the bad. The proportional control term gives me the clean positive-semidefinite damping with none of that baggage. Good — proportional control isn't just a heuristic transplant from controls; in this system it is a cleaner way to get the penalty method's known benefit.

Now derivative. The proportional term reacts to the present; the derivative term should let me react to the *future* — anticipate. The intuition is: if cost is rising fast toward the limit, raise `lambda` now, before cost actually crosses `d`. So I want a term on the rate of change of cost. In `lambda`'s differential equation a term on `g_dot` would be proportional control, which I've used; a term that acts on the *trend* of the constraint enters `lambda_dot` one derivative higher — as the second derivative of the constraint:

  lambda_dot = alpha g(x) + gamma g_ddot(x).

Let me grind out `g_ddot`. `g_dot = sum_j (dg/dx_j) x_dot_j`, so

  g_ddot = d/dt( sum_j (dg/dx_j) x_dot_j ) = sum_j ( x_dot_j sum_k (d^2g/dx_j dx_k) x_dot_k + (dg/dx_j) x_ddot_j ).

There's the complication: `g_ddot` contains `x_ddot` — the acceleration appears on both sides once I substitute back. Pushing `lambda_dot` into `x_i_ddot` and collecting,

  x_ddot + A x_dot + alpha g grad g + gamma grad g ( x_dot^T grad^2 g x_dot ) + gamma grad g grad^T g x_ddot = 0.

The `x_ddot` terms couple through the matrix `B = I + gamma grad g grad^T g`, which is identity plus a positive-semidefinite outer product, hence positive definite and invertible. Left-multiply by `B^{-1}` to decouple the acceleration:

  x_ddot + B^{-1} A x_dot + ( alpha g(x) + gamma x_dot^T grad^2 g x_dot ) B^{-1} grad g = 0.

Now I can read off what the derivative term does, and it's genuinely different from proportional. Two effects. First, `B^{-1}` has eigenvalues no greater than one (strictly smaller in the `grad g` direction when `gamma > 0`, unchanged in directions orthogonal to `grad g`). It scales the restoring direction and changes the damping operator to `B^{-1} A`, which can redirect the damping vector whenever `A x_dot` is not already aligned with an eigenvector of `B`. Second, and this is the predictive part, the scalar `gamma x_dot^T grad^2 g x_dot` now multiplies `B^{-1} grad g` — a force quadratic in the velocity, modulated by the curvature of the constraint along the direction of motion. In the acceleration equation the sign is negative, so picture moving along a direction where `g` curves upward (`x_dot^T grad^2 g x_dot > 0`): the new term pushes along `-B^{-1} grad g`, decreasing `g`. If at that moment `g > 0` too — I'm already in violation and the curvature says it's about to get worse — then this anticipatory force adds to the ordinary restoring force; I brake harder before overshooting. If instead `g` curves downward along my velocity, the new force points the other way, partly canceling the restoring force, easing off. So the derivative term reads the *trend* and acts ahead of the violation — exactly the anticipation I wanted. It's also more delicate: it's the term most prone to amplifying noise and destabilizing, because second derivatives of a noisy signal are wild. I'll keep that flag for when I get to estimation.

One asymmetry I want to bake in for the cost case specifically. Derivative control on cost should *brake* increases in cost — that's the dangerous direction, toward violation. But when cost is *decreasing*, I don't want the derivative term fighting that; a falling cost is good, let it fall. So I should rectify the derivative term: use `(rate of cost increase)_+`, the positive part, so it acts against increases and stays silent on decreases. That's a one-sided derivative, appropriate because the constraint is one-sided (an inequality, not an equality).

So the full controller is the combination: proportional damps, derivative anticipates, integral does the one thing neither of the others can. Why keep integral at all? Because at convergence I need *zero* steady-state violation, and only the integral term can supply the standing value of `lambda` that holds cost exactly at `d`. When the system has settled, the error `J_C - d` is ~0, so the proportional term `K_P (J_C - d)` vanishes and the derivative term (cost not changing) vanishes too — if those were my only terms, `lambda` would collapse to zero and cost would drift back up. The integral term has *remembered* the accumulated history, so it holds a nonzero `lambda` even when the instantaneous error is zero. Integral eliminates steady-state offset; that's its job and it's irreplaceable. Proportional and derivative shape the *transient* — the very thing I'm failing at — while integral guarantees the *asymptote*. And nicely, setting `K_P = K_D = 0` recovers the traditional Lagrangian method exactly, so this is a strict generalization, not a replacement: I've widened the space of update rules from a one-parameter family to a three-parameter one, with the old method sitting at the origin of the two new axes.

Now I move from continuous dynamics to the discrete per-iteration rule I'll actually run. Each iteration `k` I receive an estimate of the episodic cost `J_C`. Define the error `Delta = J_C - d`. The integral is the running sum of errors; the proportional reaction is `Delta` itself; the derivative is the (rectified) change in cost since last iteration, `partial = (J_C - J_C_prev)_+`. And the multiplier must stay nonnegative — it's a KKT multiplier for an inequality constraint, `lambda >= 0`. So:

  Delta   <- J_C - d
  partial <- ( J_C - J_C_prev )_+
  I       <- ( I + Delta )_+
  lambda  <- ( K_P Delta + K_I I + K_D partial )_+
  J_C_prev <- J_C

Two projections to think about. The outer `(.)_+` on `lambda` is the multiplier nonnegativity. The inner `(.)_+` on the integral `I` is anti-windup: I don't want the integrator banking a big *negative* reservoir during a long feasible stretch, because then when a violation finally arrives the controller would first have to climb back out of that negative hole before `lambda` could respond — a delayed reaction, the exact disease I'm curing. Clamping `I` at zero from below keeps the integrator from accumulating "credit" for being safe. The rectification on `partial` I argued above. Everything else is just the three gains.

Now the part that actually bit people who tried to use big multipliers: step size. The policy gradient under the Lagrangian is `grad J - lambda grad J_C`. When `lambda` gets large — and with an aggressive controller it can — the magnitude of this combined gradient blows up, and a fixed learning rate then takes an enormous, destabilizing step in `theta`. The fix is to keep the *effective* step size consistent regardless of `lambda`. Notice

  arg max_theta ( J - lambda J_C ) = arg max_theta  (1/(1+lambda)) ( J - lambda J_C ),

since dividing the objective by the positive constant `1 + lambda` doesn't move the argmax. But it *does* normalize the gradient magnitude: the combined gradient becomes `(1/(1+lambda)) ( grad J - lambda grad J_C )`, a convex-combination-like blend whose scale stays bounded as `lambda` grows (as `lambda -> infinity` it tends to `-grad J_C`, the pure cost-reduction direction, at unit-ish scale, rather than a divergent multiple of it). In advantage form, where the policy gradient flows through a surrogate advantage, this says: combine the reward advantage `adv_r` and cost advantage `adv_c` as

  adv = ( adv_r - lambda * adv_c ) / ( 1 + lambda ).

This is the blend the inner PPO loop should use. I'll apply the same rescaling even to the plain Lagrangian baseline, so the comparison is about the *controller*, not about who got a luckier step size. If I want to keep the system control-affine, I can reparameterize `u = lambda/(1+lambda)` in `[0,1]` and weight `(1-u) grad J - u grad J_C`; same thing, sometimes handier, but the `1/(1+lambda)` form is what I'll implement. One more structural choice this forces: I should keep *separate* value functions for reward and cost, `V_R` and `V_C`, rather than folding everything into a single critic on `r + lambda c`. Because `lambda` now changes rapidly under proportional and derivative control, a single critic trained on the moving `r + lambda c` target would be perpetually stale; two fixed-target critics, blended at the policy-gradient stage by the current `lambda`, stay valid.

Now the estimation reality, which is where the derivative term could wreck me. `J_C` here is not the true expected episodic cost; it's a sample average over a minibatch of finished trajectories — noisy. The proportional and (especially) derivative terms react to that noise: a one-step difference `J_C - J_C_prev` of two noisy estimates is mostly noise. If I feed raw noisy estimates into the derivative term, I get the instability I flagged earlier. So I smooth. Keep an exponential moving average of the error for the proportional term and an EMA of the cost for the derivative term:

  delta_p <- a_p * delta_p + (1 - a_p) * Delta          (smoothed proportional input)
  cost_d  <- a_d * cost_d  + (1 - a_d) * J_C            (smoothed cost, for the derivative)

with `a_p, a_d` close to 1 (a long-ish averaging window) so a single bad minibatch can't jerk the controller. And for the derivative itself I don't take a one-step difference — too jittery even after EMA. I difference the smoothed cost against a *delayed* copy from several iterations back: keep a short queue of past smoothed costs of length `d_delay`, and compute

  pid_d <- ( cost_d - cost_ds[oldest in queue] )_+

so the derivative is a cleaner finite difference over a window of `d_delay` iterations, rectified to brake only increases. This is the practical realization of the trend term: a delayed difference of a smoothed signal, not a literal second derivative of raw samples. The integral I leave un-smoothed — it's already an accumulation, inherently low-pass.

Let me also fold the gains the way the running controller does. The integral gain `K_I` can be absorbed into the integral update itself — accumulate `Delta * K_I` into `I` directly — so the output is just `K_P * delta_p + I + K_D * pid_d` with `I` carrying its own gain. That's an equivalent reparameterization (it just relabels where the `K_I` factor lives) and it keeps the output line clean. Then clamp the output at zero for nonnegativity. The implementation can optionally clamp the normalized output to `[0, 1]`, and otherwise cap the raw penalty at a configured maximum unless a sum-normalized variant is being used. So the per-iteration controller, in the form I'd ship:

  Delta   <- J_C - d
  I       <- max( 0, I + K_I * Delta )                  # integral, gain folded in, anti-windup
  delta_p <- a_p * delta_p + (1 - a_p) * Delta          # smoothed proportional input
  cost_d  <- a_d * cost_d  + (1 - a_d) * J_C            # smoothed cost for derivative
  pid_d   <- max( 0, cost_d - cost_ds[0] )              # rectified delayed-difference derivative
  lambda  <- max( 0, K_P * delta_p + I + K_D * pid_d )  # PID output, nonnegative
  if diff_norm: clamp I and lambda to at most 1
  if not diff_norm and not sum_norm: clamp lambda to penalty_max
  push cost_d onto cost_ds (a queue of length d_delay)

The gains: `K_P` proportional, `K_I` integral, `K_D` derivative, all nonnegative. The controller leaves these as configuration knobs; in the deep-RL setting the useful regime is small gains and slowly changing smoothed inputs, with the integral as the slow memory and proportional and derivative giving fast, *shaped* responses on top of that slow baseline. That gives responsiveness without having to crank `K_I` into the regime that ruins reward.

There's a second, separable robustness problem I can now also see clearly, even if it's a different axis. Suppose I scale all rewards by a constant `rho` (different environments, or the same environment as reward magnitudes drift during learning). The optimal policy doesn't change, but the optimal multiplier `lambda*` must scale by `rho` — because `lambda` trades reward units against cost units, and I just changed the reward unit. That means to get the same learning dynamics I'd have to rescale `lambda_0`, `K_P`, `K_I`, `K_D` all by `rho`. Annoying, and it makes a single gain choice non-transferable across environments. The fix is to give `lambda` a fixed *meaning* — make `lambda = 1` always denote "reward and cost contribute equally to the update." Introduce a scaling `beta` on the cost gradient,

  grad L = (1 - u) grad J - u * beta * grad J_C,

and choose `beta` as the ratio of gradient magnitudes,

  beta = || grad J || / || grad J_C ||,

so that at `lambda = 1` the two objectives contribute equal-magnitude gradients and `lambda*` is encouraged toward 1 regardless of reward scale. It's cheap — both gradient norms are already lying around in the algorithm — and it makes one set of gains transfer across reward scales, with `lambda_0 = 1` becoming the obvious initialization. A policy-space variant could balance by KL divergences of the reward-only and cost-only updates instead of gradient norms, but the gradient-norm ratio is simpler to compute from the first-order ingredients already in the loop. This is a complementary knob; the core contribution is the PID controller on `lambda`.

So let me put it into the code I'd actually run, filling the two open slots in the constrained-PPO harness — the multiplier update and the advantage blend.

```python
import numpy as np
import torch
from collections import deque


class PIDLagrangian:
    """PID controller on the Lagrange multiplier for a safe-RL inequality constraint.
    K_P=K_D=0 recovers the traditional (integral-only) Lagrangian update."""

    def __init__(self, pid_kp, pid_ki, pid_kd, pid_d_delay,
                 pid_delta_p_ema_alpha, pid_delta_d_ema_alpha,
                 sum_norm, diff_norm, penalty_max,
                 lagrangian_multiplier_init, cost_limit):
        self._pid_kp = pid_kp
        self._pid_ki = pid_ki
        self._pid_kd = pid_kd
        self._pid_d_delay = pid_d_delay
        self._pid_delta_p_ema_alpha = pid_delta_p_ema_alpha
        self._pid_delta_d_ema_alpha = pid_delta_d_ema_alpha
        self._sum_norm = sum_norm
        self._diff_norm = diff_norm
        self._penalty_max = penalty_max
        self._pid_i = lagrangian_multiplier_init       # integral accumulator
        self._cost_ds = deque(maxlen=self._pid_d_delay)
        self._cost_ds.append(0.0)
        self._delta_p = 0.0                            # EMA of the error
        self._cost_d = 0.0                             # EMA of the cost
        self._cost_limit = cost_limit
        self._cost_penalty = 0.0                       # controller output

    @property
    def lagrangian_multiplier(self):
        return self._cost_penalty

    def pid_update(self, ep_cost):
        delta = float(ep_cost - self._cost_limit)            # Delta = J_C - d
        # integral: accumulate the error (gain folded in), anti-windup clamp at 0
        self._pid_i = max(0.0, self._pid_i + delta * self._pid_ki)
        if self._diff_norm:
            self._pid_i = max(0.0, min(1.0, self._pid_i))
        # smoothed proportional input (EMA over the noisy minibatch error)
        a_p = self._pid_delta_p_ema_alpha
        self._delta_p = a_p * self._delta_p + (1 - a_p) * delta
        # smoothed cost, for a clean derivative
        a_d = self._pid_delta_d_ema_alpha
        self._cost_d = a_d * self._cost_d + (1 - a_d) * float(ep_cost)
        # derivative: rectified difference of smoothed cost vs a delayed copy (brakes increases)
        pid_d = max(0.0, self._cost_d - self._cost_ds[0])
        # PID output, projected nonnegative (lambda >= 0)
        pid_o = self._pid_kp * self._delta_p + self._pid_i + self._pid_kd * pid_d
        self._cost_penalty = max(0.0, pid_o)
        if self._diff_norm:
            self._cost_penalty = min(1.0, self._cost_penalty)
        if not (self._diff_norm or self._sum_norm):
            self._cost_penalty = min(self._cost_penalty, self._penalty_max)
        self._cost_ds.append(self._cost_d)                   # advance the delay queue


def compute_adv_surrogate(adv_r, adv_c, lam):
    """Step-size-consistent blend: arg max (J - lam*J_C) == arg max (J - lam*J_C)/(1+lam),
    which keeps the policy-gradient magnitude bounded as lam grows."""
    return (adv_r - lam * adv_c) / (1.0 + lam)
```

And to wire it into the fixed PPO outer step, the two slots become:

```python
class ConstrainedPPO:
    def _init_controller(self):
        self._pid = PIDLagrangian(**self.cfg.lagrange_cfgs)

    def _update_multiplier(self, ep_cost):
        ep_cost = float(ep_cost)
        assert not np.isnan(ep_cost), 'cost is nan'
        self._pid.pid_update(ep_cost)
        self._lambda = self._pid.lagrangian_multiplier

    def _compute_adv_surrogate(self, adv_r, adv_c):
        return compute_adv_surrogate(adv_r, adv_c, self._lambda)
```

Let me retrace the causal chain. I started with a dual safe-RL solver that converges to a feasible policy yet violates the constraint throughout training, with the cost oscillating about the limit and a quarter-cycle lag between violation and multiplier. That lag is the fingerprint of integral control — and indeed the traditional multiplier update is exactly an integrator on the constraint error, the impoverished case of a feedback controller whose plant is the policy-optimization step. Recognizing that opened the space of update rules from one knob to a real controller. Adding a proportional term, traced through Platt and Barr's second-order collapse, injects a positive-semidefinite `beta grad g grad^T g` into the damping matrix — the same damping the quadratic penalty method gives, but without touching the primal objective and without the penalty's bundled indefinite Hessian term — so it damps the oscillation cleanly. Adding a derivative term decouples the acceleration through `B = I + gamma grad g grad^T g` and creates a velocity-quadratic, curvature-modulated force that acts *ahead* of violations, rectified to brake cost increases without impeding decreases. The integral term stays because only it eliminates steady-state violation. Discretized, the controller is the running integral plus proportional reaction plus a rectified delayed difference, each projected for nonnegativity and anti-windup; the noisy minibatch cost estimates are tamed with EMA smoothing on the proportional and derivative inputs and a delayed-difference window for the derivative. The large-`lambda` step-size blowup is handled by rescaling the objective by `1/(1+lambda)`, giving the advantage blend `(adv_r - lambda adv_c)/(1+lambda)` and motivating separate fast-valid reward and cost critics. And reward-scale sensitivity, a separate axis, can be handled by weighting the cost gradient by `||grad J||/||grad J_C||` so `lambda = 1` always means "balanced." Setting `K_P = K_D = 0` returns the original method, confirming this is a strict, controls-grounded generalization of the Lagrangian baseline that keeps its simplicity while fixing its primary failure.
