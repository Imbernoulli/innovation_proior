OK, let me think this through from scratch. I have a rocket in powered descent, and I want it to land softly at a target while spending as little fuel as possible. Fuel is not just another cost; it is the mass margin I could have delivered as payload. I also cannot treat the engine as an ideal acceleration knob. It has a minimum stable thrust, a maximum thrust, a pointing limit, and the vehicle has to stay above the terrain and arrive with essentially zero velocity. If I solve this with a generic nonlinear optimizer, I may get a local answer after an unpredictable number of iterations. That is not good enough for onboard guidance. I need a formulation whose optimum is global and whose solve time is controlled.

The physics are simple enough to write first. Let r be position, ṙ velocity, m mass, T_c commanded thrust, g gravity, and S(ω) the skew matrix for the planet rotation rate. Then

  r̈ = g + T_c/m − 2 S(ω) ṙ − S(ω)^2 r,   ṁ = −α ‖T_c‖,

where α = 1/(I_sp g_0). If I ignore the rotation terms for a moment, the core difficulty is still there: thrust divided by mass, and mass depleted by thrust magnitude. Integrating the mass equation gives m(tf) = m0 − α∫_0^tf ‖T_c(t)‖ dt, so minimizing propellant is the same as maximizing final mass.

The engine constraint is the part that breaks convexity:

  ρ_min ≤ ‖T_c(t)‖ ≤ ρ_max.

The upper bound is a ball. The lower bound is the outside of a ball. Put them together and I get an annulus, and an annulus is not convex: two legal thrust vectors at the lower magnitude can average to zero thrust, which is illegal. Dropping the lower bound would give a convex ball, but then the optimizer can coast through the hole. That trajectory can look cheap mathematically and still be impossible for the engine.

Pontryagin's maximum principle tells me why there might be a way out. The thrust direction enters the Hamiltonian through the velocity costate, and the magnitude enters linearly through the running fuel term and the mass costate. Once the direction is chosen, the magnitude is not naturally an interior value; it is driven to an endpoint of the interval. This is the old bang-bang soft-landing structure: thrust hard, throttle to the floor, thrust hard, with switches governed by a scalar switching function. The nonconvex hole is present, but the optimum is already trying to live on the boundary of it.

So I introduce a scalar Γ that bounds the thrust magnitude:

  ‖T_c(t)‖ ≤ Γ(t),    ρ_min ≤ Γ(t) ≤ ρ_max,

and I use Γ in the relaxed mass equation, ṁ = −αΓ. This lifted set in (T_c, Γ) is convex: a second-order cone sliced by two scalar bounds. It has enlarged the feasible set, because Γ can sit at ρ_min while ‖T_c‖ is smaller. The only way this helps is if the optimizer is forced back onto ‖T_c‖ = Γ.

Run the maximum-principle argument on the relaxed problem. For fixed Γ, the Hamiltonian's direction-dependent term is λ_v^T T_c/m. If λ_v is nonzero, maximizing over ‖T_c‖ ≤ Γ pushes T_c to the edge of the ball and aligns it with λ_v, so ‖T_c‖ = Γ. Slack can remain only where λ_v = 0. Suppose λ_v vanished over a time interval. The adjoint equations are linear, and the controllability condition for the translational dynamics prevents a nontrivial costate from hiding with zero velocity component over an interval; propagated through the adjoint system, all multipliers would be forced to vanish, contradicting the maximum principle's nontriviality condition. Therefore λ_v can vanish only on a measure-zero set, and the relaxed optimum satisfies

  ‖T_c*(t)‖ = Γ*(t)   almost everywhere.

That is the lossless part: the convex lift did add unphysical points, but under the controllability and transversality conditions the optimum does not use them. With glide-slope state constraints I have to be more careful, because a state-constraint multiplier enters the adjoint equations. The same conclusion is safe when the trajectory touches the glide-slope boundary only at isolated instants rather than riding it over a finite arc, which is the normal powered-descent geometry.

Now the dynamics still contain T_c/m. The natural variables are the mass-normalized thrust and slack,

  u = T_c/m,   σ = Γ/m,   z = ln m.

Then the translational acceleration becomes g + u − 2Sṙ − S^2r, and the mass equation becomes

  ż = ṁ/m = −α Γ/m = −α σ.

The dynamics are linear in [r; ṙ; z] and [u; σ], and the cone is still simple:

  ‖u‖ ≤ σ.

The thrust box moves into the variables as

  ρ_min e^(−z) ≤ σ ≤ ρ_max e^(−z).

That is the price of removing mass from the dynamics. I have one scalar exponential left, and the direction of the approximation matters. Use the maximum-thrust mass profile as a lower bound on log mass,

  z0(t) = ln(m0 − α ρ_max t).

The minimum-thrust profile gives the upper envelope,

  z0(t) ≤ z(t) ≤ ln(m0 − α ρ_min t).

Now δ = z − z0 is nonnegative. That sign is crucial. Since e^(−δ) = 1 − δ + δ^2/2 − δ^3/6 + ..., the quadratic truncation 1 − δ + δ^2/2 lies above e^(−δ) for δ ≥ 0, while the tangent 1 − δ lies below e^(−δ). The lower thrust floor is

  ρ_min e^(−z) ≤ σ.

If I enforce the stronger condition

  μ1(t)[1 − δ + δ^2/2] ≤ σ,    μ1(t) = ρ_min e^(−z0(t)),

then I am safely above the true floor, and the left side is a convex quadratic in z. A convex quadratic bounded above by an affine variable is a second-order-cone-representable constraint. The upper thrust ceiling is

  σ ≤ ρ_max e^(−z).

Here the quadratic would put σ below a convex function, which is nonconvex. The tangent gives the usable convex side:

  σ ≤ μ2(t)[1 − δ],    μ2(t) = ρ_max e^(−z0(t)).

This is conservative because 1 − δ ≤ e^(−δ) for δ ≥ 0. The log-mass envelopes are not optional decoration; they make the Taylor signs meaningful and keep the convex bounds tied to the physical thrust interval.

The objective also needs a careful interpretation. σ is not the physical fuel rate Γ; it is Γ/m. But z(tf) = z(0) − α∫σ dt, so minimizing ∫σ dt is exactly maximizing terminal log mass, and because log is monotone it is equivalent to maximizing terminal mass. In a discretized program I can minimize −z_N directly or minimize the quadrature of σ; with the linear z update they represent the same fuel objective up to constants.

The remaining constraints have to stay conic or affine. A glide-slope cone with vertical axis e1 can be written as

  e1^T r ≥ tan(γ_gs) ‖r_lat‖,

which is a second-order cone; if I need the state-constraint proof in a purely affine form, I can approximate the circular cone by planar facets. A speed cap is just ‖ṙ‖ ≤ v_max. The pointing constraint is more interesting. The original form is

  n^T T_c ≥ ‖T_c‖ cos θ_p.

For θ_p > 90 degrees this set is nonconvex. In the lifted normalized variables I write

  n^T u ≥ σ cos θ_p.

That is a half-space in (u, σ), even when cos θ_p is negative. It is a relaxation while σ may exceed ‖u‖, but the same equality ‖u*‖ = σ* at the optimum recovers the original pointing constraint.

For discretization, I should not lose the factors. If I take u and σ piecewise linear on each interval [t_k, t_{k+1}], then the no-rotation translational update is

  ṙ_{k+1} = ṙ_k + (Δt/2)(u_k + u_{k+1}) + gΔt,
  r_{k+1} = r_k + (Δt/2)(ṙ_k + ṙ_{k+1}) + (Δt^2/12)(u_{k+1} − u_k),
  z_{k+1} = z_k − (αΔt/2)(σ_k + σ_{k+1}).

With rotation terms, the same idea is implemented through the linear state-transition matrices for ẋ = Ax + B(g + u). A simpler Euler transcription,

  x_{k+1} = x_k + (A x_k + B(g + u_k))Δt,   z_{k+1} = z_k − ασ_k Δt,

is the compact version I can write directly in a small CVXPY implementation; the conic constraints and the lossless lift are unchanged, but the higher-order transcription above is the one where the Δt/2 and Δt^2/12 factors belong.

The final-time variable sits outside a fixed-grid SOCP, so I wrap the solve in a one-dimensional search over tf. The fuel curve for this landing problem has a single useful minimum over the feasible bracket, so golden-section search is enough: choose tf, solve the SOCP, compare fuel, and shrink the bracket. If the prescribed target is unreachable, I should not stop at infeasible. I first solve a closest-landing problem that minimizes the lateral miss distance subject to the same safe-descent constraints. Then I solve the fuel problem again while holding that achieved miss-distance radius fixed. That second solve is what restores the fuel-optimal trajectory among the closest reachable landings.

Here is the compact solver structure I end up with. I keep the parameter-and-solver shape simple, while making the upper thrust Taylor bound, the log-mass envelope, and the Δt factor in the log-mass update explicit.

```python
from dataclasses import dataclass, field
import numpy as np
import cvxpy as cp

@dataclass
class DescentParameters:
    x0: np.ndarray = field(default_factory=lambda: np.array([2400., 450., -330., -10., -40., 10.]))
    target: np.ndarray = field(default_factory=lambda: np.array([0., 0.]))
    m0: float = 2000.0
    m_dry: float = 1700.0
    alpha: float = 5e-4
    rho_min: float = 0.2 * 24000.0
    rho_max: float = 0.8 * 24000.0
    tf: float = 50.0
    dt: float = 1.0
    glide_slope_deg: float = 30.0
    pointing_deg: float = 120.0
    velocity_max: float = 90.0
    final_velocity_max: float = 2.0
    omega: np.ndarray = field(default_factory=lambda: np.array([2.53e-5, 0., 6.62e-5]))
    gravity: np.ndarray = field(default_factory=lambda: np.array([-3.71, 0., 0.]))

    def __post_init__(self):
        self.K = int(round(self.tf / self.dt))
        self.N = self.K + 1
        self.zi = np.log(self.m0)
        self.zf = np.log(self.m_dry)
        self.glide_tan = np.tan(np.deg2rad(self.glide_slope_deg))
        self.pointing_cos = np.cos(np.deg2rad(self.pointing_deg))
        self.e1 = np.array([1., 0., 0.])
        self.e2 = np.array([0., 1., 0.])
        self.e3 = np.array([0., 0., 1.])
        self.E = np.vstack([self.e2, self.e3])
        S = skew(self.omega)
        self.A = np.block([[np.zeros((3, 3)), np.eye(3)],
                           [-S @ S,             -2.0 * S]])
        self.B = np.vstack([np.zeros((3, 3)), np.eye(3)])

def skew(w):
    return np.array([[0., -w[2], w[1]],
                     [w[2], 0., -w[0]],
                     [-w[1], w[0], 0.]])

class PoweredDescentGuidance:
    def __init__(self, params: DescentParameters):
        self.params = params

    def _decision_variables(self):
        p = self.params
        x = cp.Variable((6, p.N))       # [position; velocity]
        z = cp.Variable((1, p.N))       # log mass
        u = cp.Variable((3, p.N))       # T_c / m
        gamma = cp.Variable((1, p.N))   # Gamma / m
        return x, z, u, gamma

    def _mass_profiles(self):
        p = self.params
        t = p.dt * np.arange(p.N)
        z_lo = np.log(p.m0 - p.alpha * p.rho_max * t)
        z_hi = np.log(p.m0 - p.alpha * p.rho_min * t)
        mu1 = p.rho_min * np.exp(-z_lo)
        mu2 = p.rho_max * np.exp(-z_lo)
        return z_lo, z_hi, mu1, mu2

    def _set_common_constraints(self, x, z, u, gamma):
        p = self.params
        constraints = [
            x[:, 0] == p.x0,
            z[0, 0] == p.zi,
            z[0, -1] >= p.zf,
            p.e1 @ x[:3, -1] == 0.0,
            p.e1 @ x[3:, -1] == 0.0,
            cp.norm(x[3:, -1]) <= p.final_velocity_max,
        ]

        for k in range(p.N - 1):
            constraints.append(
                x[:, k + 1] == x[:, k] + (p.A @ x[:, k] + p.B @ (p.gravity + u[:, k])) * p.dt
            )
            constraints.append(z[0, k + 1] == z[0, k] - p.alpha * gamma[0, k] * p.dt)

        z_lo, z_hi, mu1, mu2 = self._mass_profiles()
        dz = z[0, :] - z_lo
        constraints.extend([
            cp.norm(u, axis=0) <= gamma[0, :],
            p.e1 @ u >= cp.multiply(gamma[0, :], p.pointing_cos),
            z[0, :] >= z_lo,
            z[0, :] <= z_hi,
            cp.multiply(mu1, 1.0 - dz + 0.5 * cp.square(dz)) <= gamma[0, :],
            gamma[0, :] <= cp.multiply(mu2, 1.0 - dz),
            x[0, :] >= cp.norm(x[1:3, :], axis=0) * p.glide_tan,
            cp.norm(x[3:6, :], axis=0) <= p.velocity_max,
        ])
        return constraints

    def solve_minimum_error(self):
        p = self.params
        x, z, u, gamma = self._decision_variables()
        miss = cp.norm(p.E @ x[:3, -1] - p.target)
        smooth = cp.sum_squares(gamma[0, 1:] - gamma[0, :-1])
        objective = cp.Minimize(5.0 * miss - 0.5 * z[0, -1] + cp.norm(x[3:, -1]) + 0.1 * smooth)
        constraints = self._set_common_constraints(x, z, u, gamma)
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.CLARABEL)
        return problem.status, x, u, gamma, z

    def solve_minimum_fuel(self, landing_error_radius):
        p = self.params
        x, z, u, gamma = self._decision_variables()
        smooth = cp.sum_squares(gamma[0, 1:] - gamma[0, :-1])
        objective = cp.Minimize(-z[0, -1] + 0.05 * smooth + cp.norm(x[3:, -1]))
        constraints = self._set_common_constraints(x, z, u, gamma)
        constraints.append(cp.norm(p.E @ x[:3, -1] - p.target) <= landing_error_radius)
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.CLARABEL)
        return problem.status, x, u, gamma, z

    @staticmethod
    def recover_thrust(u, z):
        if u.value is None or z.value is None:
            return None
        return u.value * np.exp(z.value)
```
