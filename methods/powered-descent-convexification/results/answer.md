# Lossless convexification for minimum-fuel powered-descent guidance

## Problem

Compute a fuel-minimizing powered-descent trajectory with hard lower/upper thrust limits, mass depletion, glide-slope, pointing, velocity, and terminal landing constraints. The original thrust set

  ρ_min ≤ ‖T_c‖ ≤ ρ_max

is nonconvex because of the lower bound, and the dynamics contain the nonlinear term T_c/m.

## Method

Introduce a thrust-magnitude slack Γ:

  ‖T_c‖ ≤ Γ,  ρ_min ≤ Γ ≤ ρ_max,  ṁ = −αΓ.

The lifted set is convex. The relaxation is lossless for the control constraint: the maximum-principle argument aligns thrust with the velocity costate, and controllability plus nontriviality rule out the costate vanishing on an interval, so ‖T_c*‖ = Γ* almost everywhere.

Use mass-normalized variables

  u = T_c/m,  σ = Γ/m,  z = ln m.

Then

  r̈ = g + u − 2S(ω)ṙ − S(ω)^2r,  ż = −ασ,  ‖u‖ ≤ σ.

The thrust bounds become ρ_min e^(−z) ≤ σ ≤ ρ_max e^(−z). With

  z0(t) = ln(m0 − α ρ_max t),  δ = z − z0,

and the path envelope z0(t) ≤ z(t) ≤ ln(m0 − α ρ_min t), δ is nonnegative. The convex bounds are

  ρ_min e^(−z0)[1 − δ + δ^2/2] ≤ σ,
  σ ≤ ρ_max e^(−z0)[1 − δ].

The first is a conservative convex quadratic floor; the second is an affine conservative ceiling. Pointing becomes the half-space n^T u ≥ σ cos θ_p. Glide slope and velocity are second-order cones. A fixed-flight-time transcription is an SOCP; flight time is handled by a one-dimensional search. If the target is unreachable, solve closest landing first, then minimize fuel subject to the achieved miss-distance radius.

## Code

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
        x = cp.Variable((6, p.N))
        z = cp.Variable((1, p.N))
        u = cp.Variable((3, p.N))
        gamma = cp.Variable((1, p.N))
        return x, z, u, gamma

    def _mass_profiles(self):
        p = self.params
        t = p.dt * np.arange(p.N)
        z_lo = np.log(p.m0 - p.alpha * p.rho_max * t)
        z_hi = np.log(p.m0 - p.alpha * p.rho_min * t)
        return z_lo, z_hi, p.rho_min * np.exp(-z_lo), p.rho_max * np.exp(-z_lo)

    def _set_common_constraints(self, x, z, u, gamma):
        p = self.params
        cons = [
            x[:, 0] == p.x0,
            z[0, 0] == p.zi,
            z[0, -1] >= p.zf,
            p.e1 @ x[:3, -1] == 0.0,
            p.e1 @ x[3:, -1] == 0.0,
            cp.norm(x[3:, -1]) <= p.final_velocity_max,
        ]
        for k in range(p.N - 1):
            cons += [
                x[:, k + 1] == x[:, k] + (p.A @ x[:, k] + p.B @ (p.gravity + u[:, k])) * p.dt,
                z[0, k + 1] == z[0, k] - p.alpha * gamma[0, k] * p.dt,
            ]
        z_lo, z_hi, mu1, mu2 = self._mass_profiles()
        dz = z[0, :] - z_lo
        cons += [
            cp.norm(u, axis=0) <= gamma[0, :],
            p.e1 @ u >= cp.multiply(gamma[0, :], p.pointing_cos),
            z[0, :] >= z_lo,
            z[0, :] <= z_hi,
            cp.multiply(mu1, 1.0 - dz + 0.5 * cp.square(dz)) <= gamma[0, :],
            gamma[0, :] <= cp.multiply(mu2, 1.0 - dz),
            x[0, :] >= cp.norm(x[1:3, :], axis=0) * p.glide_tan,
            cp.norm(x[3:6, :], axis=0) <= p.velocity_max,
        ]
        return cons

    def solve_minimum_error(self):
        p = self.params
        x, z, u, gamma = self._decision_variables()
        miss = cp.norm(p.E @ x[:3, -1] - p.target)
        smooth = cp.sum_squares(gamma[0, 1:] - gamma[0, :-1])
        prob = cp.Problem(
            cp.Minimize(5.0 * miss - 0.5 * z[0, -1] + cp.norm(x[3:, -1]) + 0.1 * smooth),
            self._set_common_constraints(x, z, u, gamma),
        )
        prob.solve(solver=cp.CLARABEL)
        return prob.status, x, u, gamma, z

    def solve_minimum_fuel(self, landing_error_radius):
        p = self.params
        x, z, u, gamma = self._decision_variables()
        smooth = cp.sum_squares(gamma[0, 1:] - gamma[0, :-1])
        cons = self._set_common_constraints(x, z, u, gamma)
        cons.append(cp.norm(p.E @ x[:3, -1] - p.target) <= landing_error_radius)
        prob = cp.Problem(cp.Minimize(-z[0, -1] + 0.05 * smooth + cp.norm(x[3:, -1])), cons)
        prob.solve(solver=cp.CLARABEL)
        return prob.status, x, u, gamma, z

    @staticmethod
    def recover_thrust(u, z):
        if u.value is None or z.value is None:
            return None
        return u.value * np.exp(z.value)
```

The slack equality recovers the physical thrust T_c = m u with ‖T_c‖ = mσ at the optimum of the relaxed control problem. The SOCP solves the fixed-time convex transcription globally; the outer search and two-stage miss-distance/fuel solve wrap that fixed-time core.
