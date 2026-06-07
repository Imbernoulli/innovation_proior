# Context: minimum-fuel powered-descent guidance

## Research question

A spacecraft in final powered descent must fly from a known position and velocity to a soft landing at, or as close as possible to, a surface target. The engine thrust both changes the trajectory and consumes the mass margin available for payload, so the central objective is fuel: among all trajectories that satisfy the vehicle and safety constraints, choose the one that lands with maximum remaining mass.

The problem is hard because the engine cannot be treated as an ideal acceleration source. Once lit, a throttleable rocket has a lower usable thrust, set by combustion stability, and an upper thrust, set by performance. The vehicle must also stay inside a glide-slope cone above the terrain, respect a thrust pointing or tilt limit, avoid excessive speed, and end at rest with zero altitude. A useful guidance law must compute onboard in seconds, return a global optimum or a reliable infeasibility certificate, and avoid dependence on a carefully guessed initial trajectory.

## Background

The classical point-mass model uses position r, velocity ṙ, mass m, commanded thrust T_c, constant gravity g, and, in a rotating planet frame, the skew matrix S(ω):

  r̈ = g + T_c/m − 2 S(ω) ṙ − S(ω)^2 r,  ṁ = −α ‖T_c‖,

with α = 1/(I_sp g_0) in the usual rocket-equation units. Integrating the mass equation gives

  m(tf) = m0 − α∫_0^tf ‖T_c(t)‖ dt,

so minimizing fuel is equivalent to maximizing final mass.

Pontryagin's maximum principle gives the first structural clue. The Hamiltonian is affine in the thrust direction through the velocity costate and affine in the thrust magnitude through the running fuel cost and mass costate. The optimal thrust direction follows the primer vector, and the optimal magnitude is pushed to an endpoint of the admissible interval. This is the bang-bang soft-landing structure studied in the older rocket-control literature: maximum thrust, minimum thrust, maximum thrust, with switch times determined by a scalar switching function.

The lower thrust bound is the obstacle. The admissible set

  ρ_min ≤ ‖T_c(t)‖ ≤ ρ_max,  with 0 < ρ_min < ρ_max,

is an annulus in thrust space. The upper bound is a convex ball, but the lower bound is the complement of a ball, so a convex solver cannot accept the true control set directly. Removing the lower bound is not a faithful workaround, because the resulting optimizer can command coasting or arbitrarily small thrust during phases where the physical engine cannot operate.

The mass coupling adds another nonlinearity: acceleration is T_c/m, and m is a state. A pointing constraint of the form n^T T_c ≥ ‖T_c‖ cos θ_p can also be nonconvex when θ_p is obtuse, because the admissible directions cover more than a half-space but exclude a cone around the opposite direction.

Second-order cone programming is the natural computational target. Norm bounds, glide-slope cones, velocity caps, and many thrust constraints are second-order cone or affine constraints once the right variables are chosen. Interior-point SOCP solvers provide deterministic convergence behavior to a requested accuracy, which is why a convex transcription is attractive for autonomous descent.

Several empirical and structural facts shape the formulation. Minimum-fuel soft-landing profiles have the bang-bang max-min-max structure. For Mars-style powered descent, the fuel curve as a function of fixed flight time has a single useful minimum in the feasible interval, so an outer one-dimensional search over flight time is practical. For the glide-slope state constraint, the lossless-control argument is cleanest when the trajectory touches the cone only at isolated times rather than sliding along it over an interval; planetary landing geometries normally exhibit this isolated-contact behavior.

## Baselines

**Direct nonlinear optimal-control transcription.** A collocation, pseudospectral, SQP, or shooting method can discretize the original dynamics and constraints, then solve the resulting nonlinear program. This keeps the physical model close to the original problem, but the annulus constraint and mass coupling make the NLP nonconvex. The solver may converge to a local minimum, may depend strongly on the initial guess, and does not provide the bounded global certificate needed for onboard firing decisions.

**Apollo-style explicit or polynomial guidance.** Earlier powered-descent guidance laws choose an analytic acceleration or polynomial position profile and solve for coefficients that satisfy terminal conditions. These laws are simple and fast, but they do not optimize fuel under the full annular thrust band, glide slope, velocity, pointing, and large-divert constraints.

**One-dimensional soft-landing structure.** Meditch-style vertical soft-landing analysis gives useful insight into minimum-time and minimum-fuel feasibility along the vertical channel. It helps bracket feasible flight times, but it does not solve the full three-dimensional divert problem with lateral constraints and thrust-vector pointing.

**Dropping the lower thrust bound.** Replacing ρ_min ≤ ‖T_c‖ ≤ ρ_max with only ‖T_c‖ ≤ ρ_max makes the thrust set convex. The gap is physical fidelity: the solution can live inside the forbidden hole of the annulus, so the computed trajectory can be impossible for the engine to fly.

## Evaluation settings

The standard setting is planetary powered descent, especially Mars-style pinpoint landing after parachute cutoff. A scenario specifies gravity, rotation rate, initial position and velocity relative to the target, wet and dry mass, thrust lower and upper limits, specific impulse, glide-slope angle, thrust pointing angle, speed limits, and a fixed or searched flight time. The natural metrics are final mass or propellant consumed, lateral landing error, terminal speed, constraint satisfaction, solver status, solve time, and iteration count. Large-divert cases also test whether the guidance law can land safely as close as possible when the prescribed target is unreachable.

## Code framework

A starting software scaffold is a trajectory-optimization harness: parameters, dynamics matrices, a place to choose decision variables and encode the difficult vehicle constraints, and two solves for the reachable-target and closest-reachable-target cases. The empty methods below are the slots where the trajectory representation and thrust/mass constraint encoding still have to be designed.

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
        pass

def skew(w):
    pass

class PoweredDescentGuidance:
    def __init__(self, params: DescentParameters):
        self.params = params

    def _decision_variables(self):
        pass

    def _mass_profiles(self):
        pass

    def _set_common_constraints(self, x, z, u, gamma):
        pass

    def solve_minimum_error(self):
        pass

    def solve_minimum_fuel(self, landing_error_radius):
        pass
```
