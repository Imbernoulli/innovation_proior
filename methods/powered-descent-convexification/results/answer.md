# Lossless convexification for minimum-fuel powered-descent guidance

## Problem

Land a rocket softly at a target while burning minimum fuel (= maximizing landed mass), subject to gravity, mass depletion, a throttleable engine that cannot run below ρ_min nor above ρ_max, a glide-slope cone, a thrust pointing/tilt limit, a velocity cap, and fixed boundary conditions — and do it onboard, in bounded time, with a **global** optimality (or infeasibility) certificate.

Dynamics (planet frame, skew matrix S = S(ω) of the rotation rate):
- r̈ = g + T_c/m − 2 S ṙ − S² r,  ṁ = −α‖T_c‖,  α = 1/(I_sp g₀)
- minimize ∫₀^{tf} ‖T_c‖ dt  ⇔  maximize m(tf)

The obstruction: the admissible thrust set ρ_min ≤ ‖T_c‖ ≤ ρ_max is a nonconvex **annulus** (the lower bound is the complement of a ball); the dynamics are nonlinear (T_c/m); and an obtuse pointing limit n̂ᵀT_c ≥ ‖T_c‖cos θ_p is nonconvex.

## Key idea — lossless convexification

1. **Slack relaxation (lift).** Introduce Γ(t) and replace the annulus by the convex pair
   ‖T_c‖ ≤ Γ,  ρ_min ≤ Γ ≤ ρ_max, using Γ in the cost (min ∫Γ) and mass dynamics (ṁ = −αΓ).
2. **Losslessness.** Via Pontryagin's maximum principle, the optimal thrust aligns with the velocity costate (primer vector) λ_v and uses all available magnitude, so ‖T_c*‖ = Γ* wherever λ_v ≠ 0. Controllability of {A,B} plus the nontriviality (transversality) condition forbid λ_v vanishing on a positive-measure interval, so **‖T_c*‖ = Γ* almost everywhere** — the relaxation is tight. The convex optimum is the exact global optimum of the original nonconvex problem; no feasible region is removed.
3. **Change of variables.** u = T_c/m, σ = Γ/m, z = ln m make the dynamics **linear**:
   r̈ = g + u − 2Sṙ − S²r,  ż = −α σ,  with ‖u‖ ≤ σ.
4. **Convex bounds on σ.** The box becomes ρ_min e^{−z} ≤ σ ≤ ρ_max e^{−z}. Expand e^{−z} about z₀(t)=ln(m_wet − αρ_max t), δ=z−z₀:
   - floor: μ₁[1 − δ + δ²/2] ≤ σ (3 terms → convex quadratic ≤ σ = SOC), μ₁=ρ_min e^{−z₀};
   - ceiling: σ ≤ μ₂[1 − δ] (2 terms → linear; a quadratic upper bound would be nonconvex), μ₂=ρ_max e^{−z₀}.
5. **Pointing for free.** In the lifted space the obtuse cone becomes the half-space n̂ᵀu ≥ σ cos θ_p, convex for any θ_p. Glide slope e₁ᵀr ≥ tan(γ_gs)‖r_lat‖ and ‖ṙ‖ ≤ v_max are second-order cones.

## Final algorithm

Discretize on a grid → a finite **second-order cone program** solved by an interior-point method (bounded iterations, global optimum). Free final time tf is handled by an outer **golden-section search** (cost is unimodal in tf). For unreachable targets, a prioritized two-step solve: first minimize miss distance ‖r(tf)−target‖, then minimize fuel subject to that miss distance.

```python
import numpy as np
import cvxpy as cp

# --- vehicle + environment + grid ---
g     = np.array([-3.71, 0.0, 0.0])          # gravity [m/s^2]
omega = np.array([2.53e-5, 0.0, 6.62e-5])    # planet rate [rad/s]
m_wet = 2000.0; m_dry = 1700.0               # wet / dry-floor mass [kg]
rho1  = 0.2 * 24000.0                         # rho_min [N] (throttle floor)
rho2  = 0.8 * 24000.0                         # rho_max [N] (throttle ceiling)
alpha = 5e-4                                  # mass flow per N = 1/(Isp g0)
gamma_gs = np.deg2rad(30.0)                   # glide-slope half-angle
theta_p  = np.deg2rad(120.0)                  # thrust tilt limit (obtuse)
v_max    = 90.0
x0 = np.array([2400., 450., -330., -10., -40., 10.])   # [r; v]
N, dt = 50, 1.0
zi, zf = np.log(m_wet), np.log(m_dry)

S = np.array([[0, -omega[2], omega[1]],
              [omega[2], 0, -omega[0]],
              [-omega[1], omega[0], 0]])
A = np.block([[np.zeros((3,3)), np.eye(3)],
              [-S @ S,          -2*S]])
B = np.vstack([np.zeros((3,3)), np.eye(3)])
e1 = np.array([1., 0., 0.])

def solve_min_fuel(tf=None):
    # decision variables: state, log-mass, normalized thrust, slack
    x     = cp.Variable((6, N))     # [r; v]
    z     = cp.Variable((1, N))     # z = ln m
    u     = cp.Variable((3, N))     # u = T_c / m
    sigma = cp.Variable((1, N))     # sigma = Gamma / m  (the slack)

    cons  = [x[:,0] == x0, z[0,0] == zi, z[0,N-1] >= zf,
             e1 @ x[:3,N-1] == 0, e1 @ x[3:,N-1] == 0]  # soft touchdown

    z0  = np.array([np.log(m_wet - alpha*rho2*dt*k) for k in range(N)])
    mu1 = rho1 * np.exp(-z0)        # rho_min e^{-z0}
    mu2 = rho2 * np.exp(-z0)        # rho_max e^{-z0}

    for k in range(N-1):
        cons += [x[:,k+1] == x[:,k] + (A @ x[:,k] + B @ (g + u[:,k])) * dt]   # linear dynamics
        cons += [z[:,k+1] == z[:,k] - alpha * sigma[:,k] * dt]               # log-mass depletion

    cons += [cp.norm(u, axis=0) <= sigma[0,:]]                    # slack cone (lossless lift)
    cons += [e1 @ u >= sigma[0,:] * np.cos(theta_p)]              # pointing as a half-space
    dz = z[0,:] - z0
    cons += [mu1 * (1 - dz + cp.square(dz)/2) <= sigma[0,:]]      # floor: SOC
    cons += [sigma[0,:] <= mu2 * (1 - dz)]                        # ceiling: linear
    cons += [x[0,:] >= cp.norm(x[1:3,:], axis=0) * np.tan(gamma_gs)]  # glide slope (SOC)
    cons += [cp.norm(x[3:6,:], axis=0) <= v_max]                  # velocity cap

    prob = cp.Problem(cp.Minimize(cp.sum(sigma) * dt), cons)      # min fuel = int sigma
    prob.solve(solver=cp.CLARABEL)
    return prob.status, x, u, sigma, z
    # recover thrust: T_c = exp(z) * u ; at optimum ||u|| = sigma, so rho_min <= ||T_c|| <= rho_max exactly
```

The slack collapses onto the true thrust magnitude at the optimum, so this single convex SOCP returns the exact, globally optimal, minimum-fuel powered-descent trajectory; an outer golden-section loop over tf (and the two-step miss-distance/fuel prioritization for diverts) wraps it.
