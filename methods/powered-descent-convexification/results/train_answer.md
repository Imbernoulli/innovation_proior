A spacecraft in final powered descent has to fly from a known position and velocity to a soft landing at, or as near as possible to, a surface target, and it has to do so while spending the least fuel possible, because the propellant it burns is exactly the mass margin it could otherwise have delivered as payload. The difficulty is that the engine is not an ideal acceleration source. A throttleable rocket has a minimum stable thrust set by combustion, a maximum thrust set by performance, a thrust-pointing limit, and the vehicle must stay inside a glide-slope cone above the terrain, respect a speed cap, and arrive essentially at rest at zero altitude. Writing the point-mass dynamics with position $r$, velocity $\dot r$, mass $m$, commanded thrust $T_c$, gravity $g$, and the planet-rotation skew matrix $S(\omega)$,
$$\ddot r = g + \frac{T_c}{m} - 2S(\omega)\dot r - S(\omega)^2 r, \qquad \dot m = -\alpha\,\|T_c\|, \quad \alpha = \frac{1}{I_{sp}\,g_0},$$
and integrating the mass equation gives $m(t_f)=m_0-\alpha\int_0^{t_f}\|T_c\|\,dt$, so minimizing fuel is the same as maximizing terminal mass. The trouble is twofold. First, the admissible thrust set $\rho_{\min}\le\|T_c\|\le\rho_{\max}$ with $0<\rho_{\min}<\rho_{\max}$ is an annulus: the upper bound is a convex ball, but the lower bound is the *outside* of a ball, so two legal lower-magnitude thrust vectors can average to zero, which is illegal — the set is nonconvex. Second, the acceleration term $T_c/m$ couples a state into the control nonlinearly. The options on the table all fail in a specific way. A direct nonlinear transcription — collocation, pseudospectral, SQP, shooting — keeps the physics but inherits the nonconvexity, so it returns a local minimum, depends on the initial guess, and gives no global certificate for an onboard firing decision. Apollo-style polynomial guidance is fast but does not optimize fuel under the full constraint suite. One-dimensional soft-landing analysis brackets feasible flight times but cannot solve the three-dimensional divert. And simply dropping the lower thrust bound to make the set a convex ball is unfaithful: the optimizer then coasts through the forbidden hole, producing a trajectory the engine physically cannot fly. What we need is a formulation whose optimum is provably global, whose solve time is bounded, and that does not require a guessed trajectory — which points at second-order cone programming, if only the annulus and the mass coupling can be removed honestly.

I propose to solve this by lossless convexification of the powered-descent problem. The central move is a relaxation that I can prove costs nothing at the optimum. Pontryagin's maximum principle already shows the way: the thrust direction enters the Hamiltonian through the velocity costate $\lambda_v$, and the magnitude enters linearly through the running fuel term and the mass costate, so once the direction is fixed the optimal magnitude is driven to an endpoint of the admissible interval rather than sitting in the interior — the classic bang-bang max-min-max soft-landing structure. The optimum is already trying to live on the boundary of the nonconvex hole. So I introduce a scalar slack $\Gamma$ that bounds the thrust magnitude and use it in place of $\|T_c\|$ in the mass equation,
$$\|T_c\| \le \Gamma, \qquad \rho_{\min}\le\Gamma\le\rho_{\max}, \qquad \dot m = -\alpha\,\Gamma.$$
The lifted set in $(T_c,\Gamma)$ is convex — a second-order cone $\|T_c\|\le\Gamma$ sliced by two scalar bounds on $\Gamma$. It is strictly larger than the true annulus, because $\Gamma$ may sit at $\rho_{\min}$ while $\|T_c\|$ is smaller, which is exactly the unphysical hole. The relaxation only helps if the optimizer is forced back onto $\|T_c\|=\Gamma$, and it is: for fixed $\Gamma$, the direction-dependent term is $\lambda_v^\top T_c/m$, so when $\lambda_v\ne0$ maximizing over $\|T_c\|\le\Gamma$ pushes $T_c$ to the cone boundary and aligns it with $\lambda_v$, giving $\|T_c\|=\Gamma$. Slack can survive only where $\lambda_v=0$, but if $\lambda_v$ vanished over an interval, the linear adjoint dynamics together with the controllability of the translational system and the maximum principle's nontriviality condition would force *all* multipliers to vanish, a contradiction. Hence $\lambda_v$ vanishes only on a measure-zero set and
$$\|T_c^*(t)\| = \Gamma^*(t) \quad \text{almost everywhere}.$$
That is the lossless property: the convex lift adds unphysical points the optimum never uses. With the glide-slope state constraint a multiplier enters the adjoint equations, so the argument needs the trajectory to touch the cone only at isolated instants rather than ride it over a finite arc — the normal powered-descent geometry, where contact is isolated.

The dynamics still carry $T_c/m$, so I change variables to remove the mass nonlinearity, taking $u = T_c/m$, $\sigma = \Gamma/m$, and $z = \ln m$. The acceleration becomes $g + u - 2S\dot r - S^2 r$, which is *linear* in $[r;\dot r;z]$ and $[u;\sigma]$, and the mass equation becomes $\dot z = \dot m/m = -\alpha\Gamma/m = -\alpha\sigma$. The cone survives as $\|u\|\le\sigma$. The price is that the thrust box turns exponential: $\rho_{\min}e^{-z}\le\sigma\le\rho_{\max}e^{-z}$. There is one scalar exponential to tame, and the *direction* of every approximation must be chosen so the result is both convex and conservative — never claiming thrust the engine cannot deliver. The key device is to pin the approximation to a reference log-mass profile. The maximum-thrust profile is a lower bound on log mass,
$$z_0(t) = \ln\!\big(m_0 - \alpha\,\rho_{\max}\,t\big),$$
and the minimum-thrust profile gives the upper envelope, so $z_0(t)\le z(t)\le \ln(m_0-\alpha\rho_{\min}t)$, which makes $\delta = z - z_0$ nonnegative. That sign is what makes the Taylor truncations meaningful. Since $e^{-\delta} = 1 - \delta + \tfrac{\delta^2}{2} - \tfrac{\delta^3}{6}+\cdots$, for $\delta\ge0$ the quadratic $1-\delta+\tfrac{\delta^2}{2}$ lies *above* $e^{-\delta}$ while the tangent $1-\delta$ lies *below* it. For the lower thrust floor $\rho_{\min}e^{-z}\le\sigma$ I therefore enforce the stronger, safely-above condition with the quadratic, and for the upper ceiling $\sigma\le\rho_{\max}e^{-z}$ I use the affine tangent, which sits conservatively under the true ceiling:
$$\mu_1(t)\big[1 - \delta + \tfrac{1}{2}\delta^2\big] \le \sigma, \quad \mu_1(t)=\rho_{\min}e^{-z_0(t)}, \qquad \sigma \le \mu_2(t)\big[1-\delta\big], \quad \mu_2(t)=\rho_{\max}e^{-z_0(t)}.$$
The floor is a convex quadratic bounded above by an affine variable, which is second-order-cone-representable; the ceiling is affine. Picking the quadratic for the floor and the tangent for the ceiling is forced: the quadratic on the ceiling would put $\sigma$ below a convex function (nonconvex), and the tangent on the floor would understate the true minimum thrust. The objective also needs care, because $\sigma$ is $\Gamma/m$, not the physical fuel rate $\Gamma$; but $z(t_f)=z(0)-\alpha\int\sigma\,dt$, so minimizing $\int\sigma\,dt$ maximizes terminal log mass, and since log is monotone that maximizes terminal mass — minimizing $-z_N$ and minimizing the quadrature of $\sigma$ are the same fuel objective up to constants. The remaining constraints stay conic. The glide slope is the second-order cone $e_1^\top r \ge \tan(\gamma_{gs})\|r_{\text{lat}}\|$; the speed cap is $\|\dot r\|\le v_{\max}$. The pointing limit is the interesting one: the original $n^\top T_c \ge \|T_c\|\cos\theta_p$ is nonconvex for $\theta_p>90^\circ$, but in the lifted normalized variables it becomes $n^\top u \ge \sigma\cos\theta_p$, a half-space in $(u,\sigma)$ even when $\cos\theta_p<0$ — a relaxation while $\sigma$ may exceed $\|u\|$, but the same equality $\|u^*\|=\sigma^*$ at the optimum recovers the true pointing constraint. With $u,\sigma$ piecewise linear, the faithful no-rotation transcription carries the quadrature factors $\dot r_{k+1}=\dot r_k+\tfrac{\Delta t}{2}(u_k+u_{k+1})+g\Delta t$, $r_{k+1}=r_k+\tfrac{\Delta t}{2}(\dot r_k+\dot r_{k+1})+\tfrac{\Delta t^2}{12}(u_{k+1}-u_k)$, $z_{k+1}=z_k-\tfrac{\alpha\Delta t}{2}(\sigma_k+\sigma_{k+1})$; the compact solver below uses the matching Euler form $x_{k+1}=x_k+(Ax_k+B(g+u_k))\Delta t$, $z_{k+1}=z_k-\alpha\sigma_k\Delta t$, with the conic constraints and lossless lift unchanged. The fixed-flight-time problem is now a single SOCP solved globally; the final time, which sits outside the fixed grid, is handled by a one-dimensional golden-section search over $t_f$, valid because the fuel-versus-flight-time curve has a single useful minimum on the feasible bracket. And if the prescribed target is unreachable, I do not stop at infeasible: I first solve a closest-landing problem that minimizes the lateral miss distance under the same safe-descent constraints, then re-solve the fuel problem while holding that achieved miss-distance radius fixed, which restores the fuel-optimal trajectory among the closest reachable landings.

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
