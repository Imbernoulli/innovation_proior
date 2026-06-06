OK, let me think this through from scratch. I have a rocket on powered descent — Mars lander, say — and I want it to touch down softly at a target while burning as little fuel as possible, because fuel spent on descent is payload I didn't carry. I get to command a thrust vector and throttle setting, in real time, onboard. The constraints are physical: the engine can only throttle inside a band, the vehicle has to stay above the terrain on the way down, the thrust can't tilt too far from vertical, mass strictly drops as I burn. And the hard requirement that colors everything: this has to be solved autonomously in a bounded amount of time, and I have to *trust* the answer — a local minimum that depends on a lucky initial guess is no good when I'm about to fire the descent engine. So really the question is: can I pose minimum-fuel landing in a form where I can get the global optimum, fast, with a certificate?

Let me write the physics first. Point mass, constant gravity g, and since the planet rotates there are Coriolis and centripetal terms through the angular rate ω. Position r, velocity ṙ, mass m, commanded thrust T_c. Newton plus the rocket equation:

  r̈ = g + T_c/m − 2 S(ω) ṙ − S(ω)² r,   ṁ = −α ‖T_c‖,

where S(ω) is the skew matrix of ω and α = 1/(I_sp g₀) is propellant burned per Newton. Fuel is the whole game, and fuel is just ∫‖T_c‖ dt up to the constant α, since integrating ṁ gives m(tf) = m₀ − α∫‖T_c‖. So minimum fuel ⇔ minimize ∫₀^{tf} ‖T_c‖ dt ⇔ maximize the landed mass m(tf). Good, three ways of saying the same thing.

Boundary conditions: start at (r₀, ṙ₀, m₀), end with ṙ(tf) = 0 and the altitude component of r(tf) = 0 — a soft touchdown. Mass must not drop below the dry mass: m(tf) ≥ m_dry. And the safety constraints: stay in a glide-slope cone (don't fly into terrain on approach), keep the thrust tilt bounded, maybe cap the speed.

Now, what does the optimizer actually look like? Let me reach for Pontryagin's maximum principle, because that tells me the *structure* of the optimal control, which I'll want to exploit. I form the Hamiltonian with costates λ_r, λ_v, λ_m on r, ṙ, m. The control T_c enters two places: linearly in r̈ through T_c/m (so through λ_vᵀ T_c/m), and through the running cost ‖T_c‖ and the mass dynamics −α‖T_c‖ (so through the magnitude). The principle says the optimal T_c maximizes the Hamiltonian pointwise over the admissible thrust set. Split T_c into magnitude and direction. For the *direction*: the only direction-dependent term is λ_vᵀ T_c/m, so the optimal thrust points along λ_v — the primer vector — and I'll want ‖T_c‖ as large as the magnitude trade allows in that direction. For the *magnitude*: collect the scalar coefficient multiplying ‖T_c‖ from the cost, the mass costate, and the projection onto λ_v; it's a linear function of ‖T_c‖, so the magnitude is pushed to a boundary of its allowed interval. That's the bang-bang structure — thrust sits at ρ_max or at ρ_min, switching when a scalar switching function changes sign. Minimum-fuel rocket descent is "burn hard, throttle to the floor, burn hard." Hold onto that; it's going to matter more than it looks.

So where's the difficulty? The engine can't throttle arbitrarily. Combustion stability sets a floor ρ_min > 0; performance sets a ceiling ρ_max. So at every instant

  ρ_min ≤ ‖T_c(t)‖ ≤ ρ_max.

Stare at the set this carves out in thrust space. The upper bound ‖T_c‖ ≤ ρ_max is a ball — convex, lovely. The lower bound ‖T_c‖ ≥ ρ_min is the *complement* of a ball. That's nonconvex. The admissible thrust set is an annulus, a shell with a hole in the middle, and the hole is exactly the trouble: take two thrust vectors of magnitude ρ_min pointing opposite ways, average them, and you land at the origin — magnitude zero, inside the forbidden hole. A convex set can't have a hole. So the feasible control set is nonconvex, and it's the *lower* bound specifically that does it.

This is the crux. If the feasible set were convex and the objective convex, I'd have a convex program, and then every local minimum is global and an interior-point method solves it to fixed accuracy in a bounded number of iterations — exactly the bounded-time, certified, global-optimum behavior I need onboard. But the annulus kills convexity. And I can't just delete the lower bound — that's the obvious move and it fails. If I relax to only ‖T_c‖ ≤ ρ_max (a ball, convex), the solver will happily command thrust below ρ_min, even zero, wherever the unconstrained optimum wants to coast. But the physical engine *cannot* run below ρ_min. So the relaxed optimum is unflyable; the relaxation is unfaithful. Dropping ρ_min throws away the constraint I most need to respect.

And there are two more nonconvexities lurking. The dynamics themselves are nonlinear: T_c/m has the control divided by the mass *state*. And if I want a pointing constraint — keep the thrust within θ_p of vertical, n̂ᵀT_c ≥ ‖T_c‖ cos θ_p — that's nonconvex too once θ_p > 90°, an obtuse pie-slice. So I've got three sources of nonconvexity: the annulus lower bound, the nonlinear mass coupling, and the obtuse pointing cone. Let me take them one at a time, hardest first.

The annulus. I want to convexify it *without lying* — without enlarging the feasible set in a way that lets the optimizer escape into physically-impossible region and stay there. Here's the thing that nags at me: the bang-bang structure says the magnitude *wants* to be at a boundary anyway. The optimizer is never going to *want* to sit strictly inside the annulus in the radial direction — it wants to be pressed against ρ_max or ρ_min. So maybe I can hand it a *bigger*, convex set, and trust that the optimum it finds inside the bigger set lands on the boundary I care about anyway. Let me try to make that precise.

Introduce a new scalar variable Γ(t) — a slack — that stands in for the thrust magnitude. Replace the annulus with:

  ‖T_c(t)‖ ≤ Γ(t),    ρ_min ≤ Γ(t) ≤ ρ_max,

and use Γ, not ‖T_c‖, in the cost and the mass dynamics: minimize ∫ Γ dt, and ṁ = −α Γ. Look at what each piece is. ‖T_c‖ ≤ Γ is a second-order cone — convex. ρ_min ≤ Γ ≤ ρ_max is a box on a scalar — convex. The cost ∫Γ is linear — convex. So the whole thing is convex now. Geometrically I've lifted the flat nonconvex annulus in 3D thrust space up into a 4D set in (T_c, Γ): an ice-cream-cone solid ‖T_c‖ ≤ Γ, sliced between the planes Γ = ρ_min and Γ = ρ_max. That solid is convex, and it has no hole — because the hole got filled by allowing ‖T_c‖ to be *anything* up to Γ, including small.

But wait — that's exactly the worry from before. This relaxation *enlarges* the feasible set. Now (T_c, Γ) with Γ = ρ_min but ‖T_c‖ = 0 is allowed: thrust zero, but the "magnitude variable" Γ sitting at ρ_min. That's the unflyable coast again, smuggled in through the slack. So why would this be any better than just dropping ρ_min?

Here's the difference, and this is the whole idea. I'm not just relaxing — I'm going to *prove the relaxation is tight*: that at the optimum, ‖T_c*‖ = Γ* always, so the slack collapses onto the true magnitude and the recovered T_c* genuinely satisfies ρ_min ≤ ‖T_c*‖ ≤ ρ_max. If I can show that, then solving the convex problem hands me back an exact, feasible, globally-optimal solution of the original nonconvex one — *losslessly*, with no part of the real feasible region removed and no spurious unflyable optimum. The bang-bang intuition is what makes me believe this is possible: the optimizer wants the magnitude pressed to a boundary, so it shouldn't want to leave Γ > ‖T_c‖ slack.

Let me try to actually prove ‖T_c*‖ = Γ* at the optimum, because the whole edifice rests on it. I'll defer the dynamics-linearization for a moment and argue on the relaxed problem. Take an optimal solution of the convex problem with control direction/magnitude T_c*(t) and slack Γ*(t). Run Pontryagin on the relaxed problem. The cost is ∫Γ; the slack Γ enters the cost (coefficient from the running cost), the mass dynamics (through −αΓ, picking up the mass costate λ_m), and bounds it against ‖T_c‖ via ‖T_c‖ ≤ Γ. The thrust vector T_c enters the dynamics through λ_vᵀ T_c/m. Maximize the Hamiltonian over admissible (T_c, Γ). For the *direction* of T_c: the only T_c-direction term is λ_vᵀ T_c/m, so the optimal T_c aligns with the velocity costate λ_v and uses all the magnitude it's allowed, i.e. ‖T_c*‖ = (its cap). Its cap is Γ (from ‖T_c‖ ≤ Γ). So whenever λ_v ≠ 0, the inner maximization drives ‖T_c*‖ all the way up to Γ* — equality. The *only* way to have ‖T_c*‖ < Γ* at the optimum is if λ_v(t) = 0, because then the direction term vanishes and the maximizer is indifferent to ‖T_c‖, so it needn't push it to Γ.

So the question reduces to: can λ_v(t) = 0 on a set of positive measure? Suppose it did, on some interval. The costates obey a *linear* adjoint ODE — λ̇_v = −∂H/∂r and friends, a linear system driven by the same A as the dynamics. If λ_v ≡ 0 on an interval, then differentiating the adjoint relations forces the other costates to vanish on that interval too, and by controllability of the linear pair {A, B} (any state reachable, the standard rank/PBH condition) the entire costate trajectory is pinned to zero — but the maximum principle's transversality (nontriviality) condition forbids all multipliers vanishing simultaneously; the terminal multipliers on the constrained final state are nonzero. Contradiction. So λ_v can vanish only on a measure-zero set, and therefore

  ‖T_c*(t)‖ = Γ*(t)   almost everywhere.

That's it — that's losslessness. The slack inequality holds with equality at the optimum, so Γ* = ‖T_c*‖ ∈ [ρ_min, ρ_max], which means the recovered T_c* satisfies the original annulus constraint exactly. No feasible point of the nonconvex problem was discarded (the relaxation only added points, and those added points are provably not optimal), and the convex optimum *is* the nonconvex optimum. I get to solve the easy convex problem and the answer is the true one. The precision of what's surprising here: the relaxed problem *does* contain unflyable points like (‖T_c‖ = 0, Γ = ρ_min), they're feasible, the optimizer just provably never chooses them — the bang-bang pull I noticed at the start is exactly the costate argument made rigorous.

One honest caveat I should flag for myself: this clean argument is for the magnitude bound with the boundary conditions and the controllability. Once I add the glide-slope *state* constraint, the maximum-principle argument has to carry an extra multiplier for the active state constraint, and the tightness guarantee survives only if that constraint is active at isolated instants, not over a whole sub-arc. For planetary landing this holds — the optimal trajectory grazes the glide-slope cone at most at a point or two (a mid-flight touch and touchdown), never sliding along it for a finite time. I'll lean on that and check it on the computed trajectory rather than assume it blindly.

Now the second nonconvexity: the dynamics are nonlinear because of T_c/m. The slack already gave me a hint — I keep dividing thrust by mass. So let me change variables to *mass-normalized* quantities. Define

  u = T_c/m,   σ = Γ/m,   z = ln m.

u is just the commanded acceleration; σ is the mass-normalized slack; z is log-mass. Substitute. The translational dynamics:

  r̈ = g + T_c/m − 2Sṙ − S²r = g + u − 2Sṙ − S²r,

which is now *linear* in the state [r; ṙ] and the control u — the mass disappeared from the acceleration. And the mass dynamics: ż = ṁ/m = (−αΓ)/m = −α (Γ/m) = −α σ. Linear too. Beautiful — the change of variables traded the nonlinear coupling for a clean linear system [r; ṙ; z] with control [u; σ]. The slack relaxation in the new variables is just ‖u‖ ≤ σ (divide ‖T_c‖ ≤ Γ by m > 0). Still a second-order cone. 

But — and there's always a but — what happened to the box ρ_min ≤ Γ ≤ ρ_max? In the new variables Γ = σ m = σ e^z, so the box becomes

  ρ_min e^{−z} ≤ σ(t) ≤ ρ_max e^{−z}.

That's *exponential* in the state z. The σ ≤ ρ_max e^{−z} side and the σ ≥ ρ_min e^{−z} side are both nonconvex couplings between σ and z. I removed nonlinearity from the dynamics and it popped back up in the constraints. Whack-a-mole. But it's a *gentler* nonconvexity — a single exponential of a scalar — so maybe I can approximate it convexly without losing much, and bound the error.

z stays in a narrow range over the flight (mass changes by maybe a factor of two), so e^{−z} is well-behaved. Let me linearize/expand it around a known reference. The natural reference is the log-mass profile if I burned at maximum thrust the whole time: z₀(t) = ln(m_wet − α ρ_max t) — a known function of t, no optimization variable in it. Let δ = z − z₀, small. Taylor-expand e^{−z} = e^{−z₀} e^{−δ} = e^{−z₀}(1 − δ + δ²/2 − …).

Now I have to be careful about *which way* each inequality points, because that decides how many terms I can keep and stay convex. Take the lower bound first: ρ_min e^{−z} ≤ σ. With μ₁(t) := ρ_min e^{−z₀(t)}, this is μ₁(1 − δ + δ²/2 − …) ≤ σ. I want the left side to be a *convex* function of z (so that "convex ≤ σ" cuts out a convex region — a convex-quadratic lower-bounded-by, which is a second-order cone constraint). Keep three terms: μ₁[1 − (z−z₀) + (z−z₀)²/2]. That's a convex quadratic in z (positive coefficient on the square), and "convex quadratic ≤ σ" is exactly a rotated/second-order cone constraint. Keep it. The three-term truncation also *upper-bounds* the true e^{−δ} for the relevant range (since e^{−δ} ≤ 1 − δ + δ²/2 isn't generally true, but the truncation error is analytically bounded and tiny over the flight's small δ), so the constraint is slightly conservative on the floor — acceptable.

Now the upper bound: σ ≤ ρ_max e^{−z}. With μ₂(t) := ρ_max e^{−z₀(t)}, this is σ ≤ μ₂(1 − δ + δ²/2 − …). Here's the asymmetry. If I kept three terms I'd be requiring σ ≤ (a convex quadratic in z) — and "σ ≤ convex" is a *nonconvex* constraint (the region under a convex curve isn't convex). So I *cannot* keep the quadratic on this side. Drop to two terms: σ ≤ μ₂[1 − (z−z₀)]. That's σ ≤ affine in z — a linear constraint, convex. I must use the linear under-approximant on the ceiling precisely because requiring a variable to be below a quadratic is nonconvex. The two truncations are not arbitrary — each is the longest truncation that keeps its inequality convex.

So the continuous-time relaxed, convexified, linearized problem is:

  minimize ∫₀^{tf} σ(t) dt
  subject to  r̈ = g + u − 2Sṙ − S²r,   ż = −α σ,
              ‖u‖ ≤ σ,
              μ₁(t)[1 − (z−z₀) + (z−z₀)²/2] ≤ σ ≤ μ₂(t)[1 − (z−z₀)],
              glide slope, pointing, velocity cap, boundary conditions,
              z(0) = ln m_wet,  z(tf) ≥ ln m_dry.

Let me deal with the remaining safety constraints, keeping everything inside the second-order-cone class. Glide slope: stay within a cone of half-angle γ_gs above the landing site, i.e. the lateral distance is bounded by the vertical times tan γ_gs. With e₁ the vertical (up) axis and r_lat the lateral components, that's e₁ᵀr ≥ tan(γ_gs)·‖r_lat‖ — a second-order cone constraint directly, or I can approximate the cone by a handful of affine facets n̂_iᵀr ≤ 0 (an intersection of half-spaces) if I want a purely linear state constraint that slots into the affine-state-constraint version of the losslessness theorem. Either is convex.

Pointing. I wanted n̂ᵀT_c ≥ ‖T_c‖ cos θ_p, which was nonconvex for θ_p > 90°. But now I have the slack. In the lifted (u, σ) space, write it as n̂ᵀu ≥ σ cos θ_p. This is a *half-space* — linear in (u, σ) — and a half-space is convex *regardless of the sign of cos θ_p*, even for an obtuse tilt limit. The slack variable I introduced for the magnitude problem pays off a second time: it convexifies the pointing constraint for free. And the same tightness argument extends — at the optimum ‖u‖ = σ, so n̂ᵀu ≥ σ cos θ_p = ‖u‖ cos θ_p recovers the original pointing cone. (The losslessness for pointing needs one more controllability-type condition on the dynamics restricted to the nullspace of n̂, but it holds for this double-integrator-like system.) Velocity cap: ‖ṙ‖ ≤ v_max, a second-order cone directly.

Everything is now linear or second-order-cone. Discretize. Put down a time grid t_k, k = 0,…,N−1, spacing dt. I'll carry the state x_k = [r_k; ṙ_k] (6 components) and z_k (log-mass) and the controls u_k (3), σ_k (1). Linear dynamics in the [r; ṙ] block: with A the 6×6 system matrix (zeros and identity in the top, −S² and −2S in the bottom) and B mapping accelerations into ṙ̇, a simple forward-Euler step

  x_{k+1} = x_k + (A x_k + B(g + u_k)) dt,

and the mass update z_{k+1} = z_k − α σ_k dt. (A first-order-hold or matrix-exponential discretization is more accurate, but Euler is the transparent version and the structure is identical.) The cone ‖u_k‖ ≤ σ_k at each node; the Taylor box on (z_k, σ_k) at each node with z₀(t_k) and μ₁(t_k), μ₂(t_k) precomputed; glide slope and pointing and velocity at each node; boundary conditions at k = 0 and k = N−1 (initial state fixed, final velocity zero, final altitude zero, z_{N−1} ≥ ln m_dry). The whole thing is a finite-dimensional second-order cone program. An interior-point SOCP solver returns the global optimum (or certified infeasibility) in a bounded number of iterations. That's the property I wanted from the start.

Two loose ends. First, the final time tf is itself a variable, and it sits in the grid (dt = tf/N) and in z₀(t), so it's *outside* the convex program. But the minimum fuel is unimodal in tf — there's a single best flight time, and longer or shorter both cost more. So I wrap the SOCP in a one-dimensional golden-section search over tf: for each candidate tf, build and solve the SOCP, read off the fuel, and let the search converge to the optimum. Derivative-free, robust, and each inner solve is convex.

Second, the divert / minimum-landing-error case. If the target is reachable, I minimize fuel to it directly. But on a large divert, or with a tight constraint set, the target may be *unreachable* — and then a single fuel-minimizing problem just reports "infeasible," which is useless (I'd still like to land as close as I can). So I prioritize: first solve for the minimum *miss distance* — minimize ‖r(tf) − target‖ subject to all the constraints — to find the closest achievable landing point; then, holding that miss distance fixed as a constraint, solve a second SOCP that minimizes fuel among all trajectories achieving it. Two convex solves, and the result is the least-fuel trajectory to the closest reachable site. (Note the losslessness for this minimum-error step needs the closest-point solution to also touch the state-constraint boundary at only isolated instants — the same glide-slope-grazing fact I flagged — which holds for the landing geometry.)

Let me put it in code, the way I'd actually build it with cvxpy and an interior-point SOCP backend. Variables x (state [r;v]), z (log-mass), u (mass-normalized thrust), σ (slack); constraints are exactly the pieces I derived.

```python
import numpy as np
import cvxpy as cp

# --- problem data (vehicle + environment + grid) ---
g     = np.array([-3.71, 0.0, 0.0])          # gravity [m/s^2]
omega = np.array([2.53e-5, 0.0, 6.62e-5])    # planet angular rate [rad/s]
m_wet = 2000.0; m_dry = 1700.0               # wet / dry-floor masses [kg] (dry = wet - usable fuel)
rho1  = 0.2 * 24000.0                         # rho_min thrust [N]  (engine throttle floor)
rho2  = 0.8 * 24000.0                         # rho_max thrust [N]  (engine throttle ceiling)
alpha = 5e-4                                  # mass flow per Newton [s/m] = 1/(Isp g0)
gamma_gs = np.deg2rad(30.0)                   # glide-slope half-angle
theta_p  = np.deg2rad(120.0)                  # thrust tilt limit (obtuse -> needed the half-space trick)
v_max    = 90.0
x0 = np.array([2400., 450., -330., -10., -40., 10.])   # [r; v]
N, dt = 50, 1.0
zi = np.log(m_wet); zf = np.log(m_dry)

# skew matrix of omega -> linear Coriolis/centripetal dynamics
S = np.array([[0, -omega[2], omega[1]],
              [omega[2], 0, -omega[0]],
              [-omega[1], omega[0], 0]])
A = np.block([[np.zeros((3,3)), np.eye(3)],
              [-S @ S,          -2*S]])       # x_dot = A x + B (g + u)
B = np.vstack([np.zeros((3,3)), np.eye(3)])
e1 = np.array([1.,0.,0.])                     # vertical/up axis

# --- decision variables (the lift: u and the slack sigma; z = ln m) ---
x     = cp.Variable((6, N))                   # state [r(3); v(3)]
z     = cp.Variable((1, N))                   # log-mass z = ln m  (change of variables)
u     = cp.Variable((3, N))                   # mass-normalized thrust  u = T_c / m
sigma = cp.Variable((1, N))                   # mass-normalized slack   sigma = Gamma / m

cons = []
# boundary conditions
cons += [x[:,0] == x0, z[0,0] == zi, z[0,N-1] >= zf]
cons += [e1 @ x[:3,N-1] == 0, e1 @ x[3:,N-1] == 0]    # touchdown: zero altitude & zero vertical vel
cons += [cp.norm(x[3:6,N-1]) <= 0.0 + v_max]          # (final speed handled by terminal cost too)

# reference max-thrust log-mass profile z0(t) for the Taylor expansion of the bounds
z0 = np.array([np.log(m_wet - alpha * rho2 * dt * k) for k in range(N)])
mu1 = rho1 * np.exp(-z0)                               # rho_min e^{-z0}
mu2 = rho2 * np.exp(-z0)                               # rho_max e^{-z0}

for k in range(N-1):
    # linear translational dynamics (mass divided out -> r_ddot = g + u)
    cons += [x[:,k+1] == x[:,k] + (A @ x[:,k] + B @ (g + u[:,k])) * dt]
    # log-mass depletion: z_dot = -alpha sigma
    cons += [z[:,k+1] == z[:,k] - alpha * sigma[:,k] * dt]

# slack cone: ||u|| <= sigma  (lifts the nonconvex annulus; tight at optimum)
cons += [cp.norm(u, axis=0) <= sigma[0,:]]
# thrust pointing as a half-space n^T u >= sigma cos(theta_p) (convex even for obtuse theta_p)
cons += [e1 @ u >= sigma[0,:] * np.cos(theta_p)]
# Taylor/SOC convex approximation of  rho_min e^{-z} <= sigma <= rho_max e^{-z}
dz = z[0,:] - z0
cons += [mu1 * (1 - dz + cp.square(dz)/2) <= sigma[0,:]]   # 3-term -> convex quadratic <= sigma (SOC)
cons += [sigma[0,:] <= mu2 * (1 - dz)]                     # 2-term -> sigma <= affine (linear)
# glide slope: vertical >= tan(gamma) * lateral distance  (second-order cone)
cons += [x[0,:] >= cp.norm(x[1:3,:], axis=0) * np.tan(gamma_gs)]
# velocity cap
cons += [cp.norm(x[3:6,:], axis=0) <= v_max]

# minimum-fuel objective: integral of sigma (= integral of Gamma/m; at optimum = true fuel rate)
objective = cp.Minimize(cp.sum(sigma) * dt)
prob = cp.Problem(objective, cons)
prob.solve(solver=cp.CLARABEL)
# recover physical thrust: T_c = m u = exp(z) u ;  at the optimum ||u|| = sigma, so the
# annulus rho_min <= ||T_c|| <= rho_max holds exactly -- the convex solve gave the true optimum.
```

The causal chain, start to finish: minimum-fuel soft landing is an optimal-control problem whose magnitude part is bang-bang, but the engine's throttle floor makes the admissible thrust an annulus, whose nonconvex lower bound blocks convex optimization; dropping the floor is unfaithful, so instead I lift the magnitude into a slack Γ with ‖T_c‖ ≤ Γ and ρ_min ≤ Γ ≤ ρ_max, which is convex, and prove via the maximum principle that the velocity costate can't vanish on an interval (controllability + nontriviality), so ‖T_c*‖ = Γ* at the optimum — the relaxation is lossless; the nonlinear T_c/m dynamics I kill with u = T_c/m, σ = Γ/m, z = ln m, giving linear dynamics at the price of exponential bounds, which I tame with a three-term Taylor on the floor (convex quadratic ≤ σ, an SOC) and a two-term on the ceiling (σ ≤ affine, since σ-below-a-quadratic would be nonconvex); the obtuse pointing cone convexifies for free as the half-space n̂ᵀu ≥ σ cos θ_p in the lifted space; glide slope and velocity stay as second-order cones; discretizing on a time grid yields a finite SOCP solved to global optimality in bounded time, wrapped in a golden-section search over the free final time, with a prioritized two-step miss-distance-then-fuel variant for diverts that can't reach the target.
