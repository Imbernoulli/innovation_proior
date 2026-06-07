# Synthesis — Lossless convexification for minimum-fuel powered-descent guidance

## The pain point / research question
Land a rocket (Mars / lunar / VTVL) softly at a target with **minimum fuel** (= maximum landed/payload mass),
subject to: translational dynamics under gravity, **mass depletion** (fuel burned ∝ thrust magnitude),
a **throttleable engine that cannot go below ρ_min** (combustion stability) nor above ρ_max,
a **glide-slope** cone (stay above terrain on approach), a thrust **pointing** cone (tilt limit), velocity bound,
and fixed boundary conditions (start state, zero terminal velocity, zero terminal altitude).
Must be solvable **onboard, in bounded time, to a guaranteed global optimum** (no local-minimum gambling, no
warm-start dependence) — this is what rules out generic nonlinear programming.

## The objects
- State: r (position, 3), ṙ (velocity, 3), m (mass).
- Control: thrust vector T_c ∈ R³, magnitude ‖T_c‖.
- Dynamics (constant-g, planet-frame with Coriolis/centripetal S(ω)):
  r̈ = g + T_c/m − 2 S ṙ − S² r ;  ṁ = −α ‖T_c‖,  α = 1/(I_sp g0) (mass flow ∝ thrust).
- Minimum fuel = min ∫₀^{tf} ‖T_c(t)‖ dt  (⇔ max m(tf), since ṁ = −α‖T_c‖).

## The killer nonconvexity
Thrust magnitude must live in an **annulus**: ρ_min ≤ ‖T_c(t)‖ ≤ ρ_max.
Upper bound ‖T_c‖ ≤ ρ_max is a convex ball — fine. **Lower bound ‖T_c‖ ≥ ρ_min is the complement of a ball
— nonconvex.** Also ṁ = −α‖T_c‖ makes the dynamics nonlinear (m in denominator of T_c/m).
And a "natural" pointing constraint n̂ᵀT_c ≥ ‖T_c‖cos θ is nonconvex for θ > 90°.

## Fix 1 — slack-variable relaxation (the lift)
Introduce slack Γ(t) ∈ R. Replace the annulus by:
  ‖T_c(t)‖ ≤ Γ(t)  (convex cone),  ρ_min ≤ Γ(t) ≤ ρ_max  (convex box).
Use Γ in the cost and in the mass dynamics: min ∫ Γ dt, ṁ = −α Γ.
Geometrically: lift the nonconvex annulus in R³ into the convex solid V in R⁴ = {(T_c,Γ): ρ_min≤Γ≤ρ_max, ‖T_c‖≤Γ}.
This relaxation ENLARGES the feasible set (now ‖T_c‖ < ρ_min is allowed, when Γ ≥ ρ_min and ‖T_c‖ ≤ Γ).

## Fix 2 — change of variables (kill the nonlinear dynamics)
m in the denominator is nonlinear. Define
  u = T_c/m  (mass-normalized thrust = commanded acceleration),
  σ = Γ/m    (mass-normalized slack),
  z = ln m   (log-mass).
Then dynamics become LINEAR:
  r̈ = g + u − 2Sṙ − S²r,  ż = ṁ/m = −α Γ/m = −α σ.
Slack relaxation in new vars:  ‖u‖ ≤ σ.
But the box ρ_min ≤ Γ ≤ ρ_max becomes ρ_min e^{−z} ≤ σ ≤ ρ_max e^{−z}  (since σ = Γ/m = Γ e^{−z}) — exponential in z, NONCONVEX again on the σ-side.

## Fix 3 — Taylor / SOC approximation of the exponential bounds
Let z₀(t) = ln(m_wet − α ρ_max t)  (mass profile if burning at max thrust — a known function of t).
Expand e^{−z} about z₀ with δ = z−z₀:
- Lower bound (the one that must stay a *convex set boundary that σ lies above*):
  ρ_min e^{−z} ≈ μ₁(t)[1 − (z−z₀) + (z−z₀)²/2],  μ₁ = ρ_min e^{−z₀}.  This keeps **3 Taylor terms** → it is a convex quadratic lower-bounding function ≤ σ → a second-order-cone constraint (a convex quadratic ≤ affine-of-σ). KEEP it as a lower bound on σ.
- Upper bound:  ρ_max e^{−z} ≈ μ₂(t)[1 − (z−z₀)],  μ₂ = ρ_max e^{−z₀}.  Only **2 terms (linear)** — because requiring σ ≤ (a convex quadratic) would be nonconvex; the linear under-approximant keeps σ ≤ affine, convex.
Result (continuous-time convex problem):
  min ∫ σ dt
  s.t. r̈ = g + u − 2Sṙ − S²r,  ż = −α σ,  ‖u‖ ≤ σ,
       μ₁[1 − (z−z₀) + (z−z₀)²/2] ≤ σ ≤ μ₂[1 − (z−z₀)],
       glide slope: e₁ᵀr ≥ tan(γ_gs)·‖[r₂;r₃]‖  (or 4-facet affine approx),
       pointing: n̂ᵀu ≥ σ cos θ_p,  velocity ‖ṙ‖ ≤ v_max,
       boundary conds, z(0)=ln m_wet, z(tf) ≥ ln m_dry.

## Losslessness proof (Pontryagin / maximum principle)
Claim: at the optimum of the relaxed problem, **‖u*(t)‖ = σ*(t) a.e.** — the relaxation is TIGHT, so the relaxed
optimum is feasible AND optimal for the original nonconvex annulus problem.
Argument (costate / primer vector):
- Hamiltonian H = −σ + λ_rᵀ ṙ + λ_vᵀ(g+u−...) − α λ_z σ  (sign conventions: cost −σ for a max-Hamiltonian form). Group the σ terms: coefficient (−1 − α λ_z) is the "throttle switching function" R(t); the u terms: λ_vᵀ u with ‖u‖ ≤ σ.
- Pontryagin: maximize H over (u,σ) in the feasible (u,σ) set. For fixed σ, the inner max over u with ‖u‖≤σ is achieved at u = σ·λ_v/‖λ_v‖ whenever λ_v ≠ 0 → so ‖u*‖ = σ* and u* aligned with the primer vector λ_v. The only way ‖u*‖ < σ* at optimum is if λ_v(t) = 0.
- Costate dynamics: λ_v̇ = −∂H/∂ṙ etc. give a linear adjoint ODE; λ_v(t) ≡ 0 on a positive-measure interval would force (via controllability / normality of {A,B} — Condition 1) the entire adjoint to vanish identically, contradicting the transversality/nontriviality condition (Condition 2: the terminal multipliers are nonzero). Hence λ_v ≠ 0 a.e., so ‖u*‖ = σ* a.e.
- Therefore the slack inequality ‖u‖ ≤ σ holds with EQUALITY at the optimum → σ = ‖u‖ ∈ [ρ_min e^{−z}, ρ_max e^{−z}] → recover ρ_min ≤ ‖T_c‖ ≤ ρ_max. No feasible point of the original is removed (lossless) and the global optimum is attained by a convex SOCP.
Caveat: with state constraints (glide slope), the guarantee needs them active only at isolated instants (Theorem holds if boundary touched at ≤ finitely many points) — empirically true for Mars landing (touches glide-slope cone at most once before touchdown).

## Discretization → SOCP
Time grid t_k, k=0..N−1, dt. Piecewise-linear control / first-order-hold or zero-order-hold; here simple Euler:
  x_{k+1} = x_k + (A x_k + B(g + u_k)) dt,  z_{k+1} = z_k − α σ_k dt.
Every constraint (‖u_k‖≤σ_k cone, glide-slope cone, pointing halfspace, Taylor box) is SOC or linear → a finite SOCP,
solved by interior-point in bounded iterations (here CLARABEL via cvxpy).
Free final time tf: outer **golden-section line search** over tf (cost unimodal in tf).
Minimum-landing-error variant: when target unreachable, first minimize ‖r(tf)−target‖ (Problem 4), then minimize fuel
subject to that achieved miss distance (prioritized two-step) — this is the G-FOLD large-divert form.

## Design decisions → why
- Slack Γ vs. direct annulus: annulus nonconvex; Γ lifts to convex AND is provably tight (lossless) — so no optimality lost. Why not just drop ρ_min? Because then the optimizer would coast at ‖T‖<ρ_min which the engine physically can't do; losslessness is what lets us drop it *honestly*.
- Γ in cost (min ∫Γ) not ∫‖T‖: makes cost linear in the variable that also appears in ż; and at optimum Γ=‖T‖ so it IS fuel.
- log-mass z = ln m: turns ṁ=−αΓ multiplicative coupling (T_c/m) into linear ż=−ασ; price is exponential bounds.
- 3-term Taylor on lower bound, 2-term on upper: lower must stay ≤ σ and be convex (quadratic-≤-affine = SOC ok); upper must stay ≥ σ → σ ≤ linear (a convex-quadratic upper bound would be nonconvex), so drop to linear under-approximant. Error analytically bounded & tiny over the flight.
- α = 1/(I_sp g0): mass flow per Newton; ṁ = −α‖T‖ is the rocket equation differential form.
- Golden search on tf: cost unimodal in tf, derivative-free, robust.
- Pointing as n̂ᵀu ≥ σ cos θ: halfspace in (u,σ), convex even for θ>90° (vs nonconvex n̂ᵀu ≥ ‖u‖cosθ).
- Glide slope as SOC e₁ᵀr ≥ tan(γ)‖r_lat‖ or 4 affine facets: keeps problem SOCP; affine facets fit the affine-state-constraint LCvx theorem cleanly.

## Sources
- Primary 2007: Açıkmeşe & Ploen, "Convex Programming Approach to Powered Descent Guidance for Mars Landing", JGCD 30(5):1353. (derivation via explainer + patent restatement; researchgate full text blocked.)
- Primary 2013: Açıkmeşe, Carson, Blackmore, "Lossless Convexification of Nonconvex Control Bound and Pointing Constraints of the Soft Landing Optimal Control Problem", IEEE TCST 21(6):2104.
- US Patent 8,489,260 B2 (Acikmese, Blackmore, Scharf) — restates Problems 1–6, change of variables (43)-(50), Taylor approx (49), prioritized algorithm, Lemmas 1-3 / Theorem 1.
- Survey: Malyuta et al., "Convex Optimization for Trajectory Generation" (arXiv 2106.09125) — Part I LCvx, Problems 6/7, slack input σ, Theorem 1 + Conditions 1-2 (maximum principle), pointing relaxation, glide-slope as affine state constraint, bang-bang.
- Explainer: tealquaternion.netlify.app/post/gfold-2007 — 2007 problem statement, Hamiltonian, switching function R₁₂, T_c* = (λ₂/‖λ₂‖)Γ*.
- Code: github.com/Natsoulas/lcvx-pdg (cvxpy + CLARABEL, soft-landing). Matches the change of variables, Taylor box, glide-slope, pointing.
