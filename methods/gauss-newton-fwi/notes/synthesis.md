# Synthesis — FWI via adjoint-state method

## Pain point / goal
Reconstruct a spatial velocity model c(x) (or squared slowness m=1/c²) of the subsurface from recorded
seismograms d_obs at receivers, by minimizing the L2 waveform misfit J(m)=½ Σ_s ∫₀ᵀ ‖ S u_s(t) − d_s(t) ‖² dt,
where u_s solves the acoustic wave equation for shot s. Model has MILLIONS of grid-point parameters; one scalar
misfit. Need ∇J. Brute force / forward sensitivity = one wave simulation per parameter → millions of solves. Wall.

## The intellectual move (adjoint-state, = jameson-adjoint applied to wave eq)
Treat converged/forward wave state as constraint F(u,m)=0 (the wave PDE). Lagrangian L=h(u,m)−⟨λ,F(u,m)⟩.
Choose λ (adjoint/costate field) to annihilate the ∂J/∂u·δu term: adjoint equation ∂F/∂u^* λ = ∂h/∂u.
Then ∇J = ∂h/∂m − ⟨λ, ∂F/∂m⟩. One forward solve (u) + one adjoint solve (λ) → gradient w.r.t. ALL params.
This is the SAME associativity / right-to-left grouping move as Jameson; here F = wave operator.

## Exact equations (Plessix 2006, grounded)
Forward (time domain, m=σ²=1/c²):  L u_s = f_s, L = m ∂²/∂t² − Δ ;  u_s(0)=0, ∂_t u_s(0)=0.
Misfit (eq 29): J = ½ Σ_{s,r} ∫₀ᵀ (S_{s,r} u_s − d_{s,r})² dt.
Adjoint system (eq 32):  m ∂²λ_s/∂t² − Δλ_s = Σ_r S^T_{s,r}(S_{s,r}u_s − d_{s,r}),  FINAL conditions λ_s(T)=0, ∂_tλ_s(T)=0.
Change of variable q_s(t)=λ_s(T−t) (eq 34) → time-reversed forward solve (eq 35): q_s(0)=0, source = residual(T−t) injected at receivers.
GRADIENT (eq 33/36):  ∂J/∂m(x) = − Σ_s ∫₀ᵀ λ_s(x,t) ∂²u_s(x,t)/∂t² dt
                                = − Σ_s ∫₀ᵀ q_s(x,T−t) ∂²u_s(x,t)/∂t² dt.
So gradient = − zero-lag time correlation of (∂²u/∂t²) with the back-propagated residual field, summed over shots.
[Frequency-domain check, eq 27:  ∂J/∂m(x) = Re Σ_ω Σ_s ω² λ*_s(x,ω) u_s(x,ω) — the −ω² ↔ ∂²/∂t².]

Discrete (Devito, grounded): forward `m u.dt2 - u.laplace + damp*u.dt = src*dt²/m`;
adjoint `m v.dt2 - v.laplace - damp*v.dt = residual injected at rec`, run backward in time;
gradient `g += -u * v.dt2` (m = 1/c²).

## Velocity parameterization scaling (2/c³)
m = 1/c² ⇒ ∂m/∂c = −2/c³.  Chain rule: ∂J/∂c = (∂J/∂m)(∂m/∂c) = (−2/c³)·(∂J/∂m)
= (−2/c³)·(− Σ_s ∫ λ_s ∂²u_s/∂t² dt) = (2/c³) Σ_s ∫ λ_s ∂²u_s/∂t² dt.
So for a velocity model the gradient carries an extra factor (−∂m/∂c) = +2/c³ (equivalently 2/(ρc³) if a density factor
is kept; in constant-density acoustic it is 2/c³). Magnitude factor 2/c³; sign follows the chain rule above.

## Gauss-Newton / Hessian
Frechet/Born derivative J(δm) → linearized scattered data. Gauss-Newton Hessian H_GN = J^T J (Re part),
ignores second-derivative (multiple-scattering) term of full Hessian. Pratt-Shin-Hicks 1998. GN descent pk = -H_GN^{-1} g.
Diagonal of H_GN ~ zero-lag autocorrelation of partial-derivative (virtual-source) wavefields → acts as
geometrical-spreading / illumination preconditioner. In practice approximated by pseudo-Hessian / source-illumination
diagonal precond, or solved with conjugate gradient (no explicit Hessian).

## Non-convexity: cycle skipping + multiscale
Oscillatory seismograms → L2 misfit highly multimodal. Cycle skipping: if predicted trace is more than HALF A PERIOD
off the observed trace, gradient pushes toward matching the WRONG cycle → spurious local minimum (Virieux & Operto 2009;
necessary condition: error < half period). The lower the frequency, the wider the half-period ⇒ fewer minima.
Multiscale remedy: invert lowest frequencies / earliest arrivals first (build smooth long-wavelength background),
then progressively add higher frequencies. Frequency continuation (Bunks/Sirgue/Pratt; Sirgue & Pratt 2004).

## Ancestors (in-frame citations OK)
- Lions 1971 — optimal control of PDEs, costate/adjoint as Lagrange multiplier field.
- Bryson & Ho 1975 — backward costate sweep gives gradient w.r.t all controls (template).
- Chavent 1974 — adjoint-state in inverse problems.
- Lailly 1983 — migration = gradient of L2 misfit = backprop residual correlated with forward field.
- Claerbout imaging principle — gradient kinematically = correlation of source & receiver wavefields (RTM).
- Tarantola 1987 — least-squares inverse theory / generalized least squares (background, in-frame).
- Pratt/Shin/Hicks 1998 — Gauss-Newton & full Newton in frequency-space FWI.
- Sirgue & Pratt 2004; Bunks et al 1995 — multiscale frequency strategy.
Note: Tarantola 1984 and Virieux & Operto 2009 are the TARGET papers — must NOT be cited as artifacts in deliverables.

## Code scaffold (pre-method): a forward FD acoustic solver + cost + a getGradient stub (FD baseline) + descent loop.
Final code fills: adjoint (time-reversed residual) solver + gradient by correlating u.dt2 with backprop field + velocity update.
