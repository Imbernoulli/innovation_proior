# Synthesis — Rectified Flow (2209.03003)

## Pain point at the time (2022)
- Diffusion/score SDEs (DDPM, NCSN, Song score-SDE) generate by reversing a noising SDE; sampling needs hundreds–thousands of NN calls because the learned reverse process is stochastic and its trajectories are curved.
- Probability-Flow ODE (Song 2020) / DDIM (Song 2020) convert the SDE to a deterministic ODE with the same marginals, allowing fewer steps. But the ODE drift comes from an SDE-to-ODE derivation, so its α_t,β_t schedules (exponential α from OU process, β=√(1−α²)) produce **curved** trajectories with **non-uniform speed** — so they still need many Euler steps and can't be made 1-step.
- Neural-ODE MLE (Chen 2018, FFJORD) trains an ODE by maximizing likelihood of Z_1; but (a) requires simulating the ODE every training step + backprop-through-time, expensive and unstable; (b) **under-specified**: infinitely many ODEs hit the same Z_1 marginal via different paths, so the path is left to optimizer accidents.
- GAN/VAE one-step models: GAN minimax instability/mode collapse; VAE/normalizing-flow need tractable-likelihood architecture constraints.
- OT: gives a clean "transport between two distributions" framing and convex transport costs E[c(Z_1−Z_0)], but classical OT solvers are slow in high-d, and c-optimal ≠ better generation.

## Key idea (discovery order)
The bottleneck is *curvature*: a curved ODE needs many steps; a perfectly straight constant-speed ODE Z_t = (1−t)Z_0 + tZ_1 needs **one** Euler step. So instead of inheriting a schedule from an SDE, directly aim for the straightest transport.
- Straightest object connecting a sample pair (X_0,X_1) is the line X_t=(1−t)X_0+tX_1, with constant velocity X_1−X_0. But the "ODE" dX_t=(X_1−X_0)dt is **non-causal** (needs X_1, the future). Lines from different pairs **cross** at shared points, so there is no single-valued velocity field along them.
- Fix: **causalize** by regressing a velocity field v(x,t) onto X_1−X_0:
  min_v ∫_0^1 E[‖(X_1−X_0) − v(X_t,t)‖²] dt, X_t=(1−t)X_0+tX_1.
  Optimal v^X(x,t)=E[X_1−X_0 | X_t=x] — the *average* line direction through (x,t). This averages out crossings → an honest single-valued ODE field. Trajectories of dZ_t=v^X(Z_t,t)dt never cross (ODE uniqueness).

## Three theorems (all proved inline in reasoning)
1. **Marginal-preserving (Thm 3.3):** Law(Z_t)=Law(X_t) ∀t, so Z_1~π_1 if Z_0~π_0; (Z_0,Z_1) is a valid coupling.
   Proof: for test h, d/dt E[h(X_t)] = E[∇h(X_t)·Ẋ_t] = E[∇h(X_t)·v^X(X_t,t)] (tower property, conditioning on X_t). This says ρ_t=Law(X_t) solves the continuity equation ∂_t ρ_t + div(v^X_t ρ_t)=0 weakly (integrate-by-parts equivalence). Z_t is driven by the *same* v^X from the *same* initial law (Z_0=X_0), so Law(Z_t) solves the same continuity equation with same IC → equal by uniqueness of the continuity-equation/ODE solution. Generalizes to any differentiable interpolation X_t with v^X=E[Ẋ_t|X_t].
2. **Convex cost non-increase (Thm 3.5):** E[c(Z_1−Z_0)] ≤ E[c(X_1−X_0)] for all convex c. Double-Jensen:
   E[c(Z_1−Z_0)] = E[c(∫_0^1 v^X(Z_t,t)dt)] ≤ E[∫_0^1 c(v^X(Z_t,t))dt]  (Jensen over t)
   = E[∫_0^1 c(v^X(X_t,t))dt]  (same marginals)
   = ∫_0^1 E[c(E[X_1−X_0|X_t])]dt ≤ ∫_0^1 E[E[c(X_1−X_0)|X_t]]dt  (Jensen over conditioning)
   = E[c(X_1−X_0)]. Pareto descent over ALL convex c at once, not tied to a single c.
3. **Straightening / O(1/K) (Thm 3.7):** define straightness S(Z)=∫_0^1 E[‖(Z_1−Z_0)−Ż_t‖²]dt; non-crossing measure V((X_0,X_1))=∫_0^1 E[‖(X_1−X_0)−E[X_1−X_0|X_t]‖²]dt. With c=‖·‖² the two Jensen gaps are exactly S(Z)+V, giving the identity
   E[‖X_1−X_0‖²] − E[‖Z_1−Z_0‖²] = S(Z) + V((X_0,X_1)).
   Reflow Z^{k+1}=RectFlow((Z_0^k,Z_1^k)); telescoping ∑_{k=0}^K [S(Z^{k+1})+V((Z_0^k,Z_1^k))] = E‖X_1−X_0‖² − E‖Z_1^{K+1}−Z_0^{K+1}‖² ≤ E‖X_1−X_0‖². So min_k S(Z^k) ≤ E‖X_1−X_0‖²/K → 0.
   Derivation of S+V identity: E‖X_1−X_0‖² − E‖Z_1−Z_0‖²; use Z_1−Z_0=∫Ż_t dt and the two Jensen steps; each gap is the variance of the averaged quantity. (Gap1 = ∫E‖Ż_t − (Z_1−Z_0)‖²? actually gap from Jensen-in-t equals S; gap from conditioning equals V; verify signs.)
   Straight coupling = fixed point of Rectify = non-crossing interpolation (V=0) = flow coincides with interpolation.

## 1D / OT connection (Thm 3.10, 3.12)
- c-optimal (strictly convex c) ⇒ straight (else Rectify strictly lowers cost, contradiction).
- In 1D the unique monotone (non-crossing) coupling is jointly optimal for all convex c, and it is exactly the unique straight coupling. In d≥2 straight ⇏ c-optimal for a given c (rotational component of v^X). So straightness is necessary, generally not sufficient.

## Nonlinear extension / DDIM connection
- Replace line by any differentiable X_t connecting X_0,X_1; v^X(x,t)=E[Ẋ_t|X_t=x]; min_v ∫ w_t E‖v(X_t,t)−Ẋ_t‖²dt. Marginal-preserving still holds; cost/straightening do NOT (curve isn't a geodesic).
- X_t=α_t X_1+β_t ξ with ξ~N(0,I) recovers PF-ODE/DDIM. VP ODE: α exponential (eq vpode, a=19.9,b=0.1), β=√(1−α²). sub-VP: β=1−α². VE: α=1. These give curved / non-uniform-speed paths → can't straighten. Linear α_t=t, β_t=1−t is the canonical choice. Decouples π_0 from the schedule (no reason to force N(0,β_0²I) or approximate X_0≈β_0ξ).
- Proposition: with the OU relations η_t=−α̇/α, σ_t²=2β²(α̇/α − β̇/β), the DDPM/score ODE target Ỹ_t reduces to Ẋ_t=α̇X_1+β̇ξ, so PF-ODE ⊂ nonlinear rectified flow.

## Algorithm / code (grounded in gnobitab/RectifiedFlow)
- Train: t~U(0,1); x_t=t·x1+(1−t)·x0; target=x1−x0; loss=‖v_θ(x_t,t)−target‖². (losses.py: perturbed_data=t*batch+(1-t)*z0; target=batch-z0; square.)
- Sample: integrate dZ_t=v_θ(Z_t,t)dt from Z_0~π_0, Euler N steps or RK45. Backward: negate drift / reverse limits.
- Reflow: generate pairs (z0, ODE(z0)); retrain on that coupling (reflow_t_schedule='uniform').
- Distill k=1: train only at t=0, target x1−x0, i.e. T̂(z0)=z0+v(z0,0); LPIPS loss works better than L2 for images.
- Network: DDPM++/NCSN++ U-Net; t fed as t*999; EMA; Adam lr 2e-4.

## Design-decision → why
- Linear interpolation (not OU schedule): geodesic of ℝ^d ⇒ shortest path ⇒ straightenable + cost-decreasing; constant speed ⇒ uniform Euler error.
- Regress onto X_1−X_0 (not score): directly yields the velocity; minimizer is conditional mean of line direction; causalizes the non-causal interpolation.
- Why conditional-mean averaging is the right "fix" for crossings: a single-valued field can't follow two directions at a crossing; the L2-optimal single value is the mean → mass-preserving rewiring.
- Reflow not distill for straightening: distill approximates the *same* coupling; reflow produces a *new*, lower-cost, straighter coupling (uses marginal-preserving so it's still valid). Few reflows (1–2) — more accumulates v^X estimation error.
- π_0 arbitrary, decoupled from schedule: nonlinear framework shows α_t,β_t and π_0 are independent choices; SDE derivation conflated them.
- Distill at t=0 with one Euler step: when flow already near-straight, z0+v(z0,0)≈z1, so the t=0 term of the same objective is the distillation loss.
</content>
