# Synthesis — SPMe (Single Particle Model with electrolyte)

## The pain point
The DFN/P2D/Newman model is the standard physics-based li-ion model. It is a coupled
micro/macro PDE system: solid diffusion in a particle at *every* macroscopic point x
(pseudo-2D: x across the cell + r inside each particle), electrolyte diffusion+migration
(concentrated-solution / Stefan–Maxwell), charge conservation in solid (Ohm) and
electrolyte (MacInnes), all coupled by Butler–Volmer at the particle surface. After
finite-volume discretisation it becomes a *stiff* index-1 DAE (mixed parabolic+elliptic
→ algebraic constraints). With 30/20/30 x-points and 15 r-points the DFN stores ~1120
states; the SPMe stores ~110. DFN is too slow for BMS / control / optimization / packs,
and DAEs have convergence trouble on current steps (charge↔discharge).

The SPM is the cheap leading-order alternative: ONE representative particle per electrode,
uniform reaction current, electrolyte taken constant. It is accurate only below ~1–2C;
above that, electrolyte concentration/potential gradients dominate and SPM voltage error
grows (literature: >50 mV at high C-rate). Prior "extended SPM" models (Moura/Perez,
Prada, Han, Kemper, Rahimian, Tanim) bolt electrolyte terms onto SPM *ad hoc* — they pick
which terms to keep by hand, can't bound the error a-priori (Moura needs 6 a-posteriori
assumptions), and use POINTWISE voltage terms that mix orders.

## The idea (discovery order)
1. Write down the full dimensionless DFN.
2. Find the small parameter. Timescales (Table): discharge τ_d, electrolyte diffusion
   τ_e, solid diffusion τ_k, reaction τ_r. Key group C_e = τ_e/τ_d = electrolyte
   transport timescale / discharge timescale ≈ 4.19e-3·(C-rate) — TINY. Also σ_k (thermal
   voltage / solid Ohmic drop) and κ̂_e (thermal voltage / electrolyte Ohmic drop) are
   LARGE. Distinguished limit: σ_k = σ_k'/C_e, κ̂_e = κ̂_e'/C_e with primes O(1), as C_e→0.
3. Expand every variable in powers of C_e: f ~ f0 + C_e f1 + C_e^2 f2 + ...
4. KEY PRE-STEP before expanding: rewrite terminal voltage in ELECTRODE-AVERAGED form.
   Pointwise V along a current path (eq pointwise-voltage) = U_eq|_xn,xp + η_p−η_n +
   φ_e,p−φ_e,n + ΔΦ_Solid. Average over xn∈[0,Ln] and xp∈[1−Lp,1] → V̄ = Ūeq + η̄_p−η̄_n
   + φ̄_e,p−φ̄_e,n + ΔΦ̄_Solid. This is THE key step; without it the model must be more
   complex to reach the same accuracy.
5. Also note conservation: ∫c_e dx = 1 (total Li-ion in electrolyte conserved).

## Leading order (recovers SPM)
- electrolyte molar+flux at O(1): ∂N0/∂x=0, N0 = −ε^b D_e(c0)∂c0/∂x; BCs+conservation ⇒
  N_e0=0, c_e0=1. NO electrolyte depletion at leading order.
- McInnes + solid Ohm at O(1): ∂φ_e0/∂x=0, ∂φ_s0/∂x=0 (σ,κ blow up). Potentials uniform.
- so c_e0,φ_e0,φ_s0 indep of x; c_s0 initially indep of x ⇒ j0, j00, η0 indep of x.
  Integrate charge conservation over electrode ⇒ i_e,n0 = xI/Ln, i_e,p0=(1−x)I/Lp;
  j_n0 = I/Ln, j_p0 = −I/Lp (uniform interfacial current).
- η_n0 = 2 asinh(I/(j0n0 Ln)), η_p0 = −2 asinh(I/(j0p0 Lp)).
- φ_e,p0 − φ_e,n0 = 0, ΔΦ_Solid0 = 0.
- ⇒ SPM: C_k ∂c_s0/∂t = (1/r²)∂_r(r²∂_r c_s0), with flux BC −(a_k γ_k/C_k)∂_r c_s0|_{r=1}
  = ±I/L_k; V0 = U_p(c_sp0)−U_n(c_sn0) − 2asinh(I/(j0p0 Lp)) − 2asinh(I/(j0n0 Ln));
  j00k = (γ_k/C_r,k) c_s0^½(1−c_s0)^½.
  (Single particle = all particles behave identically in this limit, not "replace many by one".)

## First order O(C_e) (the electrolyte correction → SPMe)
- electrolyte flux derivative: ∂N1/∂x = (1/γ_e)∂i_e0/∂x; integrate ⇒ N_e1 = I x/(γ_e Ln)
  etc (steady linear profiles). Sub into flux N1 = −ε^b D_e(1)∂c_e1/∂x + (t+/γ_e)i_e0;
  integrate twice using continuity + lithium conservation ⇒ analytic quadratic-in-x
  c_e,n1, c_e,s1, c_e,p1 (the steady electrolyte concentration correction).
- McInnes at O(C_e): i_e0 = ε^b κ̂_e' κ_e(1)(−∂φ_e1/∂x + 2(1−t+)∂c_e1/∂x). Integrate ⇒
  φ_e1 = φ̃_e + 2(1−t+)c_e1 − (I/(κ̂_e'κ_e(1)))·(geometry). The 2(1−t+)c_e1 piece is the
  diffusion (concentration) potential; the I/κ piece is migration Ohmic.
- solid Ohm at O(C_e): I − i_e0 = −σ_k'∂φ_s1/∂x ⇒ φ_s1 quadratic ⇒ solid Ohmic loss
  ΔΦ̄_Solid = −(I/3)(Lp/σp + Ln/σn).
- THE crucial averaging step: at O(C_e), ∂i_e1/∂x = j_k1 with i_e1=0 at both electrode
  ends ⇒ ∫_electrode j_k1 dx = 0. So electrode-AVERAGED reaction-current correction is
  zero ⇒ averaged particle surface flux correction is zero ⇒ c̄_s1 = 0 ⇒ Ū_eq^1 = 0
  (no first-order OCV correction). This is exactly why the averaged form is well-posed:
  given the averaged current, find the averaged potential — no need to know pointwise j.
  Ad-hoc pointwise models implicitly assume j(x)=averaged j, false beyond leading order.
- linearise Butler–Volmer at O(C_e), average ⇒ η̄_k1 = −c̄_e,k1 tanh(η_k0/2)
  = c̄_e,k1 I / sqrt((j0k0 Lk)²+I²). Concentration overpotential η̄_c = 2 C_e(1−t+)(c̄_e,p1 − c̄_e,n1).
- electrolyte Ohmic loss ΔΦ̄_Elec = −(I/κ̂_e κ_e(1))(Ln/(3ε_n^b) + Ls/ε_s^b + Lp/(3ε_p^b)).
- Combine: fold reaction-overpotential correction into electrode-averaged exchange-current
  density j̄0k = (1/Lk)∫ (γ_k/C_r,k) c_s0^½(1−c_s0)^½ (1+C_e c_e1)^½ dx, since
  −2asinh(I/(j̄0k Lk)) ≈ −2asinh(I/(j0k0 Lk)) + C_e c̄_e,k1 I/sqrt(...). 
- SPMe(S) = steady electrolyte (drop ∂c_e1/∂t) — purely algebraic correction, no extra PDE/state.

## Canonical SPMe (transient electrolyte)
For current steps, transient electrolyte matters (timescale τ_e). Scale time with τ_e;
on that timescale particle concentrations frozen, exchange terms negligible. State the
COMPOSITE model (correct on both diffusion and discharge timescales): keep the ∂c_e1/∂t
term ⇒ one extra linear parabolic PDE for c_e1 across n/s/p with the leading-order current
as source. Final SPMe: TWO particle diffusion PDEs (n,p) + ONE electrolyte diffusion PDE,
all independent & LINEAR (naturally parallel, non-stiff ODEs after discretisation), plus the
algebraic voltage V = Ūeq + η̄_r + η̄_c + ΔΦ̄_Elec + ΔΦ̄_Solid.

Dimensional SPMe (eq SPMe-dimensional) combines leading+first order electrolyte into one
PDE: ε_k ∂c_e/∂t = −∂N_e/∂x + source(±I/(F L_k)); N_e = −ε^b D_e(c_typ)∂c_e/∂x + t+ I/(F)·(geom).

## Validity / error (a-priori, from the limit)
Conditions (Table conditions): C_e ≪1, RTσ_k/(FI L)≫1, RTκ/(FIL)≫1, solid & reaction
timescales ≪ 1/C_e. Error ~ max( (I L/(D_e F c_max))², (I L/D_e)²(1/(F R T))|U_k''| ).
The second term blows up when OCV is highly nonlinear (end of discharge) → 3C discrepancy.
Fix would need distinguished limit U_k'' ~ O(1/C_e), which forces solving every particle →
loses SPM cheapness, so omitted.

## CODE (PyBaMM) grounding
SPMe inherits SPM. SPM uses ConstantConcentration electrolyte diffusion, LeadingOrder
electrolyte conductivity, LeadingOrder electrode ohm. SPMe overrides:
- electrolyte diffusion → Full (Stefan–Maxwell: N = −tor D_e grad c + t+ i_e/F (+conv);
  ∂(ε c)/∂t = −div N + source/F) — the transient electrolyte PDE.
- electrolyte conductivity → Composite: builds φ_e (the O(C_e) potential), and the split
  overpotentials eta_c_av = chi RT/F (macinnes_c_e_p − macinnes_c_e_n) [concentration
  overpotential] and delta_phi_e_av = −i(L_n/(3κ_n)+L_s/κ_s+L_p/(3κ_p)) [electrolyte
  Ohmic] — EXACTLY the paper's η̄_c and ΔΦ̄_Elec.
- electrode ohm → Composite (first-order solid Ohmic loss, the −I/3 (L/σ) term).
Particle submodel (FickianDiffusion x_average=True) = the single averaged particle (SPM
leading order). chi = 2(1−t+) thermodynamic factor. Voltage assembled in BaseModel from
OCP + η_r + η_c + ΔΦ_Elec + ΔΦ_Solid.

## Design-decision → why
- Electrode-averaged (not pointwise) voltage: only well-posed object given averaged current;
  guarantees O(C_e²) accuracy; this is the step ad-hoc models miss → main accuracy gain.
- Distinguished limit σ,κ ~ 1/C_e: keeps Ohmic losses at first order (same order as
  electrolyte concentration effects); otherwise they'd drop or dominate.
- Linear electrolyte PDE (D_e(1) frozen at c=1, not D_e(c)): D_e(c) is a higher-order term;
  keeping it (Perez) gives a nonlinear PDE with no accuracy guarantee. Linear ⇒ parallel,
  non-stiff, analytic special cases.
- SPMe(S) vs SPMe: steady = algebraic, no extra state (good for microcontroller RAM);
  transient = +1 PDE, needed after current steps.
- Single averaged particle even at first order (c̄_s1=0): the averaged surface flux
  correction vanishes, so no benefit to resolving per-particle solid diffusion at O(C_e);
  keeps SPM cheapness. (Residual O(C_e) particle error is the price.)
