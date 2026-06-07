# Synthesis — Lorenz, deterministic chaos / sensitive dependence

## Sources read this run (refs/)
- `lorenz-1963.pdf` — PRIMARY: "Deterministic Nonperiodic Flow", JAS 20:130–141 (BU mirror, full 12pp read).
- `chaos-at-fifty-1306.5777.pdf` — Motter & Campbell, contains Lorenz's OWN words (the long quote from
  *The Essence of Chaos* p.134) verbatim + Laplace + Poincaré antecedents + all the facts.
- `seagulls-butterflies-grasshoppers-AJP.pdf` — Hilborn 2004, AJP 72(4):425. Quotes the 1972 AAAS talk
  verbatim; the seagull→butterfly history; the 1898 Franklin "grasshopper" antecedent.
- `essence-of-chaos-review-0910.2213.pdf` — Lurie, quotes Lorenz's ski-slope sensitive-dependence demo.
- `state-of-nonlinear-dynamics-1963.pdf` — context on the 1963 field state.
- `ghys-butterfly-effect.pdf` — Ghys, mathematician's history (Poincaré, Hadamard, Maxwell antecedents).

## SELF-ACCOUNT (the backbone) — Lorenz's own narration
From *The Essence of Chaos* (1993), p.134, quoted in Chaos at Fifty:
"At one point I decided to repeat some of the computations in order to examine what was happening in
greater detail. I stopped the computer, typed in a line of numbers that it had printed out a while
earlier, and set it running again. I went down the hall for a cup of coffee and returned after about
an hour, during which time the computer had simulated about two months of weather. The numbers being
printed were nothing like the old ones. I immediately suspected a weak vacuum tube or some other
computer trouble... but before calling for service I decided to see just where the mistake had
occurred... Instead of a sudden break, I found that the new values at first repeated the old ones,
but soon afterward differed by one and then several units in the last decimal place, and then began to
differ in the next to the last place and then in the place before that. In fact, the differences more
or less steadily doubled in size every four days or so, until all resemblance with the original output
disappeared somewhere in the second month. This was enough to tell me what had happened: the numbers
that I had typed in were not the exact original numbers, but were the rounded-off values that had
appeared in the original printout. The initial round-off errors were the culprits; they were steadily
amplifying until they dominated the solution."

Concrete detail (Tech Review / Chaos at Fifty): he rounded one variable from .506127 to .506 — three
decimal places in the printout vs six-figure internal precision. Model was 12 variables (temperature,
wind etc.). Machine: Royal McBee LGP-30, 4096 32-bit words, ~1 s/iteration. He had bought it to PIT
linear statistical forecasting against dynamical (equation) forecasting; he sought NONPERIODIC
solutions (hardest case for linear methods), found them by adding latitude/longitude-varying heating.

1972 AAAS talk (Hilborn quote): "...whether two particular weather situations differing by as little
as the immediate influence of a single butterfly will generally after sufficient time evolve into two
situations differing as much as the presence of a tornado. In more technical language, is the behavior
of the atmosphere *unstable* with respect to perturbations of small amplitude?"
Seagull origin (1963 NY Acad Sci): "one flap of a sea gull's wings would be enough to alter the course
of the weather forever." Title "butterfly" coined by Philip Merilees in Lorenz's absence. Also:
the 1963 paper's first title was "Deterministic turbulence", changed at editor's urging (Chaos at 50).

## ANTECEDENTS
- **Laplacian determinism** (the assumption being overturned): Laplace's demon — given exact initial
  state + forces, the entire future is computable; tacit corollary = APPROXIMATE knowledge gives
  approximate prediction. (Chaos at Fifty quotes Laplace directly.)
- **Poincaré (1880s, three-body)**: "small differences in the initial conditions produce very great
  ones in the final phenomena... prediction becomes impossible." Lorenz cites Poincaré 1881
  (J. de Math 7) in his own bibliography. He was likely unaware of it at discovery time (Chaos@50 p.2).
- **Numerical weather prediction**: dynamic meteorology (simulate the fluid equations) vs the
  prevailing linear-statistical forecasting. Lorenz's whole experiment was set in this rivalry.
- **Convection / Rayleigh-Bénard**: Rayleigh 1916 — fluid layer heated from below; steady solution
  (no motion) goes unstable and convection sets in when the Rayleigh number Ra = gαH³ΔT ν⁻¹κ⁻¹
  exceeds critical Rc = π⁴a⁻²(1+a²)³, min 27π⁴/4 at a²=1/2.
- **Saltzman (1962)**: "Finite amplitude free convection as an initial value problem—I", JAS 19:329.
  Expanded ψ, θ in double Fourier series → infinite ODE system; truncated; integrated numerically;
  found in some cases all but THREE variables died and the three fluctuated nonperiodically. Lorenz
  credits Saltzman explicitly (acknowledgments) for "bringing to his attention the existence of
  nonperiodic solutions." (Saltzman paper itself paywalled; fully described inside Lorenz 1963 §5.)

## THE PRIMARY RESULT (Lorenz 1963), as derived
Convection PDEs (§5, from Saltzman, 2-D rolls, free-free boundaries):
  ∂/∂t ∇²ψ = -∂(ψ,∇²ψ)/∂(x,z) + ν∇⁴ψ + gα ∂θ/∂x        (17)
  ∂/∂t θ   = -∂(ψ,θ)/∂(x,z) + (ΔT/H) ∂ψ/∂x + κ∇²θ        (18)
ψ = stream function, θ = temperature departure from no-convection state.
Truncation to a single mode each (eqs 23-24): keep one term in ψ (∝X) and two in θ (∝Y, ∝Z).
Substituting and dropping all other-frequency trig terms gives the three convection equations:
  X' = -σX + σY                  (25)
  Y' = -XZ + rX - Y             (26)
  Z' =  XY - bZ                 (27)
with τ = π²H⁻²(1+a²)κt (dimensionless time), σ = κ⁻¹ν = Prandtl number, r = Rc⁻¹Ra (relative
Rayleigh number), b = 4(1+a²)⁻¹. Physical meaning: X ∝ convection intensity; Y ∝ temperature
difference between ascending/descending currents; Z ∝ distortion of vertical temperature profile
from linearity. Chosen values: σ=10, a²=1/2 → b=8/3; critical r for instability of steady convection
r = σ(σ+b+3)(σ-b-1)⁻¹ = 470/19 ≈ 24.74 (eq 34); they pick slightly supercritical r=28.

Steady states: origin (0,0,0) = no convection; two others (eq after 32):
  X=Y=±√(b(r-1)), Z=r-1  → (±6√2, ±6√2, 27) = (±8.485, ±8.485, 27) = C, C'.
For r>1 origin is a saddle (one positive root of [λ+b][λ²+(σ+1)λ+σ(1-r)]=0, eq 32).
For r>critical(34) the two convective steady states are also unstable (complex roots cross imaginary
axis): convection itself is unsteady → the system has NOWHERE stable to settle.

Volume contraction (eqs 30-31): divergence of the flow = ∂X'/∂X + ∂Y'/∂Y + ∂Z'/∂Z = -(σ+b+1).
So any phase-space volume V obeys V' = -(σ+b+1)V → shrinks like e^{-(σ+b+1)t}, i.e. e^{-(41/3)t}
for these constants. DISSIPATIVE: every volume collapses to zero. Combined with boundedness
(trajectories trapped inside an ellipsoid, §2: Q=½ΣX² ⇒ -dQ/dt>0 outside an ellipsoid E, so all
trajectories enter and stay in a bounded region R). Bounded + every volume → 0 + no stable fixed
point + (§3) every trajectory unstable ⇒ trajectories crowd onto a zero-volume set that is neither a
point nor a closed curve nor a torus — the attractor (Lorenz's "infinite complex of surfaces", §7,
eqs 37-38; later called the strange/Lorenz attractor, fractal dim ≈ 2.06 per Chaos@50).

§3 instability of nonperiodic flow: a stable trajectory must be quasi-periodic (proved via limit
points); contrapositive — a nonperiodic trajectory is unstable; two states differing imperceptibly
"may eventually evolve into two considerably different states" → "prediction of the sufficiently
distant future is impossible." This is sensitive dependence, derived BEFORE the numerics.

§7 numerics: double-approximation (RK-like) scheme, Δτ=0.01, on the LGP-30. Y(t) shows amplified
oscillations then irregular sign changes; the (Y,Z) and (X,Y) projections trace the two-lobed orbit
(Fig 2). The "tent map" M_{n+1}=2M_n (M_n<½) / 2-2M_n (M_n>½) of successive Z-maxima (eqs 35, Fig 5)
slope > 1 everywhere ⇒ all periodic sequences unstable, the nonperiodic ones nondenumerable & of
full measure ⇒ almost all solutions nonperiodic. (Forecasting analogue argument §8.)

## KEY DESIGN CHOICES → WHY
- Forced + dissipative (nonconservative) ODEs, not conservative: a conservative quantity would be
  pinned to its initial value (can't represent forcing→response); dissipation + forcing lets the
  system reach a nontrivial sustained state and traps trajectories in a bounded R. (§1-2)
- Reduce convection to ODEs at all: continuous PDE → finite ODE by spectral truncation, so it can be
  integrated on the LGP-30 and its ultimate behavior studied. (§1, §5)
- Truncate to exactly THREE modes: fewer can't show the phenomenon; Saltzman's larger runs showed only
  three variables stayed alive, so three is the minimal faithful core. (§5)
- σ=10, b=8/3 (a²=1/2): a²=1/2 minimizes the critical Rayleigh number (the easiest convection cell);
  σ=10 is a representative Prandtl number (Saltzman's choice). (§5, §7)
- r=28: just above the steady-convection instability threshold ≈24.74, so steady convection has just
  gone unstable and the only behavior left is unsteady/nonperiodic — the regime of interest. (§7)
- Numerical scheme = double approximation (centered cannot be used: it lets X_{n+1} be non-unique,
  violating determinism; forward diff can blow up; double-approx is bounded). (§4)

## NUMERICAL ILLUSTRATION (code/lorenz.py, verified)
RK4, σ=10,b=8/3,r=28. Two starts (1,1,1) and (1,1,1.000001): |Δ| grows from 1e-6 to O(10) by t=40,
roughly doubling on a fixed timescale (matches "doubled every four days"). Orbit bounded:
X∈[-17.9,19.6], Z∈[1.0,47.8]. Steady states reproduce (±8.485,±8.485,27)=(±6√2,±6√2,27). ✓
