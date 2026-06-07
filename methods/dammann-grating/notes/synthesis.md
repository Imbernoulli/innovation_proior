# Synthesis — Dammann grating

## Pain point / research question
Multiple imaging / beam fan-out: take ONE input beam (or one object) and produce an array of N equally
bright copies arranged regularly about the optical axis, in a single passive pass, with high light
efficiency. Pre-1971: multiple images were made with beam-splitter stacks / amplitude masks (lossy,
bulky) or with absorbing gratings (a Ronchi/amplitude grating wastes most of the light into the wrong
orders and into absorption). Want: a thin, fully transparent phase element that splits into equal orders.

## Ancestors (load-bearing)
- **Fourier/Fraunhofer diffraction.** Far field of an aperture/transparency = Fourier transform of its
  complex transmittance. For a *periodic* transmittance with period d, the far field is a comb of
  discrete diffraction orders at sinθ_m = mλ/d; the complex amplitude of order m is the m-th Fourier-series
  coefficient c_m of one period of the transmittance. Order intensity = |c_m|^2. (Goodman, Fourier Optics.)
- **Phase gratings vs amplitude gratings.** An amplitude (Ronchi) grating absorbs/blocks light → low
  efficiency, strong 0 order. A *phase* grating |t|=1 everywhere is lossless: all incident power is
  redistributed among the orders, none absorbed. Sinusoidal phase grating → Bessel-function order weights
  (unequal). Binary phase grating (two levels) → simplest to fabricate (one etch depth).
- **Fourier-series view of a periodic profile.** Any design question about order weights becomes a question
  about the Fourier coefficients of one period. Free knobs = the *shape* of the period.
- **Fan-out / array generation framing**: want first N orders EQUAL, the rest as small as possible.

## The derivation (re-derived, verified numerically in code/verify_dammann_math.py)
- Restrict to a **binary** transmittance: t(x) = e^{iφ(x)}, φ ∈ {0, π} → t ∈ {+1, −1}. Two phase levels =
  one etch depth d = λ / (2(n−1)) (half-wave). Fabrication-friendly; verified in NASA paper eq. for d.
- Restrict to an **even** profile g(x)=g(−x) over one period [−1/2, 1/2] (period normalized to 1). Evenness
  → all c_m are **real** → no phase scrambling across orders, and the order spectrum is symmetric c_{−m}=c_m,
  so a symmetric fan-out about the 0 order is automatic. (This is *why* even + 0/π, not arbitrary complex.)
- Even binary profile is fully specified by **transition points** on the half-period:
  0 = x_0 < x_1 < ... < x_K < x_{K+1} = 1/2, with sign s_j = (−1)^j on [x_j, x_{j+1}).
- Fourier coefficient (m ≥ 1):
    a_m = ∫_{-1/2}^{1/2} g cos(2π m x) dx = 2 ∫_0^{1/2} g cos(2π m x) dx
        = 2 Σ_{j=0}^{K} (−1)^j ∫_{x_j}^{x_{j+1}} cos(2π m x) dx
        = (2/(π m)) Σ_{j=0}^{K} (−1)^j [sin(2π m x_{j+1}) − sin(2π m x_j)]
    Boundary terms: sin(2π m·0)=0 and sin(2π m·1/2)=sin(π m)=0 vanish; Abel/telescoping the interior:
        a_m = (2/(π m)) Σ_{j=1}^{K} 2(−1)^{j-1} sin(2π m x_j)     [each interior x_j shared by two intervals
                                                                   with opposite sign → factor 2, sign (−1)^{j-1}]
  Equivalent compact form used in code (sign (−1)^{j}, j=0..K−1 indexing interior pts t_1..t_K):
        a_m = (2/(π m)) Σ_j (−1)^j 2 sin(2π m t_j)   — verified == brute-force numeric FT to 1e-6.
    a_0 = 2 Σ_j (−1)^j (x_{j+1} − x_j).   (DC = signed length balance; controls 0-order weight.)
- **Design system**: want |a_0| = |a_1| = ... = |a_{N-1}| (first N orders equal), with the remaining power
  pushed out / efficiency maximized. K interior transitions ⇒ K free parameters ⇒ can impose ~K independent
  equal-intensity conditions. Relation reported in lit: N_d = 2N_t + 1 total equal orders from N_t transitions
  (1 transition → 1×3, 3 transitions → 1×7, etc.). Solve the nonlinear system numerically for x_j.
- **Classic 1×3, single transition** (verified): a_0 = 4x_1 − 1, a_1 = (2/π) sin(2π x_1). Set a_0^2 = a_1^2 →
  root x_1 ≈ 0.36763 → I_0 = I_1 ≈ 0.2214, efficiency η = a_0^2 + 2a_1^2 ≈ 66.42%. (matches optimizer.)
- **1×5 (2 transitions)**: optimizer → η ≈ 77.4% with uniform orders. (matches lit ~77%.)
- Efficiency η = (power in the N target orders)/(total) = (a_0^2 + 2 Σ_{m=1}^{N-1} a_m^2). Binary 0/π caps
  achievable η below the multilevel ideal — well-known ceiling commonly quoted <~86% for binary fan-outs.

## Why each choice (design-decision table)
- **Phase, not amplitude** → losslessness (|t|=1), all power redistributed not absorbed.
- **Binary (2 levels)** → single etch depth, easiest/most robust fabrication; the price is an efficiency
  ceiling and limited number of independently controllable orders.
- **0 and π specifically** → π gives maximal contrast (t=+1 vs −1, the two real values on the unit circle);
  any other Δφ leaves |t|=1 but reduces the achievable order separation / breaks the clean real-coefficient
  picture; half-wave etch d = λ/(2(n−1)).
- **Even profile** → real, symmetric coefficients ⇒ symmetric equal fan-out about axis + no inter-order phase.
- **Transition points as design variables** → they are the *only* free parameters of an even 0/π period;
  the whole design collapses to "where do the phase flips go."
- **Numerical solve** → the equal-|a_m| system is transcendental (sums of sines); no closed form beyond the
  smallest cases → Newton / nonlinear least squares / direct search with multistart.

## Empirical facts (→ context.md, sourced)
- Grating eq. sinθ_m = mλ/d; etch depth d = λ/(2(n−1)) for 0/π. (NASA Keys et al.; UMN DeMars et al.)
- GaAs IR example: n≈3.26 @12µm → d≈2.65µm; n≈3.30 @5µm → d≈1.09µm. (NASA)
- Binary fan-out efficiency ceiling commonly quoted < ~86%; 1×3 ≈ 66%, 1×5 ≈ 77% (derived/verified here).
- Design is wavelength-independent (transition pattern is on normalized coordinate); only etch depth scales.

## Canonical code
- pyMOE (INL nano, github.com/INLnano/pyMOE): sag_functions.py builds the Dammann profile from transition
  points (alternate_transitions_symmetric, dammann_grating_periodic, dammann_2d) — the *construction*.
- code/dammann_designer.py: the transition-point *optimizer* (closed-form a_m + multistart Nelder–Mead on a
  uniformity+efficiency cost). Verified: 1×3 → t≈0.368, η≈66.4%; 1×5 → η≈77.4%, uniform.
- code/verify_dammann_math.py: closed-form a_m vs brute-force numeric FT, agree to 1e-6.
