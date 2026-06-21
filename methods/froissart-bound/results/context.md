# Context: how fast can a relativistic total cross section grow?

## Research question

For the elastic scattering of two hadrons, A + B → A + B, the central observable at high energy is the total cross section σ_tot(s), where s is the squared centre-of-mass energy. Experiment shows the high-energy cross sections of strong interactions vary only slowly, and the prevailing theoretical wisdom (Pomeranchuk) is that cross sections of a process and its charge-conjugate tend to a common limit. By the optical theorem a forward amplitude growing like s^N gives σ_tot ∼ s^{N−1}, so the growth of the cross section is tied to the growth of the amplitude.

The technical question of the moment comes straight from the machinery being built around the Mandelstam double-dispersion representation: that representation, written for a four-point amplitude, is a double integral over spectral densities and converges as written only if the amplitude falls off at infinity. If the amplitude grows, one performs *subtractions* — subtract a polynomial in the energy before dispersing — and the number of subtractions equals the degree of polynomial growth one allows. So the question "how many subtractions does the Mandelstam representation need?" is identical to "how fast can the amplitude grow with energy?" — and through the optical theorem, to "how fast can σ_tot grow?".

The setting for this question: take the analyticity content of the double-dispersion representation (with a finite, unspecified number of subtractions) together with the conservation of probability (unitarity), and ask what these say about the asymptotic growth of σ_tot(s) as s → ∞ for the relativistic amplitude, expressed through the physical mass scale of the theory (the lightest exchanged particle).

## Background

**The optical theorem.** For 2 → 2 scattering with c.m. momentum k, the total cross section is fixed by the imaginary part of the forward elastic amplitude,
  σ_tot(s) = (4π/k) Im f(s, θ = 0),
where f is the c.m. scattering amplitude normalized so that dσ/dΩ = |f|². This identity (conservation of probability, the unitarity of the S-matrix at the forward point) is the bridge between the amplitude's growth and the cross section's growth.

**Partial waves and unitarity.** At fixed s the amplitude is expanded in Legendre polynomials of the scattering angle,
  f(s, t) = (1/k) Σ_{l≥0} (2l+1) a_l(s) P_l(cos θ),  cos θ = 1 + t/2k²,
with t the squared momentum transfer (t ≤ 0 in the physical region). Conservation of probability for each angular momentum channel — the S-matrix element S_l = 1 + 2i a_l must satisfy |S_l| ≤ 1 — gives the partial-wave unitarity relation
  Im a_l(s) ≥ |a_l(s)|²  ≥ 0,  and hence  0 ≤ Im a_l(s) ≤ 1.
Two facts are doing the work here: each partial wave is *capped* (its imaginary part cannot exceed unity), and the imaginary part is *positive*.

**The absorptive part.** The imaginary part in the s-channel, A(s, t) = Im_s f(s, t) (the "absorptive part"), inherits a partial-wave expansion with manifestly positive coefficients,
  A(s, t) = (1/k) Σ_{l≥0} (2l+1) Im a_l(s) P_l(cos θ).
At the forward point the optical theorem reads σ_tot = (4π/k) Im f(s, 0) = (4π/k²) Σ (2l+1) Im a_l(s), since P_l(1) = 1.

**The Mandelstam representation (Mandelstam 1958).** Postulating simultaneous analyticity in two of the invariants, the four-point amplitude M(s,t,u) for equal masses is written as a double dispersion relation,
  M(s,t,u) = (1/π²) ∬ ρ_st(s′,t′)/[(s′−s)(t′−t)] ds′dt′ + (su term) + (tu term),
over the double spectral densities ρ_st, ρ_su, ρ_tu. Chew and Mandelstam (1960) applied this to pion–pion scattering. The representation encodes the singularity structure expected from particle thresholds: the t-channel double spectral density has its support beginning at the lowest t-channel threshold, two-pion exchange at t = 4m_π². As written above the representation has *no subtractions* and presupposes that the amplitude vanishes at infinity; for a growing amplitude one subtracts a polynomial whose degree is the number of subtractions.

**Analyticity in the angle: the Lehmann ellipse (Lehmann 1958).** Microcausality plus the spectral conditions of the field theory imply that, for fixed physical s, the amplitude f(s, cos θ) — and its absorptive part A(s, cos θ) — is an analytic function of cos θ regular inside an ellipse in the complex cos θ-plane with foci at ±1. The boundary on the real axis sits at
  z_0(s) = 1 + t_0 / 2k²,
where t_0 is set by the nearest t-channel singularity. For the absorptive part, the relevant singularity is the two-pion threshold, t_0 = 4m_π². The Legendre/partial-wave expansion of A converges everywhere inside this ellipse. Because k² grows with s (k² → s/4 for large s in the equal-mass case), the ellipse *shrinks* toward the segment [−1, 1] as s → ∞: z_0 → 1 + 2t_0/s.

**Asymptotics of Legendre functions.** For x > 1 the Legendre polynomial grows with its degree, P_l(x) ∼ (x + √(x²−1))^l / √l, and the second-kind function decays, Q_l(x) ∼ (x + √(x²−1))^{−l−1}. The Laplace integral representation
  P_l(x) = (1/π) ∫_0^π (x + cos φ √(x²−1))^l dφ,  x ≥ 1,
makes these rates and the useful one-sided inequalities explicit. A convergent expansion in P_l with positive coefficients can only converge up to the boundary of the largest ellipse in which the function is analytic — so analyticity inside an ellipse of real semi-axis z_0 forces the coefficients to decay at least like (z_0 + √(z_0²−1))^{−l}.

**Polynomial boundedness.** In an axiomatic field theory the matrix elements are tempered distributions, which are polynomially bounded; correspondingly one expects the amplitude (or the absorptive part at fixed t in a suitable interval) to be bounded by a power of s: |A(s, t)| ≤ C(t) s^N for some finite integer N and t in a neighbourhood of zero. This is precisely the "finite number of subtractions" hypothesis, restated.

**The physical picture of finite-range forces.** Strong forces have finite range because they are mediated by massive quanta; the lightest is the pion. A Yukawa exchange of mass μ produces a potential ∝ e^{−μr}/r, an interaction that is appreciable only out to an impact parameter set by 1/μ (up to slowly varying factors). This is the intuitive seat of any ceiling on the cross section: a finite-range interaction can present only a finite (slowly growing) disk to the incoming flux.

## Baselines

**Unsubtracted / fixed-number-of-subtractions dispersion relations.** Single-variable fixed-t dispersion relations relate Re F to an integral of Im F over energy and were the workhorse of the period. They give powerful constraints and, at t = 0, connect to total cross sections via the optical theorem; they take the high-energy growth of the absorptive part as an *input*, the number of subtractions being supplied externally.

**The Mandelstam double-dispersion representation (Mandelstam 1958; Chew–Mandelstam 1960).** It makes analyticity in *two* invariants explicit and separates the singularities into three double-spectral terms; it underpins the S-matrix program of deriving dynamics from analyticity, crossing, and unitarity without a Lagrangian. The double spectral support records *where* the t-channel singularities sit (the two-pion threshold at 4m_π²); the number of subtractions is set by the assumed asymptotic growth of the amplitude.

**The Lehmann ellipse (Lehmann 1958).** This gives, rigorously from causality, a finite domain of analyticity in cos θ at fixed s — enough to make the partial-wave expansion converge in a neighbourhood of the physical region. The ellipse contracts onto the physical segment as s → ∞ (its real semi-axis tends to 1 like 1 + O(1/s)).

**The Pomeranchuk theorem and the constancy expectation.** Pomeranchuk's results pointed to particle and antiparticle cross sections approaching a common limit, consistent with the empirical near-constancy of high-energy cross sections. This is a statement about *differences* and *limits* under assumed boundedness of the cross sections.

## Evaluation settings

The natural yardstick is theoretical consistency, not a benchmark dataset: a result is judged by whether it follows from the stated analyticity + unitarity inputs with no extra dynamical assumption beyond a finite number of subtractions, whether it is expressed through the physical mass scale of the theory (the pion mass, since t_0 = 4m_π² is the nearest t-channel threshold for ππ, KK, πN, πΛ, … processes), and whether it reduces correctly under the optical theorem so that a statement about the forward amplitude becomes a statement about σ_tot. The relevant kinematic regime is s → ∞ at fixed (forward) momentum transfer; the observable is σ_tot(s). High-energy total-cross-section measurements (accelerator data on πp, pp, etc.) are the empirical context.

## Code framework

A small numerical scaffold that evaluates the growth law and the angular-momentum content; everything below is in pre-method terms (Legendre polynomials, the c.m. kinematics, the unitarity cap), with the one quantitative law left as an open slot.

```python
import numpy as np
from numpy.polynomial.legendre import Legendre   # P_l already available

# --- pre-method kinematics (equal-mass elastic scattering) -----------------
m_pi = 1.0                       # work in pion-mass units
def k2(s):                       # c.m. momentum squared, equal masses
    return (s - 4*m_pi**2) / 4.0

# nearest t-channel singularity (two-pion exchange) sets the angular-analyticity scale
t0 = 4*m_pi**2

def lehmann_semi_axis(s):
    """Real semi-axis z0 of the angular analyticity ellipse at fixed s."""
    return 1.0 + t0 / (2.0 * k2(s))

# --- unitarity cap on a single partial wave --------------------------------
IM_AL_MAX = 1.0                  # 0 <= Im a_l(s) <= 1

# --- optical theorem: forward sum over retained partial waves --------------
def sigma_tot_from_waves(s, im_al):
    """im_al: array of Im a_l(s), l = 0,1,...  ->  sigma_tot via optical theorem."""
    l = np.arange(len(im_al))
    return (4*np.pi / k2(s)) * np.sum((2*l + 1) * im_al)

# --- the quantity the derivation must determine ----------------------------
def num_significant_partial_waves(s):
    # TODO: the growth law we will derive (how many waves survive the
    #       angular-analyticity decay before the unitarity cap stops mattering)
    raise NotImplementedError

def cross_section_ceiling(s):
    # TODO: the asymptotic upper bound on sigma_tot(s) we are after
    raise NotImplementedError
```
