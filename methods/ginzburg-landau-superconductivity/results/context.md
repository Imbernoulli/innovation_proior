# Context: a macroscopic theory of superconductors in a magnetic field

## Research question

A superconductor expels magnetic field (the Meissner effect) and carries dissipationless
current, but only up to a critical magnetic field H_c, above which superconductivity is
destroyed. The available macroscopic theory describes the *bulk* response well yet fails
in three concrete, measured ways: (i) it cannot account for how a strong field or current
destroys superconductivity in a thin film, because its single material constant is fixed
at a given temperature and does not respond to the field; (ii) it predicts the *wrong sign*
for the surface energy at a boundary between coexisting normal and superconducting regions
of the same metal — it comes out negative, whereas a stable intermediate state requires it
to be positive; (iii) it offers no way at all to treat situations where the "amount of
superconductivity" varies from point to point.

The goal is a phenomenological theory, valid near the transition temperature T_c, that
(a) lets the local degree of superconductivity vary in space, (b) yields the surface energy
between phases from the *ordinary* material parameters rather than by hand, and (c) recovers
the known bulk behavior (Meissner screening, the H_c(T) curve) as a limit. It must be
internally consistent with electromagnetism — in particular, invariant under gauge
transformations of the vector potential.

## Background

**The Meissner state and the London description (London & London, 1935).** In a
superconductor the magnetic field is expelled from the bulk. The Londons captured this by
postulating, in addition to Maxwell's equations, a constitutive relation for the
supercurrent density j_s. Writing Λ = m/(n_s e²) (with n_s the density of "superconducting
electrons", m, e the electron mass and charge), their relations are

    Λ j_s = −(1/c) A      (in a suitable gauge),       ∇×(Λ j_s) = −(1/c) H .

Combined with Maxwell's ∇×H = (4π/c) j_s and ∇·H = 0 this gives ∇²H = H/δ² with a single
length

    δ_L = sqrt( m c² / (4π n_s e²) ) ,

so a field applied parallel to a flat surface decays into the bulk as H(z) = H_0 e^{−z/δ},
penetrating only a depth δ ≈ 10⁻⁵ cm. This reproduces the Meissner effect and the
penetration depth, and is still the workhorse for weak fields. Its load-bearing limitation
for the present problem: Λ (equivalently n_s, equivalently δ) is a *constant* at a given
temperature, independent of field strength and uniform in space. There is no room in it for
the density of superconducting electrons to be depressed near a boundary or in a strong
field, and no length scale other than δ.

**Diagnostic failures of a field-independent description.** Treating a film of half-thickness
d thermodynamically with a fixed penetration depth gives a critical field of the form
(H_c/H_c∞)² = (δ/d)·[1 − tanh(d/δ)/(d/δ)]⁻¹-type expression, i.e. a definite prediction
relating the film's enhanced critical field to δ. Measured against data this "constant" δ
is *not* constant: extracted from films of different thickness at the same temperature it
drifts markedly (for one metal near 4 K, δ ≈ 3.4×10⁻⁵ cm at d ≈ 0.3×10⁻⁵ cm but δ ≈ 2×10⁻⁵
cm at d ≈ 1.2×10⁻⁵ cm). Something about the response is changing with geometry/field that a
single fixed length cannot express.

**The surface-energy sign problem.** At a planar boundary between a normal region and a
superconducting region of the *same* metal in the critical field, the field-and-current
energy computed from the field-expulsion picture gives a surface energy of order −δ·H_c²/8π —
negative. A negative interface energy would make it energetically favorable to create ever
more interface, i.e. the homogeneous phases would be unstable against finely subdividing.
Experiment instead shows a positive surface energy (the intermediate state forms coarse
domains, not infinitely fine ones). To rescue the sign one is forced to *postulate* an extra
surface energy of non-electromagnetic origin so large — of order δ·H_c²/8π, i.e. ~10⁵ times
what the bulk-free-energy-times-atomic-length estimate (10⁻⁷–10⁻⁸ × H_c²/8π) would allow —
that introducing it has no physical justification. A genuine theory should *produce* a
positive surface energy from its own parameters.

**Landau's theory of second-order phase transitions (Landau, 1937).** A continuous
(second-order) transition is one where some quantity — an "order parameter" η, zero in the
disordered phase and nonzero in the ordered one — turns on continuously at T_c. Near T_c, η
is small, so the free energy can be expanded in it; symmetry forbids odd terms when ±η are
equivalent, so

    Φ(η) = Φ_0 + A(T) η² + B η⁴ + … ,     B > 0 ,

with A(T) = a'(T − T_c) changing sign at T_c (a' > 0). For T > T_c the minimum is at η = 0;
for T < T_c, ∂Φ/∂η² = 0 gives η² = −A/(2B) > 0, turning on as (T_c − T)^{1/2}. The free-energy
drop is −A²/(4B), and the second derivative jump produces a finite specific-heat discontinuity
at T_c — the experimental hallmark of a second-order transition. Landau applied this to
ferroelectrics (spontaneous polarization as η) and ferromagnets (spontaneous magnetization).
The transition into the superconducting state at H = 0 is, experimentally, a second-order
transition with a specific-heat jump — so its ordered phase too should be characterizable by
some order parameter that turns on at T_c.

**Two thermodynamic facts to anchor the parameters.** The condensation free-energy density
of the superconducting phase relative to the normal phase equals the magnetic energy density
of the critical field: F_n − F_s = H_c²/8π. And the critical field follows the empirically
confirmed near-parabolic law H_c(T), vanishing as (T_c − T) near T_c.

**Quantum-mechanical analogies available at the time.** The kinetic energy density of a
charged particle of mass m and charge e in an electromagnetic field, in the Schrödinger
picture, is (1/2m)|(−iℏ∇ − (e/c)A)ψ|², and the associated probability/charge current is
(eℏ/2mi)(ψ*∇ψ − ψ∇ψ*) − (e²/mc)|ψ|²A. These gauge-covariant forms — the replacement
−iℏ∇ → −iℏ∇ − (e/c)A, and the requirement that physics be unchanged under the simultaneous
ψ → ψ e^{iχ(r)}, A → A + (ℏc/e)∇χ — are standard. ℏ = 1.05×10⁻²⁷ erg·s, electron charge
e = 4.8×10⁻¹⁰ e.s.u., electron mass m = 9.1×10⁻²⁸ g.

## Baselines

**London electrodynamics (London & London, 1935).** Core idea and math as above:
supplement Maxwell with Λ j_s = −A/c, giving ∇²H = H/δ² and exponential field decay over
δ_L = sqrt(m c²/4π n_s e²). Reproduces the Meissner effect and the bulk penetration depth.
Gaps it leaves open: its single constant Λ (and hence n_s, δ) is fixed at each temperature
and uniform, so it cannot represent a spatially varying or field-suppressed degree of
superconductivity; the penetration depth it extracts from films of different thickness is
not constant against data; and the surface energy it implies between coexisting phases comes
out negative, contradicting the observed positive value and forcing an unjustified ad hoc
non-electromagnetic surface term.

**Thermodynamic critical-field treatment of films.** Core idea: equate the Gibbs free
energies of the fully-superconducting and normal states of a film of thickness 2d in a
parallel field, using the field profile from a fixed penetration depth, to get the film's
critical field as a function of d and δ. Gap: the resulting relation does not fit measured
critical fields of films of varying thickness with a single δ — the extracted δ drifts —
because it treats the superconducting density as rigid and the interface as sharp.

**Landau's second-order-transition free-energy expansion (Landau, 1937).** Core idea: expand
the free energy of a system undergoing a continuous transition in powers of a small order
parameter η, Φ = Φ_0 + A(T)η² + Bη⁴ with A(T) changing sign at T_c and B > 0, giving
η² = −A/2B below T_c and a specific-heat jump. It is a *uniform-system, zero-field*
thermodynamic theory of the transition. Gap for superconductivity: as written it has no
spatial dependence and no coupling to the magnetic field or to electric current, so by
itself it says nothing about field penetration, surface energy between phases, or the
destruction of superconductivity by a field — the very magnetic phenomena that define a
superconductor.

## Evaluation settings

The natural quantities against which any such theory would be checked, all of which exist as
measurements before the theory:
- The bulk penetration depth δ₀ of a weak magnetic field (≈10⁻⁵ cm), and its (small)
  dependence on a superimposed strong field, especially for thin films.
- The critical field H_c(T) of bulk samples and its enhancement in thin films of varying
  thickness 2d.
- The specific-heat jump at T_c (fixing the second-order character and the parabolic H_c law).
- The sign and magnitude of the surface energy at a normal/superconducting boundary,
  inferred from the coarseness of the intermediate-state domain structure.
- The destruction of superconductivity in films by an applied field and by a transport
  current.
Representative materials are the elemental superconductors (e.g. mercury, tin, aluminium,
lead) with T_c of a few kelvin. Length scales are in 10⁻⁵ cm; fields in gauss; energies per
unit area in erg/cm².

## Code framework

A small numerical scaffold for the one-dimensional problem of how the superconducting state
joins onto a normal region (or vacuum) across a planar interface, in dimensionless units —
length measured in some characteristic length to be identified, the order-quantity measured
relative to its deep-bulk value f = (quantity)/(bulk value), 0 ≤ f ≤ 1. The pieces that
already exist are the ODE integrator and quadrature; the physics to be filled in is the
equation governing f(x) across the wall and the interface-energy functional.

```python
import numpy as np
from scipy.integrate import solve_ivp, quad

# Known primitives: a generic 1-D boundary-value / first-integral integrator and a quadrature
# rule.  Dimensionless conventions: x in units of a characteristic length; f = (local degree
# of order)/(bulk value), with f -> 1 deep in the ordered region and f -> 0 in the disordered
# region.

def order_parameter_profile(L=14.0, n=4001):
    """Integrate the 1-D profile f(x) joining the ordered (f->1) and disordered (f->0)
    regions across a planar interface, in the absence of an applied field."""
    x = np.linspace(-L, L, n)

    def rhs(x, f):
        # TODO: the equation we will derive for how f varies through the wall
        pass

    # TODO: choose the boundary/initial data implied by that equation and integrate
    raise NotImplementedError


def interface_energy():
    """Energy per unit area stored in the transition layer between the two phases,
    obtained by integrating an energy-density excess across the wall."""
    # TODO: the energy-density functional whose integral over the wall gives this
    raise NotImplementedError


if __name__ == "__main__":
    # TODO: integrate the profile, then evaluate the interface energy and its sign
    pass
```
