# Context: building a mechanics of the atom from observable quantities

## Research question

By the mid-1920s the quantum theory of the atom is a patchwork of postulates and
correspondence arguments. The question to be settled is how to compute, for an atomic system,
the quantities that spectroscopy actually measures — the frequencies, intensities, and
polarizations of spectral lines — from the forces acting in the system.

The existing scheme obtains those line frequencies and intensities as relations between the
position of the electron in its orbit and its orbital period. The standing question is what a
mechanics of the atom looks like and which quantities it should take as primitive.

## Background

**The Bohr–Sommerfeld old quantum theory.** Bohr's 1913 atom has the electron moving on
classical orbits, but only a discrete set of "stationary states" is allowed, and the atom
radiates not while orbiting but only when it jumps between states. Two postulates carry the
theory. The quantization condition, sharpened by Sommerfeld (1915–16) and Wilson and
Ishiwara, restricts the classical phase integral of each degree of freedom to integer
multiples of Planck's constant,

  ∮ p dq = J = n h,

with p the momentum conjugate to a coordinate q and the integral taken over one period.
Schwarzschild and Epstein recognized that these phase integrals are the action variables J of
celestial mechanics, computed in action–angle coordinates; restricting J = nh selects a
discrete family of Keplerian orbits of varying size and eccentricity. Sommerfeld's treatment
of the relativistic fine structure of hydrogen, and the Schwarzschild–Epstein treatment of
the Stark effect, were the theory's showpieces. The second postulate is the Bohr frequency
condition: the frequency of the light emitted or absorbed in a jump between stationary states
of energy W(n) and W(m) is

  ν = ( W(n) − W(m) ) / h.

This is a radical break. Classically a bound charge radiates at its orbital frequency and the
overtones of it; Bohr makes the radiated frequency an energy *difference* divided by h, tied
to the orbital period only in one limit.

**The correspondence principle.** Bohr's bridge back to classical physics. In the limit of
large quantum numbers the transition frequency ν(n, n−α) for the jump n → n−α merges with the
α-th overtone α·ω(n) of the orbital frequency, and the line *intensity* for that jump follows
laws qualitatively like the intensity of the α-th term in the Fourier expansion of the
classical orbit. Concretely: write the classical coordinate of a periodic orbit as a Fourier
series

  x(t) = Σ_α A_α(n) e^{i α ω(n) t},

with ω(n) the orbital frequency for state n. Then the principle sets up a dictionary — the
α-th Fourier amplitude A_α(n) of the orbit is the classical counterpart of the quantum
amplitude for the transition n → n−α, and the radiated power in the α-th harmonic is the
counterpart of the transition probability. This dictionary is the systematic way the old
theory connects an orbit (a one-state object) to a spectral line (a two-state object).

**Adiabatic invariants.** Ehrenfest's adiabatic principle explains *why* the action
variables J are the right things to quantize: under a slow change of the system's parameters
the action of a periodic motion is invariant, and Burgers proved each action variable is
adiabatically invariant, so a quantum condition imposed on J is stable under slow
perturbations. This underwrites ∮ p dq = nh.

**Dispersion theory.** Classical dispersion theory (Helmholtz, Lorentz, Drude) models a
medium as charged oscillators with resonance frequencies at the absorption lines, and explains
anomalous dispersion (the index of refraction dropping near an absorption frequency). In the
Bohr atom the dispersion oscillators resonate at the *transition* frequencies (W_i − W_f)/h,
the same frequencies that enter the Balmer formula.

Ladenburg (1921) drew on Einstein's radiation theory to replace the amplitudes of the
classical dispersion formula by Einstein transition probabilities and the number of dispersion
electrons by the number of quantum jumps, producing a formula written in terms of transitions —
for the ground state. Kramers (1924) generalized it. He derived a classical dispersion formula
having the form of an action-derivative of an expression built from the amplitudes and
frequencies of the induced oscillations, then turned it into a quantum formula by three
substitutions: amplitudes → transition probabilities, orbital frequencies → transition
frequencies, and the action derivative by a difference quotient,

  d / dJ  →  (1/h) Δ / Δn .

Because in the large-n limit transition and orbital frequencies coincide and Δn = 1 is small
against n, the difference quotient merges with the derivative, so the quantum formula reduces
to the classical one for large n; Bohr's correspondence principle then licenses applying it
down to small n. A structural feature, sharpened in the Kramers–Heisenberg dispersion analysis
(1925): the resulting formula depends only on the transitions between orbits — no single orbit
appears in it. The orbital quantities of the classical starting point survive in the formula
through the transition-indexed data the substitutions leave behind.

**The Thomas–Reiche–Kuhn sum rule (1925).** Kuhn and Thomas independently derived the
high-frequency limit of the Kramers dispersion formula: a sum over all transitions from a
given state of the oscillator strengths equals a constant fixed by e, m, h. It is a relation
among observable transition quantities.

**Where the old theory stands.** The orbit-based quantization condition ∮p dq = nh gives the
hydrogen spectrum, the fine structure, and the Stark effect; for the hydrogen atom in crossed
electric and magnetic fields, the helium atom, and many-electron atoms it is applied with mixed
results. The choice of coordinates in which the condition is imposed changes the shape of the
orbit while leaving the energy alone. The quantization gives the action up to an additive
constant, surfacing as the open question of integer versus half-integer quantum numbers in
band spectra and the anomalous Zeeman effect. The quantities the theory delivers cleanly are
the spectroscopic ones: line frequencies and line intensities, i.e. transition frequencies and
(squared) transition amplitudes.

## Baselines

These are the concrete schemes a new theory of the atom would be measured against.

- **Bohr 1913 / Sommerfeld 1915–16 quantization.** Stationary states selected by ∮p dq = nh
  in action–angle variables; energies from the classical Hamiltonian on the quantized orbits;
  line frequencies from ν = (W_i − W_f)/h.

- **Bohr correspondence principle.** Estimates transition frequencies and intensities by
  matching them to the harmonics and Fourier amplitudes A_α(n) of the classical orbit in the
  large-n limit; it gives the *form* of the answer for large n by analogy.

- **Ladenburg (1921) dispersion.** Rewrites the classical dispersion formula with Einstein
  transition probabilities replacing amplitudes, so the formula speaks of jumps, for the
  ground state.

- **Kramers (1924) / Kramers–Heisenberg (1925) dispersion.** A quantum dispersion formula
  obtained by the transcription d/dJ → (1/h)Δ/Δn, expressed entirely in transition
  frequencies and transition amplitudes, with no orbit appearing. It is a formula for the
  scattering of light, expressed in observable, transition-indexed quantities.

## Evaluation settings

The natural yardsticks for a candidate mechanics, all available before any new theory exists:

- **Spectroscopic line data.** Measured frequencies and relative intensities of spectral
  lines — for the hydrogen Balmer series and fine structure, the Stark and Zeeman patterns,
  and band spectra. These are exactly the transition frequencies ω(n, n−α) and the
  intensities ∝ |amplitude|² that any theory must reproduce.

- **Benchmark dynamical systems with one degree of freedom.** The simple harmonic oscillator;
  the anharmonic oscillators ẍ + ω₀² x + λ x² = 0 and ẍ + ω₀² x + λ x³ = 0, soluble in
  classical mechanics by Fourier expansion with the coefficients as power series in λ; and the
  rigid rotator, an electron orbiting at fixed radius. These are the toy problems on which a
  candidate mechanics can be checked term by term against the classical limit.

- **Cross-checks from existing quantum results.** The Kramers–Born perturbative treatment of
  the anharmonic oscillator energy; the Thomas–Reiche–Kuhn sum rule; the Goudsmit–Kronig–Hönl
  and Ornstein–Burger intensity sum rules for multiplets and the Zeeman effect. A consistent
  new mechanics should reproduce these as special cases or limits.

- **The internal consistency tests.** Whether the energy comes out time-independent, with no
  transition-frequency components, whether ω(n, n−1) = (2π/h)[W(n) − W(n−1)] holds as a
  *consequence* rather than an input, and whether the radiation vanishes in the lowest state.

## Code framework

The available computational scaffold is the classical Fourier bookkeeping for a periodic
one-degree-of-freedom system, together with the equation of motion and the phase integral.
Spectroscopy supplies line frequencies and line intensities.

```python
import numpy as np

# Classical periodic motion as a Fourier series over harmonics of one orbital frequency,
# for a one-degree-of-freedom bound system in state labeled n.
def classical_coordinate(amplitudes, omega_n, t):
    """x(t) = sum_alpha A_alpha(n) * exp(i * alpha * omega_n * t).
    amplitudes[alpha] = A_alpha(n); index alpha runs over integer harmonics."""
    x = 0.0 + 0.0j
    for alpha, A in amplitudes.items():
        x += A * np.exp(1j * alpha * omega_n * t)
    return x

# Classical product of two such series: symmetric convolution of Fourier coefficients.
def classical_product_coefficients(A, B):
    """C_beta = sum_alpha A_alpha * B_{beta-alpha}."""
    C = {}
    betas = range(min(A)+min(B), max(A)+max(B)+1)
    for beta in betas:
        C[beta] = sum(A.get(a, 0) * B.get(beta - a, 0) for a in A)
    return C

# Classical equation of motion and phase-integral condition.
def equation_of_motion(x, xddot, force):
    """xddot + f(x) = 0 ; force(x) supplies f(x)."""
    return xddot + force(x)

def phase_integral(p, q):          # J = closed-integral p dq, set = n h classically
    raise NotImplementedError      # TODO


class LineQuantity:
    """Placeholder for a quantity described by spectral lines instead of one orbit."""
    def __init__(self, *args, **kwargs):
        pass                       # TODO

def oscillator_line_solution(omega0, lam, *args, **kwargs):
    # TODO: determine frequencies, intensities, and energy levels for
    #       xddot + omega0**2 * x + lam * x**2 = 0
    raise NotImplementedError
```
