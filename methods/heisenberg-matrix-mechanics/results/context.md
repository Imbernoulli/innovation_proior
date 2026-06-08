# Context: building a mechanics of the atom from observable quantities

## Research question

By the mid-1920s the quantum theory of the atom is a patchwork that works in a few places
and collapses everywhere else. The problem to be solved is sharp: **find a mechanics of the
atom that actually predicts what spectroscopists measure — the frequencies, intensities, and
polarizations of spectral lines — without leaning on quantities that no experiment can reach.**

The existing scheme computes those measurable line frequencies and intensities, but it
obtains them as constituent relations between quantities that cannot be observed even in
principle: the position of the electron in its orbit and its orbital period. A solution would
have to do two things at once. First, it must reproduce the empirical successes of the old
theory — the hydrogen spectrum, the fine structure, the Stark effect — where they exist.
Second, and decisively, it must keep working in the cases where the old theory fails: the
hydrogen atom in crossed electric and magnetic fields, the response of an atom to a
periodically varying field (dispersion), the helium atom, and atoms with many electrons. The
recurring symptom in every failure is the same — a calculation built on the electron's orbit
gives the wrong answer or no answer at all. So a real fix probably cannot be a better orbit
calculation; it has to be a theory whose primitive quantities are the observable ones, with
the orbit removed from the foundations entirely.

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

This is already a radical break. Classically a bound charge radiates at its orbital frequency
and the overtones of it; Bohr severs that link entirely — the radiated frequency is an energy
*difference* divided by h and has nothing to do with the orbital period except in one limit.

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
counterpart of the transition probability. This dictionary is the only systematic way the old
theory connects an orbit (a one-state object) to a spectral line (a two-state object).

**Adiabatic invariants.** Ehrenfest's adiabatic principle explains *why* the action
variables J are the right things to quantize: under a slow change of the system's parameters
the action of a periodic motion is invariant, and Burgers proved each action variable is
adiabatically invariant, so a quantum condition imposed on J is stable under slow
perturbations. This underwrites ∮ p dq = nh but does nothing to make the orbit observable.

**Dispersion theory — the crack that widened.** Classical dispersion theory (Helmholtz,
Lorentz, Drude) models a medium as charged oscillators with resonance frequencies at the
absorption lines, and explains anomalous dispersion (the index of refraction dropping near an
absorption frequency). Transplanting this into the Bohr atom is where the orbit picture
visibly fails: the dispersion oscillators resonate at the *transition* frequencies
(W_i − W_f)/h, not at the electron's orbital frequency, and Bohr had been forced to sever
those two frequencies to get the Balmer formula at all. Dispersion therefore becomes anomalous
"at the wrong frequencies" if one uses orbital frequencies — a direct, quantitative
embarrassment for the orbit.

Ladenburg (1921) took the first step out: drawing on Einstein's radiation theory he replaced
the amplitudes of the classical dispersion formula by Einstein transition probabilities and
the number of dispersion electrons by the number of quantum jumps, producing a formula written
in terms of transitions — but only for the ground state. Kramers (1924) generalized it. He
derived a classical dispersion formula having the form of an action-derivative of an
expression built from the amplitudes and frequencies of the induced oscillations, then turned
it into a quantum formula by three substitutions: amplitudes → transition probabilities,
orbital frequencies → transition frequencies, and — the key move — the action derivative by a
difference quotient,

  d / dJ  →  (1/h) Δ / Δn .

Because in the large-n limit transition and orbital frequencies coincide and Δn = 1 is small
against n, the difference quotient merges with the derivative, so the quantum formula reduces
to the classical one for large n; Bohr's correspondence principle then licenses the leap of
faith that it holds down to small n. The decisive structural feature, sharpened in the
Kramers–Heisenberg dispersion analysis (1925): **the resulting formula depends only on the
transitions between orbits — no single orbit appears in it anywhere.** A quantity that used to
be attached to one orbit has become an array of numbers attached to all the transitions
between orbits.

**The Thomas–Reiche–Kuhn sum rule (1925).** Kuhn and Thomas independently derived the
high-frequency limit of the Kramers dispersion formula: a sum over all transitions from a
given state of the oscillator strengths equals a constant fixed by e, m, h. It is a relation
purely among observable transition quantities.

**Where the old theory stands.** The honest assessment of the field: the orbit-based
quantization condition ∮p dq = nh works for hydrogen and the Stark effect, fails for crossed
fields, helium, and many-electron atoms; the choice of coordinates in which the condition is
imposed changes the shape of the "orbit" even when it leaves the energy alone, exposing the
orbit as an artifact; and an embarrassing empirical ambiguity recurs — the quantization gives
the action only up to an additive constant, surfacing as the unresolved question of integer
versus half-integer quantum numbers in band spectra and the anomalous Zeeman effect. The
quantities that always come out right are the spectroscopic ones: line frequencies and line
intensities, i.e. transition frequencies and (squared) transition amplitudes.

## Baselines

These are the concrete schemes a new theory of the atom would be measured against.

- **Bohr 1913 / Sommerfeld 1915–16 quantization.** Stationary states selected by ∮p dq = nh
  in action–angle variables; energies from the classical Hamiltonian on the quantized orbits;
  line frequencies from ν = (W_i − W_f)/h. *Gap:* the whole computation runs on orbits, which
  are unobservable and, for crossed fields and many electrons, give wrong or no results; and
  it leaves the integer/half-integer quantization unresolved.

- **Bohr correspondence principle.** Estimates transition frequencies and intensities by
  matching them to the harmonics and Fourier amplitudes A_α(n) of the classical orbit in the
  large-n limit. *Gap:* qualitative and asymptotic — it tells you the *form* of the answer for
  large n by analogy, but it is not a closed calculational scheme, and it still references the
  orbit's Fourier series.

- **Ladenburg (1921) dispersion.** Rewrites the classical dispersion formula with Einstein
  transition probabilities replacing amplitudes, so the formula speaks of jumps, not
  electrons. *Gap:* restricted to the ground state; not a general mechanics.

- **Kramers (1924) / Kramers–Heisenberg (1925) dispersion.** A quantum dispersion formula
  obtained by the transcription d/dJ → (1/h)Δ/Δn, expressed entirely in transition
  frequencies and transition amplitudes, with no orbit appearing. *Gap:* it is a formula for
  one phenomenon (scattering of light), not a mechanics — it does not by itself tell you how to
  determine the transition amplitudes and frequencies of an arbitrary bound system from its
  forces. But it is the template: it shows that an observable-only, transition-indexed
  description is possible and consistent.

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
Spectroscopy supplies line frequencies and line intensities, but the algebra and action rule
for quantities indexed by lines are still open.

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

def phase_integral(p, q):          # J = closed-integral p dq, to be set = n h classically
    raise NotImplementedError      # TODO: express this using line-indexed data


class LineQuantity:
    """Placeholder for a quantity described by spectral lines instead of one orbit."""
    def __init__(self, *args, **kwargs):
        pass                       # TODO: choose the index structure

    def __mul__(self, other):
        # TODO: define the product of two line-indexed quantities
        raise NotImplementedError

def line_quantization_rule(*args, **kwargs):
    # TODO: replace the absolute phase integral by a rule on line data
    raise NotImplementedError

def oscillator_line_solution(omega0, lam, *args, **kwargs):
    # TODO: determine frequencies, intensities, and energy levels for
    #       xddot + omega0**2 * x + lam * x**2 = 0
    raise NotImplementedError
```
