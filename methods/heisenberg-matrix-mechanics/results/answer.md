# Matrix mechanics: a mechanics of the atom from observable quantities

## The problem it solves

The Bohr–Sommerfeld old quantum theory computes spectral lines from classical electron orbits
quantized by ∮ p dq = n h. The orbit is unobservable in principle (the first short-wavelength
quantum needed to "see" it Compton-ejects the electron), the Bohr frequency condition
ν = (W(n) − W(m))/h already negates the orbital kinematics, and the orbit-based scheme fails
outright for crossed fields, helium, and many-electron atoms, while leaving the
integer/half-integer quantization unresolved. The goal: a mechanics founded **exclusively on
relations between quantities that are observable in principle** — the transition frequencies
ω(n, n−α) and transition amplitudes a(n, n−α) that spectroscopy actually delivers.

## The key idea

Refuse to write the position x(t). Keep the classical *laws* (the equation of motion ẍ + f(x)
= 0 and the phase integral) but **reinterpret every classical quantity as a two-index array
indexed by the pair of stationary states a transition connects**, oscillating at the
observable frequency ω(n, n−α). The entire content then reduces to the algebra of these
arrays — and that algebra turns out to be non-commutative matrix multiplication, paired with
an observable quantum condition.

## The multiplication rule (the central result)

Classically a product of Fourier series convolves their coefficients symmetrically:
C_β(n) = Σ_α A_α(n) B_{β−α}(n). For α > 0, A_α(n) is reinterpreted as the transition leg
X(n, n−α)e^{iω(n,n−α)t}; the negative-frequency conjugate is represented by the reversed leg,
not by flipping a sign inside a one-index Fourier label. **Demanding that each product term sit
at an allowed (Ritz) frequency**
ω(n, n−β) = ω(n, n−α) + ω(n−α, n−β) (an identity from ω = ΔW/ℏ, valid only if the second leg
begins where the first ends) forces the factors to chain through the intermediate state:

  **C(n, n−β) = Σ_α X(n, n−α) Y(n−α, n−β).**

This is the rule for multiplying two square arrays of numbers (row × column). The summed
intermediate index sits in the middle of an *ordered* chain, so swapping the factors gives a
different sum:

  **x y ≠ y x** in general.

The non-commutativity is forced by the same Ritz structure as the rule itself.

## The quantum condition

The classical phase integral, in Fourier form, is

  ∮ m ẋ² dt = 2π m Σ_α |a_α(n)|² α² ω_n = n h.

Because the action is observable only through dJ/dn = h (the absolute value is fixed only up
to a constant — the source of the half-integer ambiguity), differentiate with respect to n,

  h = 2π m Σ_α α · d/dn ( α ω_n |a_α(n)|² ),

then apply the dispersion-theory transcription
α∂Φ(n,α)/∂n → Φ(n+α,n) − Φ(n,n−α). The first term must be ordered as the line from n+α to n,
so its frequency is positive rather than sign-flipped. The two-sided sum folds to a one-sided
sum (2π → 4π), giving

  **h = 4π m Σ_{α=1}^{∞} [ |a(n+α, n)|² ω(n+α, n) − |a(n, n−α)|² ω(n, n−α) ].**

This is the **Thomas–Reiche–Kuhn sum rule** — the high-frequency limit of the Kramers
dispersion formula. The remaining additive constant is fixed by the **normal-state condition
a(n₀, n₀ − α) = 0** for all α > 0 ("no state below the lowest to radiate to"), which
simultaneously forces **integer** quantum numbers and removes the half-integer ambiguity.

## The method, stated cleanly

Equation of motion ẍ + f(x) = 0 and quantum condition (16), with all products read by the
chaining rule, **together determine the transition frequencies, energies, and transition
amplitudes of any one-degree-of-freedom system** — no inspired guesswork.

## Worked benchmark: the anharmonic oscillator

For ẍ + ω₀² x + λ x² = 0, the classical Fourier ansatz x = λa₀ + a₁cos ωt + λa₂cos 2ωt + …
becomes, on reinterpretation, recursion relations in which every square/product chains through
an intermediate state, e.g.

  ω₀² a(n,n) + ¼[a²(n+1,n) + a²(n,n−1)] = 0,
  (−ω²(n,n−2) + ω₀²) a(n,n−2) + ½ a(n,n−1) a(n−1,n−2) = 0, …

To lowest order the quantum condition gives, with the normal-state condition fixing the
constant,

  a²(n, n−1) = n h / (π m ω₀),   a(n, n−τ) = κ(τ) √(n!/(n−τ)!) → κ(τ) n^{τ/2} (large n),

and the energy W = m ẋ²/2 + m ω₀² x²/2, built from the transitions **both up and down** from
state n,

  [x²]nn = ¼[a²(n,n−1) + a²(n+1,n)],
  [ẋ²]nn = ω₀²[x²]nn,

  **W(n, n) = (n + ½) h ω₀ / 2π** — the zero-point ½-quantum falls out, with no classical
  counterpart.

The off-diagonal energy elements W(n, n−α), α ≠ 0, vanish (energy is a true constant of the
motion = diagonal array), and ω(n,n−1)/2π = (1/h)[W(n) − W(n−1)] holds as a *consequence*. For
ẍ + ω₀² x + λ x³ = 0, carried to order λ²,

  W = (n+½)hω₀/2π + λ·3(n²+n+½)h²/(8·4π²ω₀²m) − λ²·h³/(512π³ω₀⁵m²)(17n³ + 51/2 n² + 59/2 n + 21/2),

which agrees exactly with the independent Kramers–Born perturbation calculation.

## Final form

```python
import numpy as np

# (1) The kinematics: transition-indexed arrays multiply by chaining through the
#     intermediate state -- forced by the Ritz combination principle. Equivalent to
#     square-matrix multiplication; non-commutative.
def array_product(X, Y, states):
    """C(i,k) = sum_m X(i,m) Y(m,k).  X[(i,j)] = amplitude on transition i -> j."""
    return {(i, k): sum(X.get((i, m), 0) * Y.get((m, k), 0) for m in states)
            for i in states for k in states}

# (2) The quantum condition (Thomas-Reiche-Kuhn sum rule), observable-only form.
def quantum_condition_lhs(a, omega, n, alpha_max, m):
    """= h.  a[(i,j)], omega[(i,j)] on transition i -> j."""
    s = sum(abs(a.get((n + al, n), 0))**2 * omega.get((n + al, n), 0)
            - abs(a.get((n, n - al), 0))**2 * omega.get((n, n - al), 0)
            for al in range(1, alpha_max + 1))
    return 4 * np.pi * m * s

# (3) The harmonic-oscillator solution the method lands on (lowest order):
#       a^2(n, n-1) = n h / (pi m omega0)          (normal state: a(0,-1) = 0)
#       W(n, n)     = (n + 1/2) h omega0 / (2 pi)   (zero-point energy from up+down legs)
def harmonic_oscillator(n, h, m, omega0):
    a2_down = n * h / (np.pi * m * omega0)           # |a(n, n-1)|^2
    a2_up = (n + 1) * h / (np.pi * m * omega0)       # |a(n+1, n)|^2
    energy = 0.25 * m * omega0**2 * (a2_down + a2_up)
    return a2_down, energy
```
