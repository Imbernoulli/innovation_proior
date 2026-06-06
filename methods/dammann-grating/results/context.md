# Context: equal-intensity beam fan-out by a thin transparent phase element

## Research question

Given a single collimated input beam (or a single object in an imaging system), produce an array of N
equally bright copies, arranged on a regular grid about the optical axis, in one passive transmissive pass,
with as little light wasted as possible. The element must be thin, fully transparent, and manufacturable.

The difficulty is the combination of constraints. A beam can be replicated with a stack of partially
reflecting plates, but that is bulky, alignment-sensitive, and the copies are hard to equalize. An absorbing
(amplitude) mask can in principle imprint any periodicity, but it throws away most of the incident light to
absorption and dumps a large fraction into the unwanted zero order. What is wanted is a single passive
surface whose *only* effect is to redistribute the incoming power among a chosen set of equal output spots —
no absorption, equal intensities, and a fabrication recipe simple enough to etch reliably.

## Background

**Fraunhofer diffraction and the Fourier-optics picture.** When a thin transparency with complex amplitude
transmittance t(x,y) is illuminated by a plane wave, its far-field (Fraunhofer) amplitude is the Fourier
transform of t. If t is *periodic* with period d, that transform is a comb of delta functions — the
diffraction orders — located at sinθ_m = mλ/d (the grating equation, with λ the wavelength and m the integer
order). The complex amplitude landing in order m equals the m-th Fourier-series coefficient c_m of one period
of t, and the intensity of that spot is |c_m|². The angular *positions* of the spots are fixed by the period
alone; their *brightnesses* are entirely a question of the Fourier coefficients of the period's shape. So
"make N equal spots" becomes "shape one period so that |c_0| = |c_1| = … = |c_{N-1}| and the rest are small."

**Amplitude gratings waste light.** A Ronchi (bar) grating modulates |t| between 0 and 1. Because it absorbs
or blocks light, its efficiency is intrinsically low and a large share of the transmitted power stays in the
zero order; the equal-intensity goal is essentially unreachable with a purely absorbing element.

**Phase gratings are lossless.** If instead |t(x)| = 1 everywhere and only the phase varies, t(x) = e^{iφ(x)},
the element absorbs nothing: all incident power is conserved and merely redistributed among the orders. This
is the key lever — a pure phase profile can put power *where you want it* without throwing any away. A
sinusoidal phase grating spreads light into orders weighted by Bessel functions J_m, which are unequal and
not freely tunable; the order weights are not independent design knobs.

**Binary (two-level) phase is the simplest to fabricate.** A phase profile that takes only two values can be
made with a single etch: regions at one depth and regions at another, one step. Choosing the two phase levels
0 and π makes the transmittance take the two real values +1 and −1. The required surface-relief step for a
0/π element is a half-wave of optical path, d = λ / (2(n − 1)) for a medium of index n in air. (For a GaAs
infrared element, n ≈ 3.26 at 12 µm gives d ≈ 2.65 µm, and n ≈ 3.30 at 5 µm gives d ≈ 1.09 µm.) Such a binary
relief profile is wavelength-independent in its *pattern*: only the etch depth scales with λ, not the
transition layout.

**Fourier coefficients of a periodic profile are the design surface.** Because the order amplitudes are the
Fourier-series coefficients of one period, and a binary 0/π period is fully described by the positions at
which it flips sign, the entire design reduces to choosing a small set of *transition points* within the
period. The order amplitudes become closed-form trigonometric sums in those transition positions.

**Efficiency is bounded for binary elements.** With only two phase levels, the fraction of input power that
can be steered into a prescribed set of equal orders has a ceiling well below what a continuously-blazed or
many-level element could reach; the figure commonly cited for binary fan-out elements is below roughly 86%.
A single transition splitting into three equal spots reaches about two-thirds of the light in the target
orders; two transitions for five equal spots reach roughly three-quarters. These are facts about the binary
constraint, knowable from the coefficient algebra before any particular layout is fixed.

## Baselines

**Beam-splitter / partially-reflecting plate stacks.** Cascade semi-reflective surfaces to peel off copies of
the beam. Core idea: geometric splitting of amplitude at each interface. Limitation: bulky, alignment- and
coating-sensitive, equal intensities require precisely graded reflectivities, and it does not scale to a
two-dimensional regular array of many spots.

**Amplitude (Ronchi / absorbing) gratings and masks.** A periodic modulation of |t| produces diffraction
orders; an arbitrary order pattern can be approached with a computer-generated absorbing mask. Core idea:
encode the desired Fourier spectrum in a binary *opacity* pattern. Math: order m carries |c_m|² where c_m is
the Fourier coefficient of the opacity pattern. Limitation: absorption makes efficiency low, the zero order
dominates, and equalizing many orders while staying bright is not achievable.

**Sinusoidal / continuous phase gratings.** A smoothly varying phase φ(x) is lossless and far brighter than an
amplitude grating. Math: a sinusoidal phase grating of modulation depth gives order amplitudes J_m. Limitation:
the order weights follow fixed special-function curves and are not independently tunable, so you cannot demand
"the first N orders exactly equal and the rest suppressed"; and a continuous blaze is harder to fabricate
faithfully than a single etch step.

## Evaluation settings

The natural yardsticks are: (i) the **uniformity** of the target orders — the spread (max−min)/(max+min) of
their intensities, ideally zero; (ii) the **diffraction efficiency** — the fraction of incident power landing
in the chosen target orders; (iii) the suppression of unwanted orders outside the target set; and (iv)
manufacturability — number of distinct phase levels (one etch step is best) and feature sizes relative to the
lithography limit. The grating equation sinθ_m = mλ/d sets the spot spacing for a chosen period d and design
wavelength λ; standard infrared substrates (e.g. GaAs) and visible glasses with known dispersion n(λ) fix the
half-wave etch depth d = λ/(2(n−1)). Designs are specified on a normalized period coordinate, so a single
transition layout serves any wavelength once the depth is scaled.

## Code framework

The pieces that already exist: NumPy/SciPy for arrays and nonlinear optimization, an FFT for checking far
fields, and a routine to render a periodic two-level profile from a list of sign-flip positions. The contribution
will occupy the empty slots below — the closed-form order amplitudes for a given transition layout, and the
search that places the transitions.

```python
import numpy as np
from scipy.optimize import minimize

def binary_profile(x, transitions, start=+1.0):
    """Render one even period (values +/-1) given sign-flip positions on [0, 0.5]."""
    y = np.full_like(x, start)
    for i, _ in enumerate(transitions):
        lo = transitions[i]
        hi = transitions[i+1] if i+1 < len(transitions) else 0.5
        y[(np.abs(x) >= lo) & (np.abs(x) < hi)] = start * (-1.0)**(i+1)
    return y

def order_amplitudes(transitions, M):
    """Diffraction-order amplitudes a_0..a_M for an even binary period defined by `transitions`.
    # TODO: the closed form for the Fourier coefficients of a binary period — the contribution."""
    pass

def design_cost(transitions, N):
    """Objective measuring (un)uniformity of the first N orders and the efficiency in them.
    # TODO: built from order_amplitudes once it exists."""
    pass

def design(N, K, restarts):
    """Search the K transition positions so the first N orders are equal and efficiency is high.
    # TODO: multistart nonlinear optimization of design_cost."""
    pass
```
