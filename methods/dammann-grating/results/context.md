# Context: equal-intensity beam fan-out by a thin transparent phase element

## Research question

Given a single collimated input beam (or a single object in an imaging system), produce a chosen symmetric
set of equally bright copies, arranged on a regular grid about the optical axis, in one passive transmissive
pass, with as little light wasted as possible. The element must be thin, fully transparent, and manufacturable.

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
alone; their *brightnesses* are entirely a question of the Fourier coefficients of the period's shape.

**Amplitude gratings waste light.** A Ronchi (bar) grating modulates |t| between 0 and 1. Because it absorbs
or blocks light, its efficiency is intrinsically low and a large share of the transmitted power stays in the
zero order; the equal-intensity goal is essentially unreachable with a purely absorbing element.

**Phase gratings are lossless.** If instead |t(x)| = 1 everywhere and only the phase varies, t(x) = e^{iφ(x)},
the element absorbs nothing: by Parseval all incident power is conserved and merely redistributed among the
orders. A sinusoidal phase grating spreads light into orders weighted by Bessel functions J_m, which are
unequal and not freely tunable; with a single modulation-depth knob the order weights are not independent
design knobs.

**Two-level phase is the simplest to fabricate.** A phase profile that takes only two values can be made with
a single etch: regions at one depth and regions at another, one step. The surface-relief step that produces a
π phase difference is a half-wave of optical path, d = λ / (2(n − 1)) for a medium of index n in air. (For a
GaAs infrared element, n ≈ 3.26 at 12 µm gives d ≈ 2.65 µm, and n ≈ 3.30 at 5 µm gives d ≈ 1.09 µm.) Such a
relief profile is wavelength-independent in its *pattern*: only the etch depth scales with λ, not the layout.
A continuously-blazed or many-level relief can in principle steer more power into a chosen order set than a
two-level one, but it requires a gray-scale etch that is far harder to fabricate faithfully.

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
an arbitrary flat set of target orders with the rest suppressed; and a continuous blaze is harder to fabricate
faithfully than a single etch step.

## Evaluation settings

The natural yardsticks are: (i) the **uniformity** of the target orders — the spread (max−min)/(max+min) of
their intensities, ideally zero; (ii) the **diffraction efficiency** — the fraction of incident power landing
in the chosen target orders; (iii) the suppression of unwanted orders outside the target set; and (iv)
manufacturability — number of distinct phase levels (one etch step is best) and feature sizes relative to the
lithography limit. The grating equation sinθ_m = mλ/d sets the spot spacing for a chosen period d and design
wavelength λ; standard infrared substrates (e.g. GaAs) and visible glasses with known dispersion n(λ) fix the
half-wave etch depth d = λ/(2(n−1)). Designs are specified on a normalized period coordinate, so a single
normalized pattern serves any wavelength once the depth is scaled.

## Code framework

The pieces that already exist: NumPy/SciPy for arrays and nonlinear optimization, and an FFT for checking far
fields. The design routine itself is to be filled in.

```python
import numpy as np
from scipy.optimize import minimize

def design(num_orders):
    """Design a transmittance whose far field is the chosen equal-intensity order set."""
    pass
```
