# Context: equal-intensity beam fan-out by a thin transparent phase element

## Research question

Given a single collimated input beam (or a single object in an imaging system), how does one produce a chosen
symmetric set of equally bright copies, arranged on a regular grid about the optical axis, in one passive
transmissive pass? The element of interest is thin, fully transparent, and manufacturable, redistributing the
incoming power among a chosen set of output spots.

## Background

**Fraunhofer diffraction and the Fourier-optics picture.** When a thin transparency with complex amplitude
transmittance t(x,y) is illuminated by a plane wave, its far-field (Fraunhofer) amplitude is the Fourier
transform of t. If t is *periodic* with period d, that transform is a comb of delta functions — the
diffraction orders — located at sinθ_m = mλ/d (the grating equation, with λ the wavelength and m the integer
order). The complex amplitude landing in order m equals the m-th Fourier-series coefficient c_m of one period
of t, and the intensity of that spot is |c_m|². The angular *positions* of the spots are fixed by the period
alone; their *brightnesses* are set by the Fourier coefficients of the period's shape.

**Amplitude gratings.** A Ronchi (bar) grating modulates |t| between 0 and 1. It absorbs or blocks part of
the incident light, and the transmitted power distributes across the orders with a share remaining in the
zero order.

**Phase gratings are lossless.** If instead |t(x)| = 1 everywhere and only the phase varies, t(x) = e^{iφ(x)},
the element absorbs nothing: by Parseval all incident power is conserved and merely redistributed among the
orders. A sinusoidal phase grating spreads light into orders weighted by Bessel functions J_m, set by a
single modulation-depth parameter.

**Surface-relief fabrication.** A phase profile is realized as a surface-relief step in a transparent
substrate. A step that produces a π phase difference is a half-wave of optical path, d = λ / (2(n − 1)) for a
medium of index n in air. (For a GaAs infrared element, n ≈ 3.26 at 12 µm gives d ≈ 2.65 µm, and n ≈ 3.30 at
5 µm gives d ≈ 1.09 µm.) Such a relief profile is wavelength-independent in its *pattern*: only the etch depth
scales with λ, not the layout. A continuously-blazed or many-level relief is produced with a gray-scale etch;
a profile taking a small number of discrete depths is produced with a correspondingly small number of etch
steps.

## Baselines

**Beam-splitter / partially-reflecting plate stacks.** Cascade semi-reflective surfaces to peel off copies of
the beam. Core idea: geometric splitting of amplitude at each interface, with the reflectivities setting the
relative copy intensities.

**Amplitude (Ronchi / absorbing) gratings and masks.** A periodic modulation of |t| produces diffraction
orders; an order pattern can be approached with a computer-generated absorbing mask. Core idea: encode the
desired Fourier spectrum in a binary *opacity* pattern. Math: order m carries |c_m|² where c_m is the Fourier
coefficient of the opacity pattern.

**Sinusoidal / continuous phase gratings.** A smoothly varying phase φ(x) is lossless. Math: a sinusoidal
phase grating of modulation depth gives order amplitudes J_m, which follow the Bessel-function curves as the
single depth parameter is varied.

## Evaluation settings

The natural yardsticks are: (i) the **uniformity** of the target orders — the spread (max−min)/(max+min) of
their intensities, ideally zero; (ii) the **diffraction efficiency** — the fraction of incident power landing
in the chosen target orders; (iii) the suppression of unwanted orders outside the target set; and (iv)
manufacturability — number of distinct phase levels and feature sizes relative to the lithography limit. The
grating equation sinθ_m = mλ/d sets the spot spacing for a chosen period d and design wavelength λ; standard
infrared substrates (e.g. GaAs) and visible glasses with known dispersion n(λ) fix the half-wave etch depth
d = λ/(2(n−1)). Designs are specified on a normalized period coordinate, so a single normalized pattern serves
any wavelength once the depth is scaled.

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
