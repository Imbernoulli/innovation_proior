# Context: optimal magnitude approximation for linear-phase FIR filters

## Research question

Given a desired magnitude response — typically piecewise constant: gain 1 in passbands, 0 in
stopbands, with prescribed band edges and "don't-care" transition gaps — design a length-`N`
finite-impulse-response (FIR) filter with exactly linear phase whose realized magnitude
approximates the spec as well as possible in the **worst-case** (Chebyshev, L∞) sense across the
bands. A telephony channel filter, a decimation anti-alias filter, a differentiator — each comes
with a tolerance scheme (passband ripple ≤ δ_p, stopband attenuation ≥ some dB) and a fixed tap
budget `N`.

## Background

**Linear phase forces a symmetric impulse response.** An FIR filter of length `N` has frequency
response `H(e^{jω}) = Σ_{n=0}^{N-1} h(n) e^{-jωn}`. Exactly linear phase requires
`H(e^{jω}) = ± |H(e^{jω})| e^{-jα ω}` with constant group delay; the necessary and sufficient
condition is impulse-response symmetry `h(n) = h(N-1-n)` (positive symmetry) or antisymmetry
`h(n) = -h(N-1-n)` (negative symmetry), giving delay `α = (N-1)/2`. Combining the two symmetries
with `N` odd or even yields exactly **four cases**. In every case the response factors as a real
amplitude function times a pure linear-phase term, `H = G(f) · e^{j[(Lπ/2) - ((N-1)/2)2πf]}`,
`L ∈ {0,1}`, where `G(f)` is real and is the object that must approximate the desired magnitude.
`G(f)` is a finite trigonometric sum: a cosine sum for the symmetric cases, a sine sum for the
antisymmetric ones (the latter natural for differentiators and Hilbert transformers).

**Chebyshev / minimax approximation and the equioscillation theorem.** Classical approximation
theory studies, for a continuous target `D̂(f)` on a closed set `F`, the polynomial `P` of degree
`r-1` that minimizes `max_{f∈F} Ŵ(f)|D̂(f) - P(f)|`. The cornerstone result is the **alternation
(equioscillation) theorem**: `P` is the unique best weighted-Chebyshev approximation iff the
weighted error `E(f) = Ŵ(f)[D̂(f) - P(f)]` attains its maximum magnitude with *alternating sign* at
at least `r+1` points of `F`. The theory is developed for ordinary algebraic polynomials in a
real variable on a closed interval.

**The Remez exchange (1934).** Remez gave an iterative scheme to find the best Chebyshev
approximation: guess a reference set of `r+1` points; solve for the polynomial and the error level
`δ` that make the error equal `±δ` alternately on the reference; then move the reference points to
the locations where the actual error is largest (the exchange); repeat. The error level rises
monotonically toward the optimum and the references converge to the true equioscillation points.

**Window behavior.** Simple truncation of the ideal impulse response (a rectangular window) suffers
the **Gibbs phenomenon**: a roughly **9% overshoot** at the band edge that does **not** shrink as
`N` grows — increasing `N` only narrows the "ears." The rectangular window minimizes the
integral-squared error. Tapered windows (Hamming `0.54 - 0.46 cos(2πn/(N-1))`, with peak sidelobe
~40 dB down; Blackman, sidelobes < 0.0001 of the main-lobe peak at the cost of a 3× wider main
lobe; Kaiser/Dolph–Chebyshev windows that trade ripple against transition width) reduce the ripple
relative to the rectangular window.

## Baselines

**Window method.** Take the ideal (doubly infinite) impulse response `h_d(n)` — for a lowpass cut
at `ω_c`, `h_d(n) = sin(ω_c n)/(πn)` with limiting value `ω_c/π` at `n=0` — truncate it to `N` samples by multiplying by a window `w(n)`,
and delay to make it causal. In the frequency domain this convolves the ideal brick wall with the
window's Fourier transform; the window's main-lobe width sets the transition width and its sidelobes
set the ripple. The rectangular window minimizes mean-squared error; the Gibbs overshoot is ~9%.

**Frequency-sampling method.** Specify the response at `N` equispaced frequencies `H_k` and recover
the impulse response by the inverse DFT, `h(n) = (1/N) Σ_k H_k e^{j(2π/N)kn}`. The interpolated
response is exact at the sample points but ripples in between. The ripple can be reduced by leaving
a few **transition-band samples free** and choosing them (by linear programming) to cancel ripple,
since the response is *linear* in the unknown samples.

**Linear-programming design.** Because the response `G(f) = Σ α_k cos(2πkf)` is linear in the
coefficients, one can pose "minimize `δ` subject to `-δ ≤ W(f_i)[D(f_i) - G(f_i)] ≤ δ` on a dense
frequency grid" as a linear program and let a simplex solver find the minimax coefficients. This
minimizes the worst-case error and allows exact band edges, with practical scale up to ~100
parameters.

**Maximal-ripple (Herrmann–Hofstetter) equiripple design.** Solve directly for a filter whose
error equiripples: write constraint equations setting the error to `±δ` at a guessed set of extremal
frequencies and setting its derivative to zero there, producing a system of `N-1` nonlinear
equations in `N-1` unknowns (the coefficients plus the extremal locations), solved by Newton-type
iteration. This yields genuinely equiripple filters whose extremal frequencies are outputs of the
solve.

## Evaluation settings

Standard design targets of the era, used to judge any FIR design method: **lowpass, highpass,
bandpass, bandstop** magnitude specs (gain 1 in passbands, 0 in stopbands, with named transition
widths), plus **wideband differentiators** (desired response proportional to frequency) and
**Hilbert transformers** (odd-symmetric, constant magnitude). A design is characterized by the tap
count `N`, the band-edge frequencies, the desired gain and a relative weight in each band, and the
sampling rate. The natural yardsticks are the achieved worst-case (peak) deviation in each band — in
linear units for passbands and in dB of attenuation for stopbands — the transition width, and the
compute time and tap budget needed to meet a given tolerance scheme. Filters up to several hundred
taps are within scope.

## Code framework

The available primitives are forming the ideal impulse response, evaluating a real trigonometric
amplitude on a dense frequency grid, solving finite interpolation systems, and using the DFT or
inverse DFT. The shape of the design routine is: set up the weighted approximation problem on the
prescribed bands, fit the amplitude, then fold the returned coefficients into a symmetric or
antisymmetric impulse response.

```python
import numpy as np

def desired_and_weight(grid, bands, desired, weight):
    """Sample the desired amplitude D(f) and weight W(f) on a dense frequency grid.
    Per-band: constant for lowpass/bandpass specs, frequency-dependent for a
    differentiator (D proportional to f, W a relative-error weight)."""
    D = np.empty_like(grid); W = np.empty_like(grid)
    # ... fill D, W per band (and leave transition gaps out of the grid) ...
    return D, W

def fit_amplitude(grid, D, W, num_cos):
    """Find the length-`num_cos` cosine amplitude that is best in the weighted
    worst-case (minimax) sense for desired D under weight W on `grid`.
    Returns the cosine coefficients and the achieved error level."""
    # TODO: choose and implement the optimal-approximation engine
    pass

def design_fir(numtaps, bands, desired, weight=None, grid_density=16):
    grid, D, W = build_grid(numtaps, bands, desired, weight, grid_density)
    num_cos = num_amplitude_terms(numtaps)        # count of trig terms; depends on symmetry/parity case
    alpha, dev = fit_amplitude(grid, D, W, num_cos)
    h = cosine_coeffs_to_impulse_response(alpha, numtaps)  # impose h(n)=±h(N-1-n)
    return h
```
