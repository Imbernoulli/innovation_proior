# Context: the second autocorrelation (autoconvolution Hölder-ratio) inequality

## Research question

For a non-negative `f`, the autoconvolution Hölder ratio
`R(f) = ||f*f||_2^2 / (||f*f||_inf·||f*f||_1) ≤ 1`,
and `C2 := sup_f R(f)` (Barnard–Steinerberger, arXiv:1903.08731) is approached by explicit constructive
lower bounds. The question is to push the lower bound on `C2` as high as possible: over non-negative step
functions, produce a construction whose ratio `R` is as close to `1` as can be achieved, scored under the
exact evaluator.

## Construction class and scoring

Non-negative step function `f = Σ v_n·1_[n,n+1)`, `v_n ≥ 0`, translation/dilation invariant. Autoconvolution
piecewise linear, node values `L_j = (v*v)_{j−1}`, `L_0 = L_{2N} = 0`. Norms: `||f*f||_inf = max_j L_j`,
`||f*f||_1 = ½ Σ_j(L_j+L_{j+1})`, `||f*f||_2^2 = ⅓ Σ_j(L_j^2+L_jL_{j+1}+L_{j+1}^2)`. Score `R`, higher
better, `≤ 1`. Self-convolution via FFT (`O(N log N)`), so scoring a `~50000`-piece function is sub-second.
Because `R` is invariant to grid spacing and offset, the unit-grid `fftconvolve` evaluator and the published
`[-1/4,1/4]` `numpy.convolve` verifier return the identical number.

## Known reference points

Flat floor `0.6667`; Matolcsi–Vinuesa 20-step `0.88922`; AlphaEvolve 50-step `0.89628`; Boyer–Li 575-step
`0.901564`; Jaech–Joseph 539-step `~0.9016`; AlphaEvolve-V2 `~50000`-step `0.96102` (TTT-Discover
`50000`-step `0.959180`; Together AI `100000`-point `0.961206`; ImprovEvolve `0.96258`); Hölder ceiling `1.0`.
