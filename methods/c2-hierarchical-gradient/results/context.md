# Context: the second autocorrelation (autoconvolution Hölder-ratio) inequality

## Research question

For a non-negative `f`, the autoconvolution Hölder ratio `R(f) = ||f*f||_2^2 / (||f*f||_inf·||f*f||_1) ≤ 1`
is unattained, and `C2 := sup_f R(f)` (Barnard–Steinerberger, arXiv:1903.08731) is approached by explicit
constructions. A coarse `~20`-piece optimized step function reaches the high `0.88`s. The question here is how
to take an optimized coarse step-function *shape* and raise its score `R` toward the band of the
published `50`–`575`-step constructions (`0.896`–`0.9016`).

## Construction class and scoring

Non-negative step function `f = Σ v_n·1_[n,n+1)`, `v_n ≥ 0`, translation/dilation invariant. Autoconvolution
piecewise linear, node values `L_j = (v*v)_{j−1}`, `L_0 = L_{2N} = 0`. Norms: `||f*f||_inf = max_j L_j`,
`||f*f||_1 = ½ Σ_j(L_j+L_{j+1})`, `||f*f||_2^2 = ⅓ Σ_j(L_j^2+L_jL_{j+1}+L_{j+1}^2)`. Score `R`, higher
better, `≤ 1`. Self-convolution via FFT (`O(N log N)`), so optimization at `N` in the hundreds is fast.

## Known reference points

Flat floor `0.6667`; Matolcsi–Vinuesa 20-step `0.88922`; AlphaEvolve 50-step `0.89628`; Boyer–Li 575-step
`0.901564`; Jaech–Joseph 539-step `~0.9016`; AlphaEvolve-V2 record `0.96102`; ceiling `1.0`. The Boyer–Li
`575`-step construction was obtained from `~10^6` optimization trajectories.
