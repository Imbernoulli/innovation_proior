# Context: the second autocorrelation (autoconvolution Hölder-ratio) inequality

## Research question

For a non-negative function `f`, the autoconvolution `f*f(x) = ∫ f(t) f(x−t) dt` satisfies the Hölder bound

```
R(f) = ||f*f||_2^2 / ( ||f*f||_inf · ||f*f||_1 )  ≤  1,
```

unattained because an autoconvolution is never an indicator. Barnard–Steinerberger (arXiv:1903.08731)
asked how large `C2 := sup_f R(f)` can be; the answer is approached by explicit constructive lower bounds.
The simplest candidate, a flat step function, scores exactly `2/3` because its autoconvolution is a rigid
triangle — and refining the grid does not move it, since only the *shape* of the heights affects `R`. The
question here is the first real construction: how to introduce variation among the heights and search for a
non-flat profile whose autoconvolution bends away from the tent, clearing the `2/3` floor.

## Construction class and scoring

Non-negative piecewise-constant step function `f = Σ_{n=0}^{N−1} v_n·1_[n,n+1)`, `v_n ≥ 0`, translation/
dilation invariant so only heights+count matter. Autoconvolution is piecewise linear with node values
`L_j = (v*v)_{j−1}`, `L_0 = L_{2N} = 0`. Norms by exact piecewise-linear formulas:
`||f*f||_inf = max_j L_j`, `||f*f||_1 = ½ Σ_j(L_j+L_{j+1})`, `||f*f||_2^2 = ⅓ Σ_j(L_j^2+L_jL_{j+1}+L_{j+1}^2)`.
Score `R`, higher better, `≤ 1`. Self-convolution via FFT.

## Known reference points

Flat floor `0.6667`; Matolcsi–Vinuesa 20-step `0.88922`; AlphaEvolve 50-step `0.89628`; Boyer–Li 575-step
`0.901564`; AlphaEvolve-V2 record `0.96102`; ceiling `1.0`. This method works at a coarse `N = 20` — the
resolution at which the published `0.88` band was first reached — to find the basic non-trivial shape.
