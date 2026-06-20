# Context: the second autocorrelation (autoconvolution Hölder-ratio) inequality

## Research question

For a non-negative `f`, the autoconvolution Hölder ratio `R(f) = ||f*f||_2^2 / (||f*f||_inf·||f*f||_1) ≤ 1`
is unattained, and `C2 := sup_f R(f)` (Barnard–Steinerberger, arXiv:1903.08731) is approached by explicit
constructive lower bounds. Hierarchical β-annealed gradient refinement on a few hundred step-function
pieces reaches the high `0.89`s with the gradient still moving. The question here is the endpoint: lift once
more to thousands of pieces and spend a long, sharply-annealed gradient run to reach the published
step-function frontier — matching the best `~575`/`539`-piece results (Boyer–Li `0.901564`, Jaech–Joseph
`~0.9016`) — knowing the absolute record (AlphaEvolve-V2's `~50000`-piece `0.96102`) sits far above and was
bought by an evolutionary search with orders of magnitude more compute.

## Construction class and scoring

Non-negative step function `f = Σ v_n·1_[n,n+1)`, `v_n ≥ 0`, translation/dilation invariant. Autoconvolution
piecewise linear, node values `L_j = (v*v)_{j−1}`, `L_0 = L_{2N} = 0`. Norms: `||f*f||_inf = max_j L_j`,
`||f*f||_1 = ½ Σ_j(L_j+L_{j+1})`, `||f*f||_2^2 = ⅓ Σ_j(L_j^2+L_jL_{j+1}+L_{j+1}^2)`. Score `R`, higher
better, `≤ 1`. Self-convolution via FFT (`O(N log N)`), making long runs at `N` in the thousands affordable.

## Known reference points

Flat floor `0.6667`; Matolcsi–Vinuesa 20-step `0.88922`; AlphaEvolve 50-step `0.89628`; Boyer–Li 575-step
`0.901564`; Jaech–Joseph 539-step `~0.9016`; AlphaEvolve-V2 record `0.96102` (ImprovEvolve later `0.96258`);
Hölder ceiling `1.0`. This method is the step-function frontier a single bounded hierarchical gradient
constructor can reach; the gap to `0.96102` is the open part of the problem.
