# Context: the second autocorrelation (autoconvolution Hölder-ratio) inequality

## Research question

For a non-negative function `f` on the real line, the autoconvolution `f*f(x) = ∫ f(t) f(x−t) dt` is a
smooth, bump-shaped object. Hölder's inequality bounds the ratio

```
R(f) = ||f*f||_2^2 / ( ||f*f||_inf · ||f*f||_1 )  ≤  1,
```

with equality only if `f*f` is an indicator — which it never is, since the autoconvolution of a
non-negative function is continuous and spread out. Barnard and Steinerberger (arXiv:1903.08731) asked how
large `C2 := sup_f R(f)` can be made; every advance since is a constructive lower bound — an explicit `f`
with a measured `R(f)`. The question here is the very first one on this problem: what is the simplest legal
construction, and what does it score? It fixes the floor that every subsequent construction must beat.

## Construction class and scoring

The standard class (used by AlphaEvolve App. B.2, Boyer–Li, Matolcsi–Vinuesa) is the non-negative
**piecewise-constant step function** `f = Σ_{n=0}^{N−1} v_n·1_[n,n+1)`, `v_n ≥ 0`. The objective is
invariant under translation and dilation, so only the heights and their count `N` matter. The
autoconvolution of a step function is **piecewise linear**, fully determined by its integer node values
`L_j = (f*f)(j) = (v*v)_{j−1}` (with `L_0 = L_{2N} = 0`). The three norms are exact integrals of the
piecewise-linear curve:

```
||f*f||_inf = max_j L_j
||f*f||_1   = ½ Σ_j (L_j + L_{j+1})
||f*f||_2^2 = ⅓ Σ_j (L_j^2 + L_j·L_{j+1} + L_{j+1}^2)
```

and `R = ||f*f||_2^2 / (||f*f||_inf·||f*f||_1)`. The score is this ratio; higher is better, `R ≤ 1` always.

## Known reference points

Flat indicator → `2/3 ≈ 0.6667` (the floor). Matolcsi–Vinuesa 20-step → `0.88922`. AlphaEvolve 50-step →
`0.89628` (arXiv:2506.13131). Boyer–Li 575-step → `0.901564` (arXiv:2506.16750). AlphaEvolve-V2 record →
`0.96102` (~50000-step irregular). Hölder ceiling `1.0`, unattained. This method targets only the floor.
