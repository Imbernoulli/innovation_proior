# Metamer-Proof Pigment Recipe

## Problem

You run a paint-mixing lab. You are given a palette of `M` candidate pigments
and a target colour sample, and must output a **recipe**: nonnegative mixing
concentrations `c_1..c_M` over the palette, with `sum(c_i) = 1`. The recipe
will be checked not under one light, but under **several different
illuminants**, and it must look right under all of them.

Each pigment `i` is described by its Kubelka-Munk absorption and scattering
spectra `K_i(1..N)`, `S_i(1..N)`, sampled at `N` wavelength bands. When
pigments are mixed in concentrations `c_i`, the standard single-constant
Kubelka-Munk model says absorption and scattering combine **linearly**:

```
K_mix(b) = sum_i c_i * K_i(b)
S_mix(b) = sum_i c_i * S_i(b)
```

but the resulting reflectance at band `b` is a **nonlinear** function of
their ratio `r = K_mix(b) / S_mix(b)`:

```
R_mix(b) = 1 + r - sqrt(r^2 + 2r)
```

For each illuminant spectrum `L_j(1..N)` you convert a reflectance curve
`R(1..N)` to a tristimulus point using the given colour-matching functions
`xbar, ybar, zbar` (all length `N`), with the usual `Y = 100` white
normalization:

```
k_j     = 100 / sum_b L_j(b) * ybar(b)
X, Y, Z = k_j * sum_b L_j(b) * {xbar,ybar,zbar}(b) * R(b)
```

The tristimulus point is converted to CIE Lab (standard CIE76 `f`, using
`Xn, Yn=100, Zn` computed the same way from `L_j` and `R = 1`), and compared
to the target's Lab point under that same illuminant via Euclidean distance
`dE_j = sqrt(dL^2 + da^2 + db^2)`.

## Input (stdin)

```
N M K
xbar(1..N)
ybar(1..N)
zbar(1..N)
L_1(1..N)
...
L_K(1..N)
Rtarget(1..N)
cost_weight
K_1(1..N)
S_1(1..N)
...
K_M(1..N)
S_M(1..N)
cost_1 ... cost_M
```

`Rtarget` is the target reflectance spectrum (values in `(0,1)`). `K` is the
number of illuminants. `cost_weight` and `cost_i` scale the penalty for using
pigment `i` (see Scoring).

## Output (stdout)

`M` nonnegative numbers `c_1..c_M`, whitespace-separated, with
`sum(c_i) = 1` (tolerance `1e-3`). All values must be finite.

## Feasibility

- Exactly `M` tokens, each parses as a finite real number.
- No `c_i < -1e-6`.
- `sum(c_i)` within `1e-3` of `1`.

Any violation scores `0`.

## Objective (minimize)

```
F = sum_{j=1..K} dE_j  +  cost_weight * sum_{i: c_i > 1e-4} cost_i
```

Fewer, cheaper pigments help only through the second term; the first term
dominates and rewards a reflectance curve `R_mix` that stays close to
`Rtarget` **simultaneously under every given illuminant**. A recipe whose
reflectance shape only coincidentally matches the target's colour under one
particular illuminant (a *metameric* match) will generally show a large
`dE_j` under a differently-shaped illuminant, even if it looked perfect under
the first one.

## Scoring

The checker rebuilds `R_mix` from your `c_i` via the formulas above, computes
`F`, and compares it to its own internal baseline `B` (the uniform mixture
`c_i = 1/M` for all `i`), printing `Ratio: %.6f` for
`sc = min(1000, 100*B/F)/1000`. Lower `F` is better; `Ratio` is in `[0,1]`.

## Constraints

`N = 12`, `7 <= M <= 16`, `K = 3`, all spectral values in `[0, 4]`,
`Rtarget(b)` in `(0,1)`, `cost_weight` and `cost_i` in `[0.2, 3]`.

## Example (worked, illustrative shapes only — not part of any real test)

`N=2, M=2, K=1`: `xbar=[1.0,0.2], ybar=[0.2,1.0], zbar=[0.1,0.1]`,
`L_1=[1.0,1.0]`, `Rtarget=[0.5,0.5]`, `cost_weight=0.5`. Pigment 1:
`K=[0.3,0.1], S=[0.6,0.6], cost=1.0`; pigment 2: `K=[0.1,0.3], S=[0.6,0.6],
cost=1.0`.

Recipe `c=(0.5,0.5)`: `K_mix=[0.2,0.2]`, `S_mix=[0.6,0.6]`, ratio `1/3` on
both bands, so `R_mix ~= [0.4514, 0.4514]`. Converting both `R_mix` and
`Rtarget` under `L_1` gives Lab points `(72.97,0,0)` and `(76.07,0,0)`, so
`dE_1 ~= 3.10`. Both pigments are used, so the penalty is
`0.5*(1.0+1.0)=1.0`, giving `F ~= 4.10`.
