# Thin-Film Coating: Hit a Reflectance Spectrum

## Problem
Light in air (index `n0 = 1`) strikes a flat substrate of index `ns`. You deposit an
ordered stack of at most `L` thin dielectric films between air and substrate. Film `j` is
one of `M` available materials (real index `n[mi]`) with a physical thickness `d` (in nm).
At normal incidence the whole stack has a measured reflectance `R(lambda)` for each of `K`
probe wavelengths. You are given a **target reflectance curve** `Rstar(lambda)` and must
build a stack whose reflectance matches it as closely as possible while using few layers.

## Input (stdin)
```
n0 ns
M
n[0] n[1] ... n[M-1]
K
lambda_1 Rstar_1
...
lambda_K Rstar_K
L lambda0 cost dmax
```
`lambda0` is a reference design wavelength (nm); `cost` is the per-layer penalty; `dmax` is
the maximum allowed thickness (nm).

## Output (stdout)
First line: integer `m` with `0 <= m <= L`, the number of films (listed from the air side
toward the substrate). Then `m` lines, each `mi d`: material index `mi` (in `0..M-1`) and
thickness `d` (in nm, `0 <= d <= dmax`).

## Reflectance model (deterministic)
For a wavelength `lambda`, each film contributes the normal-incidence characteristic matrix
```
delta = 2*pi*n*d / lambda
Mj = [[cos delta,      i*sin delta / n],
      [i*n*sin delta,  cos delta     ]]
```
The stack matrix is the ordered product `Mstack = M1 * M2 * ... * Mm` (air-side film first).
With `[B, C]^T = Mstack * [1, ns]^T`, the input admittance is `Y = C/B` and
```
r = (n0 - Y) / (n0 + Y),   R(lambda) = |r|^2   (clamped to [0,1]).
```
An empty stack gives the bare substrate `R = ((n0-ns)/(n0+ns))^2`.

## Feasibility
All numbers finite; `0 <= m <= L`; each `mi` an integer in `0..M-1`; each `0 <= d <= dmax`.
Any violation scores `Ratio: 0.0`.

## Objective (minimize)
```
SSE = sum over the K wavelengths of (R(lambda) - Rstar(lambda))^2
F   = SSE + cost * m
```
Lower `F` is better. The `cost * m` term rewards parsimony: an extra film must earn its
spectral improvement.

## Scoring
The checker builds its own baseline `B` = the bare-substrate SSE (zero films). With your
feasible objective `F`,
```
sc    = min(1000, 100 * B / max(1e-9, F))
Ratio = sc / 1000
```
so doing nothing scores about `0.1`, and driving `F` ten times below the bare baseline caps
at `1.0`.

## Why it is hard
Reflectance is a nonlinear function of the whole matrix product, and every film couples all
`K` wavelengths at once. Nulling the error at one wavelength by dropping in a quarter-wave
film (`d = lambda/(4n)`) generally worsens the others, so patching the worst wavelength one
film at a time thrashes. A hint: at `lambda0` a quarter-wave film of index `n` transforms
admittance as `Y -> n^2 / Y`, so composing quarter-wave films acts multiplicatively on
admittance (additively on its logarithm) -- reasoning there lets you steer the whole band at
once. Each target curve was produced by a hidden stack of MORE than `L` films, so exact
reproduction is impossible; find the best `L`-film approximation.

## Constraints
`M = 3`, `13 <= K <= 15`, `3 <= L <= 4`, film indices in `[1.38, 2.35]`, substrate index
`ns` in `[1.9, 2.3]`, `dmax = 1200` nm. Runs well under the time limit.

## Example
Suppose `ns = 1.90`; the bare substrate reflects `((1-1.90)/2.90)^2 = 0.0963` at every
wavelength, so the empty stack scores `Ratio = 0.1`. A single quarter-wave film centred on
`lambda0` drives reflectance toward the target near `lambda0` but drifts away at the band
edges, lowering `F` only modestly. A multi-film design whose admittance tracks the whole
target curve (while paying its `cost * m`) lowers `F` much further and scores higher.
