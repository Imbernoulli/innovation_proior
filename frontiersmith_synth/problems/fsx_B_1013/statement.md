# Spirograph Portfolio for Annulus Coverage

## Problem

You are given an annulus (ring) with integer inner radius `r_in` and outer
radius `r_out`, subdivided into `K` equal-width concentric radial bins. You
must design a **portfolio of at most `Q` hypotrochoid ("spirograph") curves**
whose pooled sample points fill the annulus's radial bins as evenly as
possible.

Each curve `i` you output is four integers `R_i r_i d_i p_i`:

- a fixed-gear radius `R_i` and rolling-gear radius `r_i` with `2 <= r_i < R_i <= M`
- a pen offset `d_i` with `1 <= d_i <= r_i`
- a phase `p_i` with `0 <= p_i <= S-1`
- **containment**: the curve's whole radial band must lie inside the annulus:
  `r_in <= (R_i - r_i) - d_i` and `(R_i - r_i) + d_i <= r_out`

Every curve you output is sampled at `S` fixed gear-steps. Let
`g_i = gcd(R_i, r_i)` and `w_i = R_i / g_i` (the curve's rotational-symmetry
order, i.e. its petal count). For step `k = 0 .. S-1` define the exact
rational angle index `idx = (w_i*k + p_i) mod S` and the sampled radius:

```
rho = (R_i - r_i) + d_i * cos(2*pi*idx / S)
```

Every sampled `rho` of every curve must fall in `[r_in, r_out]` (guaranteed,
up to a `1e-6` tolerance, by the containment constraint above). Each sample
falls into radial bin `min(K-1, floor((rho - r_in) / binwidth))` where
`binwidth = (r_out - r_in) / K`. Pool the samples of **all** curves you
output (`q` curves, `1 <= q <= Q`, total `P = q*S` samples) into one
histogram `h[0..K-1]`.

## Objective

Maximize
```
F = 0.5 * coverage + 0.5 * entropy
```
where `coverage = (#bins with h[b] > 0) / K`, and `entropy` is the Shannon
entropy of the histogram normalized by `log(K)` (so `entropy = 1` iff samples
are spread perfectly evenly over all `K` bins, and `entropy = 0` iff they all
land in one bin).

The rotational-symmetry order `w_i` of a curve, and specifically how it
arithmetically interacts (via `gcd`) with the **fixed** sampling budget `S`,
controls how finely that curve's `S` samples sweep its own band: as `k`
ranges over `0..S-1`, the angle index `idx = (w_i*k + p_i) mod S` visits
every one of the `S` residues mod `S` iff `gcd(w_i, S) = 1`, giving a full,
fine sweep of angles (and hence radii, up to the harmless left-right
symmetry of `cos`) across the curve's whole band. If `gcd(w_i, S)` is large,
the visited indices collapse onto only `S / gcd(w_i, S)` distinct residues
(repeated `gcd(w_i,S)` times each), so the curve's `S` samples pile onto a
sparse handful of angles instead of sweeping the band, and its contribution
shrinks toward a thin sliver of bins. A curve with maximal *per-curve*
symmetry order (e.g. `R_i, r_i` coprime, so `w_i = R_i`) can still collapse
this way if `R_i` happens to share large factors with `S`. When a curve is
forced to collapse, its phase `p_i` decides *which* of the `S/gcd(w_i,S)`
surviving angles it lands on — a lever for making different curves'
collapsed points complement rather than duplicate each other.

## Feasibility

Any violated constraint (bad range, `r_i >= R_i`, a curve's band escaping
`[r_in, r_out]`, `q` out of `[1, Q]`, malformed/incomplete/non-numeric
output) scores `Ratio: 0.0`.

## Input (stdin)

One line: `r_in r_out K Q S M`.

## Output (stdout)

Line 1: `q` (number of curves used, `1 <= q <= Q`).
Then `q` lines, each `R_i r_i d_i p_i`.

## Example (illustrative toy numbers, not from the actual test set, and not necessarily near-optimal)

`r_in=10 r_out=50 K=8 Q=4 S=12 M=90`. Output
```
1
30 2 2 0
```
places one curve with band-center `28`, `d=2`: all `12` samples oscillate in
`[26,30]`, landing in only `1`-`2` of the `8` bins — weak coverage, low `F`.
A stronger portfolio spreads several curves' band-centers across `[10,50]`
**and** picks `R_i` so `w_i` is coprime to `S=12` (all `12` samples of that
curve then visit distinct angles, finely sweeping its whole local band)
while choosing phases so different curves' actually-visited bins do not
overlap.

## Scoring

The checker computes `F` for your submission as above, and also computes `F`
for its own naive internal reference portfolio `B` — a full-budget,
evenly-spaced-center portfolio with a minimal, fixed pen offset that ignores
the `R_i`/`S` arithmetic entirely. Your printed score is
`ratio = min(1.0, F / (10*B))`: matching the naive reference exactly gives
`ratio = 0.1`, and beating it by `10x` or more saturates at `ratio = 1.0`.

## Constraints

`10 <= r_in < r_out <= 500`, `4 <= K <= 200`, `3 <= Q <= 8`, `4 <= S <= 240`,
`r_out < M <= 700`. Time limit 5s, memory 512MB.
