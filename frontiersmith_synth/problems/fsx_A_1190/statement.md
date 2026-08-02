# No Pure Pixels: Inflating to the Enclosing Material Simplex

## Problem
A sensor records `N` mixed spectra over `R` wavelength bands. Each observed pixel `j` is a
**linear mixture** of `K` unknown pure-material spectra ("endmembers") `m_1,...,m_K in R^R`:

```
y_j = sum_{k=1..K} a_kj * m_k + noise_j
```

where the abundance vector `a_j = (a_1j,...,a_Kj)` is **nonnegative and sums to 1** -- a
point in the standard `(K-1)`-simplex whose corners are the endmembers themselves. You are
given only the `N` mixed spectra `y_j` and `K`; you must recover BOTH the endmember spectra
and every pixel's abundances.

The catch: in this benchmark **no pixel is ever a pure material** (every `a_kj` is bounded
away from `1`). The true endmembers therefore lie strictly **outside** the convex hull of
the data you can see. Any method that estimates an endmember by picking (or nearly
reproducing) an extreme *observed* pixel is geometrically stuck short of the truth, no
matter how much data it has -- the gap does not shrink with `N`. What such methods CAN still
see is that pixels near each **edge** of the true simplex (mixtures of only two materials)
remain densely sampled even though no single pixel is a full vertex.

## Input (stdin)
```
t
R K N
y_1  (R floats)
y_2  (R floats)
...
y_N  (R floats)
```
`t` is the test id (uninterpreted by you; ignore it or not, it changes nothing about how to
solve the instance). `R` bands, `K` endmembers, `N` pixels follow, then the `N` mixed
spectra, one per line.

## Output (stdout)
```
m_1  (R floats)          <- endmember 1's spectrum
...
m_K  (R floats)          <- endmember K's spectrum
a_1  (K floats)          <- abundance vector for pixel 1 (same order as the input pixels)
...
a_N  (K floats)          <- abundance vector for pixel N
```

## Feasibility
An output is valid iff **all** hold:
- all `K*R + N*K` values are finite;
- every endmember entry is `>= 0` (tolerance `1e-6`) and `<= 50`;
- every abundance entry lies in `[0,1]` (tolerance `1e-6`);
- each pixel's `K` abundances sum to `1` within `1e-3`;
- **reconstruction fidelity**: for every pixel, `sum_k a_kj * m_k` must be close to the
  given `y_j` -- the mean relative L2 reconstruction error over all pixels must be `<= 0.35`.
Any violation scores `Ratio: 0.0`.

## Objective
The checker matches your `K` submitted endmembers to the hidden true endmembers under the
best permutation (minimizing total relative error), then scores:
- **endmember accuracy**: for each matched pair, `exp(-relative_L2_error / 0.15)`, averaged
  over the `K` endmembers;
- **abundance accuracy**: for each pixel, `exp(-L1_error_against_hidden_truth / 0.25)`,
  averaged over all `N` pixels (using the same endmember permutation to align coordinates).

`F = 0.5 * endmember_accuracy + 0.5 * abundance_accuracy`, and `Ratio = min(1.0, F)`.
Both terms reward the SHAPE of your recovered simplex, not just fitting the observed data
(which the feasibility gate already forces) -- getting the endmembers' actual positions
right, including how far past the data cloud they must be inflated, is what raises the
score.

## Constraints
- `R = 8` bands, `K = 3` endmembers, `32 <= N <= 104` pixels.
- Every test case caps each pixel's largest abundance component strictly below `1` (no pure
  pixels anywhere); several cases use a tight cap so the naive "trust the most extreme
  observed pixel" estimate is measurably, structurally short of the truth.
- Time limit 5s, memory 512MB.

## Example
With `K=3`, if the three true material spectra are well separated and your submission's
matched endmembers are exact (`relative_L2_error = 0`) and every abundance is exact
(`L1_error = 0`), then `endmember_accuracy = abundance_accuracy = 1.0`, `F = 1.0`, and
`Ratio = 1.0`. Declaring all three endmembers equal to the mean pixel spectrum with uniform
abundances `(1/3,1/3,1/3)` is feasible but scores far lower, since it captures none of the
simplex's shape.
