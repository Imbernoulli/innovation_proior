# Shared-Basis Signal Probes

A laboratory instrument can take only **m linear measurements** of an unknown
vector `x` in `R^n` before it is discarded. The catch: `x` may come from any of
**K different signal families**. Family `k` consists of vectors that are
**s-sparse** in a known basis `B_k` (an `n x d_k` matrix with orthonormal
columns): every signal of family `k` is `x = B_k a` where `a` has at most `s`
nonzero entries. The bases are related — they were manufactured to share a
common low-dimensional subspace — but how much they overlap varies from
instance to instance. You are given the bases; you must design the measurement
apparatus once, for all families at once.

Design **m probe vectors** `p_1..p_m` (rows of a matrix `P`). Each probe has
integer entries with `|entry| <= pmax`. When the instrument later reads a
signal `x`, it produces `y = P x + eta`, where `eta` is Gaussian noise with
standard deviation `sigma = 0.2` per component.

## Input (stdin)

Line 1: `n m K s pmax`.
Then, for each family `k = 1..K`: a line with `d_k`, followed by `n` lines of
`d_k` floating-point numbers — the matrix `B_k`, row by row.

## Output (stdout)

Exactly `m` lines of `n` integers each: the rows of your probe matrix `P`.
Every entry must satisfy `|entry| <= pmax`. Any other output is infeasible
(score 0).

## Feasibility

- exactly `m` rows, each with exactly `n` integer-valued tokens;
- all entries finite, integer, and bounded by `pmax` in absolute value.

## Objective (minimize)

The checker evaluates your `P` as follows. For each family `k` it draws `T = 24`
hidden test signals (fixed seed; `s` nonzero coefficients with magnitudes in
`[0.5, 1.5]` and random signs), measures `y = P x + eta`, and reconstructs `x`
with a **fixed decoder** you cannot change: `s` iterations of Orthogonal
Matching Pursuit on the dictionary `P B_k` (column-norm-normalized correlation,
least-squares refit after each pick). The per-signal error is the relative
error `||x - x_hat|| / ||x||`. Let `E_k` be the mean error over the `T` signals
of family `k`. The objective is

```
F = max over k of E_k        (minimize the worst family)
```

## Scoring

The checker internally builds a naive baseline apparatus `P0` (probe `j`
measures coordinate `j mod n` at amplitude `pmax`) and computes its objective
`B`. Your ratio is

```
Ratio = min(1, 0.065 * B / F)
```

so the naive baseline scores 0.065, and reducing the worst-family error by a
factor of ~15 below baseline caps the score at 1.0. Scores are bit-for-bit
deterministic: the test signals, noise, and decoder are fixed.

## Constraints

- 64 <= n <= 128, m = 40, 3 <= K <= 5, 5 <= s <= 10, pmax = 7.
- 10 <= d_k <= 26; the bases are orthonormal-column matrices with a planted
  shared subspace (its dimension varies per instance).
- Time limit 5 s, memory 512 MB.

## Example (illustrative I/O form only)

Input excerpt (n=4 toy, not a real test):

```
4 2 1 1 7
3
0.5000000000 0.0000000000 0.0000000000
0.5000000000 ...
...
```

Output (m=2 rows of n=4 integers):

```
7 0 0 0
0 7 0 0
```

On the real tests, this naive coordinate-probe construction is exactly the
checker's baseline and scores `Ratio = 0.065000`; a random ±pmax matrix does
somewhat better on some tests and far worse than a structure-aware design on
others.
