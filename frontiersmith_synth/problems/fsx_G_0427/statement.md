# Turbulence-Lab Invariant Group

## Problem
A turbulence laboratory logs six anonymized probe channels `x0 … x5` for each of
many experiments. Every experiment sits at a different Reynolds number, so each
channel individually sweeps across several orders of magnitude. Hidden in the
data, however, is an (approximate) **dimensionless invariant**: a power-law
combination of the channels

```
Pi = x0**a0 * x1**a1 * x2**a2 * x3**a3 * x4**a4 * x5**a5
```

that stays (nearly) constant across experiments. Your job is to recover the
integer exponent vector `a = (a0, a1, a2, a3, a4, a5)` that makes `Pi` as
invariant as possible.

Not every channel belongs in the invariant — some are distractors whose exponent
should be `0`. Two channels are coupled in the *training* regime in a way that
does **not** persist at higher Reynolds numbers, so a group tuned only to the
training rows can look invariant yet fall apart on the held-out regime.

## Input (stdin)
```
TESTID <k>
<ncols> <ntrain>
x0 x1 x2 x3 x4 x5
<row 1: ncols floats>
...
<row ntrain: ncols floats>
```
`ncols = 6` always. All channel values are strictly positive. `TESTID` is the
case id (difficulty index) only — it carries no information about the invariant.

## Output (stdout)
A single line of exactly **6 space-separated integers** — the exponent vector:
```
a0 a1 a2 a3 a4 a5
```

## Feasibility
* Exactly 6 integers, each with `|a_i| <= 6`.
* Not all zero (the vector `0 0 0 0 0 0` gives `Pi = 1`, which is not a group and
  scores 0).
* Non-integer, non-finite (`nan`/`inf`), out-of-range, or malformed output scores 0.

## Objective (maximize)
Your group is graded on a **held-out extrapolation split** drawn from a Reynolds
regime strictly **above** your training rows. Reward generalization, not
memorization. With per-channel-centered logs, define

```
proj_j = sum_i a_i * (log x_ij - mean_i)      over held-out experiments j
s(a)   = std_j(proj_j) / ||a||_2              (scale-free spread per unit exponent)
```

Smaller `s` means stronger cancellation, i.e. a more invariant group. Dividing by
`||a||_2` makes the score independent of rescaling `a -> c*a`. The invariance and
final objective are

```
M(a) = -log10(s(a) + 1e-12)                   (higher = more invariant)
F(a) = max(0, M(a) - 0.02 * sum_i |a_i|)      (mild parsimony penalty)
```

The score is `F(a)` normalized by the objective of a fixed reference group the
grader builds itself:

```
Ratio = min(1000, 100 * F(a) / F(reference)) / 1000
```

Irreducible measurement noise and a small residual finite-Reynolds drift keep even
the best real invariant well below `1.0`.

## Scoring
* `Ratio ~ 0.1` reproduces the reference group.
* Recovering the true multi-channel invariant pushes the ratio up, but the noise
  floor leaves headroom below `1.0`.
* Any infeasible output scores `0`.

## Constraints
* `ncols = 6`, `90 <= ntrain <= 235`, exponents in `[-6, 6]`.
* Deterministic grading; the held-out split is regenerated from `TESTID` alone.

## Example (illustrative FORM only — NOT the hidden law)
Suppose (hypothetically) the invariant were the ratio of the first two channels,
`Pi = x0 / x1`. Then a correct answer would be `1 -1 0 0 0 0`. This shape is shown
only to illustrate the output format; the actual invariant for these instances is
different and must be discovered from the data.
