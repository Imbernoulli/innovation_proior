# Nested Frames

A framer nests rectangular mats inside one another. Each nesting level `k`
has a **proportion** `p_k` (a normalised ratio in `(0,1)`, how much the
frame insets relative to its parent). Across one workshop's designs, the
proportions of EVERY nest obey the SAME hidden rule: deviations from a
harmonic fixed point `p*` evolve by a 2nd-order linear recurrence,

```
(p_{k+1} - p*) = alpha * (p_k - p*) + beta * (p_{k-1} - p*)
```

with hidden constants `alpha, beta, p*`. The recurrence's characteristic
roots `lambda1, lambda2` (its "spectrum") govern whether nests converge
toward `p*` or drift away from it as they get deeper. A design's **tension
score** is `T = (p_d - p*)^2` — how far the INNERMOST (deepest) frame's
proportion strays from the harmonic ratio.

You are shown SHALLOW designs (depth `3..6`): the full noisy proportion
trace plus a noisy tension reading. You are graded on DEEP designs (depth
`10..14`) you never see — there you get only the first two proportions
`p1, p2` and a target depth `d`, and must predict `T` from those alone.

## Input (stdin)

```
n t
d_1  p_1  p_2  ... p_{d_1}  T_1
d_2  p_1  p_2  ... p_{d_2}  T_2
...
```

`t` is the test id. `n` training rows follow, each: an integer depth
`d in [3,6]`, then `d` floats (the proportion trace, noisy), then one float
(the noisy tension reading). Rows are independent designs but share the
SAME hidden recurrence for this test id.

## Output (stdout)

Print **one line**: a single arithmetic Python expression string over the
variables `p1`, `p2`, `d` using only `+ - * / **`, parentheses, and numeric
literals (no function calls, no other names). This is your predictor of `T`
for a graded design given its first two proportions and target depth.

**Illustrative FORM only — NOT the hidden law:**

```
0.7 * (p1 - 0.5) ** 2 + 0.02 * d
```

This just shows valid syntax; the real law is a different shape entirely
and you must discover it from the training traces.

## Feasibility

The expression must parse under the grammar above (arithmetic + `p1,p2,d`
only, finite numeric literals). It must evaluate to a finite number for
every graded design. Any violation scores `0`.

## Objective (minimise)

Let `F` be the mean squared error of your expression's predictions against
the graded tension readings of the deep (depth `10..14`) designs — each
reading is the true tension plus a little irreducible measurement noise
(see below). The checker also forms `B`, the mean squared error of
predicting the constant training-mean tension for every graded design (an
internal baseline). Score:

```
Ratio = min(1000, 100 * B / F) / 1000
```

A constant predictor reproduces `B` (Ratio ≈ 0.1). Lower held-out error
raises the score. Even a fully correct recurrence leaves a graded reading
that includes irreducible measurement noise, so the ceiling stays open.

## Why the training window is a trap

Over depth `3..6` the recurrence's effect is barely visible: a curve that
merely tracks how tension trends with depth (ignoring that each design's
OWN `p1,p2` fix its personal mixture of the two eigen-modes) fits the
shallow window just fine. But by depth `10..14` that mixture's dominant
mode has compounded through many more multiplications by `lambda1`, so any
model that isn't built on the actual recurrence — rather than a curve fit
to depth alone — diverges from the true tension, often by orders of
magnitude once the dominant root exceeds `1`.

## Constraints

Time limit 5 s, memory 512 MB. `n` is a few hundred rows. All scoring is
deterministic — the hidden recurrence, the graded designs, and the grading
noise are all fixed functions of the test id.
