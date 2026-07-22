# Temple Bell Ledger: Two Laws Wearing One Mask

## Problem
A temple keeps a ledger of bell-strike counts, one line per ceremony:
`a(1), a(2), a(3), ...`. The *true* count follows a hidden rule, but the rule is
not a single law. For every line `n >= 3` the clean count obeys **one of a few
short linear recurrences**, and which one is used is chosen by `n mod m` for a
**hidden small modulus `m`** (the ceremony cycle rotates which bell leads):

```
clean(n) = A[n mod m] * clean(n-1) + B[n mod m] * clean(n-2) + C[n mod m]
```

The coefficients `A[r], B[r], C[r]` and the modulus `m` are hidden. On any finite
window this mixture looks like one long, messy law; only restricting the ledger to
a single residue class `n == r (mod m)` reveals the clean short system behind that
class.

Worse, the scribe is tired: a **small fraction of the written lines are corrupted**
— replaced by a number far from the true count. The corruption does not disturb
the underlying bell process (later clean counts are still computed from the true
values), so a corrupted line is pure observation noise that must be *ignored*, not
fitted.

You are given the first `N` written lines. Predict the next `K` written lines.
Those future lines **continue to be corrupted at the same rate**, so a few of them
are inherently unpredictable — recovering the law perfectly still cannot guess
which future lines the scribe will botch.

## Input (stdin)
```
N K t
a(1)
a(2)
...
a(N)
```
`t` is the test id. Then `N` observed (possibly corrupted) integer counts, one per
line.

## Output (stdout)
Exactly `K` integers, one per line: your predictions for the written lines
`a(N+1), a(N+2), ..., a(N+K)`.

## Feasibility
The output must be **exactly `K` integers** (each a finite base-10 integer, no
decimals / `nan` / `inf`). Any other token count, or any non-integer token, scores
`Ratio: 0.0`.

## Objective (minimise)
The grader knows the hidden clean counts `clean(N+i)` and the actually-written
`obs(N+i)`. Per line it charges a clamped relative error against the **written**
ledger, normalised by the clean magnitude:
```
loss_i = min(1, |pred_i - obs(N+i)| / (1 + |clean(N+i)|))
F      = sum over i=1..K of loss_i           (your total error, minimise)
```
A corrupted future line is far from `clean`, so its `loss_i` is `1` no matter what
you predict — that is the irreducible slack that keeps the score below the ceiling.

## Scoring
The grader builds its own trivial baseline `B` = the error `F` of the
**"predict 0"** ledger, then
```
sc    = min(1000, 100 * B / max(1e-9, F))
Ratio = sc / 1000
```
so "predict 0" scores about `0.1`, driving `F` down raises the score, and a perfect
law-recovery is capped below `1.0` by the corrupted held-out lines.

## Why the obvious approach is a trap
Fitting **one** linear recurrence to the whole window (Berlekamp–Massey or a single
least-squares recurrence) absorbs the residue-class switching into a long fake law
that fits the finite window but derails within a few extrapolated steps; and being
non-robust, it is dragged off by the corrupted lines. The clean move is to test
**residue-class restrictions**: split the lines by `n mod m`, recover each class's
short law while *rejecting* outliers, and prefer the smallest, simplest system that
explains the data.

## Constraints
`N` up to ~170, `K = 50`, `2 <= m <= 3`. Time limit 5 s, memory 512 MB. Scoring is
fully deterministic.

## Example (illustrative FORM only — NOT the hidden law)
If a toy ledger obeyed the single rule `a(n) = a(n-1) + a(n-2)` (Fibonacci) with no
switching and no corruption, then from `..., 5, 8` you would emit `13, 21, 34, ...`.
The real ledger switches law by `n mod m` and contains corrupted lines; you must
discover its shape from the data.
