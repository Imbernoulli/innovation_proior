# Hearing the Shape of Pans

## Problem

Mark Kac asked whether you can hear the shape of a drum from its resonance
spectrum. Here you get the reverse task: you are handed *counting* data for
a family of rectangular baking pans and must write down a **symbolic law**
that predicts how many vibrational modes a pan has below a given spectral
threshold — accurately enough to extrapolate to a pan shape you never saw.

Each pan is an axis-aligned rectangle with side lengths `a`, `b` (positive
reals). For a spectral parameter `lam` (also positive), there is an
idealized, continuum-smoothed mode-count `N(a, b, lam)`. `gen.py` gives you
noisy observations of `N` for a batch of **compact** pans — side ratios
`a/b` between 1 and 3 (roughly square baking trays), sampled at a range of
`lam`. Your job is to write a **closed-form expression** for `N` in terms of
`a`, `b`, `lam` that also works for pans you never observed: long, narrow
**baking sheets** with side ratio 10–30. The physics of such counting laws
is well known to have (at least) two competing contributions with different
growth rates in `lam` — a bulk/area-like term and a boundary/perimeter-like
correction — plus a small constant offset. Which contribution actually
carries the signal depends on the regime; the training pans and the baking
sheets are **not** in the same regime.

## Input (stdin)

Line 1: integer `K`, the number of training rows.
Next `K` lines: four floats `a b lam N` — one noisy observation each.
(No further structure or hidden values are printed; the coefficients of the
true law are never revealed anywhere in this data.)

## Output (stdout)

Exactly one line: a single Python-style arithmetic expression using only
the variables `a`, `b`, `lam`, the constants `pi`, `e`, and the functions
`sqrt`, `log`, `exp`, `abs`, `min`, `max`, combined with `+ - * / ** %` and
parentheses. Example of the *syntax* only (not the physics — illustrative
FORM only, not the hidden law): `2 * a + log(b) / (lam ** 0.3) - e`.

## Feasibility

The output must be exactly one non-empty line, at most 2000 characters,
parsing to a valid expression over only the allowed names/functions/
operators above. Evaluating it on every held-out row must produce a finite
number (no `nan`/`inf`, no exceptions). Any violation scores `Ratio: 0.0`.

## Objective

The checker regenerates a **held-out extrapolation split**: baking-sheet
pans with side ratio 10–30, at moderate `lam`, computed from a fixed law
that is never shown to you. Let `F` = (normalized RMS error of your
expression against the true held-out values) + a small complexity penalty
for expressions with more than 40 syntax-tree nodes + a fixed small
irreducible-uncertainty floor (the same floor is added to the checker's own
baseline below, so it does not change relative standing — it only keeps a
single lucky test case from saturating the score). **Minimize `F`.**

## Scoring

The checker also computes its own internal baseline `B`: a fixed,
uncalibrated single-term guess with no perimeter dependence and no fitting
at all. Score `= min(1000, 100 * B / max(1e-9, F)) / 1000`, printed as
`Ratio: <value>`. Matching the baseline scores ≈0.1; a materially better
extrapolation scores higher, capped below 1.0 so there is always headroom
above the reference solutions.

## Constraints

`1 <= K <= 6000`. `0.1 <= a, b <= 2000` (held-out baking sheets can be as
narrow as `b ~ 0.4`). `1 <= lam <= 6000`. `N` values fit in a double. Time
limit 5s, memory 512MB.

## Example (worked score, illustrative FORM only)

Suppose two held-out rows have true values `N = 10.0` and `N = 40.0`, and a
submitted expression evaluates to `12.0` and `36.0` there. Squared errors:
`4` and `16`, mean `10`, RMS `≈3.162`. True-value scale: RMS of `10,40`
`≈29.15`. So the (illustrative) normalized error is `≈3.162/29.15 ≈ 0.1085`.
If the checker's own baseline had a normalized error of `0.20` on this toy
pair, the score would be `min(1000, 100*0.20/0.1085)/1000 ≈ 0.184`. The
real checker uses many more held-out rows, the real `N`, and the real
baseline — this is only to illustrate the arithmetic of the formula.
