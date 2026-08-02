# Silent Reflections: Indexing a Powder Pattern from Its Absences

## Problem
A powder-diffraction experiment on Cu K-alpha1 radiation (wavelength
`lambda = 1.5406` angstrom, fixed) is run on a crystal with an **orthorhombic**
lattice: three mutually perpendicular axes of length `a, b, c`, each a
DISTINCT INTEGER number of angstrom in `[4, 13]` (hidden from you, disclosed
only as this range -- so the challenge is which integers and which centering,
not an open-ended continuous fit).
Bragg's law places a reflection at scattering angle `2*theta` for every
integer triple `(h,k,l)` (not all zero, taken with `h,k,l >= 0` since sign
does not change the spacing) with

```
sin(theta) = (lambda/2) * sqrt( (h/a)^2 + (k/b)^2 + (l/c)^2 )
```

(only triples with the right-hand side `<= 1` are physically observable).
The crystal also has a hidden **Bravais centering** `P`, `I`, or `F`, which
forbids some triples outright (a *systematic absence*): `I` requires
`h+k+l` even, `F` requires `h,k,l` all-even or all-odd, `P` forbids nothing
(both rules are symmetric in `h,k,l`, so which axis you call `a`, `b`, `c`
is your free choice). Reflections closer than `0.05` degrees merge into one
observed line.

You are given the observed lines only up to a moderate angle
`theta2_given_max`. Recover the lattice constants, the centering, and an
`(h,k,l)` index for every given line -- **and** get the diffraction pattern
right out to a larger angle `theta2_full_max`, disclosed to you but whose
peaks are not. A wrong-but-plausible indexing (e.g. assuming primitive `P`
when the truth is more restrictive) can still fit every given line, because
`P` never forbids anything -- it is only refuted by the lines it wrongly
predicts, or misses, beyond what you were shown. (Illustrative FORM only,
not the hidden law: a purely cubic `a=b=c` guess is a different, unrelated
kind of mistake from a centering mistake.)

## Input (stdin)
```
testId
lambda
theta2_given_max theta2_full_max
M
q_1
q_2
...
q_M
```
`q_i` are the `M` observed `2*theta` values (degrees), strictly ascending.

## Output (stdout)
```
a b c
centering
h_1 k_1 l_1
...
h_M k_M l_M
```
`a,b,c` are positive reals; `centering` is one of `P I F`; the `M` lines
give your `(h,k,l)` index (non-negative integers, not all zero) for
`q_1 .. q_M` in order.

## Feasibility
Invalid (scores `Ratio: 0.0`) if: `a,b,c` are not finite positive numbers
`<= 40`; `centering` is not one of `P/I/F`; there are not exactly `M`
index lines; any `h,k,l` is not a non-negative integer `<= 2000`, or is
`(0,0,0)`; or any submitted `(h,k,l)` violates your own declared
`centering`'s reflection condition.

## Scoring
A feasible output is scored as a weighted sum `F` of four terms (each in
`[0,1]`, higher is better):
1. **lattice accuracy** -- relative error of `a,b,c` against the hidden
   truth, best of the 6 axis permutations (labeling of the 3 axes is a free
   convention);
2. **self-consistency** -- how closely your `a,b,c,(h,k,l)` reproduce, via
   Bragg's law, the `q_i` you claim to explain;
3. **indexing correctness** -- fraction of your `M` indices that match a
   truly-correct `(h,k,l)` for that line;
4. **extrapolation fit** -- the F1 score between the diffraction pattern
   your `a,b,c,centering` forward-predicts out to `theta2_full_max` and the
   hidden true pattern out to that same angle.

Term 4 dominates the weighted sum, so getting the given lines to fit
(terms 1-3) is necessary but far from sufficient; the centering you declare
controls which of your own predicted lines beyond `theta2_given_max` are
kept or pruned, and that is what mainly decides your score. The final
`Ratio` is a fixed rescaling of `F` (`Ratio = min(1, 0.9*F)`) so a perfect
`F` still leaves headroom above `1.0`.

## Constraints
`3 <= M <= 30`; hidden truth `a,b,c` are distinct integers in `[4,13]`
angstrom; `30 <= theta2_given_max < theta2_full_max <= 100` (degrees). Your
output `a,b,c` need not be integers. Time limit 5s, memory 512MB.

## Example
If `M=3` with `q = (20.0, 28.5, 35.0)` and you output `a b c = 6.0 6.0 6.0`,
`centering = P`, indices `(1,0,0) (1,1,0) (1,1,1)` -- this is a purely
illustrative FORM of a valid-shaped answer, not a worked score; the real
grader recomputes everything against the hidden crystal for the actual test
input.
