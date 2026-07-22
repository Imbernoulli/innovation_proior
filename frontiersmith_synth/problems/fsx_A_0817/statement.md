# The Flat Direction: Discovering an Affine Symmetry Generator

## Problem
A field notebook records a real-valued **label** at scattered points of the plane. The
label is produced by an unknown smooth function of `(x, y)`, but that function is not
arbitrary: it is **invariant under the flow of a hidden one-parameter transformation
group** of the plane. Concretely, there is a hidden 2x2 matrix `A` and vector `t` such
that moving any point along the ODE `dp/ds = A*p + t` never changes the label. Three
familiar members of this "affine generator" family are: a pure **rotation** about some
center (`A` skew-symmetric, `t` fixed so the flow's fixed point is that center), a pure
**uniform scaling** about some center (`A` a multiple of the identity), and a pure
**translation** along a fixed direction (`A = 0`). You are not told which — you must
recover it from data sampled only in a small region of the plane, then certify that it
keeps holding far outside that region.

## Input (stdin)
```
testId
G eps
x_1 y_1 label_1        # anchor point of group 1
x_2 y_2 label_2        # anchor + eps in x
x_3 y_3 label_3        # anchor - eps in x
x_4 y_4 label_4        # anchor + eps in y
x_5 y_5 label_5        # anchor - eps in y
... (repeated for all G groups, 5 lines each, always in this order)
```
`G` groups of 5 points are given (a small local finite-difference stencil around each
of `G` anchor points), all sampled from one bounded "training slab" of the plane.

## Output (stdout)
Exactly six numbers `a11 a12 a21 a22 t1 t2`, declaring your generator
`A = [[a11,a12],[a21,a22]]`, `t = (t1,t2)`. (Illustrative **form only — not the hidden
law**: a `dp/ds = p^2 + sin(p)`-style nonlinear wobble could never be declared this way
— this is only to show the output is six bare numbers, nothing else.)

## Feasibility
The output must be exactly 6 whitespace-separated finite numbers, each of magnitude
`<= 1e4`, not all (numerically) zero. Any violation scores `Ratio: 0.0`.

## Scoring
Your generator is rescaled to unit norm first (only its *direction* in the 6
coefficients matters — magnitude is neither rewarded nor punished). The checker then:
1. **Invariance test**: picks several points well outside the training slab, but on
   the *same* hidden orbit as training data (genuinely far away in the plane, not in
   the range the label was fit on), and applies your generator's finite flow `g_s` to
   each, at several fixed step sizes `s` (positive and negative). It compares the true
   label before and after: agreement decays smoothly with the discrepancy.
2. **Label-propagation test**: for the same far-away points, it searches your
   generator's flow for the step size that carries the point closest to some training
   point, copies that training label as the prediction, and checks it against the true
   far-away label.
3. A mild parsimony penalty applies if your 6 numbers do not reduce to a small number
   of significant components (a clean, minimal generator beats a kitchen-sink guess).
Let `F` be the resulting quality score and `B` the same pipeline's score for the
checker's own trivial construction (`a11=a12=a21=a22=0, t1=1, t2=0`, i.e. "guess
translation along +x and hope"):
```
sc = min(1000.0, 100.0 * F / max(1e-9, B))
Ratio = sc / 1000.0
```
Matching the baseline scores `0.1`; a much better-aligned generator caps at `1.0`.
Irreducible finite-difference/model error keeps even a correct discovery below `1.0`.

## Why the obvious fit is a trap
Pooling every point into one global linear regression and taking its flat direction
assumes translation invariance everywhere — exactly right if the truth is a
translation, but a rotation's or scaling's label has **no** globally flat linear
direction, only a locally-tangent one near the slab that drifts badly once you flow far
along it. The data is deliberately handed to you as small local stencils so the *local
gradient* at each anchor is directly readable; the insight is to use those gradients
geometrically (radial vs. tangential) to solve for a hidden center, not to fit the
label surface at all.

## Constraints
- `6 <= G <= 14`; `eps = 0.04` fixed. Time limit 5s, memory 512MB.
- Coordinates and labels are bounded floats; each `.in` file is well under 1MB.
