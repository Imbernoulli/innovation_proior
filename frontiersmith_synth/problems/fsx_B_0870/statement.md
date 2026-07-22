# Below the Tipping Point — recovering a hidden percolation scaling law

A lab studies bond percolation on a large but finite random-graph family. As the
bond-occupation probability `p` rises, the network's giant-component fraction
`S(p)` (a value that should sit in/near `[0,1]`; see Constraints) undergoes a
phase transition at a **hidden** critical probability `p_c`. The finite-size
scaling ansatz for this family is **known** exactly:

```
z(p) = (p - p_c) / W
g(z) = 0.5 * ( z + sqrt(z*z + 4) )          # smooth hinge, g(z) > 0 everywhere
S(p) = clip( A * (W * g(z))^beta , 0, 1 )
```

`W` (the crossover width) is **known**, printed in the header below. `p_c`
(the critical probability), `beta` (the critical exponent) and `A` (the
amplitude) are **hidden** and must be recovered from data. As `z -> +inf`,
`W*g(z) -> (p-p_c)`, so `S -> A*(p-p_c)^beta` — the textbook near-critical
scaling law. As `z -> -inf`, `S` decays smoothly toward 0 (finite-size
rounding below the transition). Your job is to fit `p_c`, `beta`, `A` well
enough to predict `S` **through and beyond** the transition, from data taken
only **below** it.

## Input (stdin)

```
t  N  W
p0 S0
p1 S1
...
p(N-1) S(N-1)
```

`t` is the test id; `W` is the known crossover width. Then `N` census rows
follow: bond probability `p_i` and the measured giant-component fraction `S_i`,
both **strictly sub-critical** (`p_i < p_c` for the hidden `p_c`), spanning from
deep sub-critical up to just below the transition, with small measurement noise
(occasionally slightly negative, since a near-zero true value plus noise can dip
below 0). The held-out grading points span the near-critical crossover (a little
below and at the transition) through the fully super-critical regime — a region
strictly beyond the training band and **not** given to you.

## Output (stdout): a one-shot law

Emit a single closed-form expression for `S` as a function of `p`. Use numeric
constants, `+ - * /`, unary `+/-`, and the functions `absv(a)`, `minv(a,b)`,
`maxv(a,b)`, `powv(a,b)` (power; the base is clamped to `>=0` before raising, so
a negative base never crashes the evaluator — but a zero base with a negative or
non-integer exponent is still mathematically undefined and, like any other
non-finite result, scores `0`), over the single variable `p`.

**Illustrative FORM only — NOT the hidden law:**

```
maxv(0.0, 0.4*p - 0.1) + 0.02*absv(p - 0.5)
```

This just shows the syntax; the real law's threshold, exponent, and amplitude are
different and must be discovered from the census.

## Feasibility

The expression must parse under the grammar above (known names only, finite
constants, at most 200 nodes). Any parse/known-name violation, or any non-finite
value produced while evaluating your law on the held-out points, scores `0`.

## Objective (minimise)

Let `MSE` be the mean squared error of your law's predictions against the true
(noisy) held-out targets, and `nodes` the number of expression nodes in your law.
The grader forms

```
F = MSE       * (1 + LAMBDA * nodes)
B = MSE_flat  * (1 + LAMBDA * 1)     # baseline: constant = mean(train S)
Ratio = min(1000, 100 * B / F) / 1000
```

with a small fixed `LAMBDA`. The flat baseline reproduces `B` (Ratio ≈ 0.1);
lowering held-out error raises the score, while a parsimony tax discourages
needlessly large laws. Held-out measurement noise is substantial and irreducible,
so even a very good law stays well below the ceiling — there is room to improve.
Report the highest Ratio you can.

## Why the sub-critical census is a trap

Over the observed range, `S(p)` merely looks like a slowly rising, mildly convex
curve — a flexible smooth fit (polynomial, spline, whatever) tracks it beautifully
and a practitioner is tempted to stop there, treating the given ansatz as
optional. But a smooth curve fitted only to sub-critical data — with no
reference to `p_c`, `beta`, `A` — has **no threshold and no singular exponent
baked in**: pushed past the transition it either flattens out (if it saturates)
or runs away in the wrong shape (if it does not); it has no mechanism to
reproduce the true `(p-p_c)^beta` acceleration. Recovering the singularity means
actually **fitting the given scaling form** — searching for the `p_c`, `beta`,
`A` that make the sub-critical data collapse onto the known crossover kernel —
not merely interpolating the data's smooth appearance with an unrelated curve.

## Constraints

Time limit 5 s, memory 512 MB. `N` is a few hundred rows; `p` and the true
`S` lie in `[0,1]` (measured `S_i` may stray a little outside due to noise);
scoring is fully deterministic.
