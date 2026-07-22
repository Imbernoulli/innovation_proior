# Coffee Percolating Through Sieves of Many Sizes

A sieve of mesh-count `L` is a finite lattice of tiny gaps. You pour coffee
grounds packed at density `p` (roughly in `[0,1]`) onto the sieve, repeat many
times with fresh random packings, and record the **fraction of pours that
find a clear path through the sieve**, `Pi_hat(p, L)`. You have a logbook of
readings from several **small** sieves. Predict the crossing fraction for
**much larger sieves** that never appear in your logbook.

Even a fully-packed sieve occasionally clumps and blocks every path, and a
nearly-empty one sometimes lets a trace through by capillarity, so readings
never quite touch 0 or 1: the true crossing probability has the fixed shape
`Pi_true(p, L) = 0.1 + 0.8 * sigmoid(z)` for some quantity `z` depending on
`p` and `L`. What is **hidden** is exactly how `z` depends on them: as the
sieve grows, the crossing curve both (a) shifts its midpoint toward a
critical packing density `pc`, and (b) sharpens from a gentle ramp into a
near-vertical step. These two effects are not independent — both are
governed by **one** exponent `nu`:

```
z = (p - pc) * L ** (1 / nu)
```

`pc` and `nu` are fixed for a given sieve family but unknown to you, and
**different for every test case**. (Illustrative FORM only — NOT the real
shape: `z = 3*(p - 0.5) + 0.02*L` is the kind of ad hoc formula this problem
is NOT about; the real relationship couples `p` and `L` through a single
power `L**(1/nu)`, not a linear combination.)

## Input (stdin)

```
n t
L[0]  p[0]  Pi_hat[0]
L[1]  p[1]  Pi_hat[1]
...
```

`t` is the test id. `n` logbook rows follow: sieve size `L` (integer, one of
`8, 16, 24, 32`), packing density `p` (float), and the observed crossing
fraction `Pi_hat` (float in `[0,1]`, noisy). Grading uses sieves of size
`L = 128` and `L = 512` — **never present in the logbook**.

## Output (stdout): one closed-form expression

Print a single line: an arithmetic expression over the variables `p` and
`L`, numeric constants, the binary operators `+ - * /`, parentheses, the
unary functions `sig` (logistic), `tanh`, `absv`, `sqrt` (of `|x|`), and the
one two-argument function `pw(base, exponent)` — a safe real power
(`base**exponent`; returns `0` for non-positive `base`). At most `40`
expression nodes.

Example (syntax only, not a good fit):
```
0.1 + 0.8*sig((p - 0.5) * pw(L, 0.8))
```

## Feasibility

The expression must parse under the grammar above (known names/functions
only, finite constants, node budget respected). Evaluating it must produce a
finite real number at every graded point. Any violation scores `0`.

## Objective (minimise)

The grader evaluates your expression at a fine grid of `(p, L)` points near
`pc` for `L in {128, 512}` (regenerated deterministically from the hidden
law — noisy, like your logbook readings) and forms the mean squared error
`MSE` against the true readings there, plus a small complexity penalty on
the expression's node count `nodes`:

```
F = MSE * (1 + LAMBDA * nodes)
B = MSE_of_constant_0.5 * (1 + LAMBDA * 1)     # internal baseline
Ratio = min(1000, 100 * B / F) / 1000
```

with a small fixed `LAMBDA`. A constant `0.5` reproduces the baseline
(Ratio ≈ 0.1). Getting the threshold LOCATION and the transition WIDTH right
at the untested, much larger sieve sizes drives `MSE` down; measurement
noise plus the parsimony tax keep even the exact law well below the ceiling.

## Why the logbook curve is a trap

On the small sieves in your logbook, each `L`'s crossing curve looks like an
independent, well-behaved logistic in `p` — you can fit its own midpoint and
width just fine. Extrapolating each of those two numbers to `L=128/512`
**separately** (e.g. each as its own function of `1/L`) throws away the fact
that they are locked together by the same `nu`: unless you happen to guess
the right functional shape for both simultaneously, the two extrapolation
errors compound instead of cancelling, and the predicted curve at the large
sieve sizes lands nowhere near the true, much sharper transition.

## Constraints

Time limit 5 s, memory 512 MB. `n` is at most a few hundred rows. Scoring is
fully deterministic.
