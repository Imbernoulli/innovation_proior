# Two-Mode Dispersion Inversion

You are characterising a specimen from its **dispersion curve**: how fast a
wave of frequency `f` travels through it (its phase velocity `v(f)`) depends
on the specimen's material **stiffness** `k`. You do not get to see `k`
directly — only a noisy sample of `(f, v)` measurements from a lab sweep —
and you must hand back a closed-form expression for `v(f)` that a second,
higher-frequency sweep will grade you on.

## The physics

The specimen supports **two** propagation modes ("branches"), both driven by
the same hidden stiffness `k > 0`:

```
vA(f) = CA * sqrt(k * f)                    (branch A: grows without bound)
vB(f) = sqrt(k) * (CB - CD / (f + CE))      (branch B: saturates as f grows)
```

`CA, CB, CD, CE` are **known** per-specimen geometry constants — they are
given to you in the input. `k` is the ONE unknown material property you must
recover from data. At any frequency the sensor reports whichever mode
actually propagates, which is the LOWER-velocity branch:

```
v(f) = min( vA(f), vB(f) )
```

Because `vA(f) -> 0` as `f -> 0` while `vB(f)` stays positive, and `vA(f)`
grows without bound while `vB(f)` saturates, the two branches cross **exactly
once**, at some frequency `f_cross`. Below `f_cross` the sensor is reading
branch A; above it, branch B — the identity of "the observed mode" **swaps**
there, even though the measured velocity itself stays continuous (no jump in
value, only in which formula governs it).

**Illustrative FORM only — NOT the hidden law:** an unrelated example
expression is `2.0 * f / (1.0 + f) + 0.3` — this just shows the expression
syntax; the real dispersion law has the two-branch shape described above, and
`f_cross` is different for every test case and is never told to you.

## Input (stdin)

```
n t
CA CB CD CE
f[0]  v[0]
f[1]  v[1]
...
f[n-1] v[n-1]
```

`t` is the test id. `n` training rows follow: a frequency and a **noisy**
measured velocity. All training frequencies lie in a LOW band. The grading
sweep probes a HIGHER band of frequencies (not given to you), which for most
test cases contains the mode crossing.

## Output (stdout): a closed-form expression

Print **one line**: a single arithmetic expression in the variable `f`, using
`+ - * / **`, parentheses, numeric constants, the unary functions `sqrt`,
`abs`, and the binary functions `min`, `max`. At most 40 expression nodes.

## Feasibility

The expression must parse under the grammar above (only `f`, the listed
functions, and finite numeric constants) and must evaluate to a finite number
at every graded frequency. Any violation scores `0`.

## Objective (maximise)

Let `relerr_i = |your_v(f_i) - true_v(f_i)| / true_v(f_i)` over the graded
frequencies, and `nodes` the size of your expression. The grader forms

```
F = mean_i( exp(-relerr_i / 0.20) ) / (1 + 0.01 * nodes)
B = same formula for the checker's own constant-velocity baseline
Ratio = min(1000, 100 * F / B) / 1000
```

A constant reproduces the baseline (Ratio ≈ 0.1). Lower held-out relative
error raises the score; a mild parsimony tax discourages needlessly large
expressions. Independent measurement noise on the grading sweep keeps even a
perfect recovery of the modal structure below the ceiling.

## Why a single smooth fit is a trap

Training data always sits on ONE branch (the sensor cannot see the other
branch below/above the crossing). A curve fit that treats the visible shape
as "the" dispersion law and simply extends it will fit training beautifully
and then miss the swap: it keeps growing like branch A past a crossing where
the true mode has already switched to the saturating branch B, or vice
versa. Recovering `k` from the KNOWN branch formulas — not the visible curve
shape alone — determines both branches at once.

## Constraints

Time limit 5 s, memory 512 MB. `n = 26`. Scoring is fully deterministic.
