# Strobe-Lit Fan Law — recovering a monotone speed law from Nyquist-folded readings

A shop fan's blade-pass frequency `f(x)` (in Hz) depends on a drive-level
control `x` through a fixed but undocumented **monotone power law**:
`f(x)` grows smoothly and strictly with `x`, following one hidden shape
for the whole unit. You film the fan with a **strobe camera that samples
at a fixed, comparatively low rate `fs`** (frames per second), sweeping
`x` across a range and recording one reading per frame.

The catch: a camera running at `fs` frames per second cannot tell a
blade-pass frequency `f` apart from any frequency that aliases to the same
appearance. Every reading you get back is the true frequency **folded**
into the baseband window `[0, fs/2]` by the standard Nyquist triangle fold:

```
fold(f, fs):  m = f mod fs
              return m        if m <= fs/2
              return fs - m   otherwise
```

So a reading of, say, `3.2` could really mean a true frequency of `3.2`,
or `fs - 3.2`, or `fs + 3.2`, or `2*fs - 3.2`, and so on — each reading
alone is ambiguous. Your job is to recover the true closed-form law
`f(x)`, correct on drive levels the sweep never reached.

## Input (stdin)

```
t  N  fs  F_MAX
x_0  reading_0
x_1  reading_1
...
x_(N-1)  reading_(N-1)
```

`t` is the test id, `N` the number of rows, `fs` the camera's fixed
sampling rate for this test, and `F_MAX` a loose, test-independent
physical ceiling — no fan of this class ever exceeds `F_MAX` Hz at any
drive level, training or held-out. Rows are given in **arbitrary order**.
The held-out grading grid (drive levels beyond the swept range) is **not**
given to you.

## Output (stdout): a closed-form law

Emit a single expression for `f` as a function of `x`. Allowed: numeric
constants, the operators `+ - * /`, unary `+/-`, parentheses, the single
variable `x`, and the functions `absv(a)`, `minv(a,b)`, `maxv(a,b)`,
`powv(a,b)` (computes `a` to the power `b`; `a` must evaluate positive).

**Illustrative FORM only — NOT the hidden law:**

```
5.0 + 1.5*absv(x - 3.0) / (1.0 + 0.05*x)
```

This only shows the syntax; the real law's shape, exponent and constants
must be discovered from the data.

## Feasibility

The expression must parse under the grammar above (only known
names/functions, correct arities, finite constants, at most 200
expression nodes). Any parse violation, or any non-finite or non-positive
value produced while evaluating the law on the held-out grid, scores `0`
for that test.

## Objective (minimise)

Let `pred_k` be your law evaluated at held-out drive level `x_k`, and
`true_k` the (noisy) true, **unaliased** frequency there. The grader forms
the mean **squared LOG error** (it rewards matching growth *rate*, not just
scale) plus a small parsimony tax on expression size `nodes`:

```
F = mean_k (log(pred_k) - log(true_k))^2 * (1 + LAMBDA * nodes)
B = mean_k (log(Rbar)   - log(true_k))^2 * (1 + LAMBDA * 1)
    # Rbar = flat geometric mean of YOUR OWN training READINGS
Ratio = min(0.90, 0.1 * (B / F) ** GAMMA)
```

with small fixed constants `LAMBDA, GAMMA` (`0 < GAMMA < 1`), capped below
1 so the score never saturates. Predicting the flat average of the raw
(folded) readings reproduces `B/F = 1` (Ratio = 0.1). A law that recovers
the true growth rate drives `F` toward the held-out measurement-noise
floor and `B/F` far above 1; the sub-linear `GAMMA` compresses that
spread so a merely-plausible-shaped law does not saturate, and noise plus
the finite sample keep even a strong recovery below the ceiling.

## Why the low camera rate is a trap

A reading is only ever `fold(f_true(x), fs)`, never `f_true(x)` itself.
Fitting the raw readings as if they were direct measurements works fine
only while the sweep never crosses a Nyquist zone boundary. Once it does,
the folded readings stop increasing with `x` — they zig and zag — and any
regression fit through them settles on a far shallower apparent growth
rate than the truth, which then collapses badly on the held-out grid where
the true frequency has grown well past what any folded reading could show.
The fix is not to ignore the fold but to invert it: since `f_true(x)` is
monotone, its Nyquist zone index is **non-decreasing** along sorted `x` —
a lattice-consistency constraint linking every row's unknown fold count to
every other row's. Some sweeps also carry a deliberate gap in `x` where the
zone index jumps by several steps at once, defeating any purely local,
one-fold-at-a-time correction.

## Constraints

Time limit 5 s, memory 512 MB. `N` ranges 26–44 rows depending on `t`
(harder, more heavily aliased sweeps get more rows). Scoring is fully
deterministic.
