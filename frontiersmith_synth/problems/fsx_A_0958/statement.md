# Small-Signal Log: Recover the Saturating-Resonant Response Law

## Problem

A sealed device converts a driving signal into a response. You can dial two
knobs — the drive **frequency** `f` and the drive **amplitude** `a` — and
read the resulting **response** `y` off a meter. A technician has logged a
batch of readings, but only from a **small-signal, sub-resonant** test
window: modest amplitudes and frequencies safely below wherever the device's
natural resonance sits.

Physically, devices like this behave in a well-known way: the response
**saturates** (compresses, flattens toward a ceiling) as drive amplitude
grows large, and it **resonates** — rises to a peak then rolls off — as
drive frequency approaches a hidden natural frequency. The two effects
combine as a single **product**: an amplitude-dependent saturating gain
curve multiplying a frequency-dependent resonant curve. The exact resonance
location, resonance width, saturation scale, and overall gain are hidden
and were never in your test window.

Your job: from the small-signal log alone, output a closed-form expression
for the response that **extrapolates correctly** into a region you never
measured — large drive amplitudes (deep saturation) combined with
frequencies at, and beyond, the hidden resonance peak.

## Input (stdin)

```
N
f0 a0 y0
f1 a1 y1
...             (N rows total)
```
`f, a` are the drive frequency and amplitude (positive floats); `y` is the
measured response (positive, noisy — meter readings are not exact). `N`
grows with the difficulty ladder (60 to 114 rows).

## Output (stdout)

One line: a closed-form Python expression in `f, a`. Allowed tokens only:
- arithmetic: `+ - * / ** %`, parentheses, numeric literals;
- unary functions (exactly one argument each): `exp, log, sqrt, sin, cos, tanh, abs`;
- constants `pi`, `e`.

Any other name, attribute access, indexing, or multi-line output is
rejected. At most 400 AST nodes, at most 5000 characters, single line.

## Feasibility

The expression must parse under the whitelist above and evaluate to a
**finite** real value of moderate magnitude (`|value| <= 1e6`) at every
held-out point. Anything else (empty, unparseable, unknown symbol,
`nan`/`inf`, absurd magnitude) scores **0**.

## Objective (minimize)

The grader regenerates a deterministic **held-out region** — amplitudes
several times larger than anything you trained on, crossed with frequencies
at and beyond the (hidden) resonance peak, including corner combinations
that isolate the saturation effect from the resonance effect and combine
both — evaluates your expression there, and forms the loss

```
err = RMSE_heldout
F   = err * (1 + ALPHA * complexity)      (ALPHA = 0.003, complexity = # AST nodes)
```

so raw accuracy trades against expression simplicity: a huge formula that
merely contorts itself to the small-signal rows pays a price.

## Scoring

Let `B` be the held-out RMSE of the internal constant predictor
(`y = mean(train y)`). Then

```
sc = min(1000, 100 * B / max(1e-9, F))
Ratio = sc / 1000
```

Reproducing the constant baseline gives `Ratio ≈ 0.1`. Recovering the true
saturating-resonant shape drives held-out RMSE down and pushes the ratio
well above that — but measurement noise on the small-signal log and the
complexity penalty keep a perfect score out of reach.

## Constraints

- `60 <= N <= 114` training rows.
- Training frequencies and amplitudes are always strictly below the hidden
  resonance / saturation onset; held-out amplitudes and frequencies are a
  genuinely disjoint, larger region, with several points specifically
  isolating one effect (amplitude-only or frequency-only extrapolation)
  from the combined effect.
- Deterministic scoring only; the held-out region and ground-truth law are
  fixed per instance and regenerated inside the grader — no wall-time, no
  randomness in the score.

## Example (illustrative FORM only — NOT the hidden law)

Suppose (hypothetically) the response were just an unrelated additive
combination with no saturation or resonance at all, e.g.
`y = 3.0 + 0.5*f - 0.2*a*f`. A submission in that shape would be written
on stdout as:

```
3.0 + 0.5*f - 0.2*a*f
```

This is only to show the required output SHAPE (one line, allowed tokens).
The actual hidden law must be discovered from the data.
