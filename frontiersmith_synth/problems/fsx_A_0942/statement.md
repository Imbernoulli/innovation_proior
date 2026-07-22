# Pressure Vessel Leak Forensics

A sealed pressure vessel leaks through a worn seal: its level `L` (in
arbitrary "units") obeys `dL/dt = -c * L^alpha` for hidden positive
constants `c` and `alpha`, with `alpha` roughly between `1.2` and `1.8`
(the seal geometry makes the leak super-linear in level, but you don't
know exactly how super-linear).

The vessel is also topped off from a tanker truck, most days. The facility
logs each **day's total delivered volume** — possibly one or two unlogged
deliveries summed together — but never individual times, counts, or
amounts. A level sensor is read once per day, at day's end, with small
measurement noise.

You are given a telemetry log recorded while the vessel stayed in a
**low-level regime**. You must recover the leak law well enough to predict
the leak *rate* in a **higher, unseen level regime** that the grader checks
against but never shows you.

**Illustrative FORM only — NOT the hidden mechanism:** if someone told you
"the mechanism looks like `rate = 0.02 + 0.3/(1+L)`", that is just an example
of the kind of arithmetic shape an answer can take — not a hint about the
real (power-law) family, which you already know from the paragraph above.

## Input (stdin)

```
D t
L0
Q_1 E_1
Q_2 E_2
...
Q_D E_D
```

`t` is the test id. `D` days of telemetry follow `L0` (the level reading
before day 1). Each day `d` gives `Q_d` (that day's total delivered volume,
`>= 0`, possibly `0`) and `E_d` (the level reading at the *end* of day `d`).
One day = one fixed time unit (`T_day = 1`). Deliveries can land at any
unlogged instant within their day.

## Output (stdout)

Print **exactly one line**: a single closed-form arithmetic expression for
the leak rate as a function of the level, using the variable `L`. Allowed:
`+ - * /`, `**` (power), parentheses, numeric literals — nothing else (no
function calls, no other variable names). Example shape: `0.014*L**1.6`.

## Feasibility

The line must parse under the grammar above (only `L`, numeric constants,
and `+ - * / **`), be under 300 characters / 40 expression nodes, and
evaluate to a **finite real number at every held-out level** the grader
checks. Any violation scores `0`.

## Objective (minimise)

The grader regenerates held-out levels in the higher regime (never shown to
you) with a noisy observed leak rate at each, and forms `F`, the mean
squared error between `log(your prediction)` and `log(observed rate)`
there. It also computes `B`: the same metric for its own **trivial**
reference — a proportional fit `rate = k*L` (exponent forced to `1`)
calibrated directly from your day-to-day level drops, *without* correcting
for delivery volumes. Then

```
Ratio = min(900, 100 * B / F) / 1000
```

A `rate = k*L` submission reproduces `B` (Ratio ~= 0.1). The `900` cap (not
`1000`) keeps a fixed sliver of headroom above every submission, including
the unreachable exact law, so the score never saturates.

## Why the day-to-day deltas are a trap

If you just take `(E_(d-1) - E_d) / T_day` as "the decay rate that day" and
fit a power law to it, delivery days look like anomalously *small*
(sometimes negative) decay — the delivery is baked invisibly into the
delta. Fitting through those points, instead of removing the delivery's
effect, drags your exponent and constant away from the truth, and
extrapolating to the higher held-out regime the error compounds.

## Worked example (of the scoring mechanism only, not the hidden law)

Suppose day 5 shows `Q_5 = 1.20`, `E_4 = 6.00`, `E_5 = 5.30`. With *true*
(noise-free) levels, the volume lost to leakage that day is *exactly*
`Q_5 + E_4 - E_5 = 1.90`, **no matter when the 1.20 units arrived, or as one
delivery or two** — mass balance doesn't care about timing or count. Your
`E` readings carry a little sensor noise, so this is only a close estimate
in practice — but unbiased, unlike ignoring `Q_d` altogether. Separately: if
a held-out point has observed rate `0.420` and your expression evaluates to
`0.500` there, it contributes `(ln 0.500 - ln 0.420)^2 ~= 0.0311` to `F`.

## Constraints

`D` is at most a few dozen. Time limit 5 s, memory 512 MB. Scoring is
fully deterministic.
