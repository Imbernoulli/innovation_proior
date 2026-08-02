# Winter Data, Summer Ceiling

A field of solar panels feeds an inverter that turns DC array power into
grid AC power. Every inverter has a hard capacity: however much DC power
the panels could produce, the inverter's AC *output* can never exceed its
own rating — the excess is simply clipped away. You are given a logger
trace from **one array**, recorded over a **calm winter**, and must predict
its power output on a **fast, sunny summer day** for the same array.

The catch: winter sun is weak. On the days you have data for, irradiance
never comes close to the level that would make the inverter clip, so the
visible relationship between sunlight and power looks like a clean,
unbounded curve. Summer is different — irradiance regularly exceeds the
level the inverter can pass through, and the true output goes flat at a
ceiling you have never once observed.

## Input (stdin)

```
n t N
G[0]  T[0]  P[0]
G[1]  T[1]  P[1]
...
G[n-1] T[n-1] P[n-1]
```

`t` is the test id. `N` is this array's DC **nameplate capacity** (kW),
constant for the whole trace. Each row gives irradiance `G` (W/m^2),
ambient temperature `T` (deg C), and the measured AC power `P` (kW) at one
moment. Rows are i.i.d. samples, not a time series — order carries no
information.

Physically: power scales roughly linearly with irradiance (normalised
against the standard 1000 W/m^2 test condition used to rate `N`), and
falls off mildly as the panels get hotter (a standard temperature derate).
The exact per-array efficiency and temperature coefficient are **not**
given — fit them from the data. Separately, the inverter enforces a hard
ceiling: industry fleets like this one typically clip somewhere between
**58% and 88% of nameplate `N`**, but the exact fraction for THIS array is
never given and is never approached in your winter data.

The held-out grading trace is a **different, sunnier** period for the same
array (same `N`, same hidden efficiency, same hidden clip fraction); it is
NOT given to you.

## Output (stdout): one closed-form expression

Emit exactly one line: an arithmetic expression in `G`, `T`, `N`, using
`+ - * /`, parentheses, numeric constants, and the functions `min(a,b)`,
`max(a,b)`, `absv(x)`. No other names, no assignments, no state.

**Illustrative FORM only — NOT the hidden law** (shows the syntax, not the
shape of the real relationship, which you must discover from the data):

```
max(0, 0.05 * T * G - 12)
```

## Feasibility

The expression must parse under the grammar above (known names/functions
only, finite constants, at most 40 expression nodes, at most 20000 bytes).
Any violation, or any non-finite value produced while evaluating it on the
held-out rows, scores `0`.

## Objective (maximise)

Let `MSE` be the mean squared error of your expression against the true
held-out power, and `nodes` the number of expression nodes you used. The
grader forms

```
F = MSE * (1 + LAMBDA * nodes)
B = MSE_of(0.4 * N) * (1 + LAMBDA * 1)      # internal baseline
Ratio = min(1000, 100 * B / F) / 1000
```

with a small fixed `LAMBDA`. Predicting a flat `0.4*N` reproduces the
baseline (Ratio ~= 0.1). Lowering held-out error raises the score; a small
parsimony tax discourages needlessly large expressions. Report the highest
Ratio you can — 1.0 is not reachable (sensor and microclimate noise, plus
the unavoidable uncertainty in this array's exact clip fraction, keep even
a very good model below the ceiling).

## Why the winter branch is a trap

On winter data, `G` never gets near the clip, so `P` looks like a single
smooth function of `G` (and mildly of `T`) with no bend in it — any
reasonable curve fit will look excellent. That same curve, extrapolated
into summer irradiance levels, keeps climbing long after the true array has
gone flat. The fix is not a better curve fit on the visible data; it is
recognising that a ceiling must exist at all, and putting a physically
sensible number on it even though it was never observed.

## Constraints

`n` is between 220 and 390 rows. Time limit 5 s, memory 512 MB. Scoring is
fully deterministic.
