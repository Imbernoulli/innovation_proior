# Scrap Tickets from the Annealing Oven

## Problem
A batch oven anneals parts. Every job runs a **two-segment schedule**: an optional
ramp segment (heat/cool to setpoint `S1` and hold it for `D1` seconds, starting
from the ambient temperature `T0 = 20`), then a hold segment (drive to setpoint
`S2` and hold it for `D2` seconds). Internally the part's temperature obeys a
**first-order thermal lag towards whatever setpoint is currently commanded**,
with **thermal mass making heating and cooling relax at different, hidden
rates** (`k_heat` when the setpoint is above the current temperature, `k_cool`
when it is below). Both rates are fixed once per test but never disclosed.

Quality control only checks the **final `w` seconds of the hold segment**
(`w` is given per job): the part *passes* iff the true temperature stayed
inside a target window `[Lo, Hi]` for that entire final stretch; otherwise it's
scrapped. Because the check window only spans the tail of the hold, the
temperature is guaranteed monotone within it, so passing is equivalent to both
its endpoints lying in `[Lo, Hi]`. You never see the continuous temperature
trace or the rates — only, for each **training** job, its full schedule and the
binary pass/fail ticket a worker wrote down.

You must then predict, for a batch of **held-out** jobs (schedules only, no
ticket), a **normalized margin**: how comfortably (or badly) each job would
pass, defined as `margin / (W/2)` where `margin` is the smaller of "how far
above `Lo`" and "how far below `Hi`" the check window's bounds land, and
`W = Hi - Lo`. A margin of 0 means "exactly on the boundary"; near +1 means
"dead center"; a large negative value means "badly scrapped." Held-out jobs
use much shorter durations and a wider setpoint range than any training job —
genuine extrapolation, not interpolation.

## Input (stdin)
```
t N Q
S1 D1 S2 D2 w Lo Hi label      (N lines, training tickets; label is 0 or 1)
S1 D1 S2 D2 w Lo Hi            (Q lines, held-out queries, in query order)
```
All temperature/duration fields are floats; `label` is an integer.

## Output (stdout)
Exactly `Q` whitespace-separated finite base-10 floating point numbers: your
predicted normalized margin for each held-out query, **in the order given**.

## Feasibility
The output must parse as exactly `Q` finite floats (no `nan`/`inf`, no
missing/extra tokens). Any violation scores `Ratio: 0.0`.

## Objective and Scoring
For each query the checker knows the true normalized margin `m_i` (it re-derives
the hidden rates and re-simulates — you never see them). Your per-query error is
`e_i = min(4.0, |pred_i - m_i|)` (clipped so one wild outlier can't dominate).
Let `F = average(e_i)`. The checker also computes `B`, the same clipped-error
average for the "predict 0 for everything" construction. Your score is
```
Ratio = min(0.9, 0.1 * B / max(1e-9, F))
```
so "predict 0" scores exactly `0.1` (F equals B), driving `F` below `B` raises
your score, and a near-perfect physics recovery is capped at `0.9` (headroom
is deliberately left above a perfect reference). **You are maximizing this
Ratio.**

## Why the obvious approach is a trap
Fitting a single symmetric relaxation constant against the training pass/fail
bits (and, for convenience, assuming every hold starts from ambient — i.e.
ignoring the ramp segment) reproduces the training tickets tolerably, because
most training jobs are long enough, and simple enough, for that compromise
constant to look fine. It does not generalize: the held-out jobs are short
(so the heating/cooling asymmetry dominates the outcome) and often two-segment
(so the ramp's ending temperature — not ambient — is what the hold actually
starts from). Only recovering the two separate rates, and correctly composing
both segments, transfers.

## Constraints
`N` up to 220, `Q` up to 24, all temperatures/durations bounded and finite.
Time limit 5 s, memory 512 MB. Fully deterministic.

## Example (illustrative FORM only — NOT the hidden law)
If (hypothetically) the response were instantaneous with no lag at all, a job
with `S2=100, Lo=90, Hi=110` would simply score `margin = 10`, `W/2 = 10`,
normalized margin `1.0`. The real oven has genuine thermal lag with two
different hidden rates — you must discover both from the tickets.
