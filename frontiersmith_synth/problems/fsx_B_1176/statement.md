# Whole-House Meter: Disaggregating Appliances by Edge Signature

## Problem

A whole-house smart meter reports one aggregate power trace `agg[1..T]`
(watts, integer, one sample per time step). Behind that single number are
`A` appliances, each a tiny **state machine with exactly two states, OFF
(power 0) and ON (power `P`)** -- but an appliance is not a constant load:
once it switches ON it is legally required to STAY on for a bounded number
of steps before it may switch OFF again, and symmetrically for OFF. So the
same instantaneous power reading can come from different appliances, and
the same appliance can legally be silent for a long or a short while
depending only on timing, not on the reading itself.

Appliance `a` has power `P_a` and legal dwell windows
`[minOn_a, maxOn_a]` (steps it must/may stay ON once switched on) and
`[minOff_a, maxOff_a]` (same for OFF). All appliances start OFF one step
before the trace begins. Your job: reconstruct one state sequence per
appliance, `x_a[1..T] in {0,1}`, that is individually legal for its own
state machine and, together, explains the observed aggregate trace.

**The trap**: several appliances may share the exact same power `P_a`
(e.g. two different 1200W loads) while having very different dwell
windows (a short-burst load vs. a long-run one). Reading "which appliance
is on" off the instantaneous aggregate alone is therefore ambiguous --
and worse, several appliances can be ON *simultaneously*, so the
aggregate at any instant is a sum, not a label. What IS almost always
unambiguous is a **step change (edge)**: it equals exactly one
appliance's own ON/OFF power delta, and only appliances currently
*legally eligible* to transition (already dwelt at least their own
minimum) can be its source. Chaining edges, filtered by each appliance's
own legal timing, separates appliances a naive level-match cannot.

## Input (stdin)
```
T A
wT wA
P_1 minOn_1 maxOn_1 minOff_1 maxOff_1
...
P_A minOn_A maxOn_A minOff_A maxOff_A
agg_1 agg_2 ... agg_T
```
`wT`, `wA` are the scoring weights below (read and use them -- they vary
per test).

## Output (stdout)
```
A
x_1[1] x_1[2] ... x_1[T]
x_2[1] ... x_2[T]
...
x_A[1] ... x_A[T]
```
First the appliance count `A` (must equal the input), then, for each
appliance in input order, its `T` state tokens (each `0` or `1`),
whitespace-separated (line breaks are cosmetic).

## Feasibility
- Exactly `1 + A*T` tokens; the first must equal `A`; every state token
  must be `0` or `1` (finite integers only).
- Split appliance `a`'s sequence into maximal constant-state runs. Every
  run's duration must not exceed the dwell max for its state
  (`maxOn_a`/`maxOff_a`). Every run that is **not** the first or last run
  in the sequence must also meet the dwell min (`minOn_a`/`minOff_a`) --
  the very first and last runs may be shorter, since the true segment
  could start/end outside the observed window.
Any violation (for any appliance) scores `Ratio: 0.0`.

## Objective (maximize)
Let `recon[t] = sum_a P_a * x_a[t]` and `rawFit = max(0, 1 -
sum_t|agg[t]-recon[t]| / sum_t agg[t])`. Let `rawAcc` be the average, over
appliances, of the fraction of steps where your `x_a[t]` matches the
(hidden, unrevealed) true state. Both raw fractions have a floor well
above 0 (a roughly-balanced ON/OFF sequence overlaps a phase-blind guess
on many steps "by luck"), so the score reshapes them super-linearly,
punishing that floor while still letting a genuinely correct
reconstruction saturate: `traceFit = rawFit^2`,
`meanAcc = max(0, 2*rawAcc - 1)`, `F = wT*traceFit + wA*meanAcc`.

## Scoring
The checker also evaluates `F` on its own naive reference (every
appliance cycling at its own *minimum* legal dwell, off then on,
repeating, ignoring the trace entirely) to get baseline `B > 0`. Your
score is `min(1.0, 0.1 * F / B)`, printed as `Ratio: <value>`.

## Constraints
`3 <= A <= 6`, `36 <= T <= 62`, `100 <= P_a <= 2000`. Time limit 4s.

## Example (illustrative form only, not a real test case)
Two appliances, `P=[100,100]`, both `minOn=2,maxOn=3,minOff=2,maxOff=3`,
`T=6`, `agg=[0,100,100,200,100,0]`. One legal reconstruction: appliance 1
ON at steps 2-4, appliance 2 ON at steps 3-5 -- both runs length 3, both
legal, `recon` matches `agg` exactly. Mechanics only; real instances have
more appliances and shared power levels among them.
