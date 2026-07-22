# One Feed Line, Many Ponds: Staggered Growth-to-Harvest Scheduling

## Problem
A biofuel plant operates `P` algae ponds sharing a **single feed pipeline** over a
horizon of `T` discrete steps (`t = 0..T-1`). Each pond `p` starts with initial
biomass `b0_p` and passes through exactly two phases:

- **Growth**: while pond `p` is growing, it may be fed `f_{p,t} >= 0` units at
  step `t`. Feeding below the pond's own **activation threshold** `tau_p` keeps
  the culture alive but triggers **no cell division that step** (feed is
  effectively wasted on maintenance). Feeding at or above `tau_p` grows biomass
  by the concave amount `a_p * sqrt(f_{p,t})`.
- **Harvest**: at its chosen **switch step** `switch_p`, pond `p` stops growing
  and its entire accumulated biomass is converted to fuel in one shot, at an
  efficiency that **decays with elapsed plant age**: `e0_p * decay_p ^ switch_p`
  (`0 < decay_p < 1`), independent of any other pond.

Every step, the pipeline can deliver at most `C` total feed units, split however
you like across the ponds that are actively growing that step:
`sum_p f_{p,t} <= C` for every `t`.

You choose, for every pond `p`: a **start** step `start_p` (growth begins),
a **switch** step `switch_p >= start_p` (harvest happens), and a feed amount
for each of pond `p`'s active growth steps `t in [start_p, switch_p)`. Before
`start_p` the pond simply holds its initial biomass `b0_p` unfed.

## Input (stdin)
```
P T
C
a_1 b0_1 e0_1 decay_1 tau_1
...
a_P b0_P e0_P decay_P tau_P
```
`1 <= P <= 15`, `1 <= T <= 45`. All other values are positive decimals.

## Output (stdout)
`P + 1` lines followed by `P` feed rows, interleaved as:
```
P
start_1 switch_1
f_{1,start_1} f_{1,start_1+1} ... f_{1,switch_1 - 1}
start_2 switch_2
f_{2,start_2} ...
...
```
The feed row for pond `p` has exactly `switch_p - start_p` numbers (possibly
zero, i.e. an empty line if `switch_p == start_p`).

## Feasibility
- `0 <= start_p <= switch_p <= T` (integers) for every pond.
- Every printed feed value is finite and `>= 0`.
- For every step `t`, the sum of feed values assigned to that step across all
  ponds is `<= C` (tolerance `1e-4`).
Any violation, or a malformed/truncated/non-finite output, scores `Ratio: 0.0`.

## Objective
Maximize total fuel `F = sum_p e0_p * decay_p ^ switch_p * B_p(switch_p)`,
where `B_p(switch_p) = b0_p + sum` of `a_p * sqrt(f_{p,t})` over active steps
`t` with `f_{p,t} >= tau_p` (steps below threshold contribute 0 growth).

## Scoring
The checker builds its own trivial baseline `B`: feed the **entire** line to
pond 1 alone for the whole horizon, and let every other pond harvest
immediately at `t = 0` with zero growth. `B` is always feasible and positive.
```
sc = min(1000.0, 100.0 * F / max(1e-9, B))
Ratio = sc / 1000.0
```

## Example
`P=2, T=3, C=1.0`; pond 1: `a=2.0 b0=1.0 e0=1.0 decay=0.9 tau=0.4`; pond 2:
`a=1.0 b0=1.0 e0=1.0 decay=0.9 tau=0.4`. (Illustrative shape only — not the
generator's actual distribution.)

Baseline: pond 1 alone gets `f=1.0` all 3 steps (`>= tau`): growth `2*sqrt(1)=2`
per step, `B_1(3) = 1+2+2+2 = 7`, fuel `= 1 * 0.9^3 * 7 = 5.103`. Pond 2 never
grows: fuel `= 1 * 0.9^0 * 1 = 1.0`. `B = 6.103`.

A submission with pond 1 growing only step 0 (`start=0, switch=1, f=[1.0]`)
and pond 2 growing steps 1-2 (`start=1, switch=3, f=[1.0, 1.0]`): pond 1's
`B=1+2=3`, fuel `=1*0.9^1*3=2.7`; pond 2's `B=1+1+1=3`, fuel `=1*0.9^3*3
=2.187`. `F = 4.887`, so `sc = 100*4.887/6.103 = 80.08`, `Ratio = 0.08008`.

## Constraints
`1 <= P <= 15`, `1 <= T <= 45`, `0 < C`, `0 < tau_p < C`, `0 < decay_p < 1`.
Time limit 5s, memory 512MB.
