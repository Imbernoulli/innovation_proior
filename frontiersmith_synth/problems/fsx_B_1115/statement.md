# Vaccine Line: Batch Sizes Through Three Stages

## Problem
A vaccine production line has three sequential stages — Formulation (1), Fill (2), and
Finish & Pack (3) — feeding a demand schedule of `T` shipment pulses. Schedule production
**lots** at each stage so every pulse is satisfied exactly, while minimizing the total raw
material drawn into Stage 1.

Stage 1 draws raw material from an unlimited source. Stages 2 and 3 draw material from the
output buffer of the stage before them. Every lot, at any stage, works the same way: at an
integer hour `t` you choose a draw quantity `R` (from the raw source for Stage 1, from
Buffer(i-1) for stage `i=2,3`). A **fixed setup scrap** `S_i` (material wasted in the
changeover) is subtracted: the lot's net output `O = R - S_i` (must be `>= 0`) is added to
Buffer(i) at hour `t`.

Material resting in a buffer **decays**: a quantity `Q` sitting in Buffer(i) for `h` waiting
hours shrinks to `Q*(1-d_i)^h` before it can be drawn on again (`d_i` is that buffer's hourly
decay fraction — one rate for the Formulation->Fill buffer, one for the Fill->Finish buffer,
one for the Finish->Pack buffer holding finished doses awaiting shipment). At any hour where a
buffer receives an addition and is also drawn on, the addition is applied first, then the
draw(s) against it.

Demand pulse `j` (hour `t_j`, quantity `D_j`) is a **mandatory** draw of exactly `D_j` from
Buffer 3 at hour `t_j` — you do not schedule it, it always happens. Buffer 3 must hold at
least `D_j` (post-decay, pre-draw) at every pulse.

## Input (stdin)
```
T
t_1 t_2 ... t_T
D_1 D_2 ... D_T
S1 S2 S3
d1 d2 d3
```
`t_j` strictly increasing integers (hours), `t_1 >= 0`. `D_j` positive integers. `S1,S2,S3`
positive integers (setup scrap per lot). `d1,d2,d3` are decay fractions per hour, `0<d_i<1`.

## Output (stdout)
```
N
stage_1 time_1 R_1
...
stage_N time_N R_N
```
`N` production lots in any order; each line gives `stage in {1,2,3}`, an integer `time` in
`[0, t_T]`, and the draw quantity `R` (a nonnegative number, up to 6 decimals).

## Feasibility
All of the following must hold, else `Ratio: 0.0`:
- `1 <= N <= 5000`; every `stage in {1,2,3}`; every `time` an integer in `[0, t_T]`; every `R`
  finite with `0 <= R <= 1e9`.
- For every lot, `R >= S_stage` (a lot must at least cover its own scrap).
- Simulating Buffer 1, Buffer 2, Buffer 3 forward in time (your additions, decay between
  touches, the mandatory demand draws on Buffer 3) must never leave any buffer below `-1e-6`.

## Objective
Minimize `F`, the total raw material drawn into Stage 1 across all Stage-1 lots (the sum of
`R` over lines with `stage == 1`).

## Scoring
The checker's own baseline `B` is the always-feasible one-lot-per-pulse plan: produce and
consume every stage in the very same hour as each pulse, so no decay is ever incurred:
`B = sum(D_j) + T*(S1 + S2 + S3)`. With your total raw material `F`:
```
sc = min(1000.0, 100.0*B / max(1e-9, F))
Ratio = sc / 1000.0
```
Matching `B` scores `0.1`; using `10x` less raw material caps the score at `1.0`.

## Constraints
`4 <= T <= 12`, `3 <= t_T <= 400`, `1 <= D_j <= 500`, `10 <= S_i <= 500`,
`0.0005 <= d_i <= 0.09`. Time limit 4s, memory 512m.

## Example
`T=2`, times `0 10`, demand `50 50`, `S1=S2=S3=10`, `d1=d2=d3=0.05`. One-lot-per-pulse draws
`50+30` raw material at each pulse: `B = 100 + 2*30 = 160`. A batched plan instead produces
one Stage-1 lot and one Stage-2 lot at `t=0`, sized so that after `10` hours of Buffer-2 decay
it still yields enough for Stage 3 to split into the two pulses (Stage 3 still lots twice, so
its scrap is paid twice, but Stage 1/2 setup is paid only once each). If the decay lost over
those `10` hours is smaller than the two setups saved, `F < B` and `Ratio` exceeds `0.1` — the
trade-off depends on how the decay rate compares to setup cost at each buffer, not on any
single stage's batch size alone.
