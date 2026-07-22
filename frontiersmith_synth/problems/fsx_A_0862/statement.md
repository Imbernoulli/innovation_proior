# Breaker Drill: Water-Filling K Robot Arms Under One Power Cap

## Problem

A factory is running a blackout drill. `K` robot arms sit on **one shared
breaker**. Arm `i` must move its end-effector through a total distance of
exactly `D[i]` units along its own fixed path (no arm depends on any other
arm — they can move at the same time or at different times, your choice).

Time is discretized into ticks `1, 2, 3, ...`. On each tick, you choose an
integer speed `v_i(t) >= 0` for every arm (0 means idle that tick). Moving
at speed `v` for one tick advances that arm's progress by exactly `v`
distance units and draws power `v^2` (a robot arm moving twice as fast
draws four times the current — the exact quadratic cost rule of the
breaker box). The **breaker trips** if the *combined* instantaneous power
of all arms moving in the same tick ever exceeds the shared cap `P`:

```
sum over all arms i of v_i(t)^2  <=  P     (every tick t)
```

Each arm must reach its target distance **exactly** — no undershoot, no
overshoot — and every arm's progress is monotonically non-decreasing (you
cannot "unmove"). The schedule ends the moment every arm has reached its
target; the number of ticks you use is the **makespan**.

## Input (stdin)

```
K P A
D[1] D[2] ... D[K]
```
`2 <= K`, `P >= 1`, `A >= 1`, each `D[i] >= 1`. One arm's distance is
deliberately far larger than the rest (a long chain sharing the breaker
with many short ones) — the exact mix is only visible by reading `D[]`,
never assume it is uniform or sorted.

## Output (stdout)

`T` lines, one per tick used, each with exactly `K` non-negative integers:
```
v_1(t) v_2(t) ... v_K(t)
```
(space separated, one line per tick `t = 1..T`).

## Feasibility

Reject (score 0) if: any line has a wrong token count or an unparseable /
negative speed; any tick's combined power `sum v_i(t)^2` exceeds `P`; any
arm's cumulative progress ever exceeds its target `D[i]`; or, at the last
line, any arm's cumulative progress is not **exactly** `D[i]`.

## Objective

Minimize
```
cost = A * T + sum over all ticks t, all arms i of  v_i(t)^2
```
i.e. a fixed penalty `A` for every tick the drill runs, plus the total
electrical energy burned (sum of every tick's power draw). This is a pure
time-vs-energy trade-off: finishing sooner costs more energy per unit of
distance (running a arm at speed `v` for a fixed distance burns energy
proportional to `v` itself — quadratic power times inversely-proportional
time), and the shared cap limits how many arms can pay that price at once.

## Scoring

The checker computes your cost `F` and its own baseline `B`: the fully
serial, speed-1-only construction (move exactly one arm at a time, one
distance unit per tick — always feasible since `1^2 = 1 <= P`). For
minimization, `ratio = min(1, 0.1 * B / F)`, printed as `Ratio: <value in
[0,1]>`. Lower `F` scores higher, capped below 1 so there is always
headroom to do better.

## Constraints

`K` up to a few dozen, `D[i]` up to a few thousand, time limit 5s, memory
512MB.

## Example (worked, illustrative shape only — real instances plant a sharp
size skew between one long chain and many short ones, not shown here)

`K=2, P=5, A=3`, `D = [3, 2]` (solo max speed here is 2, since `2^2=4<=5`
but two arms both at speed 2 need `8 > 5`). Feasible schedule:
```
2 0
1 2
```
Tick 1: arm 1 moves 2 (power 4). Tick 2: arm 1's last unit (1) plus arm 2
at speed 2 (power `1+4=5<=5`) — both reach `[3,2]` exactly, so `T=2`. Cost
`= 3*2 + (4+5) = 15`. Whether crowding more arms into fewer ticks beats
spreading them over more, gentler ticks is exactly what `P`, `A`, and the
real shape of `D[]` in your input determine — read them, don't assume this.
