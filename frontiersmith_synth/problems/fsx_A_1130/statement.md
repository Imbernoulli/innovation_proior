# Thermal Timetable: Milking Drifting Lift Without Crowding

## Problem
A fleet of `G` gliders launches over a 2D plane on integer coordinates. Each glider has
altitude, starting at `A0` and capped at `Amax`. Time advances in discrete steps `t = 1..T`.

There are `K` thermals. Thermal `k` has an initial center `(cx,cy)`, a constant integer
drift velocity `(vx,vy)`, a capture radius `R`, and a lift constant `L`; at time `t` its
center is `(cx+vx*t, cy+vy*t)`. There are `M` beacons, each a fixed point `(bx,by)` with
capture radius `Rb` and value `Val`.

**Each step**, every still-flying glider moves by an integer vector `(dx,dy)` with
`dx^2+dy^2 <= Vmax^2`. After moving, look at which thermal disc (if any) currently
covers the glider's position; if several do, it uses the **lowest-indexed** one. Let `n`
be the number of still-flying gliders simultaneously inside that same thermal this step.
That glider's altitude changes by `floor(L / (1 + n^2))` (integer division). A glider
touching no thermal instead loses `Sink` altitude. Altitude is clamped to `[0, Amax]`; the
moment it reaches `0` the glider **lands out**: frozen in place, all its remaining moves
must be `(0,0)`, and it earns nothing further.

`floor(L/(1+n^2))` is **concave in the crowd size** `n`: the total lift a thermal delivers
per step, `n * floor(L/(1+n^2))`, is largest at small `n` and falls as more gliders pile
in. A thermal is a machine with limited *simultaneous* throughput, not a place — the same
thermal can profitably serve many gliders **staggered over time** as it drifts.

A beacon is captured (once, by whoever gets there first) if any still-flying glider's
position lands within `Rb` of it at any moment `t = 0..T` (before or after landing,
frozen position included). Score = sum of the values of all beacons captured by the fleet.

## Input (stdin)
```
G K M T Vmax Sink Amax A0
sx_1 sy_1            (G lines: glider launch points)
...
cx_1 cy_1 vx_1 vy_1 R_1 L_1     (K lines: thermals)
...
bx_1 by_1 Rb_1 Val_1            (M lines: beacons)
...
```

## Output (stdout)
`G` lines. Line `g` lists `2*T` integers `dx_1 dy_1 dx_2 dy_2 ... dx_T dy_T`: glider `g`'s
move at each step `t=1..T`.

## Feasibility
Output is valid iff: exactly `G*2*T` integer tokens are present; every move satisfies
`dx^2+dy^2 <= Vmax^2`; and once a glider has landed out, all its later moves are `(0,0)`.
Any violation scores `Ratio: 0.0`.

## Scoring
Let `F` be the total beacon value captured by your fleet. Let `B` be the checker's own
trivial construction: every glider dashes in a straight line for the beacon nearest its own
launch point, ignoring thermals entirely (`B >= 1` always, guaranteed reachable).
```
sc = min(1000.0, 100.0 * F / max(1e-9, B))
Ratio = sc / 1000.0
```
Matching the straight-dash baseline scores `0.1`; a fleet that reaches `10x` its total
value caps at `1.0`.

## Constraints
`5 <= G <= 13`, `3 <= K <= 8`, `6 <= M <= 15`, `24 <= T <= 48`, `Vmax=3`, `Sink=3`,
`Amax=90`, `A0=48`. Coordinates fit in `[-90,90]`. Time limit 5s, memory 512m.

## Example
`Vmax=1`. Glider starts at `(0,0)`, `A0=6, Amax=20, Sink=1`. One thermal at `(3,0)`,
static (`vx=vy=0`), `R=1, L=10`. One beacon at `(3,3)`, `Rb=1, Val=9`. `T=6`.
Moves `(1,0),(1,0),(1,0),(0,1),(0,1),(0,1)` give positions `(1,0),(2,0),(3,0),(3,1),
(3,2),(3,3)`. The glider is alone (`n=1`) inside the thermal disc from `t=2` through
`t=4` (`floor(10/2)=5` altitude each such step): altitude runs
`6 -> 5 -> 10 -> 15 -> 20(capped) -> 19 -> 18`, always positive, and it reaches `(3,3)`
at `t=6` capturing the beacon: `F=9`. This is only illustrative FORM, not a trap case —
on the generated (larger) tests, straight-dash baselines land out with modest reach while
a fleet that times its thermal visits to avoid crowding reaches far more total value.
