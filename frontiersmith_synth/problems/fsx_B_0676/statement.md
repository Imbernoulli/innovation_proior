# River Staircase: Head-Coupled Release Scheduling

## Problem

A river carries water through `N` dams in series, numbered `1..N` from
upstream to downstream. Dam `i` has a reservoir of capacity `C_i`, a
turbine that can pass at most `Rmax_i` units of water per time step, and a
travel delay `delay_i` (steps) for water leaving dam `i-1` to arrive at dam
`i` (dam 1 has no upstream). The simulation runs for `T` discrete steps.

At the start of step `t`, dam `i` holds `level_i[t]` units of water. You
choose a release `release_i[t]`, the amount passed through the turbine at
that step, with

```
0 <= release_i[t] <= min(level_i[t], Rmax_i)
```

Water arriving at dam `i` during step `t` is

```
inflow_i[t] = localInflow_i[t] + passthrough_{i-1}[t - delay_i]
```

(the second term is 0 for `i = 1`, or if `t - delay_i < 1`).
`localInflow_i` is given in the input (tributaries feeding straight into
dam `i`, or, for dam 1, the headwater series — which may include sustained
flood pulses).

After release and inflow, the pre-spill level is
`pre_i[t] = level_i[t] - release_i[t] + inflow_i[t]`. If this exceeds
`C_i`, the excess is force-spilled over the spillway (bypasses the
turbine, generates no power) and `level_i[t+1] = C_i`; otherwise
`level_i[t+1] = pre_i[t]` and there is no spill. Released water AND
spilled water both continue downstream after the same delay:
`passthrough_i[t] = release_i[t] + spill_i[t]`.

The power generated at dam `i`, step `t`, is

```
power_i[t] = release_i[t] * gain_i(level_i[t]),   gain_i(L) = a_i + b_i * sqrt(L / C_i)
```

`gain_i` is a concave, increasing function of the level BEFORE this
step's release/inflow (higher head -> more energy per unit of water
released). `a_i, b_i, C_i, Rmax_i, delay_i` and the initial level
`init_i = level_i[1]` are all given per-dam in the input.

**Objective (maximize):** total energy `F = sum_{i,t} power_i[t]` over the
whole horizon.

The scoring formula's shape (not its constants) is: your `F` is compared
against an internal feasible reference construction `B`; the printed ratio
grows with `F/B` and saturates once `F` reaches about `10x` that
reference — so both squeezing out extra energy and avoiding forced spills
matter, but no single instant dominates the score.

## Input (stdin)

```
N T
C_1 Rmax_1 a_1 b_1 delay_1 init_1
...
C_N Rmax_N a_N b_N delay_N init_N
localInflow_1[1] ... localInflow_1[T]
...
localInflow_N[1] ... localInflow_N[T]
```
All values are floats except `N, T, delay_i` (integers). `delay_1 = 0`.

## Output (stdout)

`N` lines; line `i` has exactly `T` space-separated non-negative floats:
`release_i[1] ... release_i[T]`.

## Feasibility

For every `i, t`: `release_i[t]` finite, `>= 0`, and
`<= min(level_i[t], Rmax_i)` under the physics above (levels are
determined mechanically from your releases, not submitted). Wrong line/
token counts, non-finite values, negative releases, or any step that
overdraws a reservoir or exceeds turbine capacity scores `Ratio: 0.0`.

## Constraints

`4 <= N <= 8`, `40 <= T <= 400`, `600 <= C_i <= 1800`,
`0.15*C_i <= Rmax_i <= 0.30*C_i`, `1 <= delay_i <= 5` for `i>1`,
time limit 2-5s per case, each `.in` well under 5 MB.

## Example (worked, illustrative shape only)

2 dams, `T=3`, no delay effect shown for brevity. Suppose dam 1:
`C=1000, Rmax=300, a=0.2, b=0.8, init=500`, no inflow. If you release
`300` at step 1 when `level=500`: `gain = 0.2+0.8*sqrt(0.5) ≈ 0.766`,
`power ≈ 229.8`, new level `=200`. Releasing the same `300` units later
when the level has recovered to a higher value yields strictly more
power for the identical water — this is the core trade-off: draining a
reservoir now to maximize this instant's output can cost you head (hence
energy) for every future release through that dam, and a dam that is
never given spare buffer force-spills for free during a flood pulse. Real
`a_i, b_i, C_i, Rmax_i, delay_i` and the inflow tables live in the input.
