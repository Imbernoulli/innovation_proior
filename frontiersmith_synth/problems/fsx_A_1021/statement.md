# Depot Patience: Cooldown-Aware Overnight Charging

## Problem
An electric bus depot has `N` buses returning overnight and `C` charging berths shared
by all of them. Bus `i` arrives at tick `a_i`, with battery temperature `theta0_i`
(degrees) and needs `E_i` units of energy before the common morning deadline `D`
(ticks `0..D-1`; every bus must be full by the end of tick `D-1`). Chargers deliver
integer power levels `0..Pmax` per tick; only `C` buses may be actively charging
(power `> 0`) at any single tick, no matter how many berths physically exist — that is
the shared **berth capacity**.

Battery temperature obeys discretized Newton cooling plus a charging heat term, ticked
forward from the bus's own arrival:
```
theta[a_i] = theta0_i
theta[t+1] = AMBIENT + (theta[t] - AMBIENT) * RHO + HEAT * power[t]      for t = a_i .. D-1
```
with fixed constants `AMBIENT = 25`, `THETA_SAFE = 35`, `RHO = 0.82`, `HEAT = 2.0`
(same for every instance). A bus that never charges just cools toward `AMBIENT`.

State of charge is the fraction of required energy delivered so far, capped at 1:
`soc[t] = min(1, (sum of power[a_i..t-1]) / E_i)`. Degradation accrues every tick as a
**product** of heat-above-safe and how full the battery already is:
```
degrade_i = sum_{t=a_i}^{D-1}  c_i * max(0, theta[t] - THETA_SAFE)^2 * soc[t]
```
where `c_i` (a per-bus sensitivity, given in the input) is fixed for that bus. Crucially,
while `soc[t] = 0` the contribution is exactly zero **regardless of temperature** — a bus
that hasn't started charging yet degrades nothing while it cools.

## Input (stdin)
```
N D C Pmax
a_1 theta0_1 E_1 c10_1
...
a_N theta0_N E_N c10_N
```
`c10_i` is `10 * c_i` (integer); use `c_i = c10_i / 10.0`. All of `N, D, C, Pmax, a_i,
theta0_i, E_i, c10_i` are non-negative integers, `0 <= a_i < D`.

## Output (stdout)
Exactly `N * D` whitespace-separated integers, in row-major order: bus `0`'s `D` power
values `power_0[0] .. power_0[D-1]`, then bus `1`'s, and so on (conventionally printed
as `N` lines of `D` integers, one line per bus, but the checker only counts tokens, not
newlines). `power_i[t]` is the charging power bus `i` draws at tick `t` (each in
`0..Pmax`; must be `0` for `t < a_i`, since the bus is not physically present yet).

## Feasibility
An output is valid iff **all** hold:
- exactly `N * D` integer tokens, each in `[0, Pmax]` (any other count is rejected);
- `power_i[t] = 0` for every `t < a_i`;
- `sum(power_i[a_i .. D-1]) >= E_i` for every bus (it leaves fully charged);
- for every tick `t`, at most `C` buses have `power_i[t] > 0` (shared berth cap).
Any violation scores `Ratio: 0.0`.

## Objective
Minimize `F = sum_i degrade_i` (total overnight battery degradation), simulated exactly
by the recurrence above from the submitted power arrays.

## Scoring
The checker builds its own always-feasible reference schedule `B`: every bus charges
starting the instant it arrives, first-come-first-served over the `C` berths, at FULL
power `Pmax` (the obvious "charge as fast as possible" instinct). `B` is that
schedule's own `F`, computed with the same degradation recurrence. With minimization
normalization:
```
sc = min(1000.0, 100.0 * B / max(1e-9, F))
Ratio = sc / 1000.0
```
Matching the reference exactly scores `Ratio = 0.1`; cutting degradation to a tenth of it
caps at `1.0`.

## Constraints
- `5 <= N <= 40`, `10 <= D <= 130`, `1 <= C <= 8`, `Pmax = 4`.
- The generator always plants an instance where the reference construction `B` is
  reachable (some feasible schedule always exists).
- Time limit 5s, memory 512m.

## Example
Two buses, `D=10`, `C=1`, `Pmax=4`. Bus 0: `a=0, theta0=50, E=4, c10=10`. Bus 1:
`a=0, theta0=26, E=4, c10=10`. Charging bus 0 immediately at full power keeps it near
its hot steady state while its SoC ramps to 1 — expensive. Delaying bus 0 a few ticks
(free, since its SoC is still 0) lets `theta` decay toward 25 first, so the *same* later
full-power burn accrues far less `theta_excess^2 * soc`. (This worked numeric example is
illustrative of the mechanism only, not a literal transcript of the checker's baseline.)
