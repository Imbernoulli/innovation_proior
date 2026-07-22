# Overnight Ferry Charging: Water-Filling Under Session Fees

## Problem
An electric ferry docks overnight and must recharge before its first morning
crossing. The dock's grid connection is metered in `T` consecutive 15-minute
steps of duration `dt` hours each. In step `t` you choose a charging rate
`r_t` (kW). The charger cannot idle at an arbitrary trickle: in every step it
is either **off** (`r_t = 0`) or drawing **at least its minimum operating
rate** `r_min`, i.e. `r_t = 0` or `r_min <= r_t <= Rmax`.

Charging is not lossless: at rate `r_t` the charger dissipates
`alpha_t * r_t^2` kW as heat, so drawing energy at a high rate is
disproportionately wasteful. The grid bills for **every** kW drawn
(the loss is paid for, not subtracted from delivered energy), at the
step's spot price `p_t` per kWh.

Two more charges apply. First, each time the charger is switched on it pays a
**fixed connection fee** `Fee` for that **session** -- a session is a maximal
run of consecutive steps with `r_t > 0`; turning the charger off and back on
later starts a brand-new session and a brand-new fee. Second, the dock
operator levies a **demand charge** `D` per kW on the single **peak** rate
used anywhere during the night (this charge is paid once, on `max_t r_t`, not
per step).

You must deliver at least the energy target `E_target` (kWh) by morning. Your
task: choose the rate profile `r_1, ..., r_T` that minimizes the total bill.

## Input (stdin)
```
T dt Rmax Fee D r_min
p_1 p_2 ... p_T
alpha_1 alpha_2 ... alpha_T
E_target
```
`T` is an integer; all other values are floats. `p_t > 0` is the price
($/kWh) in step `t`; `alpha_t > 0` is that step's quadratic loss coefficient.

## Output (stdout)
Exactly `T` whitespace-separated real numbers: `r_1 r_2 ... r_T` (kW), on one
or more lines.

## Feasibility
An output is valid iff **all** hold:
- exactly `T` numeric, finite tokens are printed;
- every `r_t` is `0` (tolerance `1e-6`) or lies in `[r_min, Rmax]` (tolerance
  `1e-6`) -- a value strictly between the "off" tolerance and `r_min` is a
  hard violation, not a cheap way to fake a trickle;
- delivered energy `sum_t r_t * dt >= E_target` (tolerance `1e-4`).
Any violation scores `Ratio: 0.0`.

## Objective
Minimize the total bill
```
F = sum_t p_t * r_t * dt        (energy cost)
  + sum_t alpha_t * r_t^2 * dt  (quadratic loss cost)
  + Fee * (number of charging sessions)
  + D * max_t r_t                (demand charge on the peak rate)
```

## Scoring
The checker builds its own always-feasible reference: one **flat** rate
`r_avg = E_target / (T * dt)` held for the entire night (a single session).
Let `B` be that flat profile's bill (by the same formula above) and `F` your
bill. Since this is a minimization,
```
sc = min(1000.0, 100.0 * B / max(1e-9, F))
Ratio = sc / 1000.0
```
A profile matching the flat baseline scores `0.1`; a profile with one tenth
the bill saturates at `1.0`.

## Constraints
`4 <= T <= 60`, `dt = 0.25`, `1 <= Rmax <= 100`, `0 < r_min <= Rmax`,
`p_t, alpha_t > 0`, `0 < E_target < Rmax * dt * T`, and
`E_target / (T * dt) >= r_min` (the flat overnight rate the checker's own
reference profile runs at is itself a legal, above-minimum rate).

## Example (worked score, illustrative only -- NOT a real test case)
`T=2, dt=1, Rmax=10, Fee=5, D=1, r_min=1`, prices `[1, 1]`, `alpha=[1,1]`,
`E_target=4`. Flat baseline: `r_avg=2` both steps -> bill
`= (1*2+1*2) + (1*4+1*4) + 5*1 + 1*2 = 4+8+5+2 = 19 = B`.
A candidate `r=[4,0]`: `0` is off, `4 >= r_min` OK, delivered `4*1=4 >= 4` OK;
bill `= 1*4 + 1*16 + 5*1 + 1*4 = 4+16+5+4 = 29` (worse -- one big session
avoids a second fee but the quadratic loss and demand charge explode). A
candidate `r=[2,2]` equals the baseline itself (`Ratio=0.1`). The real test
cases give much richer price landscapes where spreading moderately across a
few chosen cheap windows, rather than either extreme, wins.
