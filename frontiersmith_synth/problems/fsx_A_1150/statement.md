# Ward Block Utility Plant: The Cascade Dispatch Ledger

## Problem
A hospital's central utility plant must, on every timestep of a dispatch window, meet three
simultaneous demands: **steam** (sterilizers, laundry), **chilled water** (ward and OR cooling),
and **electricity** (everything else). It has one boiler and four ways of turning boiler steam
and grid power into what the wards need:

- **Boiler**: burns fuel to make high-pressure (HP) steam, `b` units, up to capacity `Cap_b`.
  Fuel consumed is `a_b*b + c_b*b^2` (the quadratic term prices the extra irreversibility of
  running the boiler harder).
- **Turbine**: any portion `x` of the boiler's HP steam (`0<=x<=b`) can be routed through a
  back-pressure turbine instead of straight to the steam header. It yields `eps_p*x`
  electricity and `eps_s*x` low-pressure (LP) exhaust steam, with `eps_p+eps_s<1` (the turbine
  itself destroys some exergy). This is FREE beyond the boiler fuel already counted above.
- **Absorption chiller**: converts LP exhaust steam `z<=eps_s*x` into `COP_abs*z` chilled water.
- **Electric chiller**: converts electricity `e_chill` into `COP_elec*e_chill` chilled water
  (`COP_elec` is far larger than `COP_abs` — a first-law "single-output efficiency" ranking
  always prefers it).
- **Grid import**: buys electricity `e_grid` at fuel-equivalent cost `a_g*e_grid + c_g*e_grid^2`.

Steam not routed to the turbine (`b - x`) satisfies steam demand directly.

## Input (stdin)
Line 1: `T a_b c_b Cap_b eps_p eps_s COP_abs COP_elec a_g c_g`.
Then `T` lines, one per timestep: `S Pw Ch` — integer steam, electricity, and chill demand.

## Output (stdout)
Exactly `T` lines. Line `t` gives five non-negative numbers for that timestep:
```
b x z e_chill e_grid
```
`b` = boiler HP steam output, `x` = HP steam routed through the turbine, `z` = LP steam sent to
the absorption chiller, `e_chill` = electricity sent to the electric chiller, `e_grid` =
electricity bought from the grid.

## Feasibility
For every timestep `t` (tolerance `1e-6`), all five numbers finite and `>=0`, and:
- `x <= b <= Cap_b`
- `z <= eps_s*x` (can't feed the absorption chiller more LP steam than the turbine made)
- `b - x >= S` (steam demand met)
- `eps_p*x + e_grid >= Pw + e_chill` (electricity demand met)
- `COP_abs*z + COP_elec*e_chill >= Ch` (chill demand met)

Any violation, wrong token count, or non-finite value scores `Ratio: 0.0`.

## Objective (minimize)
Total primary fuel across the window:
```
F = sum_t [ a_b*b_t + c_b*b_t^2 + a_g*e_grid_t + c_g*e_grid_t^2 ]
```

## Scoring
The checker builds an internal baseline `B`: the fully-**dedicated** schedule (boiler exactly
for steam, electric chiller exactly for chill, grid for the rest of the power), with every
quantity generously oversized (unoptimized — a "just make it work" construction, not tightly
sized). With `F` your total fuel:
```
sc    = min(1000, 100 * B / max(1e-9, F))
Ratio = sc / 1000
```
Matching the baseline scores about `0.1`; using ten times less fuel caps at `1.0`. Because each
dedicated converter is individually "efficient" per unit of its own output, ranking converters
that way and never routing steam through the turbine still beats the oversized baseline — but it
leaves fuel on the table whenever chill and power demand are both large enough that the
turbine's electricity **and** its LP-steam by-product both find a use. Whether that cascade pays
off, and by how much, depends on the timestep's demand mix, not on any converter's standalone
efficiency rating.

## Constraints
`6 <= T <= 150`. All coefficients are positive; `Cap_b` always leaves headroom above every
timestep's steam demand.

## Example
`T=1`, `a_b=1.0, c_b=0.001, Cap_b=100, eps_p=0.1, eps_s=0.8, COP_abs=0.8, COP_elec=3.0,
a_g=4.0, c_g=0.05`, demand `S=20, Pw=20, Ch=30`.

Dedicated (`x=0`): `b=20` (exactly meets steam), `e_chill=30/3=10`, `e_grid=20+10=30`.
Fuel `= (20 + 0.001*20^2) + (4.0*30 + 0.05*30^2) = 20.4 + 165.0 = 185.4`.

Cascading with `x=15`: `b=35`, LP steam `y=0.8*15=12` all sent to the absorption chiller
(`z=12`) covering `0.8*12=9.6` of chill; the remaining `20.4` chill needs `e_chill=6.8`
electricity. Total electricity needed `=20+6.8=26.8`, and the turbine already supplies
`0.1*15=1.5`, so `e_grid=25.3`. Fuel `= (35 + 0.001*35^2) + (4.0*25.3 + 0.05*25.3^2) =
36.225 + 133.205 = 169.43` — cheaper than dedicated (though not necessarily the checker's true
optimum), illustrating that pushing more steam through the turbine can beat the dedicated split
when chill demand is large relative to power demand.
