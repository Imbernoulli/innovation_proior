# Placing Active Sites Along a Diffusion-Fed Strip

## Problem
A catalytic strip has `L` positions in a line (0-indexed), each either empty
or carrying one active site. Reactant comes from bulk reservoirs held at a
fixed concentration `C0` touching BOTH ends of the strip (the cells just
outside position `0` and position `L-1` are reservoir cells, permanently at
`C0`). You choose a subset of at most `B` positions to activate; every active
site is identical, and empty positions never react (they only pass reactant
along by diffusion).

An active site's usable turnover capacity is cut by two things: (1) CROWDING
-- every OTHER active site within `r_screen` cells competes for the same
local electronic/steric environment, reducing capacity multiplicatively; (2)
POISONING -- capacity falls with how much the site has already turned over
earlier in the run (poison accumulates with use, never clears). What actually
LIMITS turnover moment to moment, though, is DIFFUSION: reactant travels in
from the two reservoirs through the strip one cell at a time, and a site can
only convert what has already diffused to its own cell.

The run has `T=12` reaction cycles. In cycle `t`, active site `i` has
capacity `k_i = v_max * (1/(1+gamma*n_i)) * (1 - poison_i)`, where `n_i` is
the count of OTHER active sites with `|i-j| <= r_screen`, and `poison_i`
(starts at 0, capped at 1) carries over from earlier cycles. The reactant
field `c` (shared across cycles, starting at `C0` everywhere) is then
advanced by `R=25` explicit micro-steps of size `dt=0.1`:
```
for each micro-step (using the OLD c from the start of the micro-step):
  for each active site i:  produced_i = k_i * c[i] * dt
  for every cell i:        c[i] += dt*D*(c[i-1]+c[i+1]-2*c[i]) - produced_i
                            c[i] = clamp(c[i], 0, C0)
```
(`c[-1]` and `c[L]` are the fixed reservoir cells, always `C0`; inactive
cells have `produced_i=0` and only diffuse.) Summing `produced_i` over the 25
micro-steps of cycle `t` gives that cycle's turnover `u_i` for each active
site; then `poison_i += poison_rate * u_i` (capped at 1), and `sum(u_i)` over
all active sites is added to the running total conversion. This repeats for
all `T` cycles, with `c` and every `poison_i` carried forward the whole run.

Your goal: maximize the TOTAL conversion (the sum of every cycle's `sum(u_i)`
over the whole run).

## Input (stdin)
```
L B
D v_max gamma r_screen poison_rate C0
```
`L` positions, activation budget `B` (`B <= L`). `D` is the diffusion
coefficient, `v_max` the per-site max rate constant, `gamma` the crowding
coefficient, `r_screen` (integer) the crowding radius in cells, `poison_rate`
the per-unit-turnover poisoning rate, `C0` the reservoir concentration.

## Output (stdout)
Exactly `L` integers, each `0` or `1` (any whitespace layout). Token `i`
(0-indexed) is `1` if position `i` carries an active site, else `0`.

## Feasibility
Exactly `L` tokens, each parsing as `0` or `1`; the count of `1`s must be
`<= B`. Any wrong count, non-integer/out-of-range token, or budget overrun
makes the whole answer infeasible (score 0).

## Scoring
The checker runs the exact protocol above on your placement to get your
total conversion `F`, and on its own reference placement -- all `B` sites
packed into one contiguous block centered in the strip -- to get `F_ref`
(always positive and feasible). It reports
`Ratio = min(1000, 100*F/F_ref) / 1000`. Matching the reference placement
scores ~0.1; doing 10x better than it caps at 1.0 -- no placement is
engineered to reach that cap.

## Constraints
`16 <= L <= 46`, `4 <= B <= 17` with `B <= 0.4*L`, `0.02 <= D <= 3.3`,
`0.35 <= v_max <= 0.65`, `2 <= r_screen <= 3`, `0.15 <= gamma <= 0.35`,
`0.03 <= poison_rate <= 0.08`, `0.85 <= C0 <= 1.15`. Time limit 5s.

## Example (illustrative FORM only -- not a real hidden case)
`L=6, B=2`: one placement clusters both sites at positions 2 and 3
(adjacent); another spaces them at positions 0 and 5 (each next to a
reservoir, far apart, no crowding between them). Which wins depends on `D`:
with fast diffusion both patterns stay near full supply and the gap is
small; with slow diffusion the clustered pair shares one quickly-depleted
pocket AND crowds each other, while the far-apart pair each keeps tapping a
fresh, unshared reservoir all run long -- decided by running the protocol
above, not by inspection.
