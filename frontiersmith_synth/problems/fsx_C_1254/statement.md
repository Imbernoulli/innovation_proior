# Turning Off the Parts Nobody Is Using: Power-Gating Partition

## Problem

You are given `N` hardware blocks and `K` activity traces over a shared window of `T`
timesteps. In trace `k`, block `i` is either doing useful work (`1`) or idle (`0`) at each
timestep. Design ONE power-gating scheme, fixed once, replayed against all `K` traces:

1. Partition the `N` blocks into at most `D` **power domains**.
2. Pick one non-negative integer **gating threshold** `theta_d` for every domain `d`.

A domain is a shared power rail: its leakage RATE is `L` times its own **size** (the number of
blocks assigned to it) — bundling more blocks onto one rail makes every timestep it is live
proportionally pricier, no matter how many of those blocks are actually working. A domain must
be ON (drawing its full rate) at any timestep when at least one member is active in that trace
(a domain's "demand" is the OR of its members). Between active periods a domain accumulates an
idle run of length `Lr`. The rule is fixed: if `Lr >= theta_d`, the domain is powered OFF for
that run, paying a one-time **wakeup energy `W`** (independent of size) before its next active
period — or nothing if the run runs off the end of the trace. If `Lr < theta_d`, the domain is
kept needlessly ON through the whole run, still paying its full rate.

**MINIMIZE** total energy over every domain and every one of the `K` traces:
```
rate(d) = L * (blocks assigned to domain d)
energy  = sum over (domain d, trace, timestep) of rate(d) while ON, where an idle run of
          length Lr counts as ON for its whole length UNLESS gated (Lr>=theta_d), in which
          case it costs a flat W instead (0 if it is the trace's final run).
```
Finer domains expose more idle time to gate, and each rail is cheaper per ON step — but every
gate triggered costs the flat `W`, so gating a short idle run can cost more than it saves, and
a rail carrying MANY blocks should gate more eagerly than one carrying few. Sizing thresholds,
and choosing which blocks may safely share a rail, against the *actual idle-run-length
distribution* — not a fixed instinct — is the whole game.

## Input (stdin)
```
N D K T
L W
<K*N activity strings, each of length T, chars '0'/'1'>
```
The `K*N` strings are grouped by trace (trace 1's `N` blocks, then trace 2's, ...).

## Output (stdout)
```
Du
dom_1 dom_2 ... dom_N
theta_1 theta_2 ... theta_Du
```
`Du` domains actually used (`1 <= Du <= D`). `dom_i` is the domain id (`1..Du`) assigned to
block `i`. `theta_j` is the gating threshold for domain `j`.

## Feasibility
Rejected (score 0) if: any token is missing / non-integer / out of the declared bounds;
`Du` is outside `[1,D]`; any `dom_i` is outside `[1,Du]`; or any `theta_j` is outside
`[0, 1000000]`.

## Scoring
Let `F` = the total energy above, replayed by the checker for your `(partition, thresholds)`
against every one of the `K` traces. Let `B = L*N*T*K` — the energy of the checker's own
trivial reference (one domain holding all `N` blocks, rate `L*N`, held ON the whole horizon
on every trace, never gated). `B` is a fixed constant, independent of the trace content. Score:
```
ratio = min(1.0, 0.1 * B / max(1e-9, F))
```
Lower `F` (relative to `B`) scores higher; "just leave it all on" scores exactly 0.1.

## Example (worked, not to scale with the real tests)
One domain, one block (rate `L=10`), `W=50` (breakeven idle length `50/10=5`: gating pays off
once a run reaches length `6`). Trace, `T=16`: `0000000` `111` `00` `1111` (idle 7, active 3,
idle 2, active 4 — neither idle run is trailing):
- threshold `0` (gate every idle run): active `70` + gates `50+50=100` = `170`.
- threshold `1000000` (never gate): active `70` + idle kept on `70+20=90` = `160`.
- threshold `6` (gate runs `>=6` only): active `70` + length-7 gates (`50`) + length-2 stays
  on (`20`) = `140`.
Reading the run's *length* against the breakeven point (`140`) beats both "always gate" (`170`,
worse than never gating!) and "never gate" (`160`).

## Constraints
`1 <= N <= 40`, `1 <= D <= 10`, `1 <= K <= 10`, `1 <= T <= 200`, `1 <= L <= 1000`,
`0 <= W <= 1000`. Time limit 5s, memory 512MB.
