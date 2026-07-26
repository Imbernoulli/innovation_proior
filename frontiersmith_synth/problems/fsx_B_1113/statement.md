# Merge-Fix Wake Batching: Two Runways, One Choke Point

## Problem
`N` aircraft are holding above an airport, each with an earliest **ready time**
`r_i` (the earliest tick it may begin its approach) and a **weight class**
`c_i in {H, M, L}` (Heavy / Medium / Light). Every aircraft must first cross a
single shared **approach fix** (all traffic funnels through one merge point
upstream of both runways) and then land on one of **two parallel runways**.

You must output, for every aircraft: which runway it uses, the tick it
crosses the fix, and the tick it lands.

### Constraints the schedule must satisfy
1. **Readiness**: an aircraft cannot cross the fix before it is ready:
   `fix_time_i >= r_i`.
2. **Transit**: landing follows the fix crossing by at least `D` ticks (fixed,
   given in the input): `landing_time_i >= fix_time_i + D`.
3. **Shared-fix separation**: because every aircraft funnels through the same
   merge point regardless of which runway it eventually uses, any two
   aircraft's fix crossings — sorted by time, over **all** `N` aircraft
   together — must be at least `FIXSEP` ticks apart. This is what couples the
   two runways: you cannot get unlimited combined throughput just by using
   both of them.
4. **Wake-vortex separation (per runway, sequence-dependent)**: on a single
   runway, consider the aircraft landing on it sorted by landing time. Between
   any two **consecutive** ones (previous class `a`, next class `b`) the gap
   must be at least `S[a][b]` ticks, where `S` is a 3x3 matrix given in the
   input. A Heavy leaving a strong wake in front of a trailing Light needs a
   much larger gap than two same-class landings back to back — `S` is NOT
   symmetric and NOT class-independent.

Any violation of 1-4 scores `Ratio: 0.0`.

## Input (stdin)
```
N
D FIXSEP
S[H][H] S[H][M] S[H][L]
S[M][H] S[M][M] S[M][L]
S[L][H] S[L][M] S[L][L]
fw[H] fw[M] fw[L]
r_1 c_1
...
r_N c_N
```
`S[a][b]` = minimum landing-gap (same runway) when class `a` lands
immediately before class `b`. `fw[c]` = fuel burned per tick a class-`c`
aircraft spends holding (from `r_i` until it lands). `r_i` is an integer,
`c_i` is one of the letters `H`, `M`, `L`.

## Output (stdout)
Exactly `N` lines (line `i` describes aircraft `i`, matching input order):
```
runway_i fix_time_i landing_time_i
```
`runway_i in {1, 2}`; all values integers.

## Objective
Minimize the total fuel burned holding in the stack:
```
F = sum_i fw[c_i] * (landing_time_i - r_i)
```
Every extra tick an aircraft waits (whether stuck behind the shared fix or
behind a wake-separation gap on its runway) costs fuel at its class's rate.

## Scoring
The checker builds its own feasible reference `B`: process aircraft strictly
in ready-time order (never reordering by class) and alternate them onto the
two runways round-robin (1, 2, 1, 2, ...), honoring the fix and wake
constraints but never choosing which runway is better. Given your feasible
`F`:
```
Ratio = min(1000, 100 * B / F) / 1000
```
so matching that arrival-order, blind-alternation reference scores `0.1`;
cutting total fuel to a tenth of it caps at `1.0`. There is no known
closed-form optimum — this is a coupled release-time / two-machine /
sequence-dependent scheduling problem.

## Constraints
`6 <= N <= 40`, `D, FIXSEP` small positive integers, `1 <= S[a][b] <= 20`,
`1 <= fw[c] <= 10`. Deterministic exact integer scoring; no randomness.

## Example
`N=4`, `D=1`, `FIXSEP=1`, every `S[a][b]=2`, `fw=[1,1,1]`, all four aircraft
ready at `r=0`, class `H`. The round-robin reference sends them fix times
`0,1,2,3` and runways `1,2,1,2`; runway 1's landings are `1,3` (its second
landing needs `S[H][H]=2` after the first) and runway 2's are `2,4`, giving
`B = (1-0)+(2-0)+(3-0)+(4-0) = 10`, so matching it scores `Ratio = 0.1`. The
real payoff of picking the *better* runway and of batching by class only
shows up once classes interleave and `N` grows — that is where the ladder's
harder cases live.
