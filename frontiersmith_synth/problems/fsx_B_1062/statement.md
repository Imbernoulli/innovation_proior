# Relay Throughput Spacing

## Problem
You are deploying wireless relays for `m` independent source/destination pairs that share
one deployment area. Pair `i` must move data from `S_i` to `D_i`, either directly or through
a chain of relay nodes you place. You have a shared budget of `R` relay nodes total, split
however you like across the `m` pairs.

All radios (sources, relays, destinations) transmit with the same power `P`. A transmission
over distance `d` arrives with power `P / (1+d)^alpha` (`alpha` = path-loss exponent). Routing
uses a synchronized round schedule: in round `r`, the `r`-th hop of **every** pair whose path
has at least `r` hops transmits simultaneously. All of that round's simultaneous transmitters
interfere with each other's receivers — a receiver hearing its intended transmitter also hears
every other pair's round-`r` transmitter, attenuated the same way. So the SINR at a round-`r`
receiver of pair `i` is
```
SINR = signal / (N0 + sum of P/(1+dist(other_tx, this_rx))^alpha over every OTHER active pair)
```
(`N0` = noise floor). The rate of that hop is `log2(1+SINR)` bits/s/Hz. A pair's end-to-end
throughput is the **bottleneck**: the minimum hop rate along its own path (the classic
weakest-link multi-hop capacity). Because `log2(1+SINR)` is concave, shortening hops always
raises SINR but with ever-smaller marginal reward — while every extra hop is one more
simultaneous transmitter that raises interference for whoever else is active that round. Your
job: place relays and pick each pair's route (an ordered sequence of relay positions from
`S_i` to `D_i`) to maximize the **worst pair's** throughput.

## Input (stdin)
```
m R P alpha N0 Xmax Ymax
sx_1 sy_1 dx_1 dy_1
...
sx_m sy_m dx_m dy_m
```
`m` pairs, integer relay budget `R`, transmit power `P`, path-loss exponent `alpha`, noise
floor `N0`, and a deployment box `[0,Xmax] x [0,Ymax]` containing every `S_i`, `D_i`.

## Output (stdout)
`m` groups of whitespace-separated tokens, in input order (the checker reads the whole stream
as one token sequence, so exact line breaks do not matter — one group per line is simplest but
not required). Group `i`: an integer `k_i >= 0` (relays used for pair `i`), followed by `k_i`
coordinate pairs `x y` giving that pair's relay chain **in order from `S_i` to `D_i`** (do not
repeat `S_i`/`D_i` themselves):
```
k_i x_1 y_1 x_2 y_2 ... x_{k_i} y_{k_i}
```
`sum(k_i) <= R`.

## Feasibility
- All `m` groups parse in order (the token stream must not run out early); every `k_i >= 0`;
  all numbers finite.
- Every relay coordinate lies in `[0,Xmax] x [0,Ymax]` (tolerance `1e-6`).
- `sum(k_i) <= R`.
Any violation scores `Ratio: 0.0`.

## Objective (maximize)
For each pair build its path `S_i -> relay_1 -> ... -> D_i`. Run the round-synchronized
schedule above to get every hop's rate, take each pair's minimum hop rate, then
`F = min` of that over all `m` pairs.

## Scoring
`B` is the checker's own fixed baseline: **every** pair gets exactly one relay, parked at its
own straight-line midpoint (ignores interference and hop-count trade-offs; `R >= m` always
holds, so this is always feasible). With your feasible `F`:
```
sc    = min(1000, 100 * F / max(1e-9, B))
Ratio = sc / 1000
```
so the one-relay-per-pair baseline scores about `0.1`, and ten-times-better caps at `1.0`.

## Constraints
`2 <= m <= 8`, `2 <= R <= 40`, `6 <= P <= 12`, `2.0 <= alpha <= 2.15`, `0.12 <= N0 <= 0.2`,
`Xmax = Ymax = 60`. Runs in well under the time limit for these sizes.

## Example
*(Illustrative FORM only — smaller than any real test, just to show the arithmetic.)*
`m=2`, `R=3`, `P=12`, `alpha=2.0`, `N0=0.2`. Pair 1 (hard, long): `S=(0,0) D=(20,0)`. Pair 2
(easy, short): `S=(0,4) D=(6,4)`.

Baseline (1 relay each, at `(10,0)` and `(3,4)`): pair 1's bottleneck is `~0.4287`, pair 2's is
`~1.2667`, so `B = 0.4287`.

Spending the 3rd relay on the *harder* pair (2 relays for pair 1, 1 for pair 2) instead of
splitting evenly raises pair 1's bottleneck to `~0.6542` (pair 2 stays above it), so
`F = 0.6542` and `Ratio = min(1000, 100*0.6542/0.4287)/1000 ~= 0.153`. This beats both giving
the odd relay to the already-easy pair and chaining every pair to the shortest possible hops.
