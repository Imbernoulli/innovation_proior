# Prepositioned Doses: Hedging Against the Worst Seed City

## Problem
A public-health office manages `N` cities connected by a mobility graph (edges carry a
contact-rate weight). Health planners must **preposition** `S` vaccine doses across the
cities *before* anyone is sick, in one shot, once. Afterwards nature picks one of `K`
possible outbreak scenarios to actually happen; each scenario names a seed city and a
seasonal contact-rate multiplier for that outbreak. You do not get to react after the
seed is revealed — your allocation is scored against every scenario and only the worst
one counts.

Given city `i` with population `P_i`, dosing it costs `cost_i = ceil(P_i * alpha / 100)`
doses to bring it to its MAXIMUM protection level `0.5`, where `alpha` (percent) is
given in the input (doses beyond `cost_i` are wasted). Partial dosing gives partial
protection: with `d_i` doses at city `i`, `p_i = min(0.5, d_i / cost_i)` — protection
reduces but never fully eliminates transmission risk, and it is capped low enough that
fully dosing the seed city alone can never make *its own* local outbreak fizzle out
(`beta_local*(1-0.5) > gamma`): the only way to shrink the worst case is to stop a
scenario's outbreak from cascading through the network, not to out-vaccinate any one
city. A protection level of `p_i` multiplies city `i`'s transmission by `(1 - p_i)`
**both as a susceptible target and as an infectious source** — a heavily-protected city
is a *throttle* in the mobility graph for however that scenario's spread would
otherwise route through it.

For each scenario `(seed, beta)`, infection spreads over a fixed number of daily steps
`T` under a discrete-time metapopulation SIR process (all cities start fully
susceptible except one infected individual in `seed`):
```
foi_i(t)   = beta_local*(1-p_i)*I_i(t)  +  beta * sum_{(i,j,w) in edges} (w/1000)*(1-p_i)*(1-p_j)*I_j(t)
new_inf_i  = min(S_i(t), foi_i(t))
S_i(t+1)   = S_i(t) - new_inf_i
I_i(t+1)   = I_i(t) + new_inf_i - gamma*I_i(t)
```
(`S_i, I_i` are population *fractions*; `beta_local = 0.45`, `gamma = 0.18`, fixed
constants.) The scenario's damage is `total_infected = sum_i P_i*(1 - S_i(T))`, the
total number of people who were ever infected by day `T`.

## Input (stdin)
```
N K T alpha budget
P_1
...
P_N
M
u_1 v_1 w_1
...
u_M v_M w_M
seed_1 beta_1
...
seed_K beta_K
```
`0 <= u,v < N`, edges are undirected, `w` is a per-mille contact weight. `beta` is given
as a percent (e.g. `110` means multiplier `1.10`). `budget` is the total dose count `S`.

## Output (stdout)
Exactly `N` whitespace-separated non-negative integers `d_1 ... d_N`: doses prepositioned
at each city, in city order.

## Feasibility
- Exactly `N` integer tokens, each `>= 0`.
- `sum(d_i) <= budget`.
Any violation scores `Ratio: 0.0`.

## Objective
Minimize `F`, the **worst-case** `total_infected` over all `K` scenarios (using your
allocation's `p_i` values throughout the simulation described above).

## Scoring
Let `B` be the checker's own baseline: `F` computed with **zero doses everywhere** (the
unmitigated worst outbreak). Minimization normalization:
```
sc = min(1000.0, 100.0 * B / max(1e-9, F))
Ratio = sc / 1000.0
```
Spending doses can only reduce `F` below `B`, so `Ratio >= 0.1`; achieving `F = B/10` or
lower saturates the cap.

## Example (worked, illustrative shape only)
3 cities, `P = [1000, 1000, 20]`, one edge `(0,2,500)`, one edge `(2,1,500)`, `alpha=20`
(so `cost = [200, 200, 4]`), `budget = 4`, one scenario seeded at city 0. Dosing the
*bridge* city 2 to its cap (`d = [0,0,4]`) costs only 4 doses and throttles the only
route from city 0 to city 1 down to 50% strength, keeping most of the outbreak
contained to city 0 — far cheaper than dosing city 0 or city 1 directly (200 doses
each, more than the entire budget), and cheaper than trying to out-vaccinate city 0's
own local outbreak (which no amount of dosing city 0 alone can fully extinguish).

## Constraints
`4 <= N <= 40`, `3 <= K <= 6`, `10 <= T <= 35`, `1 <= P_i <= 5000`,
`1 <= w <= 300`, `10 <= alpha <= 20`, `70 <= beta <= 140`, time limit 5s, memory 512MB.
