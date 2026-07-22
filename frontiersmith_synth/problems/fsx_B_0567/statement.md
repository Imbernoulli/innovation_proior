# Twin-Store Energy Arbitrage: Battery vs. Hydrogen Vault

## Problem
You arbitrage electricity across `T` hourly slots with two energy stores that lose energy in
fundamentally different ways.

- **Battery** — a *proportional* leak: energy sitting in it is multiplied by `rB` (`0<rB<1`)
  every hour. Its round trip is otherwise loss-free.
- **Hydrogen vault** — *no* leak (energy keeps forever), but every unit that leaves it is
  scaled by a fixed conversion efficiency `etaV` (`0<etaV<1`), a one-time tax paid regardless
  of how long the energy was stored.

You submit a charge/discharge schedule for both stores; the market simulator replays it and
scores your terminal cash.

## Input (stdin)
```
T rB etaV powB powV capB capV
p[0]      p[1]      ... p[T-1]
buy[0]    buy[1]    ... buy[T-1]
sell[0]   sell[1]   ... sell[T-1]
```
`p[t]` is the integer price at hour `t`. `powB,powV` are per-hour power caps; `capB,capV` are
storage capacities (all positive integers); `rB,etaV` are decimals. `buy[t]` and `sell[t]` are
0/1 flags: you may only **charge** a store (buy from the grid) at an hour with `buy[t]=1`, and
only **discharge** it (sell to the grid) at an hour with `sell[t]=1`. Every hour is one or the
other or neither; the two flags are never both 1.

## Output (stdout)
`T` lines, line `t` holds two real numbers `uB[t] uV[t]` — the signed action for the battery
and the vault at hour `t`. Positive = **charge** (buy `u` units from the grid into the store);
negative = **discharge** (remove `|u|` units and sell to the grid).

## Simulation & Feasibility
Start with empty stores and zero cash. For each hour `t`, in order:
1. Battery leaks: `sB <- sB * rB` (the vault does not leak).
2. Apply `uB[t]`: `sB <- sB + uB[t]`;  `cash <- cash - uB[t]*p[t]`.
3. Apply `uV[t]`: `sV <- sV + uV[t]`; if `uV[t] >= 0` then `cash <- cash - uV[t]*p[t]`,
   else `cash <- cash + etaV*(-uV[t])*p[t]` (a discharge earns the *taxed* price).

An output is **infeasible** (`Ratio: 0.0`) if any hold: the token count is not `2T`; any value
is non-finite; `|uB[t]|>powB` or `|uV[t]|>powV`; a charge (`u>0`) occurs where `buy[t]=0` or a
discharge (`u<0`) where `sell[t]=0`; or after any step `sB<0`, `sB>capB`, `sV<0`, or `sV>capV`
(tolerance `1e-6`). Energy left stored at the end is worth nothing.

## Objective
**Maximize** `F =` terminal cash.

## Scoring
Let `B` be the checker's internal baseline: the profit of the single best **battery-only**
round trip — over all `t1<t2`, buy `powB` units at `t1`, hold, and sell what remains
(`powB * rB^(t2-t1)`) at `t2`, taking the most profitable such pair. With maximization
normalization:
```
sc    = min(1000.0, 100.0 * max(0,F) / max(1e-9, B))
Ratio = sc / 1000.0
```
Reproducing that single best battery trade scores `Ratio ≈ 0.1`; ten times its profit caps at `1.0`.

## The tension
The battery has the cheaper round trip, so ranking media by efficiency sends every parcel to
it. But the battery's loss **compounds** with hold time (`rB^d`) while the vault's tax is a
**fixed** `etaV` no matter the duration. There is a crossover horizon `d* = ln(etaV)/ln(rB)`:
parcels held shorter than `d*` are cheaper in the battery, parcels held longer are cheaper in
the vault. The prices contain profitable trades whose spread only pays off over long holds,
where the battery leaks the gain away — those belong in the vault. The winning schedule
segments trades by intended hold duration instead of by any single "which store is better" rank.

## Constraints
`T ≤ 400`; `0.9 ≤ rB < 1`; `0.8 ≤ etaV < 1`; prices are positive integers.
Time limit 5s, memory 512m.

## Example
With `rB=0.98`, `etaV=0.85`, so `d*≈8.0`. A parcel bought at 900 and sold 4 hours later at
1080: the battery keeps `0.98^4·1080≈997`, netting `+97/unit`; the vault would net
`0.85·1080−900≈+18`, so this short hold belongs in the battery. Held 40 hours instead against a
sell price of 1200: the battery keeps `0.98^40·1200≈535` (a `−365/unit` loss) while the vault
nets `0.85·1200−900=+120`. Same two stores, opposite routing — decided purely by hold duration.
