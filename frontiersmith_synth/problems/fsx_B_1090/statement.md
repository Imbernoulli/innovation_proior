# Storm-Proof Battery Bids — scenario-consensus commitment plan

You operate a grid battery that must publish, once, a fixed week-ahead
commitment vector: how much energy to sell (discharge) or buy (charge) in
each hour. The market settles against **one** of `S` published future
scenarios — each scenario is a full hourly price curve plus a set of
**outage hours** in which the interconnection is down. Every scenario is
plausible; your grade is your profit in the **worst** scenario.

## Problem

Battery parameters: energy capacity `Emax`, power limit `Pmax`, charge
efficiency `eta`, initial charge `E0`, void penalty `lam`, shortfall penalty
`mu`. Each scenario `s` publishes its prices `p_s[t]`, outage flags
`o_s[t] ∈ {0,1}`, and a terminal valuation `rho_s` paid per unit of energy
still stored at the horizon.

You output committed grid exchanges `q[1..T]`: `q[t] > 0` sells `q[t]`
MWh (discharge), `q[t] < 0` buys `|q[t]|` MWh (charge, of which
`eta·|q[t]|` is stored).

## Feasibility (committed trajectory)

With `E[0] = E0`: discharging removes `q[t]`, charging adds `eta·|q[t]|`.
Require `|q[t]| ≤ Pmax` and `0 ≤ E[t] ≤ Emax` for every hour, assuming all
bids execute. Any violation scores 0.

## Settlement in scenario s

Hours are processed in order; `E` starts at `E0`.

- **Outage hour** (`o_s[t]=1`): the commitment is void — no energy moves,
  no revenue — and you pay the contract penalty `lam·|q[t]|`.
- **Discharge** (`q[t] > 0`): delivered `d = min(q[t], E)`; you earn
  `p_s[t]·d` and pay shortfall `mu·(q[t] − d)`; `E ← E − d`.
- **Charge** (`q[t] < 0`): accepted `a = min(|q[t]|, (Emax − E)/eta)`; you
  pay `p_s[t]·a` plus shortfall `mu·(|q[t]| − a)`; `E ← E + eta·a`.

After hour `T` the residual stored energy is sold at `rho_s` per unit.

Because earlier voids change the physical charge level, a plan that assumes
every bid executes can be clipped in some scenarios — shortfall penalties
apply to the undelivered part. Profit `F = min_s profit_s`.

## Scoring

`B` = the worst-scenario profit of the do-nothing plan (`min_s rho_s·E0`,
positive). Your ratio is `min(1, max(0, F / (10·B)))`. Doing nothing scores
exactly 0.1; reaching ten times the do-nothing value caps at 1.0.

## Input (stdin)

```
T S
Emax Pmax eta E0 lam mu
rho_1
p_1[1] ... p_1[T]
o_1[1] ... o_1[T]
... (S scenario blocks of 3 lines each)
```

## Output (stdout)

`T` whitespace-separated numbers: the committed exchanges `q[1..T]`.

## Constraints

- `T = 168`, `19 ≤ S ≤ 28`; all prices, `lam`, `mu`, `rho_s` positive;
  `0.88 ≤ eta ≤ 0.90`; `E0 = Emax/2`; `Pmax = Emax/4`.
- Time limit 5 s, memory 512 MB. Scoring is exact double arithmetic,
  fully deterministic.

## Worked example

`T=4, S=2`, `Emax=10, Pmax=4, eta=0.9, E0=5, lam=20, mu=25`.
Scenario A: `rho=18`, `p=[10,12,40,44]`, `o=[0,0,0,0]`.
Scenario B: same prices and `rho`, but `o=[0,0,1,1]` — the two peak hours
are void. `B = 18·5 = 90`.

*Peak-chaser* `q = [-4, 0, 3, 1]` (feasible: `E = 5→8.6→8.6→5.6→4.6`).
Scenario A: `−40 + 120 + 44 + 18·4.6 = 206.8`. Scenario B: the charge
settles (`−40`, `E=8.6`), both peak sales void (`−60 − 20`), residual
`18·8.6 = 154.8` → profit `34.8`. `F = 34.8`, ratio `34.8/900 = 0.0387` —
worse than doing nothing.

*Consensus* `q = [-4, 0, 0, 0]` commits only where no scenario voids:
both scenarios give `−40 + 18·8.6 = 114.8`, ratio `114.8/900 = 0.1276`.

The mean price curve never tells you which hours some scenario voids; the
worst scenario is where the score lives. Uncommitted stored energy keeps
its terminal value in every scenario — flexibility you did not sell cannot
be voided.
