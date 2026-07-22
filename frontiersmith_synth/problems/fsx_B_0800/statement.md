# Lagoon Quotas: Pulsing Through the Depensation Dip

## Problem
A lagoon fishery co-op sets a weekly catch quota `Q_t` for `T` weeks (five years). Stock
biomass `S_t` follows a **depensation** (Allee-effect) recruitment curve: growth is
*negative* when the stock is below a threshold `A` (population too sparse to breed
efficiently), positive between `A` and carrying capacity `K`, and negative again above `K`.
Weeks `t = dstart .. dend-1` are a printed **low-recruitment regime window** (e.g. a
multi-month drought) in which the growth rate is scaled down by `drought_mult < 1`.

Each week, before harvest, the stock grows:
`S'_t = S_t + r_t * S_t * (S_t/A - 1) * (1 - S_t/K)`, where `r_t = r_base` normally and
`r_t = r_base * drought_mult` during the regime window.

**Effort race.** `N` independent fishers each have a cost `c_i` and a weekly catch
capacity `cap_i`. Fisher `i` fishes this week iff the expected value density
`p_t * (S'_t / K) > c_i` (price times post-growth stock fraction), where `p_t = p_base *
price_season[t mod 52]` is a seasonal price. If it fishes, it attempts `cap_i`. The
realized catch is `H_t = min(Q_t, attempted_t, S'_t)` — the quota only binds if it is the
tightest constraint; a quota the fishers cannot profitably reach goes unused.
`S_{t+1} = S'_t - H_t`.

**Seasonal quota schedule.** Weeks whose `t mod 52` lies in a printed set `closed_mod`
(a recurring legal spawning closure) MUST have `Q_t = 0`.

## Input (stdin)
```
T
K A Acol gamma p_base Qcap
S0 r_base
dstart dend drought_mult
n_closed
<n_closed ints: closed_mod values, each in [0,52)>
52
<52 floats: price_season[0..51]>
N
<N lines: c_i cap_i>
```

## Output (stdout)
`T` lines; line `t+1` (for `t = 0..T-1`) is one real number `Q_t >= 0`.

## Feasibility
All must hold, checked by replaying the exact recurrence above:
- every `Q_t` is finite and in `[0, Qcap]`;
- `Q_t = 0` whenever `t mod 52` is in `closed_mod`;
- the post-harvest stock `S_{t+1}` never drops below the collapse threshold `Acol` at
  **any** week — a single breach folds the fishery and scores `Ratio: 0.0` for the whole
  submission (no partial credit for the weeks before the collapse).
Any violation scores `Ratio: 0.0`.

## Objective
Maximize the discounted revenue `F = sum_{t=0}^{T-1} gamma^t * p_t * H_t`.

## Scoring
The checker builds its own internal baseline `B`: a flat constant quota
`Qb = min(0.012*S0, Qcap)`, applied every legally open week for all `T` weeks (ignoring
the depensation dynamics and the regime window), replayed through the same recurrence.
```
sc = min(1000.0, 100.0 * F / max(1e-9, B))
Ratio = sc / 1000.0
```
Reproducing the baseline scores `Ratio ~ 0.1`; a strategy that banks `10x` the baseline's
discounted revenue caps at `Ratio = 1.0`.

## Constraints
- `T = 260`, `N = 300`, `2 <= n_closed <= 10`.
- `0 < Acol < A < K`; `S0 > A`; `0 < drought_mult < 1`; `0 < dstart < dend <= T`.
- Time limit 5s, memory 512MB. Each `.in` file is a few KB.

## Example (illustrative recurrence ONLY — not a full worked instance)
Suppose for one week `S_t = 400000`, `A = 250000`, `K = 1000000`, `r_t = 0.07`:
`growth = 0.07 * 400000 * (400000/250000 - 1) * (1 - 400000/1000000)
        = 0.07 * 400000 * 0.6 * 0.6 = 10080`, so `S'_t = 410080`. If `p_t = 10`, the value
density is `10 * 0.41008 = 4.1008`; every fisher with `c_i < 4.1008` fishes. Suppose their
summed capacity is `attempted_t = 60000` and `Q_t = 15000`: then `H_t = min(15000, 60000,
410080) = 15000`, contributing `10 * 15000 = 150000` (times the week's discount factor)
to `F`, and `S_{t+1} = 395080`. A submission that reproduces the checker's flat-quota
baseline for all 260 weeks scores `Ratio ~ 0.1`; a quota schedule that voluntarily closes
during the printed regime window and banks bursts near `Qcap` while the stock is safely
above `A` can push `F` well past `10x` the baseline.
