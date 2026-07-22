# Casino Floor Router: Surviving the Cashier

## Problem
You are designing gambler-flow rules for a casino floor. The floor has `Nc` **corridor
pits** (quiet, low-key tables) arranged in a fixed one-way loop `0 -> 1 -> ... -> Nc-1 -> 0`,
and `K` **feature stations** (flashy jackpot clusters) arranged in their own one-way loop
`0 -> 1 -> ... -> K-1 -> 0`. A gambler currently at corridor pit `i` may, on the next step,
either continue the loop to pit `(i+1) mod Nc`, take a **shortcut door** to a fixed feature
station `shortcut[i]`, or head to the **cashier** (leaving for good). A gambler at feature
station `k` may take a **return door** back to a fixed corridor pit `hubret[k]`, continue the
feature loop to station `(k+1) mod K`, or head to the cashier. `shortcut[]` and `hubret[]` are
part of the input (fixed floor plan, not yours to choose).

Regulators impose per-door **minimum flow floors** (a door that exists must carry at least
its floor probability): global floors `f_ring, f_short, f_hret, f_hring` apply to the
loop-continue, shortcut, return, and feature-loop doors respectively; each corridor pit `i`
also has its own cashier floor `lo[i]` (small — quiet pits rarely force a cash-out), and each
feature station `k` has its own cashier floor `hi[k]` (large — big payouts trigger mandatory
cash-out prompts). You choose every door's exact probability (each row's three doors must
sum to exactly `1`, each `>=` its floor). This is a genuinely free design: a pit or station's
leftover probability, above the mandatory floors, may be routed to whichever of its non-cashier
doors you like.

## Input (stdin)
```
Nc K
f_ring f_short f_hret f_hring
lo[0] lo[1] ... lo[Nc-1]
hi[0] hi[1] ... hi[K-1]
shortcut[0] shortcut[1] ... shortcut[Nc-1]
hubret[0] hubret[1] ... hubret[K-1]
```

## Output (stdout)
Exactly `Nc + K` lines, each with 3 floats summing to `1`:
- Lines `1..Nc` (corridor pit `i`, 0-indexed): `p_loop p_short p_cash`
- Lines `Nc+1..Nc+K` (feature station `k`): `p_ret p_ring p_cash`

## Feasibility
All checks use tolerance `1e-4` on the row sum and `1e-6` on floors:
- Exactly `3*(Nc+K)` finite numbers, no `nan`/`inf`.
- Each of the three values on a corridor row is `>= f_ring`, `>= f_short`, `>= lo[i]`
  respectively; each on a feature row is `>= f_hret`, `>= f_hring`, `>= hi[k]` respectively.
- Every row sums to `1` (within tolerance).
Any violation scores `Ratio: 0.0`.

## Objective (maximize)
Drop the cashier column: the remaining `(Nc+K) x (Nc+K)` matrix `Q` (loop/shortcut/return/
feature-loop entries only) is **substochastic** — it governs how long a gambler survives on
the floor before being forced to the cashier. The **quasi-stationary survival rate** is the
Perron (spectral-radius) eigenvalue of `Q`: the long-run per-step probability of NOT having
cashed out yet, conditioned on still being on the floor. This is a property of `Q`'s **cycle
structure** (not a per-pit sum) — the checker computes it by a fixed, deterministic power
iteration (no randomness, no time limit dependence). Maximize this value.

## Scoring
Let `B` be the Perron value of the checker's own internal reference routing (a fixed, feasible,
positive construction: mostly shortcut/return doors, a small trickle continuing each loop).
With your feasible survival rate `F`:
```
sc    = min(1000, 100 * F / max(1e-9, B))
Ratio = sc / 1000
```
Reproducing the reference scores about `0.1`; ten times better caps at `1.0`.

## Constraints
`6 <= Nc <= 130`, `2 <= K <= 40`. All floors lie in `(0, 1)` with room to spare (a feasible
row always exists). Runs comfortably within the time limit.

## Example
`Nc=3, K=1`, floors `f_ring=0.03 f_short=0.01 f_hret=0.02 f_hring=0.02`, `lo=[0.05,0.05,0.05]`,
`hi=[0.85]`, `shortcut=[0,0,0]`, `hubret=[0]`. (Illustrative only — the graded cases use much
larger `Nc, K` where no single pit's behavior can be read off by eye.) One feasible output:
every corridor pit routes almost all its slack around the loop (`0.94 0.01 0.05` on each of
the 3 corridor rows), and the feature station — bound by `hi[0]=0.85`, so `p_ret+p_ring<=0.15`
— is given `0.13 0.02 0.85`. The interior matrix's dominant eigenvalue is then close to the
loop's own retained weight (`~0.94`), since that cycle never touches the high-cashier feature
station.
