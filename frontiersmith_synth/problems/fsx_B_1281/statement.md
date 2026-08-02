# Additionality Desk: Buying Real Tonnes Under a Budget

## Problem
You run a carbon-offset purchasing desk with a fixed dollar budget. A broker offers `n`
candidate projects. Each project `j` is described by 9 integers:

```
price  claimed_tonnes  reported_baseline  reference_baseline  irr_no_carbon  irr_threshold  reversal_risk  perm_years  buffer_pct
```

- `price` — dollars charged per claimed tonne; `cost_j = price_j * claimed_tonnes_j`.
- `claimed_tonnes` — the nominal reduction the seller markets.
- `reported_baseline` / `reference_baseline` — the project's own counterfactual emission-rate
  claim vs. an independent regional reference rate (same units). When `reported` is inflated
  well above `reference`, the project is overstating how bad "no intervention" would have been,
  so most of its claimed reduction is fictitious.
- `irr_no_carbon` / `irr_threshold` — the project's estimated financial return *without* carbon
  revenue, and the return needed to go ahead anyway. If `irr_no_carbon >= irr_threshold` the
  project would likely happen regardless of your purchase (not additional).
- `reversal_risk` — annual probability (basis points, out of 10000) that the reduction is undone
  (fire, harvest, policy change) during the commitment period.
- `perm_years` — the commitment/monitoring horizon in years.
- `buffer_pct` — percent of losses absorbed by an insurance buffer pool if reversal occurs.

Every project's **true, verifiable, permanent reduction** is
```
inflation_penalty = min(1, reference_baseline / reported_baseline)
fin_additionality  = clip((irr_threshold - irr_no_carbon) / 1000, 0, 1)
additionality      = min(inflation_penalty, fin_additionality)
survival           = (1 - reversal_risk/10000) ** perm_years
permanence         = 1 - (1 - buffer_pct/100) * (1 - survival)
effective_tonnes_j = claimed_tonnes_j * additionality * permanence
```
`additionality` takes the *worse* of the two independent red flags (baseline gaming and the
would-happen-anyway test) — either one alone is enough to discredit a project.

## Input (stdin)
```
n budget
price_1 claimed_1 reported_1 reference_1 irr_no_carbon_1 irr_threshold_1 reversal_1 perm_years_1 buffer_1
...
price_n claimed_n reported_n reference_n irr_no_carbon_n irr_threshold_n reversal_n perm_years_n buffer_n
```
All values are integers. `1 <= n <= 45`. `budget` and every `cost_j` fit in 32-bit range.

## Output (stdout)
```
k
i_1 i_2 ... i_k
```
`k` is how many projects you buy; the second line lists their 1-based indices (any order).

## Feasibility
- `0 <= k <= n`; every `i_t` in `[1,n]`, pairwise distinct.
- `sum(cost_{i_t}) <= budget`.
Any violation, or non-finite output, scores `Ratio: 0.0`.

## Objective
Maximize `F = sum(effective_tonnes_{i_t})` over the purchased set.

## Scoring
The checker builds its own trivial reference portfolio `B`: look only at the first `min(n,6)`
projects **in the order they are listed** in the input (no quality signal used at all) and
first-fit them into the budget; `B` is that portfolio's own `F`. Then
```
sc = min(1000, 100 * F / max(1e-9, B))
Ratio = sc / 1000
```
Reproducing the trivial portfolio scores `0.1`; a portfolio `10x` more true tonnes per
dollar caps at `1.0`.

## Constraints
Time limit 5s, memory 512m. `2 <= n <= 45`; each `.in` well under 5 MB.

## Example
Project A = `(price=5, tonnes=1000, reported=800, reference=100, irr_no_carbon=1400,
irr_threshold=1000, reversal=900, perm=5, buffer=0)` — illustrative FORM only, not a real
test case. Its `inflation_penalty = min(1, 100/800) = 0.125` (baseline inflated 8x) and its
`fin_additionality = clip((1000-1400)/1000,0,1) = 0` (would happen anyway), so
`additionality = min(0.125, 0) = 0`: it is worth **0** true tonnes no matter how many are
"claimed" or how cheap. A smaller, pricier project with an honest baseline and
carbon-dependent financing is genuinely worth its price.
