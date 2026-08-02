# Royalty Audit: Defensible Recovery Under a Sampling Budget

## Problem

A royalty/transaction population of `N` line items is split into `K = 3` strata
(e.g. marquee accounts, standard accounts, long-tail accounts). Every item has a
publicly known **reported value** and a publicly known **audit cost**. You are given
a total **audit budget** and, for each stratum, a **historical prior** estimate
(mean, standard deviation) of that stratum's true error rate — a rough guide, not a
guarantee. You must choose which items to actually audit, within budget.

Auditing an item reveals its real error dollars, but that information is only
public *to you* after you commit to your plan — the checker reveals it when scoring.
Your score is not "dollars you personally uncovered"; it is **defensible recoverable
value**: money you can either prove exactly (because you looked at everything) or
project onto the unaudited rest of a stratum because your sample of it was, by a
stated rule, a fair sample — not a hand-picked one.

**Illustrative FORM only** (not this problem's actual numbers): if a stratum has
$500,000 total reported value and your sample of it has a mean observed error rate
of 4%, a defensible sample projects to a $20,000 claim for that stratum; a
non-defensible sample of the very same items only banks a small fraction of the
dollars it literally touched.

## Input (stdin)

```
T N K Cmax
THRESH
prior_mean_rate[0] prior_stdev_rate[0]
...
prior_mean_rate[K-1] prior_stdev_rate[K-1]
id_1 stratum_1 reported_value_1 audit_cost_1
...
id_N stratum_N reported_value_N audit_cost_N
```
`T` is this test's index (ignore it). Strata are numbered `0..K-1`. `ids` are
`1..N`. `THRESH` is this test's extrapolation-defensibility threshold (see Scoring).

## Output (stdout)

The ids (any order, whitespace-separated) of the items you choose to audit.

## Feasibility

- Every token must parse as an integer in `[1,N]`; duplicates are rejected.
- The sum of `audit_cost` over your chosen ids must be `<= Cmax`.
- Any violation scores 0.

## Scoring

Fix `Z = 1.645`. For each stratum `h`, let `n` = number of your audited items in
`h`, `nb` = the stratum's population size:

- **`n == nb`** (you audited every item in the stratum): its claim is the *exact*
  true total error dollars for the whole stratum — certain, no extrapolation risk.
- **`n >= 2` and both of the following hold** on your audited items in `h`:
  - *Representative*: rank every item of stratum `h` by reported value; let each
    audited item's **percentile** be its rank divided by `(nb-1)`. The **mean**
    percentile of your audited items in `h` must lie in `[0.37, 0.63]` — i.e. your
    sample must be centered on the stratum's value range, not cherry-picked from
    one end.
  - *Precise*: let `mean_rate`/`sd` be the mean/sample-stdev (ddof=1) of the
    observed error *rate* (error dollars / reported value) on your audited items.
    The relative margin `RM = Z * sd / (sqrt(n) * mean_rate)` must be `<= THRESH`.

  Then the claim is `mean_rate * (stratum h's total reported value)`.
- **Otherwise** (n<2, or fails either check above): the claim is `0.32 *` (the sum
  of actual error dollars on the audited items in `h`) — real money you found, but
  not statistically defensible to project onto the rest of the stratum.

Your total score is the sum of claims across all `K` strata. **Maximize** it.
Auditing only the biggest-ticket items in a stratum finds real dollars but sits at
one end of the value range — it typically fails the representativeness check above,
so most of that stratum's true value is left unclaimed no matter how large the
audited transactions were.

## Constraints

`70 <= N <= 130`, `K = 3`, values/costs are positive integers `<= 200000`,
`0.30 <= THRESH <= 0.45`. Time limit: 5s. Each `.in` file `<= 5 MB`.
