# Allocating Risk, Not Capital

A book holds **N capital sleeves**. Each sleeve has a **calm-regime** return
distribution (moderate volatility, low cross-sleeve correlation) and a
**stress-regime** return distribution that is *much* more volatile AND much more
correlated with a common shock -- and the sleeves that already look the most
volatile in calm times are exactly the ones whose stress-regime correlation to
that shock is highest. You are shown a large sample of calm-regime scenarios and
only a **small** sample of stress-regime scenarios (real crisis history is
scarce). You must choose capital weights before knowing which regime the future
holds.

## Input (stdin)

```
testId
N
cap_1 ... cap_N
group_cap
K  cluster_1 ... cluster_K      (0-indexed sleeve ids of the high-calm-vol cluster)
C_CALM
C_CALM lines, each N floats     (calm-regime scenario returns per sleeve)
C_STRESS
C_STRESS lines, each N floats   (stress-regime scenario returns per sleeve)
```

`cap_i` is the maximum weight sleeve `i` may receive. `group_cap` bounds the
**total** weight placed in the cluster sleeves combined -- a hard concentration
limit on the sleeves most exposed to the common stress shock.

## Output (stdout)

Exactly `N` floats `w_1 ... w_N`: the capital weight of each sleeve (one line,
or any whitespace layout).

## Feasibility

- exactly N finite tokens; all `w_i >= 0`;
- `w_i <= cap_i` for every sleeve (tolerance 1e-6);
- `sum(w_i) = 1` (tolerance 1e-4);
- `sum(w_i for i in cluster) <= group_cap` (tolerance 1e-6).
Any violation scores `Ratio: 0.0`.

## Objective (maximize)

The checker draws a fresh, larger, **held-out** sample from the *same* calm and
stress regimes (never shown to you) and scores

```
F(w) = mean(held-out calm return of w)  /  CVaR_alpha(held-out stress loss of w)
```

i.e. the return your weights earn in normal times **per unit of tail loss** they
realize in a genuine stress draw. `CVaR_alpha` is the mean loss over the worst
`alpha` fraction of held-out stress scenarios **ranked by your own portfolio's
loss** -- so which scenarios count as "tail" depends on how *your* sleeves
co-move under stress, not on any one sleeve in isolation. This is where the
stress-regime correlation matters: two portfolios with the same total exposure
can realize very different tail losses depending on whether that exposure sits
in sleeves that crash together or not.

## Scoring

The checker also builds an internal baseline: chase the single highest-calm-
volatility sleeve up to its cap, spill into the rest of its cluster (subject to
`group_cap`), then fill the remaining book by capacity. Call its objective `B`.
Your ratio is `min(1, 0.23 * F / B)` (so `B` itself scores 0.23, and driving `F`
comfortably above the naive baseline caps the score at 1.0). Scores are
bit-for-bit deterministic.

## Constraints

- 4 <= N <= 8, `0.10 <= cap_i <= 0.70`, `group_cap = 0.45`.
- `C_CALM = 2000`, `C_STRESS = 30` (the stress sample is deliberately much smaller
  and noisier than the calm sample -- you must estimate the stress joint
  structure from it, not read it off).
- Time limit 5 s, memory 512 MB.

## Example (illustrative I/O form only, N=2 toy -- not a real test)

Input excerpt:
```
1
2
0.60 0.60
0.60
1 1
2
0.0100 0.0050
-0.0080 0.0300
2
-0.1500 -0.4000
-0.2000 -0.3500
```

Output:
```
0.5 0.5
```

On the real tests, equal-capital allocation looks diversified (every sleeve
gets the same slice) but ends up with just as much weight in the
high-calm-volatility cluster as a naive momentum strategy that chases the best
calm-period track record -- and both realize far more tail loss per unit of
return than an allocation that equalizes each sleeve's marginal contribution to
the stress-regime CVaR.
