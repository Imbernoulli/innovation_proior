# Renewal Repricing Against the Pool That Remains

## Problem
An insurer sells policies in `K` independent price tiers (auto, home, ...). Tier `i`
renews once this cycle at a single new price `p_i` for every policyholder currently in
that tier. Within tier `i` the book is **not homogeneous**: it is split into risk
*buckets* `j = 1..B_i`. Bucket `j` currently holds `n_{i,j}` policyholders and its future
claim size follows a small discrete **loss distribution** given as `(value, probability)`
pairs (probabilities are integers per-mille, summing to `1000`); its expected loss is
`E_{i,j} = sum(value * probability) / 1000`.

A named competitor quotes a fixed price `c_i` for tier `i`. Whenever your price `p_i`
exceeds `c_i`, some policyholders shop away -- but **not uniformly**: each bucket `j`
has its own departure thresholds `Tlo_{i,j} <= Thi_{i,j}`. Writing `gap = p_i - c_i`,
the fraction of bucket `j` that departs is
```
depart(gap) = 0                                   if gap <= Tlo
            = 1                                   if gap >= Thi
            = (gap - Tlo) / (Thi - Tlo)            otherwise
```
Cheap-to-insure buckets have LOW thresholds (they leave at a small gap); expensive
buckets have HIGH thresholds (they are stuck with you even at a large gap) -- so
raising price can shed exactly the customers you wanted to keep. Regulation also caps
how far `p_i` may move from last cycle's price `p0_i`: `p_i` must lie in
`[p0_i - floor(p0_i*cap_i/100), p0_i + floor(p0_i*cap_i/100)]`.

Realized underwriting profit for tier `i` at price `p_i`, using the buckets that
actually remain (i.e. the POST-selection pool), is
```
remaining_{i,j} = n_{i,j} * (1 - depart(p_i - c_i))
profit_i(p_i)   = p_i * sum_j remaining_{i,j}  -  sum_j remaining_{i,j} * E_{i,j}
```
Total objective `F = sum_i profit_i(p_i)`.

## Input (stdin)
```
K
p0_1 c_1 cap_1 B_1
n_{1,1} m_{1,1}
v_1 pr_1 v_2 pr_2 ... v_m pr_m
Tlo_{1,1} Thi_{1,1}
... (repeat for buckets 1..B_1 of tier 1, then tiers 2..K identically)
```
`3 <= K <= 6`; `3 <= B_i <= 5`; `2 <= m_{i,j} <= 3`; `1 <= n_{i,j} <= 3000`;
`0 <= v <= 500`; each bucket's `pr` values are positive integers summing to `1000`;
`1 <= p0_i, c_i <= 500`; `0 <= cap_i <= 100`; `0 <= Tlo_{i,j} <= Thi_{i,j} <= 5000`.

## Output (stdout)
`K` whitespace-separated integers `p_1 ... p_K`, the new price for each tier in input
order.

## Feasibility
For every tier `i`, `p_i` must be an integer with
`p0_i - floor(p0_i*cap_i/100) <= p_i <= p0_i + floor(p0_i*cap_i/100)`.
Any violation (wrong token count, non-integer token, out-of-band price) scores
`Ratio: 0.0`.

## Objective
Maximize total realized profit `F = sum_i profit_i(p_i)` as defined above, using the
buckets that actually remain at your chosen price -- not the buckets you started with.

## Scoring
Let `B` be the profit of the checker's own baseline: freezing every tier at last
cycle's price (`p_i = p0_i`), scored with the same post-selection formula. Then
```
sc = min(1000.0, max(0.0, 100.0 * F / max(1e-9, B)))
Ratio = sc / 1000.0
```
Matching the frozen baseline scores `0.1`; `10x` more profit caps at `1.0`.

## Constraints
Time limit 5s, memory 512MB.

## Example
One tier, `p0=100 c=90 cap=20`, two buckets: `(n=100, E=50, Tlo=15, Thi=25)` and
`(n=50, E=160, Tlo=30, Thi=60)`. Band: `[80, 120]`.
Baseline (`p=100`, gap `10`): both buckets fully retained (`10 <= 15` and `10 <= 30`),
so `B = 100*(100-50) + 50*(100-160) = 5000 - 3000 = 2000`.
Submitting `p=110` (gap `20`): bucket 1 departs `(20-15)/(25-15)=0.5`, leaving `50`;
bucket 2 fully retained, leaving `50`. `F = 110*100 - (50*50 + 50*160) = 11000 - 10500
= 500`. `sc = 100*500/2000 = 25`, so `Ratio = 0.025000`.
