# Orchard Ledger: Splitting Rows for Equal-Ripeness Inspectors

## Problem
A prime `p` is fixed. The orchard's trees are the nonzero residues
`1, 2, ..., n` where `n = p - 1`, numbered by their position along the row.
You must split all `n` trees into `k` orchard **plots of exactly equal size**
`n / k` each (every tree assigned to exactly one plot).

Two families of inspectors audit your plots for "equal ripeness" balance,
and they use completely different accounting rules -- a plot that looks
balanced to one family can look wildly lopsided to the other.

**Row inspectors** look at a window: a contiguous run of `w` consecutive
tree positions `t, t+1, ..., t+w-1` (inside `[1,n]`). For each plot, the
inspector compares how many of that plot's trees fall inside the window
against the proportional share the plot "should" get (the window length
times the plot's fraction of all trees).

**Graft inspectors** use the orchard's OTHER structure. Fix the smallest
primitive root `g` of `p` (the smallest integer whose powers
`g^0, g^1, ..., g^{n-1} mod p` list every one of `1..n` exactly once). Every
tree `x` therefore has a unique GRAFT INDEX `idx(x)` in `[0, n-1]` with
`x = g^idx(x) mod p`. A graft inspector is given a small integer `d`; for
each plot it buckets that plot's trees by `idx(x) mod d` and compares the
count landing in each of the `d` buckets against the proportional share.

Row inspectors see the orchard laid out by POSITION; graft inspectors see it
laid out by GRAFT INDEX -- an unrelated, multiplicatively-defined order.

## Input (stdin)
```
p k
s_1 s_2 ... s_k
m_row
t_1 w_1
...
t_{m_row} w_{m_row}
m_graft
d_1
...
d_{m_graft}
```
`p` is prime and `n = p-1` is divisible by `k`; every `s_i = n/k`. Each row
window satisfies `1 <= t_j` and `t_j + w_j - 1 <= n`. Each graft divisor
satisfies `2 <= d_j <= n`.

## Output (stdout)
Print `n` integers `c_1 c_2 ... c_n` (any whitespace), where `c_x` is the
plot number (`1..k`) assigned to tree `x`, for `x = 1, 2, ..., n` in order.

## Feasibility
Valid iff **all** hold: exactly `n` tokens are printed, each parses as an
integer in `[1, k]`, and for every plot `i` the number of trees assigned to
it equals `s_i` **exactly**. Any violation scores `Ratio: 0.0`.

## Objective (minimize)
For every inspector (row or graft), compute its WORST-PLOT imbalance: the
largest, over all `k` plots, of `|actual count - proportional share|`. Your
raw score `F` is the MEAN of these worst-plot imbalances taken over all
`m_row + m_graft` inspectors. Lower `F` is better -- ignoring either
inspector family entirely leaves that family's imbalance to dominate the
mean.

## Scoring
The checker also builds its own simple reference partition (consecutive
positions cut into `k` blocks of size `s_i`) and computes that partition's
own `F`, calling it `B`. Your score is
```
Ratio = min(1000, 100 * B / max(1e-9, F)) / 1000
```
Matching the reference exactly gives `Ratio: 0.100000`; beating it
substantially pushes the ratio well above that, though it is capped at
`1.0` and real solutions should not reach the cap.

## Constraints
`73 <= p <= 8101`, `4 <= k <= 30`, `1 <= m_row <= 3`, `2 <= m_graft <= 4`.
Time limit 5s, memory 512MB.

## Example
Toy case (not a real test): `p=5` (`n=4`, trees `{1,2,3,4}`), `k=2` plots of
size 2, `g=2` (`idx(1)=0, idx(2)=1, idx(4)=2, idx(3)=3`), row window
`t=1,w=2`, graft test `d=2`. Reference `{1,2},{3,4}`: row counts `(2,0)` vs
expected `(1,1)` -- deviation `1`; graft buckets for `{1,2}` are
`idx={0,1}` -> counts `(1,1)` vs expected `(1,1)` -- deviation `0`. So
`F=(1+0)/2=0.5=B`, `Ratio: 0.100000` (it IS the reference). Partition
`{1,3},{2,4}`: row counts `(1,1)` vs `(1,1)` -- deviation `0`; graft
buckets also `(1,1)` vs `(1,1)` for both plots (`{1,3}` has `idx={0,3}`,
`{2,4}` has `idx={1,2}`, one even/odd each) -- deviation `0`. So `F=0`,
`Ratio: 1.000000` -- this tiny instance happens to admit a perfect split.
Real instances never reach `F=0` everywhere at once; row and graft pull in
different directions, and a good split must serve both.
