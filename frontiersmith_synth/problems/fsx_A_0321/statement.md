# Vineyard Emitter Placement: Yield-Weighted Cap Set

## Problem
A terraced vineyard is addressed by an `n`-digit **ternary** code: every plot has an
address `d_1 d_2 ... d_n` with each `d_i` in `{0,1,2}` (three terrace levels along
each of `n` slope axes), so there are `3^n` plots in all. You install drip **emitters**
on a subset of plots to maximise total water delivery.

Two engineering rules constrain the layout:

1. **Rocky plots.** A given list of plots is bedrock and can never host an emitter.
2. **No resonant triple.** Water hammer builds up if three emitters sit on a common
   *ternary line*. Three distinct plots `x`, `y`, `z` are collinear exactly when, in
   every coordinate, `x_i + y_i + z_i \equiv 0 \pmod 3`. No three switched-on emitters
   may be collinear. (Equivalently: the set of chosen plots is a **cap set** in
   `F_3^n` — it contains no full line `{a, a+t, a+2t}`.)

Each plot has an integer **water yield** `w(plot) >= 1`. Choose which emitters to switch
on to maximise the sum of their yields, subject to the two rules.

## Input (stdin)
```
n
m
<m lines: each a rocky plot as an n-character ternary string>
w_0 w_1 w_2 ... w_{3^n - 1}
```
The `3^n` yields are listed in **canonical index order**: the plot with address
`d_1...d_n` has index `sum_i d_i * 3^(n-i)` (leftmost digit most significant). Yields
are whitespace-separated and may span several lines.

## Output (stdout)
The addresses of the plots you switch on, one `n`-character ternary string per line
(any order). Duplicates, rocky plots, malformed tokens, or any collinear triple make
the whole submission infeasible.

## Feasibility
The chosen set must: use only valid `n`-digit ternary addresses, contain no repeats,
avoid every rocky plot, and contain **no three collinear plots**. Any violation scores 0.

## Objective
Maximise `F = sum of yields of the chosen plots`.

## Scoring
Let `B` be the total yield of the internal trivial construction — the `{0,1}^n`
sub-cube (all addresses using only terraces `0` and `1`, minus any rocky plots), which
is always a valid cap set. The score is
```
sc    = min(1000, 100 * F / max(1e-9, B))
Ratio = sc / 1000
```
Reproducing the `{0,1}^n` set scores `Ratio ~= 0.10`; a `10x`-better layout caps at `1.0`.
There is no known closed-form optimum — larger dimensions (up to `n = 8`, `6561` plots)
are the evaluation regime where a good *priority ordering* matters most.

## Constraints
- `4 <= n <= 8`, so `81 <= 3^n <= 6561` plots.
- Number of rocky plots `m ~= 0.03 * 3^n`.
- `1 <= w(plot) <= 1000`. All arithmetic is exact integer / mod-3.

## Example (worked score)
Take a tiny `n = 2` vineyard (9 plots) with no rocky plots and unit yields.
`F_3^2` has maximal caps of size 4, e.g. `{00, 01, 10, 22}` — check no triple sums to
`0 (mod 3)` coordinate-wise. Its yield is `4`. The trivial `{0,1}^2 = {00,01,10,11}`
also has size 4 and yield `B = 4`, so this particular layout scores
`Ratio = 100*4/4 / 1000 = 0.10`. With non-uniform yields the two sets diverge: steering
the greedy toward the heaviest plots (while dodging resonant triples) is what pulls the
ratio above the baseline.
