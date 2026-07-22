# Corridor Rebates: Budgeted Subsidies for a Heat-Pump Cascade

## Problem
A village has `n` households. Household `i` will install a heat pump once its
**pressure** reaches its threshold `theta_i`. Pressure has two sources:

- **Rebate.** You may hand household `i` a whole-number rebate `x_i >= 0`. It contributes
  `x_i * r_i`, where `r_i` is `i`'s rebate responsiveness.
- **Peers.** Influence flows along `m` directed neighbor links `u -> v`: once an upstream
  neighbor `u` has adopted, it adds a fixed weight `W` to `v`'s pressure. A household with
  `k` already-adopted upstream neighbors gets `k * W`.

Adoption is a deterministic cascade run to a fixpoint: repeatedly, any not-yet-adopted
household `i` with `x_i * r_i + W * (adopted upstream neighbors) >= theta_i` adopts. This
process is monotone, so the final adopter set is unique and order-independent.

You have a total rebate budget `B`. Choose the rebates to maximize the number of adopters.

## Input (stdin)
```
n m B W
theta_1 r_1
...
theta_n r_n
u_1 v_1        (m directed edges u -> v, meaning u influences v)
...
u_m v_m
```

## Output (stdout)
`n` whole numbers `x_1 ... x_n` (whitespace-separated), the rebate per household.

## Feasibility
Valid iff there are exactly `n` values, every `x_i` is an integer with `x_i >= 0`, and
`sum_i x_i <= B`. Any violation scores `Ratio: 0.0`.

## Objective
Run the cascade from your rebates and maximize `F`, the number of adopters at the fixpoint.

## Scoring
Let `B0` be the number of adopters produced by the checker's own **uniform** construction:
give every household the same rebate `floor(B / n)`, then run the cascade. With
maximization normalization:
```
sc = min(1000.0, 100.0 * F / max(1e-9, B0))
Ratio = sc / 1000.0
```
Reproducing the uniform allocation scores `Ratio = 0.1`; ten times as many adopters caps
at `1.0`.

## What makes it hard
Spending a rebate to *self-activate* a household buys exactly one adopter. But a small
rebate given where an upstream neighbor has already adopted can push a household over its
threshold for a fraction of its self-cost — and that household then influences the next one
downstream. The marginal value of a rebate is the length of the cascade it continues, not
the recipient's own adoption. The obvious "convert the cheapest households first" rule
never funds these continuations and leaves most of the budget's reach on the table.

## Constraints
- `1 <= n <= 100000`, `0 <= m <= 2*n`, `1 <= W`, `1 <= theta_i`, `1 <= r_i`, `B >= 1`.
- All quantities are integers. Time limit 5s, memory 512m, each input <= 5 MB.

## Example
Two households, one link `1 -> 2`, `W = 100`, `theta = [150, 110]`, `r = [1, 1]`, `B = 160`.
Uniform gives each `floor(160/2) = 80`: neither reaches its threshold, so `B0 = 1` (clamped).
Instead spend `x = [150, 10]`: household 1 adopts (`150 >= 150`), then household 2 sees
`10 + 100 = 110 >= 110` and adopts. `F = 2`, `sc = 100*2/1 = 200`, `Ratio = 0.2`.
