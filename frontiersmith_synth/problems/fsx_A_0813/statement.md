# The Guild Ledger — Recovering an Exact Compounding Rule from Rounded Coins

## Problem
A medieval lenders' guild runs several accounts. Every day, each account's
balance (an exact integer number of coins — the guild never deals in
fractions) grows by a fixed but unknown daily rate `r`, and the result is
rounded **down** to the nearest whole coin:

```
b <- floor(b * (1 + r))
```

Additionally, the guild collects a maintenance fee on the **last day of every
30-day month** (day indices `0..29` are month 0, `30..59` are month 1, and so
on) — but *only* in months whose account balance was still below a hidden
poverty line `theta` coins on the **first day of that month** (since balance
only grows within a month except for this fee, the first day's balance is
also the month's minimum). The fee is a fixed integer number of coins,
deducted immediately after that day's growth is applied. On days that are not
a month's last day, or in months that clear the poverty line, no fee is
charged.

The guild's true daily rate `r` is an exact **rational number** `p/q` in
lowest or non-lowest terms (`0 <= p < q`), not a decimal — this matters
because `floor()` is *not* continuous in `r`: two rates that agree to six
decimal places can still make `floor(b*(1+r))` round differently on plenty of
days once `b` grows large, and those single-coin disagreements accumulate.
Your job: from several accounts' daily balances over 200 days, recover the
*exact* `(p, q, fee, theta)` that generated them.

**Illustrative FORM only — not the hidden law of this problem:** e.g. a rule
like `x <- x + 2*floor(x/3)` on an unrelated integer sequence is the *shape*
of "an exact integer update rule you must recover from data"; the real rule
here follows the ledger mechanics described above, not this example.

## Input (stdin)
```
t K T L
b0_1 b1_1 b2_1 ... bT_1
b0_2 b1_2 b2_2 ... bT_2
...
b0_K b1_K b2_K ... bT_K
```
`t` is the test id. `K` accounts are logged for `T` days each (`T+1` balances
per account, `b0` = starting balance). `L = 30` is the month length (day `d`
of the whole run belongs to month `d // L`; it is the month's last day iff
`d % L == L-1`).

## Output (stdout)
Four whitespace-separated integers on one line:
```
p q fee theta
```
representing your recovered rate `r = p/q`, fee, and poverty line.

## Feasibility
The submission scores `Ratio: 0.0` unless ALL hold:
- exactly 4 tokens, each parses as a finite integer;
- `1 <= q <= 10^7`, `0 <= p < q` (a daily growth rate strictly under 100%);
- `0 <= fee <= 10^7`; `0 <= theta <= 10^9`.

## Objective (minimize)
The grader picks 4 **fresh** starting balances (never shown to you) and rolls
the SAME ledger mechanics forward for 300 days — well past your 200 training
days — twice: once under the true hidden `(p*, q*, fee*, theta*)`, once under
your submitted `(p, q, fee, theta)` (your rollout tracks its own month-start
balances; it does not see the true trajectory). Let `F` be the mean absolute
coin error between the two rollouts over all account-days.

## Scoring
```
B = mean absolute error of the "nothing changes" rollout (r=0, fee=0)
eps = B / 8
sc = min(1000, 100 * (B + eps) / max(1e-9, F + eps))
Ratio = sc / 1000
```
Predicting no growth at all reproduces `B` exactly, scoring `Ratio = 0.1`.
Because of the `eps` floor, even an exact recovery (`F = 0`) caps at
`Ratio = 0.9` — there is always headroom above any single strategy.

## Constraints
`1 <= t <= 10^5`; `4 <= K <= 6`; `T = 200`; `L = 30`; time limit 5s, memory
512m. Each `.in` file is well under 5 MB.

## Example (worked score)
Suppose `B = 5000` and a submission's rollout achieves `F = 500`
(`eps = 625`): `sc = 100*(5000+625)/(500+625) = 100*5625/1125 = 500`, so
`Ratio = 0.5`. A submission with `F = 0` gets
`sc = 100*5625/625 = 900`, i.e. `Ratio = 0.9`.
