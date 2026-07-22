# Vesting Lock-in Allocator

## Problem

You manage a pool of capital across `T` periods and `K` investment instruments.
Each instrument `k` has a fixed **lock duration** `L_k`: any amount you commit
to instrument `k` is frozen (irreversible — there is no early withdrawal, ever)
for exactly `L_k` consecutive periods starting the period you commit it, and
compounds period-by-period at that instrument's own rate. Only once the lock
matures does the grown amount return to your pool of free cash, available to
commit again (possibly to a different instrument).

Every instrument's per-period rate for every period is given to you up front
as a full `T x K` table — you know the entire future rate schedule when you
decide what to do today. Rates can be positive or negative.

At the start of each period `t = 1..T`, you may commit any non-negative
amounts of your *currently free* cash to any subset of instruments (you do not
have to commit it all — cash can sit idle for a period, earning nothing, and
try again later). Because a commitment is frozen the instant it is made, the
choice you make at period `t` determines exactly which rates that money will
ride for the next `L_k` periods — so making the most of a locked instrument
means timing the commitment so its lock lines up with that instrument's best
stretch of the schedule, not just its rate on the day you commit.

## Input (stdin)

```
T K
C0
L_1 L_2 ... L_K
rate[1][1] rate[1][2] ... rate[1][K]
rate[2][1] ...
...
rate[T][1] ... rate[T][K]
```
`C0` is your starting free cash (integer). `L_k` (integer, `1 <= L_k <= T`) is
instrument `k`'s lock duration. `rate[t][k]` is an integer in basis points
(rate for period `t` = `rate[t][k] / 10000`); it can be negative.
Constraints: `1 <= T <= 50`, `1 <= K <= 8`, `1 <= C0 <= 10^7`,
`-2000 <= rate[t][k] <= 3000`.

## Output (stdout)

`T` lines, each with `K` non-negative numbers: `commit[t][1] ... commit[t][K]`
— the amount newly committed to each instrument at the start of period `t`.

## Feasibility

At every period `t`, the sum of that period's commitments may not exceed the
cash currently free (starting cash, plus everything from earlier commitments
whose lock has matured exactly at `t`, plus any cash left idle from before —
tracked by simulation). All commit amounts must be finite and non-negative.
Any violation scores 0.

## Simulation / Objective

A commitment of amount `x` to instrument `k` made at period `t` matures with
value `x * prod_{s=t}^{min(t+L_k-1, T)} (1 + rate[s][k]/10000)` and becomes
free cash again at period `t + L_k` (or, if the lock has not fully matured by
the end of period `T`, that partially-compounded value is simply counted as
part of your final wealth). Your **terminal wealth** `F` is your total cash
after period `T`: idle cash still on hand plus the (possibly partial) value of
every position, matured or not. Maximize `F`.

## Scoring

The checker replays your commitments exactly as specified above (rejecting
any feasibility violation with `Ratio: 0.0`) to get `F`, and compares it to
the baseline `B = C0` (the trivial strategy of never committing anything):

```
Ratio = min(1.0, F / (10 * C0))
```

## Example (worked, illustrative shape only — not the full mechanic)

`T=2, K=1, C0=100, L_1=2`, rates `[[1000],[1000]]` (10% both periods).
Committing all 100 at period 1 locks it for both periods:
`F = 100 * 1.10 * 1.10 = 121`. `Ratio = min(1, 121/1000) = 0.121`.

## Constraints recap

`1 <= T <= 50`, `1 <= K <= 8`, `1 <= C0 <= 10^7`, `1 <= L_k <= T`,
`-2000 <= rate[t][k] <= 3000` (basis points). Time limit 5s, memory 512MB.
