# One Comet, Many Clocks

## Problem
A comet's brightness index on night `k` is `a(k) = tr(M^k)`, the trace of the
`k`-th power of a hidden `d x d` integer "resonance matrix" `M` (`3 <= d <= 5`,
entries bounded, fixed once per test). By a standard algebraic fact `a(k)`
obeys one **exact, order-`d`, integer linear recurrence** for all `k > d` (the
recurrence coming from `M`'s characteristic polynomial) — but `M` itself, and
that recurrence, are hidden.

A handful of small robotic observatories log the comet every night. Each
observatory's clock has its own small-prime **modulus**: on its logging
night it writes down `a(k)` **reduced mod its own prime**, not the true
value. Which observatory logs which night rotates through an undisclosed
repeating schedule — you only see, per logged night, *which* prime was used.
No single observatory's prime is ever big enough alone to pin the true
(signed) recurrence coefficients down — only by combining evidence across
*several* primes can the exact integer law be recovered (Chinese Remainder
Theorem).

You must then answer forecast queries: for a future night `K` (`K` can be up
to `10^15`) and a modulus `P` (sometimes one of the logged primes, often a
brand-new one no observatory ever used), report `a(K) mod P`.

## Input (stdin)
```
t d N Q
k_1 p_1 r_1
...
k_N p_N r_N
K_1 P_1
...
K_Q P_Q
```
`t` is the test id, `d` the matrix dimension. Then `N` logged nights: night
`k_i`, observatory prime `p_i`, and the logged value `r_i = a(k_i) mod p_i`
(nights are given in increasing `k` order). Then `Q` forecast queries, each a
`(K, P)` pair, in no particular order.

## Output (stdout)
Exactly `Q` integers, one per line (or whitespace-separated): your predictions
for `a(K_1) mod P_1, ..., a(K_Q) mod P_Q`, in the order the queries were
given. Each token must be a finite base-10 integer.

## Feasibility
The output must parse as exactly `Q` base-10 integers (no decimals, no
`nan`/`inf`, no missing/extra tokens). Any violation scores `Ratio: 0.0`.

## Objective and Scoring
For each query the true value is `a(K_i) mod P_i`. Since a modulus wraps
around, "how wrong" a guess is is measured by the **wrapped (circular)
distance**, normalized to `[0,1]`:
```
dist_i   = min(|pred_i - actual_i| mod P_i, P_i - (|pred_i - actual_i| mod P_i))
loss_i   = dist_i / (P_i / 2)
F        = average(loss_i)              # your total normalized error (lower = better)
```
The checker also computes its own baseline `B` = the same `F` for the
"predict 0 for everything" construction. Your score is
```
Ratio = min(0.88, 100 * B / max(1e-9, F)) / 100
```
so "predict 0" scores about `0.1`, driving `F` below `B` raises your score,
and a fully exact recovery is capped at `0.88` (headroom is deliberately left
above a perfect reference solution). **You are maximizing this Ratio.**

## Why the obvious approach is a trap
Fitting a linear recurrence **separately for each observatory's own prime**
(a Berlekamp-Massey-style window solve mod that one prime) answers queries
against a modulus you've already seen exactly — but for a brand-new modulus,
reusing "the coefficients solved mod my best-covered prime" as the presumed
true integer law is only valid when the true coefficients already fit under
*that one prime's* Nyquist bound. The generator avoids that regime, so this
recipe aliases, and forecasts for new clocks come out essentially arbitrary.
The exact law only falls out of **fusing** several primes together.

## Constraints
`3 <= d <= 5`, `N` up to ~300, `Q` up to 20, forecast nights up to `10^15`,
moduli up to ~`10^7`. Time limit 5 s, memory 512 MB. Fully deterministic.

## Example (illustrative FORM only — NOT the hidden law)
If (hypothetically, not the real rule) the brightness obeyed the toy Fibonacci
recurrence `a(k) = a(k-1) + a(k-2)` with `a(1)=1, a(2)=1`, and a single
observatory logged every night mod `7`, you would see rows like `1 7 1`,
`2 7 1`, `3 7 2`, `4 7 3`, `5 7 5`, `6 7 1` (`a(6)=8 mod 7`), and a query
`K=10 P=7` should predict `a(10) mod 7 = 55 mod 7 = 6`. The real instance uses
several clock primes, a rotating schedule, and a different hidden law
entirely — you must discover it from the data.
