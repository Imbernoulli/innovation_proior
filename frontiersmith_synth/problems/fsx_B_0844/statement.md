# Lottery Machine Forensics

## Problem

A retired lottery machine drew numbers for years using a simple hidden rule:

```
x_{k+1} = (a * x_k + c) mod m
```

starting from a secret first draw `x_1`. The operator lost the machine's manual but kept the
logbook of the first `N = 120` draws `x_1, ..., x_N` (exact integers, each in `[0, m)`). All you
are told about the hidden law:

- `m` is the product of exactly **4 distinct primes**, each between 101 and 4999 (which primes,
  and the values of `a`, `c`, `x_1`, are all secret).
- `a` and `c` are fixed unknown residues in `[0, m)`.

Given the logbook, predict the exact draw at each of `Q` requested **future** indices
`k_1 < k_2 < ... < k_Q` (indices can be as large as `10^18`, since the machine kept running long
after the logbook ends — you cannot simulate that many steps one at a time within the time
limit).

## Input (stdin)

```
N
x_1 x_2 ... x_N
Q
k_1 k_2 ... k_Q
```

`N = 120` always. `k_1 < k_2 < ... < k_Q`, all strictly greater than `N`, up to `10^18`.

## Output (stdout)

`Q` integers — your predicted draws at `k_1, ..., k_Q`, in that order, space- or
newline-separated. Every value must be a plain nonnegative decimal integer (no signs, no
decimal points, no scientific notation).

## Feasibility

Your output must contain **exactly** `Q` tokens, each a nonnegative integer literal of at most 30
decimal digits (every legitimate answer is under `m < 10^15`, so this is generous headroom, not a
real constraint). Any missing or extra token, any non-numeric token, any token longer than 30
digits, or a negative number scores that test case `0`.

## Scoring

`m` factors into the 4 secret primes `p_1, p_2, p_3, p_4`. For each query you earn credit equal
to the **fraction of these 4 primes for which your guess agrees with the true draw modulo that
prime** (possible values `0, 1/4, 1/2, 3/4, 1`; an exactly correct guess mod `m` always earns the
full `1`, since agreeing mod every prime factor of `m` forces agreement mod `m`). Your score on a
test case is the average credit over its `Q` queries, then rescaled so a hopeless guess sits near
the low end and a fully correct reconstruction sits high but below the maximum reported ratio.
This rewards partially recovering the hidden structure (e.g. nailing 3 of the 4 residue
components) even when the whole modulus isn't cracked.

**Illustrative FORM only** (not the hidden law — a different, simpler shape, shown only to
clarify the input/output mechanics): suppose a toy sequence obeyed `y_{k+1} = 3*y_k mod 7` with
`y_1 = 2`, so the logged values `y_1..y_5` are `2, 6, 4, 5, 1`. Since `6/2 = 3 (mod 7)`, you could
read off the multiplier and answer `y_8 = 2 * 3^7 mod 7 = 6`. The real task's hidden law is a
different, composite, unknown modulus with an unknown additive term — this single-division trick
does not directly carry over, and dividing by the wrong quantity mod the wrong modulus can fail
outright.

## Constraints

- `N = 120` fixed, `Q = 8` fixed.
- Every `k_i <= 10^18`.
- Time limit 5s, memory 512MB.
- `m` (product of the 4 secret primes, each in `[101, 4999]`) is at most roughly `6 * 10^14`, well
  within 64-bit range but far too large to search or brute-force-simulate.

## Example (worked score, illustrative numbers only)

Say for some test case the true draw at a query is `T = 8675309` and its (secret) prime
factorization context has `m = p_1 p_2 p_3 p_4`. If your guess `G` satisfies `G ≡ T (mod p_1)`,
`G ≡ T (mod p_2)`, `G ≡ T (mod p_3)`, but `G != T (mod p_4)`, that single query earns credit
`3/4 = 0.75`. Averaging such credits over all `Q` queries of a test case, then rescaling, gives
that test case's reported ratio.
