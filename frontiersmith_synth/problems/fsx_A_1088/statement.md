# Cheapest Clockwork: A Long Cycle That Still Remembers Its Anchors

## Problem

You must design a permutation `f : Z_n -> Z_n` and hand it over as a
**straight-line arithmetic program** ("clockwork"): a short list of scalar
instructions, each operating on 8 registers `r0..r7`, all arithmetic mod `n`.
Register `r0` starts holding the input `x`; every other register starts at
`0`. The program's output is the final value of `r0`. Your only goal is to
minimize the number of instructions, subject to `f` satisfying two
constraints that pull in opposite directions:

1. **Anchors.** `f` must fix a given set `S` of points pointwise: `f(s) = s`
   for every `s` in `S`.
2. **Reach.** the cycle of the permutation `f` that contains `0` must have
   length at least `L`.

A single instruction like "add 1" gives you a full `n`-cycle for free (one
instruction!) -- but it fixes nothing at all, so any nonempty `S` kills it.
Making a handful of points stick usually costs you cycle length, or a lot of
instructions, or both. `n` is **guaranteed to be the product of two distinct
primes**, `n = p * q`, with `p` much smaller than `q`; nothing else about the
factorization, or about why `S` is what it is, is told to you.

## Instruction set

Each instruction is one line, 4 whitespace-separated tokens, using register
indices in `[0,8)` and constants `K` in `[0,n)`:

```
ADD  a b d     r[d] = (r[a] + r[b])  mod n
MUL  a b d     r[d] = (r[a] * r[b])  mod n
ADDC a K d     r[d] = (r[a] + K)     mod n
MULC a K d     r[d] = (r[a] * K)     mod n
```

## Input (stdin)

```
n
k
s_1 s_2 ... s_k
L
```
`2 <= n <= ~20000`, `0 <= k <= 6`, each `s_i` in `[0,n)`, `1 <= L <= n`.

## Output (stdout)

```
m
<instruction 1>
...
<instruction m>
```
`1 <= m <= 150`.

## Feasibility

The checker executes your program on **every** `x` in `[0,n)` (exact integer
arithmetic). It rejects (score `0`) unless:
- the resulting map `f` is a bijection of `{0,...,n-1}`,
- `f(s) = s` for every `s` in `S`,
- the cycle of `f` containing `0` has length `>= L`.

Malformed, oversized, non-finite, or out-of-range output is rejected the same
way.

## Objective and scoring

Feasible submissions are scored purely on instruction count `m` (fewer is
better) against a fixed internal reference op-budget `B` the checker keeps to
itself:
```
ratio = min(1, 0.1 * B / m)
```
A do-nothing-clever construction lands near `0.1`; a construction using
roughly `B/10` instructions saturates near the top of the scale. There is no
bonus for going below the true minimum the checker knows about.

## Worked example (illustrative only -- not one of the real tests)

`n = 15 = 3*5`. Say `S = {7, 12}`, `L = 4`. The 2-instruction program
```
2
MULC 0 7 0
ADDC 0 3 0
```
computes `f(x) = 7x+3 mod 15`. Check: `f(7) = 52 mod 15 = 7`, `f(12) = 87 mod
15 = 12` -- both anchors hold. Following `0 -> 3 -> 9 -> 6 -> 0` gives a
cycle of length 4, meeting `L=4`. This `f` is a bijection (`gcd(7,15)=1`), so
it is feasible, and at 2 instructions it scores near the top of the scale.

## Constraints recap
`2 <= n <= ~20000`, `0 <= k <= 6`, `1 <= L <= n`, `1 <= m <= 150`, register
indices in `[0,8)`, constants in `[0,n)`. Time limit 5s.
