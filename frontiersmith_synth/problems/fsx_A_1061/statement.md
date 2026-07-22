# Two-Scale Echo Suppression — Channel Sets Across a Coarse-Band Chain

## Problem

A relay must pick `k` distinct radio channels out of `M` available slots, `A = {a_1,...,a_k} \subseteq \{0,...,M-1\}`. Two receivers listen for echoes on this set, at two different resolutions:

- The **fine receiver** works modulo the full slot count `M`. It reports an echo whenever two channel *pairs* sum to the same value mod `M`: formally it counts
  `E_M(A) = #{(a,b,c,e) in A^4 : a+b == c+e (mod M)}`.
- The **coarse receiver** only resolves *bands* — it works modulo a given divisor `D` of `M` (many fine channels share one band). It reports the same kind of pair-sum echo, but modulo `D`:
  `E_D(A) = #{(a,b,c,e) in A^4 : a+b == c+e (mod D)}` (equivalently, the fine-receiver formula applied to the band labels `a_i mod D`).

Both counts always include the "trivial" matches where `{a,b} = {c,e}` as an unordered pair — those can never be avoided — so `E_n(A) >= 2k^2 - k` for every set and every modulus `n`, with equality exactly when no other collisions occur (a **Sidon set** mod `n`). The relay's total interference score to **minimize** is

```
TOTAL(A) = E_M(A) + W * E_D(A)
```

where `W` is a given positive integer weight (the coarse band matters more when many fine channels share it). Note: `D` here is typically much smaller than `k`, so by pigeonhole the band labels `a_i mod D` *must* repeat — you cannot make the coarse receiver see a Sidon set; the best you can do is spread the repeats as evenly as possible across the `D` bands while still keeping the fine receiver happy.

## Input (stdin)

```
M D k W
```

- `M` — the full channel modulus.
- `D` — the coarse-band divisor of `M` (`D` divides `M`, and `gcd(D, M/D) = 1`).
- `k` — the number of channels to select, `4 <= k <= 20`.
- `W` — a positive integer weight on the coarse term.

## Output (stdout)

Print exactly `k` **distinct** integers, each in `[0, M)`, separated by whitespace — your channel set `A`.

## Feasibility

Feasible iff the output has exactly `k` tokens, each a finite integer in `[0, M)`, all distinct. Any violation (wrong count, out of range, duplicate, non-integer/non-finite token) scores `0`.

## Objective

Minimize `TOTAL(A) = E_M(A) + W * E_D(A)` as defined above, computed with exact integer arithmetic.

## Scoring

The checker builds its own trivial reference set `A0 = {0, D, 2D, ..., (k-1)D} mod M` (all channels crammed into coarse band `0`) and computes `B = TOTAL(A0)`. Then

```
Ratio = min(1000, 100 * B / TOTAL(A)) / 1000
```

Matching the trivial reference scores about `0.1`; a set with a substantially smaller `TOTAL` scores higher, up to a cap of `1.0`.

## Constraints

- `4 <= k <= 20`, `1 <= D < k`, `D | M`, `gcd(D, M/D) = 1`, `1 <= W <= 10^6`, `M <= 5*10^6`.
- The checker runs in `O(M + k^2)` exact integer arithmetic; scoring is fully deterministic.

## Example (worked score)

`M=10, D=2, k=3, W=1`, submission `A = {0,2,5}` (band labels `0,0,1`):
`E_M(A)=17`, `E_D(A)=41`, `TOTAL=58`.
Checker's trivial reference `A0={0,2,4}` (band labels `0,0,0`): `E_M(A0)=19`, `E_D(A0)=81`, `B=19+1*81=100`.
`Ratio = min(1000, 100*100/58)/1000 = 0.1724`.
