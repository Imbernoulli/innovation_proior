# Hash Function Versus the Published Key-Family Sweep

## Problem

You must design a hash function that maps 64-bit keys into `M = 1024` buckets. Unlike a
textbook exercise, your hash will not be judged on random keys: it is judged **openly**
against a fixed, *published* sweep of structured "hostile" key families (arithmetic
progressions with various strides, contiguous bit-plane clusters, and low-entropy
float-style encodings with a fixed exponent field). The families and their exact
parameters are given to you in the input — the sweep is white-box. Your goal is to keep
the **worst bucket, over every family, over the whole sweep** as lightly loaded as
possible.

A hash built from a single multiplicative mixer, applied blindly, is knowably vulnerable:
some published family's stride will interact badly with *any one* fixed mixer, or with the
wrong choice of which bits you finally read off. The published sweep is specifically built
so this happens. You must read the sweep and reason about which mixer/extraction choice
survives which families — not just apply one standard trick and hope it averages out.

## Input (stdin)

```
M F
<family_1>
<family_2>
...
<family_F>
```
`M` is always 1024. `F` is the number of families (roughly 3–6). Each family line is one of:
```
AP start stride count        keys[i] = (start + i*stride) mod 2^64,  i = 0..count-1
COSET base lo width          keys enumerate all 2^width values of a contiguous bit
                              window [lo, lo+width) over the fixed base (a "bit-plane
                              cluster": every other bit is frozen to base)
FLOAT exp mant_base mant_stride count
                              keys[i] = (exp << 40) | ((mant_base + i*mant_stride) mod 2^40)
                              (fixed high 24-bit "exponent" field, moving low 40-bit
                              "mantissa", wrapping — a low-entropy float-style cluster)
```
All values are non-negative integers; `start/base/mant_base/exp` fit in 64/64/40/24 bits
respectively.

## Output (stdout) — the artifact

```
S REDUCE
<stage_1>
...
<stage_S>
```
`S` (0–4) is the number of mixing stages, applied to each key in order; `REDUCE` is
`MODM` (bucket = final value mod M) or `TOPBITS` (bucket = final value's top 10 bits,
i.e. `value >> 54`). Each stage is one of:
```
MUL a b        v <- (a*v + b) mod 2^64                 (0 <= a,b < 2^64)
ROT r          v <- rotate_left(v, r) over 64 bits      (0 <= r <= 63)
XORFOLD r      v <- v XOR (v >> r)                      (1 <= r <= 63)
SALT t idx v_0 ... v_{2^t-1}
               idx_field <- (v >> idx) & (2^t - 1); v <- v XOR v_{idx_field}
               (1 <= t <= 12, 0 <= idx <= 63, 2^t table values, each < 2^64)
```
`v` starts equal to the raw key. The total number of SALT table entries printed (summed
over all SALT stages) must not exceed 8192.

## Feasibility

The output must parse exactly as above with no missing/extra tokens, all integers
non-negative, finite, and within the ranges stated (this rejects `nan`/`inf`/garbage/huge
values automatically). Any violation scores `Ratio: 0.0`.

## Objective

For each family, hash every one of its keys and find that family's most-loaded bucket
(its peak count). Your objective is the **maximum of these peaks over all families in the
sweep** — minimize it.

## Scoring

The checker builds its own reference hash (one fixed multiplicative mixer + top-bit
extraction) and computes its peak-of-peaks `B` on the same sweep. Writing your own
peak-of-peaks as `F`:
```
ratio = min(1000, 100 * B / F) / 1000
```
Better (lower `F`) scores higher, capped at 1.0. Ratio 0 means infeasible output.

## Constraints

Time limit 5s, memory 512MB. Total keys per test case ≤ 40000.

## Example (worked, illustrative only — NOT the sweep's real shape)

Suppose the sweep is a single family `AP 0 1 4` (keys `0,1,2,3`), with real `M=1024`.
Output `1 TOPBITS` / `MUL 1 0` (identity multiply, then read the top 10 bits) sends all
four keys to bucket 0, since `0,1,2,3 >> 54` are all `0` — peak load 4, a poor score.
Output `0 MODM` (no stages, bucket = key mod 1024) instead sends them to buckets
`0,1,2,3` — peak load 1, a much better score. This toy example only shows that
*which* stages you compose and *which* extraction you finish with both matter; it is far
too small to exhibit the sweep's real traps, which involve much larger, specifically
chosen families.
