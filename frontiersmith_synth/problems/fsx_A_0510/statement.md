# Ring Beacons: Clean-Beat Reuse over a Composite Frequency Ring

## Problem
A ring network reuses beacon frequencies drawn from `Z_n = {0, 1, ..., n-1}`, where the
ring size is **composite**: `n = m1 * m2 * m3` with the three moduli **pairwise coprime**
prime powers. You activate a set `B` of tower base-offsets (residues mod `n`).

When two active towers sit at offsets `b` and `b'`, their beacons produce a *beat* at the
difference `d = (b - b') mod n`. A beat is **clean** only if it stays clear of forbidden
interference in **every** frequency band simultaneously: for each factor `i`, the reduced
value `d mod m_i` must lie in a given allowed set `A_i` of that band. Formally the clean
targets are

```
T = { d in Z_n : (d mod m_i) in A_i  for all i = 1,2,3 }.
```

You want your activated towers to *realize* as many distinct clean beat frequencies as
possible. Because the ring factors through the Chinese Remainder Theorem, a beat `d`
corresponds to the residue triple `(d mod m1, d mod m2, d mod m3)`, and it is clean iff each
coordinate is allowed. Note `T` itself is the CRT product `A_1 x A_2 x A_3`.

## Input (stdin)
```
3
m1 m2 m3
k
s1
a_{1,1} a_{1,2} ... a_{1,s1}
s2
a_{2,1} ... a_{2,s2}
s3
a_{3,1} ... a_{3,s3}
```
`m1,m2,m3` are pairwise coprime prime powers; `n = m1*m2*m3`. `k` is your activation budget.
`A_i = { a_{i,1}, ..., a_{i,si} }` is the allowed residue set for factor `i` (each in
`[0, m_i)`, and `0 in A_i`).

## Output (stdout)
```
j
b_1
b_2
...
b_j
```
`j` (with `0 <= j <= k`) is the number of towers you activate, followed by their distinct
base-offsets `b_t in [0, n)`.

## Feasibility
`j <= k`; every `b_t` an integer in `[0, n)`; all `b_t` distinct. Any violation
(too many towers, out-of-range, duplicate, or non-integer) scores `0`.

## Objective (maximize)
Maximize the number of **distinct clean beat frequencies** realized:
```
F = | { (b_t - b_u) mod n : 1<=t,u<=j } cap T |.
```
(The self-beat `0` is clean and counts once.)

## Scoring
Let `Bsc` be the checker's internal reference `F` for a fixed random set of `k` residues.
Your ratio is
```
Ratio = min(1000, 100 * F / Bsc) / 1000.
```
Matching the random reference gives about `0.1`; realizing ten times its clean coverage
caps at `1.0`. The maximum is genuinely open: no known construction is proven optimal, so
strong reference solutions leave headroom above them.

## Constraints
`n` up to about `8 * 10^8`; `k = 60`; 10 independent cases. Time limit 5 s per case,
memory 512 MB. A brute-force search over `Z_n` is hopeless; the coprime factorization is the
lever — the moduli `m_i` are small.

## Example (illustrative shape only)
Suppose `B = B1 x B2 x B3` is a residue product (each tower's residues chosen from a small
per-band set). Then every beat's residue triple is a per-band difference, so the clean beats
you realize are exactly the product of the per-band clean differences:
`F = |D(B1) cap A1| * |D(B2) cap A2| * |D(B3) cap A3|`, where `D(S)` is the difference set of
`S`. Thus a budget of `k = k1*k2*k3` towers realizes the product of three small per-band
coverages. Whether a product layout, or some other layout, maximizes `F` for the given
`A_i` is up to you.
