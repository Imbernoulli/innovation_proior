# Carillon Round: A Melody That Answers Its Own Echoes

## Problem

You write ONE melody: a sequence of pitch classes (integers mod 12, like
notes on a 12-tone bell tower). It is performed as a **round** (a canon):
alongside the original, `K` extra bell towers each play a **delayed,
transposed copy** of the very same melody. Voice `i` is described by a
delay `d_i >= 1` (it starts `d_i` notes late) and a transposition `t_i`
(every pitch class it plays is shifted up by `t_i`, mod 12). Every note has
duration exactly 1 tick, so if your melody is `c_0, c_1, ..., c_{m-1}`,
voice `i` sounds pitch `(c_j + t_i) mod 12` at absolute time `j + d_i`.

Whenever two of these voices (the original counts as voice 0, delay 0,
transposition 0) sound at the same absolute time, their two pitch classes
must be **consonant**: the input gives a 12x12 0/1 table `C` (symmetric)
where `C[a][b] = 1` means pitch classes `a` and `b` may sound together.
Your melody must be consonant with *every* transformed echo of itself, at
every moment two echoes overlap — this is the only feasibility rule.

## Input (stdin)

```
K L
C[0][0] ... C[0][11]
...
C[11][0] ... C[11][11]
d_1 t_1
...
d_K t_K
```
`K` (1..3) voices besides the original; `L` is the maximum melody length
you may output. `C` is symmetric with `C[i][i] = 1` for all `i` (a note is
always consonant with an exact unison of itself). Delays `d_i` are pairwise
distinct positive integers; transpositions `t_i` are in `[1, 11]`.

## Output (stdout)

```
m
c_0 c_1 ... c_{m-1}
```
`1 <= m <= L`, each `c_k` an integer pitch class in `[0, 11]`.

## Feasibility

For every pair of voices and every absolute time both are sounding, the two
sounding pitch classes must satisfy `C[a][b] = 1`. Any violation, any
malformed/out-of-range/mismatched-length output scores `Ratio: 0.0`.

## Objective

Let `H` be the Shannon entropy (base 2) of the pitch-class distribution of
your `m` notes, normalized `H_norm = H / log2(12)` (so `H_norm` is 0 for a
drone, 1 for a perfectly uniform mix of all 12 classes). Let `TG` be the
number of *distinct contour trigrams*: for each consecutive pair of notes
take the sign of their shortest signed pitch-class step (`-1`/`0`/`+1`),
then count distinct triples of three consecutive signs.

```
F = m * (H_norm + 0.05) + TG
```

The checker also builds its own trivial reference melody (a single drone
pitch held as long as it stays consonant with its own echoes) to get a
baseline `B > 0`, and reports
`Ratio = min(1000, 100 * F / B) / 1000`.

Longer AND more pitch-diverse melodies score higher — but a long, low-
diversity drone or simple repeating arpeggio still only edges past the
baseline: it survives the consonance rule cheaply while contributing almost
nothing to `H_norm` or `TG`.

## Constraints

`1 <= K <= 3`, `8 <= L <= 40`, `2 <= d_i <= 10`, `1 <= t_i <= 11`. Time
limit 5s, memory 512MB.

## Example (worked score, illustrative shape only)

Suppose `K=1`, `L=6`, `d_1=3`, `t_1=1`, and `C` happens to make the melody
`0 1 2 0 1 2` fully consonant with its own delay-3/transpose-1 echo (i.e.
`C[c[k]][(c[k-3]+1) % 12] = 1` for `k=3,4,5`). Its pitch distribution is
uniform over `{0,1,2}`, so `H = log2(3) ~= 1.585`, `H_norm ~= 0.4406`.
Contour: steps are `+1,+1,-2->sign -1,+1,+1,-2->sign -1` — signs
`(1,1,-1,1,1,-1)`, giving distinct trigrams `(1,1,-1)` and `(1,-1,1)`, so
`TG = 2`. `F = 6 * (0.4406 + 0.05) + 2 = 2.944 + 2 = 4.944`. Compare this
against whatever drone-length `B` the checker finds feasible for this
particular `C` and delay — a longer, more evenly-spread, more contoured
melody scores higher, but only if it survives every echo it creates against
itself.
