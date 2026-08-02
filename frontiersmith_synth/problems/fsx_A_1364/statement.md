# Dual-Quota Batch: A Common Independent Set with a Maximality Certificate

## Problem
You are given `n` candidate items, numbered `1..n`. Two INDEPENDENT classification
schemes are imposed on them:

- **Scheme A** assigns every item a class in `1..K1`. Class `c` may contribute at most
  `cap1[c]` items to your chosen batch.
- **Scheme B** assigns every item a class in `1..K2`, with its own per-class quota
  `cap2[c]`.

Choose a subset `I` of items (your *batch*) that respects **both** quota systems
simultaneously: for every scheme-A class, the number of chosen items in it is at most
`cap1[c]`; likewise for every scheme-B class. (In matroid language, `I` must be
independent in the partition matroid of Scheme A *and* the partition matroid of
Scheme B — a *common independent set*.) You want the batch as LARGE as possible.

Optionally, you may also submit a **maximality certificate**: a subset `A` of the `n`
items (with complement `Ā = {1..n} \ A`). Define `r1(A) = sum over scheme-A classes c
of min(cap1[c], |A items in class c|)` and `r2(Ā)` analogously for Scheme B. It is a
mathematical fact that for ANY valid batch `I` and ANY such split, `|I| <= r1(A) +
r2(Ā)`. If your certificate achieves EQUALITY `r1(A) + r2(Ā) = |I|` for the batch you
submitted, that equality is a machine-checkable proof that your batch is as large as
any batch could possibly be — and you earn a bonus (see Scoring). No certificate can
achieve equality unless your batch really is optimal, so there is no way to bluff it.

## Input (stdin)
```
n K1 K2
c1_1 c2_1
c1_2 c2_2
...
c1_n c2_n
cap1[1] cap1[2] ... cap1[K1]
cap2[1] cap2[2] ... cap2[K2]
```
Line `i+1` gives item `i`'s scheme-A class `c1_i` (1..K1) and scheme-B class `c2_i`
(1..K2).

## Output (stdout)
```
m
i_1 i_2 ... i_m
flag
[k a_1 a_2 ... a_k]     (only if flag = 1)
```
`m` is the batch size and `i_1..i_m` its distinct 1-indexed item ids. `flag` is `0` (no
certificate) or `1` (certificate follows): if `1`, print `k` then the `k` item ids of
the set `A`. Whitespace/newline layout is free-form (all fields are read as a token
stream); if you have no certificate, just print `0` for `flag` and stop.

## Feasibility
Any of the following makes the whole submission score `Ratio: 0.0`: `m` out of
`[0,n]`; a listed item id out of `[1,n]` or repeated; the chosen batch exceeding
`cap1[c]` for some scheme-A class `c`; or exceeding `cap2[c]` for some scheme-B class.
A malformed or absent certificate simply forfeits the bonus — it never invalidates an
otherwise-feasible batch.

## Objective and Scoring
Let `F = m` if no valid certificate was supplied, or `F = 1.2 * m` if `flag = 1` and
`r1(A) + r2(Ā) = m` exactly (as defined above; checked exactly in integer arithmetic).
Let `B` be the size of the checker's own reference batch, built by filling Scheme A's
classes greedily in input order (ignoring Scheme B entirely), then dropping from that
list, in the same order, any item that would break a Scheme-B quota. `B` is always a
valid, positive batch. Then:
```
sc = min(1000.0, 100.0 * F / max(1e-9, B))
Ratio = sc / 1000.0
```
Matching the reference batch scores `0.1`; `10x` its size (with bonus) caps at `1.0`.

## Constraints
- `1 <= n <= 200`, `1 <= K1,K2 <= n`, all `cap1[c], cap2[c] >= 1`.
- Time limit 5s, memory 512m.

## Example (worked, illustrative shape only)
`n=4`, items: `1:(A=1,B=2)`, `2:(A=1,B=1)`, `3:(A=2,B=2)`, `4:(A=3,B=3)`,
`cap1=[1,1,1]`, `cap2=[1,2,1]`. Reference batch (Scheme-A-first, then Scheme-B-repair):
fills item1 for class A=1, skips item2 (A=1 full), keeps item3 (A=2), keeps item4
(A=3) → `{1,3,4}` M1-feasible; Scheme-B-repair: item1 (B=2, ok), item3 (B=2, quota 2,
now full), item4 (B=3, ok) → `B=3`. A batch `{2,3,4}` is also feasible (A: 1,2,3 each
once; B: 1,2,3 each once) and size `m=3`, matching `B`, so `F=3`, `Ratio=0.1` — unless
a valid certificate is attached, which could raise it toward `0.12`.
