# Mirror Grammar

## Problem
You are given one long string `T` over a small alphabet. Your task is to output a
**straight-line program (SLP)** — a sequence of numbered rules, each defining one new
symbol — that derives exactly `T`, using as few total operations as possible.

Each rule is one of three kinds:
- `T c` — a new symbol expanding to the single character `c`.
- `C j k` — a new symbol expanding to `expand(j) + expand(k)` (concatenation of two
  earlier symbols).
- `R j` — a new symbol expanding to the reverse of `expand(j)`.

Rules are numbered 1..M in the order they appear (the i-th output line defines symbol
`i`); a rule may only reference symbols with smaller indices (no forward references, no
cycles). The **last rule** in your output is the start symbol and must expand to exactly
`T`.

`T` is built by a hierarchical process: a short random seed block is repeatedly wrapped
with fresh random padding and combined with (a) a **reversed copy of itself** and (b) a
**literal doubled copy** of a small local chunk, then wrapped again at the next level.
This plants both literal repeats and nested palindromic mirrors at every level, without
telling you where the level boundaries are — you must find them by reading `T`.

## Input (stdin)
One line: the string `T` (lowercase letters `a`-`d` only, no other whitespace).

## Output (stdout)
`M` lines, one rule per line, in the format above.

## Feasibility
- Every reference (`j`, `k` in `C`/`R` rules) must point to a strictly earlier line.
- Every intermediate expansion must not blow up absurdly (checker enforces a generous
  size cap; any expansion that is actually useful is never longer than `T`).
- The final (last) rule must expand to exactly `T`, character for character.
- Any violation scores `Ratio: 0.0`.

## Objective (minimize)
Total operation count: a `T` rule costs 1, a `C` rule costs 2, an `R` rule costs 1. Sum
this over all `M` rules — this is your grammar's size. Smaller is better.

## Scoring
The checker also builds its own trivial construction internally: one `T` rule per
character of `T` plus a left-to-right chain of `C` rules combining them, whose cost is
`B = 3n - 2` for `n = |T|`. Writing `F` for your total operation count, the score is

```
ratio = min(1000, 100 * B / F) / 1000
```

so matching the trivial baseline scores `0.1`, and a 10x-smaller grammar caps the score
at `1.0`.

## Constraints
- `1 <= |T| <= 5000`.
- Alphabet: `a`, `b`, `c`, `d`.
- Time limit 5s, memory 512MB.

## Example (worked score, illustrative only — not one of the generator's cases)
For `T = "abccba"` (n = 6), one valid SLP:
```
T a
T b
T c
C 1 2
C 4 3
R 5
C 5 6
```
Rule 4 makes `"ab"`, rule 5 makes `"abc"`, rule 6 reverses it to `"cba"`, rule 7
concatenates `"abc" + "cba" = "abccba" = T`. Operation count: three `T` rules (1 each),
three `C` rules (2 each), one `R` rule (1): `F = 3 + 6 + 1 = 10`. Baseline
`B = 3*6 - 2 = 16`. Score `= min(1000, 100*16/10)/1000 = 0.16`.

A submission that ignored the mirror and rebuilt `"cba"` from scratch with three more
`T`/`C` rules would use more operations for a worse score — recognizing and reusing the
reversed block is exactly the kind of saving that matters at scale, where blocks are
wrapped many levels deep.
