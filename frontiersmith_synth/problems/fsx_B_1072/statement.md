# Rotation-Free Ribbon: Necklace-Class Coverage

## Problem
You are given an alphabet size `a`, a factor length `k`, and a cyclic-string length `L`.
You must output ONE cyclic string `s` of length `L` over the alphabet `{0, 1, ..., a-1}`
(cyclic means position `L-1` is adjacent to position `0`, wrapping around).

Consider all `L` cyclic windows of length `k`: the window starting at position `i` is
`s[i], s[i+1 mod L], ..., s[i+k-1 mod L]`. Two length-`k` strings are in the same
**necklace class** iff one is a cyclic rotation of the other (e.g. for `k=3`, `"001"`,
`"010"`, and `"100"` are all the same necklace class). Your score is the number of
*distinct necklace classes* that appear among the `L` windows -- **not** the number of
distinct window strings. Two windows that are rotations of each other count only once.

## Input (stdin)
```
a k L
```

## Output (stdout)
A single line: the length-`L` string `s`, each character a digit in `{0, ..., a-1}`.

## Feasibility
An output is valid iff it is a single token, has length exactly `L`, and every character
is a digit strictly less than `a`. Any violation scores `Ratio: 0.0`.

## Objective
Maximize `F`, the number of distinct necklace classes among the `L` cyclic length-`k`
windows of `s`.

## Scoring
Let `B` be the checker's own trivial baseline: the constant string `"000...0"` (length
`L`), which always yields exactly 1 necklace class, so `B = 1`. With maximization
normalization:
```
sc = min(1000.0, 100.0 * F / max(1e-9, B))
Ratio = sc / 1000.0
```
Reproducing the baseline scores `Ratio = 0.1`; hitting 10 or more distinct necklace
classes caps the score at `1.0`.

**The trap.** A natural first idea is to build `s` the way you would build a de Bruijn
sequence: greedily walk forward, at each step choosing the smallest next character such
that the newest length-`k` window is a substring you haven't printed before. This
maximizes *distinct substrings*, but it has no notion of rotation: many of the "new"
substrings it happily accepts are just cyclic rotations of a necklace class it already
covered, so a large fraction of the `L` window-slots gets wasted re-covering ground
already scored. Because a necklace class has up to `k` different rotation-representatives
that all look like "new" substrings to a rotation-blind walk, this waste compounds.

**The insight.** Work with necklace-*canonical* forms directly: a length-`k` string is
its own necklace's canonical representative iff it is already the lexicographically
smallest among its own rotations (a Lyndon-word-style minimal form). Enumerate distinct
canonical representatives and concatenate them back-to-back with **zero overlap**. Every
block-aligned window then reproduces its representative exactly, so each block of `k`
characters buys exactly one *guaranteed-fresh* necklace class -- no rotation waste at
all. The windows that straddle two blocks are pure bonus: they may land on yet another
uncovered class for free, but can never undo the guaranteed per-block gain.

## Constraints
- `2 <= a <= 4`, `3 <= k <= 7`, `9 <= L <= 42`, and `L` is a multiple of `k`.
- `L` is always strictly less than `k` times the total number of necklaces of length `k`
  over an `a`-letter alphabet, so a full necklace cover is never reachable.
- Time limit 5s, memory 512m.

## Example
Let `a=2, k=3, L=6` (illustrative only -- not one of the actual graded instances).
Output `s = "001011"`. Its 6 cyclic windows are `001, 010, 101, 011, 110, 100`. Necklace
classes: `{001,010,100}` and `{011,101,110}` both appear (the other two classes of length
3, `{000}` and `{111}`, do not). So `F = 2`, `B = 1`, `Ratio = min(1000, 200)/1000 = 0.2`.
A necklace-aware choice does better: `s = "001111"` places the canonical rep of
`{001,010,100}` next to the canonical rep of `{111}`. Its windows are
`001, 011, 111, 111, 110, 100`, hitting THREE classes (the two aligned blocks plus a free
bonus hit on `{011,101,110}` from a boundary-crossing window), for `F = 3`,
`Ratio = 0.3` -- strictly better, using the same length, by exploiting rotation
structure instead of raw substring novelty.
