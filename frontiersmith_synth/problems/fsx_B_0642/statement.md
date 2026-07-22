# Long Square-Free Word Hitting a Target Letter Mix

## Problem
A word over the alphabet `{0,1,2,3}` is **square-free** if it contains no factor of the
form `u u` for any nonempty block `u` (no substring that is two immediately consecutive
identical copies of anything, of any length). Square-free words over 4 letters can be
extended forever, but making one long is easy -- making one that *also* hits a
prescribed letter mix, and a prescribed pattern of which letter tends to follow which,
is not: greedily biasing your letter choice toward the mix you want quietly changes
what follows what, and biasing toward the transition you want quietly drifts the mix.

You are given a target length `L`, a target per-letter frequency `w0..w3`, and a target
**successor preference** `s0..s3` (the letter you'd like to see right after each
letter, when possible). Output a square-free word of length at most `L` over
`{0,1,2,3}` that gets both objectives as close as you can, simultaneously.

## Input (stdin)
```
L
w0 w1 w2 w3
s0 s1 s2 s3
```
`L` (integer, 80 <= L <= 500) is the target length. `w0..w3` are nonnegative integers
summing to 10000 (the target frequency of letter `i` is `wi/10000`). `s0..s3` is a
permutation of `{0,1,2,3}` with `si != i` for all `i` (the desired successor of letter
`i` is `si`).

## Output (stdout)
```
N
word
```
`N` (0 <= N <= L) is the length of your word; `word` is a string of exactly `N`
characters from `{0,1,2,3}`. If `N = 0`, the second line may be empty.

## Feasibility
`word` must be square-free: for every period `p >= 1` and every start `i` with
`i + 2p <= N`, the block `word[i..i+p-1]` must differ from `word[i+p..i+2p-1]`. Any
violation, any malformed/out-of-range output, or any non-`{0,1,2,3}` character scores
`Ratio: 0.0`.

## Objective (maximize)
Let `N` be your word's length, `p_i` the realized frequency of letter `i`
(`count_i / N`), and `TV = 0.5 * sum_i |p_i - w_i/10000|` the total-variation distance
to the target mix. Let `match` be the fraction of adjacent pairs `(word[t], word[t+1])`
with `word[t+1] == s[word[t]]` (the target successor realized). Then
```
unigram_score = 1 - TV                (in [0,1])
succ_score    = match                  (in [0,1], 0 if N < 2)
F = (N / L) * (0.25 * unigram_score + 0.75 * succ_score)
```
Both the mix and the transition pattern matter, and the transition term is weighted
more heavily -- a word that nails the letter counts but ignores who-follows-whom scores
far below one that nails both.

## Scoring
Let `B` be the same `F` computed on the checker's own internal reference word (a
square-free scaffold with letter 0 spliced in as a separator, which realizes an exact
flat 0.25/0.25/0.25/0.25 mix but does not aim at `s` at all). With your feasible `F`,
```
sc    = min(1000, 100 * F / max(1e-9, B))
Ratio = sc / 1000
```
so the checker's own reference scores about `0.1`, and a word ten times better caps at
`1.0`.

## Constraints
`4 <= N <= L <= 500` in practice (`80 <= L <= 500` across all tests); alphabet size is
fixed at 4; time limit comfortably covers an `O(L^2)` construction and check.

## Example
`L = 6`, `w = (5000,2000,2000,1000)`, `s = (3,0,1,2)`. The word `010213` is square-free
(check every period up to 3: no two adjacent halves match). Its letter counts are
`(2,2,1,1)` so `p = (0.333,0.333,0.167,0.167)`, giving
`TV = 0.5*(0.167+0.133+0.033+0.067) = 0.2`, so `unigram_score = 0.8`. Its adjacent
pairs are `(0,1),(1,0),(0,2),(2,1),(1,3)`; comparing
against `s = (3,0,1,2)` the pairs `(1,0)` and `(2,1)` match (`s[1]=0`, `s[2]=1`), 2 of 5,
so `succ_score = 0.4`. `F = 1 * (0.25*0.8 + 0.75*0.4) = 0.5`. The checker's own reference
word for `L=6` is `012301`, giving `B = 0.2`, so this submission scores
`Ratio = min(1, 0.1*0.5/0.2) = 0.25`.
