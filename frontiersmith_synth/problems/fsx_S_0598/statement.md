# Quotient Quorum: Most Inequivalent Words modulo a Rewriting Congruence

You are handed a finite **string-rewriting system** over the digit alphabet
`{0, 1, ..., k-1}` and a length bound `L`. Each rule `l -> r` is
**length-preserving** (`|l| == |r|`) and shortlex-decreasing (either shorter — never
happens here — or the same length with `r` lexicographically smaller than `l`).

A single **rewrite step** replaces one occurrence of some rule's left side `l` inside
a word by that rule's right side `r`. Two words are **equivalent** (`~`) if one can be
turned into the other by a finite chain of rewrite steps applied in **either
direction**, where every word along the chain has length in `[1, L]`. This `~` is the
rewriting congruence restricted to bounded words; it partitions all words of length in
`[1, L]` into equivalence classes.

Your job: output a set of words that touches **as many distinct classes as possible**.
Because two equivalent words count as the *same* class, raw distinctness is an illusion
— what matters is distinctness **modulo `~`**.

## Input (stdin)
```
k L Nmax
m
l_1 r_1
...
l_m r_m
```
`k` = alphabet size (`2 <= k <= 4`), `L` = maximum word length (`5 <= L <= 7`),
`Nmax` = the maximum number of words you may output (it equals the total number of
classes). Then `m` rules, each as two digit-strings `l r`.

## Output (stdout)
Up to `Nmax` words, one per line (whitespace-separated also accepted). Each word must be
a non-empty string over `{0..k-1}` of length at most `L`.

## Feasibility
The submission is rejected (score 0) if it contains more than `Nmax` tokens, or any
token is empty, longer than `L`, or contains a symbol outside `{0..k-1}`.

## Objective (maximize)
Let `F` be the number of **distinct congruence classes** represented among your words
(words landing in the same class collapse to one). The checker computes the true
congruence by closing the bounded rewrite graph, so it is `F` — not your raw word count
— that is scored.

## Scoring
The checker builds a reference density `B = round(TotalClasses / 10)` and reports
```
Ratio = min(1000, 100 * F / B) / 1000
```
A do-nothing spread of short words sits near `0.1`; hitting one representative of every
class approaches `1.0`. The score is deterministic and depends only on your word set.

## Why the obvious move fails
The tempting approach is to enumerate words and keep the ones that are distinct **as
strings** — e.g. the `Nmax` shortlex-first words. But many short strings already lie in
the same class: `l` and `r` of every rule are equivalent, and — crucially — the system
is **not confluent**. Divergent rules let one word reduce to two *different* irreducible
words that are nonetheless equivalent (they meet only by passing through a longer word).
Counting by surface form badly over-counts; after the congruence collapses them, the
real class count crashes.

The lever is to reason about the **quotient**, not the tokens: normalize toward
canonical forms and complete the system so that words reachable from a common peak are
identified. The better you approximate the true classes, the fewer of your `Nmax` slots
you waste on words that secretly coincide.

## Constraints
- `2 <= k <= 4`, `5 <= L <= 7`, `1 <= m <= 8`; every rule is length-preserving and
  shortlex-decreasing, so normalizing a single word always terminates.
- Time limit 5 s, memory 512 MB. Scoring is exact and reproducible.

## Example (illustrative — not an input of this task)
Alphabet `{0,1}`, `L = 2`, rule `10 -> 01`. Classes of words up to length 2:
`{0}`, `{1}`, `{00}`, `{11}`, `{01, 10}`. There are 5 classes. Outputting
`0 1 00 11 01` hits all 5; outputting `0 1 01 10 00` also lists 5 strings but `01` and
`10` share a class, so it hits only 4 — the raw count misleads.
