# Scarce-Rhyme Sonnet Lattice

## Problem
You are handed a synthetic lexicon and must compose a 14-line sonnet with the classic
Shakespearean rhyme scheme **ABAB CDCD EFEF GG**. Every word in the lexicon has exactly 2
syllables, so every line is exactly **5 words** (10 syllables). Each word carries three fixed
properties: a 2-bit **stress pattern** (`"01"`, `"00"`, `"10"`, or `"11"`, read as
unstressed/stressed per syllable), a **rhyme class** (an integer), and an **initial phoneme**
(a letter).

The canonical iambic foot is `"01"` (unstressed-stressed). A word placed as the `s`-th word of
a line (`s = 0..4`, 0-indexed) occupies syllables `2s` and `2s+1` of that line; wherever its
stress bits differ from `"01"`, that syllable position is an **inversion**. The lexicon
deliberately has only a **handful of rhyme classes** — far fewer than the 7 rhyme-scheme pairs
that need one — so several pairs are forced to share a class, and within a class only a few
words carry the cheap `"01"` stress. Reading and rhyme-planning are therefore inseparable: which
class you commit each pair to determines how much word variety and stress budget is left for
every other pair, especially the closing couplet.

## Input (stdin)
```
W BUDGET
STRESS_1 RHYME_1 INIT_1
...
STRESS_W RHYME_W INIT_W
```
`W` words, 0-indexed. `STRESS_i` is a 2-character string over `{0,1}`. `RHYME_i` is an integer
rhyme class. `INIT_i` is a single letter. `BUDGET` is the maximum total inversions allowed
across the whole poem.

## Output (stdout)
Print exactly 14 lines, each with 5 space-separated integers — word indices (0-indexed, valid
range `[0, W)`) — giving that line's words in order.

## Feasibility
1. Exactly 14 lines, each with exactly 5 valid integer word indices.
2. **Rhyme**: for each scheme pair (lines 1&3, 2&4, 5&7, 6&8, 9&11, 10&12, 13&14), the LAST
   (5th) word of both lines must share the same rhyme class.
3. **Meter-inversion budget**: summed over all 70 word-placements, the total number of syllable
   positions where a word's stress bit differs from the canonical `"01"` foot must not exceed
   `BUDGET`.
Any violation scores `Ratio: 0.0`.

## Objective
Let `U` be the 70 words in reading order (line-major). Maximize
```
F = distinct_word_count(U)
  + longest_alliteration_chain(U)
  + 6 * entropy_bits(inversion positions)
  + 2 * distinct_rhyme_words
```
- `distinct_word_count` = number of distinct word indices used anywhere in the poem.
- `longest_alliteration_chain` = the longest run of consecutive words in `U` that (a) all share
  the same initial phoneme and (b) are pairwise distinct (a repeated word index ends the run).
- `inversion positions`: every mismatched syllable (a value in `0..9`, the syllable's position
  within its line) contributes one entry to a multiset pooled over the whole poem;
  `entropy_bits` is that multiset's Shannon entropy in bits (0 if there are no inversions).
- `distinct_rhyme_words` = number of distinct word indices among the 14 rhyme-critical (5th)
  words specifically — this isolates the payoff of *which classes you reserved for which pairs*
  from whatever the other 56 (non-rhyme) words separately achieve: concentrating every pair on
  one class caps this term low even if the free words elsewhere look fine.

## Scoring
The checker builds its own baseline: it finds the single rhyme class with the most `"01"`-stress
words and cycles only through that class's distinct words for the entire poem (trivially valid,
0 inversions, no cross-class reasoning). Call its objective value `B`. With
`sc = min(1000, 100*F/B)`, print `Ratio: sc/1000` — reproducing the baseline scores `Ratio ≈ 0.1`.

## Constraints
- `20 <= W <= 70`. `BUDGET >= 0`. Time limit 5s, memory 512m.

## Example
Suppose the richest class has 3 distinct `"01"` words (indices 2, 5, 9). The baseline poem
cycles `2,5,9,2,5,9,...` for all 70 slots: `distinct_word_count=3`,
`longest_alliteration_chain=3` (cycling period 3 breaks the "pairwise distinct" rule on the 4th
repeat), `entropy=0` (0 inversions), `distinct_rhyme_words=3` (only 3 distinct words ever land
on a 5th-word slot), so `B = 3 + 3 + 0 + 2*3 = 12`. A submission reaching `F = 24` (e.g. by
spreading pairs across several classes and building longer alliteration runs) scores
`Ratio = min(1000, 100*24/12)/1000 = 0.2` — larger lexicons and tighter rhyme scarcity keep `B`
and the achievable `F` both larger, leaving real headroom between baseline, competent, and
expert play.
