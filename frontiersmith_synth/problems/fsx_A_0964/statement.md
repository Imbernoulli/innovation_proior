# The Kessar Tablets

Excavators found a hoard of clay tablets in the extinct **Kessar** script. Every
surviving word is transliterated as a string over the three attested glyphs
`A`, `B`, `C`. A prior epigrapher has already aligned a training corpus: pairs
`(word, root)` where `word` is an attested surface form and `root` is its
reduced canonical root, produced by repeatedly applying the (unknown) grammar
rules of Kessar until no rule applies any more.

Kessar's grammar consists of a hidden set of rewrite rules. Each rule has a
left-hand side `L` (a substring of length 2 to 5 over `{A,B,C}`) and a
right-hand side `R` (a *single* glyph). Applying a rule replaces one
occurrence of `L` in the current string with `R` — the string strictly
shortens every time a rule fires, so repeated application always terminates.
Different rules' left-hand sides may **overlap** (e.g. one rule's `L` may be a
prefix of another rule's `L`); when several rules could fire at the
left-most position where *something* matches, Kessar's grammar has a fixed
(hidden) **priority order** that breaks the tie — you never see this order
directly, only its effect on the training pairs.

**Illustrative FORM only — NOT the hidden grammar:** a toy rule set might be
`{AB->C, BCA->A, CCAB->B}` with priority `BCA->A` > `AB->C` > `CCAB->B`. The
real Kessar grammar for this tablet is different; you must recover its shape
from the data.

The corpus you are given only contains **short** attested words (length ≤ 9).
You will be graded on **long** inscriptions (length 24–36) that were never
observed — composing many rule firings, including firings of longer rules
that essentially never manifest inside a length ≤ 9 window but dominate long
inscriptions.

## Input (stdin)

```
N t
w_1 r_1
w_2 r_2
...
w_N r_N
```
`t` is the tablet id. Each of the `N` following lines is an attested word
`w_i` and its fully-reduced root `r_i` (both non-empty strings over
`{A,B,C}`).

## Output (stdout): your reconstructed grammar

Emit zero or more lines, one candidate rule per line:
```
priority  L  R
```
`priority` is any finite number (lower fires first when multiple rules
match the same left-most position); `L` is 2–5 glyphs; `R` is exactly one
glyph. Order of lines does not matter — only the `priority` values do (ties
break by line order).

## Feasibility

Every line must parse as exactly three whitespace-separated tokens: a finite
number, a glyph string of length 2–5 using only `A`,`B`,`C`, and a single
glyph in `A`,`B`,`C`. At most 80 rules. An empty submission, any malformed
line, any out-of-alphabet character, or a non-finite priority scores `0`.

## Objective (maximize)

The grader regenerates the same hidden grammar from `t`, draws 250 held-out
long inscriptions (never shown to you), and reduces each one twice: once with
the true hidden grammar (leftmost, highest-priority rule fires first,
repeated to a fixed point) and once with **your** submitted rule set under
the identical leftmost / highest-priority procedure. For each held-out word
let `sim = 1 - lev(your_root, true_root) / max(|your_root|,|true_root|,1)`
(normalized Levenshtein similarity, `1.0` = exact match). Let `F` be the mean
`sim` over the 250 held-out words, and `B` the mean `sim` achieved by the
**identity baseline** (predicting the root equal to the raw word, i.e.
submitting no rules). The score is
```
Ratio = min(1000, 100 * F / B) / 1000
```
Reproducing the training pairs verbatim is not enough — the grader never
checks training reproduction, only extrapolation to long, never-seen
compositions.

## Why short words are a trap

In a word of length ≤ 9, a rule with a 4- or 5-glyph left-hand side almost
never gets the chance to fire — most of what you observe is explained by the
short (length-2/3) rules alone. A learner that only ever looks for 2- and
3-glyph substitutions will reproduce the training corpus perfectly and still
be systematically wrong on long inscriptions, where the longer rules fire
constantly and interact with rules discovered from shorter windows. Recovering
the true grammar requires checking, length by length, whether the rules found
so far already explain each training pair — and only crediting a new, longer
rule when they provably don't.

## Constraints

`N` is at most a few hundred. Time limit 5 s, memory 512 MB. Scoring is fully
deterministic.
