# Twelve Corpora, One Dictionary

## Problem
A data platform stores `D = 12` fixed documents drawn from unlike domains (logs, code,
prose-like text, sequence-like text, ...). Each document is a string over lowercase
letters. You must design **one shared dictionary** — a set of short substrings — that a
fixed, simple compressor will use to shrink *all twelve* documents at once.

The compressor is a deterministic **greedy longest-prefix-match tokenizer**: scanning a
document left to right, at every position it uses the *longest* dictionary entry that
matches the text starting there; if no entry matches, it emits the single next character
as a literal token. The compressed size of a document is its number of emitted tokens
(dictionary tokens and literal tokens both cost exactly one token). A document's
**compression ratio** is `original_length / compressed_size` (higher = more compression).

Your dictionary has a shared **budget**: the total number of characters across all chosen
entries may not exceed `K`, and the number of entries may not exceed `M`. Every entry must
have length between `minlen` and `maxlen` (inclusive), and entries must be pairwise
distinct.

The catch: you are scored on the **worst** document, not the average. A dictionary that
squeezes one huge, easy-to-compress document while leaving a small document untouched
scores no better than doing nothing for that small document. Some substrings are private
to one domain; others recur across several domains — spending one entry on such a
cross-domain stem can raise several documents' ratios for a single budget cost, which is
not the same thing as that stem simply having a high total occurrence count.

## Input (stdin)
```
D K M minlen maxlen
doc_1
doc_2
...
doc_D
```
`D` documents follow, one per line, each a nonempty string of lowercase letters
(`a`-`z`), no whitespace inside a document.

## Output (stdout)
```
E
entry_1
entry_2
...
entry_E
```
Print the number of dictionary entries `E`, then the `E` entries, one per line. `E = 0`
is legal (an empty dictionary).

## Feasibility
An output is valid iff **all** hold:
- each entry is a string of lowercase letters with `minlen <= length <= maxlen`;
- the `E` entries are pairwise distinct;
- the sum of entry lengths is `<= K`;
- `E <= M`.
Any violation scores `Ratio: 0.0`.

## Objective
For a feasible dictionary, compress every document with the longest-prefix-match
tokenizer described above and compute each document's ratio `original_length /
compressed_size`. Maximize `F = min` over the 12 documents of this ratio.

## Scoring
Let `B` be the checker's own trivial construction: the **empty dictionary**. With no
entries, every character is a literal token, so every document's ratio is exactly `1.0`
and `B = 1.0`. With maximization normalization:
```
sc = min(1000.0, 100.0 * F / max(1e-9, B))
Ratio = sc / 1000.0
```
Reproducing the empty dictionary scores `Ratio = 0.1`; pushing the worst document's ratio
to `10x` the baseline caps the score at `1.0`.

## Constraints
- `D = 12`; document lengths range roughly `300`-`4200` characters.
- `4 <= minlen <= maxlen <= 9`; `K` and `M` are given per test (typically `K` around
  `180`-`260` characters, `M` around `40`).
- Time limit 5s, memory 512m.

## Example
*(Illustrative FORM only — a tiny hypothetical instance, not from the real generator.)*
Suppose `D = 2` (in the real tests `D = 12`), `minlen = 3`, `maxlen = 5`, documents
`"cabcabcabx"` (length 10) and `"zzcabqq"` (length 7). The empty dictionary gives
`B = 1.0` (both ratios are `1.0`). Now submit the dictionary `{"cab"}` (cost 3 chars).
Tokenizing doc 1: `cab|cab|cab|x` -> 4 tokens -> ratio `10/4 = 2.5`. Tokenizing doc 2:
`z|z|cab|q|q` -> 5 tokens -> ratio `7/5 = 1.4`. The worst document is doc 2, so
`F = 1.4`, giving `sc = 100*1.4/1.0 = 140`, `Ratio = 0.14`. A dictionary entry that only
helped doc 1 (e.g. a longer private substring of doc 1) would leave doc 2 at ratio `1.0`
and score no better than the empty dictionary — spending the shared budget on whichever
document is *currently worst off* is what raises the score.
