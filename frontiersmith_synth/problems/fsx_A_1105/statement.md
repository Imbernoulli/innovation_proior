# Deadhand Grammar: Compress a String by Committing Replacement Rules

## Problem

You are given a string `S` of length `n` over an alphabet of the first `q`
lowercase letters. You compress it by writing an **ordered list of grammar
rules**. Rule `i` introduces a fresh nonterminal (written `#i`) and a
right-hand side pattern `P_i`: a sequence of 2..`L` tokens, where each token is
either a base letter or a reference `#j` to an **earlier** rule (`j < i`).

The rules are applied to the axiom (initially `S`) **in order**. Applying rule
`i` scans the current axiom left to right and replaces **every non-overlapping
occurrence** of the token sequence `P_i` by the single token `#i`: after a
replacement at some position, the scan continues immediately after the
replaced span (so occurrences of a self-overlapping pattern cannot share
characters). Replacements are **irreversible**: the consumed characters become
one opaque token, and later rules can only match it whole.

The **grammar size** is

```
F = (# tokens left in the final axiom) + sum over rules of (len(P_i) + c)
```

where `c` is a per-rule overhead given in the input. Smaller is better.

## Input (stdin)

One line with five integers: `n K c L q`, then one line with the string `S`
(exactly `n` characters from the first `q` lowercase letters). You may commit
**at most `K` rules**.

## Output (stdout)

First a line with `m` (0 ≤ m ≤ K), the number of rules, then `m` lines, the
`i`-th being the pattern `P_i`. A pattern is a string of base letters and
references `#j` (1 ≤ j < i) with no separators, e.g. `ab#1c`. Its token count
must be between 2 and `L`.

## Feasibility

- 0 ≤ m ≤ K; exactly `m` pattern lines follow.
- Every token is a base letter (first `q` letters) or `#j` referencing an
  earlier rule; pattern token length in [2, L].
- Each rule must match **at least one** occurrence in the axiom at the moment
  it is applied (dead rules are infeasible).

Any violation scores 0.

## Objective and Scoring

Minimize `F`. The checker builds the trivial grammar itself (zero rules,
size `B = n`) and reports

```
Ratio = min(1.0, 0.1 * B / F)
```

so doing nothing scores 0.1, and a 10x size reduction caps the score at 1.0.

## Constraints

- 400 ≤ n ≤ 6000, 4 ≤ q ≤ 7, 6 ≤ K ≤ 14, 2 ≤ c ≤ 6, 8 ≤ L ≤ 12.
- Time limit 5 s, memory 512 MB. Deterministic scoring: the same output always
  yields the same ratio.

## Example (worked score)

Instance: `n=16, K=3, c=2, L=6, q=5`, `S = abxcabxcabxcabxc`.

- Output `0` (no rules): `F = 16`, Ratio = 0.100.
- Output
  ```
  1
  abxc
  ```
  The pattern `abxc` occurs 4 times; the axiom becomes `#1 #1 #1 #1` (4
  tokens). `F = 4 + (4 + 2) = 10`, Ratio = min(1, 0.1*16/10) = **0.160**.
- The most-frequent-pair move looks tempting (`ab`, `bx`, `xc` each occur 4
  times), but committing to pair `ab` yields `F = 12 + (2+2) = 16` — zero net
  gain after overhead — and spending further pair rules on `xc`, then `#1#1`,
  only returns to `F = 16`. Chasing the hottest pair while ignoring rule
  overhead, the rule budget, and the repeats a replacement forecloses is the
  classic losing line here.

## Notes

The instances mix several regimes: long tandem repeats whose period is worth
ONE rule, decoy pairs whose frequency is inflated by scattered occurrences
(and whose commitment misaligns overlapping repeats), and a rule budget `K`
too small to compress everything pair by pair. There is no known optimal
scheme; graded credit for every improvement.
