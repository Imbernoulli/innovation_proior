# Palindrome-Bounded Word Stretching

## Problem
You work over a small alphabet of `a` letters, written as the decimal digits
`0, 1, ..., a-1`. A **palindromic factor** of a word is any contiguous
substring that reads the same forwards and backwards (e.g. `"010"`, `"0110"`,
or a single letter, which is trivially a length-1 palindrome).

You must produce a word (a finite string over the alphabet) in which **no
palindromic factor is longer than `p`**. Subject to that rule you want the
word to be as long as possible, up to a given target length `L`. You do not
have to reach `L` -- you are simply scored by how far you get before the rule
first breaks.

## Input (stdin)
```
a p L
```
Three integers on one line: the alphabet size `a`, the palindrome-length
bound `p`, and the target length `L`.

## Output (stdout)
A single line containing one string of decimal digits, each digit in
`{0, ..., a-1}`, with length between `1` and `L`. Print nothing else (no
extra tokens, no blank output).

## Feasibility
Your output is rejected (`Ratio: 0.0`) if any of the following holds:
- it is not exactly one whitespace-free token,
- its length is `0` or greater than `L`,
- any character is not a decimal digit, or is a digit `>= a`.

A rejected output never reaches the scoring step below.

## Objective
Let `F` be the length of the longest **prefix** of your string that contains
no palindromic factor of length greater than `p` (so `F` equals the length of
your whole string if it is entirely clean, i.e. if you never violate the
bound at all). You want to **maximize `F`**.

## Scoring
The checker builds its own trivial reference word: the naive cyclic pattern
`0, 1, 2, ..., a-1, 0, 1, 2, ...` repeated up to length `L`. It scores that
reference word by the exact same clean-prefix rule above to get a baseline
length `B > 0`. Your score is
```
sc = min(1000.0, 100.0 * F / B)
Ratio = sc / 1000.0
```
printed on the checker's last line. Matching the baseline's clean length
scores `Ratio = 0.1`; getting a clean prefix `10x` longer than the baseline
caps the score at `1.0`.

## Constraints
- `2 <= a <= 4`
- `p >= 5`, and `L` is at most a few hundred (small scale).
- Time limit 5s, memory 512MB.

## Example (illustrative, not one of the graded cases)
Suppose `a = 2`, `p = 2`, `L = 6`. The baseline cyclic word is `"010101"`
(length 6). Checking it against the rule: its length-3 prefix `"010"` is
already a palindrome of length 3, which is `> p = 2`, so the baseline's clean
prefix has length `B = 2` (the prefix `"01"` is fine; adding the third
character breaks the rule).

Now say a participant outputs `"011001"`. Scanning left to right: `"0"`
clean, `"01"` clean, `"011"` -- its longest palindromic factor is `"11"`
(length 2, OK), still clean, `"0110"` -- the whole thing is a palindrome of
length 4 `> 2`, so the rule breaks exactly here. The clean prefix before that
point has length `F = 3` (`"011"`). The score is
```
sc = min(1000.0, 100.0 * 3 / 2) = 150.0
Ratio = 0.15
```
A participant who instead manages to keep the **whole** target-length word
clean (never breaking the rule at all) gets `F = L`, scoring proportionally
higher -- how close you can push `F` to `L` while respecting a tight bound
`p` on a small alphabet is exactly the open-ended question this problem
poses.
