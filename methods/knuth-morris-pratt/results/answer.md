# The Knuth–Morris–Pratt string-matching algorithm

## Problem

Find all occurrences of a nonempty pattern `W` of length `m` inside a text `S` of length `n`. The naive scanner re-tries the pattern at every text start, costs `O(nm)` in the worst case (e.g. `a^k b` in `a^N b`), and **backs the text pointer up** after a failed partial match — which forces an editor reading from a file to buffer characters it has already passed.

## Key idea

Scan the text **strictly left to right, never moving the text pointer backward**. When a mismatch occurs after matching `j−1` pattern characters, the matched prefix already determines how far the pattern can be safely slid; that "how far" depends only on the pattern, so precompute it once into a *failure table*. Recovery on mismatch is done entirely by shrinking the pattern pointer through the table; the text pointer only ever advances. The result is `O(m + n)` time, `O(m)` memory, alphabet-independent.

The table is, in spirit, the linear-time simulation of a backtracking automaton with its repeated sub-computations shared — the mechanism that makes a possibly-exponential two-way pushdown machine recognizable in linear time (Cook, 1972), distilled down to a single array indexed by matched-prefix length.

## The algorithm

Maintain a text pointer `k` and a pattern pointer `j`, with the pattern aligned so `pattern[1]` sits at text position `k − j + 1`.

- If `text[k] = pattern[j]`: advance both.
- Else: `j := next[j]` (slide the pattern), repeat until match or `j = 0` (slide fully past, advance text). `k` never decreases.

The table. Let `f[j]` = length+1 of the longest proper border of `pattern[1..j−1]` (largest `i < j` with `pattern[1..i−1] = pattern[j−i+1..j−1]`; `f[1] = 0`). The refined failure table is equivalently the largest `i < j` satisfying that same border condition and `pattern[i] ≠ pattern[j]`, or `0` if no such `i` exists. Set `next[1] = 0`; for `j > 1` it can be computed from `f` by avoiding a guaranteed-failing re-comparison:
```
next[j] = f[j]        if pattern[j] ≠ pattern[f[j]]
        = next[f[j]]  if pattern[j] = pattern[f[j]]
```
It is built by matching the pattern against itself, in `O(m)`. The extra final slot stores the restart border length after a complete match, so overlapping matches are found.

Correctness rests on the invariant (with `p = k − j`): `text[p+i] = pattern[i]` for `1 ≤ i < j`, and no full match begins left of `p`. Running time: the text pointer advances ≤ `n` times; every `j := next[j]` strictly decreases `j`, and every decrease can be charged to a previous increase of `j` caused by advancing the text, so it fires ≤ `n` times total — `O(n)` matching, `O(m)` preprocessing.

Sharpness: with the refined `next`, the number of consecutive `j := next[j]` steps while one text character is scanned is at most `1 + log_φ m` (`φ` = golden ratio), and Fibonacci strings `b₁=b, b₂=a, bₙ=bₙ₋₁bₙ₋₂` achieve it. The unrefined `f` loses this per-character bound (pattern `a^m` would do `m` steps at one character) though it keeps overall linearity.

## Working code

```python
def preprocess(W):
    # Failure / "next" table from the pattern alone (O(m)).
    # For k < len(W), T[k] resumes after a mismatch at W[k] without
    # moving the text pointer. T[len(W)] restarts after a full match.
    # T[0] = -1 is the sentinel for "no prefix survives."
    T = [0] * (len(W) + 1)
    T[0] = -1
    pos, cnd = 1, 0          # cnd = current border length (the f[j] role)
    while pos < len(W):
        if W[pos] == W[cnd]:
            T[pos] = T[cnd]   # refinement: resuming at cnd would re-mismatch
        else:
            T[pos] = cnd
            while cnd >= 0 and W[pos] != W[cnd]:
                cnd = T[cnd]  # slide the pattern against itself
        pos += 1
        cnd += 1
    T[pos] = cnd
    return T


def search(S, W):
    T = preprocess(W)
    matches = []
    k = 0   # text pointer — only ever advances
    j = 0   # pattern pointer
    while k < len(S):
        if W[j] == S[k]:
            k += 1
            j += 1
            if j == len(W):
                matches.append(k - j)   # match starts at k - j
                j = T[j]
        else:
            j = T[j]                    # slide pattern, text pointer fixed
            if j < 0:
                k += 1
                j += 1
    return matches
```

Verified: `preprocess("ABCDABD") == [-1, 0, 0, 0, -1, 0, 2, 0]`; `search("babcbabcabcaabcabcabcacabc", "abcabcacab") == [15]`; `search("aaaaaaaaaaaaaab", "aaaaaaab") == [7]`.
