# Resonance offsets: translation-invariant pattern matching on an integer stream

## Research question

You are given a *text* sequence of `n` integers `t[0..n-1]` and a *pattern* sequence of `m` integers
`p[0..m-1]`. All values may be **negative, zero, or positive**. The pattern *resonates* at start
position `i` of the text (with `0 <= i <= n - m`) if there exists a single constant offset `c` such
that

```
t[i + j] = p[j] + c   for every j in [0, m - 1].
```

In words: the window `t[i..i+m-1]` is an exact copy of the pattern shifted up or down by one fixed
amount. The shift `c` may differ between positions and may itself be negative or zero. Report how many
positions resonate, and list them in increasing order.

This is the integer-stream analogue of substring search: it shows up when matching melodies up to
transposition, sensor traces up to a baseline drift, or price series up to an additive bias. Getting
the *length-edge* corners right — empty pattern, single-element pattern, all-negative data, the
overflow of a difference of two large values — is where naive code breaks.

## Input / output contract

- Input (stdin), whitespace-separated:
  - line/token group 1: `n` (`0 <= n <= 2*10^5`),
  - then `n` integers `t[i]` (`-10^9 <= t[i] <= 10^9`),
  - then `m` (`0 <= m <= 2*10^5`),
  - then `m` integers `p[j]` (`-10^9 <= p[j] <= 10^9`).
- Output (stdout), exactly two lines:
  - line 1: the number `k` of resonance positions,
  - line 2: the `k` positions in increasing order, space-separated (an **empty line** if `k = 0`).
- Time limit: 1 second. Memory: 256 MB.

Example: for `t = [-1, 0, -2, 2, 3, 1]` and `p = [5, 6, 4]`, the answer is `2` with positions
`0 3`. At position 0 the offset is `c = -6` (`-1,0,-2 = 5,6,4 minus 6`); at position 3 the offset is
`c = -3` (`2,3,1 = 5,6,4 minus 3`).

## Background

Because the offset `c` is free, absolute values do not matter — only the *shape* of the sequence
does. Subtracting consecutive entries cancels `c`: if `t[i+j] = p[j] + c` for all `j`, then
`t[i+j+1] - t[i+j] = p[j+1] - p[j]`. So a resonance is exactly a place where the length-`(m-1)`
sequence of **consecutive differences** of the pattern occurs, as a contiguous block, inside the
length-`(n-1)` sequence of consecutive differences of the text. That reduces the problem to ordinary
exact substring search over an integer alphabet, which Knuth–Morris–Pratt solves in linear time.

Two families of approach are on the table before committing:

- **Naive windowed check.** For every start `i`, fix `c = t[i] - p[0]` and verify all `m` entries.
  This is `O(n*m)` and obviously correct, but quadratic in the worst case (e.g. a pattern that almost
  matches everywhere). Fine as a brute-force oracle; too slow for the stated limits.
- **KMP on the difference sequences.** Build `pd[j] = p[j+1] - p[j]` and `td[j] = t[j+1] - t[j]`,
  then run KMP to find every occurrence of `pd` in `td`. This is `O(n + m)`. The open questions are
  the index bookkeeping (a match ending in the difference array maps back to a text start position)
  and — crucially — the **degenerate lengths** `m = 0` and `m = 1`, where the difference sequence is
  empty and the generic KMP loop says nothing.

## Evaluation settings

Judged on hidden tests covering: patterns with negatives and zeros mixed in; all-negative text and
pattern; the empty pattern (`m = 0`); the single-element pattern (`m = 1`, which matches at *every*
position); patterns longer than the text (`m > n`, zero positions); `n = 0`; and large `n = 2*10^5`
with values near `±10^9`, so that a single difference `t[i+1] - t[i]` can reach `2*10^9` and overflow
a 32-bit integer.

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n, m;
    if (!(cin >> n)) return 0;
    vector<long long> t(n);
    for (auto &x : t) cin >> x;
    cin >> m;
    vector<long long> p(m);
    for (auto &x : p) cin >> x;

    // TODO: count and list the start positions where p resonates in t (match up to a
    //       single additive offset). Handle the m = 0 and m = 1 base cases first.

    return 0;
}
```
