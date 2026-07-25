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

This shows up when matching melodies up to transposition, sensor traces up to a baseline drift, or
price series up to an additive bias. Getting the *length-edge* corners right — empty pattern,
single-element pattern, all-negative data — is where naive code breaks.

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

## Evaluation settings

Judged on hidden tests covering: patterns with negatives and zeros mixed in; all-negative text and
pattern; the empty pattern (`m = 0`); the single-element pattern (`m = 1`); patterns longer than the
text (`m > n`); `n = 0`; and large `n = 2*10^5` with values near `±10^9`.

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
