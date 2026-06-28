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

Single self-contained C++17 program. It reads the pattern `W` on the first line of stdin and the text `S` on the second, and prints the 0-based start positions of every occurrence (space-separated; an empty line if none). Lengths use `long long` so indices never overflow on long texts.

```cpp
// Knuth-Morris-Pratt exact string matching, single-file C++17.
// Reads two lines from stdin: line 1 = pattern W, line 2 = text S.
// Prints the 0-based start positions of every occurrence of W in S
// (space-separated on one line; an empty line if there are none).
#include <bits/stdc++.h>
using namespace std;

// Failure / "next" table from the pattern alone (O(m)).
// For k < m, T[k] resumes after a mismatch at W[k] without moving the text
// pointer. T[m] restarts after a full match (so overlapping matches are found).
// T[0] = -1 is the sentinel for "no prefix survives."
vector<long long> preprocess(const string& W) {
    long long m = (long long)W.size();
    vector<long long> T(m + 1, 0);
    T[0] = -1;
    long long pos = 1, cnd = 0;     // cnd = current border length (the f[j] role)
    while (pos < m) {
        if (W[pos] == W[cnd]) {
            // border extends AND the next chars agree, so resuming at cnd would
            // just re-mismatch the same text char -- short-circuit it
            // (this is the pattern[i] != pattern[j] refinement).
            T[pos] = T[cnd];
        } else {
            T[pos] = cnd;
            while (cnd >= 0 && W[pos] != W[cnd])
                cnd = T[cnd];       // slide the pattern against itself
        }
        pos += 1;
        cnd += 1;
    }
    T[pos] = cnd;
    return T;
}

// Find every occurrence of W in S; the text pointer k only ever advances.
vector<long long> search(const string& S, const string& W) {
    vector<long long> matches;
    long long n = (long long)S.size(), m = (long long)W.size();
    if (m == 0) return matches;
    vector<long long> T = preprocess(W);
    long long k = 0;                // text pointer -- only ever advances; never backs up
    long long j = 0;                // pattern pointer
    while (k < n) {
        if (W[j] == S[k]) {
            k += 1;
            j += 1;
            if (j == m) {                   // full match ends just before k
                matches.push_back(k - j);
                j = T[j];                   // resume to find further matches
            }
        } else {
            j = T[j];                       // slide the pattern, keep k fixed
            if (j < 0) {                    // sentinel: nothing survives
                k += 1;
                j += 1;
            }
        }
    }
    return matches;
}

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);
    string W, S;
    getline(cin, W);
    getline(cin, S);
    vector<long long> matches = search(S, W);
    for (size_t i = 0; i < matches.size(); ++i) {
        if (i) cout << ' ';
        cout << matches[i];
    }
    cout << '\n';
    return 0;
}
```

Verified by compiling with `g++ -O2 -std=c++17` and running: pattern `abcabcacab` in text `babcbabcabcaabcabcabcacabc` prints `15`; pattern `aaaaaaab` in `aaaaaaaaaaaaaab` prints `7`; the overlapping case `aba` in `caababa` prints `2 4`.
