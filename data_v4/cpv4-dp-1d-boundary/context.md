# Cutting a measured rod into billets of bounded length

## Research question

A rod is divided into `n` consecutive unit-segments numbered `0..n-1`; segment `i` carries an integer
"imbalance contribution" `v[i]` (it may be negative). You operate a saw that must cut the whole rod
into contiguous **billets**, where each billet is a maximal run of consecutive segments. The saw has a
mechanical limitation: every billet must contain between `L` and `R` segments, inclusive — never fewer
than `L`, never more than `R`. The billet covering segments `[j, i)` (segments `j, j+1, ..., i-1`)
costs

```
K + |v[j] + v[j+1] + ... + v[i-1]|
```

i.e. a fixed setup fee `K` per cut piece plus the absolute value of the imbalance accumulated over that
billet. You must cut the *entire* rod (cover all `n` segments, with billets that tile `[0, n)` exactly)
and you want to **minimize the total cost** over all billets. If it is impossible to tile `[0, n)` with
billets each of length in `[L, R]`, report `-1`. The empty rod (`n = 0`) needs no cuts and costs `0`.

This is a one-dimensional partition DP. The whole difficulty is the *boundary arithmetic*: a billet is a
half-open range `[j, i)` of length `i - j`, and the transition has to translate "length between `L` and
`R` inclusive" into the exact inclusive/exclusive bounds on the predecessor index `j`. Get that window
off by one and you silently admit billets that are one segment too short (or too long), or you forbid a
legal cut — and the error only shows up on tightly-constrained inputs.

## Input / output contract

- Input (stdin): the first line has four integers `n K L R`
  (`0 <= n <= 2*10^5`, `0 <= K <= 10^9`, `1 <= L <= R <= 50`).
  The second line has `n` integers `v[i]` (`-10^9 <= v[i] <= 10^9`), whitespace-separated.
  (When `n = 0` the second line is empty or absent.)
- Output (stdout): a single line with the minimum total cost, or `-1` if no valid tiling of `[0, n)`
  into billets of length in `[L, R]` exists.
- Time limit: 1 second. Memory: 256 MB.

Example: for `n = 6`, `K = 2`, `L = 2`, `R = 3`, `v = [3, 3, -5, -5, 3, 3]`, the answer is `6`
(cut into `[0, 3)` and `[3, 6)`: each billet sums to `1`, costing `2 + |1| = 3`, total `6`).

## Background

The cost of a billet depends only on the *sum* of `v` over its segments, so prefix sums
`S[i] = v[0] + ... + v[i-1]` (with `S[0] = 0`) turn the imbalance of `[j, i)` into `S[i] - S[j]` in
`O(1)`. Two families of approach are on the table before committing:

- **Fixed-length greedy / heuristic.** Always cut billets of one chosen length (say always `R`, or always
  `L`), or greedily extend a billet until adding the next segment would stop reducing `|sum|`. This is
  `O(n)` and trivial, but because `v` has negatives the optimal split is the one that keeps each billet's
  running sum near zero, and that balance point does not align with any fixed length — so the open
  question is whether any greedy rule is actually optimal.
- **Linear partition DP.** Let `dp[i]` be the cheapest way to tile the prefix `[0, i)`. The last billet
  ends at `i` and starts at some `j` with `L <= i - j <= R`, so `dp[i] = min over valid j of
  dp[j] + K + |S[i] - S[j]|`. This is `O(n * (R - L + 1))`, which with `R <= 50` is near-linear; the open
  question is the *exact* range of `j` to scan — the inclusive/exclusive boundary that is the heart of
  this task.

## Evaluation settings

Judged on hidden tests covering: feasible tilings with mixed positive/negative `v`; infeasible instances
where `L` is too large to tile (answer `-1`), including the sharp `n` slightly below `L` and
`L = R` exact-divisor mismatches; `n = 0` (answer `0`); single segment with `L = 1`; all-`v`-equal arrays;
tight bands `L = R` where every billet length is forced; and large `n = 2*10^5` with `|v|` near `10^9`
(so the accumulated cost exceeds a 32-bit integer and the per-`i` window must stay `O(R)`).

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    long long K;
    int L, R;
    if (!(cin >> n >> K >> L >> R)) return 0;
    vector<long long> v(n);
    for (auto &x : v) cin >> x;

    // prefix sums S[0..n], S[i] = v[0] + ... + v[i-1]
    vector<long long> S(n + 1, 0);
    for (int i = 0; i < n; i++) S[i + 1] = S[i] + v[i];

    // TODO: dp[i] = min cost to tile [0, i) with billets of length in [L, R];
    //       a last billet [j, i) is legal iff L <= i - j <= R. Print dp[n] or -1.
    long long answer = 0;

    cout << answer << "\n";
    return 0;
}
```
