# Counting binary strings with no long run of ones, modulo p

## Research question

A binary string is *good* if it never contains a run of `k` or more consecutive
ones (equivalently, the longest block of `1`s has length at most `k-1`). Given
`N`, `k`, and a modulus `p`, count how many good binary strings of length `N`
there are, and report the count modulo `p`.

You must answer several independent queries in one run.

## Input / output contract

- Input (stdin):
  - The first line contains an integer `T` — the number of queries.
  - Each of the next `T` lines contains three integers `N k p`:
    - `N` — the length of the strings (`0 <= N <= 10^18`),
    - `k` — the forbidden run length (`1 <= k <= 50`); a string is good iff it
      has no run of `k` or more consecutive ones,
    - `p` — the modulus (`2 <= p <= 10^9`); `p` is **not** guaranteed to be prime.
- Output (stdout): `T` lines, the `i`-th being the number of good strings of
  length `N_i` modulo `p_i`.
- Constraints: `1 <= T <= 2*10^5`.
- Time limit: 2 seconds. Memory: 256 MB.

The empty string (`N = 0`) is good, so its count is `1` (reported as `1 mod p`).

### Sample

Input:

```
3
3 2 1000000007
3 3 1000000007
4 4 1000000007
```

Output:

```
5
7
15
```

Explanation. For `N = 3, k = 2` the good strings are exactly those with no two
adjacent ones: `000, 001, 010, 100, 101` — that is `5`. For `N = 3, k = 3` only
the single all-ones string `111` is forbidden, leaving `8 - 1 = 7`. For
`N = 4, k = 4` only `1111` is forbidden, leaving `16 - 1 = 15`.

## Evaluation settings

Judged on hidden tests covering: tiny lengths; the boundary `N = k` and
`N = k+1`; `k = 1`; `k = 2`; the full range up to `N = 10^18` with `k` up to
`50`; prime moduli such as `10^9 + 7` and `998244353`; and composite moduli.
Many queries are batched in a single input, so the per-query cost must be
small.

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int T;
    if (!(cin >> T)) return 0;
    while (T--) {
        long long N, k, p;
        cin >> N >> k >> p;

        // TODO: count good binary strings of length N (no run of k or more
        // consecutive ones), modulo p, and print the result on its own line.
        long long answer = 0;

        cout << answer << "\n";
    }
    return 0;
}
```
