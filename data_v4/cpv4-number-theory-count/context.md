# Counting distinct fraction values in an N x N grid

## Research question

Fix an integer `N`. Look at every ordered pair `(a, b)` with `1 <= a <= N` and `1 <= b <= N`, and
read it as the rational number `a / b`. Many different pairs name the *same* number: `1/2`, `2/4`,
and `3/6` are all the value `0.5`. The question is how many **distinct rational values** the whole
grid produces. Formally, two pairs `(a, b)` and `(c, d)` denote the same value iff `a*d = b*c`; count
the size of the set of distinct values, i.e. the number of equivalence classes.

You must answer this for `T` independent queries, each with its own `N`.

This is a deduplication-counting problem: the naive `N*N` count is wildly too large because of the
collisions, and the natural "fix" — count each reduced fraction once — is exactly where an off-by-one
or a double-count of the symmetric pair (`p/q` vs `q/p`) or the self-value `1/1` creeps in. Getting
the bookkeeping of *which fractions to count and how many times* exactly right is the whole task.

## Input / output contract

- Input (stdin): the first token is `T` (`1 <= T <= 5`), the number of queries. Then `T` integers
  follow, one per query: `N` (`1 <= N <= 10^7`), whitespace-separated.
- Output (stdout): for each query, a single line with the number of distinct rational values `a / b`
  with `1 <= a, b <= N`.
- Time limit: 2 seconds. Memory: 256 MB.

Example: for `N = 3` the grid pairs reduce to the seven distinct values
`{1/3, 1/2, 2/3, 1/1, 3/2, 2/1, 3/1}`, so the answer is `7`. Note `1/1` is produced by `(1,1)`,
`(2,2)`, and `(3,3)` but is counted once.

## Evaluation settings

Judged on hidden tests covering: the minimum `N = 1` (answer `1`, only the value `1/1`); small `N`
where the distinct values can be enumerated by hand; several mid-range `N`; and the maximum
`N = 10^7` repeated across all `T` queries, where an `O(N^2)` approach is hopeless.

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int t;
    if (!(cin >> t)) return 0;
    vector<int> ns(t);
    int maxN = 1;
    for (int i = 0; i < t; i++) {
        cin >> ns[i];
        maxN = max(maxN, ns[i]);
    }

    // TODO: compute the number of distinct fraction values a/b with
    // 1 <= a, b <= N for each query and print it.

    for (int i = 0; i < t; i++) {
        long long ans = 0;
        cout << ans << "\n";
    }
    return 0;
}
```
