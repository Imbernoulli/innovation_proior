# Loading a locker with a fire-safety buffer

## Research question

A self-storage company rents a single locker whose interior is a row of `K` integer space
units (positions `1 .. K`). Fire code requires that the **last `g` units must be left empty**
as a buffer, so only the first `K - g` units may actually be occupied. You are shown `n`
candidate items; item `i` occupies `s[i]` contiguous-equivalent space units and is worth
`v[i]`. You may store any subset of the items (each item at most once). A subset is *legal*
if the **total space it occupies is at most the usable amount** `U = K - g`. Among all legal
subsets, output the **maximum total value**. The empty subset is always legal, so the answer
is at least `0`.

## Input / output contract

- Input (stdin): the first line has three integers `n`, `K`, `g`
  (`1 <= n <= 2000`, `0 <= K <= 2*10^5`, `0 <= g <= 2*10^5`). Then `n` lines follow, the
  `i`-th containing `s[i]` and `v[i]` (`1 <= s[i] <= 2*10^5`, `1 <= v[i] <= 10^9`).
- Output (stdout): a single line with the maximum achievable total value.
- Time limit: 1 second. Memory: 256 MB.

Note that `g` may be `>= K`; then the usable amount `U = K - g` is non-positive and must be
treated as `0` (no item fits, answer `0`). An item with `s[i] > U` simply cannot be stored.

Example: `n = 4`, `K = 10`, `g = 3` (so `U = 7`), items `(s,v) = (3,8), (4,9), (5,10), (2,5)`.
The answer is `17`: items `0` and `1` occupy `3 + 4 = 7 <= 7` for value `8 + 9 = 17`, which beats
any single item and also beats the other space-`7` combination `(5,10)+(2,5)=15`.

## Evaluation settings

Judged on hidden tests covering: the buffer exactly consuming the locker (`g = K`, so
`U = 0`), the buffer exceeding the locker (`g > K`), no buffer at all (`g = 0`, plain
knapsack), items that exactly fill the usable space, items strictly too large for `U`,
single-item and `n = 2000` cases, and values near `10^9` with many items selected.

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    long long K, g;
    if (!(cin >> n >> K >> g)) return 0;
    vector<long long> s(n), v(n);
    for (int i = 0; i < n; i++) cin >> s[i] >> v[i];

    // Usable space U = K - g (clamp to 0 if non-positive).
    // TODO: choose a subset of items (each at most once) with total space at
    //       most U; print the maximum achievable total value.
    long long answer = 0;

    cout << answer << "\n";
    return 0;
}
```
