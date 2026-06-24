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

This is a 0/1 knapsack whose capacity is not given directly but is `K - g`, a *difference of
two boundaries*. Getting the usable capacity exactly right — and iterating the DP table over
exactly the right inclusive range — is the whole game: a one-unit slip in `U` or in the loop
bounds silently lets in (or wrongly forbids) a configuration that sits right on the edge.

## Input / output contract

- Input (stdin): the first line has three integers `n`, `K`, `g`
  (`1 <= n <= 2000`, `0 <= K <= 2*10^5`, `0 <= g <= 2*10^5`). Then `n` lines follow, the
  `i`-th containing `s[i]` and `v[i]` (`1 <= s[i] <= 2*10^5`, `1 <= v[i] <= 10^9`).
- Output (stdout): a single line with the maximum achievable total value.
- Time limit: 1 second. Memory: 256 MB.

Note that `g` may be `>= K`; then the usable amount `U = K - g` is non-positive and must be
treated as `0` (no item fits, answer `0`). An item with `s[i] > U` simply cannot be stored.

Example: `n = 4`, `K = 10`, `g = 3` (so `U = 7`), items `(s,v) = (3,8), (4,9), (5,10), (2,5)`.
The answer is `15`. Wait — check it: items `0` and `1` occupy `3 + 4 = 7 <= 7` for value
`8 + 9 = 17`, which beats any single item and beats `(5,10)+(2,5)=15`. The answer is `17`.

## Background

The constraint "total occupied space at most `U`, each item used at most once" is the
textbook 0/1 knapsack. Two design questions are live before committing:

- **What is the capacity, exactly?** It is `U = K - g`, a subtraction of two given boundaries.
  Because the locker positions are `1 .. K` and the *last* `g` of them are reserved, the
  usable positions are `1 .. (K - g)`, i.e. exactly `K - g` units. Whether that should be
  `K - g`, `K - g + 1`, or `K - g - 1` is precisely the inclusive/exclusive boundary that has
  to be nailed down by counting, not guessed.
- **How to iterate the DP.** The standard one-dimensional table `dp[c]` = best value using
  occupied space at most `c` is filled per item by scanning `c` from high to low so each item
  is used at most once. The open question is the exact range of `c` (inclusive of `U`?) and
  the lower cutoff (`c >= s[i]`), both of which are off-by-one-prone.

## Evaluation settings

Judged on hidden tests covering: the buffer exactly consuming the locker (`g = K`, so
`U = 0`), the buffer exceeding the locker (`g > K`), no buffer at all (`g = 0`, plain
knapsack), items that exactly fill the usable space, items strictly too large for `U`,
single-item and `n = 2000` cases, and values near `10^9` with many items selected so the
total exceeds the 32-bit range (`sum` can reach `~2*10^12`, requiring 64-bit accumulation).

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
    // TODO: 0/1 knapsack with capacity U; print the best value reachable with
    //       total occupied space at most U.
    long long answer = 0;

    cout << answer << "\n";
    return 0;
}
```
