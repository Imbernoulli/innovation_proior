# Gift card with a spending window: maximum joy in a price band

## Research question

A store gift card must be spent so that the **total price of the items you buy lands inside a fixed
band** `[L, R]` — at least `L` and at most `R`, **both endpoints inclusive**. There are `n` items;
item `i` has an integer price `p[i] >= 1` and an integer **joy** `v[i]` that may be negative (some
items are duds you would normally avoid, but a dud might be the only way to nudge the total price up
into the band). You buy each item at most once. Among all subsets whose total price `s` satisfies
`L <= s <= R`, output the **maximum total joy**. If no subset has its price in the band, output
`IMPOSSIBLE`.

The empty subset has price `0` and joy `0`; it is a legal purchase **only when `L = 0`** (so that
`0` falls inside `[L, R]`).

## Input / output contract

- Input (stdin): the first line has three integers `n`, `L`, `R`
  (`0 <= n <= 2000`, `0 <= L <= R <= 100000`). Then `n` lines follow, line `i` holding two integers
  `p[i]` (`1 <= p[i] <= 100000`) and `v[i]` (`-10^9 <= v[i] <= 10^9`).
- Output (stdout): a single line — the maximum total joy over all subsets whose total price is in
  `[L, R]` inclusive, or the word `IMPOSSIBLE` if there is no such subset.
- Time limit: 1 second. Memory: 256 MB.

Example: for

```
5 10 12
4 3
5 4
6 5
7 -1
3 2
```

the answer is `9`. Buying items 2 and 3 (1-indexed) costs `5 + 6 = 11`, which is inside `[10, 12]`,
for joy `4 + 5 = 9`. (Buying items 1, 2, 5 also reaches price `12` with joy `3 + 4 + 2 = 9`.) No
in-band subset beats `9`.

## Evaluation settings

Judged on hidden tests covering: `n = 0`; bands with `L = 0` (empty subset legal) and `L > 0`
(empty subset illegal); single-item inputs where the item's price is below `L`, inside `[L, R]`,
exactly equal to `L` or to `R`, or above `R`; all-negative joys (the answer is the *least bad*
in-band subset, which can be negative); inputs with no in-band subset (`IMPOSSIBLE`); items with
`p[i] > R` (must be ignored); and large `n` and `R` near the limits.

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    long long L, R;
    if (!(cin >> n >> L >> R)) return 0;

    vector<long long> p(n), v(n);
    for (int i = 0; i < n; i++) cin >> p[i] >> v[i];

    // TODO: among subsets whose total price s satisfies L <= s <= R (inclusive),
    //       output the maximum total joy, or "IMPOSSIBLE" if none exists.

    return 0;
}
```
