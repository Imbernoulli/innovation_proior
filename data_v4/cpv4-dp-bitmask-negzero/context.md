# Parcel-to-slot assignment with optional delivery (max net profit)

## Research question

A depot has `n` parcels and `m` delivery slots. Putting parcel `i` into slot `j` earns a **net
profit** `p[i][j]`, which may be **positive, zero, or negative** (a negative entry means the fuel,
handling, and penalty costs of that pairing outweigh its revenue). Each slot can hold **at most one**
parcel and each parcel goes into **at most one** slot. Crucially, delivery is *optional*: you may
leave any parcel undelivered and any slot empty.

Choose an assignment (a partial matching between parcels and slots) that **maximizes the total net
profit**. Because the empty assignment — deliver nothing — is always allowed, the answer is never
below `0`.

This is a weighted bipartite matching restricted to a small number of slots, which is exactly the
shape that a **subset (bitmask) DP over the occupied slots** solves cleanly. The interesting corners
are the ones the optionality creates: an all-negative profit matrix (deliver nothing, answer `0`),
zeros (delivering for `0` profit is allowed but never helps), and the empty instance `n = 0`.

## Input / output contract

- Input (stdin): the first line has two integers `n` and `m`
  (`0 <= n <= 18`, `1 <= m <= 18`). Then follow `n` lines, each with `m` integers; the `j`-th
  integer on line `i` is `p[i][j]` (`-10^9 <= p[i][j] <= 10^9`), whitespace-separated.
- Output (stdout): a single line with the maximum achievable total net profit.
- Time limit: 1 second. Memory: 256 MB.

Example: for

```
3 3
5 -2 1
-3 4 0
2 1 6
```

the answer is `15` (parcel 0 -> slot 0 = 5, parcel 1 -> slot 1 = 4, parcel 2 -> slot 2 = 6).

## Background

With `m` slots, at most `m` parcels are ever delivered, and the only thing that ties parcels together
is the "each slot used once" constraint. Two families of approach are on the table before committing
to one:

- **Greedy.** Repeatedly take the largest remaining positive `p[i][j]` whose parcel and slot are both
  still free, and lock them together. It is `O(nm log(nm))` and short; the open question is whether
  grabbing the single biggest entry can ever block a more profitable matching elsewhere.
- **Subset DP over slots.** Process parcels one at a time; track, for every subset `mask` of slots,
  the best profit achievable with exactly those slots occupied. Each parcel is either left
  undelivered or placed in one currently-free slot. This is `O(n * 2^m * m)`; the open questions are
  the exact base case (what does "before any parcel" mean?) and how the optional / negative entries
  interact with that base case so the all-negative instance returns `0` rather than something
  negative.

## Evaluation settings

Judged on hidden tests covering: all-positive matrices, matrices with negatives and zeros, the empty
instance (`n = 0`), a single parcel (`n = 1`), an all-negative matrix (answer should be `0`), the case
`n > m` and `n < m`, and large `n = m = 18` with `|p[i][j]|` near `10^9` (so the running profit can
exceed a 32-bit integer).

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n, m;
    if (!(cin >> n >> m)) return 0;
    vector<vector<long long>> p(n, vector<long long>(m));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < m; j++) cin >> p[i][j];

    // TODO: subset DP over occupied slots; deliver nothing is always allowed,
    // so the answer is at least 0.
    long long answer = 0;

    cout << answer << "\n";
    return 0;
}
```
