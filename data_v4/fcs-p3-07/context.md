# Josephus survivor for a very large circle

## Research question

`n` people stand in a circle, labelled `1, 2, ..., n` clockwise. Starting the count at person `1`,
you count off `k` people and the `k`-th is eliminated; counting then resumes with the next living
person and continues the same way. Exactly one person is left at the end. Output that survivor's
original label.

You must answer `q` independent instances `(n, k)` of this question.

This is the classical Josephus problem. The catch is the scale: `n` can be as large as `10^9`, far
too large to simulate the circle or to fill a survivor table, while `k` is small. The whole exercise
is to commit to an algorithm whose cost depends on `k` and `log n` rather than on `n`.

## Input / output contract

- Input (stdin): the first line is `q` (`1 <= q <= 10^5`), the number of queries. Each of the next
  `q` lines contains two integers `n` and `k` with `1 <= n <= 10^9` and `1 <= k <= 50`.
- Output (stdout): for each query, one line with a single integer — the original label
  (`1`-indexed, in `[1, n]`) of the lone survivor.
- Time limit: 1 second. Memory: 256 MB.

Example:

```
5
1 1
2 2
7 3
10 2
41 3
```

produces

```
1
1
4
5
31
```

The last line is the famous Josephus instance: 41 people, every 3rd eliminated, survivor at
position 31.

## Background

Fix the standard `0`-indexed reformulation. Let `r(m)` be the surviving seat (in `0..m-1`) when the
process runs on `m` people, eliminating every `k`-th. Eliminating the first victim (seat `k-1`,
`0`-indexed) and then renumbering the remaining `m-1` survivors starting from the seat after the
victim gives the well-known recurrence

```
r(1) = 0
r(m) = (r(m-1) + k) mod m       for m = 2, 3, ..., n
```

and the answer for `n` people, converted back to a `1`-indexed label, is `r(n) + 1`.

Two routes are on the table before committing:

- **Fill the recurrence directly.** Loop `m` from `2` to `n`, each step a single add-and-mod. This is
  `O(n)` per query and `O(1)` memory. Correct and trivial, but for `n = 10^9` across up to `10^5`
  queries it is hopelessly slow.
- **Batch the recurrence.** Because `k` is small, for most values of `m` the update `r -> r + k` does
  not reach the modulus `m+1`, so no wrap occurs and many consecutive steps just shift `r` upward by
  `k` each time. Those runs can be jumped in one arithmetic move, collapsing the work to `O(k log n)`
  per query (intuitively, each genuine wrap shrinks the gap `m - r` by a factor related to `k`). The
  open questions are the exact size of each safe jump and the boundary handling so the batched result
  equals the step-by-step recurrence.

## Evaluation settings

Judged on hidden tests covering: `n = 1` (the survivor is always person `1`); `k = 1` (every count of
one eliminates the current person, so eliminations sweep `1, 2, ..., n` and person `n` survives);
small circles where the answer is checkable by hand or by direct simulation; mid-size `n` cross-checked
against the `O(n)` recurrence; and the stress regime of up to `10^5` queries with `n` near `10^9` and
small `k`, where only a per-query cost sublinear in `n` finishes in time. All arithmetic stays within
`64`-bit range (`k*step` and `r` never exceed about `10^9`), but using a `64`-bit type avoids any
intermediate-overflow worry.

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int q;
    if (!(cin >> q)) return 0;
    while (q--) {
        long long n, k;
        cin >> n >> k;

        // TODO: compute the 1-indexed Josephus survivor for n people, every k-th
        // eliminated, in time sublinear in n (n can be up to 1e9, k is small).
        long long survivor = 1;

        cout << survivor << "\n";
    }
    return 0;
}
```
