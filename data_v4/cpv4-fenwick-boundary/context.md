# Badge scanner: present-count over an inclusive value range

## Research question

A door scanner tracks which **badges** are currently inside a building. Each badge has a distinct
integer id. We process a chronological log of `q` events of three kinds:

- `+ b` — badge `b` enters (it was *not* inside just before this event),
- `- b` — badge `b` leaves (it *was* inside just before this event),
- `? lo hi` — a query: **how many badges are currently inside whose id lies in the inclusive range
  `[lo, hi]`** (that is, `lo <= b <= hi`).

For every `?` event, output the count. The building starts empty. Ids and query bounds can be as
large as `10^9`, but only at most `q` distinct ids ever appear in `+`/`-` events, so the values are
sparse over their range.

This is the canonical "dynamic prefix-count over values" task that a Fenwick (binary indexed) tree
solves in `O(log)` per operation. The whole difficulty is at the *boundary*: the query range is
**inclusive on both ends**, the values are compressed, and a single off-by-one in how the lower bound
is translated into a prefix subtraction silently drops or double-counts the badge that sits exactly
on `lo`.

## Input / output contract

- Input (stdin): the first token is `q` (`1 <= q <= 2*10^5`). Then `q` events follow, one per line
  in chronological order:
  - `+ b` or `- b` with `1 <= b <= 10^9`;
  - `? lo hi` with `1 <= lo, hi <= 10^9` (note `lo > hi` is allowed and denotes an empty range).
  - Guarantees: a `+ b` event never repeats a badge already inside; a `- b` event always refers to a
    badge currently inside. So at any instant each id is present at most once (set semantics).
- Output (stdout): for each `?` event, in order, one line with the number of badges currently inside
  whose id is in `[lo, hi]`.
- Time limit: 1 second. Memory: 256 MB.

Example:

```
8
+ 10
+ 30
+ 50
? 10 50
? 11 49
- 30
? 10 50
? 60 5
```

Answers: `3`, `1`, `2`, `0`. The first query counts `10, 30, 50` (both endpoints inclusive); the
second narrows to the open span and keeps only `30`; after `30` leaves, `[10,50]` holds `10` and `50`;
the last has `lo > hi`, an empty range.

## Background

The state is a dynamic set of present ids; each query asks for a count of present ids in a value
window. Two families of approach are on the table before committing:

- **Rescan per query.** Keep a hash set of present ids and, for each `?`, iterate the set counting
  those in `[lo, hi]`. This is `O(size)` per query — up to `O(q)` each — so worst case `O(q^2)`,
  about `4*10^10` operations at `q = 2*10^5`. Correct but far too slow under a 1 s limit.
- **Fenwick tree over compressed values.** Coordinate-compress the ids that appear in `+`/`-`
  events into `1..N`, keep a Fenwick tree of present-counts, and answer a window `[lo, hi]` as
  `fsum(#values <= hi) - fsum(#values < lo)`. Each update and query is `O(log N)`, total
  `O(q log q)`. The open question is the *exact* translation of the inclusive bounds `lo` and `hi`
  — which side is `<=` and which is `<` — when `lo`/`hi` need not be actual ids.

## Evaluation settings

Judged on hidden tests covering: queries whose bounds coincide with present ids (inclusive endpoints
must be counted), bounds that fall strictly between ids or outside the id range, `lo > hi` (empty),
queries before any insertion, ids re-entering after leaving, and large `q = 2*10^5` with ids spread
across `[1, 10^9]` (so compression is mandatory and counts fit in 32 bits but the code uses 64-bit
sums for safety).

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int N;                 // size of the Fenwick tree (number of distinct badge values)
vector<int> fen;       // 1-indexed Fenwick storing present-counts over compressed badges

void fadd(int i, int delta) {       // i is 1-indexed
    for (; i <= N; i += i & (-i)) fen[i] += delta;
}
long long fsum(int i) {             // sum over compressed positions [1..i], i may be 0
    long long s = 0;
    for (; i > 0; i -= i & (-i)) s += fen[i];
    return s;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int q;
    if (!(cin >> q)) return 0;

    // TODO: read events, coordinate-compress the badge ids that appear in +/- events,
    //       maintain the Fenwick tree, and answer each "? lo hi" as a present-count over [lo, hi].

    return 0;
}
```
