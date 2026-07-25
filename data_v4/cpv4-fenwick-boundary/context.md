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

## Evaluation settings

Judged on hidden tests covering: queries whose bounds coincide with present ids, bounds that fall
strictly between ids or outside the id range, `lo > hi` (empty), queries before any insertion, ids
re-entering after leaving, and large `q = 2*10^5` with ids spread across `[1, 10^9]`.

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int q;
    if (!(cin >> q)) return 0;

    // TODO: read the q events, maintain the set of currently-present badge ids, and
    //       answer each "? lo hi" with the count of present ids in [lo, hi].

    return 0;
}
```
