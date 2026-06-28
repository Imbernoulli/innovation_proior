# Fully persistent versioned sequence with reverse and range-sum

## Research question

You maintain a sequence of integers that evolves through a stream of `q` operations,
but with a twist that ordinary balanced trees do not give you: **every operation may
act on any past version of the sequence, not just the most recent one.** Each
modifying operation does not change the version it reads; it produces a brand-new
version that is appended to the version list, leaving the old version fully intact and
still queryable. This is *full persistence* — old states never get clobbered, and the
history is a tree of versions, not a single timeline.

There are three operation types:

- **insert** a value at a given position of some version `v`, creating a new version;
- **reverse** a contiguous subrange `[l, r]` of some version `v`, creating a new version;
- **query** the sum of a contiguous subrange `[l, r]` of some version `v` (read-only,
  no new version is created).

Version `0` is the empty sequence. The `k`-th modifying operation (counting `insert`
and `reverse`, but not `query`) creates version `k`. Output the answer to every query,
in input order.

The point of the problem is to support `insert`, `reverse`, and range-`sum` on a
sequence in logarithmic time **while keeping every historical version addressable and
correct** — including reversing or inserting into a version that was itself produced by
an earlier reverse.

## Input / output contract

- Input (stdin): the first token is `q` (`0 <= q <= 10^5`), the number of operations.
  Then `q` operations follow, one per line, each beginning with a type tag:
  - `1 v p x` — *insert*: in version `v`, insert value `x` so that it becomes the
    element at 0-indexed position `p`. Here `0 <= p <= len(v)` (so `p = len(v)` appends
    at the end). `-10^9 <= x <= 10^9`. Creates the next version.
  - `2 v l r` — *reverse*: in version `v`, reverse the subarray at 0-indexed positions
    `[l, r]` inclusive, with `0 <= l <= r < len(v)`. Creates the next version.
  - `3 v l r` — *query*: output the sum of the elements at 0-indexed positions `[l, r]`
    inclusive of version `v`, with `0 <= l <= r < len(v)`. Does **not** create a version.
  - In every operation, `v` refers to an already-existing version id (`0 <= v <`
    current number of versions). All position/range references are guaranteed in range
    for the version they target.
- Output (stdout): for each `query` operation, one line with the requested sum.
- Time limit: 3 seconds. Memory: 1024 MB.

### Worked sample

Input:

```
9
1 0 0 5
1 1 1 1
1 2 2 2
1 3 3 4
1 4 4 3
3 5 1 3
2 5 1 3
3 6 1 1
3 5 1 1
```

The five inserts build, version by version, `v1=[5]`, `v2=[5,1]`, `v3=[5,1,2]`,
`v4=[5,1,2,4]`, `v5=[5,1,2,4,3]`. Then:

- `3 5 1 3` asks for the sum of positions `1..3` of `v5 = [5,1,2,4,3]`, i.e. `1+2+4 = 7`.
- `2 5 1 3` reverses positions `1..3` of `v5`, creating `v6 = [5,4,2,1,3]`.
- `3 6 1 1` asks position `1` of `v6`, which is `4`.
- `3 5 1 1` asks position `1` of `v5`. Crucially this is still `1`: producing `v6` did
  **not** touch `v5`. Persistence is what makes this line correct.

Output:

```
7
4
1
```

## Background

Two structural demands collide here.

- **Sequence operations by position.** Insert-at-position and reverse-a-subrange need a
  balanced binary search tree keyed not by value but by *position* — an "implicit-key"
  tree, where a node's in-order rank is determined by subtree sizes rather than by a
  stored key. Split (cut off the first `k` elements) and merge (concatenate two
  sequences) are the primitives; insert is split + a singleton + two merges, and
  reverse is two splits, a lazy "reverse this whole subtree" flag, and two merges. The
  lazy flag is what makes reverse `O(log n)` instead of `O(r-l)`: flipping an entire
  subtree's order is recorded as one bit and only "pushed" down when that subtree is
  next traversed.

- **Full persistence.** Every past version must remain valid. A normal balanced tree
  mutates nodes in place, which destroys the old version. To keep history, modifications
  must be **non-destructive**: instead of editing a node, allocate a fresh copy of it.

The tension is that lazy propagation *is* a mutation — pushing a reverse flag rewrites a
node and reorders its children — and that seems to fight persistence head-on. Resolving
that tension is the core of the problem.

## Evaluation settings

Judged on hidden tests covering: long single-version chains grown by `insert` (so the
sequence reaches `~10^5` elements and tree depth is maximal); heavy use of `reverse`,
including reversing a version that was already reversed (nested/composed reversals);
operations that branch off **old** versions rather than the latest, exercising
persistence directly; queries whose ranges are whole subtrees, partial subtrees, and
single elements; degenerate ranges (`l == r`), length-one reverses, and `q = 0`. Values
near `10^9` over ranges of `~10^5` elements push sums past 32 bits, so accumulators must
be 64-bit.

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

    // Version 0 is the empty sequence. Each insert/reverse appends a new version
    // that shares as much structure as possible with the version it was built from;
    // each query reads a version without creating one.

    for (int i = 0; i < q; i++) {
        int type; cin >> type;
        if (type == 1) {
            long long v, p, x; cin >> v >> p >> x;
            // TODO: insert x at position p of version v -> new version
        } else if (type == 2) {
            long long v, l, r; cin >> v >> l >> r;
            // TODO: reverse positions [l, r] of version v -> new version
        } else {
            long long v, l, r; cin >> v >> l >> r;
            // TODO: output sum of positions [l, r] of version v
        }
    }
    return 0;
}
```
