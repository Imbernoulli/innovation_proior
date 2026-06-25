# Minimum inspection checkpoints along a pipeline

## Research question

A maintenance crew monitors a long straight pipeline laid out along a number line. A survey has
flagged `n` suspect **segments**; segment `i` occupies the closed integer interval `[l_i, r_i]`
(a leak could be anywhere on that stretch, endpoints included). The crew can install an inspection
**checkpoint** at any integer coordinate `x`. A checkpoint at `x` inspects segment `i` exactly when
`l_i <= x <= r_i` — that is, the checkpoint must sit on the segment, and sitting on either endpoint
counts.

Installing checkpoints is expensive, so the crew wants the **minimum number of checkpoints** such
that every flagged segment is inspected by at least one of them. Output that minimum count.

This is the "stab all intervals with the fewest points" problem. The twist that decides correctness
is that the intervals are **closed**: two segments that merely touch at a single coordinate (say
`[1,5]` and `[5,9]`, sharing only the point `5`) can be inspected by one shared checkpoint. Whether
your code treats a touching endpoint as "already covered" or "still uncovered" is a one-symbol
decision — `<` versus `<=` — and it is the whole problem.

## Input / output contract

- Input (stdin): the first token is `n` (`0 <= n <= 2*10^5`). Then follow `n` lines, each with two
  integers `l_i` and `r_i` (`-10^9 <= l_i <= r_i <= 10^9`), the inclusive endpoints of segment `i`.
  Tokens may be separated by arbitrary whitespace.
- Output (stdout): a single line with the minimum number of checkpoints. If `n = 0`, the answer is
  `0`.
- Time limit: 1 second. Memory: 256 MB.

Example: for the four segments `[1,5]`, `[5,8]`, `[9,12]`, `[10,14]` the answer is `2` — one
checkpoint at coordinate `5` inspects both `[1,5]` and `[5,8]` (they touch at `5`), and one at
coordinate `12` inspects both `[9,12]` and `[10,14]`.

## Background

The constraint is global: a single checkpoint can serve many segments at once, and the right
placement of one checkpoint changes which segments the next one must cover. Two families of approach
are on the table before committing:

- **Greedy by earliest right endpoint (exchange argument).** Sort the segments by their right
  endpoint. Take the not-yet-inspected segment with the smallest right endpoint, place a checkpoint
  at that right endpoint, and mark every segment that this checkpoint inspects as done; repeat. The
  open questions are *why* placing the checkpoint at the right endpoint (rather than anywhere else on
  the segment) is safe, and *exactly* which already-placed-checkpoint test correctly recognises an
  inspected segment when intervals are closed and may touch.
- **Coordinate-compressed interval DP / matching.** Compress endpoints and reason about a maximum
  set of pairwise point-disjoint segments. This is heavier to implement and the boundary subtlety
  (do touching intervals count as disjoint?) reappears in the disjointness test, so it offers no
  shortcut around the core decision.

## Evaluation settings

Judged on hidden tests covering: `n = 0`; a single segment; many segments sharing one common point;
chains of segments that touch end-to-end (`[1,5],[5,9],[9,13],...`) where the closed-endpoint sharing
is decisive; nested segments; fully disjoint segments (answer equals `n`); duplicate and degenerate
`l_i = r_i` segments; negative coordinates; and large `n = 2*10^5` with coordinates spanning the full
`[-10^9, 10^9]` range.

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    if (!(cin >> n)) return 0;
    vector<pair<long long,long long>> seg(n); // (right endpoint, left endpoint)
    for (int i = 0; i < n; i++) {
        long long l, r;
        cin >> l >> r;
        seg[i] = {r, l};
    }

    // TODO: sort by right endpoint and greedily place checkpoints, getting the
    // closed-interval "already inspected" boundary test exactly right.
    long long answer = 0;

    cout << answer << "\n";
    return 0;
}
```
