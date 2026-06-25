# Re-carpeting a hallway with fixed-width runner rugs

## Research question

A hallway floor is a row of `n` wooden panels, numbered `1..n` from one end. Panel `i` has a
*roughness* value `a[i] >= 1` (how uneven that plank is). You want to cover the **entire** hallway
with **runner rugs**. Each rug is laid over a contiguous block of panels and is described by an
**inclusive** interval `[l, r]` of panel indices (it covers panels `l, l+1, ..., r`). The rugs must
not overlap and together must cover every panel, so the rugs form a partition of `1..n` into
consecutive blocks.

Two rules govern a single rug over `[l, r]`:

- A rug can span **at most `L` panels**, i.e. its inclusive length `r - l + 1` must be `<= L`.
- The rug must be padded to absorb the roughest panel underneath it, so the rug over `[l, r]` costs
  `K + max(a[l], a[l+1], ..., a[r])`, where `K` is a fixed per-rug manufacturing fee and the `max`
  is taken over the **inclusive** range `[l, r]`.

Choose a set of rugs covering the whole hallway and **minimize the total cost**. Output that minimum.

Because every single panel forms a legal length-1 rug, a covering always exists, so the answer is
always finite.

## Input / output contract

- Input (stdin): the first line has three integers `n`, `K`, `L`
  (`1 <= n <= 5000`, `0 <= K <= 10^9`, `1 <= L <= n`).
  The second line has `n` integers `a[1..n]` (`1 <= a[i] <= 10^9`), whitespace-separated.
- Output (stdout): a single line with the minimum total cost.
- Time limit: 1 second. Memory: 256 MB.

Example: for `n = 5`, `K = 2`, `L = 2`, `a = [1, 5, 5, 1, 5]` the answer is `17`
(rugs `[1,1]`, `[2,3]`, `[4,5]` cost `3 + 7 + 7`).

## Background

The covering is a partition of the index line `1..n` into consecutive inclusive blocks, with a
per-block cost and a per-block length cap. Two families of approach are on the table before
committing:

- **Greedy laying.** Walk left to right and, at each step, lay the longest legal rug (or the
  locally cheapest rug). It is `O(n)` and trivial to write; the open question is whether a local
  choice of where to end the current rug can be globally optimal when the cost only charges for the
  roughest panel under each rug.
- **Partition dynamic programming.** Let `dp[i]` be the minimum cost to cover the first `i` panels.
  The last rug covers some inclusive suffix `[j+1, i]` of legal length, and `dp[i]` is the best over
  all legal split points `j`. This is `O(n * L)`; the open question is the exact recurrence — in
  particular which index the running `max` and the length cap are measured against, since `[l, r]`
  is inclusive on both ends.

## Evaluation settings

Judged on hidden tests covering: `L = 1` (every panel is its own rug), `L = n` (one rug may cover
everything), `L = n - 1` (the cap just barely forbids the full hallway), `K = 0` (only the roughness
matters), large `K` (the per-rug fee dominates and pushes toward fewer, longer rugs), arrays with a
single very rough panel that wants to be isolated, `n = 1`, and large `n = 5000` with values near
`10^9` (so the total can exceed a 32-bit integer).

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    long long K;
    int L;
    if (!(cin >> n >> K >> L)) return 0;
    vector<long long> a(n + 1);               // 1-indexed roughness a[1..n]
    for (int i = 1; i <= n; i++) cin >> a[i];

    // TODO: compute the minimum total cost to cover panels 1..n with rugs,
    // each an inclusive interval [l, r] of length <= L costing K + max(a[l..r]).
    long long answer = 0;

    cout << answer << "\n";
    return 0;
}
```
