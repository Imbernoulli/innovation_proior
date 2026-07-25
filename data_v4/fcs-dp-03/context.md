# Consolidating an append-only shard log

## Research question

A streaming service stores an event log as `n` immutable **shards** laid out left to right in
write order. Shard `i` holds `w[i]` bytes. The compactor must collapse the whole log into a single
shard, and it may only **fuse two shards that are currently adjacent** in the layout. Fusing two
adjacent shards rewrites both into one new shard whose size is the sum of the two sizes, and the
**cost of that fuse equals the size of the shard it produces** (every byte of both inputs is copied
once). The new shard sits in the position the two inputs occupied, so the relative order of all
remaining shards is preserved and only adjacent fuses are ever available.

You choose the order of fuses. After `n - 1` fuses exactly one shard remains. Output the **minimum
possible total fusing cost**.

## Input / output contract

- Input (stdin): the first token is `n` (`0 <= n <= 2000`); then `n` integers `w[i]`
  (`1 <= w[i] <= 10^9`), whitespace-separated.
- Output (stdout): a single line with the minimum total fusing cost.
- For `n = 0` and `n = 1` no fuse is ever performed, so the cost is `0`.
- Time limit: 2 seconds. Memory: 256 MB.

Worked example: for `w = [4, 1, 2, 3]` the answer is `19`.
One optimal schedule: fuse the `1` and `2` (cost `3`, sizes become `[4, 3, 3]`), fuse the two `3`s
(cost `6`, sizes become `[4, 6]`), fuse the last pair (cost `10`). Total `3 + 6 + 10 = 19`. No
schedule does better.

## Evaluation settings

Judged on hidden tests covering: tiny inputs (`n = 0, 1, 2`), uniform weights, strictly ascending
and descending weights, "spiky" alternating tiny/huge weights, near-maximal weights at `n = 2000`,
and many random mid-sized cases cross-checked against a slow but transparent reference. A solution
that is only fast enough for small `n` will fail.

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
    vector<long long> w(n);
    for (auto &x : w) cin >> x;

    // TODO: compute the minimum total cost to fuse all shards into one,
    // where each fuse may only combine two currently-adjacent shards and
    // costs the size of the shard it produces.
    long long answer = 0;

    cout << answer << "\n";
    return 0;
}
```
