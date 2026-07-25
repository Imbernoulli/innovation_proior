# Counting visible buildings in a window with point updates

## Research question

You are given `n` building heights `h[0..n-1]`. A *survey* fixes a contiguous window `[l, r]`
(`0 <= l <= r <= n-1`) and looks at the buildings from the left edge inward: a building at position
`k` (`l <= k <= r`) is **visible** if it is *strictly taller than every building to its left inside
the window*, i.e. `h[k] > h[j]` for all `l <= j < k`. The leftmost building of any window is always
visible. The number of visible buildings in `[l, r]` is exactly the number of **strict prefix maxima**
of the subarray `h[l..r]`.

Buildings are also renovated over time, so heights change. You must support, in any order:

- a **point update** that sets one building's height, and
- a **survey query** that reports how many buildings are visible in a window.

## Input / output contract

- Input (stdin):
  - line 1: two integers `n` and `q` (`1 <= n <= 2*10^5`, `1 <= q <= 2*10^5`).
  - line 2: `n` integers `h[0..n-1]` (`1 <= h[i] <= 10^9`).
  - then `q` lines, each an operation:
    - `1 p x` — set `h[p] = x` (`0 <= p <= n-1`, `1 <= x <= 10^9`).
    - `2 l r` — survey the window `[l, r]` (`0 <= l <= r <= n-1`): output the number of visible
      buildings.
- Output (stdout): for each query of type `2`, one line with the visible count.
- Time limit: 2 seconds. Memory: 256 MB.

Example: with `h = [3, 1, 4, 1, 5, 9]`, the window `[0, 5]` has visible buildings `3, 4, 5, 9`
(answer `4`), and the window `[1, 3] = [1, 4, 1]` has visible buildings `1, 4` (answer `2`).

## Evaluation settings

Judged on hidden tests covering: single-element arrays and `l == r` windows; arrays with many equal
heights; strictly increasing windows (every building visible) and strictly decreasing windows (only
the first visible); queries that split across several canonical segments; updates interleaved with
queries; and large `n = q = 2*10^5` with heights up to `10^9`.

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int n, q;
vector<long long> h;       // current heights
vector<long long> mx;      // segment-tree node maxima

void build(int node, int lo, int hi) {
    if (lo == hi) { mx[node] = h[lo]; return; }
    int mid = (lo + hi) / 2;
    build(2 * node, lo, mid);
    build(2 * node + 1, mid + 1, hi);
    mx[node] = max(mx[2 * node], mx[2 * node + 1]);
}

void pointSet(int node, int lo, int hi, int pos, long long val) {
    if (lo == hi) { mx[node] = val; return; }
    int mid = (lo + hi) / 2;
    if (pos <= mid) pointSet(2 * node, lo, mid, pos, val);
    else            pointSet(2 * node + 1, mid + 1, hi, pos, val);
    mx[node] = max(mx[2 * node], mx[2 * node + 1]);
}

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);
    if (!(cin >> n >> q)) return 0;
    h.assign(max(1, n), 0);
    for (int i = 0; i < n; i++) cin >> h[i];
    mx.assign(4 * max(1, n), LLONG_MIN);
    if (n > 0) build(1, 0, n - 1);

    string out;
    for (int i = 0; i < q; i++) {
        // TODO: handle update `1 p x`, and survey `2 l r` = number of visible
        // buildings (strict prefix maxima) in h[l..r].
    }
    cout << out;
    return 0;
}
```
