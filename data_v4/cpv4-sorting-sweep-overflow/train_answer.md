**Problem.** There are `n` data streams; stream `i` is active on the half-open interval `[s_i, e_i)` and consumes `w_i` bandwidth while active. At an instant `t` the load is the sum of `w_i` over streams with `s_i <= t < e_i`. Read `n` and the triples `s e w` from stdin; print the maximum load over all instants `t`. With `n` up to `2*10^5` and `w_i` up to `10^9`, the answer can reach `2*10^14`.

**Key idea — sort-and-sweep.** The load only changes at stream boundaries, so emit a `+w` event at each start `s_i` and a `-w` event at each end `e_i`, sort the `2n` events by time, scan left to right keeping a running load `cur`, and track `best = max(best, cur)`. `O(n log n)` time, `O(n)` memory. Because all weights are positive, the peak is attained at some start, so the sweep over events captures it.

**The half-open tie rule.** At a shared time, a stream that ends and one that starts do *not* overlap (`[2,5)` and `[5,9)` are disjoint at `t = 5`). So at equal times, **ends must be processed before starts**, otherwise the running load shows a phantom overlap. Encode each event as a pair `(time, delta)` with `delta = +w` for a start and `delta = -w` for an end, then sort the pairs ascending: at equal time the negative delta (end) sorts before the positive delta (start) automatically — the tie rule is just numeric order, with no comparator to get backwards.

**Pitfalls.**
1. *Int overflow (the trap).* The natural `int cur, best` and `int w` overflow: three weight-`10^9` streams over one instant give `3*10^9`, which wraps a signed 32-bit `int` to a negative value and prints a stale, wrong answer; the full bound `2*10^14` wraps modulo `2^32` to `552894464`. Every load-bearing quantity — `w`, the event delta, `cur`, `best` — must be `long long`. No crash, no warning, just wrong on large tests.
2. *Tie order reversed.* Processing starts before ends at equal time invents an overlap that no real instant has (two abutting streams report `2w` instead of `w`). Ends before starts; the signed-delta pair sort enforces it.

**Edge cases.** `n = 0` → no events → `0`. Empty interval `s_i == e_i` → skipped (active at no instant). Single stream → its own weight. Disjoint streams sharing a boundary → never summed together. Nested streams → peak in the innermost overlap.

**Complexity.** `O(n log n)` time (the sort dominates), `O(n)` extra space.

**Code.**

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    if (!(cin >> n)) return 0;

    // Each stream i is active on the half-open interval [s_i, e_i) with weight w_i.
    // Build sweep events: at time s a +w event, at time e a -w event.
    // Sort by time; at equal time, process the -w (end) events before the +w (start)
    // events so that a stream ending exactly when another starts does NOT overlap it.
    vector<pair<long long, long long>> ev;
    ev.reserve(2 * n);
    for (int i = 0; i < n; i++) {
        long long s, e, w;
        cin >> s >> e >> w;
        if (s >= e) continue;              // empty interval contributes nothing
        ev.push_back({s, +w});             // start: add weight
        ev.push_back({e, -w});             // end:   remove weight
    }

    // Sort by time. For equal time, ends (negative delta) come before starts (positive
    // delta). Sorting the pair (time, delta) ascending does exactly that, because a
    // negative delta sorts before a positive one at the same time.
    sort(ev.begin(), ev.end());

    long long cur = 0, best = 0;           // 64-bit: load can reach ~2e5 * 1e9 = 2e14
    for (auto &p : ev) {
        cur += p.second;
        best = max(best, cur);
    }

    cout << best << "\n";
    return 0;
}
```
