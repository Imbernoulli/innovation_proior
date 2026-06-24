**Problem.** A garage logs `n` cars; car `i` occupies the **half-open** interval `[s_i, e_i)` —
present at instant `s_i`, gone by `e_i`. A car leaving at time `t` and a car arriving at `t` reuse
the same spot and do **not** overlap. Report the maximum number of cars present at any single
instant (the minimum spots needed). Records with `s_i >= e_i` occupy nothing and are ignored; if no
effective cars remain the answer is `0`. Read `n` then the pairs from stdin; print one integer.

**Key idea — event sweep with `+1 / -1`.** Coverage `f(t) = #{i : s_i <= t < e_i}` is a step
function that jumps `+1` when `t` reaches an arrival and `-1` when `t` reaches a departure, so its
maximum is attained among the `2n` event coordinates. Emit a `+1` event at each `s_i`, a `-1` event
at each `e_i`, sort by coordinate, sweep left to right keeping a running count `cur`, and report its
maximum. `O(n log n)`.

**The one decision that makes it correct — the tie-break.** At a coordinate where a departure
`e_j = t` and an arrival `s_k = t` coincide, half-open semantics say the departing car is already
gone (`t < e_j` is false) while the arriving car is present (`s_k <= t` is true). So **at a shared
coordinate, apply all ends (`-1`) before all starts (`+1`).** Encode each event as `(coordinate,
type)` with `type 0 = end`, `type 1 = start`; then a plain lexicographic `pair` sort puts ends
before starts at any tie — exactly what `[s, e)` needs. Reversing this (starts first) silently
computes the answer for *closed* intervals `[s, e]`, off by one.

**Pitfalls.**
1. *Inverted tie-break (the off-by-one).* If starts sort before ends at equal coordinate, two merely
   touching intervals `[1,4)` and `[4,7)` are counted together, giving `2` where the half-open answer
   is `1`. A two-car touching trace exposes it instantly: encode `end=0, start=1` so ends fire first.
2. *Degenerate records.* A record with `s_i >= e_i` covers no instant. Pushing its events anyway lets
   the running count drift negative and can corrupt the maximum; skip such records before emitting
   events.
3. *Coordinate compression temptation.* Coordinates reach `10^9`, but the event sweep needs no array
   indexed by coordinate, so no compression and no second copy of the boundary question.

**Edge cases.** `n = 0` → `0`; a single car → `1`; fully disjoint intervals → `1`; many intervals
sharing one endpoint count only the live ones (touching does not overlap); nested intervals peak in
the middle; degenerate `s >= e` records contribute `0`.

**Complexity.** `O(n log n)` time for the sort, `O(n)` space for the `2n` events. The running count
never exceeds `n <= 2*10^5`, so it fits any integer type; coordinates fit `long long` with room.

**Code.**

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    if (!(cin >> n)) return 0;             // n = 0 -> no cars -> answer 0

    // Each car occupies the half-open interval [s, e): present at s, gone by e.
    // A car ending at time t and a car starting at time t do NOT overlap.
    // We build events: a start contributes +1 AT coordinate s; an end contributes
    // -1 AT coordinate e. We sweep coordinates in increasing order. At a shared
    // coordinate, ENDS must be applied before STARTS, because [a,t) and [t,b) are
    // disjoint -- the leaving car frees the instant before the arriving car claims it.
    // event = (coordinate, type) with type 0 = end (-1), type 1 = start (+1).
    vector<pair<long long,int>> ev;
    ev.reserve((size_t)2 * n);
    for (int i = 0; i < n; i++) {
        long long s, e;
        cin >> s >> e;
        // Guard: a degenerate interval with s >= e occupies no instant; skip it.
        if (s >= e) continue;
        ev.push_back({s, 1});   // start
        ev.push_back({e, 0});   // end
    }

    // Sort by coordinate ascending; within equal coordinate, type 0 (end) before
    // type 1 (start). Since we encode end as 0 and start as 1, a plain pair sort
    // on (coordinate, type) gives ends first at ties -- exactly what half-open needs.
    sort(ev.begin(), ev.end());

    long long cur = 0, best = 0;
    for (auto &p : ev) {
        if (p.second == 1) cur += 1; else cur -= 1;
        best = max(best, cur);
    }

    cout << best << "\n";
    return 0;
}
```
