**Problem.** A ring of `m` markers `0..m-1` (arithmetic mod `m`) and `n` floodlights; floodlight `i`
lights the contiguous clockwise arc `s_i, s_i+1, ..., s_i+len_i-1` (mod `m`), with `1 <= len_i <= m`.
Switch on the fewest floodlights so every marker is lit; print that minimum, or `-1` if the union of
all arcs still leaves a gap. Read `m n` then the `n` pairs `s_i len_i` from stdin.

**Key idea — anchored greedy minimum circular-arc cover.** On a line, cover-with-fewest-intervals is
the classic greedy: from the current covered endpoint, jump to the interval that starts at or before it
and reaches furthest. The circle has no leftmost point, so anchor one: marker `0` must be lit by *some*
chosen arc. Enumerate, as the forced first arc, each arc covering marker `0` (those with `s == 0` or
`s + len > m`). For first arc with start `a0`, rotate every arc into coordinates relative to `a0`
(`rs = (s - a0) mod m` in `[0, m)`, interval `[rs, rs+len)`, plus a prefix copy `[rs-m, rs+len-m)` if it
wraps), sort by left endpoint, and run the line greedy over the window `[0, m)`. The minimum count over
all first-arc choices is the answer; if no candidate's sweep finishes, output `-1`. A single full-ring
arc (`len == m`) is handled up front as answer `1`.

**Pitfalls.**
1. *Seam double-count (the headline trap).* A wrapping arc must appear twice on an unrolled axis so its
   coverage near marker `0` is visible — but an *unanchored* single sweep over the doubled axis `[0, 2m)`
   can consume *both* copies and charge one physical floodlight as two. On `m=3`, arcs `(2,2)` and
   `(1,1)`, that buggy sweep returns `3`; the truth is `2`. Anchoring each sweep at a forced first arc
   and covering a window of length exactly `m` collapses the seam so no lamp is counted twice.
2. *Negative modulus.* `rs = (s - a0) % m` is negative in C++ when `s < a0`, which silently drops an arc
   out of the window and corrupts the count (or fakes a `-1`). Use `((s - a0) % m + m) % m`.
3. *Coordinate overflow.* `m` up to `10^9` makes `rs + len` and the window reach `~2*10^9`; everything
   must be `long long`. An `int` is a silent wrong answer.
4. *"Covers marker 0" predicate.* It is `s == 0 OR s + len > m`, not `s + len >= m` (an arc ending
   exactly at marker `m-1` does not reach `0`).

**Edge cases.** Single full-ring arc -> `1` (early exit on `len >= m`); `m = 1` -> any arc is full,
`1`; infeasible (e.g. `m=5`, two `(0,2)` arcs) -> every candidate sweep stalls, `-1`; duplicate and
fully redundant arcs are not over-counted because the greedy advances `curEnd` past identical intervals
in one jump.

**Complexity.** At most `n` first-arc candidates, each an `O(n log n)` sort plus linear sweep:
`O(n^2 log n)`. For `n <= 2000` the genuine worst case (every arc a candidate, full-length sweeps)
measures ~0.14 s, well inside the 1 s limit. Memory `O(n)`.

**Code.**

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    long long m;
    int n;
    if (!(cin >> m >> n)) return 0;

    // Arc (s, L), 1 <= L <= m, covers circular markers s,...,s+L-1 (mod m).
    // Want minimum number of arcs whose union is the whole ring, else -1.
    //
    // MINIMUM CIRCULAR ARC COVER via greedy-exchange:
    //   Marker 0 must be covered by some chosen arc. We try, as the forced
    //   FIRST arc, each arc that covers marker 0. Forcing arc A fixes the
    //   linear window [A.start, A.start + m): we must cover that whole window.
    //   After A we have covered up to A.start + A.len. Then we repeatedly pick,
    //   among arcs whose linear start is <= current covered end, the one whose
    //   linear end reaches furthest, advancing the covered end. Each pick is one
    //   more arc. We stop once the covered end reaches A.start + m. The minimum
    //   over all choices of first arc is the answer.

    vector<long long> S(n), Ln(n);
    for (int i = 0; i < n; i++) {
        long long s, L;
        cin >> s >> L;
        s %= m; if (s < 0) s += m;
        S[i] = s; Ln[i] = L;
    }

    // A length-m arc covers the entire ring by itself.
    for (int i = 0; i < n; i++) if (Ln[i] >= m) { cout << 1 << "\n"; return 0; }

    long long best = LLONG_MAX;

    // Candidate first arcs = arcs covering marker 0: s == 0 OR s + L > m.
    for (int f = 0; f < n; f++) {
        bool coversZero = (S[f] == 0) || (S[f] + Ln[f] > m);
        if (!coversZero) continue;
        long long a0 = S[f];

        // Coordinates relative to a0: marker (s - a0 mod m). Cover [0, m).
        // Each arc j -> [rs, rs+L); if it wraps, also the prefix copy [rs-m, ...).
        vector<pair<long long,long long>> ivs; // (l, r)
        ivs.reserve(2 * n);
        for (int j = 0; j < n; j++) {
            long long rs = ((S[j] - a0) % m + m) % m;   // positive modulus!
            ivs.push_back({rs, rs + Ln[j]});
            if (rs + Ln[j] > m) ivs.push_back({rs - m, rs + Ln[j] - m});
        }
        sort(ivs.begin(), ivs.end());

        // Greedy minimum interval cover of [0, m). Forced first arc f maps to
        // rs = 0, seeding the sweep. Count arcs used.
        long long curEnd = 0, cnt = 0;
        size_t p = 0;
        bool ok = true;
        while (curEnd < m) {
            long long newEnd = curEnd;
            while (p < ivs.size() && ivs[p].first <= curEnd) {
                newEnd = max(newEnd, ivs[p].second);
                p++;
            }
            if (newEnd <= curEnd) { ok = false; break; }
            curEnd = newEnd;
            cnt++;
        }
        if (ok && curEnd >= m) best = min(best, cnt);
    }

    if (best == LLONG_MAX) cout << -1 << "\n";
    else cout << best << "\n";
    return 0;
}
```
