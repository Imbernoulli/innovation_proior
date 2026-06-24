**Problem.** A receiver logs `n` pulses at integer times `t[0..n-1]` (unsorted, possibly repeated). A snapshot triggered at integer time `s` records every pulse with `s <= t[i] < s + L` — a **half-open** window `[s, s + L)` of fixed width `L`; a pulse at exactly `s + L` is *not* recorded. Output the minimum number of snapshots so every pulse is recorded. Read `n`, `L`, and the times from stdin; print the count.

**Key idea — greedy exchange (leftmost-uncovered sweep).** Sort the pulses. Repeatedly take the earliest still-uncovered pulse `p`, open a snapshot whose window starts exactly at `t[p]` (window `[t[p], t[p] + L)`), record every pulse the window reaches, and repeat. Count the windows.

**Why it is optimal.** Some snapshot must record the earliest pulse `p0`; its start `s` satisfies `s <= t[p0]`. Sliding that window right until `s = t[p0]` loses nothing on the left (no pulse is earlier than `p0`) and can only push the right edge `s + L` further right, reaching at least as many pulses. So an optimal solution starts its first window at `t[p0]`; remove the pulses it covers and induct on the suffix. Hence opening each window at the leftmost orphan is globally optimal.

**The boundary — the whole point.** Because the window is half-open, a pulse stays inside iff `t[i] < s + L`, a **strict** comparison. The reach test must be `t[i] < cover_end`, where `cover_end = t[p] + L`. Writing `<=` instead pulls a pulse sitting on the open right edge into the current window when it should have started a new snapshot.

**Pitfalls.**
1. *Inclusive/exclusive off-by-one.* Use `t[i] < cover_end`, not `<=`. Trace `t = [0, 3]`, `L = 3`: the correct answer is `2` (the pulse at `3 = 0 + L` is on the open edge, so it needs its own window). The buggy `<=` version returns `1`. The general giveaway: pulses spaced *exactly* `L` apart each need their own snapshot.
2. *Overflow.* With `t[i], L` up to `10^9`, the right edge `cover_end = t[i] + L` reaches `~2*10^9`, past 32-bit `int`. Hold the times and `cover_end` in `long long`.

**Edge cases (all handled by the sweep):** `n = 0` -> `0`; `n = 1` -> `1`; all-identical times -> `1` (one window catches coincident pulses); pulses spaced exactly `L` apart -> each needs its own snapshot; `L = 1` with distinct times -> one snapshot per distinct time; unsorted / duplicate input handled by the initial `sort`.

**Complexity.** `O(n log n)` time (the sort dominates; the sweep is one linear pass), `O(1)` extra space beyond the input.

**Code.**

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    long long L;
    if (!(cin >> n >> L)) return 0;
    vector<long long> t(n);
    for (auto &x : t) cin >> x;

    sort(t.begin(), t.end());

    // A snapshot at integer time s captures every pulse with s <= t < s + L
    // (the right end s+L is EXCLUDED -- half-open window [s, s+L)).
    // Greedy: take the earliest still-uncaptured pulse p, place s = t[p].
    // That window covers exactly the pulses with value in [t[p], t[p]+L).
    long long snapshots = 0;
    int i = 0;
    while (i < n) {
        snapshots++;
        long long cover_end = t[i] + L;        // exclusive right boundary
        while (i < n && t[i] < cover_end) i++;  // strict: t[i] == cover_end is NOT covered
    }

    cout << snapshots << "\n";
    return 0;
}
```
