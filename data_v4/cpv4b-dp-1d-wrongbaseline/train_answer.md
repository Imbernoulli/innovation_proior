**Problem.** Slots `a[0..n-1]` sit on a closed loop (`n-1` wraps to `0`). Ship one **non-empty contiguous run** of length `1..n` — the run may wrap past the seam, each slot used at most once — and maximize the shipped sum. This is the maximum circular subarray sum with the non-empty-segment rule. Read `n` and the values from stdin, print the maximum.

**Key idea — guarded circular Kadane.** A circular run is either non-wrapping (a plain subarray) or wrapping (a suffix glued to a prefix). Compute, in one pass:

- `best` = best non-wrapping run = ordinary Kadane (`cur = max(a[i], cur + a[i])`, `best = max(best, cur)`).
- `total` = sum of all slots.
- `worst` = minimum non-empty subarray sum = Kadane with min/`+` flipped.

A wrapping run is "the whole loop minus a contiguous gap left out", and that gap is a non-wrapping subarray, so the best wrapping run is `total - worst`. The naive answer is `max(best, total - worst)`.

**Why the textbook formula is wrong here.** `total - worst` is only a *legal* run when the dropped gap is a proper, non-empty subarray. When the minimum subarray is the *entire belt* — which happens exactly when `worst == total` (the all-negative regime) — `total - worst` equals `0`, which corresponds to the **empty** run. The empty run is forbidden by this contract, so the bare formula returns `0` instead of the correct (negative) answer. Concretely on `a = [-3, -1, -4]`: `best = -1`, `total = -8`, `worst = -8`, so `total - worst = 0` and the textbook `max(-1, 0) = 0` — but the true answer is `-1` (ship the single least-negative slot). The fix is the guard:

```
if (worst == total) answer = best;            // wrap would be empty -> illegal
else                answer = max(best, total - worst);
```

`best` is plain Kadane on a non-empty array, hence always a legal non-empty run, so the fallback is safe.

**Pitfalls.**
1. *The empty-wrap degeneracy.* Never ship `max(best, total - worst)` unguarded. Guard on `worst == total`; on all-negative belts it flips a wrong `0` to the correct least-negative element. (Trace `[-3,-1,-4]` to see the `0`.)
2. *Overflow.* With `n` up to `2*10^5` and `|a[i]|` up to `10^9`, `total` reaches `~2*10^14`; use `long long`. An `int` is a silent wrong-answer on large tests.

**Edge cases.** `n = 1` (`[-7]` -> `-7`; `[9]` -> `9`: the guard fires since `worst==total`, and `best` is the lone slot); all-positive (best run is the whole loop, `best == total`, and `total - worst < total` since dropping a positive gap shrinks it); zeros (`worst==total==0` -> `best=0`); all-negative (guard fires). `n = 0` is guarded defensively though the contract says `n >= 1`.

**Complexity.** `O(n)` time (one interleaved pass of two Kadane scans), `O(1)` extra space.

**Code.**

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    if (!(cin >> n)) return 0;
    vector<long long> a(n);
    for (auto &x : a) cin >> x;
    if (n == 0) { cout << 0 << "\n"; return 0; }  // contract guarantees n>=1, but be safe

    // Linear (non-wrapping) maximum subarray sum, non-empty (Kadane).
    long long best = a[0], cur = a[0];
    long long total = a[0];
    // Linear minimum subarray sum, non-empty (Kadane on negated logic).
    long long worst = a[0], curMin = a[0];
    for (int i = 1; i < n; i++) {
        cur = max(a[i], cur + a[i]);
        best = max(best, cur);
        curMin = min(a[i], curMin + a[i]);
        worst = min(worst, curMin);
        total += a[i];
    }

    // Wrapping candidate: total minus the minimum interior subarray.
    // If worst == total then every element is in the minimum subarray, i.e.
    // the "complement" wrap would be empty; that is illegal (segment must be
    // non-empty), so we must NOT take the wrap in that case.
    long long answer;
    if (worst == total) {
        // All elements lie in the minimum subarray => array is all non-positive
        // in the sense that the best non-empty pick is just the linear best.
        answer = best;
    } else {
        answer = max(best, total - worst);
    }

    cout << answer << "\n";
    return 0;
}
```
