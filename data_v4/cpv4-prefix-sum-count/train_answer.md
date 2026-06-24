**Problem.** Given `n` readings `a[0..n-1]` (possibly negative or zero) and a period `m`, count the
contiguous windows `[l, r]` (`0 <= l <= r <= n-1`) whose sum `a[l] + ... + a[r]` is divisible by `m`
(`0` counts as a multiple). Read `n`, `m`, and the values from stdin; print the count.

**Why direct enumeration fails.** There are up to `n*(n+1)/2 ≈ 2*10^10` windows for `n = 2*10^5`, so
the `O(n^2)` enumerate-and-test reference is both too slow for 1 second and produces a count that
overflows 32-bit. It survives only as the brute-force oracle.

**Key idea — prefix-residue bucketing.** Let `S[0] = 0` and `S[k] = a[0] + ... + a[k-1]`. The window
`[l, r]` has sum `S[r+1] - S[l]`, divisible by `m` iff `S[r+1] ≡ S[l] (mod m)`. So balanced windows
are in bijection with unordered index pairs `0 <= i < j <= n` whose prefix residues are equal. For a
residue class of size `c` that is `C(c, 2) = c*(c-1)/2` pairs; equivalently, stream the prefixes and,
for each new residue, add the number of earlier prefixes with that residue *before* incrementing its
bucket. Both give `0 + 1 + ... + (c-1) = c*(c-1)/2` per class. `O(n + m)` time, `O(m)` space.

**Correctness.** The streaming `answer += cnt[pref]` over a residue class encountered in order sums to
exactly `c*(c-1)/2`, the number of unordered pairs in that class, and the bijection above maps each
such pair to one balanced window and back. Seeding `cnt[0] = 1` injects the empty prefix `S[0] = 0`,
the left endpoint of every window that starts at index `0`.

**Pitfalls.**
1. *Double-count via `c^2`.* The pairs in a class of size `c` number `c*(c-1)/2`, not `c*c/2`:
   `c*c` counts ordered pairs including the diagonal `i = i`, so `c*c/2` overcounts. A trace of
   `m = 2, a = [2, 4]` (truth `3`) returns `4` under `c*c/2`.
2. *Dropped empty prefix.* Without seeding `cnt[0] = 1`, every window starting at index `0` is lost
   — a systematic undercount. A trace of `m = 5, a = [5]` (truth `1`) returns `0`.
3. *Negative residues.* C++'s `%` keeps the dividend's sign, so a negative running sum yields a
   negative remainder and indexes `cnt` out of bounds. Normalize with `((x % m) + m) % m`; the double
   mod is needed so a value like `-2` maps to `m-1`, not to `m`. Trace `m = 3, a = [-2, 2]`.
4. *Overflow.* For `m = 1` the answer is `n*(n+1)/2 ≈ 2*10^10`; the accumulator must be `long long`.

**Edge cases.** `n = 0` → `0` (loop never runs). `n = 1` → `1` iff `a[0] % m == 0`. `m = 1` → all
`n*(n+1)/2` windows. Zeros in `a` extend a residue class and need no special case. `cnt` uses
`m <= 10^6` longs (8 MB), within memory.

**Complexity.** `O(n + m)` time, `O(m)` extra space.

**Code.**

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    long long m;
    if (!(cin >> n >> m)) return 0;
    vector<long long> a(n);
    for (auto &x : a) cin >> x;

    // prefix[r] = (a[0]+...+a[r-1]) mod m, with prefix[0] = 0 (empty prefix).
    // A window [l, r-1] (0-based, 0 <= l <= r-1) has sum divisible by m iff
    // prefix[l] == prefix[r] (as residues). Count unordered pairs i < j with
    // prefix[i] == prefix[j]: for each residue with count c, that is c*(c-1)/2.
    // Negative values: normalize residue into [0, m-1].
    vector<long long> cnt(m, 0);
    long long pref = 0;
    cnt[0] = 1; // the empty prefix prefix[0] = 0
    long long answer = 0;
    for (int i = 0; i < n; i++) {
        pref = ((pref + a[i]) % m + m) % m; // normalize into [0, m-1]
        // every earlier prefix equal to this residue closes a balanced window ending at i
        answer += cnt[pref];
        cnt[pref]++;
    }

    cout << answer << "\n";
    return 0;
}
```
