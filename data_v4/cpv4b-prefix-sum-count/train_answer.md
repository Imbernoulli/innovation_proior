**Problem.** Given a ledger `a[0..n-1]` (values any sign, zeros allowed) and a target `S`, count the
non-empty contiguous runs `[l, r]` whose sum `a[l] + ... + a[r]` equals exactly `S`. Read `n`, `S`,
and the values from stdin; print the count.

**Key idea — prefix-sum hashing.** With `P[0] = 0` and `P[k] = a[0] + ... + a[k-1]`, the run `[l, r]`
sums to `S` iff `P[r+1] - P[l] = S`, i.e. `P[l] = P[r+1] - S`. Sweep the right edge `r+1 = 1..n`
keeping a frequency map of the prefix sums seen so far; at each step add the stored frequency of
`P[r+1] - S` to the answer. The invariant that makes this correct: while processing `P[r+1]`, the map
holds exactly `{P[0], ..., P[r]}` — every valid left endpoint and nothing else. So **query the map,
then insert `P[r+1]`**, and seed `P[0] = 0` once before the loop. `O(n)` time, `O(n)` space.

**Pitfalls.**
1. *Length-0 double-count (the big one).* If you insert the current prefix `P[r+1]` before querying,
   it matches itself whenever `P[r+1] - S = P[r+1]` — i.e. for **every** position when `S = 0` —
   counting the empty run `[r+1, r]`. On `a = [1, -1]`, `S = 0` this returns `3` instead of `1`.
   Query first, insert second.
2. *Empty-prefix seeding.* Seed `P[0] = 0` exactly once. Seeding it twice turns the all-zero
   `[0,0,0]`, `S = 0` answer from the correct `6` into `9`.
3. *Overflow.* The count reaches `n*(n+1)/2 ~ 2*10^10` (all-zero array), and prefix sums reach
   `~2*10^14`, with `S` up to `10^14`; use `long long` for the prefix, the map keys/values, `S`, and
   the answer. An `int` is a silent wrong-answer on large tests.

**Edge cases.** `n = 0` -> `0` (empty loop). `n = 1` matches iff the lone value equals `S`.
Unreachable `S` -> `0`. Negative `a[i]` make `P[r+1] - S` negative — fine, the map keys on the exact
signed value (this is exact-sum, not divisibility, so there is no modulo-sign trap). `S = 0` with
cancellation (e.g. `[5,-5,5,-5]` -> `4`) is handled because a current prefix never matches itself.

**Complexity.** `O(n)` expected time, `O(n)` space.

**Code.**

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    long long S;
    if (!(cin >> n >> S)) return 0;
    vector<long long> a(n);
    for (auto &x : a) cin >> x;

    // Count non-empty subarrays [l, r] with a[l] + ... + a[r] == S.
    // Prefix sums: P[0] = 0, P[k] = a[0] + ... + a[k-1].
    // [l, r] sums to S  <=>  P[r+1] - P[l] = S  <=>  P[l] = P[r+1] - S.
    // Sweep r+1 = 1..n. BEFORE inserting P[r+1], the map holds exactly
    // {P[0], ..., P[r]} = all valid left endpoints l in [0, r], so a length-0
    // subarray (l = r+1) can never be matched. Order matters: query then insert.
    unordered_map<long long, long long> cnt;
    cnt.reserve(n * 2 + 16);
    cnt.max_load_factor(0.7);

    long long pref = 0;        // P[0] = 0, the empty prefix (a valid left endpoint l = 0)
    cnt[pref] = 1;             // seed P[0] exactly once
    long long answer = 0;
    for (int i = 0; i < n; i++) {
        pref += a[i];          // now pref = P[i+1]
        auto it = cnt.find(pref - S);   // need P[l] = P[i+1] - S
        if (it != cnt.end()) answer += it->second;
        cnt[pref] += 1;        // insert P[i+1] AFTER querying
    }

    cout << answer << "\n";
    return 0;
}
```
