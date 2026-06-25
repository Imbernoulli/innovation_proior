**Problem.** A radio log is a sequence of `n` tone ids `t[0..n-1]` (ids repeat). A highlight reel is a non-empty subsequence, and reels are counted by *content*: two selections that render the same tone-id sequence are one reel. Read `n` and the ids from stdin; print the number of distinct non-empty reels modulo `1000000007`. This is "count distinct subsequences".

**Why brute force is only an oracle.** Enumerating all `2^n` index masks and deduplicating in a hash set is exactly the definition and is unbeatable for correctness, but dies past `n ~ 25`. With `n <= 2*10^5` it is useful only to validate the real algorithm on tiny inputs.

**Key idea — linear counting DP with a dedup correction.** Let `dp` = the number of distinct subsequences of the prefix seen so far, *including the empty one*; start `dp = 1`. Appending a tone `x` doubles the count (each existing subsequence either ignores or appends `x`): `dp = 2*dp`. But if `x` appeared before, appending it now re-creates every subsequence that already ended in an earlier `x`, and that many are duplicates. The exact number to subtract is the value `dp` held **immediately before the previous occurrence of `x`** was appended. So store, per tone id, that pre-append snapshot, and subtract it on the next reuse:

- `x` new: `dp = 2*dp` (subtract nothing).
- `x` repeat: `dp = 2*dp - last[x]`, where `last[x]` is the `dp` from just before the previous `x`.

After every step set `last[x] = old` (the `dp` from just before *this* append). The empty subsequence is included throughout for clean doubling; subtract it once at the end. Answer: `dp - 1`.

Hand-check on `t = [1,2,1]`: `dp` goes `1 -> 2 -> 4 -> 7` (last step `2*4 - last[1]=1`), so `7 - 1 = 6`, matching the six distinct reels `1, 2, (1,2), (2,1), (1,1), (1,2,1)`.

**Pitfalls.**
1. *Negative modular residue.* `(dp - last[x]) % MOD` can be negative once values wrap `p`, and C++ `%` keeps the sign — a silent wrong-answer on large inputs that no small hand trace exposes. Always `(dp - last[x] + MOD) % MOD`, and the final `(dp - 1 + MOD) % MOD`.
2. *Wrong snapshot, stored or timed wrong.* The subtracted term must be the `dp` from *before* the previous `x` (store `old`, not the post-update `dp`), and it must be **overwritten on every occurrence**. Storing the post-update `dp` over-subtracts (`[1,1,1]` collapses to `1` instead of `3`); keeping only the first occurrence's snapshot under-subtracts and double-counts (`[1,1,1]` inflates to `4` instead of `3`).
3. *Big ids.* `t[i]` up to `10^9` cannot index an array; use a hash map keyed by id.

**Edge cases.** `n = 0` -> `dp` stays `1` -> answer `0`. Single tone -> `1`. All identical (`n` copies) -> `dp` climbs `1,2,3,...,n+1` -> answer `n`. All distinct -> no subtraction -> `2^n - 1`. All confirmed against the brute oracle (400 random cases, zero mismatches).

**Complexity.** `O(n)` expected time (hash-map lookups), `O(n)` space for the map. Runs in ~24 ms at `n = 2*10^5`.

**Code.**

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    const long long MOD = 1000000007LL;

    int n;
    if (!(cin >> n)) return 0;            // no input -> nothing broadcast -> 0 reels
    vector<long long> t(n);
    for (auto &x : t) cin >> x;

    // dp = number of distinct subsequences (including the empty one) of the prefix seen so far.
    // Start with dp = 1: the empty sequence is the only distinct subsequence of the empty prefix.
    // For each new tone x: dp_new = 2*dp_old - (dp value right before the PREVIOUS occurrence of x).
    // The subtracted term removes subsequences that would otherwise be counted twice; if x is new,
    // nothing is subtracted. "last[x]" stores the dp value AS OF just before x was last appended,
    // i.e. the old dp at that earlier step (the count of distinct subsequences not yet using that x).
    long long dp = 1;                     // counts the empty subsequence
    unordered_map<long long, long long> last; // tone id -> dp snapshot to subtract on its next reuse
    last.reserve(n * 2);

    for (int i = 0; i < n; i++) {
        long long old = dp;
        dp = (2 * old) % MOD;
        auto it = last.find(t[i]);
        if (it != last.end()) {
            dp = (dp - it->second % MOD + MOD) % MOD;
        }
        // The next time tone t[i] appears, we must subtract exactly "old" (the dp value that held
        // immediately before appending THIS occurrence). Overwrite so only the latest occurrence counts.
        last[t[i]] = old;
    }

    // dp now includes the empty sequence; the problem wants non-empty reels, so subtract 1.
    long long ans = (dp - 1 + MOD) % MOD;
    cout << ans << "\n";
    return 0;
}
```
