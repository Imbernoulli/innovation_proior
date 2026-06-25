**Problem.** Given a ledger `a[0..n-1]` of signed integers and a fixed target `S`, count the contiguous windows `[l, r]` (`0 <= l <= r <= n-1`) whose sum equals exactly `S`. Read `n` and `S`, then the `n` values, from stdin; print the count.

**Key idea — prefix-sum + hash-map sweep.** With prefixes `P[0]=0`, `P[k]=a[0]+...+a[k-1]`, a window `[l, r]` sums to `S` iff `P[r+1] - P[l] = S`. So the answer counts index pairs `(i, j)`, `i < j`, with `P[j] - P[i] = S`. Sweep `j` left to right keeping `seen[value]` = how many *earlier* prefixes had that value; at each step add `seen[P[j] - S]` to the answer, then insert `P[j]`. Seed `seen[0] = 1` for the empty prefix `P[0]`. `O(n)` expected time, `O(n)` memory.

**Pitfalls.**
1. *32-bit overflow of the answer (the headline trap).* With `n` up to `2*10^5`, the count can reach `n(n+1)/2`. On the all-zeros array with `S = 0` every window matches, giving `200000*200001/2 = 20000100000 ~ 2*10^10`. That exceeds `INT_MAX = 2147483647`; a 32-bit accumulator silently wraps `20000100000` to the *negative* value `-1474736480`. The accumulator must be `long long`. No small sample exposes this — only a large traced case does.
2. *32-bit overflow of the prefix / S.* A prefix total reaches `2*10^14` and `|S|` up to `2*10^14`, so `prefix - S` spans about `+-4*10^14`. The prefix, the target, and the map's keys must all be `long long`, or the lookup keys themselves get truncated.
3. *Bookkeeping.* Seed the empty prefix `seen[0] = 1` (else windows starting at `l = 0` are never counted), and **query before inserting** the current prefix (else a prefix matches itself when `S = 0`, counting phantom zero-length windows).

**Edge cases.** `n = 0` -> `0` (loop never runs). Single element: `S=5, a=[5]` -> `1`; `S=0, a=[7]` -> `0`. Negative `S` reachable (`S=-2, a=[1,-3,1]` -> `2`). Huge unreachable `S` -> `0` with no overflow in `prefix - S`. All-`10^9` values with `S=10^9` -> `200000` while `prefix` climbs to `2*10^14`.

**Complexity.** `O(n)` expected time (hash map), `O(n)` extra space.

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

    // count[p] = how many prefix sums P[0..j-1] equal value p have been seen so far.
    // We accumulate the answer as we sweep j from 0..n-1.
    unordered_map<long long, long long> seen;
    seen.reserve(n * 2);
    seen.max_load_factor(0.7f);

    long long prefix = 0;        // P[j] after consuming a[0..j-1]; prefix sum can exceed 32-bit
    long long answer = 0;        // number of subarrays; can reach ~n^2/2, exceeds 32-bit
    seen[0] = 1;                 // empty prefix P[0] = 0 seen once

    for (int i = 0; i < n; i++) {
        long long x;
        cin >> x;
        prefix += x;             // prefix = P[i+1]
        // subarrays ending at index i with sum S correspond to a prior prefix == prefix - S
        auto it = seen.find(prefix - S);
        if (it != seen.end()) answer += it->second;
        seen[prefix] += 1;
    }

    cout << answer << "\n";
    return 0;
}
```
