**Problem.** A row of `n` containers has positive weights `w[0..n-1]`. A left crane removes a prefix
of `i` containers, a right crane removes a suffix of `j` containers; they must not overlap
(`i + j <= n`) and they share one fuel tank, so `pref[i] + suf[j] <= B`, where `pref[i]`/`suf[j]` are
the prefix/suffix weight sums. Maximize the number removed, `i + j`. Read `n`, `B`, and the weights
from stdin; print the maximum count.

**Why the obvious greedy is wrong.** "Keep a pointer at each end and always lift whichever end's next
container is lighter, until neither fits" looks optimal — spend the least fuel per container to fit
the most. But the budget is *shared and global*, so a locally cheap pick can starve the other end. On
`w = [3, 4, 6, 6, 3, 4]` with `B = 10`, the greedy lifts `w[0] = 3` then `w[1] = 4` (fuel `7`), after
which both ends cost `6`/`4` and nothing more fits, for **2** containers. But taking the single
`w[0] = 3` on the left and the last two `w[5] = 4`, `w[4] = 3` on the right costs `3 + 7 = 10 <= B`
and removes **3**. Greedy is discarded.

**Key idea — two-pointer sweep over prefix lengths.** Precompute `pref` and `suf`. For each affordable
prefix length `i` (i.e. `pref[i] <= B`), the best suffix length is the largest `j` with
`suf[j] <= B - pref[i]` and `j <= n - i`; the answer is `max_i (i + J(i))`. The pointer is `O(n)`
because `J(i)` is **non-increasing** in `i`: as `i` grows, `pref[i]` rises (shrinking the fuel cap)
and `n - i` falls (tightening the overlap cap), and a tighter cap can only lower the feasible `j`. So
start `j = n` and slide it *inward* as `i` increases. (Numeric check on `w = [1,7,7,8,2,8]`, `B = 11`:
`pref = [0,1,8,15,...]`, `suf = [0,8,10,18,...]`, giving `J(0)=2, J(1)=2, J(2)=0` and answer
`1 + 2 = 3`.)

**Pitfalls.**
1. *Pointer direction.* The feasible suffix length *decreases* with `i`, so the pointer must **shrink**
   from `n`, not grow from `0`. A grow-only pointer keeps a stale, over-budget `j`: on `w = [2, 5]`,
   `B = 6` it reports the infeasible pair `i = 1, j = 1` (cost `2 + 5 = 7 > 6`) for `2` instead of `1`.
2. *Overlap cap.* The `j <= n - i` clamp is what enforces no double-counting. Omit it and a loose
   budget makes the prefix and suffix both grab everything: on `w = [1,1,1,1]`, `B = 100` you get `8`
   instead of `4`. Clamp `j` to `n - i` *before* the fuel shrink.
3. *Overflow.* With `n` up to `2*10^5`, `w[i]` up to `10^9`, and `B` up to `10^18`, all sums and the
   budget need `long long`. Compute `B - pref[i]` only after checking `pref[i] <= B` so it never goes
   negative.

**Edge cases (all handled by the sweep + caps):** `n = 0` -> `0`; `B = 0` -> `0`; `B >=` total weight
-> `n` (everything, via the overlap cap); a single too-heavy container -> `0`; an optimum entirely on
one side is reached at `i = 0` (all suffix) or `i = pmax, j = 0` (all prefix).

**Complexity.** `O(n)` time (one upward `i` pass plus one monotone inward `j` pointer), `O(n)` space
for the prefix/suffix arrays.

**Code.**

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    long long B;
    if (!(cin >> n >> B)) return 0;
    vector<long long> w(n);
    for (auto &x : w) cin >> x;

    // Left crane removes a prefix of i containers; right crane removes a suffix
    // of j containers. They share one fuel budget B, so pref[i] + suf[j] <= B,
    // and they must not overlap, so i + j <= n. Maximize i + j.
    //
    // pref[i] = sum of first i containers; suf[j] = sum of last j containers.
    vector<long long> pref(n + 1, 0);
    for (int i = 0; i < n; i++) pref[i + 1] = pref[i] + w[i];
    vector<long long> suf(n + 1, 0);
    for (int j = 0; j < n; j++) suf[j + 1] = suf[j] + w[n - 1 - j];

    // Sweep i (prefix count) upward over every affordable prefix. For each i the
    // suffix may use at most n - i containers and at most B - pref[i] fuel. As i
    // increases, pref[i] grows so the suffix budget shrinks; the overlap bound
    // n - i also shrinks. Hence the largest affordable suffix count j is
    // non-increasing in i, so a single pointer sliding inward gives O(n) total.
    long long best = 0;
    int j = n; // largest suffix count we will ever consider; shrinks as i grows
    for (int i = 0; i <= n; i++) {
        if (pref[i] > B) break;            // no longer affordable; larger i is worse
        if (j > n - i) j = n - i;          // respect the no-overlap bound
        while (j > 0 && (suf[j] > B - pref[i])) j--; // shrink until suffix fits
        best = max(best, (long long)i + j);
    }

    cout << best << "\n";
    return 0;
}
```
