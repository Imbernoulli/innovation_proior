**Problem.** Given `n` integer frequencies `a[0..n-1]`, a modulus `m`, and a target residue `t` with `0 <= t < m`, count the unordered pairs `{i, j}`, `i != j`, with `(a[i] + a[j]) mod m == t`. Read `n m t` and the frequencies from stdin, print the count. With `n` up to `2*10^5`, the answer can approach `C(n, 2) ≈ 2*10^10`.

**Why the obvious brute force is too slow.** Testing every pair is `O(n^2) ≈ 4*10^10` operations at the top of the range — far past a 1-second limit. It is correct, so it serves as the oracle, but it cannot be the submission.

**Key idea — residue bucketing.** The predicate depends only on residues: `{i, j}` is resonant iff `(a[i] mod m) + (a[j] mod m) ≡ t (mod m)`. Let `cnt[r]` count frequencies with residue `r`. As `r` runs over `0..m-1`, the residue that completes it is forced, `s = (t - r) mod m`. Count index pairs by case, visiting each unordered residue pair once:

- `r < s` (distinct residues): add `cnt[r] * cnt[s]` — every element of bucket `r` pairs with every element of bucket `s`.
- `r == s` (self-paired residue, i.e. `2r ≡ t`): add `C(cnt[r], 2) = cnt[r] * (cnt[r] - 1) / 2` — choose 2 distinct indices within the bucket.
- `r > s`: skip; this residue pair was already counted when the loop variable was the smaller residue.

This is `O(n + m)`.

**Correctness.** Every resonant index pair has a unique unordered residue pair `{r, s}` with `r + s ≡ t`. The `r < s` branch counts all cross-bucket index pairs exactly once (the symmetric visit at `r > s` is skipped). The `r == s` branch counts unordered within-bucket pairs as `C(cnt[r], 2)`, excluding an index paired with itself and not double-counting order. Reproducing the sample (`m = 5`, `t = 3`, residue counts `cnt[1]=cnt[2]=cnt[4]=2`): `cnt[1]*cnt[2] + C(cnt[4],2) = 4 + 1 = 5`, matching the expected `5`.

**Pitfalls.**
1. *Overflow (the headline trap).* The answer reaches `C(2*10^5, 2) ≈ 2*10^10` and a single product `cnt[r]*cnt[s]` (or `cnt[r]*(cnt[r]-1)`) reaches `~4*10^10`, both past `INT_MAX ≈ 2.1*10^9`. Because C++ evaluates `cnt[r]*cnt[s]` in the operands' type before adding to `answer`, making only `answer` 64-bit is not enough — `cnt` itself must be `long long` so the products are computed in 64-bit. An `int` here is a silent wrong-answer (the `int` version prints `672547168` on the all-even `n = 2*10^5` case instead of `19999900000`), not a crash.
2. *Self-pair vs. square.* For `r == s` the count is `C(cnt[r], 2)`, never `cnt[r]^2`; the latter counts ordered pairs and self-pairings. `cnt[r]*(cnt[r]-1)` is always even, so the `/2` is exact.
3. *Double counting.* Without the `r < s` guard, each distinct residue pair is added twice.
4. *Negative modulo.* `(t - r)` can be negative; use `((t - r) % m + m) % m` so the index `s` stays in `0..m-1`.

**Edge cases.** `n = 0` and `n = 1` → `0` (no pairs). `m = 1` (forces `t = 0`) → every pair resonant, `C(n, 2)`, also a big-answer case. Self-pair bucket with `cnt = 1` → `0`. Frequencies up to `10^9` reduce fine; the raw sum `a[i]+a[j]` never appears because the fast solution works purely in residues.

**Complexity.** `O(n + m)` time, `O(m)` space.

**Code.**

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    long long m, t;
    if (!(cin >> n >> m >> t)) return 0;     // n machines, modulus m, target residue t

    // cnt[r] = how many frequencies are congruent to r modulo m.
    vector<long long> cnt(m, 0);
    for (int i = 0; i < n; i++) {
        long long x;
        cin >> x;
        long long r = ((x % m) + m) % m;     // safe modulo for any sign (here x >= 0)
        cnt[r]++;
    }

    // We want pairs (i<j) with (a_i + a_j) % m == t.
    // Pair residues (r, s) with (r + s) % m == t. For r != s the count is cnt[r]*cnt[s]
    // (each unordered residue pair counted once); for the self-paired residue r == s it
    // is cnt[r]*(cnt[r]-1)/2 (choose 2 within the bucket).
    long long answer = 0;
    for (long long r = 0; r < m; r++) {
        long long s = ((t - r) % m + m) % m; // residue that completes r to t (mod m)
        if (r < s) {
            answer += cnt[r] * cnt[s];       // cnt[r]*cnt[s] can exceed 32-bit: long long
        } else if (r == s) {
            answer += cnt[r] * (cnt[r] - 1) / 2;
        }
        // r > s already handled when the loop variable was s
    }

    cout << answer << "\n";
    return 0;
}
```
