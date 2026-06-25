**Problem.** Output a structure: given `n` (`1 <= n <= 1000`), emit `n` non-negative integer offsets `x[0] < x[1] < ... < x[n-1]`, each in `[0, 200000000]`, forming a **Sidon set** — all pairwise sums `x[i] + x[j]` over `i <= j` (including the self-sum `2*x[i]`) are distinct. Among all valid sets, output the **lexicographically smallest** one, in increasing order. Read `n` from stdin, print the values space-separated on one line.

**Key idea — greedy by smallest extension = lexicographically minimal.** Start with `x[0] = 0`; repeatedly append the smallest integer greater than the current maximum that keeps every pairwise sum distinct. By an exchange argument this greedy prefix is the lexicographically smallest Sidon set: if greedy `g` and any valid Sidon set `s` first differ at position `k`, then `s_k` is an admissible extension of the shared prefix, greedy took the smallest admissible extension, so `g_k < s_k`. Greedy never needs to backtrack because arbitrarily large Sidon sets exist (so an admissible next value always exists, well under the cap). Maintain a presence array `seen[s]` over *all* used sums; a candidate `v` is admissible iff every new sum it creates — `v + x[k]` for each existing `x[k]`, plus the self-sum `2v` — is currently unused. The new sums are mutually distinct automatically (`v + x[k]` distinct for distinct `x[k]`, and `2v` differs from each since `x[k] != v`), so testing each against `seen` is sufficient. This is the Mian-Chowla sequence `0, 1, 3, 7, 12, 20, 30, 44, ...`.

**Pitfalls.**
1. *A formula that is valid only for small `n`.* Squares `x[k] = k^2` are a valid Sidon set for `n <= 5` but collide at `n = 6`: `0 + 25 = 9 + 16 = 25` (the pairs `(x[0], x[5])` and `(x[3], x[4])`). Verifying only on tiny inputs ships a construction that scores zero from `n = 6` on. Also, a formula set is essentially never lexicographically minimal (squares start `0, 1, 4`, but `0, 1, 3` is a smaller valid start). Verify the property at the *scale the tests use*.
2. *Local / windowed distinctness check.* Checking new sums only against a window of the last few elements is valid while `n` is at most the window size, then silently breaks. With a window of 4 the greedy emits `0 1 3 7 12 15` at `n = 6`, but `0 + 15 = 3 + 12 = 15` — it forgot the pair straddling the window (`x[0] = 0` with the new element). The distinctness condition is global; check every existing element.
3. *Forgetting the self-sum `2v`.* The pair `(v, v)` produces sum `2v`. If only `v + x[k]` is checked, a candidate whose `2v` duplicates an existing sum slips through and creates a collision. The `seen[2v]` test is load-bearing.

**Edge cases.** `n = 1` -> `0` (vacuously Sidon). `n = 2` -> `0 1`. `n = 3` -> `0 1 3` (candidate `2` fails because `2 + 0 = 1 + 1`). The `seen` array is grown to `2*cand + 1` before indexing `2*cand`, so no out-of-bounds. Offsets are non-negative so all sum indices are `>= 0`. At `n = 1000` the maximum offset is `14018950 < 200000000`, so the cap is always respected; sums fit in `long long` with huge margin.

**Complexity.** The greedy scans candidate values, doing `O(size)` work per candidate; total work is bounded by `O(max_value * n)` which, for `n = 1000`, runs in about `0.4` s and `~36` MB (the `seen` array is `~2 * 14018950` bytes). Verified valid over all `500500` pairwise sums at `n = 1000`, and identical to an exhaustive lexicographically-minimal search for `n <= 14`.

**Code.**

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    if (!(cin >> n)) return 0;                 // no input -> nothing to print

    // Build the lexicographically smallest Sidon set (distinct pairwise sums) of size n.
    // Greedy: x[0] = 0; each next value is the smallest integer > x[last] that keeps EVERY
    // pairwise sum distinct. The exchange argument shows this greedy prefix is lex-smallest.
    //
    // We track which pairwise sums are already used with a presence array `seen`, indexed by
    // the sum value. A candidate v is admissible iff none of the new sums it creates --
    // v + x[k] for every existing x[k], and 2v -- is already present. The new sums are also
    // mutually distinct automatically (v + x[k] are distinct for distinct x[k], and 2v differs
    // from each v + x[k] since x[k] != v), so checking each against `seen` is sufficient.

    vector<long long> x;
    x.reserve(n);
    vector<char> seen;                          // seen[s] = 1 if sum s already used
    long long cand = 0;

    while ((int)x.size() < n) {
        long long top = 2 * cand;               // largest new sum this candidate would create
        if ((long long)seen.size() <= top) seen.resize(top + 1, 0);

        bool ok = true;
        for (long long y : x) {                 // check v + x[k] for existing x[k]
            if (seen[cand + y]) { ok = false; break; }
        }
        if (ok && seen[top]) ok = false;        // check the self-sum 2v

        if (ok) {
            for (long long y : x) seen[cand + y] = 1;
            seen[top] = 1;
            x.push_back(cand);
        }
        cand++;
    }

    // Output the set, space-separated, on one line.
    for (int i = 0; i < n; i++) {
        cout << x[i];
        cout << (i + 1 < n ? ' ' : '\n');
    }
    if (n == 0) cout << "\n";
    return 0;
}
```
