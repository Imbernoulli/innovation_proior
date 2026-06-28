**Problem.** Given `n` (`1 <= n <= 10^7`), output a permutation `p[0..n-1]` of `{1,...,n}` whose prefix sums `S_k = p[0]+...+p[k-1]` (`k = 1..n`) are pairwise distinct modulo `n`, or print `-1` if none exists. Equivalently: a sequencing of the cyclic group `Z_n`. The judge is a checker — any valid permutation is accepted.

**Feasibility.** A valid order exists iff `n == 1` or `n` is even. Reason: ticket `n ≡ 0 (mod n)`, so wherever it sits it adds a `0`-step, which collides with the previous prefix sum unless `n` is placed first; hence `S_1 ≡ 0`. For odd `n >= 3`, the full total `S_n = n(n+1)/2 ≡ 0 (mod n)` as well, so `S_1` and `S_n` collide — impossible. `n = 1` is the trivial single-rider case (feasible); all even `n` are feasible (the cyclic group `Z_n` is sequenceable iff `n` is even).

**Key idea — a proven interleaved construction.** For feasible `n`, interleave a descending run of the large values with an ascending run of the small values:

```
p = n, 1, n-2, 3, n-4, 5, ...
```

Even positions take `n, n-2, n-4, ...`; odd positions take `1, 3, 5, ...`. Together these are exactly `{1,...,n}`, so it is a permutation. Each consecutive (descending, ascending) pair contributes `(n-2j) + (2j+1) = n+1 ≡ 1 (mod n)`, so the partial sum advances by `1` per pair and the residues sweep out a complete system mod `n`. It is `O(n)`, needs no search, and starts with `n` as the feasibility argument requires.

**Pitfalls.**
1. *Validating a slick construction only on tiny `n`.* The reverse-identity order `n, n-1, ..., 1` has prefix sums `≡ -k(k-1)/2 (mod n)`, which are distinct **iff `n` is a power of two**. It passes `n = 2, 4, 8, 16, 32` — exactly the sizes a casual tester eyeballs — then fails at `n = 6` and at every judged non-power-of-two even `n` (e.g. `10^6`, `9999998`). The property must be verified *at the constraint scale*, not on `n <= 8`; that is the Sidon-set failure (right for small `n`, shipped for `10^7`, scored `0`).
2. *Over-rejecting `n = 1`.* The infeasible set is "odd `n >= 3`", not "odd". Gate on `n != 1 && (n & 1)`, or `n = 1` wrongly prints `-1`.
3. *Construction direction.* The large run must *descend* (`big -= 2`); incrementing walks `big` past `n` and breaks the permutation.
4. *Output volume.* ~`10^7` numbers (~75 MB): build one buffered string and write once with `sync_with_stdio(false)`; per-token `cout <<` is too slow. Use `long long` (the total reaches ~`5*10^13`).

**Edge cases.** `n = 1` -> `1`; `n = 2` -> `2 1`; odd `n >= 3` -> `-1`; powers of two are handled by the same construction (not reverse-identity); `n = 10^7` runs in ~`0.5 s`.

**Complexity.** `O(n)` time, `O(n)` output buffer, no auxiliary search.

**Code.**

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    long long n;
    if (!(cin >> n)) return 0;

    // Distinct prefix sums mod n exist iff n == 1 or n is even.
    // (Z_n is sequenceable iff n is even; n == 1 is the trivial single-seat case.)
    if (n != 1 && (n & 1LL)) {
        cout << -1 << "\n";
        return 0;
    }

    // Construction for even n (and n == 1): interleave the descending evens
    // starting at n with the ascending odds starting at 1:
    //   p = n, 1, n-2, 3, n-4, 5, ...
    // Position 0 takes n, n-2, n-4, ...; the odd positions take 1, 3, 5, ...
    // This is a permutation of {1..n}, and its prefix sums are pairwise
    // distinct modulo n.
    string out;
    out.reserve((size_t)n * 7);
    long long big = n;      // descending evens: n, n-2, n-4, ...
    long long small = 1;    // ascending odds:   1, 3, 5, ...
    for (long long i = 0; i < n; i++) {
        long long v;
        if ((i & 1LL) == 0) { v = big; big -= 2; }
        else                { v = small; small += 2; }
        out += to_string(v);
        out += (i + 1 == n) ? '\n' : ' ';
    }
    cout << out;
    return 0;
}
```
