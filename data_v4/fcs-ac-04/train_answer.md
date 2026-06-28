**Problem.** There are `n` piles; pile `i` holds `a[i]` stones (`1 <= a[i] <= 10^9`, `0 <= n <= 10^5`). Players alternate. A move picks a pile of size `x > 1` and replaces it with `y` stones where `y` is a proper divisor of `x` (`y | x`, `1 <= y < x`); a pile of size `1` is dead. The player who cannot move loses (normal play). Read `n` and the sizes from stdin; print `First` if the first player wins under optimal play, else `Second`.

**Why the obvious approach is hopeless.** Memoized minimax over the *combined* multiset of piles is exponential in the number of piles: even three piles of `12` need on the order of `12^3` joint states, and with `n` up to `10^5` the joint state space is the product over `10^5` piles — unenumerable. That minimax is only good as a small-case oracle, not the solution.

**Key idea — Sprague–Grundy + the per-pile Grundy is `Omega`.** A move touches exactly one pile, so the position is a disjunctive sum of `n` independent one-pile games. By the Sprague–Grundy theorem the value of the sum is the XOR of the per-pile Grundy values, and the first player wins iff that XOR is nonzero. For a single pile of size `x`, `G(x) = mex { G(y) : y a proper divisor of x }`. Computing the first values gives `0,1,1,2,1,2,1,3,2,2,1,3,1,2,2,4` for `x = 1..16`, which is exactly **`Omega(x)`, the number of prime factors of `x` counted with multiplicity**. So:

> answer = XOR over all piles of `Omega(a[i])`; print `First` iff it is nonzero.

**Proof that `G(x) = Omega(x)`.** Induction on `x`. Base `Omega(1) = 0 = G(1)`. For `x`, write `x` as a multiset of `m = Omega(x)` primes. Every proper divisor `y` has `Omega(y) in {0,...,m-1}` (a proper divisor drops at least one prime factor and cannot keep all `m`), and *every* value `t in {0,...,m-1}` is realized by the product of some `t` of those primes. So by the inductive hypothesis the option Grundy values are exactly `{0,1,...,m-1}` — a complete prefix missing only `m` — hence `mex = m = Omega(x)`.

**Computing `Omega(x)` for `x <= 10^9`.** Trial division up to `sqrt(x) <= sqrt(10^9) < 31623` suffices: strip each prime `p <= sqrt(x)` counting multiplicities, and any cofactor `> 1` left over is a single prime (it cannot be a product of two factors both above `sqrt(x)`), so it adds exactly one. Sieve the primes up to `31623` once (linear sieve), then divide only by those. This is the right tool at `10^9` — Pollard–Rho is for `10^{12}`+ scales, and sieving `Omega` up to `10^9` is impossible in memory.

**Pitfalls to get right.**
1. *Wrong game model.* This is **not** ordinary Nim; do not XOR the sizes `a[i]`. A move is "replace `x` by a divisor", and the correct per-pile value is `Omega(x)`, derived and proved, not assumed.
2. *Integer overflow in the trial-division guard.* The loop bound `p * p > x` must be computed in 64-bit: `(long long)p * p`. With `p` near `31623`, an `int` square is near `10^9` (still inside `int` *here*), but the guard's correctness must not depend on `LIM` staying below `~46340`; promoting to `long long` removes the latent silent-wrong-answer.
3. *The leftover cofactor.* After stripping all primes `<= sqrt(current x)`, an `x > 1` remainder counts as one prime — forgetting this undercounts `Omega` for any value with a large prime factor.

**Edge cases (all fall out of `Omega(1)=0` and the final nonzero-XOR test).** `n = 0` → XOR `0` → `Second` (First cannot move). All piles size `1` → each `Omega = 0` → `Second`. Single prime → `Omega = 1` → `First`. Two equal piles (e.g. `[8,8]`, `[4,9]`) → equal Grundy values XOR to `0` → `Second` (mirroring strategy). Largest values: `Omega <= 30`, so the XOR fits in an `int`; worst case `10^5` large primes factor in well under the time limit.

**Complexity.** `O(LIM)` for the sieve plus `O(n * pi(sqrt(amax)))` for factorization — about `10^5 * 3400` divisions worst case — `O(1)` extra space beyond the sieve. Comfortably within 2 s / 256 MB.

**Code.**

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    if (!(cin >> n)) return 0;              // no piles -> First cannot move -> Second wins

    // Sieve primes up to sqrt(max a[i]) = sqrt(1e9) < 31623, so 31623 suffices.
    const int LIM = 31623;
    vector<int> spf(LIM + 1, 0);            // smallest prime factor; 0 marks "prime/unset"
    vector<int> primes;
    for (int i = 2; i <= LIM; i++) {
        if (spf[i] == 0) { spf[i] = i; primes.push_back(i); }
        for (int p : primes) {
            if ((long long)p * i > LIM) break;
            spf[p * i] = p;
            if (p == spf[i]) break;
        }
    }

    // For each pile x, the Grundy value of the "move to a proper divisor" game is
    // Omega(x) = number of prime factors of x counted with multiplicity.
    // Sprague-Grundy: the whole game's value is the XOR of per-pile Grundy values;
    // the first player wins iff that XOR is nonzero.
    int nimXor = 0;
    for (int i = 0; i < n; i++) {
        long long x;
        cin >> x;
        int omega = 0;                      // Omega(x), prime-factor count with multiplicity
        for (int p : primes) {
            if ((long long)p * p > x) break;
            while (x % p == 0) { x /= p; omega++; }
        }
        if (x > 1) omega++;                 // remaining cofactor is a single prime
        nimXor ^= omega;                    // XOR the per-pile Grundy values
    }

    cout << (nimXor != 0 ? "First" : "Second") << "\n";
    return 0;
}
```
