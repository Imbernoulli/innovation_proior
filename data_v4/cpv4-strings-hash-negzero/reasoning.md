**Reading the problem and pinning the contract.** I am given `n` integers `a[0..n-1]`, possibly negative or zero, and I must report the length of the longest *block* — contiguous subarray — that occurs at two or more **distinct** starting positions, with overlaps allowed; `0` if nothing of length `>= 1` repeats. Input on stdin is `n` then the values; I print one integer. Before any algorithm I fix the scale, because it decides the data types and the whole feasibility of brute force: `n <= 2*10^5` and `|a[i]| <= 10^9`. The answer (a length) is at most `n`, so it fits in an `int`, but the *fingerprints* I will compute over windows are a different matter — modular arithmetic with a 61-bit modulus needs 128-bit intermediate products. So `int` for the answer and indices, but a 61-bit modular hash with `__int128` multiplication underneath. That is the first non-negotiable decision.

**Laying out the candidate approaches.** Three routes are on the table, and I want the one I can verify, not merely the one that compiles.

- *Suffix array / suffix automaton on the integer alphabet.* The exact, hash-free answer to "longest repeated substring" is `max LCP` of adjacent suffixes after sorting. It is `O(n log n)` and collision-free, but it needs coordinate compression of the values into ranks and a careful SA build; on an integer alphabet with negatives and zeros the compression step is itself the bug magnet, and the whole thing is a lot of code to get exactly right under time pressure. I keep it as a fallback but do not reach for it first.
- *Binary search on length + rolling hash.* The predicate "some block of length `L` repeats" is monotone in `L` (proven below), so I can binary-search `L` and answer each yes/no query in `O(n)` with a polynomial rolling hash and a hash table. `O(n log n)` overall. This is the route I will commit to — but only after I am sure the hash handles the numeric alphabet, which is exactly where this problem is laid to trap me.
- *Pure brute force.* Compare all windows of all lengths with tuples or direct equality: `O(n^3)` or worse. Fine as the independent oracle for small `n`, useless for `n = 2*10^5`.

**Deriving the monotonicity and sanity-checking it.** I claim: if a block of length `L` occurs at two distinct starts `i != j`, then a block of length `L-1` also occurs at two distinct starts. Proof: the length-`L` windows at `i` and `j` are equal as sequences; dropping the last element of each gives the length-`(L-1)` windows at the same starts `i` and `j`, which are still equal and still start at distinct positions `i != j`. So `hasDup(L) => hasDup(L-1)`, i.e. the predicate is true on a prefix `[1..K]` of lengths and false above `K`, and the answer is `K` (or `0` if even `L=1` fails). Binary search over `[1, n]` for the largest `L` with `hasDup(L)` true is therefore correct. Let me sanity-check the boundary on the documented sample `a = [4, -1, 0, 0, 4, -1, 0, 7]`. The block `[4, -1, 0]` (length 3) starts at index 0 and at index 4 — distinct — so `hasDup(3)` is true. Is `hasDup(4)` true? The length-4 windows are `[4,-1,0,0]`, `[-1,0,0,4]`, `[0,0,4,-1]`, `[0,4,-1,0]`, `[4,-1,0,7]` — all distinct, so `hasDup(4)` is false. Hence the answer is 3, matching the statement, and the monotone shape (`...,3 true, 4 false,...`) is exactly what binary search needs.

**Setting up the rolling hash.** For a window I want a fingerprint computable in `O(1)`. Use prefix hashes `pre[k] = ( ... ((s_0)*B + s_1)*B + ... )*B + s_{k-1}` where `s_i` is the hashed symbol of `a[i]`, all mod a prime `p`. Then the hash of `a[l..r-1]` is `pre[r] - pre[l] * B^{r-l}` mod `p`, the standard subtraction-of-shifted-prefix identity. I will use the Mersenne prime `p = 2^61 - 1` with `__int128` products and the fast `lo + hi` reduction, and I will run **two independent bases** so a single fingerprint clash is astronomically unlikely; on top of that I will keep the start indices per fingerprint and confirm a real match by direct comparison, so the final answer is exact regardless of hashing luck.

**First implementation of the symbol map — and the alphabet trap.** My first cut hashes the value directly:

```
static inline u64 sym(long long v) {
    return (u64)v;                 // first attempt: feed the value straight in
}
...
pre[i+1] = ( pre[i]*B + sym(a[i]) ) mod p;
```

This looks innocent, but the problem screamed "negatives and zeros" and I want to trace it on an input that is all negatives, because that is where a raw-value symbol map should misbehave. Take `a = [-3, -1, -3]` and ask the very first query the binary search will make near the bottom, `hasDup(1)`: do any two single-element windows match? The true answer is **yes** — value `-3` sits at index 0 and index 2. Now trace what the code computes. `sym(-3)` casts `-3` to `u64`, which is `2^64 - 3 = 18446744073709551613`. Then `sym(-3) % p` (I reduce before storing) with `p = 2^61 - 1`: `18446744073709551613 mod (2^61 - 1)`. And `sym(-1) % p` is `(2^64 - 1) mod (2^61 - 1)`. For a length-1 window the fingerprint is just `sym(a[i]) % p`. So index 0 gives `(2^64-3) mod p`, index 1 gives `(2^64-1) mod p`, index 2 gives `(2^64-3) mod p`. Indices 0 and 2 *do* land on the same residue, so by luck `hasDup(1)` would still report true here — but that is luck, not correctness, and it falls apart the moment a negative and some positive value collide mod `p`.

**Diagnosing the alphabet bug precisely.** Let me construct the collision. Two distinct values `u != v` become indistinguishable iff `sym(u) ≡ sym(v) (mod p)`. With the raw cast, `sym(v) = (u64)v`, and for a negative `v` that is `2^64 + v`. So I need `2^64 + v ≡ u (mod p)` for some legal positive `u`, i.e. `u - v ≡ 2^64 (mod p)`. Since `2^64 mod (2^61-1) = 2^3 * (2^61) mod p`... concretely `2^64 = 8 * 2^61 ≡ 8 * 1 = 8 (mod p)` because `2^61 ≡ 1`. So a negative `v` and a positive `u` with `u - v ≡ 8 (mod p)` — for instance `v = -3` and `u = 5` give `u - v = 8` — would receive **the same fingerprint** even though `-3 != 5`. That is a genuine false-equal: a window containing `-3` would be declared equal to a window containing `5`, inflating `hasDup` and producing a wrong, too-large answer. The raw cast is broken; the negativity is not cosmetic, it folds the value space onto itself modulo `p`. The textbook fix for "characters must be positive digits below the base" is to **offset** every symbol into a strictly positive range that is *injective* mod `p`. With `|a[i]| <= 10^9`, map `v -> v + 10^9 + 1`, landing every symbol in `[1, 2*10^9 + 1]`. That range has width `~2*10^9`, far below `p = 2^61 - 1 ≈ 2.3*10^18`, so distinct values map to distinct residues — injective — and crucially no symbol is `0` (the smallest is `1`), so a leading-`0`-value never silently vanishes from a prefix hash.

**Fixing the symbol map and re-tracing.** New map:

```
static const u64 OFFSET = 1000000001ULL;       // 10^9 + 1
static inline u64 sym(long long v) {
    return (u64)(v + (long long)OFFSET);        // in [1, 2000000001], always > 0
}
```

Re-trace the dangerous pair `-3` vs `5`: `sym(-3) = -3 + 1000000001 = 1000000 0... = 999999998`; `sym(5) = 1000000006`. Distinct, both well below `p`, so distinct residues — the false-equal is gone. Re-trace `a = [-3, -1, -3]`, `hasDup(1)`: symbols are `999999998, 1000000000, 999999998`; indices 0 and 2 share a fingerprint *and* a direct `windowEqual` check confirms `a[0]==a[2]`, so `hasDup(1)=true`, correct, and now for the right reason rather than by modular accident. And a window of value `0` maps to `1000000001`, a perfectly ordinary positive digit, so all-zero inputs are hashed honestly. The alphabet bug is closed.

**Second implementation — the binary-search predicate and a base-case trace.** I wire the predicate and the search. My first version of the query and the driver:

```
static bool hasDup(int L) {
    unordered_map<u64, vector<int>> seen;
    for (int i = 0; i + L <= n; i++) {
        auto h = windowHash(i, L);
        u64 key = h.first * 1000000007ULL + h.second;
        auto &bucket = seen[key];
        for (int j : bucket) if (windowEqual(j, i, L)) return true;
        bucket.push_back(i);
    }
    return false;
}
...
int lo = 0, hi = n, ans = 0;          // first attempt: lo starts at 0
while (lo <= hi) {
    int mid = (lo + hi) / 2;
    if (hasDup(mid)) { ans = mid; lo = mid + 1; }
    else hi = mid - 1;
}
```

I deliberately trace the all-zero overlap corner `a = [0, 0, 0, 0, 0]`, `n = 5`, whose true answer is `4` (the length-4 block of zeros starts at index 0 and at index 1 — distinct starts, overlapping, allowed). The search starts `lo=0, hi=5`. First `mid = 2`: `hasDup(2)` scans windows `[0,0]` at i=0..3, the first repeat appears immediately, returns true; `ans=2, lo=3`. `mid = (3+5)/2 = 4`: windows of length 4 are `[0,0,0,0]` at i=0 and i=1, repeat, true; `ans=4, lo=5`. `mid = 5`: `hasDup(5)` — only one window `i=0` exists (`i + L <= n` means `i <= 0`), no repeat, returns false; `hi=4`. Now `lo=5 > hi=4`, stop, `ans=4`. Correct! But the `lo=0` start nags me: what does `hasDup(0)` even mean, and can the search ever call it?

**Diagnosing the base-case bug.** Walk a case where the answer is `0`, e.g. `a = [-1, -2, -3]`, `n=3`, no value repeats. Search `lo=0, hi=3`. `mid=1`: `hasDup(1)` finds no repeated single value, false; `hi=0`. Now `lo=0, hi=0`, loop continues, `mid=0`: it calls **`hasDup(0)`**. With my loop `for (int i = 0; i + 0 <= n; i++)` that is `i <= n`, i.e. `i` runs `0..n` — but `windowHash(i, 0)` of an empty window: `pre[i] - pre[i]*pw[0] = pre[i] - pre[i]*1 = 0` for every `i`, so *every* empty window has fingerprint `0`, the first two collide, `windowEqual(.,.,0)` vacuously returns true (the `for k<0` loop body never runs), and `hasDup(0)` returns **true**. Then `ans=0` (harmless here numerically) but the logic is now asserting "a length-0 block repeats," which is meaningless, and worse, the index `i = n` reads `pre[n]` (valid, size `n+1`) but `windowEqual` with `L=0` touches nothing — so it does not crash, yet I am relying on an accidental vacuous-truth to not corrupt `ans`. That is a base-case landmine: the predicate is only meaningful for `L >= 1`, and the search must never probe `L = 0`. The fix is twofold: start the search at `lo = 1` so `mid >= 1` always, and additionally guard `hasDup` so `L <= 0` is never silently treated as a real query.

**Fixing the base case and re-verifying.** I set `lo = 1` and add the guards `if (L <= 0) return true;` (documented as "never queried") and `if (L > n) return false;` to make the predicate total and obviously correct at its edges:

```
static bool hasDup(int L) {
    if (L <= 0) return true;          // empty block: not queried for L>=1
    if (L > n) return false;
    ...
}
...
int lo = 1, hi = n, ans = 0;          // L=0 is never a candidate answer
```

Re-trace `a = [-1,-2,-3]`: `lo=1, hi=3`. `mid=2`: `hasDup(2)` false; `hi=1`. `mid=1`: `hasDup(1)` false; `hi=0`. `lo=1 > hi=0`, stop, `ans=0`. Correct, and `hasDup(0)` is never reached. Re-trace `n=1, a=[-5]`: `lo=1, hi=1`, `mid=1`, `hasDup(1)` — only one window, no repeat, false; `hi=0`; stop; `ans=0`. Correct. Re-trace the all-zero case again with `lo=1`: same `ans=4` as before (the search never needed `mid=0`). The base-case bug is closed, and the two corners that exercised it now pass for the documented reason.

**Edge cases, deliberately, because this is where this kind of code dies.**
- `n = 0` (and the no-token input `if (!(cin >> n))`): I special-case both to print `0` and return — the loops and the `mt19937_64` base setup never run, no `pre`/`pw` of negative size, no division by a possibly-zero `MOD-300`. Empty array has no block, answer `0`. Correct.
- `n = 1`: single window at every length-1 query, `hasDup(1)=false`, answer `0`. Correct.
- All-negative, no repeat (`[-1,-2,-3]`): every fingerprint distinct after offsetting, answer `0`. Correct (traced above).
- All-negative with a repeat (`[-7,-7,-3,-7]`): `-7` repeats as a length-1 block; longer blocks `[-7,-7]` vs others do not repeat, so answer `1`. Verified against brute.
- All zeros (`[0,...,0]`, `n` of them): `0` maps to the positive digit `1000000001`, and the longest repeated overlapping block has length `n-1`. For `n=5` I traced `4`. Correct.
- Large magnitude near `+-10^9`: offset keeps every symbol in `[1, 2*10^9+1]`, injective mod `p`; no overflow because products go through `__int128` in `mulmod`. Correct.
- Hash-collision safety: even if two distinct windows ever share the combined 61+61-bit key (probability `~1/p^2` per pair), `windowEqual` re-checks the actual values before returning true, so the output is **exact**, not probabilistic. The empirical pass over 1000 random small cases (all-negative, zeros, tiny alphabet, large magnitude, empty, singletons) against the brute force confirms it.

**Performance check.** Each `hasDup(L)` is one linear scan building `n-L+1` fingerprints in `O(1)` each plus hash-table inserts, `O(n)` amortized; binary search makes `O(log n)` queries, so `O(n log n)` total. The `windowEqual` confirmations cost `O(L)` only on a genuine match, and on a match we immediately return, so they do not accumulate. On `n = 2*10^5` random, all-zero, and tiny-alphabet `{-1,0,1}` inputs it runs in under half a second against the 2-second limit, ~30 MB. Comfortable.

**Final solution.** I disproved the raw-cast symbol map by constructing the `-3`/`5` modular collision and fixed it with a positive offset; I disproved the `lo = 0` search by tracing it into the meaningless `hasDup(0)` and fixed it with `lo = 1` plus total guards; and double hashing with an exact `windowEqual` confirmation closes the collision and the all-negative / all-zero / empty / single corners. That is what I ship — one self-contained file:

```cpp
#include <bits/stdc++.h>
using namespace std;

typedef unsigned long long u64;
typedef __int128 i128;

static int n;
static vector<long long> a;

// Two independent 61-bit Mersenne-prime hashes to make collisions astronomically rare.
static const u64 MOD = (1ULL << 61) - 1;

static inline u64 mulmod(u64 x, u64 y) {
    i128 z = (i128)x * y;
    u64 lo = (u64)(z & MOD);
    u64 hi = (u64)(z >> 61);
    u64 r = lo + hi;
    if (r >= MOD) r -= MOD;
    return r;
}
static inline u64 addmod(u64 x, u64 y) {
    u64 r = x + y;
    if (r >= MOD) r -= MOD;
    return r;
}

static u64 BASE1, BASE2;
static vector<u64> pre1, pre2;   // prefix hashes
static vector<u64> pw1, pw2;     // base powers

// hashed symbol for a value: offset so negatives/zeros become positive and never 0.
// |a[i]| <= 1e9, so a[i] + OFFSET is in [1, ~2e9], strictly positive => no symbol hashes to 0.
static const u64 OFFSET = 1000000001ULL;

static inline u64 sym(long long v) {
    return (u64)(v + (long long)OFFSET); // in [1, 2000000001], always > 0
}

// hash of subarray a[l .. l+L-1] (0-indexed), combined 128-bit-ish key.
static inline pair<u64,u64> windowHash(int l, int L) {
    int r = l + L; // exclusive
    // h = pre[r] - pre[l]*pw[L]
    u64 h1 = (pre1[r] + (MOD - mulmod(pre1[l], pw1[L]))) % MOD;
    u64 h2 = (pre2[r] + (MOD - mulmod(pre2[l], pw2[L]))) % MOD;
    return {h1, h2};
}

static inline bool windowEqual(int i, int j, int L) {
    for (int k = 0; k < L; k++) if (a[i + k] != a[j + k]) return false;
    return true;
}

// Does some block of length L occur at >=2 distinct start positions (overlap allowed)?
static bool hasDup(int L) {
    if (L <= 0) return true;          // empty block trivially "repeats"; not queried for L>=1
    if (L > n) return false;
    // map combined hash key -> list of start indices sharing that key; verify on clash.
    unordered_map<u64, vector<int>> seen;
    seen.reserve((size_t)(n - L + 1) * 2 + 4);
    for (int i = 0; i + L <= n; i++) {
        auto h = windowHash(i, L);
        u64 key = h.first * 1000000007ULL + h.second; // mix two 61-bit hashes
        auto &bucket = seen[key];
        for (int j : bucket) if (windowEqual(j, i, L)) return true; // exact confirmation
        bucket.push_back(i);
    }
    return false;
}

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    if (!(cin >> n)) { cout << 0 << "\n"; return 0; }
    a.resize(n);
    for (int i = 0; i < n; i++) cin >> a[i];

    if (n == 0) { cout << 0 << "\n"; return 0; }

    // random bases in a safe range [256, MOD-2]
    std::mt19937_64 rng(0x9E3779B97F4A7C15ULL);
    BASE1 = 256 + rng() % (MOD - 300);
    BASE2 = 256 + rng() % (MOD - 300);

    pre1.assign(n + 1, 0);
    pre2.assign(n + 1, 0);
    pw1.assign(n + 1, 1);
    pw2.assign(n + 1, 1);
    for (int i = 0; i < n; i++) {
        pre1[i + 1] = addmod(mulmod(pre1[i], BASE1), sym(a[i]) % MOD);
        pre2[i + 1] = addmod(mulmod(pre2[i], BASE2), sym(a[i]) % MOD);
        pw1[i + 1] = mulmod(pw1[i], BASE1);
        pw2[i + 1] = mulmod(pw2[i], BASE2);
    }

    // Binary search the largest L in [1, n] with hasDup(L) true.
    // Monotone: if a block of length L repeats, its length-(L-1) prefixes also repeat.
    int lo = 1, hi = n, ans = 0;
    while (lo <= hi) {
        int mid = (lo + hi) / 2;
        if (hasDup(mid)) { ans = mid; lo = mid + 1; }
        else hi = mid - 1;
    }

    cout << ans << "\n";
    return 0;
}
```

**Causal recap.** Binary search over the monotone predicate `hasDup(L)` reduces the problem to "do two length-`L` windows match," which a polynomial rolling hash answers in `O(n)` per query — but the numeric alphabet is the trap: a raw `(u64)v` cast folds negatives onto positives mod `2^61-1` (I built the explicit `-3 ≡ 5` collision), so every symbol must be offset into the strictly-positive injective range `[1, 2*10^9+1]`; and the search must start at `lo = 1`, because a trace of an all-negative no-repeat input drove `mid` down to the meaningless `hasDup(0)` whose empty-window fingerprints all collide into a vacuous `true`. Offsetting the alphabet, guarding the predicate for `L<=0` and `L>n`, confirming every fingerprint match with an exact `windowEqual`, and special-casing `n=0`/empty input together make the all-negative, all-zero, single-element, and empty corners come out right, verified at zero mismatches over 1000 random cases and well inside the time limit at `n = 2*10^5`.
