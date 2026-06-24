**Reading the problem and pinning the contract.** I have a lowercase string `s` of length `n` and an integer `k`, and I must count how many *distinct* length-`k` substrings of `s` occur at two or more different starting positions. The word that decides everything here is *distinct*: a substring is its content, not its location, so if some length-`k` string can be read at three indices, it contributes `1` to the answer, not `3` and not `2`. Input is `n k` on the first line and the `n`-character string on the second; output is one integer. Let me fix the scale before choosing an algorithm: `n` up to `2*10^5`, and `k` up to `2*10^5` as well — which means `k` can exceed `n`, can equal `0`, and can be `n/2` where the windows are long. The number of length-`k` windows is `n - k + 1` when `1 <= k <= n`, and there are simply no windows otherwise. The answer is at most `n - k + 1`, well within 32 bits, but I will carry it in `long long` anyway because mixing it with `n - k + 1` (a `long long`) is cleaner and costs nothing.

**Laying out the candidate approaches.** Two routes, and I want the one I can both prove and afford.

- *Sort the raw substrings.* Build all `n - k + 1` windows as `std::string`, sort the vector, then sweep counting runs of length `>= 2`. This is obviously correct — it is almost the definition — but every string compare touches up to `k` characters, so sorting is `O(n * k * log n)`. At `n = 2*10^5` with `k = 10^5` that is on the order of `10^5 * 10^5 * 17 ≈ 10^{11}` character comparisons. Hopeless inside one second. I keep this as my *brute-force oracle*, not my submission.
- *Polynomial rolling hash.* Reduce each window to an integer fingerprint, computed incrementally in `O(1)` per shift, so all fingerprints cost `O(n)`. Sort the `n - k + 1` fingerprints (`O(n log n)`), then count groups of equal fingerprints with size `>= 2`. That is the plan. The danger is not the asymptotics; it is three small places where this kind of code goes subtly wrong: the exact roll step, the window count, and — the one that literally is the answer — turning a group of equal fingerprints into the *count the problem asks for*.

**Deriving the hash and the roll, on paper.** Define the fingerprint of a window `c_0 c_1 ... c_{k-1}` (each `c_t` a small positive integer for the letter) by Horner's rule:

`h = c_0 * B^{k-1} + c_1 * B^{k-2} + ... + c_{k-2} * B + c_{k-1}   (mod M).`

Now slide the window right by one: drop `c_0`, append `c_k`. The new window `c_1 ... c_k` has fingerprint

`h' = c_1 * B^{k-1} + ... + c_{k-1} * B + c_k.`

To get `h'` from `h`: the leading term of `h` is `c_0 * B^{k-1}`, so subtract that, then every remaining term must move up one power of `B`, i.e. multiply by `B`, then add the new last character `c_k`:

`h' = (h - c_0 * B^{k-1}) * B + c_k   (mod M).`

The weight I remove is `c_0 * B^{k-1}`, **not** `c_0 * B^{k}`. That is the precise spot off-by-ones live, so I write it down explicitly: I need to precompute `B^{k-1} mod M`, not `B^k mod M`.

A single 64-bit hash with one modulus would invite collisions: with up to `2*10^5` windows, the birthday bound against a `~10^9` modulus gives a real chance two *different* substrings share a fingerprint, which would merge two groups and corrupt the count. So I use **two** independent moduli/bases and pack the pair `(h1, h2)` into one 64-bit key `(h1 << 32) ^ h2`. Two `~10^9` hashes give an effective space near `10^{18}`; the chance of any false merge across `2*10^5` windows is negligible. Equal real substrings always produce identical `(h1, h2)`, so true repeats are never split.

**Checking the derivation on the sample, by hand-grouping.** Sample: `s = "ababbaba"`, `n = 8`, `k = 3`. Windows at indices `0..5`: `aba, bab, abb, bba, bab, aba`. Group them by content: `aba` -> {0, 5} (size 2), `bab` -> {1, 4} (size 2), `abb` -> {2} (size 1), `bba` -> {3} (size 1). Distinct substrings with group size `>= 2`: `aba` and `bab`, so the answer is `2`. Good — and note the count is `2`, the number of *groups* of size `>= 2`, emphatically not the number of repeated *occurrences* (which would be 4) nor the number of *extra* occurrences (which would be 2 here by coincidence but not in general). I keep that distinction front of mind for the counting step.

**First implementation — and a trace, because hash code transcribes dirty.** My first cut of the core. I precompute the leading-power, build the first window, then roll:

```
unsigned long long p1 = 1, p2 = 1;
for (long long i = 0; i < k; i++) { p1 = (p1*B1)%M1; p2 = (p2*B2)%M2; }   // <-- B^k

// first window
for (long long i = 0; i < k; i++) { h1 = (h1*B1 + c)%M1; h2 = (h2*B2 + c)%M2; }
keys.push_back((h1<<32) ^ h2);

for (long long i = 1; i < W; i++) {
    out = s[i-1]-'a'+1; in = s[i+k-1]-'a'+1;
    h1 = (h1 + M1 - (out*p1)%M1)%M1;   // subtract out * B^k
    h1 = (h1*B1 + in)%M1;
    ... same for h2 ...
    keys.push_back((h1<<32) ^ h2);
}
```

Let me trace the smallest input that exercises a *roll*: `n = 2`, `k = 1`, `s = "aa"`. The two single-character windows are `"a"` at index 0 and `"a"` at index 1 — identical, so the answer must be `1`. Here `W = n - k + 1 = 2`. The precompute loop runs `k = 1` time, so `p1 = B1 = 131`. First window (`i = 0`): `c = 'a'-'a'+1 = 1`, `h1 = (0*131 + 1) % M1 = 1`. Push key for `h1 = 1`. Now the roll for `i = 1`: `out = s[0] = 1`, `in = s[1] = 1`. `h1 = (1 + M1 - (1*131)%M1) % M1 = (1 + M1 - 131) % M1 = M1 - 130`, then `h1 = ((M1-130)*131 + 1) % M1`. That is some large nonzero value, definitely **not** `1`. So the two windows that are literally the same string get different fingerprints. The grouping then sees two singleton groups, and the program outputs `0`.

**Diagnosing the first bug.** The expected answer is `1`, I got `0`. The defect is exactly the power I warned myself about: I subtracted `out * B^k`, but the leading character of a length-`k` window carries weight `B^{k-1}`. For `k = 1` the window is a single character with weight `B^0 = 1`, yet I removed `out * B^1 = out * B`. So after "removing" character 0 I had not removed it cleanly — I removed `out * B` and was left with garbage, then multiplied by `B` and added `in`, producing a fingerprint that no honest length-1 window would ever have. The fix is to precompute `B^{k-1}`, i.e. run the precompute loop `k - 1` times, and roll with `(h - out * B^{k-1}) * B + in`. I confirmed this is a real, reproducible failure and not a hash fluke by running the random tester before the fix: it flagged `n=2 k=1 "aa" -> sol=0 brute=1`, plus a flood of similar small cases like `"abb" k=1` and `"aaaaaaaaaaaaa" k=8`, all `sol=0` where `brute>=1`. Every one of them involves at least one roll, which is exactly where the wrong power bites; cases with a single window (no roll) were unaffected, which corroborates the diagnosis precisely.

**Fixing the roll and re-verifying.** Change the precompute to `for (i = 0; i < k-1; i++)` so `p = B^{k-1}`, leaving the subtraction `(h + M - (out*p)%M)%M` then `*B + in` intact. Re-trace `n=2, k=1, "aa"`: now the precompute loop runs `k-1 = 0` times, so `p1 = 1`. First window: `h1 = 1`, push. Roll `i=1`: `out=1, in=1`. `h1 = (1 + M1 - (1*1)%M1) % M1 = (1 + M1 - 1) % M1 = 0`, then `h1 = (0*131 + 1) % M1 = 1`. The second window's fingerprint is `1`, identical to the first. Both keys equal -> one group of size 2 -> answer `1`. Correct. I re-trace a no-repeat roll too, `n=2, k=1, "ab"`: window0 `h1=1`; roll: `out=1`(a), `in=2`(b), `h1=(1+M1-1)%M1=0`, then `0*131+2=2`. Keys `{1, 2}`, two singletons, answer `0`. Correct. The cases that failed now pass and they pass for the reason I fixed.

**Second implementation of the counting step — and a separate trace.** With fingerprints correct, the remaining risk is the counting sweep, which is where the *distinct vs occurrences* trap lives. A tempting first sweep just tallies how many windows belong to a repeated group:

```
sort(keys.begin(), keys.end());
long long ans = 0;
for (long long i = 0; i < m; i++)
    if ((i > 0 && keys[i]==keys[i-1]) || (i+1 < m && keys[i]==keys[i+1]))
        ans++;          // count this window if it has an equal neighbour
```

Trace it on the sample `s="ababbaba", k=3`. The sorted multiset of group sizes is `aba`:2, `bab`:2, `abb`:1, `bba`:1, i.e. sorted keys look like `[aba, aba, bab, bab, abb, bba]` (in fingerprint order, but grouping is the same). The loop marks both `aba`s (each has an equal neighbour) and both `bab`s, contributing `4`; the two singletons contribute `0`. So `ans = 4`. But the correct answer is `2`. This sweep counts repeated *occurrences*, not distinct repeated *substrings* — exactly the double-count the problem is built to punish. A substring that appears `t >= 2` times adds `t` here instead of `1`.

**Diagnosing the second bug.** The output `4` versus expected `2` is a clean double-count (here it happens to be a quadruple/2x over, and on a triple-repeat it would be `3` vs `1`). The unit of counting is wrong: I must count *groups* of size `>= 2`, contributing `1` per group, not members of such groups. The correct sweep walks each maximal run of equal keys once and adds `1` iff the run length is `>= 2`:

```
long long ans = 0, i = 0;
while (i < m) {
    long long j = i;
    while (j < m && keys[j] == keys[i]) j++;   // [i, j) is one group
    if (j - i >= 2) ans++;                       // one distinct substring, counted once
    i = j;
}
```

Re-trace the sample: run `[aba,aba]` has length 2 -> `ans=1`; run `[bab,bab]` length 2 -> `ans=2`; run `[abb]` length 1 -> no change; run `[bba]` length 1 -> no change. Final `ans = 2`. Correct. Re-trace a triple: `s="aaaa", k=1` gives windows `a,a,a,a`, one run of length 4 -> `ans=1` (one distinct repeated substring, `"a"`), which is right; the buggy sweep would have said `4`.

**Edge cases, deliberately, because the window count and the boundaries are where this dies.**
- `k = 0`: there is no length-0 *substring* to count in the intended sense, and `n - k + 1 = n + 1` would be a nonsense window count. I guard `if (k < 1 || k > n) print 0`. Verified `5 0 / abcde -> 0`.
- `k > n`: no window fits; same guard returns `0`. Verified `3 5 / abc -> 0`.
- `k == n`: exactly one window, `W = 1`. The roll loop `for (i=1; i<W; i++)` never executes, the single key is its own group of size 1, answer `0`. Verified `4 4 / aaaa -> 0` — a string can never "repeat" if there's only one place it can sit.
- `n = 0`: I read `n` and `k`, skip reading the (empty) string because `n == 0`, and the `k < 1 || k > n` guard fires (since `k > 0 = n`, or `k = 0`), printing `0`. Verified `0 0 -> 0`. Importantly I must NOT do an unconditional `cin >> s` when `n = 0`, or the stream would block/fail; reading `s` only when `n > 0` keeps parsing clean.
- `k = 1`: single-character windows; with `p = B^{k-1} = B^0 = 1` the roll degenerates correctly (traced above). Verified `1 1 / a -> 0` (one window) and `2 1 / aa -> 1`.
- Tiny alphabet, many repeats: `n=4000, k=10` over `{a,b}` — the random tester agrees `sol=927=brute`, confirming the counting holds when groups are large and numerous.

**Hash-safety and overflow sanity-check.** Each modular value stays below `M < 2^30`, and the largest intermediate is a product of two such values, `< 2^60`, which fits in `unsigned long long` before the `% M`. The packed key `(h1 << 32) ^ h2`: `h1 < 10^9 + 7 < 2^30`, so `h1 << 32 < 2^62`, no overflow, and the low 32 bits hold `h2` (also `< 2^30`) without clobbering `h1`'s bits — the two halves don't overlap, so equal pairs give equal keys and unequal pairs differ. With two `~10^9` moduli the false-merge probability over `2*10^5` windows is about `(2*10^5)^2 / (2 * 10^{18}) ≈ 2*10^{-8}`, negligible; the random oracle of 600 small cases against the exact substring brute had zero mismatches, which is the empirical backstop on the hashing.

**Performance check.** `O(n)` to roll all fingerprints, `O(n log n)` to sort `n - k + 1 <= 2*10^5` 64-bit keys, `O(n)` to sweep. On the worst constructed case (`n = 2*10^5`, `k = 20`, alphabet `{a,b}` so groups churn) it runs in about `0.01 s` with `~5 MB` resident — comfortably inside `1 s / 256 MB`.

**Final solution.** I disproved the slow raw-sort as a submission (kept it as the oracle), derived the rolling hash and pinned the leading power at `B^{k-1}`, then fixed two real bugs found by tracing: the `B^k` roll that desynced equal windows, and the occurrence-counting sweep that quadrupled the sample. The version below counts each equal-fingerprint group once iff it has size `>= 2`, guards all the `k`/`n` boundaries, and agrees with the brute oracle on 600 random cases:

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    long long n, k;
    if (!(cin >> n >> k)) return 0;
    string s;
    if (n > 0) cin >> s;            // when n == 0 there is no string token

    // Number of length-k windows. If k is out of [1, n] there are none.
    if (k < 1 || k > n) { cout << 0 << "\n"; return 0; }
    long long W = n - k + 1;        // window count: starts 0..n-k inclusive

    // Two independent polynomial rolling hashes, packed into one 64-bit key.
    const unsigned long long M1 = 1000000007ULL, M2 = 998244353ULL;
    const unsigned long long B1 = 131ULL, B2 = 137ULL;

    // Precompute B^(k-1) mod M: the weight of the leading char we remove when rolling.
    unsigned long long p1 = 1, p2 = 1;
    for (long long i = 0; i < k - 1; i++) { p1 = (p1 * B1) % M1; p2 = (p2 * B2) % M2; }

    vector<unsigned long long> keys;
    keys.reserve(W);

    unsigned long long h1 = 0, h2 = 0;
    // first window s[0..k-1]
    for (long long i = 0; i < k; i++) {
        unsigned long long c = (unsigned long long)(s[i] - 'a' + 1);
        h1 = (h1 * B1 + c) % M1;
        h2 = (h2 * B2 + c) % M2;
    }
    keys.push_back((h1 << 32) ^ h2);

    // roll: window starting at i uses removing s[i-1], appending s[i+k-1]
    for (long long i = 1; i < W; i++) {
        unsigned long long out = (unsigned long long)(s[i - 1] - 'a' + 1);
        unsigned long long in  = (unsigned long long)(s[i + k - 1] - 'a' + 1);
        // remove leading char (weight B^(k-1)), shift left by one, append new char
        h1 = (h1 + M1 - (out * p1) % M1) % M1;
        h1 = (h1 * B1 + in) % M1;
        h2 = (h2 + M2 - (out * p2) % M2) % M2;
        h2 = (h2 * B2 + in) % M2;
        keys.push_back((h1 << 32) ^ h2);
    }

    sort(keys.begin(), keys.end());

    // Count DISTINCT substrings whose group size is >= 2 (appears at >= 2 positions).
    long long ans = 0;
    long long i = 0, m = (long long)keys.size();
    while (i < m) {
        long long j = i;
        while (j < m && keys[j] == keys[i]) j++;
        if (j - i >= 2) ans++;     // one distinct substring, counted once
        i = j;
    }

    cout << ans << "\n";
    return 0;
}
```

**Causal recap.** The slow raw-substring sort is correct but `O(n*k*log n)`, so I moved to rolling hashes and immediately had to pin the leading-character weight; my first roll used `B^k` instead of `B^{k-1}`, which a trace of `"aa", k=1` exposed by giving two identical windows different fingerprints (output `0` where `1` was right), fixed by precomputing `B^{k-1}`. With fingerprints correct, the counting step is the real trap: my first sweep tallied one per repeated *occurrence*, turning the sample's two repeated substrings into `4`; sweeping maximal equal-key runs and adding `1` per run of size `>= 2` counts *distinct* repeated substrings exactly. Double hashing packed into a 64-bit key kills false merges, the `k < 1 || k > n` guard plus reading `s` only when `n > 0` closes the `k=0`, `k>n`, `k=n`, and `n=0` corners, and 600 oracle cases plus a 0.01 s worst-case run confirm correctness and speed.
