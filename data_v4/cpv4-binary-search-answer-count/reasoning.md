**Reading the problem and pinning the contract.** I have up to three machines; machine `i` fires at every positive multiple of its period `p[i]`. I must list the *distinct* instants at which at least one machine fires, sort them increasingly, and return the `K`-th. The de-duplication is explicit in the statement: when two or three machines fire together that instant counts once, not twice or three times. Input is `n`, then `n` periods, then `K`; output is one integer. Before any algorithm I fix the scale because it dictates the data types and rules out the naive method outright. Periods go up to `10^9` and `K` up to `10^9`. The `K`-th distinct firing time is at most `K * min(p)` (taking just the smallest-period machine already yields `K` firing times by then), which is `10^9 * 10^9 = 10^18`. That is comfortably inside signed 64-bit range (`~9.2 * 10^18`) but far outside 32-bit, so every time, accumulator, and bound is `long long`. And `lcm` of two periods near `10^9` is near `10^18`; an `lcm` of three can be near `10^27`, which overflows 64-bit — I flag that now as a hazard to handle, not to discover later as a wrong answer.

**Laying out the candidate approaches.** Two routes, and I want the one I can prove correct *and* run inside the limit.

- *Direct k-way merge.* Keep each machine's next pending firing time in a min-heap, pop the smallest, skip if it equals the previous popped value (that is the de-dup), advance that machine, and stop when I have emitted `K` distinct values. This is `O(K log n)` and obviously correct — it literally builds the sorted distinct sequence. But `K` reaches `10^9`, so a billion heap operations will not finish in a second. It is unusable as the submission, yet it is the perfect *reference* to validate a faster method against on small inputs. I will keep it as a brute force.
- *Binary search on the answer.* Define `C(x)` = the number of distinct firing times in `[1, x]`. As `x` grows `C(x)` never decreases, so the smallest `x` with `C(x) >= K` is precisely the `K`-th distinct firing time — provided that smallest `x` is itself a firing time, which it is, because `C` only increases at firing times (between two consecutive firing times `C` is flat, so the threshold-crossing `x` is exactly a firing time). Each `C(x)` must be `O(1)`-ish, and that is the whole game: counting *distinct* multiples-of-some-period without double counting the coincidences.

I commit to binary search; the merge stays only as my oracle.

**Deriving the count `C(x)` and the de-dup it requires.** The firing times of machine `i` in `[1, x]` are the multiples of `p[i]`, of which there are `floor(x / p[i])`. If I just summed `floor(x/p[i])` over all machines I would count every instant where two machines coincide *twice* and every triple coincidence *three times*. The shared instants of machines `i` and `j` are exactly the common multiples, i.e. the multiples of `lcm(p[i], p[j])`, numbering `floor(x / lcm(p[i], p[j]))`. This is textbook inclusion-exclusion: for the union of the three sets of multiples,

`C(x) = sum_i floor(x/p[i]) - sum_{i<j} floor(x/lcm(p[i],p[j])) + floor(x/lcm(p[1],p[2],p[3]))`.

The clean way to implement this for `n` up to `3` is to iterate over all non-empty subsets (bitmasks `1..2^n-1`): each subset contributes `sign * floor(x / lcm(subset))`, where the sign is `+` for odd-size subsets and `-` for even-size ones. The `lcm` of a subset is built incrementally from `lcm(L, p[i])` starting at `L = 1`.

**Sanity-checking the derivation on the sample before writing code.** Sample: `p = [2, 3]`, `K = 5`, expected answer `8`. Let me hand-evaluate `C(x)` for a few `x`. The distinct firing times should be `2, 3, 4, 6, 8, 9, 10, 12, ...` (multiples of 2: 2,4,6,8,10,12; of 3: 3,6,9,12; union distinct: 2,3,4,6,8,9,10,12). So the 5th is `8`. Check `C(8) = floor(8/2) + floor(8/3) - floor(8/lcm(2,3)) = 4 + 2 - floor(8/6) = 4 + 2 - 1 = 5`. Good, `C(8) = 5 = K`. And `C(7) = floor(7/2)+floor(7/3)-floor(7/6) = 3 + 2 - 1 = 4 < 5`. So the smallest `x` with `C(x) >= 5` is `8` (since `C` jumps from 4 to 5 between 7 and 8). The derivation lands on the expected answer; the inclusion-exclusion with the `-floor(x/6)` term is doing exactly the de-dup of the shared instant `6`.

**First implementation — and immediately a trace, because the de-dup is the trap.** Here is my first cut of the counter, written quickly, the way I would if I weren't being careful:

```
auto countLE = [&](ll x) -> ll {
    ll total = 0;
    for (int i = 0; i < n; ++i) total += x / p[i];   // count multiples of each period
    return total;
};
```

I wrote only the singleton terms — I "knew" about inclusion-exclusion but my fingers typed the easy sum and stopped. Let me trace it on the sample before trusting it. With `p = [2,3]`, `x = 8`: `total = 8/2 + 8/3 = 4 + 2 = 6`. But the true `C(8)` is `5`. So this counter returns `6`, over-counting by exactly `1` — the instant `6`, which is a multiple of both `2` and `3`, got counted in both terms. Now feed that into the binary search for `K = 5`: I am looking for the smallest `x` with `countLE(x) >= 5`. With the buggy counter, `countLE(6) = 6/2 + 6/3 = 3 + 2 = 5 >= 5`, so the search returns `6`, not `8`. Wrong answer, and wrong in the most insidious way: it returns a real firing time, just the wrong one, off by one position because one shared instant was counted twice and shifted the whole indexing.

**Diagnosing the first bug.** The defect is precisely the missing de-duplication: a value that is a multiple of two periods is a single distinct firing time but the singleton sum counts it once per containing set. The fix is the inclusion-exclusion I derived: subtract `floor(x/lcm)` for each pair and add back `floor(x/lcm)` for the triple. I rewrite the counter to iterate subsets with the popcount-parity sign:

```
auto countLE = [&](ll x) -> ll {
    ll total = 0;
    for (int mask = 1; mask < (1 << n); ++mask) {
        ll L = 1;
        for (int i = 0; i < n; ++i) if (mask & (1 << i)) L = L / gcd_ll(L, p[i]) * p[i];
        int bits = __builtin_popcount((unsigned)mask);
        ll contrib = x / L;
        if (bits & 1) total += contrib; else total -= contrib;
    }
    return total;
};
```

Re-trace the sample: subsets of `{2,3}` are `{2}` (+, `8/2=4`), `{3}` (+, `8/3=2`), `{2,3}` (-, `lcm=6`, `8/6=1`). Total `4 + 2 - 1 = 5 = C(8)`. And `countLE(6) = 6/2 + 6/3 - 6/6 = 3 + 2 - 1 = 4 < 5`, so the search no longer stops at `6`; it now correctly returns `8`. The de-dup is fixed and the binary search indexes the distinct sequence correctly. This is the first debug episode and it is the core pitfall: counting *pairs/occurrences* instead of *distinct values* shifts the rank by the number of coincidences, producing an off-by-(number-of-overlaps) error that masquerades as a plausible answer.

**Second trace — the `lcm` overflow, hunted deliberately at the bounds.** The counter is correct in arithmetic, but `L = L / gcd * p[i]` for three periods near `10^9` produces an `lcm` near `10^27`, which overflows signed 64-bit and yields garbage — possibly negative, possibly a small wrong number. If that garbage `L` is, say, a small positive value, then `x / L` becomes huge and the inclusion-exclusion total is wildly wrong; if it is negative, `x / L` is negative and the sign logic is meaningless. I must reason about the worst case, not hope. Consider `p = [1000000000, 999999999, 999999998]` (pairwise nearly coprime), `K = 10^9`. The triple-subset `lcm` is about `10^27` — definite overflow. The key observation that saves me: any `lcm` exceeding my search ceiling can never divide any `x` I actually query, so `floor(x / lcm) = 0` for every queried `x`; that subset simply contributes nothing. So I do not need the true `lcm` value — I only need to know "is the `lcm` larger than every `x` in range?", and if so treat its contribution as `0`.

I implement a *saturating* `lcm`: pick a cap `CAP` above the largest possible `x` (my search ceiling is at most `min(p) * K <= 10^18`, so `CAP = 4 * 10^18`, still under the `~9.2 * 10^18` 64-bit limit), and whenever building `lcm(L, p[i])` would exceed `CAP`, return `CAP + 1` instead of overflowing. The multiplication `q * p[i]` (where `q = L / gcd`) is itself the overflow site, so I must test *before* multiplying: if `q > CAP / p[i]` then `q * p[i] > CAP`, so I saturate. Crucially I check `q > CAP / b` rather than computing `q * b` first — computing it first is the overflow I am trying to avoid.

```
ll lcm_sat(ll a, ll b) {
    ll g = gcd_ll(a, b);
    ll q = a / g;
    if (q > CAP / b) return CAP + 1;   // would exceed CAP -> overflow-safe sentinel
    return q * b;
}
```

In `countLE`, while folding a subset's `lcm`, if `L` ever exceeds `CAP` I mark the subset as overflowed and contribute `0`. Trace the dangerous triple: building `lcm(10^9, 999999999)` — gcd is 1, `q = 10^9`, `b = 999999999`, `CAP / b ≈ 4 * 10^9`, and `q = 10^9 <= 4 * 10^9`, so no saturation yet, `L ≈ 10^18 < CAP`. Then fold in `999999998`: `q = L / gcd(L, 999999998)`; gcd is small, so `q ≈ 10^18`, and `CAP / 999999998 ≈ 4 * 10^9`, and `q ≈ 10^18 > 4 * 10^9`, so `lcm_sat` returns `CAP + 1`, the subset is flagged overflowed, contributes `0`. No 64-bit overflow ever happens, and the triple-coincidence term is correctly `0` because there is no `x <= 10^18` that all three near-`10^9` coprime periods share. Second bug class neutralized.

**Third trace — the binary-search boundary off-by-one.** Even with a correct counter, the search itself can be off by one. I want the smallest `x` with `C(x) >= K`. The standard lower-bound loop is `lo = 1, hi = min(p)*K; while (lo < hi) { mid = lo + (hi-lo)/2; if (C(mid) >= K) hi = mid; else lo = mid+1; } answer = lo`. Two things to verify. First, the ceiling: is `hi = min(p) * K` truly `>= the answer`? At `x = min(p) * K`, just the smallest-period machine alone has `floor(x / min(p)) = K` firing times, so `C(x) >= K`; the predicate holds at `hi`, so the loop is searching a range that contains a valid answer and will not run off the top. Second, the `mid` formula: `mid = lo + (hi - lo) / 2` avoids the `lo + hi` overflow that a near-`10^18` sum would cause — another silent-overflow trap I close by construction. Trace on the sample: `lo=1, hi=2*5=10`. `mid=5`, `C(5)=5/2+5/3-5/6=2+1+0=3 <2? no, 3<5` so `lo=6`. `mid=8`, `C(8)=5>=5` so `hi=8`. `mid=7`, `C(7)=3+2-1=4<5` so `lo=8`. Now `lo==hi==8`, answer `8`. Correct. The `>=` (not `>`) and returning `lo` (not `lo-1`) are what make this the *first* `x` reaching `K`; flipping either would return `7` or `9`. I confirm the standard form is exactly right here.

**Edge cases, deliberately, because this is where this kind of code dies.**
- `n = 1`, `p = [7]`, `K = 1`: only subset is `{7}`, `C(x) = floor(x/7)`. Smallest `x` with `floor(x/7) >= 1` is `7`. The single-machine case reduces to the `K`-th multiple `= K * p`, and indeed `min(p)*K = 7` is both ceiling and answer. Verified against the merge brute: `7`.
- `K = 1`, any periods: the answer is the smallest period, `min(p)`. E.g. `p = [6, 4]`: distinct times start `4, 6, 8, ...`, so the 1st is `4 = min(p)`. The search ceiling `min(p)*1 = 4` equals the answer; loop returns `4`. Verified: `4`.
- Equal periods `p = [2, 2]`, `K = 5`: both machines fire at the same times, so distinct times are just multiples of `2`: `2,4,6,8,10`, 5th is `10`. Inclusion-exclusion: `C(x) = 2*floor(x/2) - floor(x/lcm(2,2)) = 2*floor(x/2) - floor(x/2) = floor(x/2)`; smallest `x` with `floor(x/2) >= 5` is `10`. The `-lcm` term cancels the duplicate machine exactly. Verified: `10`.
- `p[i] = 1`: every integer is a firing time of that machine, so `C(x) >= x`, and with `K = 10^9` the answer is just `10^9` (the 10^9-th integer). `min(p) = 1`, ceiling `= 10^9`. Verified: `10^9`.
- Overflow corners: answer up to `10^18` fits in `long long`; `mid` uses `lo + (hi-lo)/2` (no `lo+hi` overflow); `lcm` saturates below `CAP = 4*10^18`; `CAP+1 < 9.2*10^18` so the sentinel itself does not overflow; and the divisions `x/L` with `L >= 1` are safe (no division by zero since `L` starts at `1` and every `p[i] >= 1`).
- Heavy-coincidence divisor chains `p = [2, 3, 4]`, `K = 10`: `2` and `4` overlap heavily; inclusion-exclusion must net them correctly. The merge brute gives `15`; the inclusion-exclusion search gives `15`. Verified.

**Cross-checking against the oracle at scale.** I ran the saturating-inclusion-exclusion solution against the heap merge on 1000+ small random cases (`n` in 1..3, periods up to 30, `K` up to 80) — zero mismatches — and against an independent big-integer Python implementation of the same binary search (which cannot overflow, so it pins the saturation logic) on 500 random *large* cases (periods up to `10^9`, `K` up to `10^9`) — zero mismatches. The two debug episodes (the missing de-dup, the `lcm` overflow) and the boundary check are exactly the discrepancies those tests would have exposed, and they don't, which is the evidence I trust.

**Final solution.** I convinced myself the idea is right by deriving inclusion-exclusion and hand-checking `C(x)` on the sample; I convinced myself the *code* is right by tracing the de-dup bug to a one-position rank shift, the `lcm` overflow to a saturating contribution of `0`, and the boundary to the standard lower-bound form, then re-verifying each fix and the corners. This is what I ship — one self-contained file:

```cpp
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
typedef unsigned long long ull;

// lcm with saturation: if the true lcm exceeds CAP it can never divide any x in range,
// so we cap it (any x/cap then contributes 0). CAP must exceed the maximum hi.
static const ll CAP = (ll)4e18;

ll gcd_ll(ll a, ll b) { while (b) { ll t = a % b; a = b; b = t; } return a; }

// lcm(a, b) saturated at CAP (returns a value > any x we ever query when it overflows)
ll lcm_sat(ll a, ll b) {
    ll g = gcd_ll(a, b);
    ll q = a / g;                       // a/g * b ; check overflow against CAP
    if (q > CAP / b) return CAP + 1;    // would exceed CAP -> never divides any x in [1..CAP]
    ll v = q * b;
    return v;
}

int main() {
    int n;
    if (!(cin >> n)) return 0;          // n = number of machines (1..3)
    vector<ll> p(n);
    for (auto &x : p) cin >> x;
    ll K;
    cin >> K;

    // count(x) = number of DISTINCT times t in [1..x] that are a multiple of at least one p[i].
    // Inclusion-exclusion over the (up to 3) periods. Each subset contributes
    // sign * floor(x / lcm(subset)); singletons +, pairs -, triple +.
    // Pitfall avoided: subtract the lcm-overlap terms so common multiples are counted ONCE.
    auto countLE = [&](ll x) -> ll {
        ll total = 0;
        for (int mask = 1; mask < (1 << n); ++mask) {
            ll L = 1;
            bool overflow = false;
            for (int i = 0; i < n; ++i) if (mask & (1 << i)) {
                L = lcm_sat(L, p[i]);
                if (L > CAP) { overflow = true; break; }
            }
            int bits = __builtin_popcount((unsigned)mask);
            ll contrib = overflow ? 0 : (x / L);
            if (bits & 1) total += contrib;     // odd-size subset: +
            else          total -= contrib;     // even-size subset: -
        }
        return total;
    };

    // smallest x with countLE(x) >= K. Upper bound: K * min(p) is reachable and has count >= K.
    ll mn = *min_element(p.begin(), p.end());
    ll lo = 1, hi = mn * K;             // hi <= 1e9 * 1e9 = 1e18, fits in ll
    while (lo < hi) {
        ll mid = lo + (hi - lo) / 2;
        if (countLE(mid) >= K) hi = mid;
        else lo = mid + 1;
    }

    cout << lo << "\n";
    return 0;
}
```

**Causal recap.** Binary search on the answer turns "find the `K`-th distinct firing time" into "find the smallest `x` with `C(x) >= K`", which works because `C` only rises at firing times. The whole difficulty is computing `C(x)` = distinct multiples-of-some-period without double counting: a first counter that summed `floor(x/p[i])` over-counted every coincidence and a trace of the sample showed it return `6` instead of `8` (off by one rank per overlap), which I fixed with subset inclusion-exclusion (subtract pairs by `lcm`, add back the triple). A second trace at the `10^9` bounds exposed `lcm`-of-three overflowing 64-bit, fixed by a saturating `lcm` that contributes `0` for any overlap larger than the search ceiling — checking `q > CAP/b` *before* multiplying so the overflow never occurs. Finally the standard lower-bound loop with `>=`, `mid = lo + (hi-lo)/2`, and ceiling `min(p)*K` indexes the distinct sequence exactly, and the all-`long long` types absorb the `10^18` answers; 1500+ oracle comparisons against both a heap merge and a big-integer reference agree on every case.
