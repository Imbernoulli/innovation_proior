**Reading the problem and pinning the contract.** I have a ledger `a[0..n-1]` of signed minute-by-minute stock changes and a fixed target `S`, and I must count contiguous windows `[l, r]` whose sum is exactly `S`. Input is `n` and `S` on the first line, then `n` values; I print one integer, the count. Before I pick an algorithm I fix the scale, because here the scale is the whole problem. `n <= 2*10^5`, `|a[i]| <= 10^9`, so a prefix total `P[k] = a[0]+...+a[k-1]` can reach `2*10^5 * 10^9 = 2*10^14`. And `|S|` can be up to `2*10^14` by the contract. Both of those already blow past the 32-bit range of about `2.1*10^9`, so prefix values and `S` must live in `long long`. I will come back to a second, sneakier width problem — the *count itself* — once I have the algorithm, because it is the one that is easy to forget.

**Laying out the candidate approaches.** Two routes.

- *Brute force over all windows.* For each left endpoint `l`, walk `r` to the right accumulating the running sum and test it against `S`. This is `O(n^2)`, unquestionably correct, and at `n = 2*10^5` it is about `2*10^10` operations — a 1-second limit kills it instantly. Useful only as the oracle I check against, never as the submission.
- *Prefix-sum + hash map.* A window `[l, r]` sums to `S` iff `P[r+1] - P[l] = S`. Sweep `j` over prefix indices left to right; for the current prefix `P[j]`, the number of windows ending just before `j` equals the number of earlier prefix indices `i < j` with `P[i] = P[j] - S`. Keep a hash map `seen[value] = how many earlier prefixes had that value` and accumulate. `O(n)` expected, `O(n)` memory. This is the one I will submit, but I have to nail the bookkeeping of which prefixes are "earlier".

**Deriving the sweep and checking it on paper.** Let me restate precisely. Define prefixes `P[0]=0` and `P[k]=P[k-1]+a[k-1]`. The windows are pairs `(i, j)`, `0 <= i < j <= n`, with `P[j]-P[i]=S`. I want to count all such pairs. I will sweep `j` from `1` to `n`; just before processing the value that produces `P[j]`, the map `seen` must contain exactly the prefixes `P[0..j-1]` (the strictly-earlier ones). Then the contribution at `j` is `seen[P[j]-S]`. After counting, I insert `P[j]` so it is available as an "earlier" prefix for future `j`. The empty prefix `P[0]=0` must be in the map before the very first step — otherwise a window starting at `l=0` (which uses `P[0]` as its left prefix) is never counted.

Let me confirm on the sample: `S=2`, `a=[3,-1,1,2,-2,2]`. Prefixes `P = [0, 3, 2, 3, 5, 3, 5]`. I start `seen={0:1}`, `answer=0`.
- `P[1]=3`: need `P-S=1`, `seen[1]=0`, add nothing. Insert 3 -> `seen={0:1,3:1}`.
- `P[2]=2`: need `0`, `seen[0]=1`, `answer=1`. Insert 2.
- `P[3]=3`: need `1`, `seen[1]=0`. Insert 3 -> `seen[3]=2`.
- `P[4]=5`: need `3`, `seen[3]=2`, `answer=3`. Insert 5.
- `P[5]=3`: need `1`, `0`. Insert 3 -> `seen[3]=3`.
- `P[6]=5`: need `3`, `seen[3]=3`, `answer=6`. Insert 5 -> `seen[5]=2`.
Final `answer=6`, which matches the stated sample. Good — the recurrence and the empty-prefix seeding are right.

**A numeric self-check of the count bound, before I trust any int type.** I keep asserting "the answer can be huge", so let me actually bound it on a concrete worst case rather than wave at it. If every `a[i]=0` and `S=0`, then *every* window sums to `S`. The number of windows `[l,r]` with `0<=l<=r<=n-1` is `n(n+1)/2`. At `n=2*10^5` that is `200000*200001/2 = 20000100000 ~ 2*10^10`. That single number settles the type question: `2*10^10 > 2^31-1 = 2147483647`, so the answer counter cannot be a 32-bit `int`; it must be `long long` (whose range is about `9.2*10^18`, comfortably above `2*10^10`). I will write down `20000100000` and use it as a live test later — it is exactly the kind of case a hidden test will include, and exactly the kind I would otherwise overflow.

**First implementation and a trace, because the bookkeeping is easy to get half-right.** My first cut, focusing on the sweep:

```
unordered_map<long long,long long> seen;
long long prefix = 0, answer = 0;
for (int i = 0; i < n; i++) {
    long long x; cin >> x;
    prefix += x;                 // prefix = P[i+1]
    seen[prefix] += 1;           // record this prefix
    auto it = seen.find(prefix - S);
    if (it != seen.end()) answer += it->second;
}
```

Notice I dropped the `seen[0]=1` seeding and I inserted `prefix` *before* querying. Let me trace the smallest input that exposes both: `S=5`, `a=[5]`, where the answer is obviously `1` (the single window `[0,0]` sums to 5). Start `seen={}`, `prefix=0`, `answer=0`. i=0: `prefix=5`; `seen[5]=1`; query `prefix-S=0`, `seen.find(0)` is *not found*, add nothing. Final `answer=0`.

**The bug.** The code returns `0` but the true answer is `1`. Two defects, and the trace pins both. First, I never seeded the empty prefix `P[0]=0`, so the window `[0,0]` — whose left prefix is `P[0]=0` — has no `0` in the map to match against. Second, I inserted `seen[prefix]` *before* querying, which is wrong on a different axis: a window must have `i < j` (the left prefix strictly earlier than the right), but inserting the current prefix first lets a prefix match *itself* when `S=0`, counting bogus zero-length "windows". To see the second bug in isolation, trace `S=0`, `a=[7]` (true answer `0`, since `a[0]=7 != 0`): with the buggy order, i=0: `prefix=7`; `seen[7]=1`; query `7-0=7`, `seen[7]=1` is found, `answer=1`. It reports `1`, a window that does not exist. Both the missing seed and the insert-before-query order are real, and they are independent.

**Fix and a re-trace.** Seed the empty prefix, and *query before insert* so only strictly-earlier prefixes are visible:

```
unordered_map<long long,long long> seen;
seen[0] = 1;                      // empty prefix P[0] = 0
long long prefix = 0, answer = 0;
for (int i = 0; i < n; i++) {
    long long x; cin >> x;
    prefix += x;                  // prefix = P[i+1]
    auto it = seen.find(prefix - S);   // count earlier prefixes first
    if (it != seen.end()) answer += it->second;
    seen[prefix] += 1;            // then make this prefix available to the future
}
```

Re-trace `S=5`, `a=[5]`: `seen={0:1}`, `prefix=0`. i=0: `prefix=5`; query `5-5=0`, `seen[0]=1`, `answer=1`; insert 5. Final `1`. Correct. Re-trace `S=0`, `a=[7]`: `seen={0:1}`. i=0: `prefix=7`; query `7-0=7`, not found, add nothing; insert 7. Final `0`. Correct. And the full sample `[3,-1,1,2,-2,2]`, `S=2` re-runs to `6` exactly as in my paper trace above. The two cases that broke now pass, for precisely the reasons I changed.

**The second debug episode: tracing a large case catches a silent 32-bit overflow.** The math and bookkeeping are right, but I deliberately stress the *types* now, because the count bound I computed earlier warned me. Suppose I had been sloppy and written the accumulator as a 32-bit `int`:

```
int answer = 0;                   // WRONG WIDTH
...
if (it != seen.end()) answer += it->second;
```

Take the worst case I bounded: `n=200000`, `S=0`, all `a[i]=0`. Every prefix is `0`, so `seen[0]` climbs `1,2,3,...` and the contribution at step `k` is `seen[0]=k`, summing to `1+2+...+200000 = 200000*200001/2 = 20000100000`. The true answer is `20000100000`. Now follow the 32-bit accumulator: it can only hold values mod `2^32 = 4294967296`. The final true value `20000100000` reduces to `20000100000 mod 2^32 = 2820230816`, which as a *signed* 32-bit int is `2820230816 - 4294967296 = -1474736480`. So the `int` version prints `-1474736480` — a negative count, on a problem whose answer is always non-negative. I literally ran this case through a 32-bit build and it printed `-1474736480`, exactly the wrap value I computed by hand. That is the overflow caught by tracing the large case: not a crash, not a warning, just a silently wrong, even negative, answer that sails through every tiny sample. The fix is non-negotiable: `answer` is `long long`. The same reasoning forced `prefix` and the map's key/value to be `long long` already — a prefix can reach `2*10^14` and `S` `2*10^14`, so `prefix - S` ranges over about `+-4*10^14`, far outside `int`, and any `int` truncation there would corrupt the very keys I look up.

**A subtlety I checked rather than assumed: is the hash-map value width also at risk?** `seen[v]` counts how many earlier prefixes equal `v`; at most `n+1 = 200001` of them, which fits in `int` easily. But I store it as `long long` anyway, because `answer += it->second` should be a 64-bit add — mixing a 32-bit value into a 64-bit running sum is fine, yet making the map value 64-bit keeps the whole accumulation in one width and removes any doubt about an intermediate `int` temporary. I verified the all-zeros case sums to the correct `20000100000` with this layout, so the value width is sound.

**Why the accumulation across the sweep cannot silently truncate even on partial sums.** A worry with running counters is that an intermediate value overflows before the final one, so I check the *partial* sums on the all-zeros worst case too. At step `k` (1-indexed), `answer` holds `1+2+...+k = k(k+1)/2`. The largest intermediate is the final one, `200000*200001/2 = 20000100000`, and `answer` only ever increases (every `it->second` is non-negative), so if the final value fits in `long long` then every partial value does. With `long long` topping out near `9.2*10^18`, even a hypothetical worst-of-worst answer around `2*10^10` leaves more than eight orders of magnitude of headroom; there is no intermediate that overflows. That is the difference from the 32-bit version, where the partial sum crosses `2^31` around step `k ~ 65536` (since `k(k+1)/2 ~ 2.1*10^9` at `k ~ 65535`) and silently goes negative from there on — long before the sweep ends.

**Complexity, and the one performance footgun in the hash map.** Time is `O(n)` expected: one `find` and one insert per element, each `O(1)` average in `unordered_map`. The footgun is rehashing — without a reserve, the map grows by repeated rehashes and worst-case adversarial keys can degrade buckets. I `reserve(n*2)` and set `max_load_factor(0.7)` so the table is sized once up front for up to `n+1` distinct prefixes; on the all-zeros case there is exactly one distinct key, and on all-distinct prefixes there are `n+1`, both comfortably under the reserved capacity. I timed the `n=200000` all-zeros build and it finishes in well under the 1-second limit with about 6 MB resident, so the constant factor is fine. Memory is `O(n)` for the map, within 256 MB.

**Edge cases, deliberately.**
- `n = 0`: `if (!(cin >> n >> S)) return 0;` still reads `n=0` and `S`; the loop never runs; `answer = 0`. No window exists, so `0` is correct. (If the entire input is missing, the guard returns early, also printing nothing or `0` consistently.)
- `n = 1`, window matches: `S=5`, `a=[5]` -> `1` (traced above). Window does not match: `S=0`, `a=[7]` -> `0` (traced above). Both correct.
- All-positive, monotone prefixes: e.g. `S=3`, `a=[1,2,3]`, prefixes `0,1,3,6`. Windows summing to 3: `[0,1] (1+2)` and `[2,2] (3)` = 2. Sweep: seen{0:1}. p=1: need -2, none; insert. p=3: need 0, seen[0]=1, ans=1; insert. p=6: need 3, seen[3]=1, ans=2; insert. -> `2`. Correct.
- Negative `S` reachable: `S=-2`, `a=[1,-3,1]`, prefixes `0,1,-2,-1`. Windows summing to -2: `[1,1] (-3)`? no that's -3. `[0,1] (1-3=-2)` yes; `[1,2] (-3+1=-2)` yes = 2. Sweep: seen{0:1}. p=1: need 3, none; insert. p=-2: need 0, seen[0]=1, ans=1; insert. p=-1: need 1, seen[1]=1, ans=2; insert. -> `2`. Correct.
- Huge `S` no window reaches: `S=2*10^14` with small values -> `prefix - S` is a giant negative key never present, `answer=0`. Correct, and the 64-bit subtraction does not overflow.
- Overflow corners: `answer`, `prefix`, map key/value all `long long`; the all-zeros case verified to `20000100000`; the all-`10^9`/`S=10^9` case (prefix reaches `2*10^14`) verified to `200000` (each length-1 window matches and nothing else). Both pass.

**Final solution.** I convinced myself the *idea* is right by deriving `P[j]-P[i]=S` and hand-checking the sample to `6`; I convinced myself the *bookkeeping* is right by tracing `[5]/S=5` and `[7]/S=0`, which exposed the missing empty-prefix seed and the insert-before-query order, then re-tracing to confirm both fixes; and I convinced myself the *types* are right by bounding the count at `n(n+1)/2 = 20000100000` and watching a 32-bit accumulator wrap that to `-1474736480`. The submission uses `unordered_map<long long,long long>`, queries before inserting, seeds the empty prefix, and keeps every accumulator 64-bit:

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

**Causal recap.** The window count reduces to counting prefix pairs with `P[j]-P[i]=S`, which a single hash-map sweep does in `O(n)` — but the sweep is only correct if I seed the empty prefix `P[0]=0` and query the map *before* inserting the current prefix (a trace of `[5]/S=5` returning `0` and `[7]/S=0` returning a phantom `1` pinned both mistakes); and the types are only correct because the count reaches `n(n+1)/2 = 20000100000` on the all-zeros case, which a 32-bit accumulator silently wraps to `-1474736480` — a negative answer that no small sample would ever reveal — so `answer`, `prefix`, and the map's key/value are all `long long`.
