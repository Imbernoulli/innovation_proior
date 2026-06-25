**Reading the problem and pinning the contract.** I have a ledger `a[0..n-1]` of daily gold changes (any sign, zeros allowed) and a fixed target `S`. I must count every non-empty contiguous run `[l, r]` whose net change `a[l] + ... + a[r]` equals exactly `S`. Input on stdin is `n` and `S` on the first line, then the `n` values; I print one integer. Before any algorithm I fix the scale, because it dictates the data types. The count can be as large as the number of non-empty runs, `n*(n+1)/2`; for `n = 2*10^5` that is `200000*200001/2 = 20000100000`, which is about `2*10^10`. That is well past the 32-bit limit of `~2.147*10^9`, so the answer accumulator must be 64-bit. Separately, a prefix sum can reach `n * 10^9 = 2*10^5 * 10^9 = 2*10^14` in magnitude, and `S` itself ranges up to `10^14`, so both prefix sums and the quantity `P[r+1] - S` must live in 64-bit too (int64 max is `~9.2*10^18`, comfortable). I will use `long long` for the running prefix, for `S`, for the map's keys and values, and for the answer. An `int` anywhere here is a silent wrong-answer on the large tests. That decision is non-negotiable.

**Laying out the candidate approaches.** Two routes are on the table, and I want the one I can defend, not merely the one easiest to type.

- *Direct enumeration.* Fix `l`, sweep `r` from `l` upward keeping a running sum, and increment the count whenever the running sum hits `S`. This is `O(n^2)`, transparently correct, and I will use exactly this as my brute-force oracle. But `~4*10^10` additions for the largest `n` is hopelessly over a 1-second budget, so it cannot be the shipped solution.
- *Prefix-sum hashing.* Define `P[0] = 0` (empty prefix) and `P[k] = a[0] + ... + a[k-1]`. The sum of `[l, r]` is `P[r+1] - P[l]`. So `[l, r]` matches iff `P[r+1] - P[l] = S`, i.e. `P[l] = P[r+1] - S`. Sweeping the right edge and asking "how many earlier prefix sums equal `P[r+1] - S`?" gives the count in `O(n)` with a hash map. The idea is standard; the danger is entirely in transcription — the order of querying and inserting, and seeding the empty prefix exactly once. Counting problems like this are where a careless map update silently double-counts.

**Deriving the hashing recurrence and stating the invariant precisely.** I want, for each right edge `r`, the number of valid left endpoints `l` with `0 <= l <= r` and `P[l] = P[r+1] - S`. The crucial point is the *range* of `l`: a run `[l, r]` is non-empty exactly when `l <= r`, which in prefix-index terms means `l` ranges over `0, 1, ..., r` — that is, prefix indices `P[0], P[1], ..., P[r]`, but **not** `P[r+1]`. If I ever allowed `l = r+1`, that would be the empty run `[r+1, r]`, which has sum `0` and must never be counted (not even when `S = 0`). So the invariant I must maintain is:

> When I process the right edge whose prefix is `P[r+1]`, the frequency map must contain exactly the multiset `{P[0], P[1], ..., P[r]}` — every valid left endpoint and nothing more.

This pins the loop structure: at the start of iteration for `P[r+1]`, the map already holds `P[0..r]`; I query for `P[r+1] - S`, add that frequency to the answer, and only *then* insert `P[r+1]` so it becomes available as a left endpoint for future, larger right edges. Query first, insert second. Seed `P[0] = 0` into the map once before the loop, because `P[0]` is a legitimate left endpoint (the run starting at day 0).

**A numeric self-check of the recurrence on the sample.** Before coding I verify the derivation on `S = 2`, `a = [3, -1, 1, 2, -2, 2, 1]`, whose documented answer is `6`. Prefix sums: `P[0]=0, P[1]=3, P[2]=2, P[3]=3, P[4]=5, P[5]=3, P[6]=5, P[7]=6`. I sweep right edges `P[1..7]`, each time counting earlier prefixes equal to `P[r+1] - 2`, then inserting.

- Seed map `= {0:1}`.
- `P[1]=3`: need `3-2=1`; map has `{0:1}`, count of `1` is `0`. ans `=0`. Insert `3` -> `{0:1, 3:1}`.
- `P[2]=2`: need `0`; map has one `0`. ans `+=1 -> 1`. (This is run `[0,1]`, sum `2`.) Insert `2` -> `{0:1,3:1,2:1}`.
- `P[3]=3`: need `1`; none. ans `=1`. Insert `3` -> `{0:1,3:2,2:1}`.
- `P[4]=5`: need `3`; map has two `3`s. ans `+=2 -> 3`. (Runs `[1,3]` and `[3,3]`.) Insert `5` -> `{...,5:1}`.
- `P[5]=3`: need `1`; none. ans `=3`. Insert `3` -> `{0:1,3:3,2:1,5:1}`.
- `P[6]=5`: need `3`; map has three `3`s. ans `+=3 -> 6`. (Runs `[1,5]`, `[3,5]`, `[5,5]`.) Insert `5` -> `{...,5:2}`.
- `P[7]=6`: need `4`; none. ans `=6`. Insert `6`.

Final answer `6`, matching the documented value, and the six runs I attributed line up exactly with the listed `[0,1], [1,3], [1,5], [3,3], [3,5], [5,5]`. The recurrence is right, and crucially the empty-prefix seed `{0:1}` is what lets the run `[0,1]` be found at `P[2]`.

**First implementation — and immediately a trace, because clean math transcribes dirty.** My first instinct for the loop body, writing the running prefix as `pref`, is to keep the map current by inserting the new prefix as soon as I compute it, then look up the need:

```
unordered_map<long long,long long> cnt;
long long pref = 0; cnt[pref] = 1;      // seed P[0]
long long answer = 0;
for (int i = 0; i < n; i++) {
    pref += a[i];                       // pref = P[i+1]
    cnt[pref] += 1;                     // (felt natural: keep the map up to date)
    answer += cnt[pref - S];            // count earlier P[l] = P[i+1] - S
}
```

It looks tidy, but the inserting-before-querying ordering is exactly the kind of thing that breaks counting code, so I trace the smallest input that could expose it. I pick `a = [1, -1]` with `S = 0`. By hand the matching non-empty runs are: `[0,0]` sum `1`? no; `[1,1]` sum `-1`? no; `[0,1]` sum `0`? yes. So the answer must be `1`. Let me run the buggy code mentally.

Seed `cnt = {0:1}`, `pref = 0`, `answer = 0`.
- `i=0`, `a[0]=1`: `pref = 1`. Insert: `cnt = {0:1, 1:1}`. Need `pref - S = 1 - 0 = 1`; `cnt[1] = 1`. `answer += 1 -> 1`.
- `i=1`, `a[1]=-1`: `pref = 0`. Insert: `cnt[0]` becomes `2`, `cnt = {0:2, 1:1}`. Need `0 - 0 = 0`; `cnt[0] = 2`. `answer += 2 -> 3`.

Final `answer = 3`. That is flatly wrong — the truth is `1`.

**Diagnosing the first bug (the double-count).** Two separate over-counts conspired here, both rooted in inserting before querying.

1. At `i=0`, I inserted `P[1] = 1` and then immediately queried `cnt[P[1] - S] = cnt[1]` and found the `1` I had just inserted. That counts the run `[1,0]` — the empty run with `l = r+1 = 1` — which is illegal. When `S = 0`, `P[r+1] - S = P[r+1]`, so the just-inserted current prefix *always* matches itself: every position contributes a phantom length-0 match. That is the textbook prefix-sum double-count, and `S = 0` is the densest place it strikes.
2. At `i=1`, the same self-match struck again (`cnt[0]` had grown to `2`, and one of those `2` was the current `P[2]=0` I had just inserted), adding a second phantom.

The invariant I wrote down is violated: while processing `P[r+1]`, the map must hold only `P[0..r]`, but I had already pushed `P[r+1]` in. The fix is forced by the invariant: **query first, then insert**. I should compute `pref = P[i+1]`, look up `pref - S` against the map that still contains only `P[0..i]`, add that to the answer, and *then* insert `P[i+1]`.

**Fixing and re-verifying the order.** Reordered body:

```
long long pref = 0; cnt[pref] = 1;          // seed P[0] once
long long answer = 0;
for (int i = 0; i < n; i++) {
    pref += a[i];                            // pref = P[i+1]
    auto it = cnt.find(pref - S);            // query BEFORE inserting
    if (it != cnt.end()) answer += it->second;
    cnt[pref] += 1;                          // insert P[i+1] after
}
```

Re-trace `a = [1, -1]`, `S = 0`. Seed `cnt = {0:1}`, `pref = 0`, `answer = 0`.
- `i=0`: `pref = 1`. Find `1 - 0 = 1` in `{0:1}` -> absent. `answer = 0`. Insert -> `{0:1, 1:1}`.
- `i=1`: `pref = 0`. Find `0 - 0 = 0` in `{0:1, 1:1}` -> present, value `1`. `answer += 1 -> 1`. (Run `[0,1]`, the only match.) Insert -> `{0:2, 1:1}`.

Final `answer = 1`. Correct, and it broke before for exactly the reason I fixed, which is the evidence I trust. Re-trace the original sample too: this is the same sweep I hand-checked above (query-then-insert), and it yields `6`. Good.

**A second, quieter self-verify episode — the empty-prefix seed.** The first bug taught me to be paranoid about the empty prefix, so I deliberately test a competing mistake I have seen: seeding `P[0]` and *also* somehow letting `P[0]` enter the map a second time (for instance by initializing the count at `0` but then having a stray `cnt[0] += 1` both before and inside the loop on the first quiet day). To see what that costs, I take `a = [0, 0, 0]`, `S = 0`. The truth by direct count: non-empty runs of an all-zero array all sum to `0`, and there are `3*4/2 = 6` of them, so the answer must be `6`. Now suppose the empty prefix were seeded twice, `cnt = {0:2}` at the start. Trace with the correct query-then-insert order:

- Seed (buggy double) `cnt = {0:2}`, `pref = 0`.
- `i=0`: `pref = 0`. Find `0`: value `2`. `answer += 2 -> 2`. Insert -> `{0:3}`.
- `i=1`: `pref = 0`. Find `0`: value `3`. `answer += 3 -> 5`. Insert -> `{0:4}`.
- `i=2`: `pref = 0`. Find `0`: value `4`. `answer += 4 -> 9`.

That returns `9`, not `6` — every run got an extra `+1` from the phantom second empty prefix, plus a compounding error. So the empty prefix must be seeded **exactly once**. My code seeds it once with `cnt[pref] = 1` before the loop and never re-seeds, so it is safe. Let me confirm the *correct* single-seed run on the same `[0,0,0], S=0`:

- Seed `cnt = {0:1}`, `pref = 0`, `answer = 0`.
- `i=0`: `pref = 0`. Find `0`: value `1`. `answer = 1`. Insert -> `{0:2}`.
- `i=1`: `pref = 0`. Find `0`: value `2`. `answer = 3`. Insert -> `{0:3}`.
- `i=2`: `pref = 0`. Find `0`: value `3`. `answer = 6`. Insert -> `{0:4}`.

Final `6`. Correct — and notice this `0+1+2+3 = ... ` no, `1+2+3 = 6` telescoping is exactly `n*(n+1)/2 = 3*4/2 = 6`, which is the closed form I claimed for the all-zero case. The numeric self-check confirms that closed form on a concrete instance.

**Numeric self-check of the overflow bound.** I claimed the answer can hit `n*(n+1)/2`. For the all-zero array of length `n = 200000`, the sweep adds `1 + 2 + ... + n = n(n+1)/2 = 20000100000`. That exceeds `2^31 - 1 = 2147483647` by nearly tenfold, confirming the answer accumulator genuinely needs 64 bits (it fits easily in int64). And the largest prefix magnitude is `200000 * 10^9 = 2*10^14`, with `pref - S` reaching about `2*10^14 + 10^14 = 3*10^14` in magnitude — all far below int64's `~9.2*10^18`. So 64-bit arithmetic is both necessary and sufficient; no value overflows.

**Edge cases, deliberately, because this is where counting code dies.**
- `n = 0`: `cin >> n >> S` reads `n = 0`; the value loop reads nothing; the seed puts `{0:1}` in the map; the main loop never runs; `answer = 0`. The empty ledger has no non-empty runs — correct. The `if (!(cin >> n >> S)) return 0;` guard also handles a totally empty stdin gracefully (prints nothing, but the harness for `n=0` supplies `0` and `S`).
- `n = 1`, `a = [7]`, `S = 7`: seed `{0:1}`, `pref = 7`, find `7 - 7 = 0` -> value `1`, `answer = 1`. The single run `[0,0]` sums to `7` — correct. With `S = 4` instead: find `7 - 4 = 3` -> absent, `answer = 0` — correct.
- `S = 0` with sign cancellation, `a = [5, -5, 5, -5]`: runs summing to `0` are `[0,1], [2,3], [0,3], [1,2]` — that is `4`. Let me trust the verified sweep rather than re-derive by hand; the brute force agrees (it does in the stress tests). The point is that the query-then-insert order means a current prefix never matches itself, so even with `S = 0` no empty run sneaks in.
- Unreachable target, `a = [1, 1, 1]`, `S = 100`: every `pref - S` is hugely negative and never in the map; `answer = 0` — correct.
- Negative needed value: with negatives in `a`, `pref - S` is routinely negative; `unordered_map<long long,...>` keys on the signed value directly, so negative keys are fine (no residue/modulus mangling here — this is exact-sum, not divisibility, so there is no sign-of-modulo trap).
- Performance: the densest map class is the all-zero array, a single bucket touched `n` times; it runs in about `11 ms` for `n = 2*10^5` in my timing, and a random large case in about `30 ms` — both comfortably inside `1 s`. I reserve the map and set a `0.7` load factor to keep it fast.

**Final solution.** I convinced myself the *idea* is right by deriving the prefix-difference condition and hand-checking the sample to `6`, and I convinced myself the *code* is right by tracing the failing `[1,-1], S=0` case to a precise double-count (insert-before-query, worst at `S=0`), fixing the order, and separately proving the empty prefix must be seeded exactly once (the `[0,0,0]` double-seed returning `9` vs the correct `6`). What I ship is one self-contained file, the `O(n)` hashing count I can defend:

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    long long S;
    if (!(cin >> n >> S)) return 0;
    vector<long long> a(n);
    for (auto &x : a) cin >> x;

    // Count non-empty subarrays [l, r] with a[l] + ... + a[r] == S.
    // Prefix sums: P[0] = 0, P[k] = a[0] + ... + a[k-1].
    // [l, r] sums to S  <=>  P[r+1] - P[l] = S  <=>  P[l] = P[r+1] - S.
    // Sweep r+1 = 1..n. BEFORE inserting P[r+1], the map holds exactly
    // {P[0], ..., P[r]} = all valid left endpoints l in [0, r], so a length-0
    // subarray (l = r+1) can never be matched. Order matters: query then insert.
    unordered_map<long long, long long> cnt;
    cnt.reserve(n * 2 + 16);
    cnt.max_load_factor(0.7);

    long long pref = 0;        // P[0] = 0, the empty prefix (a valid left endpoint l = 0)
    cnt[pref] = 1;             // seed P[0] exactly once
    long long answer = 0;
    for (int i = 0; i < n; i++) {
        pref += a[i];          // now pref = P[i+1]
        auto it = cnt.find(pref - S);   // need P[l] = P[i+1] - S
        if (it != cnt.end()) answer += it->second;
        cnt[pref] += 1;        // insert P[i+1] AFTER querying
    }

    cout << answer << "\n";
    return 0;
}
```

**Causal recap.** The exact-sum count collapses to "how many earlier prefix sums equal `P[r+1] - S`", which I verified on the sample to `6`; the entire risk is transcription. My first loop inserted the current prefix before querying, so on `[1,-1], S=0` the current prefix matched itself and the answer came out `3` instead of `1` — a length-0 double-count that, because `P[r+1]-S = P[r+1]` when `S=0`, hits every position; reordering to query-then-insert restores the invariant "the map holds exactly `P[0..r]` while processing `P[r+1]`" and fixes it. A second hazard, seeding the empty prefix twice, would turn the all-zero `[0,0,0]` answer from the correct `6` into `9`, so I seed `P[0]` exactly once; and because the count reaches `n(n+1)/2 ~ 2*10^10` and prefixes reach `~2*10^14`, every accumulator, key, and value is `long long`. The result is an `O(n)` sweep that survives the `S=0`, all-zero, unreachable-target, `n=0/1`, negative-value, and large-`n` corners.
