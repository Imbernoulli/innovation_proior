**Reading the problem and pinning the contract.** I am given `n` positive integers `a[0..n-1]` and must sum, over every out-of-order pair `(i, j)` with `i < j` and `a[i] > a[j]`, the product `a[i] * a[j]`. Equal values do not count — the comparison is strict. The empty selection of pairs is fine; a non-decreasing sequence yields `0`. Input on stdin is `n` then the `n` values; I print one integer. Before any algorithm I fix the scale, because it dictates the data types and the asymptotics. Constraints are `n <= 10^5` and `1 <= a[i] <= 30000`. Two numbers jump out. First, the number of pairs is `C(n, 2) ~ 5*10^9`, so the naive double loop is dead on arrival under a 1-second limit; I need `O(n log n)`. Second, the arithmetic blows up: a single product is up to `30000 * 29999 ~ 9*10^8`, already most of the way to the 32-bit signed ceiling of `2.1*10^9`, and the *total* over all inversions can reach about `C(10^5, 2) * 30000^2 ~ 4.5*10^18`. That total fits in a signed 64-bit `long long` (ceiling `~9.2*10^18`) but is nowhere near fitting in `int`. So the final accumulator must be `long long`, and — I will come back to this — anything that accumulates value sums internally must be `long long` too. An `int` anywhere on the hot path is a silent wrong-answer on the large tests, not a crash.

**Laying out the candidate approaches.** Two routes can hit `O(n log n)`, and I want the one whose bookkeeping I can verify, not merely the one that compiles.

- *Merge-sort accounting.* The textbook inversion count sorts via merge sort and, each time an element from the right half is emitted before remaining left-half elements, adds the number of those remaining left elements. I could instead add the *sum of products*: when a right element `y` is placed and there are some left elements `x_1, ..., x_k` still pending (all greater than `y`, since both halves are sorted and they outranked `y`), the contribution is `y * (x_1 + ... + x_k)`. That is computable in bulk if I track the suffix sum of the pending left block. It works, but the index/position bookkeeping during merge is fiddly and easy to get subtly wrong, and I have to be careful that "pending left elements" really are the ones with original index less than `y`'s — which holds because the left half is entirely to the left of the right half.
- *Fenwick sweep over compressed values.* Sweep `j` from left to right. Before processing `j`, the tree already contains all earlier elements. I keep, indexed by value, the **sum of values** of inserted elements. Then the total value of earlier elements strictly greater than `a[j]` is one suffix query, and multiplying by `a[j]` gives *all* of `j`'s inversion contributions at once: `a[j] * (sum of earlier values > a[j])`. Summing over `j` gives the answer. This is `O(n log n)` with a clean, local invariant.

The Fenwick sweep has the cleaner invariant — "the tree holds every element strictly to the left of the current one" is trivially maintained by inserting after querying — so I commit to it. The merge approach would also work, but its correctness rides on getting merge boundaries exactly right, whereas the sweep's correctness rides on one clearly stated prefix-query identity.

**Deriving the sweep and checking the identity on paper.** Fix `j`. The inversions whose *larger* index is `j` are exactly the earlier positions `i < j` with `a[i] > a[j]`. Each contributes `a[i] * a[j] = a[j] * a[i]`. Summing over those `i`,

    contribution(j) = a[j] * ( sum of a[i] over i < j with a[i] > a[j] ).

Every inversion has a unique larger index `j`, so summing `contribution(j)` over all `j` counts every inversion exactly once — no double counting, no omission. Now I need, online, "the sum of values of already-inserted elements whose value is strictly greater than `a[j]`". A Fenwick tree indexed by value-rank, storing the sum of values at each rank, gives prefix sums `query(r) = sum of inserted values with rank <= r`. The sum with rank strictly greater than `a[j]`'s rank `r` is `query(m) - query(r)`, where `m` is the number of distinct values. Strictly greater is the point: using `query(r-1)` instead would be "earlier values `>= a[j]`'s value via the wrong endpoint", and using `query(r)` as the *lower* bound (subtracting it) correctly excludes ties at exactly `a[j]`. Values can be up to `30000`, so I could index by raw value, but to be safe and general I coordinate-compress to ranks `1..m`.

Let me confirm the identity on the sample `a = [3, 1, 4, 1, 5]`, expected `10`. Distinct values `{1,3,4,5}` get ranks `1=1, 3=2, 4=3, 5=4`. Sweep, tree initially empty (storing value-sums by rank):

- `j=0`, `a=3` (rank 2). Earlier values `> 3`: none (tree empty). contribution `3*0 = 0`. Insert value `3` at rank 2.
- `j=1`, `a=1` (rank 1). Earlier values `> 1`: the `3`. Sum `= 3`. contribution `1*3 = 3`. Insert `1` at rank 1.
- `j=2`, `a=4` (rank 3). Earlier values `> 4`: none (`3` and `1` are smaller). contribution `4*0 = 0`. Insert `4` at rank 3.
- `j=3`, `a=1` (rank 1). Earlier values `> 1`: `3` and `4`, sum `= 7`. contribution `1*7 = 7`. Insert `1` at rank 1.
- `j=4`, `a=5` (rank 4). Earlier values `> 5`: none. contribution `5*0 = 0`. Insert `5`.

Total `0 + 3 + 0 + 7 + 0 = 10`. Matches. The identity and the "strictly greater = `query(m) - query(r)`" range are right.

**First implementation — and immediately a trace, because clean math transcribes dirty.** My first cut keeps a single Fenwick over value-sums and does the sweep:

```
vector<int> bitSum(m + 1, 0);          // (first cut: int storage)
auto add = [&](int i, int d){ for(; i<=m; i+=i&(-i)) bitSum[i]+=d; };
auto qr  = [&](int i){ int s=0; for(; i>0; i-=i&(-i)) s+=bitSum[i]; return s; };
long long answer = 0;
for (int j=0;j<n;j++){
    int v=a[j], r=rankOf(a[j]);
    int greater = qr(m) - qr(r);        // (first cut: int)
    answer += (long long)v * greater;
    add(r, v);
}
```

The `answer` is `long long`, so I told myself the overflow worry is handled. But two things nag me: the Fenwick stores `int`, and `greater` is `int`. I trace the smallest input that could expose it. Take `a = [30000, 30000, 30000, 29999, 29999, 29999]` — three big values then three slightly smaller ones, so every one of the `3*3 = 9` cross pairs is an inversion, each contributing `30000 * 29999 = 899970000`. The true answer is `9 * 899970000 = 8099730000`.

Walk the code. For the first three `j` (all value `30000`), earlier-greater sum is `0` (nothing is strictly greater than `30000`), so they add nothing; the tree accumulates value-sum `90000` at `30000`'s rank. For `j=3,4,5` (value `29999`), the earlier-greater sum is the value-sum of the inserted `30000`s. At `j=3` that is `90000` (three `30000`s); `greater = 90000`, contribution `29999 * 90000`. But hold on — `qr` returns `int`, and the *accumulated value-sum in the tree* for the `30000` bucket is `90000`, which is fine as an `int`. The product `29999 * 90000 = 2699910000` overflows `int` *if computed in int* — but I cast `v` to `long long` first, so `(long long)v * greater` is `long long` multiplication. So this particular product is safe. Where does it actually break?

**Diagnosing the first bug (int in the tree path).** I push the trace further to where the *tree's stored sum itself* gets large, and to where `greater` itself overflows. Consider `n` large with many big values inserted: the Fenwick bucket for value `30000` accumulates `count * 30000`. With, say, `100000` copies of `30000`, that bucket holds `3*10^9`, which overflows the `int` storage `bitSum` and the `int` return of `qr`. The moment `qr(m) - qr(r)` is computed in `int`, `greater` is garbage (it has already wrapped), and no amount of casting `v` to `long long` afterward recovers it — I am multiplying by a corrupted `int`. So the defect is precise: **the Fenwick tree stores and returns `int`, but the value-sums it accumulates exceed `2^31`.** The `(long long)` cast I sprinkled on the product is in the wrong place; the overflow happens upstream, inside the tree, before the multiply. To make the failure concrete and reproducible at small scale, I build an all-`int` variant (tree, `qr`, and `answer` all `int`) and feed it the 6-element case above: it prints `-490204592`, a negative number, while the true answer is `8099730000`. The `int` accumulator wrapped around `2^31` after just a few `~9*10^8` products. That is the silent-overflow signature: a plausible-looking, often *negative*, wrong number — never a crash.

**Fixing the first bug.** Make the Fenwick tree store `long long`, and have `qr` return `long long`. Then `greater` is `long long`, the value-sums never wrap, and `v * greater` is `long long * long long`. While here I also make the multiplication unambiguously 64-bit by giving `v` type `long long` so I never rely on a cast being in exactly the right spot:

```
vector<long long> bitSum(m + 1, 0);
auto add = [&](int i, long long d){ for(; i<=m; i+=i&(-i)) bitSum[i]+=d; };
auto qr  = [&](int i){ long long s=0; for(; i>0; i-=i&(-i)) s+=bitSum[i]; return s; };
long long answer = 0;
for (int j=0;j<n;j++){
    long long v = a[j];
    int r = rankOf(a[j]);
    long long greater = qr(m) - qr(r);
    answer += v * greater;
    add(r, v);
}
```

Re-run the all-`long long` version on `[30000,30000,30000,29999,29999,29999]`: it prints `8099730000`. Correct, and it matches the brute force. The bug was upstream of the multiply, exactly as the trace predicted, and moving everything on the value path to 64-bit fixes it.

**Second trace — the ordering of insert vs. query.** A different, subtler way to be wrong: inserting `a[j]` into the tree *before* querying, or using a non-strict range. I trace `a = [2, 2]` (two equal values; equal is *not* an inversion, so the answer must be `0`). Ranks: value `2` is rank `1`, `m = 1`.

- `j=0`, `v=2`, `r=1`. `greater = qr(1) - qr(1) = 0`. contribution `0`. Then `add(1, 2)` — tree bucket `1` holds `2`.
- `j=1`, `v=2`, `r=1`. `greater = qr(1) - qr(1) = (2) - (2) = 0`. contribution `0`. Then insert.

Total `0`. Correct — the strict range `query(m) - query(r)` excludes the equal earlier element because it lands at rank `r`, inside `query(r)`, which I subtract off. Now I deliberately check the *wrong* variants to be sure I picked the right one. If I had used `qr(m) - qr(r-1)` ("greater-or-equal"), then at `j=1` I would get `qr(1) - qr(0) = 2 - 0 = 2`, contribution `2*2 = 4` — wrongly counting the tie as an inversion. And if I had inserted *before* querying, at `j=0` the element would see itself: `greater` would still be `0` here, but on a descending input it would let an element pair with itself or with a later-but-already-inserted equal — breaking the `i < j` requirement. So the two ordering decisions are load-bearing: **query first, then insert**, and **strict-greater via `query(m) - query(r)`**. The `[2,2] -> 0` trace confirms both, and the wrong-variant check shows what each guards against.

**Re-verifying the identity end to end on a decreasing case.** I want one more independent check where every pair is an inversion, because that maximally stresses the accumulation. Take `a = [4, 3, 2]`, strictly decreasing. Inversions: `(4,3), (4,2), (3,2)` with products `12, 8, 6`, total `26`. Sweep: `j=0` value `4`, greater-sum `0`, add `4`. `j=1` value `3`, earlier `>3` is `{4}`, sum `4`, contribution `3*4=12`, add `3`. `j=2` value `2`, earlier `>2` is `{4,3}`, sum `7`, contribution `2*7=14`, add `2`. Total `0+12+14 = 26`. Matches the hand count. Good — and note `12 + 14 = 26` regroups the three pairwise products `12 + 8 + 6`, which is the same sum sliced by larger-index instead of by pair. The identity holds.

**Edge cases, deliberately, because this is where this kind of code dies.**
- `n = 0`: `cin >> n` succeeds with `n = 0`, the value loop and the sweep never run, `answer` stays `0`. (If input is truly empty, `cin >> n` fails and I `return 0` having printed nothing — but the contract always provides `n`, and `n = 0` prints `0`.) Correct.
- `n = 1`, e.g. `a = [7]`: the sweep runs once, finds no earlier elements, contribution `0`, answer `0`. Correct — a single element has no pairs.
- Strictly increasing, `[1, 2, 3, 4]`: every `j` finds no earlier element strictly greater, every contribution `0`, answer `0`. Correct.
- All equal, `[5, 5, 5, 5]`: ties are never inversions; every `greater` is `0`; answer `0`. Correct (verified by the `[2,2]` trace logic at scale).
- Coordinate compression with `m` distinct values: ranks run `1..m`, the Fenwick has size `m+1`, `query(m)` covers everything, no off-by-one in the tree bounds. With `n = 0` the compressed array is empty, `m = 0`, and the loop never queries — safe.
- Overflow: the final accumulator, the tree storage, and `greater` are all `long long`; the worst-case total `~4.5*10^18` fits with room under `9.2*10^18`, and a single product `~9*10^8` fits trivially in 64-bit. The first-bug trace proved an `int` anywhere on this path wraps to a negative wrong answer.
- Output: exactly one integer and a newline; `cin >>` consumes arbitrary whitespace, so the input format is robust to spacing/newlines.

**Final solution.** I convinced myself the *idea* is right by deriving the larger-index decomposition and hand-checking it on `[3,1,4,1,5]` and `[4,3,2]`, and I convinced myself the *code* is right by tracing a big-value case to a precise upstream int-overflow in the Fenwick path (the all-`int` variant prints the negative `-490204592` instead of `8099730000`) and by tracing `[2,2]` to confirm the strict-greater range and the query-before-insert order. That is what I ship — one self-contained file, the `O(n log n)` Fenwick sweep with every value-path quantity in 64-bit:

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    if (!(cin >> n)) return 0;             // n = 0 (or empty input) -> answer 0
    vector<int> a(n);
    for (auto &x : a) cin >> x;

    // Coordinate-compress the values so the Fenwick index space is O(n).
    vector<int> srt(a);
    sort(srt.begin(), srt.end());
    srt.erase(unique(srt.begin(), srt.end()), srt.end());
    int m = (int)srt.size();
    auto rankOf = [&](int v) {
        return int(lower_bound(srt.begin(), srt.end(), v) - srt.begin()) + 1; // 1..m
    };

    // Two Fenwick trees indexed by compressed value:
    //   cnt[r] = how many earlier elements have value-rank r,
    //   sum[r] = sum of those earlier element VALUES (this is the part that overflows int).
    vector<long long> bitCnt(m + 1, 0), bitSum(m + 1, 0);
    auto add = [&](vector<long long> &bit, int i, long long delta) {
        for (; i <= m; i += i & (-i)) bit[i] += delta;
    };
    auto query = [&](vector<long long> &bit, int i) {
        long long s = 0;
        for (; i > 0; i -= i & (-i)) s += bit[i];
        return s;
    };

    long long answer = 0;
    // Sweep left to right. For element j (value v, rank r), every earlier element with a
    // strictly greater value forms an inversion (i<j, a[i] > a[j]); each contributes a[i]*a[j].
    // Summed over those earlier elements that is v * (sum of their values).
    for (int j = 0; j < n; j++) {
        long long v = a[j];
        int r = rankOf(a[j]);
        // earlier elements with rank in (r, m]  =>  value strictly greater than a[j].
        long long greaterValueSum = query(bitSum, m) - query(bitSum, r);
        answer += v * greaterValueSum;     // v fits int but the product / accumulator do not
        // insert this element
        add(bitCnt, r, 1);
        add(bitSum, r, v);
    }

    cout << answer << "\n";
    return 0;
}
```

**Causal recap.** The pair-sum-over-inversions answer decomposes cleanly by *larger index* — each `j` contributes `a[j]` times the sum of earlier values strictly greater than `a[j]` — which a Fenwick tree of value-sums computes online in `O(n log n)` (the merge-sort alternative works but its merge bookkeeping is more error-prone), and I checked that identity on `[3,1,4,1,5] -> 10` and `[4,3,2] -> 26`; the first implementation stored the tree value-sums in `int`, so on big-value inputs the accumulated bucket and the `query(m)-query(r)` difference wrapped past `2^31` *before* the `long long` multiply could help — the all-`int` variant returns the tell-tale negative `-490204592` instead of `8099730000` — fixed by making the tree storage, the query return, and `greater` all `long long`; a second trace of `[2,2] -> 0` locked in the strict-greater range `query(m) - query(r)` and the query-before-insert order (which together forbid ties and self/future pairs); and the empty, single, increasing, and all-equal corners all fall out of the same recurrence with the final `long long` accumulator absorbing the `~4.5*10^18` worst case.
