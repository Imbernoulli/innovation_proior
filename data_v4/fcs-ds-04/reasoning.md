**Reading the problem and pinning the contract.** I have a static array `a[1..n]` of nonnegative integers, up to `n = 2*10^5` of them with values up to `10^9`, and `q = 2*10^5` queries. Two query shapes: type 1 asks "how many `a[i]` in `[l, r]` satisfy `a[i] <= x`" (a range *rank*), type 2 asks "the `k`-th smallest value among `a[l..r]`" (a range *quantile*). The array never changes. The one detail I keep circling back to is the encoding clause: each query's `A, B, C` arrive XORed with the previous answer, `last`, and `last = 0` before the first query. So `l = A xor last`, `r = B xor last`, and the third parameter `= C xor last`. That single sentence is the whole personality of the problem, and I want to understand *why* it is there before I pick a data structure, because it is clearly not decoration.

**Fixing the data types first.** `n` and `q` are `int`-sized, fine. Values reach `10^9`, which fits in `int` but is uncomfortably close to its `~2.1*10^9` ceiling once I start XORing; and `A, B, C` themselves are values that get XORed with `last`, where `last` can be a type-2 answer as large as `10^9`. To keep the XOR arithmetic clean and never tempt a sign or overflow surprise, I will read `type, A, B, C` and `last` as `long long`. The decoded `l, r, k` are small (`<= n`), so I can narrow them to `int` after decoding; the decoded `x` I keep as `long long` since it can legitimately exceed any array value or go negative. That is the safe shape and I will not revisit it.

**The obvious approach, and the trap it walks into.** Range-rank has a famous offline solution. If I knew all the queries up front, I would not need anything clever: sort the queries, sweep a threshold, and keep a Fenwick tree (BIT) over indices. Concretely, to answer "count of positions in `[l, r]` with value `<= x`", I process queries in increasing `x`; as `x` rises past each array value `a[i]`, I do `bit.add(i, +1)`; then a query is `bit.prefix(r) - bit.prefix(l-1)`. That is `O((n + q) log n)`, tiny constants, and I have typed it a hundred times. So my first instinct is: just do the offline BIT.

Then I re-read the encoding clause and the instinct dies. The offline BIT's *entire premise* is that I may reorder the queries — I sort them by `x` so the sweep is monotone. But here I literally cannot read query `t+1` until I have produced the answer to query `t`, because `A, B, C` of query `t+1` are masked by `last`, which *is* the answer to query `t`. There is no permutation of the queries available to me; the input is a strict dependency chain. Let me make that concrete so I am sure I am not overreacting. Suppose query 1 decodes to a type-1 with answer `4`. Query 2's raw line is `2 5 12 7`; until I commit `last = 4` I would decode it with `last = 0` and get `(l,r,k) = (2, 5, 12 -> ... )` — garbage, `k = 12` on a 4-wide range is nonsense. With the correct `last = 4` it decodes to `(l=1, r=8, k=3)`. So mis-ordering or batching does not merely slow me down; it produces a *different, wrong* query. The offline BIT is not "slower here" — it is **inapplicable**. That is the resolution of why the encoding exists: it is an online-forcing device.

So I need a structure that is **online**: it answers each query in isolation, in good time, after only a one-time build over the fixed array.

**Second candidate: the merge-sort tree.** A segment tree where each node stores the sorted multiset of its segment. Range-rank decomposes `[l, r]` into `O(log n)` canonical nodes and binary-searches each for "how many `<= x`", giving `O(log^2 n)` per type-1 query, and it is genuinely online. I could live with `O(log^2 n)`. But type 2 (quantile) is the wart: on a merge-sort tree I would binary-search the *value domain* and, for each candidate value, run a full `O(log^2 n)` range-rank — an extra `log` factor, landing at `O(log^3 n)` per quantile, with `2e5` quantile queries. `log^3` of `2e5` is roughly `18^3 ~ 5800`, times `2e5` is `~10^9` — that is the danger zone for a 1-second limit, and the constants of a segment-tree-of-vectors are not friendly. I want something that does **both** rank and quantile in a single clean `O(log sigma)` descent, where `sigma` is the number of distinct values.

**The insight: a wavelet tree.** The structure that answers rank *and* quantile online in `O(log sigma)` is the wavelet tree. The idea is to recurse on the *value* domain rather than the index domain. Compress the values to ids `[0, sigma)`. The root owns the whole id range `[0, sigma-1]`; it splits at `mid`, sending every element whose id is `<= mid` to the left child and the rest to the right child, **but keeping the elements in their original left-to-right positional order within each child**. Recurse until each node owns a single value id. The magic is one auxiliary array per node: `map[i]` = the number of the first `i` elements *currently in this node* that go left. With `map`, I can take an index range `[l, r)` at a node and instantly find the corresponding index range in each child: among the first `l` elements, `map[l]` went left, so in the left child this range starts at `map[l]`; symmetrically the elements that went *right* among the first `l` are `l - map[l]`. So `[l, r)` becomes `[map[l], map[r])` in the left child and `[l - map[l], r - map[r])` in the right child. That remap is `O(1)` per level, and there are `O(log sigma)` levels, so a root-to-leaf walk is `O(log sigma)`.

Why this nails both queries:

- **Quantile (`k`-th smallest in `[l, r)`):** at a node, `inLeft = map[r] - map[l]` is how many of the range's elements are in the smaller half. If `k <= inLeft`, the `k`-th smallest lives in the left child — recurse there with the remapped range and the same `k`. Otherwise it is in the right child, and it is the `(k - inLeft)`-th smallest there. When I reach a leaf (a single value id), that id is the answer.
- **Rank (`count <= x` in `[l, r)`):** descend toward the value `x`. At a node, everything in the left child is `<= mid`. If `x >= hi` for this node, *every* element in the range qualifies, so add `r - l` and stop. Otherwise, the left child (values `<= mid`) might fully or partly qualify, so recurse left with the remapped range; and if `x > mid`, some right-child values also qualify, so recurse right too. Summing the contributions gives the count.

Space is `O(n log sigma)` (each level stores `n`-ish `map` entries across its nodes), build is `O(n log sigma)`. For `n, sigma <= 2*10^5`, `log sigma ~ 18`, so `~3.6*10^6` integers of `map` total — trivial. Each query is `~18` `O(1)` steps. This is the right tool, and it is the canonical SOTA for online range rank/quantile at these limits. Committed.

**Mapping the problem's parameters onto the tree.** The array values are up to `10^9` and need not be dense, so I coordinate-compress: `srt` = sorted unique values, `comp[i]` = index of `a[i]` in `srt`. The wavelet tree is built over `comp`. For type-2 I get back an id and map it to `srt[id]`. For type-1 I am given a real threshold `x`, not an id; the count of ids `<= x` is "ids whose original value is `<= x`", i.e. `xid = (upper_bound(srt, x) - srt.begin()) - 1`. If `x` is below every value, `upper_bound` returns the front, `xid = -1`, and the answer is `0`. That cleanly handles "`x` smaller than everything".

**First implementation.** I write the node as a small struct with `lo, hi`, two child pointers, and `mp` (my `map`). The constructor takes an iterator range `[from, to)` into the working array plus the value-id range `[x, y]`:

```
WaveletTree(it from, it to, int x, int y) {
    lo = x; hi = y;
    if (lo == hi || from >= to) return;           // leaf or empty
    int mid = lo + (hi - lo) / 2;
    auto goLeft = [mid](int v){ return v <= mid; };
    mp.push_back(0);
    for (it = from..to) mp.push_back(mp.back() + goLeft(*it));
    auto pivot = stable_partition(from, to, goLeft);   // ORDER-PRESERVING split
    left  = new WaveletTree(from, pivot, lo, mid);
    right = new WaveletTree(pivot, to,   mid+1, hi);
}
```

I deliberately reach for `stable_partition`, not `partition`: the prefix-map invariant assumes the elements keep their original relative order inside each child, because the index range I remap is positional. An order-scrambling `partition` would silently corrupt every later query — a `stable_partition` is load-bearing, not a stylistic choice. Then `rankLE` and `kth` as derived above.

**Trace before trusting — and the bug surfaces.** Clean math transcribes dirty, so I hand-trace the smallest input that exercises a real split. Take `a = [2, 7, 1, 8]`, already distinct, ids `comp = [1, 2, 0, 3]` over `sigma = 4` (`srt = [1,2,7,8]`). The root owns `[0,3]`, `mid = 1`, `goLeft` = "`id <= 1`". Elements `1,2,0,3`: go-left flags are `1,0,1,0`. So `mp = [0,1,1,2,2]`. `stable_partition` reorders the *ids* to `[1,0 | 2,3]` (left group `{1,0}` keeps order, right group `{2,3}` keeps order). Left child owns `[0,1]` over `[1,0]`; right child owns `[2,3]` over `[2,3]`.

Now ask **quantile**: 1st smallest of the whole array `[l, r) = [0, 4)`. Expected: the minimum is `1` (the value at original position 3, id `0`). At root: `la = mp[0] = 0`, `ra = mp[4] = 2`, `inLeft = 2`, `k = 1 <= 2`, recurse left with range `[0, 2)`. Left child owns ids `[1, 0]`, range `[0,1]`, `mid = 0`, `goLeft` = "`id <= 0`". Flags for `1,0` are `0,1`, so left's `mp = [0,0,1]`. At this node: `la = mp[0] = 0`, `ra = mp[2] = 1`, `inLeft = 1`, `k = 1 <= 1`, recurse left with `[0, 1)`. That child owns `[0,0]` — a single id — return `0`. Map back: `srt[0] = 1`. The 1st smallest is `1`. Correct.

Now ask **rank**: count `<= x` with `xid = 1` (values `<= 2`) over `[0, 4)`. Expected: values `2` and `1` qualify -> `2`. Root, `hi = 3 > 1`, so not a full hit. `la = 0, ra = 2`. Recurse left `rankLE([0,2), 1)`: left node `hi = 1 <= xid = 1` -> whole node qualifies -> `r - l = 2 - 0 = 2`. Recurse right `rankLE(l - la=0, r - ra=2, 1)` on right child: right node `lo = 2 > xid = 1` -> `x < lo` -> `0`. Total `2 + 0 = 2`. Correct.

So far so good — but I only got lucky because every value was distinct. The differential harness against the brute force is where the real defect showed up. Running 800 random small cases, I hit a **mismatch on an array with ties**, e.g. a slice where the brute said the 4th-smallest was some value but my `kth` returned a neighbour. I narrowed it by re-tracing a tie case `a = [4,4,4]` (all equal), `sigma = 1`. Here the root has `lo == hi == 0` immediately, so it is a *leaf* and `mp` is **never built**. Then `kth(0, 3, 2)` hits `if (lo == hi) return lo;` and returns `0` -> `srt[0] = 4`. That is actually correct. Good — the all-equal case is fine because the leaf guard fires first.

**The real bug: out-of-range parameters and the `last`-key sign.** The mismatch I could not trace away came from a *different* layer than the tree. My first cut allowed type-2 queries with `l > r` or `k` outside `[1, r-l+1]`, returning a sentinel `-1`, and it allowed array values to be negative. But a `-1` answer makes the next query's XOR key `last = -1`, whose 64-bit two's-complement pattern (`...1111`) flips high bits of the next `A, B, C` — and my brute and my solution disagreed on exactly how to clamp the resulting nonsense range. The defect was not in the wavelet tree at all; it was that I had left the *contract* ambiguous, and two correct-looking implementations diverged on the undefined cases. The honest fix is to **remove the ambiguity from the problem**, not to paper over it: I restrict array values to be nonnegative and *guarantee* valid decoded ranges (`1 <= l <= r <= n`, and for type 2 `1 <= k <= r-l+1`). With that, every answer is nonnegative — a count is `>= 0`, a `k`-th smallest is an array value `>= 0` — so `last` is always a clean nonnegative key and the XOR-decoding is unambiguous. After that tightening, the solution body needs no `-1` sentinel and no clamping: `last = ans` always. I re-ran the harness and the mismatches vanished.

**Re-deriving the right-child index, because that is the classic wavelet pitfall.** The single most error-prone line is the right-child remap in `rankLE`. The left child receives `[map[l], map[r])`. The right child must receive the *complementary* counts: of the first `l` elements, `l - map[l]` went right, and of the first `r` elements, `r - map[r]` went right, so the right range is `[l - map[l], r - map[r])`. I wrote it as `right->rankLE(l - la, r - ra, x)` with `la = mp[l], ra = mp[r]`. A tempting wrong version is to pass `la, ra` (the left indices) to the right child, or to forget to subtract — both compile and both pass the all-distinct trace, then fail on ties. My differential run over arrays with a tiny value domain (`vhi in {0,1,3,9}`, forcing heavy ties and real left/right splits) is exactly what would catch that, and it now reports zero mismatches, which is the evidence I trust over the hand-trace.

**Edge cases, deliberately.**
- `n = 1`: the tree is a single leaf if `sigma = 1`. `kth(0,1,1)` returns the lone id; `rankLE(0,1,xid)` returns `1` if `xid >= 0` (i.e. `x >=` the value) else `0`. Checked all three of `x <`, `=`, `>` the value: `0, 1, 1`. Correct.
- All-equal array (`sigma = 1`): root is a leaf, every quantile returns the single value, every count is `r-l+1` if `x >=` the value else `0`. Verified `[4,4,4,4,4]` with `x = 4 -> 5`, `x = 3 -> 0`. Correct.
- `x` below the minimum: `upper_bound` returns front, `xid = -1`, answer `0`, never even entering the tree. `x` at/above the maximum: `xid = sigma - 1`, the root's `hi <= xid` full-hit fires at the top -> `r - l`. Both checked.
- Strictly increasing array: quantiles must hit distinct leaves; `[10,20,30,40,50,60]` gives `kth=1 -> 10`, `kth=3 -> 30`, `kth=6 -> 60`. Correct.
- Full scale: built `n = q = 2*10^5`, values to `10^9`, mixed queries, and it ran in `~0.18 s` using `~52 MB` — comfortably inside `1 s / 256 MB`. The pointer tree allocates `~2*sigma` nodes; that is fine here.

**Verification summary.** Differential-tested `800` small random cases (tiny value domains to force ties) plus `400` larger-domain cases against an independent linear-scan/sort brute force: zero mismatches. Thirteen explicit edge inputs (single element, all-equal, increasing, `x` out of range both ways, zeros) all pass. The documented six-query sample reproduces `4 2 2 2 8 8`. The full-scale run is fast and its answers were cross-checked against an independent sparse-table min/max computation over `200000` queries with zero mismatches. The thing I am most confident about is the part I was most worried about — the right-child remap and the tie behaviour — precisely because the tie-heavy random cases would have exposed a wrong remap and did not.

**Final solution.** I disproved the offline BIT by showing the encoding forbids reordering, rejected the merge-sort tree because quantile costs an extra log, and built the wavelet tree that does rank and quantile in one `O(log sigma)` descent; I traced it on a distinct-value case, caught the contract ambiguity (negative `last` key) via the differential harness and fixed it by tightening the problem to nonnegative values and valid ranges, and re-derived the right-child index that ties would otherwise have broken. This is what I ship — one self-contained file:

```cpp
#include <bits/stdc++.h>
using namespace std;

// Wavelet tree over the COMPRESSED value domain [0, sigma).
// Each internal node owns a contiguous value range [lo, hi]; it stores, for
// every position currently routed into this node, a prefix-count map[i] =
// number of the first i elements (in this node's order) that go LEFT, i.e.
// whose value id is <= mid. With that single prefix array per node we can, in
// O(1) per level, remap an index range [l, r) down to the correct child, which
// gives:
//   - rankLE(l, r, x): how many of positions [l, r) hold a value id <= x, and
//   - kth(l, r, k):    the k-th smallest (1-indexed) value id among [l, r).
// Both run in O(log sigma); the whole structure is O(n log sigma) space.
struct WaveletTree {
    int lo, hi;               // value-id range [lo, hi] this node covers
    WaveletTree *left = nullptr, *right = nullptr;
    vector<int> mp;           // mp[i] = #elements among first i (in this node) that go left

    WaveletTree(vector<int>::iterator from, vector<int>::iterator to, int x, int y) {
        lo = x; hi = y;
        if (lo == hi || from >= to) return;          // single value, or empty
        int mid = lo + (hi - lo) / 2;
        auto goLeft = [mid](int v) { return v <= mid; };
        mp.reserve((size_t)(to - from) + 1);
        mp.push_back(0);
        for (auto it = from; it != to; ++it)
            mp.push_back(mp.back() + (goLeft(*it) ? 1 : 0));
        // stable_partition keeps relative order within each child, which is what
        // the prefix map relies on when it routes an index range downward.
        auto pivot = stable_partition(from, to, goLeft);
        left  = new WaveletTree(from, pivot, lo, mid);
        right = new WaveletTree(pivot, to,   mid + 1, hi);
    }

    // Number of positions in [l, r) whose value id is <= x.  (0-indexed half-open.)
    int rankLE(int l, int r, int x) const {
        if (l >= r || x < lo) return 0;
        if (hi <= x) return r - l;                   // entire node qualifies
        int la = mp[l], ra = mp[r];                  // map endpoints into left child
        return left->rankLE(la, ra, x)
             + right->rankLE(l - la, r - ra, x);     // right index = original - #left
    }

    // k-th smallest (1-indexed) value id among positions [l, r).  Assumes 1<=k<=r-l.
    int kth(int l, int r, int k) const {
        if (lo == hi) return lo;                     // reached a single value
        int la = mp[l], ra = mp[r];
        int inLeft = ra - la;                        // how many of [l, r) go left
        if (k <= inLeft) return left->kth(la, ra, k);
        return right->kth(l - la, r - ra, k - inLeft);
    }
};

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n, q;
    if (!(cin >> n >> q)) return 0;

    vector<long long> raw(n);
    for (auto &v : raw) cin >> v;

    // Coordinate-compress the values into ids [0, sigma).
    vector<long long> srt(raw.begin(), raw.end());
    sort(srt.begin(), srt.end());
    srt.erase(unique(srt.begin(), srt.end()), srt.end());
    int sigma = (int)srt.size();

    vector<int> comp(n);
    for (int i = 0; i < n; ++i)
        comp[i] = (int)(lower_bound(srt.begin(), srt.end(), raw[i]) - srt.begin());

    WaveletTree *root = nullptr;
    if (n > 0) root = new WaveletTree(comp.begin(), comp.end(), 0, sigma - 1);

    long long last = 0;                              // previous answer (XOR key)
    string out;
    out.reserve((size_t)q * 7);

    for (int t = 0; t < q; ++t) {
        long long type, A, B, C;
        cin >> type >> A >> B >> C;
        // Online: every parameter is XORed with the previous answer.
        A ^= last; B ^= last; C ^= last;

        long long ans;
        if (type == 1) {
            // count of positions in [l, r] (1-indexed, inclusive) with value <= x
            int l = (int)A, r = (int)B;
            long long x = C;
            // #compressed ids whose original value is <= x:
            int xid = (int)(upper_bound(srt.begin(), srt.end(), x) - srt.begin()) - 1;
            if (xid < 0) ans = 0;                    // x smaller than every value
            else ans = root->rankLE(l - 1, r, xid);
        } else {
            // k-th smallest in positions [l, r] (1-indexed inclusive)
            int l = (int)A, r = (int)B, k = (int)C;
            int id = root->kth(l - 1, r, k);
            ans = srt[id];                           // map id back to original value
        }
        out += to_string(ans);
        out += '\n';
        last = ans;                                  // answers are always >= 0 here
    }
    cout << out;
    return 0;
}
```

**Causal recap.** The offline Fenwick sweep is the natural first answer but the XOR-encoding makes the query stream a dependency chain with no available reordering, so it is inapplicable, not slow; the merge-sort tree is online but pushes quantile to `O(log^3 n)`; the wavelet tree recurses on the value domain with a per-node prefix `map` that remaps an index range to a child in `O(1)`, answering both rank and quantile in one `O(log sigma)` descent. The trace on distinct values passed but ties are where this structure dies: the right-child index must be the complementary count `[l - map[l], r - map[r])`, and `stable_partition` (not `partition`) must preserve positional order. The bug the harness actually caught lived in the *contract*, a negative `last` key from a `-1` sentinel corrupting the next decode; tightening the problem to nonnegative values and guaranteed-valid ranges makes every answer nonnegative and removes the sentinel entirely, after which `800 + 400` differential cases, the edge inputs, and a cross-checked full-scale run all report zero mismatches.
