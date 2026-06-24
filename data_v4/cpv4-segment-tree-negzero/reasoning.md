**Reading the problem and pinning the contract.** I maintain `a[0..n-1]`, values can be negative or zero, and I answer two operation types interleaved: `1 i x` sets `a[i]=x`, and `2 l r` asks for the maximum sum of a **non-empty** contiguous subarray inside `a[l..r]`. The non-empty clause is the whole personality of this problem: an all-negative range cannot return `0` by "taking nothing" — it must return its largest single element, which is negative; an all-zero range returns `0`; a range with any positive returns a positive. Before any algorithm I fix the scale, because it dictates types: `n, q <= 2*10^5` and `|a[i]| <= 10^9`, so a range *total* can reach `2*10^5 * 10^9 = 2*10^14`, well past the 32-bit ceiling of about `2.1*10^9`. Every stored quantity (`pre`, `suf`, `tot`, `best`) and every accumulator must be `long long`. An `int` here is a silent wrong-answer on the large tests, non-negotiable.

**Why the obvious thing is too slow.** The static maximum subarray is Kadane in `O(n)`. The lazy plan is: keep `a` as a plain array, apply `1 i x` in `O(1)`, and answer each `2 l r` with a fresh Kadane scan over `a[l..r]`. That is obviously correct, and I will in fact use exactly that as my independent brute-force oracle. But its cost is `O(nq)` in the worst case, up to `4*10^10` element touches, which cannot pass a 2-second limit. So it is the oracle, not the shipment. I need point updates and arbitrary-range queries both in `O(log n)`, which points at a segment tree.

**Candidate approaches and the one I can prove.** The standard trick for "best subarray under merges" is to make each segment-tree node summarize its range with four numbers, chosen so that two adjacent summaries fuse in `O(1)` without re-examining the underlying elements:

- `tot`  = sum of the whole range (used to thread prefixes/suffixes through a child),
- `pre`  = maximum sum of a **non-empty** prefix of the range (subarray starting at the left end),
- `suf`  = maximum sum of a **non-empty** suffix of the range (subarray ending at the right end),
- `best` = maximum sum of any **non-empty** subarray inside the range.

The merge of a left summary `L` and right summary `R` into the parent `P` is:

- `P.tot  = L.tot + R.tot`.
- `P.pre  = max(L.pre, L.tot + R.pre)` — a best prefix of `P` either stays inside `L`, or spans all of `L` and continues into a best prefix of `R`.
- `P.suf  = max(R.suf, R.tot + L.suf)` — symmetric, anchored at the right end.
- `P.best = max(L.best, R.best, L.suf + R.pre)` — the best subarray lies entirely in `L`, entirely in `R`, or straddles the boundary as (best suffix of `L`) + (best prefix of `R`).

A leaf over a single value `v` is `Node{pre=v, suf=v, tot=v, best=v}`: the only non-empty subarray is the element itself. Build is `O(n)`, update and query are `O(log n)`. The risk is *not* the high-level idea — it is two transcription traps that the negatives-and-zeros corner is purpose-built to expose: the **identity element** for empty/out-of-range pieces, and the **sign handling** in the cross-boundary term. I will not trust either until I have traced them.

**Deriving and sanity-checking the merge on paper.** Let me confirm `best` is right before coding, on `a = [-2, 1, -3, 4, -1, 2]`, the documented sample, querying the whole range. The true answer is `[4, -1, 2] = 5`. Split `[-2,1,-3] | [4,-1,2]`.

Left `[-2,1,-3]`: subarrays and sums — `-2; 1; -3; -1; -2; -4`. So `L.best = 1` (the lone `1`). `L.tot = -4`. Best non-empty prefix: `-2; -1; -4` -> `L.pre = -1`. Best non-empty suffix: `-3; -2; -4` -> `L.suf = -2`.

Right `[4,-1,2]`: subarrays — `4; -1; 2; 3; 1; 5`. `R.best = 5` (the whole thing). `R.tot = 5`. Best prefix: `4; 3; 5` -> `R.pre = 5`. Best suffix: `2; 1; 5` -> `R.suf = 5`.

Merge: `P.best = max(L.best=1, R.best=5, L.suf + R.pre = -2 + 5 = 3) = 5`. Correct. `P.pre = max(L.pre=-1, L.tot + R.pre = -4 + 5 = 1) = 1`, matching the real best prefix of the full array (`[-2,1] = -1`? no — `[-2,1,-3,4,-1,2]` prefixes are `-2,-1,-4,0,-1,1`, max `1`). Good, `P.pre = 1`. The formulas hold; the idea is sound. Now the dangerous part is the *empty* pieces.

**First implementation — and the identity trap.** A range query that the recursion splits will, on the out-of-range branches, return an "identity" node, and I merge that identity with real nodes. My first instinct for the identity (the empty segment) is the naive zero node:

```
Node identity() { return Node{0, 0, 0, 0}; } // pre=suf=tot=best=0
```

This *feels* harmless — "an empty range contributes nothing." Let me trace it on the corner the problem is built around: `a = [-5, -2, -8]`, query `2 0 2`, true answer `-2` (the best non-empty subarray is the single `-2`, since every sum here is negative). Suppose the query recursion hits a boundary where it merges a real leaf node for `[-2]` = `Node{-2,-2,-2,-2}` with an out-of-range `identity()` = `Node{0,0,0,0}` on its right (this happens whenever the queried range ends and the sibling is outside `[l,r]`). Merge with `L = {-2,-2,-2,-2}`, `R = {0,0,0,0}`:

`P.best = max(L.best=-2, R.best=0, L.suf + R.pre = -2 + 0 = -2) = 0`.

**The bug.** It returns `0`, but `0` is not achievable — there is no non-empty subarray summing to `0` anywhere in `[-5,-2,-8]`. The identity's `best=0` (and `pre=suf=0`) leaked a *phantom empty subarray* of value `0` into the maximum, and on an all-negative range that phantom beats every real (negative) candidate. This is exactly the "wrong base case / sign handling" defect: I encoded the empty segment as if an empty subarray summing to `0` were a legal answer, but the contract forbids the empty subarray. The fix is that the identity must make `pre`, `suf`, and `best` *impossible to win a max* — they have to be a very negative sentinel — while `tot` stays `0` (an empty range genuinely sums nothing when threaded through `tot`). So:

```
const long long NEG = LLONG_MIN / 4;
Node identity() { return Node{NEG, NEG, 0, NEG}; }
```

`tot = 0` because an empty piece adds nothing to a spanning prefix/suffix; `pre=suf=best=NEG` because you cannot start, end, or place a non-empty subarray in nothing, so these must never beat a real value.

**Re-tracing the fix on the all-negative corner.** `L = {-2,-2,-2,-2}` (leaf), `R = identity() = {NEG, NEG, 0, NEG}`. Merge:

- `P.tot = -2 + 0 = -2`.
- `P.pre = max(L.pre=-2, L.tot + R.pre = -2 + NEG) = max(-2, very_negative) = -2`.
- `P.suf = max(R.suf=NEG, R.tot + L.suf = 0 + (-2) = -2) = -2`.
- `P.best = max(L.best=-2, R.best=NEG, L.suf + R.pre = -2 + NEG) = -2`.

Now `P` equals the real `[-2]` summary, exactly as if the identity were not there. The phantom `0` is gone; merging with identity is a true no-op. Tracing the all-negative `2 0 2` end to end now yields `-2`. Fixed, and fixed for the precise reason I diagnosed: the base case had to forbid the empty subarray.

**Why `NEG = LLONG_MIN/4` and not `LLONG_MIN`.** I must never *add* to a value that could already be near the 64-bit floor, or it underflows into a huge positive garbage that then wins the max — a second, subtler sign bug. Where do I add to a possibly-`NEG` quantity? In `L.tot + R.pre` and `R.tot + L.suf` and `L.suf + R.pre`. The `tot` values are bounded by `|tot| <= 2*10^14`. The `pre/suf` values are either real (bounded by `2*10^14`) or `NEG`. So the worst addition is `NEG + (something as low as -2*10^14)`. With `NEG = LLONG_MIN/4 ~ -2.3*10^18`, that sum is about `-2.3*10^18 - 2*10^14`, still far above `LLONG_MIN ~ -9.2*10^18` — no underflow. If I had used `LLONG_MIN` itself, `LLONG_MIN + (negative)` underflows and the result could come back positive, silently corrupting `best`. The `/4` is the guard that keeps every `NEG + real` safely negative without wrapping.

**Second debug episode — the leaf for a negative single element.** Even with the identity fixed, I want to be sure a *leaf* holding a negative value is summarized correctly, because that is the other place a wrong base case hides. Suppose, in an early draft, I had written the leaf "defensively" as `Node{max(v,0LL), max(v,0LL), v, max(v,0LL)}`, reasoning that "a prefix/best should never be negative." Trace it on `a = [-7]`, query `2 0 0`. The leaf becomes `Node{0,0,-7,0}`, so `best = 0`. The query returns `0`. But the true answer is `-7`: the only non-empty subarray of `[-7]` is `[-7]` itself, value `-7`. The `max(v,0)` clamping injected the forbidden empty subarray again, this time at the leaf. So the leaf must be the unclamped `Node{v, v, v, v}` — for `v = -7` that is `Node{-7,-7,-7,-7}`, `best = -7`, correct. The lesson is consistent: *nowhere* may I let `0` stand in for "took nothing," because non-empty is mandatory. The correct leaf is simply the element, sign and all.

**Re-checking that zeros still work after banning the `0` phantom.** I just spent two episodes purging `0` as a default — I must make sure a genuine zero element still produces `0` when it should. Trace `a = [-4, 0, -1]`, query `2 0 2`, true answer `0` (the subarray `[0]`). Leaves: `[-4]={-4,-4,-4,-4}`, `[0]={0,0,0,0}`, `[-1]={-1,-1,-1,-1}`. Merge `[-4]` with `[0]`: `best = max(-4, 0, suf(-4)+pre(0) = -4) = 0`; `pre = max(-4, -4+0=-4) = -4`; `suf = max(0, 0+(-4)=-4) = 0`; `tot = -4`. So `[-4,0]` -> `{pre=-4, suf=0, tot=-4, best=0}`. Merge that with `[-1]={-1,-1,-1,-1}`: `best = max(0, -1, suf=0 + pre=-1 = -1) = 0`. The whole-range `best = 0`, from the real `[0]` element, not from any phantom. Zeros survive precisely because the *element* `0` produces a legitimate `{0,0,0,0}` leaf, while the *identity* (genuinely empty) produces `{NEG,NEG,0,NEG}`. The distinction between "an element that happens to be zero" and "nothing at all" is exactly what the two different nodes encode.

**Query recursion and avoiding an identity merge where I can.** I write `query(node, lo, hi, l, r)` returning a `Node`. Standard segment-tree query: if the node range is disjoint from `[l,r]`, return `identity()`; if fully covered, return the stored node; otherwise recurse both children and merge. One refinement keeps things clean: if the query lies entirely in the left child (`r <= mid`) I return only the left recursion, and symmetrically for the right (`l > mid`), so I merge two real partial nodes only when the range truly straddles `mid`. That isn't required for correctness — the identity is now a real no-op — but it minimizes reliance on the sentinel and reads clearly. Update is the textbook point update: descend to the leaf, set it to `leaf(val)`, recombine on the way up.

**A full trace of the documented sample to validate end to end.** `a = [-2, 1, -3, 4, -1, 2]`.
- `2 0 5`: whole array, derived above as `best = 5` (`[4,-1,2]`). Output `5`.
- `2 0 2`: range `[-2,1,-3]`, computed above as `L.best = 1`. Output `1`.
- `1 2 5`: set `a[2] = 5`, array becomes `[-2, 1, 5, 4, -1, 2]`.
- `2 0 5`: now the best is `[1,5,4,-1,2] = 11` (or `[1,5,4]=10`, but extending through `-1` to `2` adds `+1`, giving `11`). Output `11`.
- `2 0 0`: range `[-2]`, the only subarray is `[-2]`. Output `-2`.

Expected output `5 / 1 / 11 / -2`. I will confirm the compiled program reproduces exactly this.

**Edge cases, deliberately, because this is where this kind of code dies.**
- All-negative range `[-5,-2,-8]`, query `2 0 2`: answer `-2`, the least-bad single element — verified above that the fixed identity no longer leaks `0`.
- All-zero range `[0,0,0]`, query `2 0 2`: every subarray sums to `0`, answer `0`; the real `{0,0,0,0}` leaves produce `best=0`. Correct.
- Single negative element `[-7]`, `2 0 0`: answer `-7`; the unclamped leaf gives `best=-7`. Correct.
- Mixed `[-4,0,-1]`, `2 0 2`: answer `0` from the `[0]` element — traced above.
- `l == r` queries in general: the recursion bottoms out at one leaf, returning that element's value as `best`, exactly the single-element subarray.
- Overflow: all four fields are `long long`; the largest `|tot|` and `|best|` are `~2*10^14`, comfortably inside 64 bits, and `NEG + real` cannot underflow because `NEG = LLONG_MIN/4`.
- Update flipping signs: setting an element makes the leaf `{x,x,x,x}` and recombination rebuilds every ancestor summary, so a range can flip from positive-containing to all-negative and back with no stale state.
- Output: one line per type-`2` query, in order; I buffer into a string and write once to avoid per-line flush overhead under `q = 2*10^5`.

**Final solution.** I convinced myself the *idea* is right by hand-deriving the four merge formulas and checking `best`, `pre`, `suf` on the sample; I convinced myself the *base cases* are right by tracing two real bugs — a `0`-valued identity and a `max(v,0)`-clamped leaf — each of which injected a forbidden empty subarray and lost on the all-negative corner, and by fixing both to a `NEG`-sentinel identity (with `tot=0`) and an unclamped leaf, then re-tracing the all-negative, all-zero, single-negative, and mixed corners to passing. That is what I ship — one self-contained file:

```cpp
#include <bits/stdc++.h>
using namespace std;

struct Node {
    long long pre, suf, tot, best;
};

const long long NEG = LLONG_MIN / 4; // sentinel for "no element": best/pre/suf impossible

// Identity for merge: an empty segment. tot = 0 (sums nothing); pre/suf/best = NEG
// because a non-empty subarray cannot be formed from nothing, so it must never win a max.
Node identity() { return Node{NEG, NEG, 0, NEG}; }

Node leaf(long long v) { return Node{v, v, v, v}; }

Node merge(const Node &L, const Node &R) {
    Node res;
    res.tot  = L.tot + R.tot;
    res.pre  = max(L.pre, L.tot + R.pre);
    res.suf  = max(R.suf, R.tot + L.suf);
    res.best = max(max(L.best, R.best), L.suf + R.pre);
    return res;
}

int n, q;
vector<Node> tree;
vector<long long> a;

void build(int node, int lo, int hi) {
    if (lo == hi) { tree[node] = leaf(a[lo]); return; }
    int mid = (lo + hi) / 2;
    build(node * 2, lo, mid);
    build(node * 2 + 1, mid + 1, hi);
    tree[node] = merge(tree[node * 2], tree[node * 2 + 1]);
}

void update(int node, int lo, int hi, int pos, long long val) {
    if (lo == hi) { tree[node] = leaf(val); return; }
    int mid = (lo + hi) / 2;
    if (pos <= mid) update(node * 2, lo, mid, pos, val);
    else update(node * 2 + 1, mid + 1, hi, pos, val);
    tree[node] = merge(tree[node * 2], tree[node * 2 + 1]);
}

Node query(int node, int lo, int hi, int l, int r) {
    if (r < lo || hi < l) return identity();
    if (l <= lo && hi <= r) return tree[node];
    int mid = (lo + hi) / 2;
    if (r <= mid) return query(node * 2, lo, mid, l, r);
    if (l > mid)  return query(node * 2 + 1, mid + 1, hi, l, r);
    return merge(query(node * 2, lo, mid, l, r),
                 query(node * 2 + 1, mid + 1, hi, l, r));
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    if (!(cin >> n >> q)) return 0;
    a.resize(n);
    for (auto &x : a) cin >> x;

    tree.assign(4 * max(n, 1), identity());
    if (n > 0) build(1, 0, n - 1);

    string out;
    for (int k = 0; k < q; k++) {
        int type;
        cin >> type;
        if (type == 1) {
            int i; long long x;
            cin >> i >> x;
            update(1, 0, n - 1, i, x);
        } else {
            int l, r;
            cin >> l >> r;
            Node res = query(1, 0, n - 1, l, r);
            out += to_string(res.best);
            out += '\n';
        }
    }
    cout << out;
    return 0;
}
```

**Causal recap.** The high-level segment-tree-of-summaries idea was never in doubt once I derived and checked the four merge formulas; the danger was entirely in the base cases that the negatives-and-zeros corner stresses. A naive `Node{0,0,0,0}` identity injected a phantom empty subarray of value `0` that beat every real candidate on an all-negative range (`[-5,-2,-8]` wrongly returning `0` instead of `-2`), and a `max(v,0)`-clamped leaf did the same at the single-element level (`[-7]` wrongly returning `0` instead of `-7`); both are the forbidden empty subarray sneaking in as a `0`. Fixing the identity to `{NEG,NEG,0,NEG}` (sums-nothing `tot`, impossible `pre/suf/best`) and the leaf to the unclamped `{v,v,v,v}` makes "nothing at all" lose every max while a genuine `0` element still legitimately wins, and choosing `NEG = LLONG_MIN/4` keeps every `NEG + real` addition from underflowing; with all four fields `long long`, the all-negative, all-zero, single-negative, mixed, `l==r`, sign-flipping-update, and overflow corners all check out.
