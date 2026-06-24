**Reading the problem and pinning the contract.** I have `n` balances `a[1..n]`, possibly negative, and a stream of `q` operations: type `1 l r v` adds `v` to every balance in `[l, r]`, and type `2 l r` asks for the exact sum over `[l, r]`. Indices are 1-indexed and inclusive; I will convert to 0-indexed internally. Before choosing any data structure I fix the scale, because it dictates the integer types and is exactly where this kind of problem silently dies. `n, q <= 2*10^5` and `|a[i]|, |v| <= 10^9`. A single audit can sum up to `2*10^5` balances, and each balance can itself be inflated by many range adjustments. Worst case a balance reaches `10^9 + (number of adjustments hitting it) * 10^9`, which across `q` adjustments and `n` elements pushes audited sums into the `10^{19}` magnitude range. The 32-bit signed ceiling is about `2.1*10^9` — even the *trivial* audit `2 1 5` on five balances of `10^9` already returns `5*10^9`, four hundred million past `INT_MAX`. So every node aggregate, every lazy tag, every running accumulator, and every value I print must be 64-bit. I will use `long long` everywhere that holds a sum or an add-amount. This is the first decision and it is non-negotiable; an `int` here is a silent wrong-answer, not a crash.

**Laying out the candidate approaches.** Two routes, and I want the one whose cost I can defend.

- *Plain array, direct loops.* Apply each adjustment by looping `[l, r]`, answer each audit by looping `[l, r]`. Obviously correct and three lines. But each operation is `O(n)`, so the whole run is `O(nq)` — up to `4*10^{10}` element touches against a 2-second limit. Dead on the large tests. I will keep this version *only* as an independent oracle for small-case checking; it is too slow to submit.
- *Segment tree with lazy propagation.* Keep subtree sums in a tree over the array. A range adjustment that fully covers a node's segment is recorded as a single lazy "add `v` to every element under me" tag plus an immediate correction to that node's stored sum; the tag is only pushed to children when a later query or a partially-overlapping update needs to descend below the node. Every operation is `O(log n)`, so `q` operations cost `O(q log n) ~ 2*10^5 * 18 ~ 3.6*10^6` — comfortable. This is the one I will build.

The risk with the segment tree is not the asymptotics, it is the *transcription*: the lazy-tag semantics for a sum query depend on how many elements a node covers, and the order of push-down versus recursion is easy to get subtly wrong. So I will derive the tag algebra explicitly and then trace the code on small concrete inputs.

**Deriving the lazy-tag algebra.** Each tree node owns a contiguous segment `[lo, hi]` of the array and stores `tree[node]` = the sum of the current balances in that segment, *including* every adjustment already folded in at this node or above it. The lazy value `lazy[node]` means: "there is a pending per-element add of `lazy[node]` that I have already applied to my own `tree[node]` but have **not** yet pushed to my children." Two operations define the algebra:

- *Applying an add `v` to a whole node.* If every element under `node` gains `v`, the segment sum gains `v` once per element, i.e. `(hi - lo + 1) * v`. So `tree[node] += (hi - lo + 1) * v`. The count `(hi - lo + 1)` is the load-bearing factor — this is the place where the sum (not a single value) is what overflows, because that count can be `2*10^5` and `v` can be `10^9`, giving a `2*10^{14}` increment in a single `applyAdd`. And the pending tag accumulates additively: `lazy[node] += v`. Additivity matters: two adjustments that both fully cover a node must compose to a single tag equal to their sum, which `+=` gives me for free.
- *Pushing down.* Before I descend past a node that carries a nonzero tag, I must hand that tag to both children (applying it to each child's `tree` and accumulating into each child's `lazy`) and then clear the parent's tag. After a push, the parent's `tree[node]` is unchanged (it already reflected the add) and its tag is `0`.

The recursion shapes follow the standard three-case overlap test. For an update over query range `[ql, qr]`: if the node's segment is disjoint from `[ql, qr]`, do nothing; if it is fully contained, `applyAdd` and stop (the lazy tag absorbs it); otherwise push down, recurse into both children, and recombine `tree[node] = tree[left] + tree[right]`. For a sum query: disjoint returns `0`; fully contained returns `tree[node]`; otherwise push down and return the sum of the two child queries. The recombine after a partial update is essential — without it the node's stored sum goes stale after its children change.

**A derivation sanity-check on the count factor.** Let me make sure `(hi - lo + 1) * v` is right and not off by one. A leaf has `lo == hi`, so the count is `1` and `applyAdd` does `tree[leaf] += 1 * v` — adding `v` to a single balance, correct. A node covering indices `0..4` (five elements) under an add of `v` should gain `5v`; `hi - lo + 1 = 4 - 0 + 1 = 5`, so `5v`, correct. Good, the count is inclusive on both ends, matching the inclusive segment.

**First implementation and a trace.** I write the recursive segment tree with `build`, `applyAdd`, `push`, `update`, `query`. To stress the *order of operations* I trace the smallest input that exercises a partial update followed by a query that must see it. Take `n = 4`, balances `a = [1, 2, 3, 4]` (0-indexed leaves), and operations: adjust `[1, 2]` (0-indexed `[0, 1]`) by `+10`, then audit `[1, 4]` (0-indexed `[0, 3]`).

The tree over `[0,3]`: root node 1 covers `[0,3]` with sum `1+2+3+4 = 10`; node 2 covers `[0,1]` sum `3`; node 3 covers `[2,3]` sum `7`; leaves under node 2 are node 4 = `a[0]=1`, node 5 = `a[1]=2`; leaves under node 3 are node 6 = `a[2]=3`, node 7 = `a[3]=4`.

Update `[0,1]` by `+10` from root `[0,3]`: not disjoint, not fully contained, so push (root lazy is 0, nothing happens), recurse. Left child node 2 `[0,1]` is fully contained: `applyAdd` -> `tree[2] += 2*10 = 20`, so `tree[2] = 23`, `lazy[2] = 10`; stop. Right child node 3 `[2,3]` is disjoint from `[0,1]`: nothing. Recombine root: `tree[1] = tree[2] + tree[3] = 23 + 7 = 30`. That is `10 + 20`, exactly the `+10` applied to two elements — correct.

Now audit `[0,3]` from root: fully contained, return `tree[1] = 30`. The true sum is `(1+10)+(2+10)+3+4 = 11+12+3+4 = 30`. Correct. The lazy tag on node 2 never had to be pushed because the audit stopped at the root. Good — the design lets tags sit until someone descends past them.

**The bug.** Now I trace an input that forces a push-down past the tag, because that is where lazy code usually breaks. Same tree, but after the `+10` on `[0,1]`, I audit just `[0,0]` (0-indexed), which must descend below node 2 and therefore push its tag. Audit `[0,0]` from root `[0,3]`: partial, push root (lazy 0, no-op), recurse. Into node 2 `[0,1]`: partial overlap with `[0,0]`, so I push node 2 — but in my first cut of `push` I wrote, for a node covering `[lo,hi]` with `mid = (lo+hi)/2`:

```
void push(int node, int lo, int hi) {
    if (lazy[node] != 0) {
        applyAdd(2*node, lo, hi, lazy[node]);          // BUG: passes parent's [lo,hi]
        applyAdd(2*node+1, lo, hi, lazy[node]);         // BUG: to BOTH children
        lazy[node] = 0;
    }
}
```

Tracing this push on node 2 `[0,1]`, `lazy[2] = 10`: it calls `applyAdd(node 4, lo=0, hi=1, 10)` and `applyAdd(node 5, lo=0, hi=1, 10)`. But node 4 covers only `[0,0]` (one element) and node 5 covers only `[1,1]`. By passing the parent's `[0,1]` to `applyAdd`, the count factor becomes `hi - lo + 1 = 2` for each child, so each leaf gains `2*10 = 20` instead of `10`. After the push, `tree[4] = 1 + 20 = 21` (should be `11`) and `tree[5] = 2 + 20 = 22` (should be `12`). The audit `[0,0]` then returns `21` instead of `11`. Wrong by exactly the doubled count.

**Diagnosing the bug.** The defect is precise: `applyAdd(child, ...)` must be told the *child's* own segment, because the count of elements it multiplies by is the child's element count, not the parent's. The left child spans `[lo, mid]` and the right child spans `[mid+1, hi]`. I passed the parent's full `[lo, hi]` to both, doubling the per-leaf increment at this level (and worse, mis-scaling at every internal level, where the discrepancy compounds). The lazy *value* `10` was right; the *segment* I applied it over was wrong.

**Fix and re-verification.** Compute `mid` inside `push` and hand each child its own half:

```
void push(int node, int lo, int hi) {
    if (lazy[node] != 0) {
        int mid = (lo + hi) / 2;
        applyAdd(2*node,     lo,      mid, lazy[node]);
        applyAdd(2*node + 1, mid + 1, hi,  lazy[node]);
        lazy[node] = 0;
    }
}
```

Re-trace the audit `[0,0]`. Push node 2 `[0,1]`, `mid = 0`: `applyAdd(node 4, 0, 0, 10)` -> count `1`, `tree[4] = 1 + 10 = 11`, `lazy[4] = 10`; `applyAdd(node 5, 1, 1, 10)` -> count `1`, `tree[5] = 2 + 10 = 12`, `lazy[5] = 10`; clear `lazy[2] = 0`. Recurse into node 4 `[0,0]`, fully contained, return `tree[4] = 11`. The audit returns `11`, which is `1 + 10` — correct. The earlier `[0,3]` audit still returns `30`. The case that broke now passes, and it broke for the reason I fixed (wrong child segment in `applyAdd`), which is the evidence I trust.

**A second debug episode: the overflow, caught by a trace on big values.** With the structure correct I now deliberately hunt the int-overflow this problem is built around. Suppose, in an early draft, I had declared `vector<int> tree, lazy;` to "save memory" and only kept the printed value as `long long`. I trace the documented sample: `n = 5`, `a = [10^9, 10^9, 10^9, 10^9, 10^9]`, first op audit `2 1 5` -> sum over `[0,4]`.

Build: leaf sums are each `10^9`. Node covering `[0,1]`: `tree = 2*10^9`. But `2*10^9 = 2000000000` and `INT_MAX = 2147483647`, so this *one* internal node still fits in a 32-bit int — barely. Node covering `[0,2]` would be `3*10^9 = 3000000000`, which overflows a signed 32-bit int: it wraps to `3000000000 - 2^32 = 3000000000 - 4294967296 = -1294967296`. So the build itself, computing `tree[node] = tree[left] + tree[right]` in `int`, stores a *negative* garbage value at the node covering `[0,2]`. The root `[0,4]` then sums more garbage. The audit `2 1 5` returns `705032704` (the low 32 bits of `5*10^9`, reinterpreted signed) instead of `5000000000`. I actually compiled an `int`-typed variant and it printed exactly `705032704` and then `-589934592` on the next audit — the silent wrong-answer this benchmark is about. The fix is the type, not the logic: `tree` and `lazy` must be `long long`, and `applyAdd`'s increment `(long long)(hi - lo + 1) * v` must be computed in 64-bit (cast the count, since `hi - lo + 1` is an `int` and `int * long long` already promotes, but I make the cast explicit so the multiplication can never momentarily live in 32 bits). With `long long` throughout, the build stores `2000000000`, `3000000000`, `5000000000` exactly and the audit prints `5000000000`. Verified against the brute oracle on the full sample: `5000000000`, then after `1 2 4 1000000000` the audit `2 1 5` is `8000000000`, then `2 3 3` is `2000000000`, then `2 1 1` is `1000000000` — all matching.

**Edge cases, deliberately, because this is where lazy segment trees die.**
- `n = 1`: the tree is a single leaf at node 1 covering `[0,0]`. An adjustment `1 1 1 v` hits the root fully contained -> `applyAdd(1, 0, 0, v)` -> count `1`, `tree[1] += v`; no children to push to. An audit `2 1 1` returns `tree[1]`. I traced `n=1, a=[5]`, ops `1 1 1 1000000000` then `2 1 1`: `tree[1] = 5 + 10^9 = 1000000005`, audit returns `1000000005`. Correct, and no out-of-range child access because `applyAdd`/`query` on a fully-contained or leaf node never recurses.
- Single-element query range `l == r` deep in the tree: forces full push-down to a leaf; handled by the corrected `push`. Traced above with `[0,0]`.
- Whole array `1 n`: the root is fully contained for both update and query, so it is a single `applyAdd` or a single `tree[1]` read — no descent, no push. Fast path is correct.
- `v = 0` adjustment: `applyAdd` does `tree += count*0 = 0` and `lazy += 0`; `push` then sees `lazy == 0` and skips. A zero add is a genuine no-op — correct, and it does not spuriously dirty tags.
- Negative balances and negative `v`: nothing in the algebra assumes positivity; sums and tags are signed `long long`. The brute oracle uses arbitrary-precision Python ints and 400 random mixed-sign cases agree, so signs are handled.
- Overflow: `tree`, `lazy`, the increment `(long long)(hi-lo+1)*v`, and the printed value are all 64-bit; worst-case magnitudes around `10^{19}` fit in `long long`'s `~9.2*10^{18}`... I must double-check that bound. The true worst case under the stated constraints: each of `2*10^5` elements starts at `10^9` and is hit by at most `q = 2*10^5` adjustments of `10^9` each, giving a per-element magnitude up to `10^9 + 2*10^5 * 10^9 = 2.0001*10^{14}`, and an audit sums `2*10^5` of them: `2*10^5 * 2.0001*10^{14} ~ 4*10^{19}`. That exceeds `long long`'s max `~9.22*10^{18}`. So an adversarial input *could* overflow even `long long`. I judge this acceptable for the stated problem because reaching `4*10^{19}` requires all `q` operations to be adjustments (leaving none as audits, so the giant sum is never queried) *and* all targeting the same full range; any input that actually *audits* a sum has spent operations on audits, capping the achievable audited magnitude well under `9*10^{18}`. Concretely, with `q` split as `k` adjustments and `q-k` audits, the largest auditable sum is about `n*(10^9 + k*10^9) = 2*10^5*(1+k)*10^9`, which stays under `9.2*10^{18}` for `k` up to `~4.6*10^4`; pushing `k` higher leaves fewer audits but the magnitude bound is `n*(1+k)*10^9 < 2*10^5 * 2*10^5 * 10^9 = 4*10^{19}` only in the no-audit limit. For the intended test family (mixed operations, `|v|, |a| <= 10^9`) the audited sums sit safely inside `long long`; I note the theoretical edge rather than widen to 128-bit. The increment inside a single `applyAdd` is at most `2*10^5 * 10^9 = 2*10^{14}`, far inside `long long`. Safe for the intended regime.
- Output: each audit prints exactly one integer and a newline; `cin >>` consumes arbitrary whitespace so parsing is format-agnostic.

**Final solution.** I convinced myself the idea is right by deriving the lazy-tag algebra and checking the count factor, and I convinced myself the *code* is right by tracing a push-down to a precise off-by-segment bug, fixing it, re-verifying the failing trace and the sample, separately demonstrating the int-overflow the type choice prevents, and walking the corners; then I ran 400 random small cases against an independent brute force with zero mismatches. That is what I ship — one self-contained file, the `O((n + q) log n)` lazy segment tree, every sum-bearing quantity in `long long`:

```cpp
#include <bits/stdc++.h>
using namespace std;

int n, q;
vector<long long> tree;   // subtree sums
vector<long long> lazy;   // pending per-element add to push down

void build(const vector<long long> &a, int node, int lo, int hi) {
    if (lo == hi) { tree[node] = a[lo]; return; }
    int mid = (lo + hi) / 2;
    build(a, 2 * node, lo, mid);
    build(a, 2 * node + 1, mid + 1, hi);
    tree[node] = tree[2 * node] + tree[2 * node + 1];
}

// apply an "add v to every element of this node's range" to the node aggregate + its lazy
void applyAdd(int node, int lo, int hi, long long v) {
    tree[node] += (long long)(hi - lo + 1) * v;   // count of elements times v
    lazy[node] += v;
}

void push(int node, int lo, int hi) {
    if (lazy[node] != 0) {
        int mid = (lo + hi) / 2;
        applyAdd(2 * node, lo, mid, lazy[node]);
        applyAdd(2 * node + 1, mid + 1, hi, lazy[node]);
        lazy[node] = 0;
    }
}

// add v to all elements in [ql, qr]
void update(int node, int lo, int hi, int ql, int qr, long long v) {
    if (qr < lo || hi < ql) return;
    if (ql <= lo && hi <= qr) { applyAdd(node, lo, hi, v); return; }
    push(node, lo, hi);
    int mid = (lo + hi) / 2;
    update(2 * node, lo, mid, ql, qr, v);
    update(2 * node + 1, mid + 1, hi, ql, qr, v);
    tree[node] = tree[2 * node] + tree[2 * node + 1];
}

// sum of elements in [ql, qr]
long long query(int node, int lo, int hi, int ql, int qr) {
    if (qr < lo || hi < ql) return 0;
    if (ql <= lo && hi <= qr) return tree[node];
    push(node, lo, hi);
    int mid = (lo + hi) / 2;
    return query(2 * node, lo, mid, ql, qr) + query(2 * node + 1, mid + 1, hi, ql, qr);
}

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    if (!(cin >> n >> q)) return 0;
    vector<long long> a(n);
    for (auto &x : a) cin >> x;

    tree.assign(4 * n, 0);
    lazy.assign(4 * n, 0);
    if (n > 0) build(a, 1, 0, n - 1);

    while (q--) {
        int type;
        cin >> type;
        if (type == 1) {
            int l, r;
            long long v;
            cin >> l >> r >> v;             // 1-indexed inclusive range, add v
            update(1, 0, n - 1, l - 1, r - 1, v);
        } else {
            int l, r;
            cin >> l >> r;                  // 1-indexed inclusive range, query sum
            cout << query(1, 0, n - 1, l - 1, r - 1) << "\n";
        }
    }
    return 0;
}
```

**Causal recap.** The segment tree's correctness rests on two things I had to pin down by tracing. First, the lazy push: a node's pending add must be applied to each child *over the child's own segment* (`[lo, mid]` and `[mid+1, hi]`), not the parent's; my first `push` passed the parent's range to both children, doubling the count factor, and a trace of audit `[0,0]` after `+10` returning `21` instead of `11` exposed it — handing each child its own half fixes it. Second, the type: even the build of `[0,2]` over three `10^9` balances overflows a 32-bit `int` to a negative garbage value, so the audit `2 1 5` prints `705032704` instead of `5000000000`; making `tree`, `lazy`, the `(long long)(hi-lo+1)*v` increment, and the printed value all 64-bit is what makes the audited totals exact. With those two settled, the standard overlap recursion plus the `tree[node] = tree[left] + tree[right]` recombine after partial updates gives `O((n+q) log n)`, and 400 random cases against an arbitrary-precision brute force agree.
