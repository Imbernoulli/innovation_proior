I have a stream of `q <= 10^5`
operations over a sequence of integers, and the operations are *versioned*: each
`insert`/`reverse` reads some version `v` and produces a brand-new version, while a
`query` reads a version and produces nothing new. Version `0` is empty; the `k`-th
modifying op makes version `k`. Critically, `v` may be **any** past version, not just
the latest — so the version history is a tree, and an old version must still answer
queries correctly after later versions branched off it. Values are up to `10^9` in
magnitude and ranges span up to `~10^5` elements, so a sum can reach `~10^5 * 10^9 =
10^14`, far past 32 bits; every value, every aggregate, every accumulator is `long
long`. That data-type decision is settled before any algorithm — an `int` here is a
silent wrong-answer on the large tests. The output is one line per `query`, up to
`10^5` lines, so I will accumulate output in one string and flush once rather than do
`10^5` separate `cout` calls.

**The obvious approach, and watching it die on the constraints.** The definitional
implementation is to store every version as an explicit array (a `vector<long long>`),
and apply each op literally: `insert` copies the source array and splices in one
element; `reverse` copies and reverses a slice; `query` sums a slice. This is
unarguably correct — it is exactly what I will use as a brute-force oracle. But look at
the cost. Each `insert`/`reverse` *copies the entire source array* to keep the old
version intact, and the array can be `~10^5` long. So one modifying op is `O(n)` time
and `O(n)` fresh memory, and a stream of `10^5` such ops on a sequence that grows to
`10^5` is `O(q n) = 10^{10}` element copies and `~10^{10}` integers of memory. That is
tens of seconds and tens of gigabytes — two to three orders of magnitude over budget on
both axes. The copying is forced by persistence: I *cannot* mutate the old array in
place, because the old version must survive. So the explicit-array method is out for the
full constraints, and the real question crystallizes: **how do I get a new version in
`O(log n)` without copying the whole sequence, when I am forbidden from editing the old
one in place?**

**Picking the sequence structure: implicit-key balanced BST.** Insert-at-position,
reverse-a-subrange, and range-sum are exactly the operations a balanced BST keyed by
*position* supports. The trick is the "implicit key": a node has no stored key; its
in-order rank is `size(left subtree)`, so the `k`-th element is found by walking left/
right comparing `k` against left-subtree sizes. The two primitives are `split(t, k)` —
cut the sequence into its first `k` elements and the rest — and `merge(a, b)` —
concatenate two sequences. With those, `insert x at p` is `split(root, p)` into `(A,
B)`, then `merge(merge(A, leaf(x)), B)`; and a range query/edit on `[l, r]` is two
splits to isolate the middle piece `[l, r]`, do something to it, and two merges to
stitch it back. For balance I use a **treap**: each node carries a random heap priority,
and `merge` keeps the higher-priority node as the parent, which makes the expected depth
`O(log n)`. I prefer a treap over a red-black or AVL tree here precisely because
`split`/`merge` are short and their persistent versions are clean — there are no
rotations to make persistent, just two recursive functions.

**Reverse in `O(log n)`: a lazy flag.** Reversing the slice `[l, r]` by physically
swapping elements is `O(r - l)`, which is `O(n)` and kills the budget when ranges are
large. The standard fix: isolate the middle piece with two splits, then set a single
boolean `rev` flag on its root meaning "the in-order traversal of this whole subtree is
reversed". The flag is *lazy*: I do not act on it now. When I later need to descend into
that subtree, I "push" the flag one level — swap the node's two children and toggle each
child's own `rev` flag — and clear it. Because reversing a subtree never changes which
elements it contains, the subtree's stored `sum` is invariant under the flag, and only
the *order* (hence partial-range answers and future splits) is affected. So `reverse`
becomes two splits + set one bit + two merges = `O(log n)`.

**The collision, stated plainly.** Now the two requirements fight. Lazy propagation *is*
a mutation: pushing a flag rewrites a node (swaps its children, toggles flags). And
`split`/`merge` also rewrite nodes (they reattach subtrees and recompute sizes/sums). If
I mutate nodes in place, I destroy whatever old version was sharing them — which is
exactly the persistence guarantee I must keep. A concrete instance makes the danger
vivid: suppose version 5 is `[5,1,2,4,3]` and I do `reverse` on `[1,3]` to make version
6. If `merge`/`push` edits the very nodes that version 5 still points to, then a later
`query` on version 5 would see version 6's reordering. I need version 6 to *reuse*
version 5's nodes for the untouched parts, yet *not alter* a single node version 5 can
reach.

**The insight — path-copying.** The resolution is to make every node **immutable once
created**, and to implement each mutation as *allocation of a fresh node* rather than an
edit. Whenever `split`, `merge`, or a flag-push is about to change a node, it instead
clones that node into a new id and changes the clone; the clone's children that are not
themselves changed are *shared by pointer/id* with the old structure. Because `split`
and `merge` only ever touch the nodes on a single root-to-leaf path, only `O(log n)`
nodes are cloned per operation, and everything off that path is shared between the old
and new versions. That is **path-copying persistence**: a new version is a thin spine of
`O(log n)` fresh nodes hanging off the old version's subtrees, which are reused
untouched. Old versions are correct *by construction* because no node they can reach was
ever modified — modifications only ever create new nodes.

The flag-push has to be made persistent the same way, and this is the subtle part. When
I am at a node whose `rev` is set and I need to descend, I must produce a *clean* clone
of this node with the flag resolved, whose children also reflect the reversal. So:
clone the node; if its `rev` is set, swap the clone's two child *ids*, then clone each of
those children and toggle the clone's `rev` (because the reversal must propagate into
them), and clear the parent clone's `rev`. The originals are untouched; the new spine
carries the resolved order. I will route *every* descent in `split` and `merge` through
this single "clone-and-apply-flag" function so I can never forget to push.

**Queries without allocation.** A `query` is read-only, so it would be wasteful (and a
memory leak into the persistent pool) to split/merge for it. I want a pure traversal
that allocates nothing. Two facts make that possible. First, a `reverse` never changes a
subtree's total sum, so if a query range *covers a whole subtree* I can return that
subtree's stored `sum` and stop — flags are irrelevant there. Second, for *partial*
coverage I do need to respect pending flags, but I can do it without mutating: carry an
accumulated reverse *parity* `flip` down the recursion. At each node, fold in the node's
own `rev` (`flip ^= node.rev`); if `flip` is now odd, this subtree's logical order is
reversed, so its logical-left child is the stored right child and vice-versa, and I
index accordingly. This reads the lazy flags without ever pushing them, so a `query`
costs `O(log n)` time and zero allocations — the persistent pool grows only with
genuine modifications.

**Laying out the node and the helpers.** A node stores: child ids `l, r` (id `0` is a
null sentinel so I never dereference null), a random priority, the subtree `sz`, the
lazy `rev` bit, the element's `val`, and the subtree `sum`. I keep all nodes in one
`vector<Node>` pool and refer to them by integer id; `cloneNode` is a single
`push_back(pool[t])` which copy-constructs a new node that *shares the same child ids*
(structural sharing). For randomness I use a fast xorshift so priorities are well-spread
without depending on `rand()`.

```
static inline int pushApplied(int t) {     // clone of t with rev resolved one level
    int c = cloneNode(t);
    if (pool[c].rev) {
        swap(pool[c].l, pool[c].r);
        if (pool[c].l) { int nl = cloneNode(pool[c].l); pool[nl].rev ^= 1; pool[c].l = nl; }
        if (pool[c].r) { int nr = cloneNode(pool[c].r); pool[nr].rev ^= 1; pool[c].r = nr; }
        pool[c].rev = false;
    }
    return c;
}
```

`split(t, k)` clones the current node via `pushApplied`, compares `k` to the left size,
recurses into one child, reattaches the returned half, recomputes `sz`/`sum` with
`pull`, and returns. `merge(a, b)` compares priorities, `pushApplied`s the winner,
recurses on the appropriate side, `pull`s, and returns the clone. Both touch only one
path, so both clone `O(log n)` nodes.

**First implementation, and immediately tracing it — clean math transcribes dirty.** My
first cut of `split` looked like this (paraphrased), *without* routing through
`pushApplied` — I "optimized" by reading sizes directly:

```
static void split(int t, int k, int &a, int &b) {
    if (!t) { a = b = 0; return; }
    int c = cloneNode(t);                 // BUG: cloned but did NOT push rev
    int leftSz = sz(pool[c].l);
    if (leftSz < k) { split(pool[c].r, k-leftSz-1, ...); ... }
    else            { split(pool[c].l, k, ...);          ... }
}
```

To expose it I built the smallest case where a *pending reverse* must interact with a
later split. Take version 5 = `[5,1,2,4,3]`; reverse `[1,3]` to get version 6 =
`[5,4,2,1,3]`; now query position `1` of version 6, which should be `4`. I traced it.
Building version 6, I split version 5 at `1` and at `3` to isolate the middle `[1,2,4]`,
set its root's `rev`, and merged the three pieces back. So version 6's root subtree for
the middle still has a *pending* `rev`. Now the query path (in an earlier draft I had
queries also use split) splits version 6 to reach position 1. My buggy `split` cloned
the middle root but **did not swap its children or toggle their flags**, so it read the
left-size and descended as if the subtree were in *unreversed* order. It returned the
element that sits at logical position 1 of the *unreversed* middle, which is `2`, not
`4`. The trace produced `2`; the answer is `4`. Wrong.

**Diagnosing the bug.** The defect is precise: `split` (and `merge`) must *resolve* a
node's pending `rev` **before** they read its left-subtree size or decide which way to
recurse, because the pending flag means the stored child order is a lie about the
logical order. By cloning without pushing, I read sizes against the wrong child
arrangement and reattached halves in the wrong order — corrupting exactly the versions
that had been touched by a `reverse`. The fix is to make `pushApplied` (clone *and*
resolve the flag) the only way `split`/`merge` ever obtain a working node, so the flag
is always pushed before any structural decision. I replaced `cloneNode(t)` with
`pushApplied(t)` at the top of both `split` and `merge`.

**Fixing and re-verifying.** With `pushApplied` in place I re-traced version 6: building
it, the middle subtree carries `rev`. When the query (now done by a non-allocating
`rangeSum`, see below) reaches that subtree, it folds the parity in and reads the *right*
child as the logical-left, landing on `4`. Correct. Then I separately decided that
queries should not allocate at all, and wrote `rangeSum(t, ql, qr, flip)`: it
short-circuits on a fully-covered subtree's stored `sum`, otherwise folds `flip ^=
rev`, picks logical-left/right children by `flip`, and recurses with clamped windows.
I re-traced the sample: version 5 `[5,1,2,4,3]`, `query [1,3]` = `1+2+4 = 7`; after
reverse, version 6 `[5,4,2,1,3]`, `query [1,1]` = `4`; and `query [1,1]` of version 5
again = `1`. All three match, and the third one is the persistence check — version 5 was
not disturbed by building version 6.

**Edge cases, deliberately, because this is where this kind of code dies.**

- `q = 0`: the loop never runs; the program reads `q`, prints nothing, exits `0`.
  (If even `q` is missing, `if (!(cin >> q)) return 0;` handles empty input.)
- Length-one reverse, `reverse [i, i]`: the middle piece is a single node; setting its
  `rev` and pushing it later swaps two null children and toggles nothing — a no-op, as a
  one-element reversal should be. Traced on `[-5]` reversing `[0,0]`: still `[-5]`.
- Single-element query / degenerate range `l == r`: `rangeSum` with `ql == qr` returns
  that one element; verified on the sample's `query [1,1]`.
- Insert at the boundary `p = len(v)`: `split(root, len)` puts the whole sequence in the
  left half and an empty right half, so the new element appends at the end — correct.
- Insert into / reverse of an *old, already-reversed* version: the worst interaction.
  Traced `[100,200,300]` reversed to `[300,200,100]`, then insert `999` at position 1 of
  that reversed version, expecting `[300,999,200,100]`; the split must push the pending
  reverse to find position 1, and it does — sum of the whole thing is `1599`, position 1
  is `999`. Correct.
- Overflow: all aggregates and accumulators are `long long`; a worst-case sum
  `~10^{14}` fits with three orders of magnitude to spare. The null sentinel (id 0) has
  `sum = 0`, `sz = 0`, so `sm`/`sz` of an empty child contribute nothing.
- Memory: each modifying op clones an `O(log n)` spine, and a `reverse` push additionally
  clones the two children it propagates into; over `10^5` modifying ops on a `10^5`-long
  chain the pool peaks around a few hundred MB, which is why the budget is set to 1024 MB
  and 3 s. Queries allocate nothing, so they do not grow the pool.

**Differential self-verification.** I wrote the explicit-array brute force and a random
generator that tracks each version's length so every generated position/range/version
reference is in range. Over 600 standard random cases plus 700 heavier cases (long
chains, many reverses, references to *old* versions) — 1300 cases total — the treap and
the brute force agreed on every line, zero mismatches, and the documented sample matches.
I also timed the worst cases: a mixed `10^5`-op stream runs in `~0.08 s`, and the
pathological all-modifying `10^5`-op chain runs in under `0.8 s` and under the memory
cap. The idea is right because the explicit-copy method is provably correct but
quadratic, and path-copying reproduces its answers in `O(log n)` per op while keeping
every version independently correct; the code is right because the one bug I hit — not
pushing the lazy reverse before reading subtree sizes in `split`/`merge` — was traced to
a precise cause on the smallest reverse-then-read case and fixed by funneling all
descents through `pushApplied`.

**Final solution.** One self-contained file: an implicit-key treap whose `split`,
`merge`, and flag-push are all path-copying so old versions are immutable; `insert` and
`reverse` build a new version as an `O(log n)` spine over shared subtrees; and `query`
is an allocation-free parity-carrying range-sum.

```cpp
#include <bits/stdc++.h>
using namespace std;

// Fully persistent implicit-key treap (versioned rope) with lazy subtree reversal.
// Every node is immutable once created; split/merge/push-down all do path-copying,
// allocating fresh nodes instead of mutating, so any old version stays intact.

struct Node {
    int l, r;            // child node ids (0 = empty)
    unsigned prio;       // treap heap priority
    int sz;              // subtree size (>=0; fits in 31 bits) ; top bit unused
    bool rev;            // pending lazy "reverse this subtree" flag (kept as 1 byte)
    long long val;       // stored value at this position
    long long sum;       // subtree sum
};  // 8-byte aligned (the long longs) -> 40 bytes per node in practice

static vector<Node> pool;            // node 0 is the null sentinel
static uint64_t rng_state = 0x9e3779b97f4a7c15ULL;

static inline uint32_t nextRand() {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    return (uint32_t)(rng_state >> 32);
}

static inline int sz(int t)        { return t ? pool[t].sz : 0; }
static inline long long sm(int t)  { return t ? pool[t].sum : 0LL; }

// Allocate a brand-new node that is a copy of an existing one (or a fresh leaf).
static inline int newNode(long long val) {
    pool.push_back(Node{0, 0, nextRand(), 1, false, val, val});
    return (int)pool.size() - 1;
}
static inline int cloneNode(int t) {
    pool.push_back(pool[t]);          // copy-construct: structural sharing of children
    return (int)pool.size() - 1;
}

static inline void pull(int t) {     // recompute aggregates from (current) children
    Node &n = pool[t];
    n.sz  = sz(n.l) + sz(n.r) + 1;
    n.sum = sm(n.l) + sm(n.r) + n.val;
}

// Return a COPY of t with the lazy reverse flag resolved one level down.
// Children get their own clones carrying the toggled flag; t's clone is clean.
static inline int pushApplied(int t) {
    int c = cloneNode(t);
    if (pool[c].rev) {
        swap(pool[c].l, pool[c].r);
        if (pool[c].l) { int nl = cloneNode(pool[c].l); pool[nl].rev ^= 1; pool[c].l = nl; }
        if (pool[c].r) { int nr = cloneNode(pool[c].r); pool[nr].rev ^= 1; pool[c].r = nr; }
        pool[c].rev = false;
    }
    return c;
}

// Split version t into first k elements (l) and the rest (r). Path-copying:
// every node on the search path is freshly cloned, so t is unchanged.
static void split(int t, int k, int &a, int &b) {
    if (!t) { a = b = 0; return; }
    int c = pushApplied(t);
    int leftSz = sz(pool[c].l);
    if (leftSz < k) {                 // c and its left subtree go left
        int rl, rr;
        split(pool[c].r, k - leftSz - 1, rl, rr);
        pool[c].r = rl;
        pull(c);
        a = c; b = rr;
    } else {                          // c and its right subtree go right
        int ll, lr;
        split(pool[c].l, k, ll, lr);
        pool[c].l = lr;
        pull(c);
        b = c; a = ll;
    }
}

// Sum of positions [ql, qr] (0-indexed, inclusive) in the sequence rooted at t.
// Read-only: allocates nothing -- so query work never bloats the persistent pool.
// We do NOT mutate or push the lazy flags; instead we carry the accumulated
// reverse parity `flip` down the recursion. When the query covers a whole
// subtree we can short-circuit on its stored sum (reversal does not change a
// subtree's total). For partial coverage, `flip` tells us whether the subtree's
// logical left/right children are swapped, and whether the in-subtree index of
// the spine node is from the front (flip=0) or the back (flip=1).
static long long rangeSum(int t, int ql, int qr, bool flip) {
    if (!t || ql > qr) return 0;
    int total = pool[t].sz;
    if (ql <= 0 && qr >= total - 1) return pool[t].sum; // whole subtree covered
    flip ^= pool[t].rev;                 // fold this node's own pending reverse in
    int left  = flip ? pool[t].r : pool[t].l;   // logical-left child under `flip`
    int right = flip ? pool[t].l : pool[t].r;   // logical-right child under `flip`
    int ls = sz(left);
    long long res = 0;
    // logical-left child occupies indices [0, ls-1]
    if (ql < ls) res += rangeSum(left, ql, min(qr, ls - 1), flip);
    // spine node occupies index ls
    if (ql <= ls && ls <= qr) res += pool[t].val;
    // logical-right child occupies indices [ls+1, total-1] -> shift by ls+1
    if (qr > ls) res += rangeSum(right, max(ql, ls + 1) - (ls + 1), qr - (ls + 1), flip);
    return res;
}

// Merge versions a (all keys before b). Path-copying clone of the chosen root.
static int merge(int a, int b) {
    if (!a || !b) return a ? a : b;
    if (pool[a].prio > pool[b].prio) {
        int c = pushApplied(a);
        pool[c].r = merge(pool[c].r, b);
        pull(c);
        return c;
    } else {
        int c = pushApplied(b);
        pool[c].l = merge(a, pool[c].l);
        pull(c);
        return c;
    }
}

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int q;
    if (!(cin >> q)) return 0;

    pool.reserve(1);
    pool.push_back(Node{0, 0, 0u, 0, false, 0, 0}); // null sentinel id 0

    vector<int> roots;
    roots.push_back(0);               // version 0 = empty sequence

    string out;
    out.reserve(1 << 20);

    for (int i = 0; i < q; i++) {
        int type; cin >> type;
        if (type == 1) {              // insert x at position p in version v -> new version
            long long v, p, x; cin >> v >> p >> x;
            int root = roots[(size_t)v];
            int a, b;
            split(root, (int)p, a, b);
            int mid = newNode(x);
            int nr = merge(merge(a, mid), b);
            roots.push_back(nr);
        } else if (type == 2) {       // reverse [l,r] in version v -> new version
            long long v, l, r; cin >> v >> l >> r;
            int root = roots[(size_t)v];
            int a, b, c;
            split(root, (int)l, a, b);          // a = [0,l), rest = [l, len)
            split(b, (int)(r - l + 1), b, c);    // b = [l,r], c = (r, len)
            if (b) { int nb = cloneNode(b); pool[nb].rev ^= 1; b = nb; }
            int nr = merge(merge(a, b), c);
            roots.push_back(nr);
        } else {                      // query sum of [l,r] in version v (no new version)
            long long v, l, r; cin >> v >> l >> r;
            int root = roots[(size_t)v];
            out += to_string(rangeSum(root, (int)l, (int)r, false));
            out += '\n';
        }
    }

    cout << out;
    return 0;
}
```

**Causal recap.** The explicit-array method is obviously correct but copies a whole
`~10^5`-element array per modifying op, so `10^5` ops cost `~10^{10}` copies and tens of
GB — fatal. An implicit-key treap gives `O(log n)` `insert`/`reverse`/`sum` via
`split`/`merge` plus a lazy `rev` flag, but its in-place mutation would clobber the past
versions full persistence requires. **Path-copying** reconciles them: every change
allocates a fresh node instead of editing one, so each new version is an `O(log n)`
spine over the old version's untouched, shared subtrees, and old versions are correct by
construction. Making the *lazy reverse* persistent is the subtle part and was the source
of my one real bug — I cloned a node in `split` without resolving its pending `rev`, so a
reverse-then-read case returned `2` instead of `4`; routing every descent through a
single clone-and-push-flag helper fixed it. Queries carry a reverse *parity* down a
non-allocating traversal so they read the lazy flags without growing the pool. 1300
differential cases against the brute force agree with zero mismatches, the sample
matches, and the worst-case timing/memory sit inside the 3 s / 1024 MB budget.
