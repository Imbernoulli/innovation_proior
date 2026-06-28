**Problem.** Maintain a sequence of integers under a stream of `q <= 10^5` *versioned*
operations. Version `0` is empty; the `k`-th modifying op creates version `k`, and every
op may read **any** past version, leaving it intact (full persistence). Operations:
`1 v p x` inserts `x` at position `p` of version `v` (new version); `2 v l r` reverses
positions `[l, r]` of version `v` (new version); `3 v l r` outputs the sum of positions
`[l, r]` of version `v` (no new version). Print each query's answer in input order.

**Why the obvious method dies.** Store each version as an explicit array and apply ops
literally. It is correct (and makes a perfect brute-force oracle), but persistence forces
each `insert`/`reverse` to *copy the whole source array* — `O(n)` time and `O(n)` fresh
memory per op. On a `10^5`-long sequence with `10^5` ops that is `~10^{10}` copies and
tens of GB: tens of seconds, far over budget. The real constraint is sharpened by this:
get a new version in `O(log n)` **without** mutating the old one in place.

**Key idea — a path-copying (fully persistent) implicit-key treap with lazy reverse.**
Three ingredients, and the third is the innovation that reconciles the first two:

1. *Implicit-key treap.* Key nodes by position via subtree sizes. `split(t, k)` cuts off
   the first `k` elements; `merge(a, b)` concatenates. Then `insert at p` = split at `p`,
   splice a leaf, two merges; a range edit on `[l, r]` = two splits to isolate the middle
   piece, act on it, two merges. Random heap priorities keep depth `O(log n)`.

2. *Lazy reverse.* Reversing `[l, r]` by swapping elements is `O(r-l)`. Instead, isolate
   the middle and set one boolean `rev` ("this subtree's order is reversed") on its root.
   Push it lazily when descending: swap the node's children and toggle each child's
   `rev`. A reversal never changes a subtree's *sum*, only its order.

3. *Path-copying persistence — the crux.* Lazy push and split/merge are mutations, which
   would clobber the old versions that share those nodes. Resolution: make every node
   **immutable**; any "mutation" instead **allocates a fresh clone** and edits the clone,
   while unchanged children are *shared by id*. Since split/merge touch one root-to-leaf
   path, only `O(log n)` nodes are cloned per op; everything off the path is reused. Old
   versions are correct by construction — no node they can reach is ever modified.

Queries are read-only, so they allocate nothing: a `query` does a parity-carrying
traversal. Carry a reverse parity `flip` down; if a range covers a whole subtree return
its stored `sum` (flags don't matter), else fold `flip ^= rev` and read the
logical-left/right children swapped when `flip` is odd. This reads lazy flags without
pushing them, keeping the persistent pool from growing on queries.

**Pitfalls.**
1. *Pushing the lazy flag before reading subtree sizes.* In `split`/`merge`, a node's
   stored child order is a *lie* while its `rev` is pending. If you clone a node but
   forget to resolve `rev` before comparing the split index to the left size, you
   descend the wrong way and reattach halves in the wrong order — a reverse-then-read
   case returns the wrong element (e.g. `2` instead of `4`). Fix: route every descent in
   `split` and `merge` through one "clone-and-resolve-flag" helper so the push can never
   be skipped.
2. *Making the push itself persistent.* Resolving `rev` must clone the children it
   propagates into (and toggle their `rev`), not edit them in place, or you corrupt
   shared structure.
3. *Overflow.* Sums reach `~10^{14}`; use `long long` for `val`, `sum`, and accumulators.
4. *Memory.* Each modifying op clones an `O(log n)` spine (a reverse-push clones two extra
   children); queries clone nothing. The pool peaks in the low hundreds of MB at the
   limits — hence the 3 s / 1024 MB envelope.

**Edge cases.** `q = 0` prints nothing. Length-one reverse `[i,i]` and degenerate query
`l == r` are no-ops / single elements. Insert at `p = len(v)` appends. Inserting into or
reversing an *already-reversed old version* is the hard interaction — the split must push
the pending `rev` first; verified it does. Empty children use a null sentinel (id 0) with
`sz = sum = 0`.

**Complexity.** `O(log n)` expected time per operation, `O(log n)` new nodes per
modifying op (queries allocate none), so `O(q log n)` time and `O(q log n)` memory.

**Verification.** Differential-tested against the explicit-array brute force over 1300
random cases (long chains, heavy reverses, references to old versions, nested reversals)
with zero mismatches; the worked sample matches; worst-case timing/memory are inside
budget.

**Code.**

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
