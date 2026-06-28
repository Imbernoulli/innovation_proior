# Size Balanced Tree

## Problem

Maintain a dynamic ordered set of distinct keys under insertion, supporting the
order statistics $\mathrm{Select}(k)$ — the $k$-th smallest key — and
$\mathrm{Rank}(v)$ — one plus the number of keys strictly less than $v$ — while
keeping the binary search tree balanced cheaply.

## Key idea

Every node already stores its subtree size $s[t] = s[\mathrm{left}[t]] +
s[\mathrm{right}[t]] + 1$ (with $s[0]=0$), because that is exactly the field
$\mathrm{Select}$ and $\mathrm{Rank}$ read. The same field becomes the balance
criterion, so no height/color/priority is stored. The **invariant** is that each
child's subtree is at least as large as either of its nephew subtrees: for every
node $t$,

$$s[\mathrm{right}[t]] \ge s[\mathrm{left}[\mathrm{left}[t]]],\ s[\mathrm{right}[\mathrm{left}[t]]]
\qquad\text{(a)}$$
$$s[\mathrm{left}[t]] \ge s[\mathrm{right}[\mathrm{right}[t]]],\ s[\mathrm{left}[\mathrm{right}[t]]]
\qquad\text{(b)}$$

Insertion is the ordinary recursive BST insert (incrementing $s$ down the path),
followed by one call to `_rebalance`, which restores (a)/(b) with rotations:

- **Outer-heavy** nephew (e.g. $s[\mathrm{left}[\mathrm{left}[t]]] > s[\mathrm{right}[t]]$):
  a single rotation (right-rotate $t$).
- **Inner-heavy** nephew (e.g. $s[\mathrm{right}[\mathrm{left}[t]]] > s[\mathrm{right}[t]]$):
  a double rotation (left-rotate $\mathrm{left}[t]$, then right-rotate $t$).

After rotating, `_rebalance` recurses on the touched nodes. Using a `flag` to test only
the side that could have broken (left-heavy when the insert descended left, else
right-heavy), the four follow-up calls
`_rebalance(left[t], False)`,
`_rebalance(right[t], True)`,
`_rebalance(t, False)`, `_rebalance(t, True)` suffice —
the calls `_rebalance(left[t], True)` and
`_rebalance(right[t], False)` are provably no-ops (from
$s[L] \le 2s[R]+1$ one derives $s[E],s[F] \le \lfloor (8s[R]+3)/9 \rfloor \le s[R]$).

## Why it is balanced

Let $f[h]$ be the minimum node count of a size-balanced tree of height $h$. A height-$h$ tree has a
height-$(h-1)$ child (size $\ge f[h-1]$) which itself owns a height-$(h-2)$ subtree
(size $\ge f[h-2]$); by the nephew condition the other child is $\ge f[h-2]$, so

$$f[0]=1,\quad f[1]=2,\quad f[h]=f[h-1]+f[h-2]+1\ (h>1),$$

a Fibonacci-like recurrence with $f[h] = F_{h+3} - 1$. By Binet's formula,
$F_m=(\alpha^m-\beta^m)/\sqrt5$, where $\alpha = (1+\sqrt5)/2$ and
$|\beta|<1$, so $F_m$ is within $1/2$ of $\alpha^m/\sqrt5$. Inverting
$f[h] \le n$ gives worst height
$\le 1.44\log_2(n+1.5) - 1.33 = O(\log n)$ — the same bound as AVL, from the size
field alone.

**`_rebalance` is $O(1)$ amortized per invocation.** Let $SD$ be the sum
of all node depths. Every rotation `_rebalance` fires strictly decreases $SD$ (single
case: by $s[\mathrm{left}[\mathrm{left}[t]]] - s[\mathrm{right}[t]] \ge 1$; double
case: by $> 1$). Since the height is $O(\log n)$, $SD$ stays in an
$O(n\log n)$ band, and each insert raises $SD$ by only $O(\log n)$; over $n$
inserts the rotating `_rebalance` calls total $O(n\log n)$. The upward insertion paths
also make $O(n\log n)$ `_rebalance` calls, and each rotating call creates only a
constant number of follow-up calls, so total `_rebalance` work is $O(n\log n)$. Thus
each `_rebalance` invocation is $O(1)$ amortized, insertion is $O(\log n)$ amortized,
and Select/Rank are $O(\log n)$ worst case from the height bound.

## Code

Single-file C++17 reading from stdin. I/O contract: read `q` operations; each is
`I v` (insert key `v`), `S k` (print the `k`-th smallest stored key), or `R v`
(print one plus the number of stored keys strictly less than `v`). One line of
output per `S`/`R`. Keys are 64-bit.

```cpp
// Size Balanced Tree: an order-statistic balanced BST whose only per-node
// field, the subtree size s, both answers the queries and drives the rotations.
//
// I/O contract: reads q (number of operations); each subsequent line is an
// operation -- "I v" insert key v, "S k" print the k-th smallest stored key,
// "R v" print 1 + (number of stored keys strictly less than v). One answer per
// S/R query is written to stdout. Keys are 64-bit (long long, overflow-safe).
#include <bits/stdc++.h>
using namespace std;

// Array-based forest of nodes; index 0 is the null node with s[0] = 0.
vector<long long> key{0};                    // node key
vector<int> lc{0}, rc{0};                     // left / right child indices
vector<long long> s{0};                       // subtree size (the only extra field)
int root = 0;

int new_node(long long v) {                   // allocate a fresh leaf
    key.push_back(v); lc.push_back(0); rc.push_back(0); s.push_back(1);
    return (int)key.size() - 1;
}

int left_rotate(int t) {                      // pull right child up
    int k = rc[t];
    rc[t] = lc[k]; lc[k] = t;
    s[k] = s[t];                              // k inherits t's old subtree
    s[t] = s[lc[t]] + s[rc[t]] + 1;
    return k;
}

int right_rotate(int t) {                     // pull left child up
    int k = lc[t];
    lc[t] = rc[k]; rc[k] = t;
    s[k] = s[t];                              // k inherits t's old subtree
    s[t] = s[lc[t]] + s[rc[t]] + 1;
    return k;
}

int rebalance(int t, bool flag) {             // restore the nephew condition at t
    if (t == 0) return 0;
    if (!flag) {                              // left side may be too heavy
        if (s[lc[lc[t]]] > s[rc[t]]) {        // outer-left nephew -> single
            t = right_rotate(t);
        } else if (s[rc[lc[t]]] > s[rc[t]]) { // inner-left nephew -> double
            lc[t] = left_rotate(lc[t]); t = right_rotate(t);
        } else {
            return t;
        }
    } else {                                  // right side (mirror)
        if (s[rc[rc[t]]] > s[lc[t]]) {        // outer-right nephew -> single
            t = left_rotate(t);
        } else if (s[lc[rc[t]]] > s[lc[t]]) { // inner-right nephew -> double
            rc[t] = right_rotate(rc[t]); t = left_rotate(t);
        } else {
            return t;
        }
    }
    lc[t] = rebalance(lc[t], false);          // repair the touched nodes
    rc[t] = rebalance(rc[t], true);
    t = rebalance(t, false);
    t = rebalance(t, true);
    return t;
}

int insert_rec(int t, long long v) {          // BST insert + one rebalance
    if (t == 0) return new_node(v);
    s[t] += 1;
    if (v < key[t]) lc[t] = insert_rec(lc[t], v);
    else            rc[t] = insert_rec(rc[t], v);
    return rebalance(t, v >= key[t]);         // flag = which way we descended
}

void insert(long long v) { root = insert_rec(root, v); }

long long select_kth(long long k) {           // k-th smallest (1-indexed)
    int t = root;
    while (t) {
        long long r = s[lc[t]] + 1;           // rank of t in its own subtree
        if (k == r)      return key[t];
        else if (k < r)  t = lc[t];
        else           { k -= r; t = rc[t]; }
    }
    return 0;                                  // k out of range
}

long long rank_of(long long v) {              // 1 + #keys strictly < v
    int t = root; long long ans = 0;
    while (t) {
        if (v <= key[t]) {
            t = lc[t];
        } else {
            ans += s[lc[t]] + 1;              // bank left subtree + node
            t = rc[t];
        }
    }
    return ans + 1;
}

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);
    int q;
    if (!(cin >> q)) return 0;
    while (q--) {
        char op; long long x;
        cin >> op >> x;
        if (op == 'I')      insert(x);
        else if (op == 'S') cout << select_kth(x) << '\n';
        else if (op == 'R') cout << rank_of(x) << '\n';
    }
    return 0;
}
```

## Complexity

- **Height:** $\le 1.44\log_2(n+1.5) - 1.33 = O(\log n)$ (Fibonacci bound).
- **`_rebalance`:** $O(1)$ amortized per invocation ($O(n\log n)$ total
  work over the $O(n\log n)$ invocations made by $n$ inserts).
- **Insert:** $O(\log n)$ amortized.
- **Select / Rank:** $O(\log n)$ worst case.
- **Per-node state:** key, two child pointers, and the subtree size $s$ — and $s$
  is the field the order-statistics queries consume, so there is no balance field
  that the queries cannot use.
