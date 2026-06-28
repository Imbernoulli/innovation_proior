# Wavelet tree (value-bisection + per-level rank bitmaps)

## Problem

For a static array of $n$ integers, answer two kinds of query on a window of
positions `a[l..r]` (1-based): the $k$-th smallest value in the window, and the
number of window elements that are $\le x$ (range rank). Values may be negative and
may repeat; $n, q$ up to $\sim 10^5$.

## Key idea

**Recursively bisect the value range, not the positions.** Compress values to dense
codes in $[0, \sigma)$. The root owns the whole code range $[0, \sigma)$ and the whole
array in original order. Bisect at $m = \lfloor(\text{lo}+\text{hi})/2\rfloor$: mark
each element with a bit — $0$ if its code $\le m$ (low half), $1$ if $> m$ (high
half) — and **stably partition** the sequence so the low (bit-0) elements, in order,
form the left child (range $[\text{lo}, m]$) and the high elements form the right
child (range $[m+1, \text{hi}]$). Recurse; a node whose range is a single code is a
leaf. Each level is a permutation of all $n$ elements, depth is $O(\log \sigma)$.

**Store only the prefix-rank of the split bitmap.** At each node keep
$z[0..\text{len}]$ with $z[i]$ = number of bit-$0$ (low) elements among the first $i$
elements of the node's sequence. Because the stable partition preserves order, the
prefix count is exactly the position map into the children: an element at node
position $p$ that is low lands at left position $z(p)$; if high, at right position
$p - z(p)$. So a contiguous half-open slice $[l, r)$ of a node maps to a contiguous
slice of each child:

$$\text{left: } [\,z(l),\ z(r)\,), \qquad \text{right: } [\,l - z(l),\ r - z(r)\,).$$

Both endpoints go through the *same* prefix function; the half-open convention
handles the off-by-one with no $\pm1$ correction. The left slice's length
$z(r) - z(l)$ is exactly how many slice elements fell in the low half.

**$k$-th smallest** descends from the root on half-open ranges. At a node with slice
$[\text{lo}, \text{hi})$, let $\text{num\_left} = z(\text{hi}) - z(\text{lo})$. If
$k \le \text{num\_left}$ the answer is in the low half: go left with range
$[z(\text{lo}), z(\text{hi}))$, same $k$. Otherwise it is in the high half: set
$k \mathrel{-}= \text{num\_left}$ and go right with range
$[\text{lo} - z(\text{lo}), \text{hi} - z(\text{hi}))$. The leaf's single code is the
answer; map it back to a value. $O(\log \sigma)$.

**Range rank** (count $\le x$) reuses the same mapping. With $x_c$ = number of codes
$\le x$, the qualifying elements are those with code in $[0, x_c)$ — a prefix of the
value domain. Recurse from the root: if a node's range is wholly below the threshold
($\text{hi} < x_c$) add the whole slice length; if wholly at/above it
($x_c \le \text{lo}$) add $0$; otherwise split the slice into both children by the
prefix map and sum. The threshold follows one root-to-leaf path, so at most two nodes
per level are split — $O(\log \sigma)$.

## Algorithm

1. Compress values to dense codes in $[0, \sigma)$ via the sorted distinct values.
2. Build by value-bisection: at each node bisect the code range, scan the sequence to
   build the prefix-zeros array while stably splitting into the low/high child
   sequences; recurse to leaves.
3. `kth(l, r, k)`: iterative left/right descent comparing $k$ to the low count.
4. `rank_leq(l, r, x)`: threshold-prefix recursion summing child counts.

## Code

```cpp
// Reads: "n q", then n integers, then q queries. A type-1 query "1 l r k" asks the
// k-th smallest value in a[l..r] (1-based l,r,k); a type-2 query "2 l r x" asks the
// number of elements in a[l..r] with value <= x. Prints one answer per query.
#include <bits/stdc++.h>
using namespace std;

struct Wavelet {
    int n, sigma, root;
    vector<long long> vals;            // sorted distinct values
    vector<int> lo, hi, lc, rc;        // per-node value interval and child ids
    vector<vector<int>> pz;            // pz[node] = prefix-zeros array

    int newNode(int vlo, int vhi) {
        lo.push_back(vlo); hi.push_back(vhi);
        lc.push_back(-1); rc.push_back(-1);
        pz.emplace_back();
        return (int)lo.size() - 1;
    }

    int buildRec(vector<int>& seq, int vlo, int vhi) {
        int node = newNode(vlo, vhi);
        if (vlo == vhi) {                       // leaf: one value, no split
            pz[node].assign(seq.size() + 1, 0);
            return node;
        }
        int mid = (vlo + vhi) >> 1;             // bisect the value range
        vector<int> z(seq.size() + 1, 0);       // prefix count of low elements
        vector<int> left, right;                // stable partition into the two halves
        for (size_t i = 0; i < seq.size(); ++i) {
            int c = seq[i];
            if (c <= mid) {                     // low half -> bit 0, goes left
                z[i + 1] = z[i] + 1;
                left.push_back(c);
            } else {                            // high half -> bit 1, goes right
                z[i + 1] = z[i];
                right.push_back(c);
            }
        }
        pz[node] = move(z);
        int lcId = buildRec(left, vlo, mid);
        int rcId = buildRec(right, mid + 1, vhi);
        lc[node] = lcId; rc[node] = rcId;
        return node;
    }

    Wavelet(const vector<long long>& a) {
        n = (int)a.size();
        vals = a;
        sort(vals.begin(), vals.end());
        vals.erase(unique(vals.begin(), vals.end()), vals.end());
        sigma = (int)vals.size();
        root = -1;
        if (n == 0) {                           // empty array: single trivial leaf
            root = newNode(0, 0);
            pz[root].assign(1, 0);
            return;
        }
        vector<int> codes(n);
        for (int i = 0; i < n; ++i)
            codes[i] = (int)(lower_bound(vals.begin(), vals.end(), a[i]) - vals.begin());
        root = buildRec(codes, 0, sigma - 1);
    }

    // k-th smallest value in a[l..r]; 1-based l,r,k.
    long long kth(int l, int r, int k) {
        int loP = l - 1, hiP = r;               // half-open slice [loP, hiP)
        int node = root;
        while (lo[node] != hi[node]) {          // until a single-value leaf
            const vector<int>& z = pz[node];
            int zl = z[loP], zr = z[hiP];
            int num_left = zr - zl;             // slice elements in the low half
            if (k <= num_left) {                // answer is low: map slice into left
                loP = zl; hiP = zr;
                node = lc[node];
            } else {                            // answer is high: drop the low count
                k -= num_left;
                loP = loP - zl; hiP = hiP - zr; // map slice into right
                node = rc[node];
            }
        }
        return vals[lo[node]];
    }

    // Number of elements in a[l..r] with value <= x; 1-based l,r.
    long long rank_leq(int l, int r, long long x) {
        // codes <= x are exactly [0, xc): xc = number of distinct values <= x
        int xc = (int)(upper_bound(vals.begin(), vals.end(), x) - vals.begin());
        return rankRec(root, l - 1, r, xc);
    }

    long long rankRec(int node, int loP, int hiP, int xc) {
        if (hiP <= loP || xc <= lo[node]) return 0;   // empty slice / node wholly above x
        if (hi[node] < xc) return hiP - loP;          // node wholly at or below x
        const vector<int>& z = pz[node];
        int zl = z[loP], zr = z[hiP];                 // threshold splits this node
        return rankRec(lc[node], zl, zr, xc)
             + rankRec(rc[node], loP - zl, hiP - zr, xc);
    }
};

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, q;
    if (!(cin >> n >> q)) return 0;
    vector<long long> a(n);
    for (int i = 0; i < n; ++i) cin >> a[i];
    Wavelet w(a);
    string out;
    for (int i = 0; i < q; ++i) {
        int t; cin >> t;
        if (t == 1) {
            int l, r, k; cin >> l >> r >> k;
            out += to_string(w.kth(l, r, k));
        } else {
            int l, r; long long x; cin >> l >> r >> x;
            out += to_string(w.rank_leq(l, r, x));
        }
        out += '\n';
    }
    cout << out;
    return 0;
}
```

## Complexity

- **Build:** $O(n \log \sigma)$ time and memory — $O(\log \sigma)$ levels, each a
  linear pass over a permutation of all $n$ elements; per node a prefix-zeros array.
- **Query:** $O(\log \sigma)$ per `kth` (one root-to-leaf descent) and per `rank_leq`
  (the threshold splits at most two nodes per level). With value compression,
  $\sigma \le n$, so both are $O(\log n)$.
