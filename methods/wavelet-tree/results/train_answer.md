I have a fixed array of $n$ integers and a stream of $q$ queries, each asking one of two things about a contiguous window `a[l..r]` (1-based): the $k$-th smallest value in the window, or how many of its elements are $\le x$ (a range rank). Values may be negative, may repeat, and $n, q$ are both around $10^5$. The honest brute force is to do exactly what each query says — copy out the slice and sort it to read off position $k$, or scan the slice and tally the elements $\le x$. Both are correct, but the slice can be the entire array, so a single order-statistic query is an $O(n \log n)$ sort and a single count is an $O(n)$ scan, giving $O(q\,n\log n)$ overall — around $10^{10}$, hopelessly slow. The reason it is slow is structural: every query re-examines its slice from scratch, sorting the same overlapping stretches again and again and discarding the work. But the array is fixed and the slice is contiguous, so the right move is to preprocess once and answer each query by *navigating* a prebuilt structure rather than touching every element of the window. The naive 2D view — index the elements by both position and value and answer a dominance count (positions in a range, values below a bound) — exposes the real obstacle: the two axes pull against each other, because sorting by value destroys the position window and keeping position order forces value filtering at query time. What I need is a structure that splits on value while still tracking position.

I propose the wavelet tree, built by value-bisection with per-level rank bitmaps. The first observation is that both queries share one primitive: counting how many elements of a window carry a value below some threshold. That count *is* the rank query, and it also drives the order statistic, since order statistic and rank are inverses on a sorted multiset. The values themselves are a nuisance — huge, negative, sparse — and none of the queries care about magnitude, only relative order, so I compress: take the sorted distinct values and replace each $a_i$ by its index in that list, a dense code in $[0, \sigma)$ where $\sigma$ is the number of distinct values. Codes preserve order, "$\le x$" becomes "code in $[0, x_c)$" for a binary-searched threshold $x_c$, and the answer's code maps back to a value at the end.

The structure is a binary tree over the *value* range. The root owns the full code range $[0, \sigma)$ and the whole array in original left-to-right order. I bisect the value range at its midpoint $m = \lfloor(\text{lo}+\text{hi})/2\rfloor$, walk the node's sequence once, and label each element with a single bit: $0$ if its code $\le m$ (low half, goes left) and $1$ if its code $> m$ (high half, goes right). Then I form the two children by a *stable* partition — the low elements, in their original relative order, become the left child's sequence (owning value range $[\text{lo}, m]$), and the high elements, in order, become the right child's (owning $[m+1, \text{hi}]$). Recurse on each child with its narrower range; a node whose range is a single code is a leaf, under which every element shares that one value. Stability is load-bearing: by keeping each half in original order, the children are themselves valid position-ordered arrays, so the same construction recurses cleanly and position structure is never scrambled within a half. Each level holds every element exactly once (each element takes exactly one bit per split), so a level is a permutation of all $n$ elements and there are $O(\log \sigma)$ levels — $O(n\log\sigma)$ stored.

The crux is mapping a query window down the tree without ever tracking individual elements, which would cost $O(\text{slice length})$ again. The window is a contiguous range at a node, and I need it to map to a contiguous range at the child in $O(1)$. The stable partition makes this exact. Let $z(i)$ be the number of bit-$0$ (low) elements among the first $i$ elements of the node's sequence — a prefix count of zeros, with $z[0]=0$ and $z[i+1] = z[i] + (1-\text{bit}_i)$. Because the left child receives the $0$-elements in order, an element at node position $p$ that is low lands at left-child position exactly $z(p)$ (the count of lows strictly before it); a high element at position $p$ lands at right-child position $p - z(p)$ (the count of highs strictly before it). Pushing a half-open range $[l, r)$ through this, the $0$-elements of $[l,r)$ occupy left-child positions $[z(l), z(r))$ and the $1$-elements occupy right-child positions $[l - z(l), r - z(r))$:

$$\text{left: } [\,z(l),\ z(r)\,), \qquad \text{right: } [\,l - z(l),\ r - z(r)\,).$$

Both endpoints go through the *same* prefix function, and the half-open convention absorbs the off-by-one with no $\pm 1$ correction — closed ranges would force juggling "before $l$" against "up to and including $r$" and invite an off-by-one. The left slice's length $z(r)-z(l)$ is precisely how many window elements fell in the low half. So the only thing I store per node is this prefix-zeros array $z$; I never need the bits or child sequences themselves at query time, and I build $z$ in the same linear sweep that emits the two child sequences — one pass per node, $O(n\log\sigma)$ total.

With this map, the $k$-th smallest is a single root-to-leaf descent on half-open ranges. Convert the 1-based window to $[l-1, r)$ at the root. At a node with current range $[\text{lo}, \text{hi})$, let $\text{num\_left} = z(\text{hi}) - z(\text{lo})$ be the count of window elements in the low half. If $k \le \text{num\_left}$, the answer is a low value: descend left with range $[z(\text{lo}), z(\text{hi}))$ keeping the same $k$. Otherwise it is a high value: subtract the low count, $k \mathrel{-}= \text{num\_left}$, and descend right with range $[\text{lo}-z(\text{lo}), \text{hi}-z(\text{hi}))$. The leaf's single code is the answer, mapped back through the value list. Each step halves the value range and is $O(1)$, so the query is $O(\log\sigma)$, and $k$ stays in range at every step — going left $k \le \text{num\_left}$, the left range's length; going right $k - \text{num\_left}$ stays within the right range's length — so the descent never falls off an end.

The range rank reuses the identical mapping. The qualifying elements are those with code in $[0, x_c)$, a prefix of the value domain, where $x_c$ is the number of distinct values $\le x$. Recurse from the root: at a node owning value range $[\text{vlo}, \text{vhi}]$, if the node sits wholly below the threshold ($\text{vhi} < x_c$) then every element of the current sub-slice qualifies — add the slice length; if it sits wholly at or above it ($x_c \le \text{vlo}$) then none do — add $0$; otherwise the threshold splits the node, so map the slice into both children by the same prefix formula and sum. Because the threshold boundary follows a single root-to-leaf path, at most two nodes per level are ever split — the rest terminate immediately — so this too touches $O(\log\sigma)$ nodes. It is exactly the inverse machinery of the order statistic, as the shared primitive promised. Duplicates need no special handling: equal codes travel together through every split, pile up at a leaf, and the prefix counts tally them by multiplicity; a degenerate all-equal array has $\sigma=1$, a single leaf, and both queries answer directly from it. Build is $O(n\log\sigma)$ time and memory, each query is $O(\log\sigma)$, and with compression $\sigma \le n$ so both are $O(\log n)$.

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
