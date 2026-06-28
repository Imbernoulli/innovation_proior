# Quicksort

## Problem

Sort N items, held contiguously in a computer's fast random-access store, into ascending key order —
quickly (far better than the N² of the naive sorts) and **in place** (no second array the size of
the data). The motivating use: putting the words of a sentence into alphabetical order so they can be
looked up in one forward pass of a dictionary held on slow magnetic tape.

## Key idea

Sort by *partitioning around a bound* instead of by swapping neighbours:

1. **Partition.** Choose one item's key as the bound. With a single pass, rearrange the segment so
   that every item with key <= bound is below a dividing line and every item with key >= bound is
   above it. One comparison against the bound sends each item to its correct *side* in one move, the
   leap that a neighbour-swap sort cannot make. Do it in place with two pointers scanning inward from
   the ends: the lower pointer stops at a key greater than the bound, the upper pointer stops at a key
   less than the bound, and the two stopped items are exchanged until the pointers cross.
2. **Recurse.** The two resulting sub-segments are independent, so sort each by the same method.
   The partition returns the two recursive ranges `lo..j` and `i..hi`, not a single pivot index.

Balanced splits give ≈ N log N: the work per level is ≈ N (one partition pass over all items at that
level) and there are ≈ log₂ N levels. A random (or median-of-sample) bound keeps the split balanced
on average and prevents already-sorted input from being a systematic worst case.

**Cost.** With a random bound, the expected comparison recurrence has the form
`C_N = (1/N) * sum_{r=1..N} (C_{r-1} + C_{N-r}) + aN + O(1)`. For the usual comparison count this is
`2N ln N + O(N)`, or about `1.39 * N log2 N`. The information-theoretic floor for a comparison sort
is `log2(N!) ~= N log2 N`, so the average is above the floor by the factor `2 ln 2 ~= 1.4`. Worst
case, if the bound is consistently extreme, is `N^2`.

**Termination.** Equality is deliberately passed over by both scans, so equal keys do not trap the
pointers. When the pointers cross, the dividing line is between `j` and `i`. If the item that supplied
the bound still lies inside one recursive side, it is exchanged to that side's edge and excluded by
moving `i` or `j`; in equality edge cases the remaining pieces are already singletons. No recursive
call receives the same whole segment back.

**The recursion / the nest.** The collection of partitioned-but-unsorted segments forms a
last-in-first-out list. With language recursion the procedure calling itself on a sub-segment keeps
this list automatically (the chain of suspended calls *is* the list). Without recursion you maintain
an explicit "nest" (a pushdown stack of segment bounds); always postponing the *larger* segment and
recursing into the *smaller* caps the nest depth at log₂ N.

## Algorithm

```
partition(a, lo, hi):
    f := random position in lo..hi
    bound := key(a[f])
    i := lo; j := hi
    scan i upward while key(a[i]) <= bound
    scan j downward while key(a[j]) >= bound
    while i < j:
        exchange a[i] and a[j]
        step i upward and j downward
        resume the two scans
    if the bound-supplying item lies inside the upper recursive side:
        exchange it with the first item of that side; move i upward
    else if it lies inside the lower recursive side:
        exchange it with the last item of that side; move j downward
    return i, j

quicksort(a, lo, hi):
    if lo < hi:
        i, j := partition(a, lo, hi)
        quicksort(a, lo, j)
        quicksort(a, i, hi)
```

## Code

Single-file C++17. Reads `N` followed by `N` integer keys from standard input and prints the keys
in ascending order, space-separated on one line. Sorts in place (no second array), using the
random-bound two-pointer partition and the explicit-nest loop that always postpones the larger piece
(so the nest depth stays at log₂ N). The recursive procedure that ALGOL would let you write is
exactly this loop with the chain of suspended self-calls replaced by an explicit pushdown nest.

```cpp
// Quicksort: in-place comparison sort by partitioning around a random bound.
// Reads from stdin: N, then N integer keys. Prints the N keys in ascending order,
// space-separated on one line. Sorts in place (no second array), expected N log N.
#include <bits/stdc++.h>
using namespace std;

static uint64_t rng_state = 0x9e3779b97f4a7c15ULL;
static inline uint64_t xorshift64() {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    return rng_state;
}
// uniform position in [lo, hi]
static inline long long rand_pos(long long lo, long long hi) {
    return lo + (long long)(xorshift64() % (uint64_t)(hi - lo + 1));
}

// Hoare-style two-pointer partition around a randomly chosen bound.
// Rearranges a[lo..hi] in place; returns (i, j) so the recursive sides are
// a[lo..j] and a[i..hi]. Equal keys are passed over by both scans so they do
// not trap the pointers; the bound item is peeled to an edge if it remains
// inside a side, guaranteeing every returned side is strictly smaller.
static pair<long long, long long> partition_seg(vector<long long>& a,
                                                long long lo, long long hi) {
    long long f = rand_pos(lo, hi);     // an actual item -> bound is in range
    long long bound = a[f];
    long long i = lo, j = hi;
    while (true) {
        while (i < hi && a[i] <= bound) ++i;   // stop at first key > bound
        while (j > lo && a[j] >= bound) --j;    // stop at first key < bound
        if (i < j) {
            swap(a[i], a[j]);                   // two items in each other's territory
            ++i; --j;
            continue;
        }
        if (i < f) {            // bound item lies in the upper recursive side
            swap(a[i], a[f]);
            ++i;
        } else if (f < j) {     // bound item lies in the lower recursive side
            swap(a[f], a[j]);
            --j;
        }
        return {i, j};
    }
}

// Explicit-nest form (no language recursion): the "nest" is a pushdown stack of
// postponed segments. Always continue on the SMALLER piece and postpone the
// LARGER one, so the nest depth is bounded by log2(N).
static void quicksort(vector<long long>& a) {
    long long n = (long long)a.size();
    if (n < 2) return;
    long long lo = 0, hi = n - 1;
    vector<pair<long long, long long>> nest;
    while (true) {
        while (lo < hi) {
            auto pr = partition_seg(a, lo, hi);
            long long i = pr.first, j = pr.second;
            long long left_lo = lo, left_hi = j;       // a[lo..j]
            long long right_lo = i, right_hi = hi;      // a[i..hi]
            long long left_size = max(0LL, left_hi - left_lo + 1);
            long long right_size = max(0LL, right_hi - right_lo + 1);
            if (left_size < right_size) {               // postpone larger (right)
                if (right_size > 1) nest.push_back({right_lo, right_hi});
                lo = left_lo; hi = left_hi;
            } else {                                    // postpone larger (left)
                if (left_size > 1) nest.push_back({left_lo, left_hi});
                lo = right_lo; hi = right_hi;
            }
        }
        if (nest.empty()) break;
        lo = nest.back().first; hi = nest.back().second;   // resume most recent
        nest.pop_back();
    }
}

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);
    long long n;
    if (!(cin >> n)) return 0;
    vector<long long> a(n);
    for (long long k = 0; k < n; ++k) cin >> a[k];
    quicksort(a);
    for (long long k = 0; k < n; ++k) {
        cout << a[k];
        if (k + 1 < n) cout << ' ';
    }
    cout << '\n';
    return 0;
}
```

## Practical refinements

- **Small segments**: below a cutoff of a few items, sort by a special-purpose routine (e.g. insertion
  sort) rather than recursing.
- **Sentinels**: place impossibly-large/small keys at the ends so the inner comparison loop can drop
  the pointer-range test.
- **Better pivots**: take the median of a small random sample to push the
  average comparison count nearer the floor and further from the N² worst case.
