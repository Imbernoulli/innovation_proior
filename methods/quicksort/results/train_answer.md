I would call this method quicksort. It is a comparison sort that rearranges a contiguous block of items in place by repeatedly partitioning segments around a chosen bound, then sorting the resulting sub-segments independently. The canonical name is quicksort, and the version I present here follows the two-pointer partition scheme with a randomly chosen pivot bound, together with an explicit last-in-first-out nest so the same idea can be expressed even when the language does not provide recursive procedures.

The motivating task is to take a short list of words held in fast random-access store and arrange them in ascending key order so quickly that the overhead is negligible inside a larger translation loop. A slow backing store, such as a magnetic tape dictionary, makes random access expensive; the practical fix is to read the tape once in forward order and look up every word along the way. That only works if the words have already been sorted, so the in-memory sort must be both fast and frugal with store.

The naive quadratic sorts fail on both counts. Bubble sort, insertion sort, and their relatives move an item past only one neighbor at a time, so a word that belongs far from its starting position needs on the order of N passes to get home, giving N-squared comparisons and movements in total. Most of those comparisons tell us almost nothing, because they ask whether two items are in roughly the right order relative to each other rather than where each item belongs in the whole array. The information-theoretic floor for any comparison sort is about N log2 N comparisons, so quadratic methods are far above what is possible.

Quicksort escapes the neighbor-by-neighbor crawl by comparing each item against a single global bound. Choose one item in the current segment, read its key, and call that value the bound. Now scan through the segment once. Every item whose key is less than or equal to the bound belongs somewhere in the lower part of the final order; every item whose key is greater than or equal to the bound belongs somewhere in the upper part. A single comparison against the bound therefore sends an item to the correct side of a dividing line, no matter how far it had to travel, replacing the slow crawl of adjacent swaps with a single leap.

To perform this rearrangement without a second array, I use two pointers that start at opposite ends of the segment. The lower pointer walks upward, passing over items already on the correct side until it stops at the first item whose key is greater than the bound. The upper pointer walks downward, passing over items already on the correct side until it stops at the first item whose key is less than the bound. Those two items are in each other's territory, so I exchange them and continue stepping the pointers inward. The process repeats until the pointers cross. At that moment every item below the crossing point is at most the bound and every item above it is at least the bound, so the segment has been partitioned in place with only exchanges.

After a partition, the lower and upper parts are independent. Nothing below the dividing line will ever need to move above it, and nothing above will ever need to move below. I therefore sort each part by the same method. The recursion bottoms out when a part contains zero or one items, because such a part is already sorted. When the splits are roughly balanced, each level of recursion processes all current items once, costing about N comparisons, and there are about log2 N levels, so the total cost is about N log2 N, close to the comparison-sort floor.

The choice of bound matters for balance. If the bound is always the largest or smallest key, the split is lopsided and the cost can degrade back to N squared. I avoid that by choosing the bound item uniformly at random from the segment. Random choice gives balanced splits on average and prevents pathological behavior on already-sorted or adversarially ordered inputs, even though a deliberately unlucky sequence of random choices could still, in principle, produce a worst case.

Termination requires a small but important detail. The lower scan stops only at keys strictly greater than the bound, and the upper scan stops only at keys strictly less than the bound, so the bound item itself is never swapped during the inward scans. After the pointers cross, if the bound item still lies inside one of the two recursive sides, I exchange it to that side's edge and exclude it from the recursive call. This guarantees that every recursive call receives a strictly smaller segment than its parent, so the recursion cannot loop.

In a language with recursive procedures, the bookkeeping of partitioned-but-not-yet-sorted segments is handled automatically: each recursive call remembers its segment and resumes after the inner call returns. The chain of suspended calls is exactly the last-in-first-out list of postponed segments. If recursion is unavailable, the same list can be maintained explicitly as a nest or stack of segment bounds. To keep the explicit nest small, I always dive into the smaller of the two parts and postpone the larger. Because the working segment at least halves after each push, the nest depth is bounded by log2 N, so only a tiny amount of extra store is needed.

Small practical refinements, such as switching to insertion sort for very short segments or choosing the median of a small sample as the bound, can improve the constant factors, but the core method remains the partition-and-recurse structure described above.

The program reads `N` followed by `N` integer keys from standard input and prints the keys in ascending order, space-separated on one line. It sorts in place, with no second array, using the random-bound two-pointer partition and the explicit-nest loop that always postpones the larger piece (so the nest depth stays at log₂ N).

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
