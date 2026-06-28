#include <bits/stdc++.h>
using namespace std;

// We must recover the ascending order of a hidden length-n sequence of DISTINCT
// keys using only sign comparisons "is key[i] < key[j]?", and we must do it
// within a budget of B = n * ceil(log2 n) comparisons. The naive "compare every
// pair to find ranks" costs n(n-1)/2 comparisons and blows the budget; the
// merge-sort comparison schedule meets it (top-down merge sort uses at most
// n*ceil(log2 n) - 2^ceil(log2 n) + 1 comparisons, comfortably <= B).
//
// Output (fully determined by the input):
//   line 1: the indices 1..n listed in ascending order of their key (the
//           recovered permutation / the order the comparisons reveal),
//   line 2: C, the exact number of pairwise sign comparisons the canonical
//           top-down merge-sort schedule spends on this input.
//
// The canonical schedule (so the count is well defined): top-down merge sort on
// the index range [lo, hi); split at mid = (lo + hi) / 2; recurse left then
// right; merge by repeatedly comparing the current left head against the current
// right head (one comparison each time both sides are non-empty), taking the
// smaller key and breaking ties toward the left (no ties occur: keys distinct).

static long long comps = 0;          // total sign comparisons performed
static vector<long long> key;        // the hidden keys, indexed by element id
static vector<int> idx;              // working array of element ids being sorted
static vector<int> buf;              // merge scratch

// Sort idx[lo, hi) ascending by key[], counting one comparison per key-vs-key
// test in the merge step. Returns nothing; idx[lo,hi) becomes sorted.
static void mergeSort(int lo, int hi) {
    if (hi - lo <= 1) return;        // 0 or 1 element: already sorted, no compares
    int mid = lo + (hi - lo) / 2;
    mergeSort(lo, mid);
    mergeSort(mid, hi);
    // merge idx[lo,mid) and idx[mid,hi) into buf[lo,hi)
    int i = lo, j = mid, k = lo;
    while (i < mid && j < hi) {
        ++comps;                                  // one sign comparison
        if (key[idx[i]] <= key[idx[j]]) buf[k++] = idx[i++];
        else                            buf[k++] = idx[j++];
    }
    while (i < mid) buf[k++] = idx[i++];
    while (j < hi)  buf[k++] = idx[j++];
    for (int t = lo; t < hi; ++t) idx[t] = buf[t];
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    if (!(cin >> n)) return 0;                    // no input -> nothing to do
    key.resize(n);
    for (int i = 0; i < n; ++i) cin >> key[i];

    idx.resize(n);
    for (int i = 0; i < n; ++i) idx[i] = i;        // element ids 0..n-1
    buf.assign(n, 0);

    mergeSort(0, n);                               // recover order; count comparisons

    // line 1: recovered order as 1-based element indices, ascending by key.
    for (int i = 0; i < n; ++i) {
        cout << (idx[i] + 1);
        cout << (i + 1 < n ? ' ' : '\n');
    }
    if (n == 0) cout << '\n';                       // empty order: still print a line

    // line 2: the comparison count of the canonical merge-sort schedule.
    cout << comps << '\n';
    return 0;
}
