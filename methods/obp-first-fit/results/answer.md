**Problem.** Online 1-D bin packing: items arrive one at a time and must be placed immediately and
irrevocably into an open bin (capacity `C`) that still fits, or a new bin. Minimise the number of
bins; report mean #bins and percent excess over the L1 lower bound `ceil(Σ items / C)`. The editable
surface is one function `priority(item, bins) -> array` scoring each currently-fitting bin; the item
goes to the `argmax` bin.

**Key idea.** First-Fit: ignore fit quality entirely and place the item in the **first** open bin
that can hold it, opening a new bin only when none can. In the harness the bins array is in a stable
positional order, so "first bin that fits" is "valid bin of smallest index." A priority that strictly
decreases with the bin's position makes the earliest valid bin the unique maximum, realising
First-Fit exactly. A new bin opens for free: when no used bin fits, the only valid bins are still-full
ones, and the earliest of those is a fresh bin.

**Why these choices.** This is the floor every later rung must beat. Greedy reuse — never opening a
new bin when an old one fits — captures most of the achievable saving and can never do worse than one
bin per item, so it is the honest, predictable baseline. Its known weakness, named here so the next
rung can attack it, is the word "first": among several bins that all fit, First-Fit takes the earliest
by accident of creation order, with no regard for the slack left behind. It will spend a roomy bin's
capacity on a small item and fail to top off a nearly-full bin — wasted capacity that becomes extra
bins. The fix is to rank by fit quality (Best-Fit), which is the next rung.

**Hyperparameters / contract.** None. The single-file program reads the instance from stdin —
capacity `C`, item count `n`, then the `n` item sizes — and prints the number of bins First-Fit uses
followed by the L1 lower bound `ceil(Σ items / C)`. Each item goes into the earliest open bin that
still fits, opening a fresh bin only when none do. Deterministic given the stream; works at any
capacity `C` and any item sizes. Capacities and the running total are `long long` to avoid overflow.

```cpp
// Online 1-D bin packing, First-Fit policy.
// Reads from stdin: capacity C, item count n, then n item sizes.
// Prints the number of bins used (and the L1 lower bound) to stdout.
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    long long C;
    int n;
    if (!(cin >> C >> n)) return 0;

    vector<long long> items(n);
    long long total = 0;
    for (int i = 0; i < n; ++i) {
        cin >> items[i];
        total += items[i];
    }

    // remaining[b] = leftover capacity of bin b, in bin-creation order.
    // First-Fit: place each item in the earliest (lowest-index) bin that still
    // fits it; if none fit, open a new bin at the end. Equivalent to the
    // priority rule "score strictly decreasing in bin index, take the argmax".
    vector<long long> remaining;
    remaining.reserve(n);
    for (int i = 0; i < n; ++i) {
        long long item = items[i];
        int chosen = -1;
        for (int b = 0; b < (int)remaining.size(); ++b) {
            if (remaining[b] >= item) { chosen = b; break; }  // earliest valid bin
        }
        if (chosen == -1) {                 // no open bin fits -> open a fresh bin
            remaining.push_back(C - item);
        } else {
            remaining[chosen] -= item;
        }
    }

    long long used = (long long)remaining.size();
    long long lb = (total + C - 1) / C;     // L1 lower bound ceil(sum/C)

    cout << used << "\n";
    cout << lb << "\n";
    return 0;
}
```
