**Problem.** Online 1-D bin packing: items arrive one at a time, placed immediately and irrevocably
into a fitting open bin (capacity `C`) or a new bin. Minimise #bins; report mean #bins and percent
excess over the L1 lower bound `ceil(Σ items / C)`. Edit one function `priority(item, bins) -> array`;
the item goes to the `argmax` bin.

**Key idea.** Best-Fit: place the item in the valid bin where it fits **most snugly** — the bin left
with the least slack `r − s` (remaining capacity minus item size) after placement, among all bins
with `r ≥ s`. Score each bin `−(r − s)` so the tightest fit is the `argmax`. This keeps small items
with nearly-full bins (closing them off near-full, the dense packing the lower bound rewards) and
*preserves* roomy bins for genuinely large future items, instead of squandering them on small items
the way First-Fit does.

**Why these choices.** A small leftover hole is doubly good: a bin filled to a sliver of capacity has
effectively retired near-full, and by sending the item to the bin that barely fits it we leave the
roomy bins intact for big arrivals — the matching that wastes the least capacity. New-bin behaviour is
automatic and correct: a still-full bin has the largest slack, hence the most negative score, so
Best-Fit opens a fresh bin only when no used bin can hold the item at all. The residual waste is the
scatter of tight-fit holes Best-Fit cannot avoid because it judges each bin *in isolation*: it is a
monotone function of a single bin's slack, so any reshaping of the per-bin score gives the same
ordering and the same choices. Beating it therefore requires a score that compares a bin *against the
others* — the lever the next rung pulls.

**Hyperparameters / contract.** None. Deterministic given the stream; works at any capacity and item
sizes. The program reads `C N` (capacity, item count) then `N` item sizes from stdin, runs the online
Best-Fit packing, and prints the number of bins used. The whole policy is the inner placement rule:
pick the open bin with minimal remaining capacity among those with `remaining ≥ size` (equivalently,
maximise `−(remaining − size)`); a still-full bin has the largest slack and the most negative score,
so a fresh bin opens only as a last resort.

```cpp
// Online 1-D bin packing, Best-Fit policy (single-file, reads stdin).
// Input:  first line "C N" (bin capacity C, number of items N);
//         then N item sizes (whitespace-separated, any layout).
// Output: the number of bins used by online Best-Fit on that stream.
// Best-Fit places each arriving item into the open bin where it fits most
// snugly -- the bin left with the least slack (remaining - size) after
// placement -- opening a new bin only when no open bin can hold it.

#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    long long C, N;
    if (!(cin >> C >> N)) return 0;

    // remaining[i] = remaining capacity of open bin i.
    vector<long long> remaining;
    remaining.reserve((size_t)N);

    for (long long k = 0; k < N; ++k) {
        long long s;
        if (!(cin >> s)) break;

        // Best-Fit: among open bins that fit (remaining >= s), pick the one
        // left with the least slack (remaining - s), i.e. minimal remaining.
        long long bestIdx = -1;
        long long bestRemaining = 0;
        for (size_t i = 0; i < remaining.size(); ++i) {
            if (remaining[i] >= s) {
                if (bestIdx == -1 || remaining[i] < bestRemaining) {
                    bestRemaining = remaining[i];
                    bestIdx = (long long)i;
                }
            }
        }

        if (bestIdx == -1) {
            // No open bin fits: open a fresh bin at capacity C.
            remaining.push_back(C - s);
        } else {
            remaining[(size_t)bestIdx] -= s;
        }
    }

    // Every opened bin is a used bin.
    cout << remaining.size() << "\n";
    return 0;
}
```
