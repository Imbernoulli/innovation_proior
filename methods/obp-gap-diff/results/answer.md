**Problem.** Online 1-D bin packing: items arrive one at a time, placed immediately and irrevocably
into a fitting open bin (capacity `C`) or a new bin. Minimise #bins; report mean #bins and percent
excess over the L1 lower bound `ceil(Σ items / C)`. Edit one function `priority(item, bins) -> array`;
the item goes to the `argmax` bin.

**Key idea.** Beat Best-Fit by scoring each bin **relationally** instead of in isolation. Best-Fit
only sees a bin's own slack, and any monotone reshaping of that gives the same choices. Here each bin
is scored by its squared distance below the **emptiest** valid bin, item-weighted, `(r − r_max)² / s`
(remaining `r`, emptiest remaining `r_max`, item size `s`) — large for already-committed bins, zero
for the freshest. Then the score array is **differenced across bins**, `score[i] ← score[i] −
score[i−1]`, so each bin is judged by the jump in relational value over its neighbour. This couples
the bins together and breaks the single-bin monotone collapse that pins everything to Best-Fit. The
effect is to consolidate already-committed bins faster — finishing off the long tail of nearly-full
bins Best-Fit leaves — keeping fresh capacity in reserve.

**Why these choices.** The emptiest-bin landmark is the cheapest genuinely relational signal in the
array; distance below it measures how committed a bin already is, and rewarding that bias pours items
into bins we are trying to finish. Dividing by `s` makes the relational pull strongest where Best-Fit
is weakest (small items it would scatter). The neighbour differencing is the engine: a per-bin score,
however relational, still risks collapsing under `argmax`, but differencing makes each bin's fate
depend on its predecessor, coupling the whole array. The square power and the plain difference are
hand-chosen from intuition, not derived — that they are *not* optimal is exactly what the final rung
(searching the functional form) addresses. Sign convention: fitting bins (`r ≥ s`) get the score
negated so feasibility is respected before differencing.

**Hyperparameters / contract.** None tunable; the form `(r − r_max)² / s` + neighbour differencing is
fixed. Returns an array the length of the valid-bin slice. Deterministic given the stream. Works at
any capacity and item sizes.

Single-file C++17 program. It reads from stdin the capacity `C`, then the stream of item
sizes (either `n` followed by `n` sizes, or sizes until EOF), runs the online bin-packing
simulator under the gap-difference `priority` rule, and prints to stdout: bins used, the L1
lower bound `ceil(sum/C)`, and percent excess over it.

```cpp
// Online 1-D bin packing with the "gap-difference" relational priority rule.
// Reads from stdin: first token is capacity C, then n, then n item sizes
// (or just C followed by item sizes until EOF). Runs the FunSearch online
// bin-packing simulator -- each item is placed irrevocably into the argmax
// bin under priority(item, valid_bin_remaining) -- and prints to stdout:
//   bins used, the L1 lower bound ceil(sum/C), and percent excess over it.
#include <bits/stdc++.h>
using namespace std;

// Relational score: squared distance below the emptiest valid bin, item-weighted,
// then differenced across neighbouring bins so each bin is judged in context.
// `rem` holds remaining capacities of the bins that can fit the item (rem >= item).
static vector<double> priority(double item, const vector<double>& rem) {
    int n = (int)rem.size();
    double max_bin_cap = rem[0];
    for (int i = 1; i < n; ++i) max_bin_cap = max(max_bin_cap, rem[i]);
    vector<double> score(n);
    for (int i = 0; i < n; ++i) {
        double d = rem[i] - max_bin_cap;
        score[i] = d * d / item;            // distance below the emptiest bin, item-weighted
        if (rem[i] > item) score[i] = -score[i];   // respect feasibility on the fitting bins
    }
    for (int i = n - 1; i >= 1; --i)        // couple each bin to its neighbour (differencing)
        score[i] -= score[i - 1];
    return score;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    double C;
    if (!(cin >> C)) return 0;

    // Read either "n then n sizes" or "all sizes until EOF".
    vector<double> items;
    double x;
    vector<double> rest;
    while (cin >> x) rest.push_back(x);
    if (!rest.empty()) {
        // If the first value is a plausible count matching the remainder, treat it as n.
        long long maybe_n = (long long)llround(rest[0]);
        if (maybe_n >= 0 && (size_t)maybe_n == rest.size() - 1 &&
            fabs(rest[0] - (double)maybe_n) < 1e-9) {
            items.assign(rest.begin() + 1, rest.end());
        } else {
            items = rest;
        }
    }

    // One pre-allocated remaining-capacity slot per item guarantees a fresh
    // bin is always available; a bin is "used" iff its remaining ever dropped below C.
    int m = (int)items.size();
    vector<double> bins(max(m, 1), C);

    double total = 0.0;
    for (double item : items) {
        total += item;
        // Gather valid bins (remaining >= item) in stable positional order.
        vector<int> valid;
        vector<double> valid_rem;
        valid.reserve(bins.size());
        valid_rem.reserve(bins.size());
        for (int i = 0; i < (int)bins.size(); ++i)
            if (bins[i] - item >= 0) { valid.push_back(i); valid_rem.push_back(bins[i]); }
        // (A fresh capacity-C bin is always among the valid ones for a sane item.)
        vector<double> sc = priority(item, valid_rem);
        int best = 0;
        for (int i = 1; i < (int)sc.size(); ++i)
            if (sc[i] > sc[best]) best = i;
        bins[valid[best]] -= item;
    }

    long long used = 0;
    for (double b : bins) if (b != C) ++used;

    long long lb = (long long)ceil(total / C - 1e-12);
    double excess = lb > 0 ? 100.0 * (double)(used - lb) / (double)lb : 0.0;

    cout << used << "\n";
    cout << lb << "\n";
    cout << fixed << setprecision(4) << excess << "\n";
    return 0;
}
```
