# Context: online one-dimensional bin packing

## Research question

A stream of items arrives **one at a time**; each has a size and must be placed **immediately and
irrevocably**, before the next item is seen, into one of the currently open bins (each of fixed
capacity `C`) that still has room for it, or into a brand-new bin. Once placed, an item never moves.
The deliverable is a single self-contained C++17 program reading the capacity and item stream from
stdin and writing the bin count, L1 lower bound, and percent excess to stdout. Inside the fixed online
simulator, the designed component is the per-valid-bin scoring rule: given the incoming item's size
and the remaining capacities of bins that can still fit it, the program scores every such bin and
places the item into the highest-scoring one. The goal is to use as **few bins** as possible over the
whole stream (lower is better).

Because the online optimum is order-dependent and NP-hard to pin down, the honest yardstick is the
**L1 lower bound** `LB = ceil(Σ items / C)` — the bins needed if every bin were filled to the brim
with zero waste. No policy can beat `LB`, so each heuristic is reported as **mean number of bins** and
as **percent excess over the lower bound**, `100 · (mean_bins − LB) / LB`.

## Background

This is the setting of FunSearch (Romera-Paredes et al., *Mathematical discoveries from program search
with large language models*, Nature 2024; code `google-deepmind/funsearch`, `bin_packing/
bin_packing.ipynb`), which used LLM-driven program search to discover online-bin-packing priority
functions that beat the classical First-Fit and Best-Fit baselines on the Beasley OR-Library and on
Weibull-distributed instances. The published metric is exactly the fraction of bins above the L1
lower bound. Published Table 1 (excess over LB; lower is better):

| Dataset | First Fit | Best Fit | FunSearch |
|---|---|---|---|
| OR3 | 5.74% | 5.37% | 3.11% |
| Weibull 5k | 4.23% | 3.98% | 0.68% |
| Weibull 100k | 4.00% | 3.79% | 0.03% |

The discovered heuristic beats Best Fit on every dataset and widens the gap as streams grow.

## The fixed substrate

The C++ scaffold below follows the FunSearch online-bin-packing skeleton. It maintains an array of bin
remaining-capacities (pre-allocated large enough that a fresh bin is always available); for each
arriving item it finds the valid bins (`remaining ≥ item`), calls `priority(item, valid_remaining)`,
places the item in the `argmax` bin, and decrements it. A bin is "used" iff its remaining capacity
ever dropped below `C`.

```cpp
#include <bits/stdc++.h>
using namespace std;

static vector<double> priority(double item, const vector<double>& rem) {
    (void)item;
    vector<double> score(rem.size(), 0.0);
    // TODO: <solve>
    return score;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    double C;
    if (!(cin >> C)) return 0;

    vector<double> items;
    double x;
    vector<double> rest;
    while (cin >> x) rest.push_back(x);
    if (!rest.empty()) {
        long long maybe_n = (long long)llround(rest[0]);
        if (maybe_n >= 0 && (size_t)maybe_n == rest.size() - 1 &&
            fabs(rest[0] - (double)maybe_n) < 1e-9) {
            items.assign(rest.begin() + 1, rest.end());
        } else {
            items = rest;
        }
    }

    int m = (int)items.size();
    vector<double> bins(max(m, 1), C);

    double total = 0.0;
    for (double item : items) {
        total += item;
        vector<int> valid;
        vector<double> valid_rem;
        valid.reserve(bins.size());
        valid_rem.reserve(bins.size());
        for (int i = 0; i < (int)bins.size(); ++i)
            if (bins[i] - item >= 0) { valid.push_back(i); valid_rem.push_back(bins[i]); }
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

## Evaluation settings

Two seeded synthetic stream families, five seeds each (0–4), 5000 items per stream, matching the
FunSearch evaluation: **Weibull(scale = 45, shape = 3) at capacity `C = 100`** (the FunSearch "Weibull
5k" regime, Castiñeiras et al. 2012) and **OR-Library-style uniform integer sizes on `[20, 100]` at
capacity `C = 150`**. Every method is run on the same seeded streams; reported as mean #bins and
percent excess over the mean L1 lower bound. The simulator reproduces the published repo numbers to
the digit (OR3 First/Best/FunSearch = 5.74%/5.37%/3.11%; Weibull 5k = 4.23%/3.98%/0.68%) as a
calibration check.

## Input-output contract

The deliverable is one self-contained C++17 source file. It reads from stdin with the same contract
used by the final program: first token `C`; then either `n` followed by `n` item sizes, or item sizes
until EOF. It writes three lines to stdout: bins used, `ceil(sum/C)`, and percent excess over that
lower bound with four digits after the decimal point. The online placement rule is implemented inside
the program; the simulator, streams, and capacity handling are otherwise fixed.
