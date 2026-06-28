# Context: online one-dimensional bin packing

## Research question

A stream of items arrives **one at a time**; each has a size and must be placed **immediately and
irrevocably**, before the next item is seen, into one of the currently open bins (each of fixed
capacity `C`) that still has room for it, or into a brand-new bin. Once placed, an item never moves.
The deliverable is a single self-contained C++17 program reading from stdin and writing to stdout.
It reads the capacity `C`, the item count `n`, then the `n` item sizes in stream order; it processes
the stream online and prints the number of bins used followed by the L1 lower bound. The goal is to
use as **few bins** as possible over the whole stream (lower is better).

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

## Code framework

The program is a single self-contained C++17 file. It reads from stdin: capacity `C`, item count `n`,
then the `n` item sizes. It writes to stdout the number of bins used, then the L1 lower bound
`ceil(Σ items / C)`, one value per line. The scaffold fixes the entry point and I/O contract while
leaving the online placement policy open.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main(){
    ios::sync_with_stdio(false);
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

    long long used_bins = 0;
    // TODO:

    long long lb = (total + C - 1) / C;
    cout << used_bins << '\n';
    cout << lb << '\n';
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

The deliverable is a single self-contained C++17 program. It is a C++ program reading from stdin and
writing to stdout.
Input is whitespace-separated: capacity `C`, item count `n`, then the `n` item sizes in stream order.
Output is exactly two lines: the number of bins used, then the L1 lower bound `ceil(Σ items / C)`.
