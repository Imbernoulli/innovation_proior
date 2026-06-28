# Context: online one-dimensional bin packing

## Research question

A stream of items arrives **one at a time**; each has a size and must be placed **immediately and
irrevocably**, before the next item is seen, into one of the currently open bins (each of fixed
capacity `C`) that still has room for it, or into a brand-new bin. Once placed, an item never moves.
The single thing being designed is the online placement rule: when an item arrives, choose one open
bin that fits it or open a new bin. The deliverable is a **single self-contained C++17 program reading from stdin**
and writing to stdout; it reads the stream, applies the rule, and prints the final number of used
bins. The goal is to use as **few bins** as possible over the whole stream (lower is better).

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


## Input-output contract

The program reads the bin capacity `C`, item count `N`, then the `N` item sizes from stdin as
whitespace-separated integers. The first two values are `C N`; the following `N` values may appear on
any later lines. It writes exactly one integer to stdout: the number of bins used by the online
placement on that stream, followed by a newline.

```cpp
#include <bits/stdc++.h>
using namespace std;
int main(){
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    // read input from stdin per the contract
    long long C, N;
    if (!(cin >> C >> N)) return 0;
    vector<long long> items;
    items.reserve(static_cast<size_t>(N));
    for (long long k = 0; k < N; ++k) {
        long long s;
        cin >> s;
        items.push_back(s);
    }

    long long bins_used = 0;
    // TODO: compute the result for this stream

    // print result to stdout
    cout << bins_used << '\n';
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

## The editable interface

All code lives in one C++17 source file. The placement rule is the editable part; the input format,
output format, online constraint, item order, bin capacity, and metric are fixed.
