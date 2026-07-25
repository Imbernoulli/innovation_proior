# Ballast quota on a research submarine

## Research question

A deep-sea submarine is trimming its ballast before a dive. There are `n` candidate ballast crates on
the dock. Loading crate `i` changes the vessel's net ballast by an integer `c[i]` kilograms, where
`c[i]` may be **positive, zero, or negative** (some "crates" are actually buoyancy floats that *remove*
effective ballast). The dive plan imposes a quota window: the crew must load a number of crates `t`
with `L <= t <= K`. Subject to that count window, they want to **maximize** the total net ballast
change `sum of c[i] over the chosen crates`.

Output that maximum total. The interesting structure is that `L` can force the crew to load crates even
when *every* available delta is negative, while `K` caps how many of the good (positive) crates they
are allowed to keep — so the optimum is not simply "take all the positives" and it is not simply "take
the top `K`". The corners that bite are all-negative inputs, the `L = 0` empty-load option, and zeros.

## Input / output contract

- Input (stdin): the first line has three integers `n`, `L`, `K`
  (`0 <= n <= 2*10^5`, `0 <= L <= K <= n`). The second line has `n` integers `c[i]`
  (`-10^9 <= c[i] <= 10^9`), whitespace-separated. When `n = 0` the second line is empty/absent.
- Output (stdout): a single line with the maximum achievable total net ballast change.
- Time limit: 1 second. Memory: 256 MB.

Example: for `n = 6`, `L = 2`, `K = 4`, `c = [5, -3, 8, -1, 0, -7]` the answer is `13`
(load the crates with deltas `8` and `5`; that is `t = 2` crates, which satisfies `2 <= t <= 4`, and
adding any of the remaining non-positive deltas would only lower the total).

## Evaluation settings

Judged on hidden tests covering: all-positive deltas; mixed positive/negative/zero; **all-negative**
arrays with `L = 0` and with `L > 0`; `L = K` (a fixed quota, no choice of count); `L = 0` versus
`L = K = 0`; `n = 0`; `n = 1`; and large `n = 2*10^5` with `|c[i]|` near `10^9`.

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    long long L, K;
    if (!(cin >> n >> L >> K)) return 0;   // empty input -> load nothing
    vector<long long> c(n);
    for (auto &x : c) cin >> x;

    // TODO: maximize the sum of a chosen subset whose size t satisfies L <= t <= K.
    long long best = 0;

    cout << best << "\n";
    return 0;
}
```
