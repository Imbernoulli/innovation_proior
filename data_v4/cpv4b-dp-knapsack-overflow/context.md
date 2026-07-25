# Planetarium projector array under a power budget

## Research question

A planetarium is upgrading its dome show. Along the rail above the audience sit `n` candidate
projector modules. Module `i`, if switched on, draws `w[i]` watts from the dome's single power
supply and contributes `b[i]` units of "wow factor" (brightness) to the show. The supply can deliver
at most `C` watts in total. The crew may switch on any subset of the modules whose combined wattage
does not exceed `C`. They want the subset that **maximizes the total brightness**. Output that
maximum total brightness.

## Input / output contract

- Input (stdin): the first line has two integers `n` and `C`
  (`1 <= n <= 2000`, `0 <= C <= 4000`). Then `n` lines follow; line `i` has two integers
  `w[i]` and `b[i]` (`1 <= w[i] <= 4000`, `1 <= b[i] <= 10^9`).
- Output (stdout): a single line with the maximum achievable total brightness. If no module fits
  (every `w[i] > C`, e.g. when `C = 0`), the best subset is empty and the answer is `0`.
- Time limit: 1 second. Memory: 256 MB.

Example: with `C = 7` and modules `(w, b) = (3, 4), (4, 5), (2, 3), (5, 6)`, the answer is `9`
(switch on modules 1 and 2: wattage `3 + 4 = 7 <= 7`, brightness `4 + 5 = 9`; the pair `(2,3)+(5,6)`
also reaches `9`, and nothing does better).

## Evaluation settings

Judged on hidden tests covering: tiny instances checkable by hand, instances where the optimal subset
deliberately leaves wattage unused, instances where no module fits (`C = 0` and `w[i] > C`),
single-module instances, and large instances with `n = 2000`, `C = 4000`, and brightnesses near
`10^9`.

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    long long C;
    if (!(cin >> n >> C)) return 0;

    vector<long long> w(n), b(n);
    for (int i = 0; i < n; i++) cin >> w[i] >> b[i];

    // TODO: compute the maximum total brightness over subsets with total wattage <= C.
    long long answer = 0;

    cout << answer << "\n";
    return 0;
}
```
