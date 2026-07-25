# Peak occupancy of a parking garage (max overlapping half-open intervals)

## Research question

A parking garage logs `n` cars. Car `i` is described by an arrival time `s_i` and a departure
time `e_i`. The car occupies a spot during the **half-open** interval `[s_i, e_i)`: it is present
*at* the instant `s_i`, and it is gone *by* the instant `e_i`. The exclusive right endpoint is the
whole point of the model — a car that leaves at time `t` and a car that arrives at time `t` can
reuse the *same physical spot*, so they are **not** considered simultaneously present.

Question: across all instants of time, what is the **maximum number of cars present at once**?
That number is the minimum number of spots the garage must have to never turn a logged car away.

## Input / output contract

- Input (stdin): the first token is `n` (`0 <= n <= 2*10^5`). Then `n` lines (whitespace is not
  significant) each with two integers `s_i e_i` (`0 <= s_i, e_i <= 10^9`).
- A record with `s_i >= e_i` describes a car that occupies *no* instant (a zero- or negative-length
  stay, e.g. a logging glitch); such a record contributes nothing and must be ignored.
- Output (stdout): a single line with the maximum number of cars simultaneously present, using the
  half-open `[s_i, e_i)` convention. If there are no (effective) cars the answer is `0`.
- Time limit: 1 second. Memory: 256 MB.

Example: for the five cars `[1,4)`, `[2,5)`, `[5,7)`, `[3,4)`, `[4,6)` the answer is `3`. At the
instant `t = 3` the cars `[1,4)`, `[2,5)`, `[3,4)` are all present; no instant has four.

## Evaluation settings

Judged on hidden tests covering: disjoint intervals (answer `1`); intervals that merely *touch*,
i.e. `e_i == s_j`; many intervals sharing one endpoint; nested intervals; the empty input (`n = 0`);
a single car; degenerate `s_i >= e_i` records mixed in; and large `n = 2*10^5` with coordinates
spread across `[0, 10^9]` so the event array is large and ties are common.

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    if (!(cin >> n)) return 0;
    vector<pair<long long,int>> ev;
    for (int i = 0; i < n; i++) {
        long long s, e;
        cin >> s >> e;
        // TODO: turn each half-open interval [s, e) into sweep events and accumulate
        //       the running count.
    }

    long long best = 0;
    // TODO: sort events and sweep to compute best.

    cout << best << "\n";
    return 0;
}
```
