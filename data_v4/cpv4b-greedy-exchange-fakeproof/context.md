# Minimum gondola trips to evacuate a ridge

## Research question

A storm is closing in on an alpine ridge and `n` climbers must be brought down by a single cable
gondola. Climber `i` weighs `w[i]` kilograms. The gondola cabin is small: each trip it can carry
**at most two** climbers, and the **combined weight** of whoever rides must not exceed the cabin's
rated capacity `C` kilograms. Every climber's individual weight satisfies `w[i] <= C`, so anyone can
always ride alone; the only question is how often two can share a cabin.

A round trip (down and back up) is slow, so the rescue team wants the **minimum number of trips**
needed to get everyone down. Output that minimum.

## Input / output contract

- Input (stdin): the first line has two integers `n` and `C`
  (`0 <= n <= 2*10^5`, `1 <= C <= 10^9`). The second line (present iff `n > 0`) has `n` integers
  `w[i]` (`1 <= w[i] <= C`), whitespace-separated.
- Output (stdout): a single line with the minimum number of gondola trips. When `n = 0` the answer
  is `0`.
- Time limit: 1 second. Memory: 256 MB.

Example: for `n = 6`, `C = 5`, `w = [1, 1, 2, 5, 5, 5]` the answer is `5`. Each of the three
`5`-kg climbers must ride alone (any partner would push the cabin over `5` kg), the `1` and the `2`
can share one trip, and the remaining `1` rides alone — `3 + 1 + 1 = 5` trips.

## Background

The constraint couples two scarce resources at once: each trip has a **slot** budget (at most two
climbers) and a **weight** budget (sum at most `C`). Simulation-style pairing rules and closed-form
counting formulas are both standard tools for this kind of scheduling problem; which one actually
produces the minimum trip count for a given instance is what needs to be worked out.

## Evaluation settings

Judged on hidden tests covering: `n = 0` and `n = 1`; everyone too heavy to pair; everyone pairs
perfectly; a mixed case of light and heavy weights; odd and even `n`; many equal weights; weights
exactly at the capacity; and large `n = 2*10^5` with `C` near `10^9`, so the running trip count and
any weight sums must use 64-bit arithmetic.

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
    if (!(cin >> n >> C)) return 0;          // empty input -> no climbers -> 0 trips
    vector<long long> w(n);
    for (auto &x : w) cin >> x;

    // TODO: compute the minimum number of gondola trips needed, respecting the
    // at-most-two-climbers-per-trip and combined-weight-<=-C limits per trip.
    long long trips = 0;

    cout << trips << "\n";
    return 0;
}
```
