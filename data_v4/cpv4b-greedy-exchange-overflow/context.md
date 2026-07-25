# Festival booth scheduling: maximize total profit under per-act deadlines

## Research question

A street festival has a single performance stage and runs for a long string of equal-length time
slots numbered `1, 2, 3, ...`. You are pitched `n` acts. Act `i` will pay the festival a fee
`p[i]` if and only if it is staged, and it must be staged in **one** time slot whose number is
**at most** its deadline `d[i]` (the act's contract expires after slot `d[i]`). Each slot can hold
**at most one** act, and each act occupies **exactly one** slot. You may leave any act unscheduled
(staging nothing in some slots is fine, and turning an act away is fine).

Choose which acts to stage and in which slots so that the **total fee collected is maximized**, and
output that maximum total fee.

## Input / output contract

- Input (stdin): the first token is `n` (`0 <= n <= 2*10^5`). Then follow `n` lines, each with two
  integers `p[i]` and `d[i]` (`0 <= p[i] <= 10^9`, `1 <= d[i] <= 10^9`), whitespace-separated.
- Output (stdout): a single line with the maximum total fee that can be collected.
- Time limit: 1 second. Memory: 256 MB.

Example: for the five acts `(p, d) = (100, 2), (60, 1), (70, 2), (40, 1), (90, 3)` the answer is
`260` — stage the `100` act in slot 2, the `90` act in slot 3, and the `70` act in slot 1; the `60`
and `40` acts have deadline 1 but slot 1 is taken, so they are turned away.

## Evaluation settings

Judged on hidden tests covering: all acts fitting (large distinct deadlines), heavy collisions
(many acts sharing one small deadline), `p[i] = 0` acts, the empty instance (`n = 0`), a single act,
deadlines far larger than `n`, and large `n = 2*10^5` with fees near `10^9`.

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
    vector<long long> p(n);
    vector<int> d(n);
    for (int i = 0; i < n; i++) cin >> p[i] >> d[i];

    // TODO: schedule acts to maximize total collected fee under the deadline/slot constraints.
    long long answer = 0;

    cout << answer << "\n";
    return 0;
}
```
