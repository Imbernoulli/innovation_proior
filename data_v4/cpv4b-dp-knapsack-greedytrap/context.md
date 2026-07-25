# Loading the deep-sea submersible

## Research question

A crewed submersible is about to dive, and the science team has to decide which sample crates to
load. The pressure hull has a single ballast budget: the crates you bring must have **total mass at
most `C`** (an integer in arbitrary mass units), or the sub cannot surface. There are `n` candidate
crates; crate `i` has integer **mass `w[i]`** and integer **scientific value `v[i]`**. Each crate is
either loaded whole or left on deck — you cannot bring half a crate. Choose a subset of crates whose
total mass does not exceed `C` so that the **total scientific value is maximized**. Bringing nothing
is allowed, so the answer is at least `0`. Output that maximum value.

This is the 0/1 (bounded-capacity, indivisible-item) knapsack in a concrete costume.

## Input / output contract

- Input (stdin): the first line has two integers `n` and `C`
  (`0 <= n <= 1000`, `0 <= C <= 2000`). Then `n` lines follow, the `i`-th holding two integers
  `w[i]` and `v[i]` (`0 <= w[i] <= 2000`, `0 <= v[i] <= 10^9`). Whitespace (including newlines) may
  be arbitrary; read token by token.
- Output (stdout): a single line with the maximum achievable total value.
- Time limit: 1 second. Memory: 256 MB.

Example: for `C = 10` and crates `(w, v) = (6, 10), (5, 7), (5, 7)` the answer is `14` (load the two
mass-5 crates).

## Evaluation settings

Judged on hidden tests covering: all-fit and none-fit instances, crates strictly heavier than `C`
(must be ignored), zero-mass crates (free value, but never double-counted), zero-value crates,
`n = 0`, `C = 0`, exact budget fills, and large `n = 1000`, `C = 2000` with values near `10^9`.

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    long long C;
    if (!(cin >> n >> C)) return 0;
    vector<long long> w(n), v(n);
    for (int i = 0; i < n; i++) cin >> w[i] >> v[i];

    // TODO: maximize total value over subsets of crates with total mass <= C (empty allowed).
    long long answer = 0;

    cout << answer << "\n";
    return 0;
}
```
