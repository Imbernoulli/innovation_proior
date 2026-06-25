# Loading the deep-sea submersible

## Research question

A crewed submersible is about to dive, and the science team has to decide which sample crates to
load. The pressure hull has a single ballast budget: the crates you bring must have **total mass at
most `C`** (an integer in arbitrary mass units), or the sub cannot surface. There are `n` candidate
crates; crate `i` has integer **mass `w[i]`** and integer **scientific value `v[i]`**. Each crate is
either loaded whole or left on deck — you cannot bring half a crate. Choose a subset of crates whose
total mass does not exceed `C` so that the **total scientific value is maximized**. Bringing nothing
is allowed, so the answer is at least `0`. Output that maximum value.

This is the 0/1 (bounded-capacity, indivisible-item) knapsack in a concrete costume. The interesting
part is not the story but a temptation the story invites: because the crates have a clean "value per
unit mass" ratio, it is very natural to reach for a ratio-greedy loadout. Getting right *why that is
wrong here* — and what to do instead — is the point.

## Input / output contract

- Input (stdin): the first line has two integers `n` and `C`
  (`0 <= n <= 1000`, `0 <= C <= 2000`). Then `n` lines follow, the `i`-th holding two integers
  `w[i]` and `v[i]` (`0 <= w[i] <= 2000`, `0 <= v[i] <= 10^9`). Whitespace (including newlines) may
  be arbitrary; read token by token.
- Output (stdout): a single line with the maximum achievable total value.
- Time limit: 1 second. Memory: 256 MB.

Example: for `C = 10` and crates `(w, v) = (6, 10), (5, 7), (5, 7)` the answer is `14` (load the two
mass-5 crates). Loading the single mass-6 crate of value 10 — which has the best value/mass ratio —
leaves only 4 units of budget, fits no other crate, and yields only `10`.

## Background

The adjacency-free selection is governed by one global resource (the mass budget `C`), and two
families of approach are on the table before committing:

- **Ratio greedy.** Sort the crates by `v[i] / w[i]` descending and load them in that order while
  they still fit; load any zero-mass positive-value crate for free. This is `O(n log n)` and a few
  lines. It is provably optimal for the *fractional* knapsack (where you may take a fraction of a
  crate), and that fact makes it dangerously plausible here. The open question is whether
  indivisibility breaks it.
- **Capacity dynamic programming.** Build, for each reachable total mass `c` from `0` to `C`, the
  best total value of a subset whose mass is exactly `c`, adding crates one at a time. This is
  `O(n * C)`; the open question is the exact recurrence, the iteration direction that keeps each
  crate usable at most once, and how to read the final answer off the table.

## Evaluation settings

Judged on hidden tests covering: instances where ratio greedy is strictly suboptimal (the core
trap), all-fit and none-fit instances, crates strictly heavier than `C` (must be ignored),
zero-mass crates (free value, but never double-counted), zero-value crates, `n = 0`, `C = 0`, exact
budget fills, and large `n = 1000`, `C = 2000` with values near `10^9` so the total value can exceed
a 32-bit integer.

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
