# Counting full-coverage rosters from candidate squads

## Research question

A company has `n` employees, numbered `0..n-1`. The HR department has pre-approved a list of `m`
candidate **squads**; squad `j` is a set of employees given as an `n`-bit bitmask `S_j` (bit `i` set
means employee `i` belongs to that squad). A **roster** picks a collection of candidate squads that
together cover **every** employee exactly once: the chosen squads are pairwise disjoint and their
union is the whole staff. In other words, a roster is a partition of the `n` employees in which every
block is one of the approved candidate squads.

Two rosters are considered the same if they use the same **set** of squads — the order in which you
list the squads does not matter. Count the number of distinct valid rosters, modulo `1_000_000_007`.

This is a set-partition counting problem driven by an allowed-blocks list. The counting is the whole
difficulty: the natural recurrence is easy to write so that the *same* partition is tallied several
times (once per order of its blocks), or so that a duplicated / empty candidate squad inflates the
count. Getting the de-duplication and the canonical block order exactly right is the point.

## Input / output contract

- Input (stdin): the first line has two integers `n` and `m` (`0 <= n <= 18`, `0 <= m <= 2*10^5`).
  The next tokens are the `m` candidate squad masks `S_0 .. S_{m-1}`, whitespace-separated, each an
  integer in `[0, 2^n - 1]`. Defensive note: a mask may be given as `0` (an empty squad) or may
  contain stray high bits; only the low `n` bits matter and empty squads are not real squads.
- Output (stdout): a single line with the number of valid rosters modulo `1_000_000_007`.
- Time limit: 3 seconds. Memory: 256 MB.

Example: with `n = 4` and candidate masks `{3, 12, 15, 5, 10, 3, 0}` (i.e. `{0,1}`, `{2,3}`,
`{0,1,2,3}`, `{0,2}`, `{1,3}`, a duplicate `{0,1}`, and an empty squad), the answer is `3`: the rosters
are `{0,1,2,3}`, `{0,1}+{2,3}`, and `{0,2}+{1,3}`.

## Background

The quantity asked for is the number of partitions of `{0..n-1}` all of whose blocks lie in a given
allowed family. When *every* non-empty subset is allowed, this count is the Bell number `B(n)`, which
already shows the combinatorial explosion: `B(18) = 682076806159`-ish before reduction. Two ideas are
on the table before committing:

- **Subset-sum / "iterate over the next block" DP over bitmasks.** Define `dp[mask]` = number of valid
  partitions of the employee set `mask`. To fill it, pick which candidate squad covers some particular
  still-uncovered employee, remove that squad, and recurse on the remainder. This is `O(3^n)` if we
  enumerate submasks. The open question is precisely *which* employee to branch on so that each
  unordered partition is produced exactly once — choose wrong and every partition is counted `k!`
  times (once per ordering of its `k` blocks).

- **Inclusion–exclusion over uncovered employees.** Count ordered sequences of disjoint allowed
  squads and then divide out orderings. This needs modular inverses of factorials and is fiddly to set
  up correctly; the division-by-order step is exactly where a subtle counting error hides.

The first route is cleaner to make provably correct, *if* the canonical-order trick is applied
faithfully.

## Evaluation settings

Judged on hidden tests covering: `n = 0` (the empty staff has exactly one roster — use no squads);
`m = 0` and rosters that are impossible (answer `0`); inputs with duplicated candidate masks and
empty (`0`) masks that must be ignored without inflating the count; the all-subsets-allowed case where
the answer must equal `B(n) mod p`; cases needing the modulus (counts far exceeding 64-bit); and the
largest `n = 18` with up to `2*10^5` candidate masks, stressing the `O(3^n)` DP within the time limit.

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

const long long MOD = 1000000007LL;

int main() {
    int n, m;
    if (scanf("%d %d", &n, &m) != 2) return 0;

    int full = (n > 0) ? ((1 << n) - 1) : 0;

    // Read the m candidate masks. Mask off out-of-range bits, drop empty squads,
    // and de-duplicate into an `allowed` table.
    vector<char> allowed(full + 1, 0);
    for (int j = 0; j < m; j++) {
        int x;
        if (scanf("%d", &x) != 1) x = 0;
        x &= full;
        if (x == 0) continue;
        allowed[x] = 1;
    }

    // TODO: count the partitions of {0..n-1} whose every block is an allowed
    // squad, where order of the blocks does not matter, modulo MOD.
    long long answer = 0;

    printf("%lld\n", answer % MOD);
    return 0;
}
```
