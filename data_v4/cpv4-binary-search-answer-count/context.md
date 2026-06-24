# The K-th distinct pulse time of synchronized machines

## Research question

A workshop runs up to three pulse machines on a shared clock. Machine `i` (for `i` in `1..n`,
with `n` between `1` and `3`) has an integer period `p[i]` and fires at every positive multiple of
its period: at times `p[i], 2*p[i], 3*p[i], ...`. At some instants two or three machines fire at
once; such an instant is still **one** moment in time, not several.

Consider the set of **distinct times** at which *at least one* machine fires, sorted increasingly.
You are given a 1-based index `K`. Output the `K`-th smallest distinct firing time.

The set is infinite, so you cannot enumerate it; and the index `K` can be as large as `10^9`, so you
cannot walk to it term by term either. The interesting part is that the obvious way to count "how
many distinct firing times are `<= x`" over-counts the instants where machines coincide — getting
that de-duplication exactly right is the whole problem.

## Input / output contract

- Input (stdin), whitespace-separated:
  - the first token is `n` (`1 <= n <= 3`), the number of machines;
  - then `n` integers `p[1..n]` (`1 <= p[i] <= 10^9`), the periods;
  - then one integer `K` (`1 <= K <= 10^9`).
- Output (stdout): a single line with the `K`-th smallest distinct firing time.
- Time limit: 1 second. Memory: 256 MB.

Example: for `n = 2`, periods `p = [2, 3]`, and `K = 5`, the distinct firing times in order are
`2, 3, 4, 6, 8, 9, 10, 12, ...` (note `6` and `12` are shared by both machines but listed once), so
the answer is `8`.

## Background

The firing times of machine `i` are exactly the multiples of `p[i]`. The union of these
arithmetic progressions, de-duplicated, is the object we index into. Two families of approach are on
the table before committing to one:

- **Direct k-way merge.** Keep the next pending firing time of each machine in a heap, pop the
  smallest, skip duplicates, and stop at the `K`-th distinct value. This is `O(K log n)` and easy to
  trust, but with `K` up to `10^9` it is far too slow for the real bounds — it is, however, an ideal
  *reference* to check a faster method against on small inputs.
- **Binary search on the answer.** Define `C(x) =` number of distinct firing times in `[1, x]`. The
  function `C` is non-decreasing in `x`, so the smallest `x` with `C(x) >= K` is exactly the `K`-th
  distinct firing time. Each evaluation of `C(x)` is the crux: it must count each shared instant once,
  which calls for inclusion-exclusion over the periods (subtract pairwise coincidences, add back the
  triple coincidence). The open question is the exact sign pattern, the `lcm` overflow, and the
  binary-search boundary.

## Evaluation settings

Judged on hidden tests covering: a single machine; two and three machines; periods that are pairwise
coprime, that share factors, and that are equal or divide one another (heavy coincidence); `K = 1`;
`K` near `10^9` with periods near `10^9` (so the answer approaches `10^18` and `lcm`s overflow 64-bit
unless guarded); and degenerate periods such as `p[i] = 1` (every integer is a firing time).

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int main() {
    int n;
    if (!(cin >> n)) return 0;
    vector<ll> p(n);
    for (auto &x : p) cin >> x;
    ll K;
    cin >> K;

    // TODO: find the K-th smallest distinct multiple-of-some-p[i] via binary search on the
    // answer, counting distinct firing times <= x with inclusion-exclusion over the periods.
    ll answer = 0;

    cout << answer << "\n";
    return 0;
}
```
