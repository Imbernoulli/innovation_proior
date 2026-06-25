# Festival roster: best partial artist-to-stage booking

## Research question

A one-day festival has `n` time-stages running in parallel and `n` candidate artists. If artist `i`
performs on stage `j` the organisers record a *net reward* `p[i][j]` — ticket and sponsorship money
minus that artist's fee and the stage's running cost. Because fees and costs can dwarf the take, a
reward may be **negative or zero**.

Each artist can perform at most once and each stage hosts at most one artist, but **nothing forces a
stage to be used and nothing forces an artist to be booked** — a stage may sit dark and an artist may
be left off the bill. You may even book nobody at all. Choose a *partial* booking (a subset of
artists, each assigned to a distinct stage) that **maximizes the total net reward**. Output that
maximum.

This is maximum-weight *partial* bipartite matching with arbitrary-sign weights. The partial-and-
signed twist is the whole point: when every reward is a loss the right answer is to book nobody and
keep `0`, and an unused stage must cost the roster nothing rather than be forced to swallow a loss.

## Input / output contract

- Input (stdin): the first token is `n` (`0 <= n <= 18`). Then follow `n` rows of `n` integers each;
  row `i`, column `j` is `p[i][j]` (`-10^9 <= p[i][j] <= 10^9`), whitespace-separated.
- Output (stdout): a single line with the maximum total net reward of any partial booking. The empty
  booking is always allowed, so the answer is at least `0`.
- Time limit: 2 seconds. Memory: 256 MB.

Example: for `n = 3` with rows `7 1 3 / 2 8 4 / 5 6 9`, the answer is `24` (artist 0 -> stage 0,
artist 1 -> stage 1, artist 2 -> stage 2: `7 + 8 + 9`).

## Background

`n` is small (`<= 18`), so an exponential-in-`n` method indexed by a subset of artists is affordable;
`2^18 = 262144` states. Two framings compete before committing to one:

- **Treat it as a full assignment.** The textbook bitmask-assignment DP books *every* stage with
  exactly one artist (a permutation) and reads `dp[full]`. It is clean, but it answers a different
  question: it cannot leave a stage empty or an artist on the bench, so on loss-heavy inputs it is
  forced to absorb negative rewards that the partial problem would simply skip.
- **Sweep stages, subset over artists.** Process stages `0..n-1` one at a time. Keep `dp[mask]` = the
  best reward after deciding the first few stages, where `mask` is the set of artists already booked.
  Each stage is either left empty (reward `+0`) or given one still-free artist. This naturally allows
  partial bookings; the open questions are the exact transition and, above all, the base case and the
  final aggregation that make the all-loss and empty inputs return `0`.

## Evaluation settings

Judged on hidden tests covering: all-positive reward matrices; matrices mixing negatives and zeros;
**all-negative** matrices (answer must be `0` — book nobody); the empty festival (`n = 0`); a single
artist and stage with a negative reward (answer `0`); matrices where leaving a particular stage dark
beats every way of filling it; and large `n = 18` with rewards near `10^9` so a full booking sum can
exceed the 32-bit range.

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    if (!(cin >> n)) return 0;
    vector<vector<long long>> p(n, vector<long long>(n));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) cin >> p[i][j];

    // TODO: maximize total net reward over all partial artist->stage matchings
    //       (each artist at most once, each stage at most one artist, empty allowed).
    long long answer = 0;

    cout << answer << "\n";
    return 0;
}
```
