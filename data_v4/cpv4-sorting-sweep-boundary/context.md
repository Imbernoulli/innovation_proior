# Peak occupancy of a parking garage (max overlapping half-open intervals)

## Research question

A parking garage logs `n` cars. Car `i` is described by an arrival time `s_i` and a departure
time `e_i`. The car occupies a spot during the **half-open** interval `[s_i, e_i)`: it is present
*at* the instant `s_i`, and it is gone *by* the instant `e_i`. The exclusive right endpoint is the
whole point of the model — a car that leaves at time `t` and a car that arrives at time `t` can
reuse the *same physical spot*, so they are **not** considered simultaneously present.

Question: across all instants of time, what is the **maximum number of cars present at once**?
That number is the minimum number of spots the garage must have to never turn a logged car away.

This is the canonical "maximum number of overlapping intervals" sweep. It shows up under many
names — peak meeting-room usage, maximum concurrent calls, train-platform requirements — and its
correctness lives or dies on one decision: how the sweep treats the instant where one interval ends
and another begins. Get the inclusive/exclusive boundary off by one and the answer is wrong only on
the inputs where endpoints coincide, which is exactly the kind of bug that survives weak testing.

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

## Background

The maximum overlap is achieved at some instant, and coverage of the time axis only ever *increases*
at an arrival and *decreases* at a departure, so it suffices to look at a finite set of events rather
than the continuous axis. Two families of approach are on the table before committing to one:

- **Coordinate sweep with `+1 / -1` events.** Emit a `+1` event at each arrival `s_i` and a `-1`
  event at each departure `e_i`, sort all `2n` events by coordinate, sweep left to right keeping a
  running count, and report the running maximum. This is `O(n log n)`. The open question — and the
  entire difficulty of the problem — is the **tie-break at a shared coordinate**: when an arrival and
  a departure land on the same instant, which is applied first? Half-open `[s, e)` demands one
  specific order; the other order silently computes the answer for *closed* intervals.
- **Difference array / bucketed counting.** If coordinates were small one could add `+1` at `s_i`,
  `-1` at `e_i` in an array and prefix-sum it. With coordinates up to `10^9` this needs coordinate
  compression, which re-introduces the same boundary question in a different disguise (which bucket
  does a departure fall into).

## Evaluation settings

Judged on hidden tests covering: disjoint intervals (answer `1`); intervals that merely *touch*,
i.e. `e_i == s_j` (must NOT count as overlapping); many intervals sharing one endpoint; nested
intervals; the empty input (`n = 0`); a single car; degenerate `s_i >= e_i` records mixed in; and
large `n = 2*10^5` with coordinates spread across `[0, 10^9]` so the event array is large and ties
are common.

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
        //       the running count so that touching intervals (e == s') do NOT overlap.
    }

    long long best = 0;
    // TODO: sort events and sweep to compute best.

    cout << best << "\n";
    return 0;
}
```
