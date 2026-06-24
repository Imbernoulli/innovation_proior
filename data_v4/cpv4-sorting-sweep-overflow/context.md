# Peak bandwidth of overlapping data streams

## Research question

A switch carries `n` data streams. Stream `i` is active during the **half-open** time interval
`[s_i, e_i)` — it begins transmitting at instant `s_i` and stops just *before* `e_i` — and while
active it consumes `w_i` units of bandwidth. At any real instant `t`, the **load** is the sum of
`w_i` over every stream that is active at `t` (those with `s_i <= t < e_i`). Output the maximum load
that occurs at any instant.

This is the classic "busiest moment" sweep: the load is a step function of `t`, and we want its peak.
The half-open semantics matter at a shared boundary — a stream ending exactly when another starts must
**not** be counted as overlapping it.

## Input / output contract

- Input (stdin): the first token is `n` (`0 <= n <= 2*10^5`). Then follow `n` lines (whitespace is
  irrelevant), each with three integers `s_i e_i w_i` where `0 <= s_i <= e_i <= 10^9` and
  `1 <= w_i <= 10^9`. A line with `s_i == e_i` describes an empty interval that is active at no instant.
- Output (stdout): a single line with the maximum load over all instants `t`. If `n = 0` (or every
  interval is empty) the answer is `0`.
- Time limit: 1 second. Memory: 256 MB.

Example: for the four streams `[0,5) w=3`, `[2,8) w=4`, `[2,6) w=2`, `[5,9) w=5`, the answer is `11`,
attained at `t = 5`: stream 1 has already ended (active only on `[0,5)`), and streams 2, 3, 4 give
`4 + 2 + 5 = 11`.

## Background

The load changes only at stream boundaries, so the peak is found by a **sweep line** over events:

- **Sort-and-sweep.** Emit a `+w` event at each start `s_i` and a `-w` event at each end `e_i`, sort
  the events by time, then scan left to right maintaining a running load and tracking its maximum. The
  open questions are (a) how to break ties when several events share a time, given the half-open
  intervals, and (b) what numeric type the running load needs.
- **Coordinate evaluation.** Because all weights are positive, the peak is attained at some stream's
  start time, so one could instead evaluate the load explicitly at each distinct start coordinate. That
  is `O(n^2)` and only useful as an independent check on small inputs.

## Evaluation settings

Judged on hidden tests covering: streams that share start/end boundaries (so the end-before-start tie
rule is exercised), nested and disjoint intervals, empty intervals (`s_i == e_i`), `n = 0`, a single
stream, and large `n = 2*10^5` with weights near `10^9` so the running load can reach
`2*10^5 * 10^9 = 2*10^14`, far beyond the 32-bit range.

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    if (!(cin >> n)) return 0;

    // Read the n streams (s, e, w) and build sweep events.
    // TODO: sweep over sorted events, maintaining the running load and its maximum.
    long long answer = 0;

    cout << answer << "\n";
    return 0;
}
```
