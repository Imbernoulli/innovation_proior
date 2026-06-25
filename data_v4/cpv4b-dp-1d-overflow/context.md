# Counting score sequences with no two consecutive peaks

## Research question

A solo arcade run lasts `n` rounds. In each round you record an integer score in the inclusive
range `0..m`. The machine's marquee flashes only on a **peak round**, defined as a round whose score
equals the maximum `m`. The cabinet has one rule baked into its firmware: it refuses to flash the
marquee twice in a row, so **no two consecutive rounds may both be peak rounds** (i.e. you may never
have `score[i] == m` and `score[i-1] == m` for adjacent rounds). Any other pattern of scores is
allowed, and every score is chosen independently subject only to that one adjacency rule.

Count how many distinct score sequences of length `n` satisfy the rule. The count grows astronomically,
so report it modulo `1000000007`.

This is a one-dimensional counting DP: the only thing the future of the sequence cares about is whether
the round just placed was a peak. Getting it right means nailing the two-state recurrence *and* the
arithmetic, because both the per-round multiplier `m` and the running counts are near `10^9`, so a
naive product silently overflows a 32-bit integer long before the modulo is taken.

## Input / output contract

- Input (stdin): two integers `n` and `m` on the first line, whitespace-separated, with
  `0 <= n <= 2*10^5` and `0 <= m <= 10^9`.
- Output (stdout): a single line with the number of valid score sequences, taken modulo `1000000007`.
- Time limit: 1 second. Memory: 256 MB.

Example: for `n = 3`, `m = 2` the answer is `22`. (Each round scores in `{0, 1, 2}`; out of the
`3^3 = 27` sequences, the `5` that contain two adjacent `2`s are excluded.)

## Background

Two facts pin down the structure. First, a non-peak round can hold any of the `m` scores `0..m-1`,
while a peak round holds exactly the one score `m`. Second, the only constraint links a round to its
immediate predecessor. So the natural state is "did the previous round peak?", giving a two-track
linear scan over the `n` rounds.

The arithmetic is where this problem bites. After the modulo, a running count can be as large as
`10^9`, and the per-step multiplier `m` can also be `10^9`. Their product is on the order of `10^18`,
which fits in a 64-bit integer but absolutely does not fit in 32 bits (whose ceiling is about
`2.1*10^9`). A solution that does the multiplication in `int` produces garbage on large inputs while
looking perfectly correct on the small worked sample, where every intermediate value stays tiny.

## Evaluation settings

Judged on hidden tests covering: the small worked sample; `n = 0` (the empty sequence, which counts
as exactly one valid sequence); `n = 1` (every score is allowed, so `m + 1` sequences); `m = 0`
(every round is forced to score `0`, which is itself the peak, so for `n >= 2` no valid sequence
exists and the answer is `0`); and large stress tests with `n = 2*10^5` and `m = 10^9`, where any
32-bit intermediate in the recurrence overflows.

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    const long long MOD = 1000000007LL;
    long long n, m;
    if (!(cin >> n >> m)) return 0;

    // TODO: count length-n score sequences over 0..m with no two consecutive
    //       peak rounds (score == m), modulo MOD.
    long long answer = 0;

    cout << answer << "\n";
    return 0;
}
```
