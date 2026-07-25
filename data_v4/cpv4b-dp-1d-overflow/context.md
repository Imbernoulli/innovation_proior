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

## Input / output contract

- Input (stdin): two integers `n` and `m` on the first line, whitespace-separated, with
  `0 <= n <= 2*10^5` and `0 <= m <= 10^9`.
- Output (stdout): a single line with the number of valid score sequences, taken modulo `1000000007`.
- Time limit: 1 second. Memory: 256 MB.

Example: for `n = 3`, `m = 2` the answer is `22`. (Each round scores in `{0, 1, 2}`; out of the
`3^3 = 27` sequences, the `5` that contain two adjacent `2`s are excluded.)

## Evaluation settings

Judged on hidden tests covering: the small worked sample; `n = 0`; `n = 1`; `m = 0`; and large
stress tests with `n = 2*10^5` and `m = 10^9`.

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
