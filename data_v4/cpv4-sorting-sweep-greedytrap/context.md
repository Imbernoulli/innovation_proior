# Maximum-profit non-overlapping booking schedule

## Research question

A small co-working space rents out a single shared studio. Over one day it receives `n` booking
requests; request `i` asks to hold the studio during the half-open time interval `[s_i, f_i)` (it
occupies the room from `s_i` up to but **not** including `f_i`) and is willing to pay a profit
`p_i > 0`. The manager may **accept any subset of requests**, but the studio is a single resource, so
**no two accepted requests may overlap** in time. Two intervals that merely touch at an endpoint
(`f_i == s_j`) do **not** overlap and may both be accepted. The empty schedule (accept nothing) is
allowed, so the achievable profit is always at least `0`.

Output the **maximum total profit** the manager can collect.

## Input / output contract

- Input (stdin): the first token is `n` (`0 <= n <= 2*10^5`). Then follow `n` lines (or just
  `3*n` whitespace-separated tokens), each giving three integers `s_i f_i p_i` with
  `0 <= s_i < f_i <= 10^9` and `1 <= p_i <= 10^9`.
- Output (stdout): a single line with the maximum total profit of a set of pairwise non-overlapping
  accepted requests.
- Time limit: 1 second. Memory: 256 MB.

Example: for the four requests `[1,10) p=50`, `[1,3) p=10`, `[3,6) p=10`, `[6,10) p=10` the answer is
`50` — accepting the single long booking beats the three short ones that tile the same span
(`10 + 10 + 10 = 30`).

## Evaluation settings

Judged on hidden tests covering: the empty instance (`n = 0`), a single request, requests that all
mutually overlap, requests that tile a span so one long booking must be weighed against many short
ones, chains where touching endpoints (`f_i == s_j`) must be allowed, large profit values, and large
`n = 2*10^5` with coordinates up to `10^9`.

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    if (!(cin >> n)) return 0;
    vector<long long> s(n), f(n), p(n);
    for (int i = 0; i < n; i++) cin >> s[i] >> f[i] >> p[i];

    // TODO: compute the maximum total profit of a subset of pairwise non-overlapping requests
    //       (intervals are half-open [s,f); touching endpoints do not overlap; empty allowed).
    long long answer = 0;

    cout << answer << "\n";
    return 0;
}
```
