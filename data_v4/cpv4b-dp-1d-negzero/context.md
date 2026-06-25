# Best deployment window for a delivery drone with one allowed re-route

## Research question

A delivery drone flies a fixed corridor of `n` stops in order. At each stop `i` it either gains or
loses charge: a signed integer `a[i]` (negative means that leg drains more than it harvests). The
operator commits the drone to one **contiguous** block of consecutive stops `[l..r]` — once it lifts
off it cannot land until the block is done — and wants the block whose **net charge is largest**.

There is one operational allowance: inside the chosen block the operator may **re-route around at
most one stop**, removing that single stop's contribution from the total (a drone can detour past one
waypoint, but only one, and only if at least one stop of the block is still actually flown). The drone
must fly **at least one stop** (an empty mission is not a mission). Output the maximum net charge.

This is a one-dimensional dynamic-programming question of the "maximum contiguous segment" family,
sharpened by two corners that trip people up: every reading can be negative or zero, and when the
whole corridor is draining the answer is the *least bad* single stop — **not** zero — because a
non-empty block is mandatory.

## Input / output contract

- Input (stdin): the first token is `n` (`1 <= n <= 2*10^5`); then `n` integers `a[i]`
  (`-10^9 <= a[i] <= 10^9`), whitespace-separated.
- Output (stdout): a single line with the maximum achievable net charge.
- Time limit: 1 second. Memory: 256 MB.

Example: for `a = [3, -1, 4, -1, 5, -9]` the answer is `11`: fly the block `[3, -1, 4, -1, 5]`
(stops 0..4, summing to `10`) and re-route around one `-1`, giving `11`.

## Background

The unconstrained "largest contiguous net charge" is the classic linear scan: carry the best block
*ending here* and either extend it or restart. The re-route allowance adds exactly one bit of state —
"have I already spent my single skip?" — so the natural formulation tracks two running quantities at
each stop:

- the best block ending here with **no** skip used yet, and
- the best block ending here with the skip **already** used.

The subtlety is entirely in the corners and the signs. Because a block must be non-empty, the running
"best ending here" can never be reset to `0`; because a skip requires something else to remain, the
"skip used" state is undefined at the very first stop; and because charges run to `10^9` over `2*10^5`
stops, the totals overflow 32-bit arithmetic. Getting the base cases and the answer-combination right
on all-negative and single-stop inputs is the crux.

## Evaluation settings

Judged on hidden tests covering: all-positive corridors (re-route never helps), corridors mixing
negatives and zeros, a single stop (`n = 1`, where no skip is ever possible), all-negative corridors
(answer is the maximum element, a negative number), blocks where the optimal move is to skip one deep
negative bridging two positive runs, and large `n = 2*10^5` with charges near `10^9` so the running
total exceeds a 32-bit integer.

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    if (!(cin >> n)) return 0;
    vector<long long> a(n);
    for (auto &x : a) cin >> x;

    // TODO: best net charge over a contiguous non-empty block, re-routing around
    //       at most one interior stop (skip requires >=1 stop still flown).
    long long answer = 0;

    cout << answer << "\n";
    return 0;
}
```
