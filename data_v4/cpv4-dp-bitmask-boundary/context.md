# Festival main-stage line-up (non-overlapping bands on a slot strip)

## Research question

A one-day festival has a single main stage split into `m` consecutive time slots numbered `0, 1, ..., m-1`.
You are offered `n` bands. If you book band `i`, it plays a contiguous block of slots: it starts at slot
`s_i` and lasts `d_i` slots, so it occupies slots `s_i, s_i+1, ..., s_i+d_i-1` — the **half-open**
interval `[s_i, s_i+d_i)`. Booking band `i` earns profit `p_i`.

A line-up is a subset of the offered bands such that

- every booked band fits **entirely inside** the strip: all of its slots lie in `[0, m)`, i.e. `s_i + d_i <= m`; and
- no two booked bands share any slot (no overlaps).

Choose a line-up (the empty line-up is allowed) that **maximizes total profit**. Output that maximum.
Because booking nothing earns `0`, the answer is always at least `0`.

## Input / output contract

- Input (stdin): the first line has two integers `m` and `n` (`1 <= m <= 16`, `0 <= n <= 2*10^5`).
  Then `n` lines follow, each with three integers `s_i d_i p_i`
  (`0 <= s_i <= m`, `1 <= d_i <= m`, `1 <= p_i <= 10^9`). A band with `s_i + d_i > m` does not fit and
  is simply unbookable (it is part of the input but can never be chosen).
- Output (stdout): a single line with the maximum total profit of a valid line-up.
- Time limit: 2 seconds. Memory: 256 MB.

Example: `m = 4`, bands `(s,d,p) = (0,2,5), (2,2,8), (3,2,100)`. The third band would occupy slots `3,4`,
but slot `4` does not exist (`3 + 2 = 5 > 4`), so it is unbookable despite its huge profit. The best
line-up is the first two bands, slots `{0,1}` and `{2,3}`, for `5 + 8 = 13`.

## Evaluation settings

Judged on hidden tests covering a range of slot configurations and band placements near the strip's
boundaries, small cases (`m = 1`, `n = 0`), and large inputs at `m = 16` with `n` up to `2*10^5` and
`p_i` near `10^9`.

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int m, n;
    if (!(cin >> m >> n)) return 0;          // m = number of slots, n = number of bands

    // Read each band (s, d, p). A band occupies the half-open slot interval [s, s+d)
    // and is bookable only if it fits inside [0, m).
    for (int i = 0; i < n; i++) {
        long long s, d, p;
        cin >> s >> d >> p;
        // TODO: build the band's occupied-slot bitmask and keep it if it fits.
    }

    // TODO: bitmask DP over the set of occupied slots to maximize total profit
    //       of a set of pairwise slot-disjoint bands (empty line-up allowed).
    long long answer = 0;

    cout << answer << "\n";
    return 0;
}
```
