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

This is a packing problem over a tiny strip, so the occupied-slot set is a natural bitmask and the whole
thing is a bitmask DP. The delicate part is the **boundary arithmetic**: a band occupies the half-open
range `[s, s+d)`, it fits exactly when `s + d <= m`, and its lowest occupied slot is `s`. Every one of
those is one `+/-1` away from a wrong-but-plausible variant, and the constraints are chosen so that those
off-by-one variants actually change the answer.

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

## Background

The occupied slots form a subset of `{0, ..., m-1}`, so a `2^m` bitmask DP is the structural fit. Two
families of approach are worth weighing before committing:

- **Greedy by profit.** Sort bands by profit and book each one if its slots are still free. This is
  `O(n log n)` and trivial, but slot packing is a global constraint: a single fat high-profit band can
  block two thin bands that together beat it, so greedy is suspect and must be stress-tested before trust.
- **Bitmask DP over occupied slots.** Let a mask encode which slots are already used. Process slots in
  increasing order: from a mask, look at its lowest free slot and either leave it empty or start a band
  whose first slot is exactly that slot. This is `O(2^m * (m + n))`-ish and is the route this problem is
  built around. The open questions are purely the boundary ones: which slots a band's mask should contain,
  what "fits" means, and which slot a band may be *started* on.

## Evaluation settings

Judged on hidden tests covering: bands that exactly reach the last slot (`s + d = m`, must be allowed);
bands that overrun by one (`s + d = m + 1`, must be rejected) including ones with enormous profit so an
off-by-one fit test is caught; a single slot (`m = 1`); no bands (`n = 0`, answer `0`); many tiny bands
that tile the strip exactly; one fat band versus several thin ones (greedy trap); and large inputs at
`m = 16` with `n` up to `2*10^5` and `p_i` near `10^9`, so the profit sum exceeds 32 bits.

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
