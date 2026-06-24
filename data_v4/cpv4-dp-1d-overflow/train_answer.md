**Problem.** A one-row marquee has `n` lamps (`0 <= n <= 90`), each working (`'.'`) or broken (`'x'`).
A *display* turns some lamps on; it is valid when no broken lamp is on and no two adjacent lamps are
both on. Count the valid displays (the all-off display always counts). Read `n` and, if `n > 0`, the
length-`n` mask `s` from stdin; print the count.

**Key idea — two-state prefix DP.** Scan the lamps left to right carrying two counts over the prefix:
`off` = number of valid partial displays whose last lamp is **off**, and `on` = number whose last
lamp is **on**. The only thing the next lamp constrains is whether the current lamp is on, so this pair
is a sufficient summary. Transitions, both reading the *previous* pair:

- `off_i = off_{i-1} + on_{i-1}`  (lamp `i` off: previous lamp may be anything)
- `on_i  = (s[i] == '.') ? off_{i-1} : 0`  (lamp `i` on: only if working *and* previous lamp off)

Base case (empty prefix): `off = 1` (the one empty display, treated as ending "off"), `on = 0`. The
answer over the whole strip is `off + on`.

**Correctness.** The two states partition all valid prefix displays by the state of the last lamp, so
their sum is the total. A lamp may be on only when it is working and its left neighbour is off, which
is exactly `on_i = off_{i-1}` gated by `s[i] == '.'`; a lamp may always be off after any predecessor,
giving `off_i = off_{i-1} + on_{i-1}`. A broken lamp contributes `0` to `on`, which both forbids
lighting it and cleanly severs adjacency for the next lamp (it sees a guaranteed "previous off"). On an
all-working strip the recurrence reproduces the Fibonacci count `F(n+2)` of `11`-avoiding binary
strings (`n=5 -> 13`), and on a split mask `".x..x."` it reproduces the run-product
`F(3)*F(4)*F(3) = 12` — two independent cross-checks.

**Pitfalls.**
1. *Overflow (the main trap).* `n <= 90` *looks* small, but the answer is a Fibonacci number:
   `F(92) ~ 7.5 * 10^18`. A 32-bit `int` accumulator silently overflows already at `n = 45`, where the
   true answer `2971215073` wraps to `-1323752223`, and at `n = 46` it wraps to a *plausible-looking*
   positive `512559680` (true value `4807526976`). Use `long long`; it fits the worst case under the
   signed-64 ceiling `9.22 * 10^18` (which is exactly why the bound stops at `90`, not `91`). No modulus
   and no big integers — the contract wants the exact value and `long long` holds it.
2. *In-place clobber.* Compute both new values from the old `(off, on)` via temporaries. Updating `off`
   first and then reading it for `on` builds the "lamp on" count on a state where the previous lamp was
   already on — the forbidden adjacency. (A trace of `".."` returning `4` instead of `3` exposes this.)

**Edge cases (all fall out of the base case + recurrence):** `n = 0` -> `1`; a single working lamp -> `2`;
a single broken lamp -> `1`; an all-broken strip -> `1`; two adjacent working lamps -> `3` (excludes
`##`).

**Complexity.** `O(n)` time, `O(1)` extra space.

**Code.**

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    if (!(cin >> n)) return 0;            // n = 0 -> exactly one pattern (all lamps off, vacuously valid)
    string s;
    if (n > 0) cin >> s;                  // s has length n: '.' = working lamp, 'x' = broken (forced off)

    // off = number of valid patterns for the processed prefix whose LAST lamp is OFF
    // on  = number of valid patterns for the processed prefix whose LAST lamp is ON
    // A lamp may be ON only if it is working ('.') and the previous lamp is OFF.
    long long off = 1, on = 0;            // empty prefix: 1 pattern, ends "off" by convention
    for (int i = 0; i < n; i++) {
        long long noff = off + on;        // this lamp OFF: previous could be either
        long long non = (s[i] == '.') ? off : 0; // this lamp ON: needs working lamp + previous OFF
        off = noff;
        on = non;
    }

    cout << (off + on) << "\n";           // total valid patterns over the whole strip
    return 0;
}
```
