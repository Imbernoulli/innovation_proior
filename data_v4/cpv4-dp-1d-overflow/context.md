# Counting valid lamp displays on a one-row marquee

## Research question

A theatre marquee is a single horizontal row of `n` lamps, numbered `0..n-1`. Each lamp is either
**working** (it can be switched on or off) or **broken** (it is stuck off and can never light). A
*display* is a choice of which lamps are on. A display is **valid** when it satisfies two rules:

- no broken lamp is on, and
- no two **adjacent** lamps are both on (the wiring overloads if two neighbours light together).

Count how many valid displays the marquee admits. The "all lamps off" display is always one of them.
Output that count as a single integer.

This is a one-dimensional counting DP: the number of binary strings over a fixed line that avoid the
pattern `11` and respect a per-position "forced zero" mask. The same kernel appears inside tiling
counts, restricted-permanent computations, and transfer-matrix arguments, so pinning down the exact
recurrence — and the integer width it needs — is the point.

## Input / output contract

- Input (stdin):
  - the first token is `n` (`0 <= n <= 90`);
  - if `n > 0`, the next token is a string `s` of length exactly `n` over the alphabet `{'.', 'x'}`,
    where `s[i] = '.'` marks a working lamp and `s[i] = 'x'` marks a broken one. (When `n = 0` there
    is no second token.)
- Output (stdout): a single line with the number of valid displays.
- Time limit: 1 second. Memory: 256 MB.

Worked example. For `n = 4`, `s = "...x"` the answer is `5`. Writing `#` for an on lamp and `.` for an
off lamp, the valid displays are `....`, `..#.`, `.#..`, `#...`, `#.#.`. Lamp 3 is broken so it never
lights, and no display turns on two neighbours.

A second example: `n = 0` prints `1` (the empty marquee has exactly the one vacuous display).

## Background

There are two routes worth weighing before committing.

- **Closed form via Fibonacci.** If every lamp works, the count of `11`-avoiding strings of length `n`
  is the Fibonacci number `F(n+2)`. A formula is `O(1)`, but it does not survive the broken-lamp mask:
  a single `'x'` cuts the line into independent runs whose contributions multiply, and getting the
  bookkeeping of run lengths and the boundaries right by hand is more error-prone than it looks.
- **Linear DP over the lamps.** Scan left to right carrying, for each prefix, how many valid displays
  end with the last lamp **off** versus **on**. This is `O(n)`, handles the `'x'` mask uniformly (a
  broken lamp simply contributes `0` to the "ends on" state), and needs no special-casing of runs.

The quantity that ends up deciding the data type is the *size of the answer*, not `n`: even though `n`
is at most `90`, the count grows like a Fibonacci number and reaches into the `10^18` range, so the
accumulators must be 64-bit.

## Evaluation settings

Judged on hidden tests covering: `n = 0`; a single lamp working and broken; all-broken strips (answer
`1`); all-working strips at the upper end of `n` (where the count is astronomically large and a 32-bit
accumulator silently overflows); and mixed masks where the broken lamps split the row into several
runs.

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    if (!(cin >> n)) return 0;
    string s;
    if (n > 0) cin >> s;            // length n, '.' = working, 'x' = broken

    // TODO: count valid displays (no broken lamp on; no two adjacent lamps both on).
    long long answer = 1;

    cout << answer << "\n";
    return 0;
}
```
