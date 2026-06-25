# Counting balanced windows by parity of XOR popcount

## Research question

A monochrome sensor returns a strip of `n` raw readings `a[0..n-1]`, each a non-negative 30-bit
integer. The calibration pipeline declares a contiguous window `[l, r]` **balanced** when the
bitwise XOR of all readings inside it,

```
a[l] XOR a[l+1] XOR ... XOR a[r],
```

has an **even number of set bits** (an even popcount). A single reading is a window of length one,
and its XOR is just that reading. You must report **how many of the `n(n+1)/2` contiguous windows
are balanced**.

The catch is the predicate: "even popcount of the XOR" is a parity-of-bits condition, not a
parity-of-value condition, and not an `XOR == 0` condition. The whole problem turns on translating
that predicate into something a prefix scan can count in one pass without re-deriving the XOR of
every window from scratch.

## Input / output contract

- Input (stdin): the first token is `n` (`0 <= n <= 2*10^5`); then `n` integers `a[i]`
  (`0 <= a[i] < 2^30`), whitespace-separated.
- Output (stdout): a single line with the number of balanced windows.
- Time limit: 1 second. Memory: 256 MB.

Example: for `a = [3, 1, 7, 4, 2, 5]` the answer is `11`.

The balanced windows are `(0,0)`, `(0,2)`, `(0,4)`, `(0,5)`, `(1,2)`, `(1,4)`, `(1,5)`, `(2,3)`,
`(3,4)`, `(3,5)`, `(5,5)` (using 0-based inclusive `(l,r)`); window `(1,4)` even has XOR `0`, whose
popcount `0` is even, and the single reading `(0,0) = 3` has popcount `2`, also even.

## Background

Two facts about windows and bits are in play, and exactly one of them is safe to lean on:

- **Prefix XOR.** Define `P[0] = 0` and `P[k] = a[0] XOR ... XOR a[k-1]`. Then the XOR of window
  `[l, r]` equals `P[r+1] XOR P[l]`. This converts every window into a pair of prefix values, which
  is the standard route to an `O(n)` counting scan — provided the predicate factors through the
  pair nicely.
- **Popcount parity.** The quantity that decides "balanced" is the parity of the number of set bits
  of that window XOR. Whether this parity is a clean function of the two endpoint prefixes — and in
  particular whether it can be confused with the *value* parity (the lowest bit) of the window XOR —
  is the crux, and is the kind of bit identity that is very easy to assert and get wrong.

The answer can be as large as `C(2*10^5 + 1, 2) ≈ 2*10^10`, which overflows 32-bit integers, so the
count must be accumulated in 64-bit.

## Evaluation settings

Judged on hidden tests covering: the empty strip (`n = 0`), a single reading (both an even-popcount
and an odd-popcount value), strips of tiny values `0..3` where value-parity and popcount-parity
frequently disagree, wide values near `2^30 - 1`, all-equal strips (every prefix lands in one parity
class so the count hits the `C(n+1, 2)` extreme and stresses 64-bit accumulation), and large random
strips at `n = 2*10^5`.

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
    // read a[0..n-1], each in [0, 2^30)

    // TODO: count contiguous windows whose XOR has an even number of set bits.
    long long answer = 0;

    cout << answer << "\n";
    return 0;
}
```
