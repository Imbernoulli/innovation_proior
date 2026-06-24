# Echo score of a transmission (repeated fixed-length windows)

## Research question

A receiver logs one long character stream `s` of length `n`. The diagnostics team wants a single
number that measures how *redundant* the stream is at a chosen resolution `L`: slide a window of
length `L` across `s`, look at every one of the `n - L + 1` windows, and for each group of windows
that contain **exactly the same content** count how many unordered pairs of positions share that
content. Summing over all distinct contents gives the **echo score**:

```
echo(s, L) = sum over distinct length-L strings w of  C(c_w, 2)
           = sum over distinct w of  c_w * (c_w - 1) / 2
```

where `c_w` is the number of windows whose content equals `w`. Intuitively, every repeated chunk of
length `L` contributes one "echo" for each pair of places it occurs. Output `echo(s, L)`. If
`L > n` there are no windows and the score is `0`.

This is a fixed-length substring multiplicity problem. The interesting subtlety is not the idea but
the magnitude: when the stream is highly repetitive, a single content can occur on the order of `n`
times, so its pair count `C(c_w, 2)` is on the order of `n^2 / 2`, and the total can be far larger
than a 32-bit integer can hold. Getting the data types right is as load-bearing as getting the
algorithm right.

## Input / output contract

- Input (stdin): the first line has two integers `n` and `L`
  (`1 <= n <= 2*10^5`, `1 <= L <= 2*10^5`). The second line has the string `s` of length exactly
  `n`, consisting of printable non-whitespace ASCII characters (the tests use lowercase letters and
  digits).
- Output (stdout): a single line with the echo score `echo(s, L)`.
- Time limit: 1 second. Memory: 256 MB.

Example: for `n = 7`, `L = 2`, `s = "ababaab"` the windows are
`ab, ba, ab, ba, aa, ab`, so `ab` occurs 3 times, `ba` twice, `aa` once, and the echo score is
`C(3,2) + C(2,2) + C(1,2) = 3 + 1 + 0 = 4`.

## Background

Two families of approach are on the table before committing to one:

- **Sort the raw substrings.** Materialize all `n - L + 1` windows as actual strings, sort them, and
  count runs of equal strings. Trivially correct, but each comparison touches up to `L` characters,
  so this is `O(n L log n)` and `O(n L)` memory — both blow up when `L` is large (e.g. `L = 10^5`,
  `n = 2*10^5`). Good as a reference brute force, not as the shipped solution.
- **Polynomial hashing.** Give each window an integer fingerprint computed in `O(1)` from prefix
  hashes, then group windows by fingerprint. This is `O(n log n)` (or `O(n)` with a hash map) and
  `O(n)` memory regardless of `L`. The open questions are the rolling-hash formula for an arbitrary
  window `[l, l+L)`, how to make collisions negligible, and — the part that bites — what integer
  width the running counts and the final sum need.

## Evaluation settings

Judged on hidden tests covering: random strings over small and large alphabets; highly repetitive
strings (e.g. all identical characters, or a short period repeated) where one content occurs `~n`
times; `L = 1`, `L = n`, and `L > n`; `n = 1`; and large `n = 2*10^5` with `L` chosen so a single
content repeats enough that the echo score exceeds `2^31` (so a 32-bit accumulator silently
overflows). Adversarial near-collision inputs are included to punish single-modulus hashing.

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n, L;
    if (!(cin >> n >> L)) return 0;
    string s;
    cin >> s;

    // TODO: fingerprint each length-L window, group equal contents, and sum C(c_w, 2).
    long long answer = 0;

    cout << answer << "\n";
    return 0;
}
```
