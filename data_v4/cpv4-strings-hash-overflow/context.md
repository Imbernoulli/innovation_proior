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

## Evaluation settings

Judged on hidden tests covering: random strings over small and large alphabets; highly repetitive
strings (e.g. all identical characters, or a short period repeated); `L = 1`, `L = n`, and `L > n`;
`n = 1`; and large `n = 2*10^5` with a range of `L` values. Adversarial inputs are included to stress
correctness.

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
