# Minimum questions to find a hidden number with one possible lie

## Research question

A hidden integer `x` lies somewhere in `[1, N]`. You may ask yes/no questions of
the form "is `x` in the set `S`?" An adaptive adversary answers each question, and
**at most one answer in the entire game may be a lie** (the adversary chooses if
and when to lie). You must be able to name `x` with certainty no matter how the
adversary plays. Output `q1(N)`: the **minimum number of questions** that always
suffices.

This is the one-lie case of the Rényi–Ulam searching game (equivalently, optimal
binary error-correcting coding with noiseless feedback).

## Input / output contract

- Input (stdin): the first token is `T` (`1 <= T <= 10^5`), the number of
  independent queries. Then `T` integers `N` (`1 <= N <= 10^9`), one per query.
- Output (stdout): for each query, a single line with `q1(N)`.
- Time limit: 1 second. Memory: 256 MB.

Example: for `N = 3` the answer is `5`, and for `N = 1` the answer is `0`
(the value is already known, so no question is needed).

## Evaluation settings

Judged on hidden tests covering: tiny ranges `N = 1, 2, 3`, boundary ranges around
powers of two, the maximum `N = 10^9`, and large batches `T = 10^5` to exercise
throughput. The reference value `q1(10^6) = 25` is a known checkpoint.

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int T;
    if (!(cin >> T)) return 0;
    while (T--) {
        long long N;
        cin >> N;

        // TODO: compute q1(N), the minimum number of yes/no questions that
        // always identifies a hidden x in [1..N] when at most one answer lies.
        long long answer = 0;

        cout << answer << "\n";
    }
    return 0;
}
```
