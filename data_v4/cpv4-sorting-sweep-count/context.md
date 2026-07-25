# Counting close pairs on a circular track

## Research question

There are `n` runners standing on a circular track of integer circumference `L`. Runner `i` stands at
position `p[i]` measured clockwise from a fixed start line, with `0 <= p[i] < L`. Several runners may
share a position. The **circular distance** between two runners is the shorter of the two arcs joining
them: if `d = |p[i] - p[j]|`, the circular distance is `min(d, L - d)`.

Count the number of **unordered pairs** of runners whose circular distance is at most `D`. Output that
count.

## Input / output contract

- Input (stdin): the first line has three integers `n`, `L`, `D`
  (`0 <= n <= 2*10^5`, `1 <= L <= 10^9`, `0 <= D <= L`).
  The second line has `n` integers `p[i]` (`0 <= p[i] < L`), whitespace-separated. When `n = 0` the
  second line is empty or absent.
- Output (stdout): a single line with the number of unordered pairs `{i, j}` (`i != j`) whose circular
  distance `min(|p[i]-p[j]|, L-|p[i]-p[j]|)` is `<= D`.
- Time limit: 1 second. Memory: 256 MB.

Example: for `n = 4`, `L = 10`, `D = 2`, positions `p = [0, 1, 5, 9]`, the answer is `3`: the pairs
`{0,1}` (distance 1), `{0,9}` (distance `min(9,1)=1`, around the back), and `{1,9}` (distance
`min(8,2)=2`, around the back). The other three pairs sit at distance 4 or 5.

## Evaluation settings

Judged on hidden tests covering: `D = 0`, `2*D >= L` (including `D = L`), heavy duplicate positions,
`n = 0` and `n = 1`, all runners at the same spot, and large `n = 2*10^5` with `L` near `10^9`.

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    long long L, D;
    if (!(cin >> n >> L >> D)) return 0;
    vector<long long> p(n);
    for (auto &x : p) cin >> x;
    sort(p.begin(), p.end());

    // TODO: count unordered pairs whose circular distance min(|pi-pj|, L-|pi-pj|) is <= D.
    long long answer = 0;

    cout << answer << "\n";
    return 0;
}
```
