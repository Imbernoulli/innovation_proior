# Minimum lamps to light every house on a straight road

## Research question

A road is a straight line (the real axis). There are `n` houses at integer positions
`x[0..n-1]` on the road. You install identical streetlamps; a lamp may be placed at **any real
position** `s` on the road, and it lights the closed segment `[s, s + L]` of length `L` (the lamp's
"left edge" sits at `s`). A house at position `p` is lit if some lamp's segment contains `p`, i.e.
`s <= p <= s + L` for that lamp.

Place the **fewest** lamps so that **every** house is lit, and output that minimum count. A house
exactly on the boundary of a segment counts as lit (the segment is closed). Several houses may share
the same position.

This is the one-dimensional "cover points by fixed-length intervals" problem dressed as street
lighting. It is the kernel that appears inside interval-covering, radar-placement, and
sensor-coverage tasks, so getting the placement rule and the boundary handling exactly right
matters.

## Input / output contract

- Input (stdin): the first line has two integers `n` and `L`
  (`0 <= n <= 2*10^5`, `0 <= L <= 2*10^9`); the second line has `n` integers `x[i]`
  (`-10^9 <= x[i] <= 10^9`), whitespace-separated. When `n = 0` the second line may be empty or
  absent.
- Output (stdout): a single line with the minimum number of lamps. With `n = 0` the answer is `0`.
- Time limit: 1 second. Memory: 256 MB.

Example: for `n = 6`, `L = 5`, houses `[2, 3, 9, 9, 14, 20]` the answer is `3`.

## Evaluation settings

Judged on hidden tests covering: a single house, many houses at one position (duplicates),
`L = 0`, very large `L`, negative and positive coordinates mixed, clustered versus spread layouts,
and large `n = 2*10^5` with coordinates near `10^9`.

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    long long L;
    if (!(cin >> n >> L)) return 0;
    vector<long long> x(n);
    for (auto &v : x) cin >> v;

    // TODO: compute the minimum number of length-L closed intervals that cover all houses.
    long long lamps = 0;

    cout << lamps << "\n";
    return 0;
}
```
