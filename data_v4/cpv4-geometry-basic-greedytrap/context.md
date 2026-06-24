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

## Background

A lamp lights a length-`L` window and may be slid anywhere, so the question is how few windows cover
all the points. Two families of approach are on the table before committing to one:

- **A position-anchored greedy.** Sort the houses; repeatedly look at the leftmost still-dark house
  and drop a lamp there, then mark everything the lamp reaches and continue. The open question is the
  *anchoring rule* — where, relative to that leftmost dark house, the lamp's window should sit so the
  greedy is actually optimal. A natural-feeling choice is to put the lamp directly over the house
  (the house at the center, or even the house at the left edge), and it is `O(n log n)` and a few
  lines. Whether the *center* placement is optimal is exactly what must be checked.
- **Discretized covering DP.** Because coordinates and `L` are integers, every distinct coverage
  pattern of a window is realized by some integer left-anchor in a bounded range; a DP over sorted
  houses that tries each such anchor is obviously correct but `O(n * range)` — fine as an oracle on
  small inputs, far too slow for `n = 2*10^5`.

## Evaluation settings

Judged on hidden tests covering: a single house, many houses at one position (duplicates),
`L = 0` (a lamp lights only its single anchor point, so the answer is the number of distinct
positions), very large `L` (one lamp suffices), negative and positive coordinates mixed, clustered
versus spread layouts, and large `n = 2*10^5` with coordinates near `10^9` (so the lamp count and
the arithmetic `x[i] + L` must not overflow 32 bits).

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
