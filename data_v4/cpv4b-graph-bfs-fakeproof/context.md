# Relay grid: total bit-disagreement of broadcast hop-counts

## Research question

A signal relay network has `n` stations (numbered `1..n`) joined by `m` two-way fiber links. Station
`1` is the broadcast origin. When the origin emits a pulse it floods the network, and each station `v`
records `d[v]`, the **minimum number of links** the pulse crosses to reach it (so `d[1] = 0`). The
network is **connected**, so every station records a finite hop-count.

The maintenance team writes each hop-count as a binary number and asks a single aggregate question
about how "bit-different" the stations are from one another. For every **unordered pair** of distinct
stations `{u, v}` they look at `d[u] XOR d[v]` and count its set bits (its Hamming weight). Summing
over all `C(n,2)` pairs gives

```
S = sum over { u < v }  of  popcount( d[u] XOR d[v] ).
```

Compute `S`. This is the total number of bit positions at which two stations' hop-counts disagree,
accumulated across every pair.

The point of interest is that `S` has a clean per-bit closed form, but the per-bit formula and the
ordered/unordered bookkeeping are exactly the kind of thing that is tempting to assert and easy to get
wrong, while the obvious `O(n^2)` double loop is far too slow at the stated scale.

## Input / output contract

- Input (stdin): the first line has two integers `n` and `m`
  (`1 <= n <= 2*10^5`, `0 <= m <= 4*10^5`). Each of the next `m` lines has two integers `u v`
  (`1 <= u, v <= n`, `u != v`) describing a link. The graph is connected; it may contain multiple
  links between the same pair (they are harmless).
- Output (stdout): a single line with the integer `S`.
- Time limit: 1 second. Memory: 256 MB.

Example: for the chain `1-2-3-4-5-6` (hop-counts `0,1,2,3,4,5`) the answer is `25`.

## Background

Single-source shortest paths in an **unweighted** graph are produced by breadth-first search in
`O(n + m)`: distances come out in non-decreasing order along the BFS frontier, and each station's
hop-count is fixed the first time it is dequeued. That part is standard.

The aggregate `S` is where the design lives. Two facts are on the table before committing:

- **Per-bit decomposition.** `popcount(x XOR y)` counts the bit positions where `x` and `y` differ,
  so `S` decomposes bit by bit: `S = sum over bits b of (number of pairs that differ at bit b)`. The
  open question is the *exact* count of pairs differing at a single bit `b` in terms of how many
  hop-counts have that bit set — a counting identity that must be derived and is easy to mis-state.
- **Why not brute force.** The direct definition is an `O(n^2)` double loop over pairs. With
  `n = 2*10^5` that is `2*10^10` pair-evaluations, hopeless under one second. The closed form must
  collapse the pair sum into per-bit station tallies.

Scale also forces the data type: `S` can be on the order of `(n/2)^2` times the number of bits, i.e.
roughly `10^{14}`, which overflows 32-bit arithmetic and requires 64-bit accumulators.

## Evaluation settings

Judged on hidden tests covering: a single station (`n = 1`, answer `0`); long chains where hop-counts
span many bit positions; trees with branching so several stations share a hop-count; dense graphs with
many cycles and multi-edges; and the maximum `n = 2*10^5` with up to `4*10^5` links, where the answer
exceeds the 32-bit range.

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n, m;
    if (!(cin >> n >> m)) return 0;

    vector<vector<int>> adj(n + 1);
    for (int i = 0; i < m; i++) {
        int u, v;
        cin >> u >> v;
        adj[u].push_back(v);
        adj[v].push_back(u);
    }

    // TODO: BFS from station 1 to get hop-counts, then compute
    //       S = sum over unordered pairs {u,v} of popcount(d[u] XOR d[v]).
    long long answer = 0;

    cout << answer << "\n";
    return 0;
}
```
