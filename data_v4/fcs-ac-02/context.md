# Construct a graph with exactly k triangles

## Research question

Given a single non-negative integer `k`, construct **any** simple undirected graph whose number
of triangles (3-cliques, i.e. unordered triples of vertices that are pairwise adjacent) is
**exactly** `k`. You may use as many or as few vertices as you like, up to a budget of `1000`
vertices. The graph must be simple: no self-loops, no repeated edges.

This is a *constructive* problem, not a search or optimization: there is no objective to maximize,
only an exact target to hit. The deliverable is an explicit graph (a vertex count and an edge list),
and it is accepted if and only if a direct triangle count of the emitted graph equals `k`. A solution
always exists in the stated range, so a correct program never needs to give up.

The interesting tension is that the triangle count of a graph does not move in convenient unit steps:
adding one edge to a dense graph can create many triangles at once, and the set of triangle counts
achievable on a *fixed* small number of vertices has gaps (for example, no graph on 4 vertices has
exactly 3 triangles). A construction therefore has to control the count precisely while staying
inside the vertex budget.

## Input / output contract

- Input (stdin): a single integer `k` with `0 <= k <= 10^8`.
- Output (stdout):
  - First line: two integers `n` and `m` — the number of vertices (`1 <= n <= 1000`) and the number
    of edges (`m >= 0`).
  - Next `m` lines: two integers `u v` (`1 <= u, v <= n`, `u != v`) describing an undirected edge.
    No edge may be repeated (in either orientation).
  - The emitted graph must contain exactly `k` triangles.
- Any valid graph with exactly `k` triangles is accepted; the checker recounts triangles directly.
- Time limit: 1 second. Memory: 256 MB.

Worked example. Input `2`. One valid output:

```
4 5
1 2
1 3
2 3
1 4
2 4
```

Vertices `{1,2,3}` form one triangle; vertex `4` is joined to `{1,2}`, creating the triangle
`1-2-4`. Total: exactly `2` triangles.

A second example. Input `7`. One valid output:

```
5 9
1 2
1 3
2 3
1 4
2 4
3 4
1 5
2 5
3 5
```

Vertices `{1,2,3,4}` form a complete graph `K4` with `C(4,3) = 4` triangles; vertex `5` is joined to
the size-3 clique `{1,2,3}`, adding `C(3,2) = 3` triangles. Total: exactly `7`.

## Background

Two facts pin down the design space.

- **A clique of size `c` has `C(c,3)` triangles.** The complete graph on `c` vertices is the densest
  possible source of triangles per vertex.
- **Joining a new vertex to a `c`-clique creates exactly `C(c,2)` new triangles** — one for every
  edge inside that clique, since the new vertex closes each such edge into a triangle. Crucially,
  joining to a *clique* gives a count that depends only on its size, not on which vertices it is.

These two formulas are the only arithmetic the construction needs. The open question, before
committing to an algorithm, is how to combine "big clique" and "attach to a sub-clique" moves so the
running triangle total lands on `k` exactly, using at most `1000` vertices for every `k <= 10^8`.

## Evaluation settings

Judged on hidden tests covering: `k = 0` (empty graph), tiny `k` (1..30), values immediately below
binomial boundaries `C(c,3) - 1` and `C(c,2) - 1` (where a naive decomposition is most likely to
misstep), medium `k` up to `2*10^5`, and the full range up to `k = 10^8` (which forces several
hundred vertices). The checker validates that the output is a simple graph with `n <= 1000` and
recounts triangles directly, so an off-by-one in the count fails immediately.

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    long long k;
    if (!(cin >> k)) return 0;

    // TODO: build a simple graph with exactly k triangles on at most 1000 vertices,
    //       then print "n m" followed by the m edges (1-indexed).

    return 0;
}
```
