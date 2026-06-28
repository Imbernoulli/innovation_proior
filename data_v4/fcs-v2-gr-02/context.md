# Immediate dominators of a directed graph from a fixed source

## Research question

You are given a directed graph with `n` nodes (numbered `1..n`) and `m` edges, together with a
distinguished **source** node `s`. A node `u` is said to **dominate** a node `v` if *every* path from
`s` to `v` passes through `u`. Among all dominators of a node `v` (other than `v` itself), exactly one
is dominated by all the others; this is the **immediate dominator** of `v`, written `idom(v)`, and it
is the parent of `v` in the *dominator tree* rooted at `s`.

Compute `idom(v)` for every node `v`:

- `idom(s) = 0` (the source is the root of the dominator tree and has no immediate dominator),
- `idom(v) = 0` as well for any node `v` that is **not reachable** from `s` (such a node has no
  dominators at all),
- otherwise `idom(v)` is the immediate dominator as defined above.

The dominator tree is a foundational object in compiler optimization (control-flow analysis, SSA
construction) and in network reliability (single points of failure between a source and a target), so
computing it correctly and *fast* — at graph sizes where the naive definition-driven method is
hopeless — is the point.

## Input / output contract

- Input (stdin): the first line contains three integers `n m s`
  (`1 <= n <= 2*10^5`, `0 <= m <= 2*10^5`, `1 <= s <= n`). Each of the next `m` lines contains two
  integers `a b` (`1 <= a, b <= n`) denoting a directed edge `a -> b`. Self-loops (`a == b`) and
  duplicate edges may appear.
- Output (stdout): a single line with `n` integers, the values `idom(1), idom(2), ..., idom(n)`
  separated by single spaces, where `0` marks the source and any unreachable node.
- Time limit: 1 second. Memory: 256 MB.

Example. Input

```
7 8 1
1 2
1 3
2 4
3 4
4 5
5 6
5 7
6 5
```

Output

```
0 1 1 1 4 5 5
```

Node `1` is the source (`0`). Nodes `2` and `3` are reached directly from `1`. Node `4` can be
reached through either `2` or `3`, so neither dominates it — its immediate dominator is `1`. Node `5`
is reachable only through `4`, nodes `6` and `7` only through `5`.

## Background

The definition itself suggests a method: for each candidate node `u`, delete `u` and recompute which
nodes become unreachable from `s`; those are exactly the nodes `u` dominates. From the full
domination relation one can read off each immediate dominator. This is `O(n)` graph traversals, i.e.
`O(n(n + m))` work — fine for `n` in the hundreds, but quadratic and far too slow at `n = 2*10^5`.

The structural facts that any fast method must exploit:

- Dominators are defined only along paths *from the source*, so only the part of the graph reachable
  from `s` matters; everything else has `idom = 0`.
- A depth-first spanning tree from `s` gives a preorder numbering in which every dominator of `v` is
  a DFS-tree ancestor of `v` — domination refines tree-ancestry. The interesting question is *which*
  ancestor is the immediate one, given that non-tree edges (forward, back, cross) create alternative
  routes that can "skip over" tree ancestors.

## Evaluation settings

Judged on hidden tests covering: a single node (`n = 1`, with and without a self-loop); graphs with
nodes unreachable from `s` (which must report `0`); DAGs; graphs with cycles and back-edges; multigraphs
with duplicate edges and self-loops; "diamond" shapes where two branches re-merge (so the merge point
is dominated by the split point, not by either branch); and large graphs at `n, m = 2*10^5` including a
single deep chain of length `2*10^5` (which forces any recursive traversal to overflow the call stack).

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n, m, s;
    if (!(cin >> n >> m >> s)) return 0;

    vector<vector<int>> g(n + 1);
    for (int i = 0; i < m; ++i) {
        int a, b;
        cin >> a >> b;
        g[a].push_back(b);
    }

    // TODO: compute idom[v] for every v in 1..n.
    //   idom[s] = 0 (root); idom[v] = 0 if v is unreachable from s;
    //   otherwise idom[v] = immediate dominator of v.
    vector<int> idom(n + 1, 0);

    for (int v = 1; v <= n; ++v) {
        cout << idom[v];
        cout << (v == n ? '\n' : ' ');
    }
    return 0;
}
```
