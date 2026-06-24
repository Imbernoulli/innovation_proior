# Best descent through a cave network

## Research question

A cave is a network of `n` numbered chambers `0..n-1`. Chamber `0` is the entrance. There are `m`
one-way tunnels; every tunnel goes from a shallower chamber to a deeper one, i.e. a tunnel always
connects some `a` to some `b` with `a < b`. (So the network is a DAG and you can never loop back.)

Each chamber `i` carries a value `v[i]`: a positive number is treasure, a negative number is a trap
that costs you, and `0` is an empty chamber. The values may be negative, zero, or positive.

An explorer enters at chamber `0` and walks along directed tunnels, always going deeper. Standing in a
chamber means collecting its value (you cannot avoid the value of a chamber you are in), and the
explorer may **stop at any chamber** reached. Since the explorer starts at chamber `0`, the value
`v[0]` is always collected, even if it is negative — there is no option to refuse to enter.

Output the maximum total value the explorer can collect on one such descent. Note that the answer can
be **negative**: if `v[0]` is very negative and nothing reachable can recover it, the best the
explorer can do is stop immediately at the entrance with total `v[0]`.

This is a maximum-weight path problem on a DAG, started at a fixed source, where the path may end
anywhere. It is the kind of subproblem that appears inside scheduling-on-precedence-graphs and
longest-path DP; getting the one-source version exactly right — including the negative-value,
all-negative, and single-chamber corners — is the point.

## Input / output contract

- Input (stdin):
  - Line 1: two integers `n` and `m` (`1 <= n <= 2*10^5`, `0 <= m <= 4*10^5`).
  - Line 2: `n` integers `v[0..n-1]` (`-10^9 <= v[i] <= 10^9`), whitespace-separated.
  - Next `m` lines: two integers `a b` each, describing a one-way tunnel `a -> b` with `0 <= a < b <= n-1`.
    Parallel tunnels may appear; they change nothing.
- Output (stdout): a single line with the maximum total value collectible on a descent that starts at
  chamber `0`.
- Time limit: 1 second. Memory: 256 MB.

Example: with chambers `v = [3, -5, 4, -2, 7, -8]` and tunnels
`0->1, 0->2, 1->3, 2->4, 2->5, 1->4, 3->5`, the answer is `14`: take `0 -> 2 -> 4`, collecting
`3 + 4 + 7 = 14`.

## Background

Two natural framings are on the table before committing:

- **Greedy / per-step.** From the current chamber, step into whichever reachable child looks best and
  keep going while it helps. Cheap to write, but "best next step" is a local decision on a global
  path, and a negative chamber on the way to a big treasure makes the locally-bad step globally
  optimal — so greedy needs to be stress-tested before it is trusted.
- **Memoized DFS / DAG DP.** Define `best[u]` = the most value collectible by a descent that *starts*
  at `u`. Because tunnels only go deeper (`a < b`), every child of `u` has a larger index, so the
  values can be filled in by a single pass over chambers in decreasing index order — that order is a
  valid reverse topological order, no explicit recursion or topological sort needed. The open question
  is the exact recurrence, and in particular which quantities may be clamped at `0` and which may not.

## Evaluation settings

Judged on hidden tests covering: all-positive networks; networks with negatives and zeros mixed;
single chamber `n = 1` (with `v[0]` positive, zero, and negative); all-negative networks (answer is
the least-bad reachable prefix, which can be negative); chambers whose every descendant is negative
(so stopping early wins); and large `n = 2*10^5`, `m = 4*10^5` with `|v[i]|` near `10^9`, so a path
sum can reach `~2*10^14` and overflow a 32-bit integer.

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
    vector<long long> v(n);
    for (auto &x : v) cin >> x;
    vector<vector<int>> adj(n);
    for (int i = 0; i < m; i++) {
        int a, b;
        cin >> a >> b;
        adj[a].push_back(b);   // directed tunnel a -> b, guaranteed a < b
    }

    // TODO: compute the maximum value collectible on a descent that starts at chamber 0,
    //       may stop at any chamber, and always collects v[0] (answer may be negative).
    long long answer = 0;

    cout << answer << "\n";
    return 0;
}
```
