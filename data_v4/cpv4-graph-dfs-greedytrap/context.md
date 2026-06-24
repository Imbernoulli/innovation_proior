# Most prestigious reading path through a citation graph

## Research question

A literature database holds `n` papers, numbered `0..n-1`. Paper `i` carries an integer
**prestige** `p[i]` (it may be negative: a retracted or contested paper can hurt your standing).
The database also stores `m` directed **citation links**; a link `u -> v` means "after reading paper
`u` you are allowed to read paper `v` next". The citation graph is **acyclic** (a paper can only cite
work that already existed, so links always point from newer to older — no directed cycle can form).

You will pick one paper to start reading, then repeatedly follow a citation link to read exactly one
next paper, and you may **stop at any time**. The papers you read form a directed path; your *score*
is the sum of the prestige values of the papers on that path, each counted once. You must read at
least the starting paper. Output the **maximum score** achievable over all choices of start paper and
all paths from it.

This is a longest-(weighted)-path problem on a DAG. It is the kind of subproblem that shows up inside
dependency planning, build-order optimization, and trace-cost analysis, so getting the directed-graph
version exactly right — including the negative-prestige and must-read-one corners — matters.

## Input / output contract

- Input (stdin):
  - The first line has two integers `n` and `m` (`1 <= n <= 2*10^5`, `0 <= m <= 2*10^5`).
  - The second line has `n` integers `p[0..n-1]` (`-10^9 <= p[i] <= 10^9`).
  - Each of the next `m` lines has two integers `u v` (`0 <= u, v < n`, `u != v`) describing a
    directed citation link `u -> v`. The graph is guaranteed acyclic. Parallel links may appear.
- Output (stdout): a single line with the maximum achievable score.
- Time limit: 1 second. Memory: 256 MB.

Example: for the graph in the Background section the answer is `22`.

## Background

Because you may stop whenever you like, the only real decision at each paper is *which* single
out-neighbour (if any) to continue into. Two families of approach are on the table before committing
to one:

- **Greedy by next-prestige.** From the current paper, always step to the reachable neighbour with the
  largest prestige `p[v]` (taking it only while it keeps the running score climbing); try every start
  and keep the best walk. It is near-linear and a dozen lines. The open question is whether a locally
  most-prestigious next hop is ever the wrong move under the global path constraint.
- **DFS-based dynamic programming on the DAG.** Define `best[u]` = the maximum score of a path that
  *starts* at `u`, and compute it by a depth-first traversal that solves successors before predecessors.
  This is `O(n + m)`; the open question is the exact recurrence (how the "stop here" option and the
  negative values enter it) and how to run the DFS without overflowing the call stack at `n = 2*10^5`.

Worked example (the documented sample). Six papers with prestige
`p = [3, 8, 2, 9, 1, 7]` and citation links
`0->1, 0->2, 2->3, 3->4, 1->5, 4->5`. The best reading path is `0 -> 2 -> 3 -> 4 -> 5` scoring
`3 + 2 + 9 + 1 + 7 = 22`.

## Evaluation settings

Judged on hidden tests covering: all-positive prestige, prestige with negatives and zeros, a single
paper (`n = 1`) with no links, isolated papers (`m = 0`), all-negative prestige (the answer is the
single largest, possibly negative, value because at least one paper must be read), long induced chains
of length `~2*10^5` (so a recursive DFS would overflow the stack and the running score can exceed a
32-bit integer), and dense DAGs with up to `2*10^5` links.

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n, m;
    if (!(cin >> n >> m)) return 0;
    vector<long long> p(n);
    for (auto &x : p) cin >> x;
    vector<vector<int>> adj(n);
    for (int e = 0; e < m; e++) {
        int u, v;
        cin >> u >> v;
        adj[u].push_back(v);
    }

    // TODO: compute the maximum score of a directed path in this DAG, where the
    // score of a path is the sum of p[] over its nodes, you may start anywhere,
    // you may stop anywhere, and you must read at least one paper.
    long long answer = 0;

    cout << answer << "\n";
    return 0;
}
```
