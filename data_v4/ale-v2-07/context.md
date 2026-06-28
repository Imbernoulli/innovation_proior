# Grid Wire Routing

## Research question

You are given a rectangular grid of cells. Some cells are blocked obstacles; the
rest are free. On the free cells live `K` **nets**, each net being an unordered
pair of distinct free cells called its **terminals**. To *route* a net you choose
a simple path of orthogonally-adjacent free cells that starts at one terminal and
ends at the other. The hard rule is **vertex-disjointness**: no two routed nets may
share a single grid cell — so routed wires neither overlap nor cross, anywhere,
including at endpoints.

Not every net can be routed simultaneously; the instances are deliberately
over-subscribed so that the free cells are genuinely contended. The task is to
**maximize the number of nets you route** under the disjointness rule. This is the
combinatorial core of detailed routing in VLSI / FPGA design and of the
maximum vertex-disjoint-paths problem on grids — both NP-hard — so there is no
exact answer to aim for, only a continuous score to push as high as the time
budget allows.

## Input / output contract

Input (stdin):

```
H W K
<H lines, each a string of length W over {'.','#'}>   # '#' = blocked, '.' = free
<K lines, each: r1 c1 r2 c2>                           # the two terminals of net i (0-indexed)
```

All `2*K` terminals are distinct free cells. Coordinates are 0-indexed with row
`r` in `[0,H)` and column `c` in `[0,W)`.

Output (stdout): first an integer `R` (the number of nets you route), then `R`
lines, one per routed net, each of the form

```
i L r0 c0 r1 c1 ... r(L-1) c(L-1)
```

where `i` is the net index (each index at most once), `L` is the number of cells
on the path, and the `L` cells are the path in order from one terminal of net `i`
to the other (either orientation is accepted). The path must be a simple path of
free cells with consecutive cells 4-adjacent, and the cells of all `R` routed nets
must be pairwise disjoint.

Time limit: about 1.8 s wall clock. Memory: 256 MB.

## Background

Routing nets one at a time is the obvious move, but it is order-sensitive: an
early net can grab a corridor that boxes a later net out, even though a small
detour of the early net would have let both through. The established strong method
for exactly this contention is **negotiated-congestion rip-up-and-reroute**
(PathFinder, from FPGA routing): let every net route on a *shared* grid where cells
have a soft cost, then iteratively rip up nets that overlap and reroute them on a
cost field that makes contested cells expensive in proportion to how many nets want
them (present congestion) and how chronically they have been fought over
(history). Over rounds the nets "negotiate" onto private detours. A maximal
vertex-disjoint subset is then committed, and large-neighbourhood search (rip up a
few routed nets, reroute the freed ones) squeezes in more.

Two reference points bound the difficulty:

- **Sequential A\* (trivial baseline).** Route nets in input order, hard-blocking
  every already-routed cell, skipping any net that is boxed in. Always feasible,
  but strands many nets because of arbitrary ordering and irreversible blocking.
- **Distance-sorted greedy.** The same, but route short nets first. Better,
  because short nets are cheap and rarely block others — yet still one-shot and
  unable to undo a bad early corridor.

## Evaluation settings

The score of an instance is `R`, the number of nets the submission routes, **but
the entire output is validated first and any feasibility violation floors the
score to 0**. A submission is feasible iff: `R` is an integer in `[0, K]`; exactly
`R` distinct net indices are listed; every listed path has length `>= 1`, lies on
free in-grid cells, has 4-adjacent consecutive cells, visits no cell twice, and
has its two endpoints equal to that net's two terminals (in either orientation);
and across all listed nets no grid cell is used twice. If all checks pass the score
is `R`; otherwise it is `0`. The scorer (`verify/score.py`) is deterministic and
implements exactly this rule.

Instances are produced by `verify/gen.py` from an integer seed: a grid of size
`28..40` by `28..40` with `8%..18%` random obstacle cells, then `K` nets whose
count is scaled to `~0.85..1.25 * (free_cells / typical_path_length)` so the grid
is over-subscribed but mostly routable, with each net's two terminals chosen
distinct and separated by a non-trivial Manhattan distance. The reported metric is
the mean of `R` over a fixed seed set, computed by the frozen scorer; each method
is run on the same instances so the numbers are directly comparable.

## Code framework

A single self-contained C++17 program that reads the instance from stdin and writes
a feasible solution to stdout. The scaffold below reads the instance and prints a
valid (possibly empty) routing; the heuristic goes where marked.

```cpp
#include <bits/stdc++.h>
using namespace std;

int H, W, K, N;
vector<char> blk;               // blocked cell?
vector<int> sr, sc, tr, tc;     // net terminals
inline int ID(int r, int c){ return r * W + c; }

int main(){
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    if(!(cin >> H >> W >> K)){ cout << 0 << "\n"; return 0; }
    N = H * W;
    blk.assign(N, 0);
    string row;
    for(int r = 0; r < H; r++){
        cin >> row;
        for(int c = 0; c < W; c++) blk[ID(r, c)] = (row[c] == '#');
    }
    sr.resize(K); sc.resize(K); tr.resize(K); tc.resize(K);
    for(int i = 0; i < K; i++) cin >> sr[i] >> sc[i] >> tr[i] >> tc[i];

    // TODO: route as many nets as possible with vertex-disjoint paths.
    // Must ALWAYS emit a feasible (disjoint) routing within the time budget.
    vector<vector<int>> path(K);   // path[i] = cell ids of net i, or empty if unrouted

    // print a feasible solution
    vector<int> routed;
    for(int i = 0; i < K; i++) if(!path[i].empty()) routed.push_back(i);
    cout << routed.size() << "\n";
    for(int i : routed){
        auto& p = path[i];
        cout << i << " " << p.size();
        for(int v : p) cout << " " << v / W << " " << v % W;
        cout << "\n";
    }
    return 0;
}
```
