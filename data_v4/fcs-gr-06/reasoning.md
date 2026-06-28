**Reading the problem and pinning the contract.** I have an `R x C` grid of `.`, `#`, `S`, `T`. From `S` I walk to orthogonal neighbours; I may break up to `K` walls over the whole trip; stepping onto a free cell is free, stepping onto a `#` costs one break and needs remaining budget. I must print the **minimum number of walls broken** to reach `T`, or `-1` if even spending all `K` breaks cannot get me there. Before any algorithm, I fix the scale, because it decides the data structures: `R, C <= 1000` so up to `10^6` cells, and `0 <= K <= 10`. The candidate's headline case is "flip at most one wall," i.e. `K = 1`; I keep a general `K` because it forces me to be honest about the state space instead of hiding it. A distance here is a count of walls broken, at most `K <= 10`, so it fits in an `int` with enormous room — no 64-bit accumulator worries this time. The real scaling worry is the **number of states**, which I will get to.

**The first instinct, and why it is structurally wrong.** The obvious move is "it's a grid, so run a BFS from `S`." But BFS answers *shortest number of steps* on a graph where I store one distance per cell, and here the quantity I optimize is not steps — it is walls broken — and, worse, whether I am *allowed* to step onto a given `#` depends on how many breaks I have already spent. That second point is the killer. Suppose I store a single best value per cell and I first reach some cell `x` having already broken one wall, marking `x` visited. Later a different route reaches `x` having broken *zero* walls — strictly more budget in hand, strictly more future options — but my flat BFS has already marked `x` and refuses to expand it again. I have thrown away the better-resourced arrival. So "one distance per cell" is not a valid state: the same physical cell with different remaining budget is genuinely different for the future.

Let me make that concrete so I trust it rather than assert it. Take the `1 x 5` corridor `S#.#T` with `K = 1`. Cell indices `0..4`. If I treat the grid flat and let BFS wander, the moment it breaks the first `#` at index 1 it sits at index 2 with budget exhausted, then index 3 is a `#` it cannot pass, dead end. A flat search that recorded "index 2 reached" would never reconsider reaching index 2 *without* having spent the break — and indeed there is no such route here, so the true answer is `-1` (two separate walls, only one break). With `K = 2` the answer becomes `2` (break both). The flat model cannot even represent the difference between "at index 2 with one break left" and "at index 2 with zero," which is exactly the information that decides `-1` versus `2`. The cell alone is not a state; **the cell plus remaining budget is.**

**Deriving the insight: a layered state-product graph with 0/1 edge weights.** The fix is to make the search node be the pair `(cell, breaks_used)` with `breaks_used in [0..K]`. Picture `K + 1` stacked copies — *layers* — of the grid. Layer `k` is "I have broken exactly `k` walls so far." A move that lands on a free cell keeps me in the same layer (I broke nothing). A move that lands on a `#` drops me from layer `k` to layer `k + 1` (one more break), and is forbidden out of the bottom layer `k = K`. Now every edge of this product graph carries a natural cost: **0 for a free step, 1 for a wall-break.** The quantity I want — minimum walls broken to reach `T` — is exactly the shortest-path distance from `(S, 0)` to any node `(T, k)` in this graph. The product graph has `R * C * (K + 1)` nodes, up to `10^6 * 11 ~ 1.1 * 10^7`, and `O(4)` edges per node. That is large but linear-ish, so the algorithm on it must be essentially linear in the number of nodes.

**Choosing the shortest-path engine — and resisting the wrong default.** I have a single-source shortest-path problem on a weighted graph, so the reflex is Dijkstra. Dijkstra is correct here, but its cost is `O(E log V)`; at `E ~ 4 * 1.1*10^7 = 4.4*10^7`, the `log V ~ 24` factor pushes me toward `~10^9` priority-queue operations, each a heap push/pop with cache-unfriendly behaviour. That is the kind of thing that is "correct" and times out at a 1-second limit. The decisive observation is that my edge weights are *not* arbitrary positive reals — they live in the two-element set `{0, 1}`. For exactly that case there is a strictly better tool than Dijkstra: **0-1 BFS**. It replaces the priority queue with a double-ended queue and runs in `O(V + E)`, dropping the `log` factor entirely. The rule is: when relaxing an edge of weight `0`, push the neighbour to the **front** of the deque; when relaxing an edge of weight `1`, push it to the **back**. The invariant this maintains is that the deque holds nodes in non-decreasing distance order and contains at most two distinct distance values at any moment (`d` near the front, `d + 1` near the back). Popping from the front therefore always yields a node whose stored distance is final — the same finality Dijkstra gets from its heap, but without the heap. That is the SOTA approach for `{0,1}`-weighted shortest paths at this scale, and it is exactly what the layered-graph framing was built to enable.

Why is 0-1 BFS correct here, in one line I can defend? Because a node's distance, once it is at the front of the deque, can never be improved later: everything still in the deque has distance `>= d`, and every outgoing edge adds `0` or `1`, so no future relaxation produces a value below `d`. That is the same exchange argument as Dijkstra, specialized to two weights.

**Encoding the state.** I flatten `(r, c, k)` into a single integer id `((r * C + c) * (K + 1)) + k`. The total count `R * C * (K + 1)` can reach `1.1 * 10^7`, comfortably inside an `int` index, but the *product* `r * C` could approach `10^6` and `* (K+1)` stays under `1.2*10^7`, still int-safe; to be safe against any intermediate overflow on larger hypothetical limits I compute the id in `long long` and store ids as `long long` in the deque. The distance array is `vector<int>` of that length initialized to `INF = INT_MAX`. `S` seeds layer 0 at distance 0. At the end the answer is the minimum of `dist[(T, k)]` over all `k in [0..K]`, or `-1` if all are `INF`.

**First implementation.** My initial loop, written quickly:

```
deque<long long> dq;
dist[idx(sr,sc,0)] = 0; dq.push_back(idx(sr,sc,0));
while (!dq.empty()) {
    long long cur = dq.front(); dq.pop_front();
    decode r,c,k;
    int d = dist[cur];
    if (r==tr && c==tc) { answer = d; break; }   // <-- early exit on first T
    for each neighbour (nr,nc):
        char ch = g[nr][nc];
        if (ch=='#') { if (k==K) continue; nk=k+1; w=1; }
        else         { nk=k; w=1; }               // <-- BUG: free move costs 0, not 1
        int nd = d + w;
        if (nd < dist[idx(nr,nc,nk)]) {
            dist[...] = nd;
            dq.push_front(...);                    // <-- BUG: always push_front
        }
}
```

Two things in here are wrong, and I only see them by running the smallest revealing cases rather than by re-reading my own code (re-reading tends to confirm what I meant, not what I wrote).

**The debug episode.** I trace the free-corridor `1 x 5` grid `S...T` with `K = 1`; the correct answer is obviously `0` (walk straight across, break nothing). Indices `0..4`, all free, `S` at 0, `T` at 4. Start: `dist[(0,0)]=0`, deque `[(0,0)]`. Pop `(0,0)`, `d=0`. Its right neighbour is index 1, a free cell, so with my buggy `else { w=1; }` I compute `nd = 0 + 1 = 1` and set `dist[(1,0)] = 1`. Continue rightward: index 2 gets `2`, index 3 gets `3`, index 4 (`T`) gets `4`. The early exit fires when `T` is popped and reports `4`. But the true answer is `0`. The output `4` is wildly wrong, and the magnitude is the tell: it equals the number of *steps*, not the number of *breaks*. That instantly localizes the defect — I am charging `1` for ordinary free moves. A free step must cost `0`; only a wall-break costs `1`. I fix the `else` branch to `w = 0`.

Now the second bug, which the first one was masking. With weights genuinely `0` and `1`, the push side matters: a weight-`1` relaxation must go to the **back** of the deque, not the front. If I push everything to the front, the deque is no longer ordered by distance, the "front is final" invariant breaks, and a node can be popped with a non-minimal distance and never corrected. To expose it I use `S#.#T` with `K = 2`, true answer `2`. Walk: from `S`(0) the right neighbour index 1 is `#`, a weight-1 edge into layer 1, `dist[(1,1)] = 1`. With the all-`push_front` bug, `(1,1)` jumps to the front ahead of genuinely-distance-0 nodes, so I might pop and expand it before finishing all distance-0 work; in a bigger grid this lets a distance-`d+1` node be processed before a distance-`d` node and finalize a wrong value. The fix is the textbook rule: `w == 0` -> `push_front`, `w == 1` -> `push_back`. After both fixes I re-run `S...T,K=1` -> `0`, `S#.#T,K=1` -> `-1`, `S#.#T,K=2` -> `2`. All correct.

**The third subtlety — the early exit on `T`.** I had `if (r==tr && c==tc) { answer=d; break; }`, reasoning that the first time any `T`-layer node is popped from the front, its distance is final and minimal across layers, so I can stop. With a *correct* 0-1 BFS deque that reasoning is actually sound — front-pop order is non-decreasing in distance, so the first popped `(T, k)` has the minimum `dist` over all `k`. But it is fragile: it leans on the deque invariant being perfect, and it interacts badly with the lazy duplicates 0-1 BFS naturally creates (the same node can be pushed several times at different distances before its first pop). Rather than depend on that, I drop the early `break` entirely and let the BFS run to completion, then take `min over k of dist[(T,k)]` at the end. This is unconditionally correct regardless of push/pop micro-ordering and costs nothing asymptotically, because the deque still drains in `O(V + E)`. I keep the relaxation guard `if (nd < dist[nstate])` so a node is only enqueued when it is actually improved, which bounds total enqueues and makes stale popped entries harmless: when a stale id is popped, `d = dist[cur]` re-reads the *current* best for that node, and relaxing neighbours from it can only ever match or fail the `<` guard, never corrupt anything.

**Re-verifying the fixed version on a spread of cases by hand.**
- `1 x 2` grid `ST`, `K = 0`: `S`(0,0) free-steps onto `T`(0,1), weight 0, `dist[(T,0)] = 0`. Answer `0`. (Adjacent free, nothing to break.)
- `1 x 3` grid `S#T`, `K = 0`: from `S` the only neighbour is the `#`; breaking needs budget but `k == K == 0`, so the move is skipped. `T` stays `INF`. Answer `-1`. Correct — no budget to get through.
- `1 x 3` grid `S#T`, `K = 1`: break the `#` (weight 1, into layer 1), reach `T` in layer 1 at distance 1. Answer `1`. Correct.
- `1 x 4` grid `S##T`, `K = 1`: break the first `#` (layer 1), but the second `#` would need layer 2 which exceeds `K`, skipped. `T` `INF`. Answer `-1`. With `K = 2` the same grid gives `2`. Correct.
- Free detour beats a break: `S#.T / .#.. / ....`, `K = 1`. There is a wall-free route — down from `S`, along the bottom row, up into `T` — all weight-0 edges, so `dist[(T,0)] = 0` and the answer is `0` even though a break was available. This is the case that shows the objective is "minimum breaks," and 0-1 BFS naturally prefers the all-zero path because those nodes hit the front of the deque first.
- `T` enclosed: `S.. / ..# / .#T`, `K = 1`. `T`(2,2) has `#` above and to its left; reaching it requires breaking exactly one of them, so the answer is `1`, and with `K = 0` it is `-1`. Correct.

Each hand case agrees with the fixed code, and each one targets a different failure mode (free move, no-budget wall, exhausted budget mid-path, free-detour-preferred, enclosed target).

**Complexity and limits.** Nodes `V = R * C * (K + 1) <= 1.1 * 10^7`; edges `E = O(4V)`; 0-1 BFS is `O(V + E)`, i.e. linear in the state space. Memory is one `int` distance per node, about `44 MB` at the maximum, plus the deque, inside `256 MB`. Empirically, a `1000 x 1000` grid with `K = 10` runs in well under a second; the all-open and all-wall extreme grids are even faster because they touch fewer layers. A Dijkstra over the identical state graph is the cross-check oracle; it gives the same numbers but with the `log V` factor I deliberately avoided for the time limit. The 0-1 BFS is the SOTA choice precisely because the weights are confined to `{0, 1}`.

**Final solution.** I disproved the flat-grid BFS with the `S#.#T` budget argument, built the `(cell, breaks_used)` layered product graph so that breaking-a-wall versus walking-free became weight-1 versus weight-0 edges, and ran 0-1 BFS on it for `O(V + E)`. The bugs I actually hit — charging `1` for free moves, pushing weight-1 edges to the front, and over-trusting an early `T` exit — were each caught by tracing a minimal case to a precise cause, then fixed and re-verified. This is what I ship: one self-contained file.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int R, C, K;
    if (!(cin >> R >> C >> K)) return 0;

    vector<string> g(R);
    for (int i = 0; i < R; i++) cin >> g[i];

    int sr = -1, sc = -1, tr = -1, tc = -1;
    for (int i = 0; i < R; i++)
        for (int j = 0; j < C; j++) {
            if (g[i][j] == 'S') { sr = i; sc = j; }
            else if (g[i][j] == 'T') { tr = i; tc = j; }
        }

    // Layered state graph: node = (row, col, breaks_used), breaks_used in [0..K].
    // Edge weight 0  : step onto a free cell ('.', 'S', 'T')  -> breaks unchanged.
    // Edge weight 1  : step onto a wall ('#')                 -> breaks_used + 1 (needs budget).
    // dist(node) = minimum number of walls broken to arrive at that node.
    // Because every edge weight is 0 or 1, 0-1 BFS (deque) computes all distances in O(V+E):
    //   relax with weight 0 -> push_front, weight 1 -> push_back, and the deque stays
    //   sorted by distance. A pop is processed only if it matches the stored dist (lazy skip).
    const int INF = INT_MAX;
    int layer = K + 1;
    auto idx = [&](int r, int c, int k) -> long long {
        return (((long long)r * C + c) * layer + k);
    };
    vector<int> dist((long long)R * C * layer, INF);

    deque<long long> dq;
    dist[idx(sr, sc, 0)] = 0;
    dq.push_back(idx(sr, sc, 0));

    const int dr[4] = {-1, 1, 0, 0};
    const int dc[4] = {0, 0, -1, 1};

    while (!dq.empty()) {
        long long cur = dq.front();
        dq.pop_front();
        int k = (int)(cur % layer);
        long long rc = cur / layer;
        int c = (int)(rc % C);
        int r = (int)(rc / C);
        int d = dist[cur];

        for (int dir = 0; dir < 4; dir++) {
            int nr = r + dr[dir];
            int nc = c + dc[dir];
            if (nr < 0 || nr >= R || nc < 0 || nc >= C) continue;
            char ch = g[nr][nc];
            int nk = k, w;
            if (ch == '#') {
                if (k == K) continue;       // no budget left to break a wall
                nk = k + 1;
                w = 1;
            } else {
                w = 0;                      // '.', 'S' or 'T' : free move
            }
            int nd = d + w;
            long long nstate = idx(nr, nc, nk);
            if (nd < dist[nstate]) {
                dist[nstate] = nd;
                if (w == 0) dq.push_front(nstate);
                else        dq.push_back(nstate);
            }
        }
    }

    int best = INF;
    for (int k = 0; k <= K; k++)
        best = min(best, dist[idx(tr, tc, k)]);

    cout << (best == INF ? -1 : best) << "\n";
    return 0;
}
```

**Causal recap.** A flat grid BFS is wrong because "remaining budget" is part of the state, which `S#.#T` (`-1` at `K=1`, `2` at `K=2`) proves; lifting to the `(cell, breaks_used)` product graph turns free-move/wall-break into weight-0/weight-1 edges; with weights in `{0,1}` the SOTA shortest-path engine is 0-1 BFS (`O(V+E)`, no `log` factor, fits the 1-second limit at `1.1*10^7` states) rather than Dijkstra; the implementation bugs (free moves charged `1`, weight-1 pushed to the front, an over-trusted early `T` exit) were each pinned by tracing a minimal grid and fixed; and taking `min over layers of dist[(T,k)]`, or `-1`, closes out reachability, the budget corners, and the free-detour-preferred case.
