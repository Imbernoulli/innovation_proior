# Reconfiguration Routing on a Grid (token sliding)

## Research question

You are given an `H x W` grid with walls. On the free cells sit `k` labelled
**tokens**, each at a distinct start cell; each token has a distinct target cell.
An **action** is a pair `(token, direction)`: one token steps one cell in one of
the four directions `U/D/L/R` into an adjacent cell. A step is legal only if the
destination is on the grid, is not a wall, and is **not currently occupied by
another token** — tokens block each other, and no two tokens may ever share a
cell. You must output a sequence of actions that brings **every** token from its
start to its target. The objective is to **minimize the total number of
actions**.

This is the *parallel token-sliding / multi-agent path-finding (MAPF)* problem:
reconfigure a labelled multiset of tokens from one placement to another, one
unit step at a time, with mutual exclusion. It is NP-hard to minimize the total
number of moves, so the task is a continuous-score heuristic optimization — there
is no exact answer to hit, only a plan whose move count we want as small as
possible while staying feasible.

## Input / output contract

- **Input (stdin):**
  - Line 1: three integers `H W k`.
  - Next `H` lines: the grid, each a string of `W` characters; `#` is a wall and
    `.` is a free cell.
  - Next `k` lines: `sr sc tr tc` for token `i` (0-indexed) — start `(sr, sc)`
    and target `(tr, tc)`, all on free cells, starts pairwise distinct, targets
    pairwise distinct.
- **Output (stdout):**
  - Line 1: `L`, the number of actions.
  - Next `L` lines: each `i d`, where `i` is a token index (`0 <= i < k`) and `d`
    is one of `U D L R`.
- **Time limit:** ~2 seconds. **Memory:** 256 MB.

Constraints in the generated instances: `12 <= H, W <= 30`; a moderate wall
density with 1-wide dead-ends eroded away; the free region is a single
4-connected component; `2 <= k <= floor(nfree/4)` and capped at 40, so there are
always at least three blank cells per token (room to slide tokens past one
another). A feasible joint reconfiguration always exists by construction.

## Background

Two reference strategies bracket the problem:

- **Sequential one-at-a-time mover (baseline).** Route the tokens one after
  another: send token 0 all the way to its target, then token 1, and so on,
  treating the other tokens as obstacles and shoving any blocker aside through
  the blank space (a sliding-puzzle maneuver). It is simple and always feasible
  on a connected board with a blank, but it pays for every blocker it has to
  detour around or park, so its total move count drifts well above the
  unavoidable minimum.

- **Prioritized planning with space-time A\* (the strong method).** Order the
  tokens by a difficulty score and plan each, in that order, with an A\* search
  over `(cell, time)` that avoids the *reserved space-time* of every
  higher-priority token (its cell at each time step), with reservations stored in
  a hash set for O(1) collision tests. Tokens move effectively in parallel and
  each follows close to its own shortest path, so the total move count sits near
  the lower bound. Deadlocks are resolved by re-prioritizing (a "windowed
  re-plan"): bump the token that failed to the front and retry. This space-time
  reservation scheme is the established non-obvious method for MAPF and is the
  lever this datapoint is about.

The natural per-token lower bound is the sum, over tokens, of the single-token
shortest-path distance from start to target (BFS over free cells, ignoring the
other tokens). No feasible plan can use fewer total moves than that.

## Evaluation settings

A deterministic local **scorer** replays the action sequence from the start
placement, enforcing every rule:

- each action's token index and direction are valid;
- each step lands on the grid, not a wall, and **not on a cell currently occupied
  by another token** (a collision floors the score);
- after all `L` actions, **every** token is exactly on its target.

If any rule is violated — an illegal/colliding move, or a token not on its target
at the end, or a wasted move when the optimum is zero — the score is **0** (the
feasibility floor). Otherwise, with `LB` the sum of single-token shortest-path
distances, the score is

```
score = 1000 * LB / max(L, LB).
```

Because `L >= LB` for any feasible plan, the score lies in `(0, 1000]`: a plan
that achieves the per-token lower bound scores 1000, one that uses twice the
lower bound scores about 500, and an infeasible plan scores 0. Higher is better.
Instances are produced by a fixed generator parameterized by an integer seed
(random grid + walls with dead-ends eroded, a single connected free region, and
random distinct start/target cells under the blank-margin rule above). The seed
set, generator, and scorer are frozen, and every solver is run on the same
instances so the mean score is directly comparable.

## Code framework

A single self-contained C++17 program reads the instance and writes a feasible
action sequence.

```cpp
#include <bits/stdc++.h>
using namespace std;

int H, W, K;
vector<string> grid;             // '#'=wall, '.'=free
vector<int> sr, sc, tr, tc;      // starts / targets

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    if (!(cin >> H >> W >> K)) { cout << 0 << "\n"; return 0; }
    grid.assign(H, string(W, '.'));
    for (int i = 0; i < H; i++) cin >> grid[i];
    sr.resize(K); sc.resize(K); tr.resize(K); tc.resize(K);
    for (int i = 0; i < K; i++) cin >> sr[i] >> sc[i] >> tr[i] >> tc[i];

    // TODO: produce a feasible action sequence (token, direction) that moves every
    // token from its start to its target with no two tokens ever on the same cell,
    // minimizing the number of actions.
    vector<pair<int,char>> actions;

    cout << actions.size() << "\n";
    for (auto &a : actions) cout << a.first << " " << a.second << "\n";
    return 0;
}
```
