# Parity-Invariant Reachability on a sliding-tile board

## Research question

You are given two configurations of a generalized sliding-tile puzzle on an `R x C`
board (the `R = 4, C = 4` case is the classic 15-puzzle). The board holds the values
`0, 1, ..., R*C-1`, where `0` denotes the empty cell (the "blank") and the other values
are distinct labelled tiles. A single **move** slides a tile that is orthogonally
adjacent to the blank into the blank's cell — equivalently, the blank swaps places with
one of its up/down/left/right neighbours.

Given a start board `A` and a target board `B` (both permutations of the same value set
`{0, 1, ..., R*C-1}`), decide whether some sequence of moves transforms `A` into `B`.
Print `YES` if it is reachable and `NO` otherwise.

The number of distinct board states is `(R*C)!`, so any approach that searches the state
graph is hopeless at the stated scale. The whole problem is to find a property that every
move preserves and that, conversely, certifies reachability — and to compute it in linear
time.

## Input / output contract

- Input (stdin):
  - The first line contains two integers `R` and `C` (`2 <= R, C` and `R*C <= 10^6`).
    The board is a genuine two-dimensional grid: both `R >= 2` and `C >= 2`.
  - The next `R` lines each contain `C` integers: board `A` in row-major order.
  - The following `R` lines each contain `C` integers: board `B` in row-major order.
  - Both `A` and `B` are permutations of `{0, 1, ..., R*C-1}` (value `0` is the blank).
- Output (stdout): a single line, `YES` if `A` can be transformed into `B` by legal
  moves, otherwise `NO`.
- Time limit: 2 seconds. Memory: 256 MB.

Worked example:

```
Input
4 4
1 2 3 4
5 6 7 8
9 10 11 12
13 14 15 0
1 2 3 4
5 6 7 8
9 10 11 12
13 15 14 0

Output
NO
```

This is the famous "14-15 swap": the target differs from the solved board only by
exchanging tiles `14` and `15` while the blank stays in the corner. It is unreachable —
the puzzle that originally made this fact notorious.

A reachable example:

```
Input
2 2
1 2
3 0
1 0
3 2

Output
YES
```

Here the blank slides up one cell (tile `2` drops into the corner), a single legal move,
so the answer is `YES`.

## Background

This is the reachability question for the `(R*C-1)`-puzzle. Two structural facts about a
move are worth stating before any algorithm:

- **It is one transposition.** Swapping the blank with a neighbour exchanges exactly two
  cell contents, i.e. it is a single transposition of the arrangement viewed as a
  permutation of cells.
- **It moves the blank by one step.** The blank's grid position changes by Manhattan
  distance exactly `1` per move.

Two families of approach are on the table before committing to one:

- **State-space search.** Run BFS/DFS/A* from `A` and test whether `B` is encountered.
  This is correct and is the natural first idea, but the reachable component can contain
  `(R*C)!/2` states; even for the `4 x 4` board that is over `10^13`, and the constraints
  go far beyond that. Search cannot pass.
- **An algebraic invariant.** Identify a quantity that is unchanged by every move
  (hence equal in `A` and `B` whenever `A` reaches `B`) and that is also *sufficient* —
  matching invariants guarantee reachability. If such an invariant can be evaluated in
  `O(R*C)`, the problem collapses to comparing two numbers.

## Evaluation settings

Judged on hidden tests covering: identical boards; single-move and few-move scrambles
(`YES`); single-transposition targets with the blank fixed (`NO`); the classic
`4 x 4` 14-15 swap (`NO`); square and rectangular boards of both even and odd width and
height; tall, wide, and large boards up to `R*C = 10^6`; and pairs whose value multisets
genuinely differ (must print `NO`, never crash). Correctness is exact (`YES`/`NO`).

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    long long R, C;
    if (!(cin >> R >> C)) return 0;
    long long n = R * C;

    vector<int> A(n), B(n);
    for (auto &x : A) cin >> x;
    for (auto &x : B) cin >> x;

    // TODO: decide whether board A can be transformed into board B by legal
    // blank-slide moves, and print "YES" or "NO".

    cout << "NO\n";
    return 0;
}
```
