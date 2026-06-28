# Wall Painting

## Research question

A target picture is given as an `N x N` grid of colours, each in `0..C-1`. You
must reproduce it with a sequence of *brush strokes*. A canvas of `N x N` cells
starts filled entirely with colour `0`. Each stroke paints a solid,
axis-aligned rectangle of the canvas with one colour; strokes are applied in the
order you emit them, and a later stroke **overwrites** whatever earlier strokes
left underneath (painter's algorithm). You may use **at most `T` strokes**.

Maximise the number of canvas cells whose **final** colour equals the target.
The budget `T` is far smaller than the `N*N` cells, so cells cannot be painted
one at a time; and the target is built from overlapping coloured rectangles plus
a few per-cell flips, so it is *almost* — but never exactly — reconstructible by
a handful of strokes. There is no closed-form optimum: this is a discrete
layered-covering problem judged by a continuous match score. The whole game is
deciding *which* rectangles to paint, in *which* colour, and in *what order* —
and being willing to pull an early, wasteful stroke back so a later one need not
paint over it.

## Input / output contract

Input (stdin), all whitespace-separated integers:

- A line `N C T`: grid side `N`, palette size `C`, and the stroke budget `T`.
- Then `N` lines of `N` integers each: the target grid `g[r][c]` in `0..C-1`,
  row-major (`r` from `0` to `N-1`, `c` from `0` to `N-1`).

Output (stdout):

- A line `Q`: the number of strokes (`0 <= Q <= T`).
- Then `Q` lines `r1 c1 r2 c2 col`: paint the rectangle of rows `r1..r2` and
  columns `c1..c2` (inclusive) with colour `col`. Required:
  `0 <= r1 <= r2 < N`, `0 <= c1 <= c2 < N`, `0 <= col < C`.

The canvas starts as colour `0` everywhere; the `Q` strokes are applied in the
printed order; the final canvas is compared to the target.

Constraints (instances): `20 <= N <= 40`; `3 <= C <= 6`; `N <= T <= 3N`. Time
limit: about 2 seconds. Memory: 256 MB.

## Background

Two ideas frame the approach before committing to one.

- **Greedy layered construction.** Fill the whole canvas with the most frequent
  target colour (one stroke), then lay down a few large solid blocks for the
  dominant colour inside coarse tiles. This is fast and always feasible, but it
  is one-shot: once a block is placed it is never reconsidered, so a poorly
  sized or poorly coloured early block strands cells that later strokes must
  waste budget repainting.

- **Local search over the stroke sequence.** Keep a fixed sequence of strokes
  and repeatedly perturb one — move an edge, translate the rectangle, recolour
  it — accepting some score-reducing moves so the search can vacate a wasteful
  stroke and re-cover the region better. This can climb out of the dead ends the
  greedy walk falls into, but only if each perturbation is *cheap*: a single
  stroke sits in a stack of overlapping strokes, and naively re-deriving the
  visible colour of every cell after an edit is `O(N*N)` (or `O(T*N*N)` if you
  replay the whole stack), which is far too slow for the millions of moves a
  metaheuristic needs.

The open question is how to make the per-move score update cost proportional to
the size of the stroke being edited, not to the whole grid.

## Evaluation settings

**Score.** The local scorer (`verify/score.py`) reads the instance and the
solution, starts a canvas of colour `0`, replays the `Q` strokes in order (later
overwrites earlier), and reports the number of cells whose final colour equals
the target. Higher is better; the maximum is `N*N`.

**Feasibility -> 0 floor.** The score is `0` (the worst possible) if the output
is infeasible for ANY reason: it is malformed, truncated, or unparseable;
`Q < 0` or `Q > T` (over budget); a stroke has `r1 > r2` or `c1 > c2`
(degenerate / mis-ordered rectangle); a stroke leaves the `N x N` grid; or a
stroke's colour is outside `0..C-1`. An empty stroke list (`Q = 0`) is feasible
and scores the number of target cells that are already colour `0` — a
non-trivial but beatable floor. The reference baseline the match ratio is
reported against is the greedy layered construction above (a single full-canvas
fill with the most common colour).

**Instances.** `verify/gen.py SEED` draws `N`, `C`, `T`, paints `6..18`
overlapping random coloured rectangles onto an all-`0` grid, then flips `5%..10%`
of cells to random colours (salt-and-pepper noise). The noise and the overlaps
make exact reconstruction impossible, so an optimisation gap always exists.
Seeds are independent, so a fixed seed set (e.g. `1..20`) is a stable benchmark.

## Code framework

A single self-contained C++17 program reading stdin and writing stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int N, C, T;
    if (scanf("%d %d %d", &N, &C, &T) != 3) return 0;
    vector<int> target(N * N);
    for (int i = 0; i < N * N; i++) scanf("%d", &target[i]);  // row-major

    // TODO: choose at most T axis-aligned rectangular strokes (r1 c1 r2 c2 col)
    //       applied in order over an all-0 canvas, to maximise matching cells.
    //       Always print a FEASIBLE solution (Q = 0 is feasible).

    vector<array<int,5>> strokes;   // each: {r1, c1, r2, c2, col}
    printf("%d\n", (int)strokes.size());
    for (auto &s : strokes)
        printf("%d %d %d %d %d\n", s[0], s[1], s[2], s[3], s[4]);
    return 0;
}
```
