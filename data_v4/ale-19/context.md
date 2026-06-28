# 2D Rectangle Strip Packing

## Research question

You are given `n` axis-aligned rectangles and a vertical **strip** of fixed integer width `W` and
**unbounded** height. Each rectangle must be placed inside the strip with its sides parallel to the
axes and its corners at integer coordinates; no two placed rectangles may overlap (touching edges is
fine). A global flag `R` says whether rotation is allowed: if `R = 1` a rectangle may be rotated by
90 degrees (swap its width and height) before placement; if `R = 0` every rectangle keeps its given
orientation. The task is to **minimize the used height** — the maximum top edge over all
rectangles. This is the classic *strip-packing* problem; it is strongly NP-hard, there is no exact
answer to read off, and the quality of a solution is judged by a continuous score, so the only lever
is the heuristic that decides where (and how) to put each rectangle.

## Input / output contract

- Input (stdin):
  - first line: `n W R` — number of rectangles `n` (`0 <= n <= 200`), strip width `W`
    (`1 <= W <= 1000`), rotation flag `R` (`0` or `1`);
  - then `n` lines, the `i`-th being `w_i h_i` — the native width and height of rectangle `i`
    (`1 <= w_i, h_i`). When `R = 0`, every `w_i <= W`; when `R = 1`, at least one of
    `{w_i, h_i}` is `<= W` so the rectangle fits in some orientation.
- Output (stdout): exactly `n` lines, the `i`-th being `x_i y_i r_i` — the integer bottom-left
  corner `(x_i, y_i)` of rectangle `i` (in input order) and its rotation bit `r_i in {0,1}`.
  `r_i = 0` places it as `(w_i, h_i)`, `r_i = 1` as `(h_i, w_i)`. Every rectangle must be placed.
  When `R = 0`, every `r_i` must be `0`.
- Time limit: about 2 seconds. Memory: 256 MB.

A placement of rectangle `i` is the axis-aligned box with corners `(x_i, y_i)` and
`(x_i + pw_i, y_i + ph_i)`, where `(pw_i, ph_i) = (w_i, h_i)` if `r_i = 0` else `(h_i, w_i)`.

## Background

Strip packing is the one-dimensional-container cousin of bin packing: the container is infinitely
tall and we pay for the height we actually use. Two textbook reference points frame the design.

- **Shelf / level packing (FFDH).** Sort rectangles by decreasing height and greedily pack them into
  horizontal *shelves*: place a rectangle left-to-right into the first shelf whose remaining width
  admits it, otherwise open a new shelf on top. This is fast and never overlaps, but it forbids a
  short rectangle from ever sitting *above* a tall one inside the same column, so it wastes the
  vertical gaps above short pieces. FFDH is the natural normalizer the scorer measures against.
- **Bottom-left / skyline placement.** Drop each rectangle as low and as far left as it will go,
  letting it rest on whatever is already there. Represented by a *skyline* — the upper frontier of
  the occupied region as a sequence of horizontal segments `(x, width, y)` partitioning `[0, W)` —
  this lets short pieces fill the pockets above other pieces, which is exactly the waste shelves
  leave behind. The placement order, and (when allowed) the per-rectangle rotation, are then the
  free decisions to optimize.

## Evaluation settings

A solution is first checked for **feasibility**; any violation floors the score to **0**:

1. the output parses as exactly `3n` integers (`n` lines of `x y r`);
2. every `r_i in {0,1}`, and if `R = 0` then every `r_i = 0`;
3. each rectangle is fully inside the strip: `0 <= x_i`, `0 <= y_i`, `x_i + pw_i <= W`;
4. the `n` placed rectangles are pairwise non-overlapping (only positive-area overlap is illegal;
   shared edges are allowed).

For a feasible solution the **used height** is `height = max_i (y_i + ph_i)` (and `0` if `n = 0`).
Lower is better. The score normalizes against the deterministic **first-fit decreasing-height
(FFDH) shelf** baseline that the scorer recomputes itself:

```
score = round(1_000_000 * baseline_height / max(1, solver_height))     (0 if INFEASIBLE)
```

So FFDH scores about `1_000_000`; a shorter packing scores more, an infeasible one scores `0`.

**Instances** are generated deterministically from an integer seed. The strip width `W` is in the
low hundreds and `n` is in `[30, 80]`. The rectangles come from a recursive *guillotine* split of a
virtual `W x Htarget` block (a few stacked bands), then each piece is jittered by a few units and,
when `R = 1`, randomly pre-rotated. So a near-perfect packing of height about `Htarget` exists in
principle, but the jitter and rotation choices make the optimum non-trivial — precisely the regime
where placement order, rotation, and a tight bottom-left rule beat a plain shelf.

## Code framework

A single self-contained C++17 program that reads the instance from stdin and writes a feasible
solution to stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int N, W, R;
    if (!(cin >> N >> W >> R)) return 0;
    vector<int> w(N), h(N);
    for (int i = 0; i < N; ++i) cin >> w[i] >> h[i];

    // A feasible fallback: stack every rectangle in a single column at x = 0,
    // rotating only if the native width does not fit the strip.
    // TODO heuristic: skyline (bottom-left) placement driven by an insertion
    // permutation + per-rectangle rotate bit, optimized by simulated annealing
    // over the permutation; each candidate is rebuilt by an O(n * #segments)
    // skyline replay, so the search is cheap and always feasible.
    long long y = 0;
    for (int i = 0; i < N; ++i) {
        int r = 0, pw = w[i], ph = h[i];
        if (pw > W && R == 1) { r = 1; pw = h[i]; ph = w[i]; }
        cout << 0 << ' ' << y << ' ' << r << '\n';
        y += ph;
    }
    return 0;
}
```
