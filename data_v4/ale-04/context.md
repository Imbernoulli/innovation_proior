# Heat-Diffusion Tile Coloring

## Research question

A thermal panel is an `N x N` grid of tiles. Each tile must be given one of two heat
coatings, `0` (cool) or `1` (warm). Two forces fight over the choice:

- **Roughness / interface energy.** Heat diffuses across the panel, so a long boundary
  between cool and warm tiles is costly. Every pair of 4-adjacent tiles with *different*
  coatings costs a fixed interface weight `W`. A perfectly smooth panel (all one colour)
  has zero interface energy.
- **Field fidelity.** Each tile has a preferred coating `t` (its local "target" from the
  thermal layout) and a non-negative field strength `h` saying how strongly it prefers it.
  Giving a tile the wrong coating costs `h`. A panel that honours every target may have a
  very ragged, high-interface layout.

Some tiles are **pinned** by manufacturing: their coating is fixed and the design must use
that value. The task is to choose a coating for every tile so as to **minimize the total
energy** — the interface (roughness) term plus the field (fidelity) term — while respecting
all pins. This is the binary image-labelling / Ising-with-a-field problem that appears in
denoising, segmentation, and material layout; getting the smoothness-versus-fidelity trade
right on the interface is the whole game.

## Input / output contract

- **Input (stdin):**
  - First line: two integers `N` and `W` (`30 <= N <= 60`, `1 <= W`).
  - Then `N` lines, each with `N` triples (so `3*N` integers per line). The triple for the
    cell in row `r`, column `c` is `h t p`:
    - `h >= 0` — the cell's field strength.
    - `t in {0,1}` — the cell's target coating.
    - `p in {-1,0,1}` — the pin flag: `-1` means the cell is free; `p in {0,1}` means the
      cell is pinned to coating `p` and the output **must** use that value there.
- **Output (stdout):** `N` lines, each with `N` space-separated bits, giving the chosen
  coating `x[r][c] in {0,1}` for every tile. (Any whitespace layout that yields exactly
  `N*N` binary tokens is accepted; the natural `N`-rows-of-`N` form is expected.)
- **Time limit:** about 2 seconds wall-clock per instance. Memory: 256 MB.

## Evaluation settings

A solution is **feasible** iff (1) it parses as exactly `N*N` binary tokens and (2) every
pinned cell carries its pinned value. For a feasible coloring the energy is

```
E = W * (# of 4-adjacent pairs whose coatings differ)
  + sum_cells  h[r][c] * [ x[r][c] != t[r][c] ]
```

both terms non-negative integers, so `E >= 0`. The score is

```
score = round( 1e9 / (1 + E) )   if the solution is feasible,
score = 0                        otherwise (the feasibility -> 0 floor).
```

Lower energy means higher score; a perfect `E = 0` coloring scores `1e9`. Any infeasible
output — wrong shape, a non-binary token, or a violated pin — floors the score to `0`.

**Instances.** A generator seeds a `random.Random`, draws `N in [30,60]` and `W in [3,12]`,
places a few Gaussian "warm shoals" of targets over a cool background (so honouring targets
is ragged), assigns heterogeneous field strengths `h` (with a fraction of `h = 0` don't-care
cells), and pins a sparse set of cells — usually to their target, occasionally to the
opposite colour to create tension — guaranteeing at least one pin of each colour so the
all-`0` and all-`1` colorings are infeasible. The seed set, generator, and scorer are frozen;
the mean score over the seeds is reported. The only editable thing is the solver.

## Code framework

A single self-contained C++17 program reading the instance on stdin and writing a feasible
coloring on stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int N, W;
vector<int> H, T, P; // field, target, pin (-1 free, else fixed bit)
inline int ID(int r, int c) { return r * N + c; }

int main() {
    if (scanf("%d %d", &N, &W) != 2) return 0;
    int M = N * N;
    H.assign(M, 0); T.assign(M, 0); P.assign(M, -1);
    for (int i = 0; i < M; i++) scanf("%d %d %d", &H[i], &T[i], &P[i]);

    vector<char> x(M, 0);
    // Feasible start: honour pins, else take the target.
    for (int i = 0; i < M; i++) x[i] = (P[i] != -1) ? (char)P[i] : (char)T[i];

    // TODO heuristic: improve on this baseline to lower the total energy while
    //                 keeping every intermediate/final coloring feasible.

    // Output N rows of N bits (pins already honoured).
    for (int r = 0; r < N; r++) {
        for (int c = 0; c < N; c++)
            printf("%d%c", x[ID(r, c)], c + 1 < N ? ' ' : '\n');
    }
    return 0;
}
```
