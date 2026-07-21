Frost spreads from every vent at once, one cell per minute through edge-adjacent open cells, so an open cell's frost time is its grid distance to the nearest vent, with a vent at `0`. I count open cells whose frost time `d` lands in the inclusive window `[L, U]`. The scale decides the algorithm before anything else: `R, C <= 1000`, so up to `10^6` cells, and every open cell can be a vent — up to `~10^6` sources. A separate BFS per vent plus an element-wise minimum is `O(R*C)` each, so `O((R*C)^2)` up to `10^12`, hopeless. One multi-source BFS seeded with every vent at distance `0` gives the nearest-vent distance for all cells in a single `O(R*C)` sweep; the queue stays distance-sorted because every edge has weight `1`. That is the only skeleton that fits. Everything hard after that lives in the counting predicate and in what "reached" means, not in the BFS.

The window is where this problem hides its off-by-ones, and each is the kind a `1000 x 1000` grid punishes at scale but a four-cell grid exposes for free:

1. Inclusive on *both* ends. `d = L` and `d = U` are both damaged. A reflexive `d < U` comparison drops the entire `d = U` shell — and on a grid that outermost shell can be the single biggest contributor, so this is not a corner case, it is potentially most of the answer.
2. The vent is at `d = 0`, not `d = 1`. Seeding sources at `1` inflates every distance by one and slides the band.
3. "Reached" must mean frost actually arrived. A cell walled off from all vents has no frost time. If I mark unreached cells with a large sentinel and test `L <= d <= U` blindly, and especially if `L = 0` where a bare `d >= L` is true for everything, unreached cells sneak in. They must be excluded explicitly, independent of `L`.

For the distance representation I use a 2D `int` array preset to `INF = INT_MAX`, and the sentinel doubles as the visited marker: `dist[r][c] == INF` means not yet reached, and because BFS assigns each cell exactly once at its shortest distance, the first write also marks it visited. A plain FIFO `deque` suffices — unit edge weights mean FIFO order already pops cells in nondecreasing distance, no priority queue needed.

The one place the documented sample would trip a shortcut is `(1,4)`: the wall at `(1,3)` blocks the direct approach, so frost has to detour down to `(2,4)` and across for distance `4`, not the Manhattan `2`. A BFS through open cells only reproduces that automatically — the distance field is grid geodesic, not `|dr|+|dc|`.

Now the counting predicate, where pitfall #1 bites first. The natural-but-wrong write is the exclusive one:

```
if (d >= L && d < U) cnt++;     // WRONG: drops the d = U shell
```

The smallest probe is a `1 x 4` corridor `*...` with window `[1, 2]`: frost times are `0 1 2 3`, so the damaged cells are `(0,1)` and `(0,2)`, answer `2`. But `d >= 1 && d < 2` counts only `d = 1`, returning `1`. The exclusive right endpoint dropped the `d = U = 2` cell exactly as feared. The window is closed on the right; the fix is `d <= U`, after which the corridor gives `2` and the sample's band `[1,2]` includes the whole `d = 2` shell — consistent with its documented count of `13`.

Pitfall #3 needs a separate guard, and it is tempting to think the `INF` sentinel already handles it. Take `*#.` with window `[0, 1000000]`: `(0,0)` is a vent at `0`, `(0,1)` is a wall, `(0,2)` is open but sealed off by the wall so frost never arrives. The correct count is `1` (just the vent). The wall has `dist = INF` and the isolated cell has `dist = INF`, and both are excluded only because `INT_MAX` happens to exceed any legal `U <= R*C <= 10^6`. That is correctness leaning on the sentinel's magnitude — fragile, and it would break outright for a semantic window like `[0, INF]`. The honest statement is "distance was never assigned," i.e. `dist == INF`, not "distance is bigger than `U`". And a wall is never a plant regardless of its stored value, so it should be skipped before the band test, not by luck. So the count loop skips walls, skips `dist == INF`, then applies the inclusive band:

```
if (g[r][c] == '#') continue;     // walls are never plants
int d = dist[r][c];
if (d == INF) continue;           // open cell frost never reached
if (d >= L && d <= U) cnt++;      // inclusive band [L, U]
```

Re-running `*#.` with `[0, 1000000]`: `(0,0)` counts, `(0,1)` skipped as wall, `(0,2)` skipped as `INF` — count `1`, now correct because the exclusion no longer depends on `INF > U`.

The corner cases follow from these two guards and mostly need no special-casing. No vents at all: the seeding loop pushes nothing, BFS never runs, every open cell stays `INF` and is skipped, answer `0` — which is right, no frost source means no damage. `L = 0`: includes vents but the explicit `INF` skip keeps unreached cells out, which a bare `d >= 0` would not. `U = 0`: only vents qualify; on the sample `[0,0]` gives `2`. `L = U`: a single shell, `d >= L && d <= U` collapsing to `d == L`. Huge `U` up to `R*C`: safe, since the real maximum distance is below `R*C` and the `INF` skip carries the exclusion. Thin `1 x C` / `R x 1` grids and walled-off pockets are handled by the same bounds check and `INF` skip. The counter fits a 32-bit int (at most `10^6` cells in band) but I use `long long` for headroom, and the guarded `if (!(cin >> ...)) return 0;` makes a truncated input print nothing rather than crash. BFS touches each cell and its four edges once: `O(R*C)` time and memory, well inside 1 s / 256 MB for `10^6` cells.

So the shape I ship is one self-contained `O(R*C)` multi-source BFS: seed the queue with every `'*'` at distance `0`, sweep outward through `'.'`/`'*'` cells skipping `'#'` and already-assigned cells, then a single count pass that skips walls, skips `INF`, and applies `L <= d <= U`. The full self-contained program is in the answer.
