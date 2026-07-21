Two numbers in the constraints decide the shape of the solution before I pick any algorithm. `n` and `p` are *both* up to `8000`, so a DP whose work scales like `p * n^2` runs to `5 * 10^11` inner iterations — dead on arrival in a two-second window. And there are up to `8000` positions as large as `10^9`, so at `p = 1` the single pump sits at the global median and the summed distance reaches `~n/2 * 10^9 ~= 4 * 10^12`, four thousand times past the `~2.1 * 10^9` ceiling of a 32-bit int. So the whole solution is boxed in from the start: 64-bit accumulators everywhere, and a transition strictly faster than the naive split scan. The I/O is plain — `n p` on the first line, the `n` sorted positions on the second, one integer out, the minimum total served distance.

**Reducing the geometry to a partition.** The pumps live anywhere on the real line and each well takes its nearest pump. Sort the pumps: as a well's coordinate increases, the index of its nearest pump never decreases, so pump 1 serves a prefix of the sorted wells, pump 2 the next run, and so on. An optimal solution therefore partitions the sorted wells into `p` contiguous blocks, one per pump — interleaving never helps. That converts a geometric placement problem into a one-dimensional partition problem, which is a form I can attack.

Within a block served by one pump at coordinate `y`, the cost is `sum |x[t] - y|`, minimised on a line at the block **median**. For a sorted block `[j, i]` the lower median index is `m = (j + i) / 2`; with prefix sums `pre[t] = x[0] + ... + x[t-1]`, the wells left of the median contribute `x[m]*(m-j) - (pre[m]-pre[j])` and those right of it contribute `(pre[i+1]-pre[m+1]) - x[m]*(i-m)`, so the block cost `C(j, i)` is an `O(1)` query. The problem is now exactly: split the sorted array into `p` contiguous blocks minimising the sum of `C`.

The layered DP writes itself. Let `dp_k[i]` be the min cost to cover the first `i` wells with `k` pumps; the last pump covers `[j, i-1]` and the first `k-1` cover the first `j` wells, so

```
dp_k[i] = min over j in [k-1, i-1] of  dp_{k-1}[j] + C(j, i-1),
dp_1[i] = C(0, i-1),   dp_0[0] = 0,
```

with answer `dp_p[n]`. Filling `p * n` cells with an `O(n)` split scan is the `O(p n^2)` I already flagged as fatal: at `n = p = 8000` those `5.12 * 10^11` iterations are roughly nine minutes, three orders past the limit. No constant factor closes a 300x gap, so I need the inner `min` to skip most split points.

**Making the split point monotone.** The escape is to show the argmin moves predictably. Let `opt_k(i)` be the smallest `j` achieving `dp_k[i]`. If `opt_k(i)` is non-decreasing in `i` — the best cut for the last block only ever slides right as I extend the prefix by one well — then within a fixed layer the scanned windows of candidate `j` telescope instead of rescanning `[0, i]` from scratch. That monotonicity follows from the block-cost matrix satisfying the **quadrangle (Monge) inequality**, `C(a,c) + C(b,d) <= C(a,d) + C(b,c)` for `a <= b <= c <= d`: absolute-deviation cost over a sorted interval grows convexly as the interval stretches to reach farther wells, so nesting two intervals never costs more than crossing them, and Monge cost in a min-plus partition DP forces the argmin monotone. That is the lever, and it turns the transition into **divide-and-conquer DP optimisation**.

For a fixed layer I fill `curLayer[i]` over `i in [lo, hi]` knowing `opt(i) in [optlo, opthi]`. Take `mid = (lo + hi) / 2`, find its best split `bestj` by scanning only `j in [optlo, min(opthi, mid-1)]`, set `curLayer[mid]`, then recurse left on `[lo, mid-1]` with window `[optlo, bestj]` and right on `[mid+1, hi]` with `[bestj, opthi]`. Monotonicity guarantees every `i < mid` optimises at `<= bestj` and every `i > mid` at `>= bestj`, so no true optimum is ever excluded. The recursion is `O(log n)` deep and on each level the scanned windows across all `mid` cover `[0, n]` a constant number of times, so one layer is `O(n log n)` and the whole DP `O(p n log n) ~= 8000 * 8000 * 13 ~= 8.3 * 10^8` cheap operations — comfortably inside budget.

**The one place the recursion can bite: the pivot.** `bestj` does double duty — it is the split point stored into `curLayer[mid]` *and* the pivot that partitions the `opt` window for both recursive calls. If a `mid` cell has no feasible candidate — every `prevLayer[j]` in its window is the `INF` sentinel of an infeasible `(k-1)`-pump prefix — and I leave `bestj` at its initial `-1`, then the right recursion receives `opthi = -1`, computes `jhi = min(-1, mid'-1) = -1`, its inner loop never runs, and an entire sub-range silently stays `INF` — including cells that *do* have valid splits. Tracing `n=3, p=2` on `0 5 10` (answer `5`, from `{0} | {5,10}`) surfaces exactly this: a stray `INF` poisons the result. The fix is to keep the pivot well-formed — if no candidate was chosen, fall back to `optlo` before recursing:

```text
curLayer[mid] = best;
if (bestj == -1) bestj = optlo;   // no feasible split; keep window valid
compute(lo, mid - 1, optlo, bestj);
compute(mid + 1, hi, bestj, opthi);
```

With that, `n=3, p=2` returns `5` and `n=3, p=3` returns `0`. I also guard the candidate sum itself, skipping any `prevLayer[j] >= INF`, so a real cost is never added onto the sentinel and `INF + cost` can't overflow.

The corners where partition DPs go wrong, checked concretely: `p = 1` runs only the base layer, `dp_1[n] = C(0, n-1)` is the whole-array median cost around `2 * 10^12` — the 64-bit path, where an `int` would be a silent wrong answer. `p = n` gives `kcap = min(p, n) = n` and a pump per well, `dp_n[n] = 0` (checked on `7 7 -> 0`). Nine equal positions make every block cost `0`, so the median formula must survive zero-width blocks — it returns `0`. For an even block like `{1, 4}` the lower median `x[(j+i)/2]` gives cost `3`, matching a hand sum, since any point between the two central order statistics is optimal and the lower median is one of them.

For confidence beyond the argument I differential-tested against an independent `O(p n^2)` brute that recomputes each block cost by an explicit scan to the median, sharing no prefix-sum code with the solution. Across the adversarial/edge battery — a single well, `p=1`, `p=n`, all-equal, `10^9`-spread overflow cases, tight-duplicate clusters — plus several hundred random small instances (`n` up to `18`, mixed value spreads), there were zero mismatches. The `n = p = 8000` stress runs well inside two seconds, and the `p=1` case returns a value in the `10^12` range, confirming the 64-bit path is live.

What I ship is one self-contained C++17 file: read and sort the positions, build the prefix sums, define the `O(1)` median `cost(j, i)`, and run the divide-and-conquer `compute` over layers `k = 1 .. min(p, n)` with two rolling arrays, printing `dp_p[n]`. The full module is in the answer.
