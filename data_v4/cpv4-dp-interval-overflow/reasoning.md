The distinctive thing about this polygon problem is not the triangulation — it is the arithmetic. A triangle on corners `a, b, c` costs the *product* `v[a]*v[b]*v[c]`, and with labels up to `2000` a single triangle already reaches `2000^3 = 8 * 10^9`, past the 32-bit signed ceiling of `~2.147 * 10^9`. A full triangulation sums `n - 2` such triangles with `n` up to `500`, so the answer climbs to roughly `498 * 8 * 10^9 ≈ 4 * 10^12` — far beyond `int`, but comfortably inside `long long` (ceiling `~9.2 * 10^18`). So the first decision, before any recurrence, is that every label, every product, and every accumulator is 64-bit; an `int` anywhere in the product is a wrong answer. The rest of the contract is plain: read `n` then the `n` labels, print one integer, and `n < 3` means no triangle and answer `0`.

The clean route is an interval DP, but the tempting shortcut is a greedy fan: pick one vertex and draw every diagonal from it, so the whole polygon fans into triangles all sharing that vertex. Choosing the smallest-labelled vertex keeps its factor in every product, which feels economical, and it is `O(n)`. But a fan forces a single vertex into all `n - 2` triangles, whereas an optimal triangulation can use diagonals that share no common vertex — a fan simply cannot express those. And since a fan is just one special case of the interval DP (the apex fixed to the same vertex at every level), it can never beat the DP and is in general strictly worse. So it survives only as a special case the DP subsumes, not as a strategy.

For the interval DP, define `dp[i][j]` = the minimum cost to triangulate the sub-polygon whose boundary runs `i, i+1, ..., j` along the arc and is closed by the chord `i–j`. The structural fact that makes this work: in any triangulation the chord `i–j` belongs to exactly one triangle, whose apex is some interior vertex `k` with `i < k < j`. That apex splits the region into the arc `i..k` (closed by chord `i–k`), the arc `k..j` (closed by chord `k–j`), and the apex triangle `(i, k, j)` itself, giving

`dp[i][j] = min over k in (i, j) of [ dp[i][k] + dp[k][j] + v[i]*v[k]*v[j] ]`.

The base case is an arc of two vertices, `dp[i][i+1] = 0`: a single boundary edge holds no triangle. The whole-polygon answer is `dp[0][n-1]`. Diagonals never cross because the recursion is nested by construction — each recursive call works strictly inside the chord that defined it. Filling by increasing arc length `len = j - i` from `2` up guarantees both shorter arcs are ready before `dp[i][j]` is computed. States are `O(n^2)` and the apex loop `O(n)`, so `O(n^3) ≈ 1.25 * 10^8` at `n = 500` — about `0.02 s`, well inside the 1-second limit, with a `500 * 500` `long long` table at ~5.6 MB.

On the square sample `v = [10, 20, 30, 40]` (expected `18000`): edges give `dp[0][1] = dp[1][2] = dp[2][3] = 0`; the length-2 arcs give `dp[0][2] = 10*20*30 = 6000` (apex `k = 1`) and `dp[1][3] = 20*30*40 = 24000` (apex `k = 2`). For the whole square `dp[0][3]`, apex `k = 1` gives `dp[0][1] + dp[1][3] + 10*20*40 = 24000 + 8000 = 32000`, apex `k = 2` gives `dp[0][2] + dp[2][3] + 10*30*40 = 6000 + 12000 = 18000`; the minimum `18000` matches the expected answer, and the two apex choices are exactly the two triangulations of a square.

Transcribing it, the reflex is to write the DP with `int` everywhere because the labels "are small":

```
vector<vector<int>> dp(n, vector<int>(n, 0));
for (int len = 2; len < n; ++len)
    for (int i = 0; i + len < n; ++i) {
        int j = i + len;
        int best = INT_MAX;
        for (int k = i + 1; k < j; ++k) {
            int cost = dp[i][k] + dp[k][j] + v[i] * v[k] * v[j];
            if (cost < best) best = cost;
        }
        dp[i][j] = best;
    }
```

The loop order is fine, but the arithmetic is not. Trace `v = [1500,1500,1500,1500]`: every triangle costs `1500^3`, and `1500 * 1500 = 2,250,000` still fits an `int`, but `2,250,000 * 1500 = 3,375,000,000` does not, so the product `v[i]*v[k]*v[j]` — evaluated in `int` because all three operands are `int` — wraps before it is ever added anywhere. Compiling this version and feeding it this all-`1500` four-vertex case confirms it: it prints `-1839934592`, a *negative* total cost for a product of positive labels — a sign flip, not silent rounding, and impossible for a real triangulation.

The fix is at the source, not a late cast — casting the sum to 64-bit would not save the per-triangle product, which overflows before it is added. I make `v` a `vector<long long>` and `dp` a `vector<vector<long long>>`, with `LLONG_MAX` as the sentinel for `best`:

```
long long best = LLONG_MAX;
for (int k = i + 1; k < j; ++k) {
    long long cost = dp[i][k] + dp[k][j] + v[i] * v[k] * v[j];
    if (cost < best) best = cost;
}
```

Now the operands are `long long`, so the multiplication happens in 64-bit and `1500^3 = 3,375,000,000` is exact; the corrected program prints `6750000000` on that same all-`1500` case. Two magnitude extremes pin the worst case: `n = 3` with `[2000,2000,2000]` gives the maximal single triangle `8 * 10^9`, and `n = 500` all-`2000` gives `3,984,000,000,000 ≈ 4 * 10^12`, both exact and both far inside `long long`. The sentinel `LLONG_MAX` is only ever compared against and overwritten by a real `cost`, never added to anything — and every arc with `j - i >= 2` has at least one apex, so it is always replaced before being stored and cannot leak an overflow of its own.

One logic corner remains, independent of the arithmetic: tiny `n`. I print `dp[0][n-1]`, and for `n = 0` that index is `dp[0][-1]` — an out-of-range read on an empty table. The clean guard is an early exit before I allocate anything: `if (n < 3) { cout << 0; return 0; }`. Then `n = 0`, `n = 1` (a point), and `n = 2` (a single edge) all return `0` directly and never touch `dp`; the length loop `len < n` is anyway empty for those cases.

I do not want to trust the recurrence on hand cases alone, so I check it against an independent brute force that assumes no optimal substructure at all: it enumerates *every* triangulation explicitly (fix the edge `(0, n-1)`, which sits in exactly one triangle whose apex `k` splits the polygon into `0..k` and `k..n-1`, recurse and concatenate the actual triangle lists), sums the real per-triangle products of each, and takes the global minimum. That enumerator is exponential but obviously correct, and instant for tiny `n`. Across 400 random cases — `n` from `0` to `7`, labels mostly small but 30% drawn near `2000` to exercise the magnitude — the DP and the enumerator agree with zero mismatches, which certifies the recurrence, the base case, the loop order, and the `n < 3` guard together, over exactly the high-magnitude inputs where the `int` version diverged.

That is the solution: the `O(n^3)` interval DP with 64-bit arithmetic throughout and an early `n < 3` return; the full self-contained program is in the answer.
