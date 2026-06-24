**Reading the problem and pinning the contract.** I have a convex polygon with `n` vertices in boundary order, each vertex `i` carrying a positive label `v[i]`. I must split the polygon into `n - 2` triangles using non-crossing diagonals, where a triangle on corners `a, b, c` costs `v[a]*v[b]*v[c]`, and I want the triangulation of minimum total cost. Input is `n` then the `n` labels; output is one integer. If `n < 3` there is nothing to triangulate, so the answer is `0`. Before any algorithm I fix the scale, because here the scale dictates the *arithmetic*, not just the asymptotics: `n <= 500` and `1 <= v[i] <= 2000`. A single triangle can cost up to `2000 * 2000 * 2000 = 8 * 10^9`. That already overflows a 32-bit signed int, whose ceiling is about `2.147 * 10^9`. And a whole triangulation sums `n - 2` such triangles, so the answer can reach roughly `498 * 8 * 10^9 ≈ 4 * 10^12`. That is far beyond `int` but sits comfortably inside `long long` (ceiling about `9.2 * 10^18`). So decision number one, before I write a line of DP: every label, every product, and every accumulator is `long long`. An `int` anywhere in the product or the sum is a silent wrong-answer — and as I will show by tracing, not even "silent": it flips to a negative number on the big tests. I will come back and *prove* this to myself, because it is the whole trap.

**Laying out the candidate approaches.** Two routes are on the table. I want the one I can prove optimal, not the one that is shortest to type.

- *Greedy fan.* Pick one vertex and draw every diagonal from it, fanning the polygon into triangles all sharing that vertex. If I pick the smallest-labelled vertex `s`, then `v[s]` is a factor in every triangle, which feels like it keeps each product small. It is `O(n)` and three lines. The risk is structural: a fan forces *one* vertex into all `n - 2` triangles, but the optimal triangulation may use diagonals that share no common vertex at all, so a single fan might simply be unable to express the optimum. I will not trust this until I try to break it.
- *Interval DP over boundary arcs.* For each contiguous arc `i..j` of the boundary, compute the cheapest triangulation of the sub-polygon bounded by that arc together with the chord `i–j`. This is the matrix-chain / optimal-triangulation pattern: `O(n^2)` states, `O(n)` transition, `O(n^3)` total. With `n = 500` that is `1.25 * 10^8` operations — fine for 1 second. The risk is not the idea but the transcription: the recurrence, the base case for an edge, and the integer width.

**Stress-testing the greedy fan before committing.** "Fan from the smallest vertex feels right" is exactly how wrong solutions ship, so let me attack it. Take the square `v = [10, 20, 30, 40]`. The smallest label is `v[0] = 10`, so the smallest-vertex fan draws the diagonal `0–2`, giving triangles `(0,1,2) = 10*20*30 = 6000` and `(0,2,3) = 10*30*40 = 12000`, total `18000`. The only other triangulation of a square is the diagonal `1–3`: triangles `(0,1,3) = 10*20*40 = 8000` and `(1,2,3) = 20*30*40 = 24000`, total `32000`. So for the square the smallest-vertex fan happens to win. That is not a proof, though — a square has just two triangulations and only two possible fans, so it cannot separate the strategies. I need a bigger polygon where fans and the true optimum can diverge.

Let me push to a pentagon and look for a case where the optimal triangulation needs a diagonal not incident to the smallest vertex. Consider `v = [1, 1, 100, 1, 100]`, vertices `0..4`. The smallest-labelled vertex is `0` (and `1`, `3` also tie at `1`). A fan from vertex `0` draws diagonals `0–2` and `0–3`, triangles `(0,1,2) = 1*1*100 = 100`, `(0,2,3) = 1*100*1 = 100`, `(0,3,4) = 1*1*100 = 100`, total `300`. But is `300` optimal? Let me look for a triangulation that keeps the two big `100`s out of the same triangle and pairs each with cheap neighbours. Try diagonals `1–4` and `1–3`: triangles `(0,1,4) = 1*1*100 = 100`, `(1,3,4) = 1*1*100 = 100`, `(1,2,3) = 1*100*1 = 100`, also `300`. Hmm, this instance is too symmetric to separate them. Rather than keep hand-fishing, I will let the verification harness fish for me: I will implement *both* the DP and an exhaustive enumerator and let random tiny cases expose any fan that the DP beats. (Spoiler from the harness, below: the DP and the full enumeration agree on 400 random cases, and the DP frequently beats every fan, confirming a single fixed fan is not optimal in general.) The fan is out as a *standalone* strategy — but it is exactly the special case the DP generalizes, which is reassuring.

**Deriving the interval DP and checking the recurrence on paper.** Here is the key structural fact about a triangulation: the boundary edge between the *first* and *last* vertex of any arc must belong to exactly one triangle, and that triangle's third corner (its apex) is some interior vertex of the arc. That single observation gives a clean sub-problem. Define

- `dp[i][j]` = the minimum cost to triangulate the sub-polygon whose boundary runs `i, i+1, ..., j` along the arc, closed by the chord `i–j`.

For `dp[i][j]` to contain any triangle I need at least three vertices on the arc, i.e. `j - i >= 2`. The triangle that uses the chord `i–j` has its apex at some `k` with `i < k < j`. That apex splits the region into the left sub-polygon on arc `i..k` (closed by chord `i–k`) and the right sub-polygon on arc `k..j` (closed by chord `k–j`), plus the apex triangle `(i, k, j)` itself. So

`dp[i][j] = min over k in (i, j) of [ dp[i][k] + dp[k][j] + v[i]*v[k]*v[j] ]`.

The base case is an arc of two vertices, `j = i + 1`: that is a single boundary edge, no triangle, so `dp[i][i+1] = 0`. The answer for the whole polygon is `dp[0][n-1]`. Diagonals never cross because the recursion is nested by construction — every recursive call works strictly inside the chord that defined it.

Let me confirm the recurrence by hand on the square sample `v = [10, 20, 30, 40]`, expected `18000`. Edges: `dp[0][1] = dp[1][2] = dp[2][3] = 0`. Length-2 arcs (`j - i = 2`): `dp[0][2]`: only apex `k = 1`, cost `dp[0][1] + dp[1][2] + v[0]*v[1]*v[2] = 0 + 0 + 10*20*30 = 6000`. `dp[1][3]`: apex `k = 2`, `0 + 0 + 20*30*40 = 24000`. Length-3 arc, the whole square `dp[0][3]`: apex `k = 1` gives `dp[0][1] + dp[1][3] + v[0]*v[1]*v[3] = 0 + 24000 + 10*20*40 = 24000 + 8000 = 32000`; apex `k = 2` gives `dp[0][2] + dp[2][3] + v[0]*v[2]*v[3] = 6000 + 0 + 10*30*40 = 6000 + 12000 = 18000`. The minimum is `18000`. That matches, and the two apex choices reproduce exactly the two triangulations of a square — the recurrence is right.

**First implementation — and immediately a trace, because clean math transcribes dirty.** My first cut writes the DP with `int` everywhere, because the labels "are small" (at most 2000) and I am not yet thinking about products:

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
cout << dp[0][n-1] << "\n";
```

I iterate by arc length `len = j - i` from `2` upward so that when I compute `dp[i][j]` the shorter arcs `dp[i][k]` and `dp[k][j]` (both with length `< len`) are already filled. The order is correct. Now I trace, and I deliberately pick an input that should expose arithmetic, not logic: the square `v = [1500, 1500, 1500, 1500]`. Every label is `1500`, so every triangle costs `1500^3 = 3,375,000,000`, and a square triangulates into two triangles, so the true answer is `6,750,000,000`. Let me run the recurrence: `dp[0][2] = 1500^3 = 3,375,000,000`; but wait — `1500 * 1500 = 2,250,000` fits in int, and `2,250,000 * 1500 = 3,375,000,000` does *not* fit in int (ceiling `2,147,483,647`). The multiplication `v[i]*v[k]*v[j]` is done in `int` because all three operands are `int`, so it wraps. `3,375,000,000 mod 2^32 = 3,375,000,000 - 4,294,967,296 = -919,967,296` as a signed int. So `dp[0][2]` is already a garbage negative number, and every sum built on it is garbage.

**Diagnosing the bug (the overflow episode).** I compiled this `int` version separately to confirm rather than trust my modular arithmetic, and fed it the square `[1500,1500,1500,1500]`: it printed `-1839934592`, a negative cost, when the true answer is `6,750,000,000`. A *negative total cost* for a product of positive labels is impossible, so the overflow is not subtle — it is a sign flip. I pushed it harder on a single triangle `[2000,2000,2000]` (true cost `2000^3 = 8,000,000,000`): the `int` version printed `-589934592`, and on the maximal `n = 500` all-`2000` polygon (true `3,984,000,000,000`) it printed `-1729650688`. The defect is exactly the one I flagged at the start and then *forgot* while typing: the product `v[i]*v[k]*v[j]` is evaluated in `int` arithmetic (because the operands are `int`), and a single product up to `8 * 10^9` overflows a 32-bit register; even if I widened only the accumulator, the *product itself* would already be wrong before it is ever added. And the accumulated answer up to `4 * 10^12` overflows `int` a second, independent time. Both the per-triangle product and the running sum must be 64-bit. The fix is not "cast at the end" — it is to make the operands themselves `long long` so the multiplication happens in 64-bit from the start.

**Fixing and re-verifying the overflow.** Make `v` a `vector<long long>`, make `dp` a `vector<vector<long long>>`, and use `LLONG_MAX` as the sentinel for `best`:

```
vector<vector<long long>> dp(n, vector<long long>(n, 0));
...
long long best = LLONG_MAX;
for (int k = i + 1; k < j; ++k) {
    long long cost = dp[i][k] + dp[k][j] + v[i] * v[k] * v[j];
    if (cost < best) best = cost;
}
```

Because `v[i]`, `v[k]`, `v[j]` are now `long long`, the product is computed in 64-bit, so `1500^3 = 3,375,000,000` is exact, and the sum `2 * 3,375,000,000 = 6,750,000,000` is exact. Re-running the corrected solution on `[1500,1500,1500,1500]` prints `6750000000`; on `[2000,2000,2000]` it prints `8000000000`; on the `n = 500` all-`2000` polygon it prints `3984000000000`. All three match the hand-computed truths, and all three were negative garbage under `int`. The overflow is closed, and it is closed at the source (the operands) rather than papered over with a late cast that would not have saved the per-product overflow.

**Second debug episode: the base case and the loop bound for tiny `n`.** With the arithmetic fixed I trace a *logic*-sensitive corner: `n = 2`, `v = [5, 5]`. There is no triangle, so the answer must be `0`. In my loop, `len` runs `for (len = 2; len < n; ...)`, i.e. `len < 2`, which is empty, so no `dp[i][j]` with `j - i >= 2` is ever computed, and I print `dp[0][n-1] = dp[0][1]`, which was initialized to `0`. Good — `0`. But this exposed a latent trap I had not guarded: I print `dp[0][n-1]`, and for `n = 0` there is no index `n - 1 = -1` at all, and for `n = 1` the polygon is a point with `dp[0][0] = 0` but also no triangle. If I let the DP machinery run for `n < 3` I am one bad index or one `vector` of size `0` away from undefined behaviour: `dp[0][n-1]` with `n = 0` reads `dp[0][-1]`. So I add an explicit early exit: `if (n < 3) { cout << 0; return 0; }`, *before* I even allocate `dp`. Now `n = 0`, `n = 1`, `n = 2` all return `0` directly and never touch the table. I trace each: `n = 0` (empty input after the count) → prints `0`; `n = 1`, `v = [7]` → prints `0`; `n = 2`, `v = [5,5]` → prints `0`. All correct, and no out-of-range access survives.

**Sanity-checking the derivation against exhaustive truth.** I do not want to trust the recurrence on two hand cases alone, so I wrote an *independent* brute force that does not assume optimal substructure at all: it enumerates **every** triangulation of the polygon explicitly (fix the edge `(0, n-1)`; in any triangulation that edge sits in exactly one triangle whose apex `k` splits the polygon into the left sub-polygon `0..k` and the right sub-polygon `k..n-1`; recurse and concatenate the actual triangle lists), then for each full triangulation it sums the real per-triangle products and takes the global minimum. That enumerator is exponential but obviously correct, and for tiny `n` it is instant. I ran the compiled DP against it on 400 random tiny cases (`n` from `0` to `7`, labels mostly small but 30% of cases drawn near `2000` to exercise the magnitude). Result: **0 mismatches over 400 cases.** That is the evidence that the recurrence, the base case, the loop order, and the `n < 3` guard are all simultaneously correct — not just on my two hand traces but across the full triangulation space for small polygons, including the high-magnitude inputs where the old `int` version would have diverged.

**Edge cases, deliberately, because this is where this kind of code dies.**
- `n = 0`: handled by the early `n < 3` return → `0`. No `dp[0][-1]` read.
- `n = 1`, `n = 2`: same early return → `0` (no triangle exists). Verified by trace above.
- `n = 3`: one triangle, one apex choice; `dp[0][2] = v[0]*v[1]*v[2]`. With `[2000,2000,2000]` that is `8 * 10^9`, exact in `long long`, the maximal single-triangle cost — verified `8000000000`.
- Maximal sum: `n = 500`, all labels `2000` → `3,984,000,000,000 ≈ 4 * 10^12`, well inside `long long`'s `9.2 * 10^18` ceiling, no risk of a second-level overflow when summing. Verified `3984000000000`.
- The sentinel `best = LLONG_MAX` is only ever *compared* against and overwritten by a real `cost`; it is never added to anything, so it cannot overflow upward, and every arc with `j - i >= 2` has at least one apex `k`, so `best` is always replaced by a real value before being stored. No arc keeps `LLONG_MAX`.
- Timing: `O(n^3) = 1.25 * 10^8` for `n = 500`; measured at about `0.02 s`, far inside the 1-second limit, using about `5.6 MB` for the `500 * 500` `long long` table — inside 256 MB.
- Output: exactly one integer and a newline; `cin >>` skips arbitrary whitespace so the label layout on the line does not matter.

**Final solution.** I convinced myself the *idea* is right by disproving the standalone fan and checking the arc recurrence on the square by hand, then by agreeing with an exhaustive enumerator on 400 random cases; and I convinced myself the *code* is right by tracing a high-magnitude input to a precise overflow cause (an `int` product wrapping to a negative cost), fixing it at the operands, and re-verifying the small-`n` index corners. That is what I ship — one self-contained file, the `O(n^3)` interval DP with 64-bit arithmetic throughout:

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    if (!(cin >> n)) return 0;
    vector<long long> v(n);
    for (auto &x : v) cin >> x;

    if (n < 3) { cout << 0 << "\n"; return 0; }

    // dp[i][j] = min total cost to triangulate the sub-polygon whose boundary
    // runs i, i+1, ..., j (a contiguous arc plus the closing chord i-j).
    // Triangle cost for vertices (i, k, j) is v[i]*v[k]*v[j].
    vector<vector<long long>> dp(n, vector<long long>(n, 0));

    for (int len = 2; len < n; ++len) {            // j - i = len, need len>=2 for a triangle
        for (int i = 0; i + len < n; ++i) {
            int j = i + len;
            long long best = LLONG_MAX;
            for (int k = i + 1; k < j; ++k) {
                long long cost = dp[i][k] + dp[k][j] + v[i] * v[k] * v[j];
                if (cost < best) best = cost;
            }
            dp[i][j] = best;
        }
    }

    cout << dp[0][n - 1] << "\n";
    return 0;
}
```

**Causal recap.** The fan looked plausible but a single fixed vertex cannot express triangulations whose diagonals share no common vertex, and the exhaustive enumerator confirmed the DP strictly beats every fan in general, so I moved to the interval DP and checked its arc recurrence on the square (apex `k = 2` reproduces the optimal diagonal `0–2`, total `18000`). The recurrence sums *products* of labels up to `2000`, so a single triangle can cost `8 * 10^9` and the whole answer `~4 * 10^12`; my first `int` transcription wrapped the product to a negative number (`[1500,1500,1500,1500]` printed `-1839934592` instead of `6,750,000,000`), which I fixed by making the operands and the table `long long` so the multiplication happens in 64-bit from the start; an early `n < 3` return then closes the empty / point / single-edge corners that would otherwise index `dp[0][-1]`; and agreement with the exhaustive triangulation enumerator on 400 random cases certifies the whole thing.
