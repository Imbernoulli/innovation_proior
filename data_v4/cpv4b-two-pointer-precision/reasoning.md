I have `n` amplifier modules, each with a positive integer gain `g[i]`, and a tolerance factor `p/q` (positive integers with `p >= q`). A pair `{i, j}` is *balanced* when the larger gain over the smaller is at most `p/q`; I count the balanced unordered pairs, reading `n p q` then the `n` gains from stdin and printing one integer. The constraint line is what fixes the whole shape of this problem: `n <= 2*10^5`, but `1 <= q <= p <= 4*10^9` and `1 <= g[i] <= 4*10^9`. Two things follow at once. `4*10^9 > 2^31 ~ 2.1*10^9`, so `p`, `q`, and the gains do not fit in a 32-bit `int` — they need `long long` on read. And the natural ratio test multiplies two of these, and `4*10^9 * 4*10^9 = 1.6*10^19` sails past the signed 64-bit ceiling `2^63 - 1 ~ 9.22*10^18`. So even `long long` cannot hold the product I most need to compute. That overflow, not the sweep, is the real difficulty here.

Division is the naive route, and it is poison. `g_hi / (double)g_lo <= p / (double)q` carries only a 53-bit mantissa (~`9*10^15` of integer resolution), and a pair's ratio can sit arbitrarily close to the bound — on a pair whose true ratio equals `p/q` exactly, the rounded quotient lands on either side and flips the count. So I refuse division. Since `g_lo, g_hi, p, q` are all strictly positive, multiplying `g_hi / g_lo <= p / q` through by the positive `g_lo` and `q` gives the equivalent

```
g_hi * q  <=  g_lo * p .
```

No division, no rounding — two integer products and one comparison. The only open question is what type holds those products.

Two routes. Quadratic enumeration compares every unordered pair directly with this cross-multiplication — obviously correct, and exactly what an independent oracle should do, but `C(2*10^5, 2) ~ 2*10^10` comparisons is hopeless under a 1-second limit. The affordable route is sort-then-two-pointers: sort the gains ascending and fix the *larger* member of a pair at index `j`, so any partner `i < j` has `g[i] <= g[j]` and the balanced test is just `g[j] * q <= g[i] * p`.

The monotonicity that makes the sweep linear: fix `j`, and as `i` ranges over `0..j-1`, `g[i]` is non-decreasing, so the right side `g[i] * p` grows while the left side `g[j] * q` stays constant — once the inequality holds at some `i = lo_j`, it holds for every larger `i` up to `j-1`, so the balanced partners form a contiguous suffix `[lo_j, j-1]`. Advance to `j+1`: `g[j+1] >= g[j]` raises the left side, so the smallest qualifying `i` cannot decrease, `lo_{j+1} >= lo_j`. The boundary is monotone non-decreasing, so a single trailing pointer suffices and the work after the sort is `O(n)`.

Running the formula on the sample `g = [10, 1, 4, 8, 13, 5]`, `p = 5`, `q = 2` (answer `8`): sorted `[1, 4, 5, 8, 10, 13]`, and for each `j` I take the smallest `lo` with `g[j]*2 <= g[lo]*5`, counting `j - lo`:

- `j=0,1` (g=1,4): no qualifying partner, `0` each.
- `j=2` (g=5): `10 <= g[1]*5 = 20`, partners `[1,1]` -> `1` (`{4,5}`).
- `j=3` (g=8): `16 <= 20`, partners `[1,2]` -> `2` (`{4,8}`, `{5,8}`).
- `j=4` (g=10): `20 <= 20` — equality, so `{4,10}` sits exactly on the bound and is counted; partners `[1,3]` -> `3`.
- `j=5` (g=13): `26 > 20` and `> g[2]*5 = 25` but `<= g[3]*5 = 40`, so `lo` advances to `3`; partners `[3,4]` -> `2` (`{8,13}`, `{10,13}`).

Total `0+0+1+2+3+2 = 8`, and `lo` moved only rightward (`0,1,1,1,1,3`), the monotonicity I derived. The on-threshold `{4,10}` counts only because the test is `<=` on exact integer products — a `double` quotient could round it out.

Written the naive way, the loop body is:

```
sort(g.begin(), g.end());
long long count = 0;
int lo = 0;
for (int j = 0; j < n; j++) {
    while (lo < j && g[j] * q > g[lo] * p) lo++;   // products in long long
    count += (long long)(j - lo);
}
```

The sample runs clean here because its values are tiny — the products never approach the 64-bit ceiling, so a `long long` comparison happens to agree. The danger only surfaces at scale, so I construct a pair that exercises it: `n = 2`, `p = 2590928623`, `q = 200071089`, `g = [3687093964, 3758591649]`. Sorted, `g_lo = 3687093964`, `g_hi = 3758591649`, and the exact cross-products are

```
g_hi * q = 751985524321735761
g_lo * p = 9552997287018131572
```

so `g_hi * q <= g_lo * p` is true — the pair is balanced, the answer should be `1`. But `g_lo * p ~ 9.55*10^18` exceeds `2^63 - 1`, so in `long long` it wraps to `-8893746786691420044`, while `g_hi * q ~ 7.52*10^17` stays intact. The loop condition `g[j] * q > g[lo] * p` becomes `751985524321735761 > -8893746786691420044`, which is true, so the code advances `lo`, declares the pair unbalanced, and prints `0`. The overflow silently flipped one product's sign and corrupted the verdict.

The fix is to form both products in 128-bit, where nothing wraps — `__int128` reaches `~1.7*10^38`, dwarfing the `1.6*10^19` worst case:

```
while (lo < j && (__int128)g[j] * q > (__int128)g[lo] * p) lo++;
```

Casting one operand promotes the whole multiplication to 128-bit, so `g_lo * p = 9552997287018131572` is represented faithfully, the comparison `751985524321735761 > 9552997287018131572` is false, `lo` holds, the pair counts, and the constructed case now prints `1`.

There is a quieter overflow in the count itself. With all-equal gains, `n = 2*10^5`, `p = q = 1`, every pair is balanced and the answer is `C(n, 2) = 200000 * 199999 / 2 = 19999900000 ~ 2*10^10`, past the 32-bit ceiling. Each `count += (j - lo)` adds a small `int`, but the running total climbs to `~2*10^10`, so `count` must be `long long` — which it is.

The corners this problem invites: `n = 0` and `n = 1` read fine and give `0` (no pair; the loop body never fires). All-equal gains with `p >= q` make every pair balanced (`g*q <= g*p` reduces to `q <= p`) — the `C(n,2)` count case above. `p = q` with distinct gains admits only equal-gain pairs, so `0`. On-threshold pairs (ratio exactly `p/q`, like `{4,10}`) are counted by `<=` on exact products — the case a `double` test misrounds. And a maximal-magnitude pair with both gains and `p, q` near `4*10^9` is exactly the `1.6*10^19` cross-product that needs `__int128`.

I cross-check against an independent `O(n^2)` oracle that compares each pair with exact rationals (`Fraction(hi, lo) <= Fraction(p, q)`) — a different mechanism entirely, no cross-multiplication of my own. A generator mixes three regimes: a *tight* mode with a small value alphabet and small `p/q` so many ratios land exactly on the bound (the misround trap), a *wide* mode with values and `p, q` up to `4*10^9` so cross-products hit `1.6*10^19` (the overflow trap), and a medium mixed regime. Across hundreds of random small cases the two-pointer solution and the rational oracle agree with zero mismatches.

What ships is one self-contained file: read into `long long`, sort, two-pointer sweep with the cross-products in `__int128` and the count in `long long`. The full module is in the answer.
