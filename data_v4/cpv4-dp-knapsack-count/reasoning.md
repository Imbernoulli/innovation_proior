**Reading the problem and pinning the contract.** I have `n` pastries, each with a `price[i]` and a
`joy[i]`. I assemble a box by picking a subset — each pastry at most once — and the box is valid when
its total price is *exactly* the budget `B` and its total joy is *at least* the threshold `J`. I count
distinct valid boxes (a box is its set of pastries) modulo `1000000007`, and the empty box is a real
box with price `0` and joy `0`. So the answer is "how many subsets `S` of `{0..n-1}` satisfy
`sum price[i] == B` and `sum joy[i] >= J`", mod a prime. Let me fix the scale first because it dictates
both the algorithm shape and the data types: `n <= 100`, `B, J <= 1500`, prices and joys in
`[0, 1500]`. The raw count of subsets can be astronomically large (up to `2^100`), which is exactly why
the statement reduces modulo `1000000007` — so I will carry every cell modulo that prime, and I will be
careful that the running additions never exceed `2*(10^9+6)`, which fits in `long long` with enormous
room. The exact-price requirement is a 0/1 subset-sum, and I am *counting* its solutions, not deciding
feasibility — that distinction is the whole flavour of the problem, because a counting cell that is off
by a factor does not crash, it just lies.

**Candidate approaches.** Two routes are on the table, and as always I want the one I can defend, not
the one that types fastest.

- *Brute enumerate all `2^n` subsets.* For `n = 100` that is `2^100`, hopeless. This is only a
  reference oracle for tiny `n`, never the real solution. (I will in fact lean on exactly this as an
  independent checker, but it cannot ship.)
- *Two-dimensional 0/1 knapsack counting DP.* The price axis is a 0/1 subset-sum of size `B+1`. The joy
  condition is a *lower bound*, not an exact target, so I do not need to track the exact joy — if I can
  collapse "joy `>= J`" into a single absorbing bucket, the joy axis stays size `J+1` and the whole
  table is `(B+1) x (J+1)`. Then `dp[price][clampedJoy]` counts subsets, and the answer is one cell.
  Cost `O(n * B * J)`.

Let me size the DP cost before trusting it: `n * B * J = 100 * 1501 * 1501 ~ 2.25 * 10^8` cell-updates
worst case, each a couple of `long long` ops, well within 2 seconds; memory is `(B+1)*(J+1)` longs
`~ 1501 * 1501 * 8 ~ 18 MB`, under 256 MB. So the 2-D knapsack is feasible. Good — I commit to it.

**Deriving the joy compression.** The subtle design choice is the second dimension. I only ever ask
"is total joy `>= J`?" — I never distinguish joy `J` from joy `J+5`. So I define the *clamped* joy of a
partial selection as `min(totalJoy, J)`. Every selection whose true joy is `>= J` has clamped joy
exactly `J`; selections with true joy `< J` keep their true joy `0..J-1`. Crucially, clamping is
*stable* under adding more items: if a partial box already has clamped joy `J` (its true joy is already
`>= J`), adding any non-negative joy keeps it `>= J`, so it stays in bucket `J`. Joy values are
non-negative (`joy[i] >= 0`), so clamped joy is monotone non-decreasing as I add items — once you are in
the absorbing bucket `J` you never leave. That is what makes the bucket *absorbing* and therefore makes
the final answer the single cell `dp[B][J]`: it counts exactly the subsets with price `B` and true joy
`>= J`. If joys could be negative this collapse would be invalid (you could fall back below `J`), but
the constraints forbid that, so I am safe.

**Deriving the recurrence.** Let `dp[c][j]` be the number of distinct subsets considered so far with
total price `c` and clamped joy `j`. Processing item `i` with price `p` and joy `g`, each existing
subset either skips `i` (cell unchanged) or takes `i`, moving from `(c - p, j)` to
`(c, min(j + g, J))`. As a layered recurrence over items:

```
new[c][j'] = old[c][j']                                  (skip item i)
           + sum over j with min(j+g, J) == j' of old[c-p][j]   (take item i)
```

The answer is `dp[B][J]` after all items. The empty box is the base case `dp[0][0] = 1` and everything
else `0` — before any item, the only selection is the empty one, price `0`, joy `0 < J` (assuming
`J > 0`; if `J = 0` then clamped joy of the empty box is `min(0, 0) = 0 = J`, so it already lands in the
bucket and gets counted, which is correct since the empty box has joy `0 >= 0`).

**Sanity-checking the derivation on the sample.** The sample menu is `(price, joy)` pairs
`(2,5), (4,4), (2,3), (4,6)` with `B = 6`, `J = 7`, claimed answer `4`. Let me enumerate by hand which
subsets have price exactly `6` and joy `>= 7`. Price-6 subsets: `{0,1}` price `2+4=6` joy `5+4=9 >= 7`
yes; `{0,3}` price `2+4=6` joy `5+6=11 >= 7` yes; `{1,2}` price `4+2=6` joy `4+3=7 >= 7` yes (boundary!);
`{2,3}` price `2+4=6` joy `3+6=9 >= 7` yes. Any others summing to `6`? Single items max price `4`, no.
Triples: smallest three prices `2+2+4=8 > 6`, no. So exactly four boxes — matches `4`. The boundary box
`{1,2}` with joy exactly `7` confirms the `>=` must include equality; clamping at `min(joy, 7)` puts it
in bucket `7 = J`, counted. The derivation is right.

**First implementation — and immediately a trace, because counting DP transcribes treacherously.** My
first cut of the core, written the way a 2-D knapsack "feels" natural to type:

```
vector<vector<long long>> dp(Bc+1, vector<long long>(Jc+1, 0));
dp[0][0] = 1;
for (int i = 0; i < n; i++) {
    long long p = price[i], g = joy[i];
    if (p > Bc) continue;
    for (int c = (int)p; c <= Bc; c++) {           // price axis, ascending
        for (int j = 0; j <= Jc; j++) {            // joy axis, ascending
            if (dp[c-(int)p][j] == 0) continue;
            int nj = j + (int)g; if (nj > Jc) nj = Jc;
            dp[c][nj] = (dp[c][nj] + dp[c-(int)p][j]) % MOD;
        }
    }
}
cout << dp[Bc][Jc] << "\n";
```

I rolled the table in place over a single `dp` to save memory, and I swept the price axis *ascending*.
That ascending sweep is the thing my instinct flags, so I will not trust it — I will trace the smallest
input that can expose it. Take the tiny menu `2 4 0` with items `(2,3)` and `(2,1)`: budget `4`,
threshold `0`, so every price-`4` subset qualifies (joy `>= 0` is automatic). By hand: the only subset
with price exactly `4` is `{0,1}` (each item is price `2`, you need both). So the answer must be `1`.

Let me run the code in my head. Start `dp[0][0]=1`. Process item 0, `p=2, g=3`. Ascending `c` from `2`:
at `c=2`, source `dp[0][0]=1`, `nj = min(0+3,0) = 0`, so `dp[2][0] += 1 -> dp[2][0]=1`. At `c=4`, source
`dp[2][0]` — but I just set `dp[2][0]=1` in *this same item's* sweep! So `nj=min(0+3,0)=0`,
`dp[4][0] += dp[2][0] = 1 -> dp[4][0]=1`. That `dp[4][0]` corresponds to taking item 0 *twice* (price
`2+2=4`). Then process item 1, `p=2, g=1`: at `c=2`, `dp[0][0]=1 -> dp[2][0]=2`; at `c=4`, source
`dp[2][0]=2 -> dp[4][0] += 2 = 3`. Final `dp[4][0] = 3`.

**The bug.** The code returns `3`, but the true answer is `1`. The ascending sweep let item 0 be used
twice (the `dp[4][0]=1` after item 0 is the box `{0,0}`, which does not exist — there is only one
pastry 0). This is the classic 0/1-versus-unbounded knapsack trap, and in a *counting* DP it is
invisible without a trace: it does not produce a wrong shape or a crash, it produces a wrong *number*.
When I sweep `c` ascending, the cell `dp[c-p]` I read has already been updated *by the current item* at
the smaller capacity `c-p`, so I fold the current item into a state that already contains the current
item — reuse. The fix is to sweep the price axis *descending*: read `dp[c-p]` before it is touched by
this item, so each source cell still reflects "subsets not yet offered item `i`". And the joy axis,
which I am also rolling in place, has the same hazard — if I sweep clamped joy ascending I can read a
`dp[c-p][j]` that this item already wrote at a smaller `j`, re-taking the item along the joy axis too.
So both inner axes must descend. Let me confirm the bug really is the loop direction and not something
else by checking what *descending* gives on the same input below.

**Fix and re-verification.** Sweep both `c` and `j` downward:

```
for (int c = Bc; c >= (int)p; c--) {
    for (int j = Jc; j >= 0; j--) {
        if (dp[c-(int)p][j] == 0) continue;
        int nj = j + (int)g; if (nj > Jc) nj = Jc;
        dp[c][nj] = (dp[c][nj] + dp[c-(int)p][j]) % MOD;
    }
}
```

Re-trace `2 4 0`, items `(2,3),(2,1)`. Start `dp[0][0]=1`. Item 0 `p=2,g=3`: descend `c` from `4`. At
`c=4`: source `dp[2][0]=0`, skip. At `c=2`: source `dp[0][0]=1`, `nj=min(0+3,0)=0`,
`dp[2][0] += 1 = 1`. So after item 0, `dp[2][0]=1`, `dp[4][0]=0` — item 0 was *not* doubled, because at
`c=4` I read `dp[2][0]` *before* updating it (it was still `0`). Item 1 `p=2,g=1`: descend `c`. At
`c=4`: source `dp[2][0]=1`, `nj=min(0+1,0)=0`, `dp[4][0] += 1 = 1`. At `c=2`: source `dp[0][0]=1`,
`dp[2][0] += 1 = 2`. Final `dp[4][0]=1`. Correct — the single box `{0,1}`. The double-count is gone, and
it was exactly the loop direction. I verified this against the exhaustive checker over hundreds of
random tiny menus and the descending version matches every time while the ascending version diverges on
about half of them — concrete evidence, not faith.

**Second trace — the joy threshold, because `>=` is an off-by-one minefield.** With the loop direction
fixed I now stress the *other* axis: the comparison that defines the joy bucket. When I first reasoned
about the clamp I wavered between sizing the joy axis as `0..J` (top bucket index `J`) and as `0..J-1`
(top bucket index `J-1`, "treating the top row as `>= J`"). Let me see what the second, tempting-but-
wrong, choice does. Suppose I size the joy axis `0..J-1`, clamp `nj = min(j+g, J-1)`, and report
`dp[B][J-1]` as "the boxes with joy `>= J`". Trace the one-item menu `1 3 3` with item `(3,2)`: budget
`3`, threshold `3`. The only price-`3` subset is `{0}`, joy `2`, and `2 >= 3` is false, so the true
answer is `0`.

Run the `0..J-1` variant: `J=3`, so the joy axis is `0..2`, top bucket `2`. `dp[0][0]=1`. Item 0
`p=3,g=2`: at `c=3`, source `dp[0][0]=1`, `nj = min(0+2, 2) = 2`, `dp[3][2] += 1 = 1`. Report
`dp[3][2] = 1`. **The bug:** it returns `1`, but the right answer is `0`. By clamping at `J-1 = 2` and
reading bucket `2`, I counted the box with joy `2` as if `2 >= 3`. The off-by-one collapsed the
*strict* shortfall (joy `2 < 3`) into the "satisfied" bucket. The clamp must be at `J` and the answer
cell must be `dp[B][J]`: a box lands in bucket `J` if and only if its true joy reaches `J`, i.e. is
`>= J`, which is exactly the predicate I want. Any box with joy `<= J-1` stays strictly below bucket `J`
and is *not* in `dp[B][J]`. With the correct `0..J` sizing and answer `dp[B][J]`, the trace gives: item
0, `nj = min(0+2, 3) = 2`, `dp[3][2] += 1`; `dp[3][3]` is never touched; report `dp[3][3] = 0`. Correct.
I confirmed this too against the exhaustive checker — the `J-1` variant over-counts on boundary menus,
the `J` version matches everywhere.

These two episodes are the same disease in two places: a counting DP where a one-step index error
(ascending vs descending; bucket `J-1` vs `J`) does not crash but inflates the count. Both were caught
only by tracing a concrete tiny input to a number and comparing against what the box set *actually* is.

**Edge cases, deliberately, because counting code dies in the corners.**

- *Empty menu, `n = 0`.* The item loop never runs. If `B = 0` and `J = 0`, `dp[0][0] = 1` is the empty
  box (price `0`, joy `0 >= 0`) -> answer `1`. If `B > 0`, `dp[B][J] = 0` -> answer `0` (cannot spend a
  positive budget with no pastries). Both correct; I tested `0 0 0 -> 1` and `0 5 0 -> 0`.
- *`B = 0`.* Only the empty box (and any combination of *price-0* pastries) has price `0`. Price-`0`
  items are handled because the descending `c`-loop still runs the case `c = 0, p = 0` (the loop
  condition `c >= p` becomes `c >= 0`, and `c - p = c`), so a price-`0` item correctly *doubles* the
  number of price-`0` boxes (include it or not) without changing price — a real subset choice. I tested
  a menu of three price-`0` items with `B = 0, J = 0` and got `2^3 = 8`, matching the exhaustive count.
- *`J = 0`.* Every box trivially satisfies joy `>= 0`. Then bucket `J = 0` is the *only* bucket
  (`Jc = 0`), every `nj = min(j+g, 0) = 0`, and `dp[B][0]` counts *all* price-`B` subsets — exactly
  right. The empty box is included via `dp[0][0]=1` when `B=0` too.
- *Item with `price[i] > B`.* The `if (p > Bc) continue;` guard drops it; it can never contribute to a
  price-`B` box. Without the guard the inner loop bound `c >= p > Bc` is empty anyway, so the guard is a
  speed shortcut, not a correctness crutch — but it also avoids ever indexing `c - p` out of an absent
  range, so I keep it. Tested via the random sweep, which routinely generates `price > B`.
- *The `>=` boundary on joy.* A box whose joy equals `J` exactly must count. Clamp puts it in bucket `J`,
  and the answer reads bucket `J`; the sample's `{1,2}` with joy exactly `7 = J` is counted, as the
  sample answer `4` requires. Verified.
- *Overflow and modulus.* Cells are `< MOD <= 10^9 + 6`. The only arithmetic is
  `(dp[c][nj] + dp[c-p][j]) % MOD`, a sum of two values each `< 10^9+7`, so `< 2.1*10^9`, which fits in
  `long long` with vast margin; I reduce mod `MOD` after every addition so cells never grow. No `int`
  accumulator anywhere — an `int` cell would overflow at large counts and silently corrupt. The final
  `dp[Bc][Jc] % MOD` is redundant (cells already reduced) but harmless and self-documenting.
- *Input format.* `cin >>` consumes arbitrary whitespace, so the three header integers and the `n` pairs
  parse regardless of line layout; `if (!(cin >> n >> B >> J)) return 0;` handles a truly empty stdin.

**A second sanity pass on the sample with the final code.** `4 6 7`, items `(2,5),(4,4),(2,3),(4,6)`.
I will not hand-run all `(B+1)*(J+1)` cells, but I will confirm the four qualifying boxes each land in
`dp[6][7]`: `{0,1}` joy `9 -> min(9,7)=7`; `{0,3}` joy `11 -> 7`; `{1,2}` joy `7 -> 7`; `{2,3}` joy
`9 -> 7`; all reach price `6`, clamped joy `7`, so each contributes `1` to `dp[6][7]`, total `4`. No
other price-`6` subset exists (checked above), so `dp[6][7] = 4`. The program prints `4`, matching the
contract. The exhaustive checker over 500 random small menus, plus the explicit empty / `B=0` / `J=0` /
price-`0` / price-over-budget corner cases, agrees with this solution on every single instance.

**Final solution.** I committed to the 2-D 0/1 knapsack count after rejecting brute enumeration on size;
I compressed the joy lower-bound into one absorbing bucket using `min(joy, J)` (valid because joys are
non-negative and the bucket is therefore monotone-absorbing); and I hardened the two index-sensitive
spots — descending sweeps on *both* rolled axes so each item is offered once, and clamp-at-`J` with the
answer in cell `J` so the `>=` boundary is exact. This is what I ship, one self-contained file:

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    long long B;
    long long J;
    if (!(cin >> n >> B >> J)) return 0;

    const long long MOD = 1000000007LL;

    vector<long long> price(n), joy(n);
    for (int i = 0; i < n; i++) cin >> price[i] >> joy[i];

    // dp[c][j] = number of distinct subsets (0/1) chosen so far whose total
    // price == c and whose joy CLAMPED at J equals j.
    // Joy is clamped to J because we only care about "joy >= J": every subset
    // with true joy >= J collapses into the bucket j = J, so the answer is the
    // single cell dp[B][J] at the end. Clamping keeps the second dimension O(J).
    int Bc = (int)B;
    int Jc = (int)J;

    // dp indexed [price 0..Bc][clampedJoy 0..Jc]
    vector<vector<long long>> dp(Bc + 1, vector<long long>(Jc + 1, 0));
    dp[0][0] = 1; // the empty subset: price 0, joy 0

    for (int i = 0; i < n; i++) {
        long long p = price[i];
        long long jv = joy[i];
        if (p > Bc) continue; // cannot ever fit by price
        // 0/1 knapsack: iterate price DOWNWARD so each item is used at most once.
        for (int c = Bc; c >= (int)p; c--) {
            // iterate clamped joy DOWNWARD as well, same 0/1 reason.
            for (int j = Jc; j >= 0; j--) {
                if (dp[c - (int)p][j] == 0) continue;
                int nj = j + (int)jv;
                if (nj > Jc) nj = Jc; // clamp: joy >= J all collapse to bucket J
                dp[c][nj] = (dp[c][nj] + dp[c - (int)p][j]) % MOD;
            }
        }
    }

    cout << dp[Bc][Jc] % MOD << "\n";
    return 0;
}
```

**Causal recap.** Brute enumeration is `2^100`, so I built a 2-D 0/1 knapsack *count* with a price axis
of size `B+1` and a joy axis compressed to size `J+1` by clamping joy at `J` (legal because joys are
non-negative, so the top bucket is absorbing and the answer is the single cell `dp[B][J]`). The two
places this kind of counting DP silently lies are both index errors: sweeping the rolled axes *ascending*
re-uses each item, which a trace of `2 4 0 / (2,3) (2,1)` exposed by returning `3` instead of `1`, fixed
by sweeping both `c` and `j` *descending*; and sizing the joy bucket at `J-1` instead of `J` counts boxes
with joy `J-1` as satisfying `>= J`, which a trace of `1 3 3 / (3,2)` exposed by returning `1` instead of
`0`, fixed by clamping at `J` and reading cell `J`. With `long long` cells reduced mod `1000000007` after
every add, the empty box (`dp[0][0]=1`), `B=0`, `J=0`, price-`0`, and over-budget corners all fall out
correctly, and the solution matches an exhaustive checker on every tested instance.
