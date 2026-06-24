**Reading the problem and pinning the contract.** I have `n` points with integer coordinates and I must find, over every unordered triple of distinct points, the triangle of maximum area, then print **twice** that area as an exact integer. The empty/degenerate case is allowed: if `n < 3` or every triple is collinear, the answer is `0`. Input on stdin is `n` followed by `n` coordinate pairs; I print one integer. Before I pick an algorithm I fix the scale, because here the scale *is* the problem: `n <= 500` and `|x|, |y| <= 10^9`. The first number tells me `O(n^3)` is fine — `500^3 / 6 ≈ 2.1 * 10^7` triples, nothing for a 2-second limit. The second number is the landmine: coordinate differences reach `2 * 10^9`, a single product `(Bx-Ax)*(Cy-Ay)` reaches `4 * 10^18`, and the cross product is a difference of two such products, so it can reach about `8 * 10^18` in magnitude. That sits *inside* `long long` (max `≈ 9.22 * 10^18`) but is roughly four billion times too big for a 32-bit `int` (max `≈ 2.1 * 10^9`). So every coordinate, every difference, every product, and the running best must be 64-bit. I write this down now: an `int` anywhere in the cross product is a silent wrong answer, not a crash — the worst kind.

**Why twice the area, and why that is exact.** I deliberately report *twice* the area rather than the area itself. Twice the signed area of triangle `ABC` is the 2-D cross product

```
cross = (Bx - Ax) * (Cy - Ay) - (Cx - Ax) * (By - Ay),
```

and twice the *area* is `|cross|`. For integer coordinates this is an integer, with no division and therefore no floating point and no rounding. If I instead printed the area, I would have a half-integer (`cross/2`) and would be tempted into `double`, which cannot represent values near `4 * 10^18` exactly (a `double` has 53 bits of mantissa, good to about `9 * 10^15`). Reporting twice the area sidesteps all of that: the answer is an exact `long long`. Good — the contract is doing me a favour, I just have to honour it with the right type.

**Laying out the candidate approaches.** Two routes, and I want the one I can defend rather than the one that looks clever.

- *Brute force over all triples.* For every `i < j < k` compute `|cross|` and keep the maximum. `O(n^3)`, trivially correct because it literally examines every triangle. With `n <= 500` it fits the time limit with room. The only thing that can go wrong is the arithmetic — which is precisely the declared pitfall — so this approach concentrates all the risk in one well-understood place.
- *Convex hull then search.* The maximum-area triangle has all three vertices on the convex hull, so I could build the hull and search only hull vertices, possibly with a rotating-calipers-style sweep. Asymptotically faster, but it is far more code, the hull build has its own collinear-point edge cases, and the "largest triangle has hull vertices" fact, while true, needs the search over the hull done correctly. For `n <= 500` this buys nothing and adds three new ways to be wrong. I reject it on a cost/benefit basis: brute force is provably correct and fast enough.

So the plan is brute force, and the entire engineering problem reduces to "compute the cross product without overflowing and take the max."

**Deriving the cross-product formula and sanity-checking it.** Let me make sure the formula is the one I think it is, on a triangle whose area I know cold. Take `A = (0,0)`, `B = (4,0)`, `C = (0,3)` — a right triangle with legs 4 and 3, area `6`, so twice the area should be `12`. Plug in: `(Bx-Ax)=4`, `(Cy-Ay)=3`, `(Cx-Ax)=0`, `(By-Ay)=0`. `cross = 4*3 - 0*0 = 12`. `|cross| = 12`. Matches. Now check orientation independence by swapping `B` and `C` (which flips orientation): `A=(0,0)`, `B=(0,3)`, `C=(4,0)`. `(Bx-Ax)=0`, `(Cy-Ay)=0`, `(Cx-Ax)=4`, `(By-Ay)=3`. `cross = 0*0 - 4*3 = -12`, `|cross| = 12`. Same magnitude, sign flipped — exactly as expected, and since I only keep `|cross|` the order of the three points inside a triple does not matter. The formula is right and the absolute value makes triple-ordering irrelevant, so iterating `i < j < k` once per triple is sufficient.

**First implementation — and I trace it immediately, because clean math transcribes dirty.** Here is my first cut of the core, written the way a tired person writes it — using `int` for the local differences because "coordinates are read into `int`-looking variables and differences feel small":

```
int best = 0;                       // <-- first mistake hiding here
for (int i = 0; i < n; i++)
  for (int j = i+1; j < n; j++)
    for (int k = j+1; k < n; k++) {
      int abx = x[j]-x[i], aby = y[j]-y[i];   // <-- int differences
      int acx = x[k]-x[i], acy = y[k]-y[i];
      int cross = abx*acy - acx*aby;          // <-- int product: overflow
      best = max(best, abs(cross));
    }
```

I do not trust this; I trace the smallest input that should expose the scale problem. Take three points `A=(0,0)`, `B=(10^9,0)`, `C=(0,10^9)` — a right triangle with legs `10^9`, area `5*10^17`, so twice the area is `10^18`. Walk the arithmetic *as 32-bit ints*: `abx = 10^9 - 0 = 1000000000` (this already fits `int`, barely — `int` max is `2147483647`). `acy = 10^9`. The product `abx*acy = 10^9 * 10^9 = 10^18` — but computed in `int`, this wraps around modulo `2^32`. `10^18 mod 2^32` is some essentially random-looking 32-bit value; the sign can even come out negative. So `cross` is garbage, `abs(cross)` is garbage, and `best` ends up nonsense instead of `10^18`.

**Diagnosing the bug.** To make the failure undeniable rather than hand-waved, I reproduce the exact computation in a tiny program: take the same `cross = (x1-x0)*(y2-y0) - (x2-x0)*(y1-y0)` with `A=(0,0)`, `B=(10^9,0)`, `C=(0,10^9)`, computed once with `int` casts and once with `long long`. The `int` version prints `-1486618624`; the `long long` version prints `1000000000000000000`. That is the whole disease in one line: the correct twice-area is `1000000000000000000`, but the 32-bit path returns `-1486618624` — wrong magnitude *and* wrong sign. The product `10^9 * 10^9 = 10^18` overflowed `int` by a factor of roughly `4.6 * 10^8`, the high bits were discarded, and what survived even crossed into the sign bit. The defect is not in the formula — the formula sanity-checked perfectly — it is purely the data type. Three places are infected: the per-edge differences `abx, aby, acx, acy`, the product/`cross`, and the accumulator `best`. The differences *barely* fit `int` (a coordinate difference can be `2*10^9 > 2.1*10^9`? — `2*10^9 = 2000000000 < 2147483647`, so a single difference squeaks in, but `-10^9 - 10^9 = -2*10^9 = -2000000000 > -2147483648`, also in range) — but the *product* of two such differences absolutely does not, and that is enough.

**Fix and re-verification.** The fix is to make every quantity along the chain `long long`: read coordinates into `long long`, compute differences as `long long`, form the product and `cross` as `long long`, and keep `best` as `long long`. Then `10^9 * 10^9` is computed in 64-bit where `10^18` is exact, and the worst case `8 * 10^18` still fits. I re-run the trace mentally on `A=(0,0)`, `B=(10^9,0)`, `C=(0,10^9)` with `long long`: `abx = 10^9`, `acy = 10^9`, `abx*acy = 10^18`, `acx = 0`, `aby = 0`, so `cross = 10^18 - 0 = 10^18`, `|cross| = 10^18`, `best = 10^18`. Correct. And I confirm with the actual compiled solution on the documented sample (square corners plus centre): it prints `1000000000000000000`, exactly the expected twice-area of a corner right triangle. The bug was the type and only the type; with `long long` end to end it is gone.

**A second, subtler trace — does `abs` itself betray me?** Even with `long long` arithmetic I have to be careful about *which* absolute-value function I call. `std::abs(int)` and the C `abs` take and return `int`; if I write `abs(cross)` where `cross` is `long long`, in some translation units that can bind to the `int` overload and truncate a perfectly-good 64-bit `cross` back into 32 bits — re-introducing the very overflow I just eliminated, but now downstream of correct arithmetic, which is even sneakier. I trace it: suppose `cross = 8*10^18` (worst case, computed correctly in 64-bit). If `abs` truncates to `int`, `8*10^18 mod 2^32` is garbage again and `best` is wrong despite the multiply being right. To kill this whole class of ambiguity I use `llabs(cross)`, which is unambiguously `long long -> long long`, or equivalently `cross < 0 ? -cross : cross` in `long long`. I pick `llabs`. Re-checking the worst case: `llabs(8*10^18) = 8*10^18`, no truncation. Safe. This is the kind of bug that passes every small test (where `cross` fits `int` anyway) and only fails on the large hidden tests — exactly why I trace the large regime explicitly instead of trusting small samples.

**Edge cases, deliberately, because this is where geometry code dies.** I check the corners by hand and against the compiled binary:

- `n = 0`: the triple loop never runs; `best` stays `0`. Output `0`. Correct — no triangle exists.
- `n = 1` (`[(5,5)]`) and `n = 2` (`[(0,0),(3,4)]`): still no triple `i<j<k`, the innermost loop body never executes, `best = 0`. Output `0`. Correct — you need three points for a triangle. I run both: both print `0`.
- *All collinear*, `[(0,0),(1,1),(2,2)]`: the one triple has `cross = (1-0)*(2-0) - (2-0)*(1-0) = 2 - 2 = 0`, `|cross| = 0`, so `best` stays `0`. Output `0`. Correct — a degenerate triangle has zero area, and `best` initialized to `0` already covers "no positive-area triangle exists." I run it: `0`.
- *Coincident points*: if two of the three points are equal, two edge vectors collapse and `cross = 0`; harmless, contributes `0`, never lowers `best`. The problem statement explicitly allows duplicates, and the cross product handles them with no special case.
- *Maximum-overflow corners*, the four corners of a `2*10^9`-wide square `(\pm 10^9, \pm 10^9)`: the largest triangle is half the square, twice-area `= 2*10^9 * 2*10^9 / ... ` — let me just compute one: `A=(10^9,10^9)`, `B=(-10^9,10^9)`, `C=(-10^9,-10^9)`. `abx = -2*10^9`, `aby = 0`, `acx = -2*10^9`, `acy = -2*10^9`. `cross = (-2*10^9)*(-2*10^9) - (-2*10^9)*0 = 4*10^18`. The binary prints `4000000000000000000` — exact, and this single product `4*10^18` is the concrete value that an `int` could never hold. This is the test that separates a correct submission from an overflowing one.
- *Output format*: exactly one integer and a newline; `cin >>` skips arbitrary whitespace, so line-broken or space-separated input both parse.

**Stress verification against an independent brute force.** Hand traces convince me of the formula and the type; to convince me of the *implementation* I run the compiled `long long` solution against a completely independent Python brute force that computes twice the area with the shoelace expansion `ax*(by-cy) + bx*(cy-ay) + cx*(ay-by)` in Python's arbitrary-precision integers (so it can never overflow, by construction) over all triples. A generator emits four regimes: tiny coordinates `[-5,5]` (dense with collinear and duplicate triples), small `[-50,50]`, full magnitude `[-10^9,10^9]` (the overflow regime), and pinned corners `\pm 10^9` (the extreme). Across 500 random cases the two agree on every single one — zero mismatches. The shoelace form in the brute is algebraically a different arrangement of the same cross product, so agreement is real evidence and not a copy of my own arithmetic. (An earlier run *appeared* to show mismatches, but tracing them showed the fault was my test harness reusing one shared temp file across loop iterations so outputs interleaved; isolating each case to its own input file made the mismatches vanish — the solution was never wrong, my scaffolding was. Worth recording, because "the test is broken, not the code" is itself a trap.)

**Final solution.** I convinced myself the idea is right by sanity-checking the cross-product formula on a known triangle and by rejecting the hull approach on cost, and I convinced myself the *code* is right by tracing the `int` path to the exact garbage value `-1486618624` it produces on `10^9 * 10^9`, fixing it to `long long` end to end (plus `llabs` to stop a truncating overload), and then re-verifying on every corner and on 500 random cases against an overflow-proof brute. That is what I ship — one self-contained file, the simple `O(n^3)` brute over triples with strictly 64-bit arithmetic:

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    if (!(cin >> n)) return 0;
    vector<long long> x(n), y(n);
    for (int i = 0; i < n; i++) cin >> x[i] >> y[i];

    // best holds twice the maximum triangle area (a non-negative integer).
    // With |coord| <= 1e9, edge differences reach 2e9 and the cross-product
    // products reach ~4e18, so all of this MUST be long long.
    long long best = 0;
    for (int i = 0; i < n; i++)
        for (int j = i + 1; j < n; j++)
            for (int k = j + 1; k < n; k++) {
                long long abx = x[j] - x[i];
                long long aby = y[j] - y[i];
                long long acx = x[k] - x[i];
                long long acy = y[k] - y[i];
                long long cross = abx * acy - acx * aby; // = 2 * signed area
                long long twiceArea = llabs(cross);
                if (twiceArea > best) best = twiceArea;
            }

    cout << best << "\n";
    return 0;
}
```

**Causal recap.** Reporting *twice* the area makes the answer an exact integer, which is what lets me avoid `double` entirely; brute force over all triples is provably correct and fast enough at `n <= 500`, so the only real risk is arithmetic scale. A trace of `10^9 * 10^9` showed the textbook `int` cross product returning `-1486618624` instead of `10^18` — overflow that wraps and even flips sign — so I made coordinates, differences, the product, `cross`, and `best` all `long long`, and a second trace showed `abs` could re-truncate a correct 64-bit value, which `llabs` fixes; then `n < 3`, all-collinear, coincident, and the `4*10^18` corner case all check out by hand and on the binary, and 500 random cases against an arbitrary-precision shoelace brute (after I noticed an *apparent* mismatch was just my harness sharing a temp file) close it out at zero mismatches.
