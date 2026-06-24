**Reading the problem and pinning the contract.** I am at the origin and I have `n` beacons `P[0..n-1]`, each a lattice point `(x_i, y_i)` whose coordinates may be negative or zero. For an ordered pair `i < j` the *signed sweep* is `sweep(i, j) = x_i*y_j - x_j*y_i`, which is the 2D cross product of `P[i]` and `P[j]` — twice the signed area of triangle `O, P[i], P[j]`. I must report the maximum sweep over all `i < j`. Two corners are stated explicitly and I want them in front of me before I write a line: the maximum sweep can be *negative* (every pair might turn clockwise), and when `n < 2` there is *no* ordered pair and I must print the word `NONE`. So the output is either `NONE` or a single integer that is allowed to be negative or zero.

Let me fix the scale first, because it dictates the type. `n <= 2000` and `|x_i|, |y_i| <= 10^6`. A single product `x_i*y_j` reaches `10^6 * 10^6 = 10^12`, and the difference of two such products reaches `2*10^12` in magnitude. That is far past the 32-bit range of about `2.1*10^9`, so every coordinate and every accumulator that touches a product must be 64-bit. I will read into `long long` and keep the running maximum as `long long`. With `int` here the large-coordinate tests are a silent wrong-answer, not a crash, which is the worst kind. That decision is non-negotiable.

**Laying out the candidate approaches.** Two routes are on the table, and I want to commit to the one I can *prove* and *afford*, not the one that looks clever.

- *Angular sort then pair extremes.* The intuition is that the most counter-clockwise sweep should pair the "most clockwise" direction with the "most counter-clockwise" one, so sort the beacons by `atan2(y, x)` and try the angular extremes. It is `O(n log n)`. But two things worry me. First, the objective is `i < j` *in input order*, not over unordered pairs, so any reordering by angle changes which pairs are even legal — I would have to be very careful that the pair I select respects the original index order. Second, `atan2` is floating point, and with collinear and zero-coordinate beacons the angle ties are exactly where float comparisons go wrong. The cleverness is buying me trouble I do not need.
- *Direct enumeration.* With `n <= 2000` there are at most `n*(n-1)/2 ~ 2*10^6` ordered pairs. Evaluating `sweep(i, j)` for every `i < j` and tracking the maximum is `O(n^2)`, uses only integer arithmetic, and respects `i < j` by construction. At ~2 million iterations of a couple of multiplications this runs in well under the 1-second limit. There is no floating point, no sort-order subtlety. The only risk is transcription: the initial value of the running maximum, the loop bounds, and the type.

I will take direct enumeration. The cost is affordable, the arithmetic is exact, and the `i < j` constraint is automatic. The angular approach would be faster asymptotically but it is solving a problem (`O(n log n)` for huge `n`) I do not have, while importing floating-point fragility I would have to defend.

**Deriving the objective and a sanity check on paper.** The quantity is `sweep(i, j) = x_i*y_j - x_j*y_i`. Let me confirm I have the sign convention and the maximum right on the stated sample: beacons `(2,0), (0,2), (-2,0), (0,-2)` as `P0, P1, P2, P3`. I enumerate every `i < j`:

- `(0,1)`: `2*2 - 0*0 = 4`
- `(0,2)`: `2*0 - (-2)*0 = 0`
- `(0,3)`: `2*(-2) - 0*0 = -4`
- `(1,2)`: `0*0 - (-2)*2 = 4`
- `(1,3)`: `0*(-2) - 0*2 = 0`
- `(2,3)`: `(-2)*(-2) - 0*0 = 4`

The maximum is `4`, matching the stated answer. Note the spread already exhibits the pitfall: pair `(0,3)` is `-4`, a *negative* sweep, sitting in the same instance as the positive `4`. If I ever initialize the running maximum to `0` I would still get `4` here and feel safe — the bug only bites when *every* pair is negative. I make a mental note to build exactly that adversarial instance later.

**First implementation — and immediately a trace, because clean math transcribes dirty.** My first cut, fresh from the framework:

```
long long best = 0;                 // running maximum
for (int i = 0; i < n; i++) {
    for (int j = i + 1; j < n; j++) {
        long long cr = x[i]*y[j] - x[j]*y[i];
        if (cr > best) best = cr;
    }
}
cout << best << "\n";
```

and separately, the `n < 2` guard prints `NONE`. The line that makes me uneasy is `long long best = 0`. The problem said in plain words that the maximum sweep can be negative, and `0` is a value the true answer can never be forced down to from above — initializing at `0` quietly asserts "the answer is at least 0," which is a claim about the data, not a neutral starting point. Let me trace the smallest input that could expose it.

Construct an instance where every ordered pair turns clockwise. Three beacons sweeping clockwise around the origin: `P0 = (0, 3)` (pointing up), `P1 = (3, 0)` (pointing right), `P2 = (4, -1)` (pointing down-right). Going `0 -> 1 -> 2` is a clockwise fan, so I expect all sweeps negative. Compute:

- `(0,1)`: `0*0 - 3*3 = -9`
- `(0,2)`: `0*(-1) - 4*3 = -12`
- `(1,2)`: `3*(-1) - 4*0 = -3`

The true maximum is `-3` (the least-negative pair `(1,2)`). Now run my first cut by hand. Start `best = 0`. Pair `(0,1)`: `cr = -9`, is `-9 > 0`? No, `best` stays `0`. Pair `(0,2)`: `cr = -12 > 0`? No. Pair `(1,2)`: `cr = -3 > 0`? No. Final `best = 0`, printed as `0`.

**Diagnosing the first bug.** The code prints `0` but the correct answer is `-3`. The defect is exactly the one I suspected: initializing `best = 0` injected a phantom candidate of value `0` that does not correspond to any real pair, and since every real sweep here is negative, that phantom won and suppressed the true maximum `-3`. This is the classic wrong-base-case / sign-handling failure for a max-over-a-set: the identity for `max` is `-infinity`, not `0`. `0` is only correct when the problem floors the answer at `0` (e.g. "empty selection allowed"), and *this* problem does no such thing — there is no empty pair, you must pick a real `(i, j)`. So the base must be a sentinel below every achievable sweep, or it must be seeded from the first real pair. The minimum achievable sweep is `-(2*10^12)`, so any sentinel comfortably below that works; `LLONG_MIN` (about `-9.2*10^18`) is far below and is the cleanest choice because it is unambiguously "no candidate yet."

**Fixing the base case and re-verifying.** Replace the initializer:

```
long long best = LLONG_MIN;         // no real pair seen yet
```

Re-trace the clockwise instance. Start `best = LLONG_MIN`. Pair `(0,1)`: `cr = -9`, is `-9 > LLONG_MIN`? Yes, `best = -9`. Pair `(0,2)`: `cr = -12 > -9`? No. Pair `(1,2)`: `cr = -3 > -9`? Yes, `best = -3`. Final `best = -3`. Correct. And re-trace the positive sample `(2,0),(0,2),(-2,0),(0,-2)`: the first pair sets `best = 4`, later pairs are `0, -4, 4, 0, 4`, none exceeds `4`, so it prints `4`. Still correct. The instance that broke now passes, and it broke for precisely the reason I fixed — that is the evidence I trust, not the fact that the sample happened to still work.

One subtlety I must double-check after choosing `LLONG_MIN`: do I ever *add* to or *subtract from* the sentinel? If I did, `LLONG_MIN - something` underflows. But `best` is only ever compared (`cr > best`) and assigned (`best = cr`); arithmetic happens only on real coordinates inside `cr = x[i]*y[j] - x[j]*y[i]`, whose magnitude is bounded by `2*10^12`. The sentinel never participates in arithmetic, so it cannot underflow. Safe.

**A second trace, on the no-pair corner, because guards are where off-by-one lives.** The contract says `n < 2` prints `NONE`. I want to be sure my guard fires for both `n = 0` and `n = 1`, and that I never read coordinates that are not there. My structure is: read `n`, then read `n` coordinate pairs into `x`, `y`, *then* check `if (n < 2) print NONE`. Trace `n = 1`, single beacon `(5, -7)`: I read `n = 1`, read one pair into `x[0]=5, y[0]=-7`, then `n < 2` is true, print `NONE`. The double loop never runs because even if I let it, `i` goes `0..0` and the inner `j = i+1 = 1` is `>= n = 1`, so the inner body never executes and `best` stays `LLONG_MIN` — but I short-circuit before that anyway. Trace `n = 0`: I read `n = 0`, the read loop runs zero times, `n < 2` is true, print `NONE`. Good. Both corners hit the same guard.

But wait — there is a quieter failure mode I almost shipped. What if I had put the `if (n < 2)` check *before* reading the coordinates and used `return 0` with no output for `n = 0`? Then `n = 0` would print nothing at all, but the contract demands the literal `NONE`. Printing nothing is a wrong answer, not an empty-but-acceptable one. So the guard must *print* `NONE`, not merely bail. Let me also make sure I do not confuse the two early exits: `if (!(cin >> n)) return 0;` handles the case of *no header token at all* (genuinely empty input stream), which is distinct from `n = 0` (a header of `0` followed by no points). For a present `n = 0` I still owe `NONE`. My final code keeps these separate: the `cin >> n` failure returns silently (there is nothing to answer), while a successfully-read `n < 2` prints `NONE`.

**Edge cases, deliberately, because this is where this kind of code dies.**

- `n = 0`: read header `0`, read zero points, `n < 2` true, print `NONE`. Correct — there is no pair.
- `n = 1`: read one point, `n < 2` true, print `NONE`. Correct.
- All-clockwise (`(0,3),(3,0),(4,-1)`): answer `-3`, a negative maximum, handled because `best` starts at `LLONG_MIN`. This is the core pitfall and it now passes.
- Collinear / zero beacons: e.g. `(1,1),(2,2)` are collinear with the origin, `sweep = 1*2 - 2*1 = 0`; the running max correctly reports `0`, distinct from `NONE`. Zeros in coordinates simply flow through the integer products; `(0,0)` against anything gives `0`, which is a legitimate candidate, not a special case.
- Duplicated points: `P[i] = P[j]` gives `x_i*y_i - x_i*y_i = 0`, a real `0` sweep; no division anywhere, so duplicates are harmless.
- Overflow: products reach `10^12`, the difference reaches `2*10^12`, both inside `long long`. The sentinel `LLONG_MIN` never enters arithmetic. With `int` the products would wrap; with `long long` there is room to spare.
- Output format: exactly one line — either `NONE` or one integer. `cin >>` skips arbitrary whitespace, so the input layout (one pair per line or all on one line) does not matter.

**A derivation cross-check on a known geometric fact.** The maximum twice-signed-area of triangle `O, A, B` for `A, B` on the unit-area diamond should be achievable by orthogonal-ish directions. For `(2,0)` and `(0,2)` the sweep is `2*2 - 0*0 = 4`, i.e. area `2`, which is exactly the area of the right triangle with legs `2` and `2` — `(1/2)*2*2 = 2`, times two is `4`. The arithmetic and the geometry agree, so my primitive is the orientation cross product I intended, not its negation or a dot product.

**Final solution.** I convinced myself the *approach* is right by rejecting the floating-point angular sort in favor of exact integer enumeration that respects `i < j` and fits the time limit, and I checked the cross-product primitive against a known area. I convinced myself the *code* is right by tracing an all-clockwise instance to the precise wrong-base-case bug (`best = 0` injecting a phantom `0` that beats every negative sweep), fixing it to `LLONG_MIN`, re-verifying, and then tracing the `n < 2` guard for both `0` and `1`. That is what I ship — one self-contained `O(n^2)` file with 64-bit arithmetic and the two corners (negative maximum, no-pair) handled explicitly:

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    if (!(cin >> n)) { return 0; }              // no header at all -> nothing to do
    vector<long long> x(n), y(n);
    for (int i = 0; i < n; i++) cin >> x[i] >> y[i];

    if (n < 2) {                                 // no ordered pair i<j exists
        cout << "NONE" << "\n";
        return 0;
    }

    // best signed area*2 over ordered pairs i<j: cross(P[i],P[j]) = x_i*y_j - x_j*y_i.
    // Must start from a REAL pair, not 0, because every cross product can be negative.
    long long best = LLONG_MIN;
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            long long cr = x[i] * y[j] - x[j] * y[i];
            if (cr > best) best = cr;
        }
    }

    cout << best << "\n";
    return 0;
}
```

**Causal recap.** The angular-sort idea was faster on paper but it both broke the `i < j` index order and pulled in floating-point angle ties on collinear/zero beacons, so I switched to exact `O(n^2)` integer enumeration that fits the `n <= 2000` limit and respects the order by construction; coordinates and accumulators are `long long` because a single product reaches `10^12` and the difference `2*10^12`, well past 32-bit; the one real bug was initializing the running maximum to `0`, which on an all-clockwise instance (`(0,3),(3,0),(4,-1)`, true answer `-3`) injected a phantom `0` candidate that suppressed every negative sweep — a trace exposed it and seeding from `LLONG_MIN` (a sentinel that never enters arithmetic, so it cannot underflow) fixes it; and the `n < 2` guard prints the literal `NONE` *after* reading the header so that a present `n = 0` is answered rather than left silent.
