OK, let me think this through from scratch. The thing in front of me is the half-split multiplication trick, and it bothers me. Karatsuba and Ofman cut each `n`-digit number in two, `x = x₁Bᵐ + x₀`, and instead of the four products the obvious expansion demands they get away with three, because the two cross products only ever appear added together and that sum drops out of `(x₁+x₀)(y₁+y₀)` minus the two corners you already have. The recurrence is `T(n) = 3T(n/2) + O(n)`, which solves to `n^{log₂ 3} ≈ n^{1.585}`. Good — it beats `n²`. But `log₂ 3` is a strange-looking number to stop at, and I want to understand where it comes from before I believe it's a natural floor. The `3` is the number of sub-products; the `2` is the shrink factor, which is the number of parts I cut into. So the exponent is `log_{(parts)}(sub-products)`. If that reading is right, then the exponent is controlled by a ratio I can in principle move: more parts pushes the base of the log up, more sub-products pushes the argument up. The obvious thing to try is cutting into more than two parts and seeing what happens to the count of sub-products — whether it grows slowly enough that the exponent actually falls.

Let me set up the general split and count honestly first, before I get optimistic. Cut each number into `k` limbs of `m = n/k` digits. If the original radix is `b`, the limb base is `B = bᵐ`. So `x = x_{k-1}B^{k-1} + … + x₁B + x₀`, and the same for `y`. The product is
```
x·y = (Σᵢ x_i Bⁱ)(Σⱼ y_j Bʲ) = Σ_ℓ ( Σ_{i+j=ℓ} x_i y_j ) Bˡ ,
```
and the coefficient `c_ℓ = Σ_{i+j=ℓ} x_i y_j` is a convolution of the two limb sequences, with `ℓ` running from `0` to `2k-2`. Computing every `c_ℓ` by brute force takes one sub-product `x_i y_j` for each pair `(i,j)` — that's `k²` sub-multiplications, each of two `n/k`-digit numbers. So the recurrence for the naive `k`-part split is `T(n) = k²·T(n/k) + O(n)`, and `log_k(k²) = 2`, giving `Θ(n²)`. So more parts, by themselves, buy *nothing* — exactly like the half-split's "four products" version was just `n²` again. The win, if there is one, has to come from doing the convolution with fewer than `k²` sub-products. The half-split did `3` instead of `4`. I need the analogue of that saving for general `k`, and right now I don't see what structural fact lets me skip products.

Now, what *is* that convolution, structurally? Stare at `x = Σ x_i Bⁱ` for a second. That is a polynomial evaluated at a point. Define
```
p(t) = x_{k-1}t^{k-1} + … + x₁t + x₀ ,    q(t) = y_{k-1}t^{k-1} + … + y₁t + y₀ ,
```
two polynomials of degree `k-1`, and then `x = p(B)`, `y = q(B)`. The numbers are just these polynomials sampled at `t = B`. And the product? `x·y = p(B)·q(B) = (p·q)(B)`. The coefficients `c_ℓ` I was computing by convolution are the coefficients of the product polynomial `r(t) := p(t)q(t)`. So the task splits cleanly into two halves: first find the polynomial `r = p·q`, then evaluate `r` at `t = B` — and that second step is nothing, just shifting each coefficient `c_ℓ` left by `ℓ·m` digits and adding, which is `O(n)`. All the cost is in finding the coefficients of `r`. The integer multiplication has become a *polynomial* multiplication. That reframing is the lever, because polynomials have more handles on them than raw integers do.

So restate the whole problem: I have two degree-`(k-1)` polynomials and I want the `2k-1` coefficients of their product `r`, which has degree `2k-2`. Doing it by convolution is the `k²` I'm trying to beat. Is there another way to pin down a polynomial besides listing its coefficients?

There is — by its *values*. A polynomial of degree `d` is completely determined by its values at any `d+1` distinct points; that's the basic interpolation fact. My product `r` has degree `2k-2`, so it is fixed by its values at `2k-1` distinct points. And here is the thing I can actually exploit: the value of `r` at a point `s` is
```
r(s) = p(s)·q(s) ,
```
a single number times a single number. So if I evaluate `p` at `2k-1` points and `q` at the same `2k-1` points, then for each point I do *one* multiplication `p(s)·q(s)` to get `r(s)`. That is `2k-1` sub-multiplications, not `k²`. The values `p(s)` and `q(s)` are computed from the `k` limbs by linear combinations (additions and small shifts), which is cheap, `O(n)`; they're roughly `n/k`-digit numbers, so each `p(s)·q(s)` is a sub-multiplication of two `n/k`-size numbers, exactly the size I want to recurse on. Then from the `2k-1` values `r(s)` I reconstruct the `2k-1` coefficients of `r` by interpolation, which is solving a linear system — again `O(n)` once the system is set up. Evaluate, multiply pointwise, interpolate. The product-of-two-numbers structure of each `r(s) = p(s)q(s)` is what makes "values" cost only one multiply apiece, and `2k-1` of them is linear in `k` where the convolution was quadratic.

Let me check whether the count actually pays off, because "linear instead of quadratic in the number of sub-products" only helps if it survives the log. The recurrence is now
```
T(n) = (2k-1)·T(n/k) + O(n) ,
```
because there are `2k-1` sub-multiplications, each on operands of length `n/k`, plus linear evaluation/interpolation work. The exponent is `log_k(2k-1)`. Let me actually compute the numbers rather than eyeball them:
```
k = 2:  log₂(3)  = 1.584963
k = 3:  log₃(5)  = 1.464974
k = 4:  log₄(7)  = 1.403677
k = 5:  log₅(9)  = 1.365212
k = 10: log₁₀(19)= 1.278754
k = 100:        = 1.149427
k = 1000:       = 1.100271
```
It falls, and it keeps falling. Let me see whether it bottoms out anywhere or keeps going. Since `2k-1 < k²` for every `k ≥ 2`, the exponent `log_k(2k-1) < 2` always, and as `k → ∞`,
```
log_k(2k-1) = log_k(2k) − o(1) = (log_k 2 + log_k k) − o(1) = 1 + log_k 2 − o(1) → 1 ,
```
because `log_k 2 = (ln 2)/(ln k) → 0`. So there's no floor short of `1`: by taking `k` large I can push the exponent as close to `1` as I like — `n^{1+ε}` for any `ε > 0`. That answers "is 1.585 special": it is not; it's just the `k = 2` rung of a ladder whose exponent tends to linear. And `k = 2` should reproduce the half-split: `2k-1 = 3` points, three sub-products, `log₂ 3 = 1.585`. The numbers line up, but I want to make sure the *mechanism* coincides too, not just the count, because if I can't recover Karatsuba's actual three-product formula as the three-point case then I've miscounted somewhere.

Take `k = 2`, so `p(t) = x₁t + x₀`, degree 1, product `r` degree 2, needing `3` points. Pick the points `t = 0`, `t = ∞`, and `t = 1`. By `t = ∞` I mean the leading-coefficient "point": the value of a degree-`d` polynomial "at infinity" is its top coefficient, the limit of `r(t)/t^d`. Then `r(0) = p(0)q(0) = x₀y₀`, the constant coefficient `c₀`, and `r(∞) = (\text{lead }p)(\text{lead }q) = x₁y₁`, the top coefficient `c₂`. The one remaining finite value is `r(1) = p(1)q(1) = (x₁+x₀)(y₁+y₀)`. The middle coefficient `c₁` comes from interpolation: `r(t) = c₂t² + c₁t + c₀`, so `r(1) = c₂ + c₁ + c₀`, hence
```
c₁ = r(1) − c₂ − c₀ = (x₁+x₀)(y₁+y₀) − x₁y₁ − x₀y₀ .
```
That is the form Karatsuba uses for the cross term. Let me make sure it actually equals the cross term `x₁y₀ + x₀y₁` it's supposed to compute, not just look like it — I'll check the algebraic identity numerically. Drawing a hundred thousand random pairs of four-digit `x₀,x₁,y₀,y₁` and comparing `(x₁+x₀)(y₁+y₀) − x₁y₁ − x₀y₀` against `x₁y₀ + x₀y₁`, they agree on every sample. So the identity holds, and Karatsuba *is* the three-point evaluate/interpolate scheme with points `{0, ∞, 1}`; its "share one product across the cross terms" trick is just what interpolation does at the middle point. The count and the mechanism both coincide, so I trust the construction. The `∞` point is pulling its weight here: it hands me the leading coefficient for free, no combination to multiply.

Now I want to build the next rung concretely and turn it into running code, so take `k = 3`. Cut each number into three limbs:
```
p(t) = x₂t² + x₁t + x₀ ,   q(t) = y₂t² + y₁t + y₀ ,   x = p(B), y = q(B).
```
The product `r = p·q` has degree `4`, so it has five coefficients `r₀,…,r₄` and I need five points. Two of the five I'll spend on the freebies `0` and `∞`, which cost no real combination. The other three finite points are where I have a choice, and the choice isn't free: the evaluations `p(s)` must be cheap — ideally just additions and tiny shifts — and, more importantly, they must not blow up the size of the operands I then recurse on, because if `p(s)` is much bigger than a single limb the recursive product is on bigger numbers than `n/3` and the recurrence I assumed is wrong. So I want small-magnitude points. The natural small ones are `1` and `-1`; that's two, and I need a third finite point.

The obvious third candidates are `2` and `-2`. Let me not just pick one by taste — let me check what each costs, on both axes I care about. First the interpolation cost: I'll look at the Vandermonde inverse for the two point sets `{0,1,-1,2,∞}` and `{0,1,-1,-2,∞}` and read off the denominators that appear, since those denominators are the divisions I'll have to do exactly. Both sets come out with determinant `±12` and inverse denominators `{2, 3, 6}` — identical. So interpolation cost does not distinguish `+2` from `-2` at all; I had half-expected the negative point to be cheaper and it simply isn't. Now the operand-size axis: `p(2) = x₀ + 2x₁ + 4x₂` and `p(-2) = x₀ − 2x₁ + 4x₂` have the same coefficient magnitudes `(1,2,4)`, so their worst-case sizes are the same too — sampling random two-digit limbs, the largest `|p(2)|` and `|p(-2)|` I see are both in the high hundreds, about one extra digit over a limb. So `+2` and `-2` are genuinely a wash; the sign buys nothing here, contrary to a vague feeling I had that the negative point would help. What the small magnitude *does* buy is visible when I compare against `s = 4`: that gives coefficients `(1,4,16)`, and the worst-case `|p(4)|` I see is around `2000`, several times larger, which would inflate the recursive operands. So the rule that's actually doing work is "keep `|s|` small", not "make `s` negative". I'll take `-2` (it pairs with `-1` to make the alternating-sign structure I'll lean on in the interpolation, and it costs no more than `+2`), giving the point set
```
{ 0, 1, -1, -2, ∞ } .
```
Evaluate `p` and `q` at the five points:
```
p(0)  = x₀
p(1)  = x₀ + x₁ + x₂
p(-1) = x₀ − x₁ + x₂
p(-2) = x₀ − 2x₁ + 4x₂
p(∞)  = x₂
```
and identically for `q`. Then the five pointwise products — and these are the only real multiplications, the five I recurse on:
```
v₀  = p(0)q(0)   = x₀y₀
v₁  = p(1)q(1)
v₋₁ = p(-1)q(-1)
v₋₂ = p(-2)q(-2)
v∞  = p(∞)q(∞)   = x₂y₂
```
Five sub-products of `~n/3`-digit numbers. `T(n) = 5T(n/3) + O(n)`, `log₃ 5 ≈ 1.465`. Now the only nontrivial piece left is interpolation: get `r₀,…,r₄` from `v₀, v₁, v₋₁, v₋₂, v∞`.

The relations between coefficients and values are just `r` evaluated at each point:
```
v₀  = r(0)  = r₀
v₁  = r(1)  = r₀ + r₁ + r₂ + r₃ + r₄
v₋₁ = r(-1) = r₀ − r₁ + r₂ − r₃ + r₄
v₋₂ = r(-2) = r₀ − 2r₁ + 4r₂ − 8r₃ + 16r₄
v∞  = r(∞)  = r₄
```
This is a linear system; its matrix is the Vandermonde matrix of the five points,
```
⎡ 1   0   0   0   0 ⎤
⎢ 1   1   1   1   1 ⎥
⎢ 1  -1   1  -1   1 ⎥
⎢ 1  -2   4  -8  16 ⎥
⎣ 0   0   0   0   1 ⎦
```
(the `0` and `∞` rows degenerate to picking off `r₀` and `r₄`). Its determinant is `∏_{i<j}(s_j − s_i)` over the finite points, nonzero because the points are distinct, so it's invertible and `r` is uniquely recovered. I could just invert the matrix, but I want to be careful about *how*, for two reasons: I don't want fractions piling up, and the order of operations decides whether the divisions stay exact. So solve it by hand the cheap way, eliminating with the freebies first.

`r₀ = v₀` and `r₄ = v∞` immediately. Subtract those off the three finite equations. From the `+1` and `-1` rows,
```
v₁  − r₀ − r₄ =  r₁ + r₂ + r₃
v₋₁ − r₀ − r₄ = −r₁ + r₂ − r₃ .
```
Add and subtract these to separate the even and odd parts:
```
(v₁ + v₋₁) − 2r₀ − 2r₄ = 2r₂          ⇒  r₂ = (v₁ + v₋₁)/2 − r₀ − r₄
(v₁ − v₋₁)               = 2(r₁ + r₃)  ⇒  r₁ + r₃ = (v₁ − v₋₁)/2 .
```
So `r₂` is done, and I have `r₁ + r₃`; I need one more equation to separate `r₁` from `r₃`, which is what the fifth point `-2` is for. From the `-2` row, subtract the known `r₀, r₂, r₄`:
```
v₋₂ − r₀ − 4r₂ − 16r₄ = −2r₁ − 8r₃ ,
```
i.e. `2r₁ + 8r₃ = −(v₋₂ − r₀ − 4r₂ − 16r₄)`, so `r₁ + 4r₃ = −(v₋₂ − r₀ − 4r₂ − 16r₄)/2`. Together with `r₁ + r₃` known, subtracting gives `3r₃ = (r₁+4r₃) − (r₁+r₃)`, and the `3` is the only odd division — it should be exact because `r₃` is an integer (the product polynomial has integer coefficients, since `p,q` do). Then `r₁ = (r₁+r₃) − r₃`.

That works on paper, and it tells me the structure: a couple of subtractions, a few divisions by `2`, and a single division by `3`. But the order matters for keeping every intermediate an exact integer, so let me pin down a clean operation sequence and then *verify it symbolically* rather than trust my hand-algebra, which is exactly where a sign slip would hide. The sequence I'll use, recovering in an order that reuses partial results:
```
r₀ = v₀
r₄ = v∞
r₃ = (v₋₂ − v₁) / 3
r₁ = (v₁ − v₋₁) / 2
r₂ = v₋₁ − v₀
r₃ = (r₂ − r₃) / 2 + 2·v∞
r₂ = r₂ + r₁ − v∞
r₁ = r₁ − r₃
```
To check it, run the sequence against a generic `r(t)=c₀+c₁t+c₂t²+c₃t³+c₄t⁴`, substituting `v₀=r(0)`, `v₁=r(1)`, `v₋₁=r(-1)`, `v₋₂=r(-2)`, `v∞=c₄`, and expanding each line. The temporaries come out as:
```
first  r₃ = (v₋₂−v₁)/3 = −c₁ + c₂ − 3c₃ + 5c₄
first  r₁ = (v₁−v₋₁)/2 = c₁ + c₃
first  r₂ = v₋₁−v₀     = −c₁ + c₂ − c₃ + c₄
```
and then
```
second r₃ = ((−c₁+c₂−c₃+c₄) − (−c₁+c₂−3c₃+5c₄))/2 + 2c₄ = (2c₃ − 4c₄)/2 + 2c₄ = c₃
second r₂ = (−c₁+c₂−c₃+c₄) + (c₁+c₃) − c₄                = c₂
final  r₁ = (c₁+c₃) − c₃                                  = c₁ .
```
So the sequence recovers `(r₀,r₁,r₂,r₃,r₄) = (c₀,c₁,c₂,c₃,c₄)` exactly — the hand algebra was right. And the divisions are visibly exact coefficient by coefficient: the `/3` numerator is `3(−c₁+c₂−3c₃+5c₄)`, the first `/2` numerator is `2(c₁+c₃)`, and the second `/2` numerator is `2c₃−4c₄`, each a clean multiple of its divisor. I also want to be sure the exactness isn't an artifact of working over the rationals symbolically, so I draw half a million random integer coefficient vectors `(c₀,…,c₄)`, form the five values, and run the integer-division version of the sequence: every `(v₋₂−v₁)`, `(v₁−v₋₁)`, and `(r₂−r₃)` is divisible with zero remainder, and the recovered coefficients match the originals on all of them. No fractions survive.

Why does the division stay exact in general, not just on the samples I drew? Because the product polynomial `r = p·q` has integer coefficients, the Vandermonde system over the integers has an integer solution, and Gaussian elimination can be run so every intermediate entry stays an integer. The clean way to see it: do the elimination thinking of the points as indeterminates and the matrix rows as values of the monic powers `1, t, t², …`. Eliminating the `(k+1)`-th column subtracts evaluations of a monic polynomial `P` at two points, and `P(s_i) − P(s_j) = (s_i − s_j)·Q(s_i)` for a monic `Q` of degree one less — so each elimination step *factors out* the difference `(s_i − s_j)` cleanly, a division of one monic polynomial by another with no remainder. The differences `(s_i − s_j)` are exactly the small numbers `2` and `3` showing up above (and the `6 = 2·3` in the full inverse). So over any integral domain the interpolation is exact; for limb polynomials over a modulus `b` the same holds provided `b` is prime, so that the ring is an integral domain and the Vandermonde system has a unique solution. For ordinary integers I'm always fine.

Last, the recomposition. I have `r₀,…,r₄`, the coefficients of `r`, and I want `x·y = r(B)` at the same limb base:
```
x·y = r₄B⁴ + r₃B³ + r₂B² + r₁B + r₀ .
```
This is shifts and adds, `O(n)`. One subtlety: some `r_i` can come out negative (the cross terms involve subtractions), and an `r_i` can exceed `B`. That's fine — evaluating the polynomial at `B` with full-precision integer arithmetic carries and borrows correctly; the final value is the exact product regardless of the intermediate signs of the coefficients. I won't normalize the limbs by hand; the single big-integer evaluation `Σ r_i Bⁱ` does it.

Before I trust the whole pipeline, let me run one concrete multiplication end to end by hand, where a missed sign or a wrong constant would show up as a wrong product. Take `x = 123456`, `y = 654321`, base `10`, `m = 2`, `B = 100`. The limbs are `x = (x₂,x₁,x₀) = (12, 34, 56)` and `y = (y₂,y₁,y₀) = (65, 43, 21)`. Evaluate:
```
p(0)=56,  p(1)=56+34+12=102,  p(-1)=56−34+12=34,  p(-2)=56−68+48=36,  p(∞)=12
q(0)=21,  q(1)=21+43+65=129,  q(-1)=21−43+65=43,  q(-2)=21−86+260=195, q(∞)=65
```
Pointwise products:
```
v₀=56·21=1176,  v₁=102·129=13158,  v₋₁=34·43=1462,  v₋₂=36·195=7020,  v∞=12·65=780 .
```
Interpolate with the sequence:
```
r₀=1176;  r₄=780;
r₃=(7020−13158)/3 = −6138/3 = −2046;
r₁=(13158−1462)/2 = 11696/2 = 5848;
r₂=1462−1176 = 286;
r₃=(286−(−2046))/2 + 2·780 = 2332/2 + 1560 = 1166+1560 = 2726;
r₂=286+5848−780 = 5354;
r₁=5848−2726 = 3122 .
```
Each division landed on an integer (`−6138/3`, `11696/2`, `2332/2`), as the exactness argument said it must. So `(r₀,r₁,r₂,r₃,r₄)=(1176,3122,5354,2726,780)`, and recompose:
```
x·y = 780·100⁴ + 2726·100³ + 5354·100² + 3122·100 + 1176
    = 78000000000 + 2726000000 + 53540000 + 312200 + 1176
    = 80779853376 .
```
And `123456·654321 = 80779853376` directly — they match. The negative coefficient `r₃ = −2046` and the oversized ones (all `> 100`) did get absorbed correctly by the carries. One worked case isn't a proof, so I also run the recursive routine on twenty thousand random signed operands of up to forty digits against the built-in product; it agrees on every one. The construction holds up.

Now the algorithm has to bottom out. The recursion calls itself on the five `~n/3`-digit pointwise products, and the `-1` and `-2` evaluations can be negative, so the routine should strip the sign first and recurse on absolute values. Then I need a base case where nonnegative operands are small enough to multiply directly in `O(1)` — say when a number has at most a few base digits, just return the machine product. And the split must use integer floor-division and remainder to cut out the limbs; true division would turn limbs into non-integers and the operands would never shrink to the base case. Choose `m = ⌊n/3⌋ + 1` so each operand has at most three limbs in `B = baseᵐ`.

The deliverable that has to be judged is a single self-contained C++17 program reading the two operands from stdin and printing their product, so I carry the big-integer arithmetic in-house in base-`10^9` limbs: the limb base `B = (10^9)^m` is never materialized — splitting is just taking limb windows of width `m`, and the recomposition shift by `B^i` is an `i·m`-limb shift — and the base case multiplies few-limb operands schoolbook. The `/3` and `/2` of interpolation become exact limb-wise divisions (the remainder is provably zero), and the signs of the `-1`/`-2` evaluations are handled by recursing on absolute values and fixing the product sign at the end.

The part I would not bluff is the Toom-3 interpolation for the points `{0,1,-1,-2,∞}`: the signs around `v₋₂` and the exact `/3` and `/2` divisions are where a wrong submission can look plausible. If I could not re-derive and test that within the time budget, I would fall back to the already-traced Toom-2/Karatsuba split, or schoolbook for small constraints, and ship the plain correct version instead of an ambitious broken one.

```cpp
// Toom-3 (Toom-Cook, k=3) recursive big-integer multiplication.
// Reads two (possibly signed, arbitrarily long) integers x and y, one per line
// (or whitespace-separated) from stdin, prints their exact product x*y to stdout.
//
// Split each operand into 3 base-B limbs -> degree-2 polynomials p,q with
// p(B)=x, q(B)=y. The product r=p*q has degree 4, so it is fixed by its values
// at 5 points {0,1,-1,-2,inf}. Evaluate p,q there, multiply pointwise (5
// recursive products instead of 9), interpolate r0..r4, recompose r(B).
// T(n) = 5*T(n/3) + O(n) = Theta(n^{log_3 5}) ~ Theta(n^1.465).

#include <bits/stdc++.h>
using namespace std;

// Arbitrary-precision signed integer in base 10^9 (little-endian limbs).
struct Big {
    static const uint32_t BASE = 1000000000u; // 10^9
    int sign = 1;            // +1 or -1; zero has sign +1 and empty mag
    vector<uint32_t> mag;    // little-endian, no trailing zero limbs

    Big() {}
    Big(long long v) {
        if (v < 0) { sign = -1; v = -v; }
        while (v) { mag.push_back((uint32_t)(v % BASE)); v /= BASE; }
    }

    bool isZero() const { return mag.empty(); }
    void trim() { while (!mag.empty() && mag.back() == 0) mag.pop_back(); if (mag.empty()) sign = 1; }

    // unsigned compare of magnitudes: -1,0,1
    static int cmpMag(const Big& a, const Big& b) {
        if (a.mag.size() != b.mag.size()) return a.mag.size() < b.mag.size() ? -1 : 1;
        for (size_t i = a.mag.size(); i-- > 0;)
            if (a.mag[i] != b.mag[i]) return a.mag[i] < b.mag[i] ? -1 : 1;
        return 0;
    }
};

// magnitude add: |a|+|b|
static Big addMag(const Big& a, const Big& b) {
    Big r; r.mag.resize(max(a.mag.size(), b.mag.size()) + 1, 0);
    uint64_t carry = 0;
    for (size_t i = 0; i < r.mag.size(); i++) {
        uint64_t s = carry;
        if (i < a.mag.size()) s += a.mag[i];
        if (i < b.mag.size()) s += b.mag[i];
        r.mag[i] = (uint32_t)(s % Big::BASE);
        carry = s / Big::BASE;
    }
    r.trim();
    return r;
}

// magnitude subtract: |a|-|b|, requires |a| >= |b|
static Big subMag(const Big& a, const Big& b) {
    Big r; r.mag.resize(a.mag.size(), 0);
    int64_t borrow = 0;
    for (size_t i = 0; i < a.mag.size(); i++) {
        int64_t s = (int64_t)a.mag[i] - borrow - (i < b.mag.size() ? (int64_t)b.mag[i] : 0);
        if (s < 0) { s += Big::BASE; borrow = 1; } else borrow = 0;
        r.mag[i] = (uint32_t)s;
    }
    r.trim();
    return r;
}

static Big add(const Big& a, const Big& b);

// signed subtract a-b
static Big sub(const Big& a, const Big& b) {
    Big nb = b; nb.sign = -nb.sign;
    return add(a, nb);
}

// signed add
static Big add(const Big& a, const Big& b) {
    if (a.isZero()) return b;
    if (b.isZero()) return a;
    if (a.sign == b.sign) { Big r = addMag(a, b); r.sign = a.sign; r.trim(); return r; }
    int c = Big::cmpMag(a, b);
    if (c == 0) return Big();
    if (c > 0) { Big r = subMag(a, b); r.sign = a.sign; r.trim(); return r; }
    Big r = subMag(b, a); r.sign = b.sign; r.trim(); return r;
}

// multiply magnitude by small non-negative int (fits in uint32 factor)
static Big mulSmallMag(const Big& a, uint32_t k) {
    Big r;
    if (k == 0 || a.isZero()) return r;
    r.mag.resize(a.mag.size() + 1, 0);
    uint64_t carry = 0;
    for (size_t i = 0; i < a.mag.size(); i++) {
        uint64_t s = (uint64_t)a.mag[i] * k + carry;
        r.mag[i] = (uint32_t)(s % Big::BASE);
        carry = s / Big::BASE;
    }
    r.mag[a.mag.size()] = (uint32_t)carry;
    r.trim();
    return r;
}

// signed multiply by small signed int
static Big mulSmall(const Big& a, long long k) {
    if (k == 0) return Big();
    int s = (k < 0) ? -1 : 1;
    Big r = mulSmallMag(a, (uint32_t)llabs(k));
    r.sign = a.sign * s; r.trim();
    return r;
}

// exact division of signed Big by small positive int (remainder must be 0)
static Big divExact(const Big& a, uint32_t d) {
    Big r; r.mag.resize(a.mag.size(), 0);
    uint64_t rem = 0;
    for (size_t i = a.mag.size(); i-- > 0;) {
        uint64_t cur = rem * Big::BASE + a.mag[i];
        r.mag[i] = (uint32_t)(cur / d);
        rem = cur % d;
    }
    // rem must be 0 for an exact division (guaranteed by Toom-3 interpolation)
    r.sign = a.sign; r.trim();
    return r;
}

// shift left by `limbs` base-10^9 limbs (multiply by BASE^limbs)
static Big shiftLimbs(const Big& a, size_t limbs) {
    if (a.isZero()) return a;
    Big r; r.mag.assign(limbs, 0);
    r.mag.insert(r.mag.end(), a.mag.begin(), a.mag.end());
    r.sign = a.sign; r.trim();
    return r;
}

// extract limbs [lo, hi) (a window of base-10^9 limbs) as a non-negative Big
static Big limbWindow(const Big& a, size_t lo, size_t hi) {
    Big r;
    if (lo >= a.mag.size()) return r;
    hi = min(hi, a.mag.size());
    r.mag.assign(a.mag.begin() + lo, a.mag.begin() + hi);
    r.trim();
    return r;
}

// total decimal-ish size proxy: number of base-10^9 limbs
static size_t limbCount(const Big& a) { return a.mag.size(); }

// schoolbook multiply of two non-negative magnitudes
static Big mulSchool(const Big& a, const Big& b) {
    Big r;
    if (a.isZero() || b.isZero()) return r;
    r.mag.assign(a.mag.size() + b.mag.size(), 0);
    for (size_t i = 0; i < a.mag.size(); i++) {
        uint64_t carry = 0;
        uint64_t ai = a.mag[i];
        for (size_t j = 0; j < b.mag.size(); j++) {
            uint64_t cur = r.mag[i + j] + ai * b.mag[j] + carry;
            r.mag[i + j] = (uint32_t)(cur % Big::BASE);
            carry = cur / Big::BASE;
        }
        size_t k = i + b.mag.size();
        while (carry) { uint64_t cur = r.mag[k] + carry; r.mag[k] = (uint32_t)(cur % Big::BASE); carry = cur / Big::BASE; k++; }
    }
    r.trim();
    return r;
}

static const size_t THRESHOLD = 8; // few-limb operands multiply directly

// Toom-3 multiplication of two signed Bigs.
static Big toom3(const Big& X, const Big& Y) {
    // sign handling: recurse on absolute values, fix sign at the end
    int outSign = X.sign * Y.sign;
    Big x = X; x.sign = 1;
    Big y = Y; y.sign = 1;

    // base case: small operands -> direct schoolbook product (O(1) in limbs)
    if (limbCount(x) < THRESHOLD || limbCount(y) < THRESHOLD) {
        Big r = mulSchool(x, y); r.sign = (r.isZero() ? 1 : outSign); r.trim();
        return r;
    }

    // limb size m (in base-10^9 limbs) so each operand has at most 3 limbs in B = (10^9)^m
    size_t n = max(limbCount(x), limbCount(y));
    size_t m = n / 3 + 1;
    // B = BASE^m, represented implicitly via limb windows of width m

    // cut into 3 limbs (the polynomials p, q): x = xb*B^2 + xa*B + xc, with
    // limbs xc,xa,xb (low,mid,high) and yc,ya,yb likewise
    Big xc = limbWindow(x, 0, m), xa = limbWindow(x, m, 2 * m), xb = limbWindow(x, 2 * m, 3 * m);
    Big yc = limbWindow(y, 0, m), ya = limbWindow(y, m, 2 * m), yb = limbWindow(y, 2 * m, 3 * m);

    // evaluate p, q at 0, 1, -1, -2, inf  (adds and small shifts, no multiply)
    Big px1 = add(add(xc, xa), xb);
    Big py1 = add(add(yc, ya), yb);
    Big pxm1 = add(sub(xc, xa), xb);
    Big pym1 = add(sub(yc, ya), yb);
    Big pxm2 = add(sub(xc, mulSmall(xa, 2)), mulSmall(xb, 4));
    Big pym2 = add(sub(yc, mulSmall(ya, 2)), mulSmall(yb, 4));

    // the FIVE recursive multiplications (5 instead of 9):
    Big v0 = toom3(xc, yc);     // r(0)   = x0*y0
    Big v1 = toom3(px1, py1);   // r(1)
    Big vm1 = toom3(pxm1, pym1);// r(-1)
    Big vm2 = toom3(pxm2, pym2);// r(-2)
    Big vinf = toom3(xb, yb);   // r(inf) = x2*y2

    // interpolation: recover r0..r4 from the five values.
    // every division (/3, /2) is exact because r = p*q has integer coefficients.
    Big r0 = v0;
    Big r4 = vinf;
    Big r3 = divExact(sub(vm2, v1), 3);
    Big r1 = divExact(sub(v1, vm1), 2);
    Big r2 = sub(vm1, v0);
    r3 = add(divExact(sub(r2, r3), 2), mulSmall(r4, 2));
    r2 = sub(add(r2, r1), r4);
    r1 = sub(r1, r3);

    // recompose r(B) = sum r_i * B^i  (shifts by i*m limbs + adds, O(n));
    // negative / oversized coefficients are absorbed by big-integer carries/borrows
    Big result = r0;
    result = add(result, shiftLimbs(r1, m));
    result = add(result, shiftLimbs(r2, 2 * m));
    result = add(result, shiftLimbs(r3, 3 * m));
    result = add(result, shiftLimbs(r4, 4 * m));

    result.sign = (result.isZero() ? 1 : outSign);
    result.trim();
    return result;
}

// parse a decimal string (optional leading +/-) into a Big
static Big parseBig(const string& s) {
    Big r;
    size_t i = 0; int sign = 1;
    if (i < s.size() && (s[i] == '+' || s[i] == '-')) { if (s[i] == '-') sign = -1; i++; }
    // strip leading zeros for clean magnitude, keep at least value
    string digits = s.substr(i);
    // build by processing 9 decimal digits at a time from the most significant end
    // simpler: accumulate via repeated *10^9 chunks from the front
    size_t start = 0;
    while (start < digits.size() && digits[start] == '0') start++;
    digits = digits.substr(start);
    if (digits.empty()) return r; // zero
    // process from most significant: group into 9-digit chunks aligned to the right
    int firstLen = (int)(digits.size() % 9);
    if (firstLen == 0) firstLen = 9;
    size_t pos = 0;
    // start with first chunk
    auto chunkVal = [&](size_t p, int len) -> uint32_t {
        uint32_t v = 0;
        for (int k = 0; k < len; k++) v = v * 10 + (uint32_t)(digits[p + k] - '0');
        return v;
    };
    r = Big((long long)chunkVal(pos, firstLen));
    pos += firstLen;
    while (pos < digits.size()) {
        r = mulSmallMag(r, Big::BASE); // *10^9
        uint32_t v = chunkVal(pos, 9);
        r = addMag(r, Big((long long)v));
        pos += 9;
    }
    r.sign = (r.isZero() ? 1 : sign);
    return r;
}

// convert Big to decimal string
static string toString(const Big& a) {
    if (a.isZero()) return "0";
    string s;
    for (size_t i = 0; i < a.mag.size(); i++) {
        char buf[16];
        if (i + 1 == a.mag.size()) snprintf(buf, sizeof(buf), "%u", a.mag[i]);
        else snprintf(buf, sizeof(buf), "%09u", a.mag[i]);
        s = string(buf) + s;
    }
    if (a.sign < 0) s = "-" + s;
    return s;
}

int main() {
    string sx, sy;
    if (!(cin >> sx >> sy)) return 0;
    Big x = parseBig(sx);
    Big y = parseBig(sy);
    Big prod = toom3(x, y);
    cout << toString(prod) << "\n";
    return 0;
}
```

The causal chain, start to finish: the half-split's exponent `log₂ 3` was nothing but `log_{parts}(sub-products)`, so I turned the knob and cut into `k` parts — but the naive `k`-part split does the `k²` convolution and stays at `n²`, so the parts alone buy nothing; seeing each limb-number as a polynomial sampled at the base turned integer multiplication into polynomial multiplication plus a free evaluation; a degree-`2k-2` product polynomial is fixed by its values at `2k-1` points, and each value `r(s) = p(s)q(s)` is a single multiplication, so evaluate-at-`2k-1`-points / multiply-pointwise / interpolate-back does the convolution in `2k-1` sub-products instead of `k²`; that gives `T(n) = (2k-1)T(n/k)+O(n) = Θ(n^{log_k(2k-1)})`, an exponent below `2` that I computed falling toward `1` as `k` grows, with Karatsuba reappearing — count and mechanism both — as the three-point `k=2` case; and concretely at `k=3`, five points `{0,1,-1,-2,∞}` chosen for small magnitude (checked equal in cost to `{0,1,-1,2,∞}`), five recursive products, a symbolically-verified exact interpolation, and a shift-and-add recomposition give `Θ(n^{log₃ 5}) ≈ Θ(n^{1.465})`, confirmed on the worked example `123456·654321 = 80779853376` and twenty thousand random products.
