OK, let me think this through from scratch. The thing in front of me is the half-split multiplication trick, and it bothers me. Karatsuba and Ofman cut each `n`-digit number in two, `x = x₁Bᵐ + x₀`, and instead of the four products the obvious expansion demands they get away with three, because the two cross products only ever appear added together and that sum drops out of `(x₁+x₀)(y₁+y₀)` minus the two corners you already have. The recurrence is `T(n) = 3T(n/2) + O(n)`, which solves to `n^{log₂ 3} ≈ n^{1.585}`. Good — it beats `n²`. But `log₂ 3` is a strange-looking number to stop at. Where did it come from? It came from `3` sub-products with a shrink factor of `2`: the exponent is `log₂(\text{number of sub-products})`. The `2` in the base is the number of parts I cut into. So the whole result is `log_{(\text{parts})}(\text{sub-products})`. That immediately makes me want to turn the knob: cut into more than two parts and see whether the exponent falls.

Let me set up the general split and count honestly first, before I get optimistic. Cut each number into `k` limbs of `m = n/k` digits. If the original radix is `b`, the limb base is `B = bᵐ`. So `x = x_{k-1}B^{k-1} + … + x₁B + x₀`, and the same for `y`. The product is
```
x·y = (Σᵢ x_i Bⁱ)(Σⱼ y_j Bʲ) = Σ_ℓ ( Σ_{i+j=ℓ} x_i y_j ) Bˡ ,
```
and the coefficient `c_ℓ = Σ_{i+j=ℓ} x_i y_j` is a convolution of the two limb sequences, with `ℓ` running from `0` to `2k-2`. Computing every `c_ℓ` by brute force takes one sub-product `x_i y_j` for each pair `(i,j)` — that's `k²` sub-multiplications, each of two `n/k`-digit numbers. So the recurrence for the naive `k`-part split is `T(n) = k²·T(n/k) + O(n)`, and `log_k(k²) = 2`, giving `Θ(n²)`. So more parts, by themselves, buy *nothing* — exactly like the half-split's "four products" version was just `n²` again. The win has to come from doing the convolution with fewer than `k²` sub-products. The half-split did `3` instead of `4`. I need the analogue of that saving for general `k`.

Now, what *is* that convolution, structurally? Stare at `x = Σ x_i Bⁱ` for a second. That is a polynomial evaluated at a point. Define
```
p(t) = x_{k-1}t^{k-1} + … + x₁t + x₀ ,    q(t) = y_{k-1}t^{k-1} + … + y₁t + y₀ ,
```
two polynomials of degree `k-1`, and then `x = p(B)`, `y = q(B)`. The numbers are just these polynomials sampled at `t = B`. And the product? `x·y = p(B)·q(B) = (p·q)(B)`. The coefficients `c_ℓ` I was computing by convolution are *exactly* the coefficients of the product polynomial `r(t) := p(t)q(t)`. So the real task has split cleanly into two halves: first find the polynomial `r = p·q`, then evaluate `r` at `t = B` — and that second step is nothing, just shifting each coefficient `c_ℓ` left by `ℓ·m` digits and adding, which is `O(n)`. All the cost is in finding the coefficients of `r`. The integer multiplication has become a *polynomial* multiplication.

So restate the whole problem: I have two degree-`(k-1)` polynomials and I want the `2k-1` coefficients of their product `r`, which has degree `2k-2`. Doing it by convolution is the `k²` I'm trying to beat. Is there another way to pin down a polynomial besides listing its coefficients?

Yes — by its *values*. A polynomial of degree `d` is completely determined by its values at any `d+1` distinct points; that's the basic interpolation fact. My product `r` has degree `2k-2`, so it is fixed by its values at `2k-1` distinct points. And here is the thing I can actually exploit: the value of `r` at a point `s` is
```
r(s) = p(s)·q(s) ,
```
a single number times a single number. So if I evaluate `p` at `2k-1` points and `q` at the same `2k-1` points, then for each point I do *one* multiplication `p(s)·q(s)` to get `r(s)`. That is `2k-1` sub-multiplications — not `k²`. The values `p(s)` and `q(s)` are computed from the `k` limbs by linear combinations (additions and small shifts), which is cheap, `O(n)`; they're roughly `n/k`-digit numbers, so each `p(s)·q(s)` is a sub-multiplication of two `n/k`-size numbers, exactly the size I want to recurse on. Then from the `2k-1` values `r(s)` I reconstruct the `2k-1` coefficients of `r` by interpolation, which is solving a linear system — again `O(n)` once the system is set up. Evaluate, multiply pointwise, interpolate. The product-of-two-numbers structure of each `r(s) = p(s)q(s)` is what makes "values" cost only one multiply apiece, and `2k-1` of them is linear in `k` where the convolution was quadratic.

Let me check the count actually pays off. The recurrence is now
```
T(n) = (2k-1)·T(n/k) + O(n) ,
```
because there are `2k-1` sub-multiplications, each on operands of length `n/k`, plus linear evaluation/interpolation work. The exponent is `log_k(2k-1)`. Let me feel out the numbers:
```
k = 2:  log₂(3)  = 1.585
k = 3:  log₃(5)  = 1.465
k = 4:  log₄(7)  = 1.404
k = 5:  log₅(9)  = 1.365
```
It falls. And it keeps falling: `2k-1 < k²` for every `k ≥ 2`, so `log_k(2k-1) < 2` always, and as `k → ∞`,
```
log_k(2k-1) = log_k(2k) − o(1) = (log_k 2 + log_k k) − o(1) = 1 + log_k 2 − o(1) → 1 .
```
So by taking `k` large I can push the exponent as close to `1` as I like — `n^{1+ε}` for any `ε > 0`. That's the answer to "is 1.585 special": it is not; it is just the `k = 2` rung of a ladder whose exponent tends to linear. And the half-split is exactly this scheme at `k = 2`: `2k-1 = 3` points, three sub-products. Let me make sure that identification is real and not a coincidence, because if Karatsuba *is* the three-point case then I trust the whole construction.

Take `k = 2`, so `p(t) = x₁t + x₀`, degree 1, product `r` degree 2, needing `3` points. Pick the points `t = 0`, `t = ∞`, and `t = 1`. By `t = ∞` I mean the leading-coefficient "point": the value of a degree-`d` polynomial "at infinity" is its top coefficient, the limit of `r(t)/t^d`. Then `r(0) = p(0)q(0) = x₀y₀`, the constant coefficient `c₀`, and `r(∞) = (\text{lead }p)(\text{lead }q) = x₁y₁`, the top coefficient `c₂`. The one remaining finite value is `r(1) = p(1)q(1) = (x₁+x₀)(y₁+y₀)`.

The middle coefficient `c₁` I get by interpolation: `r(t) = c₂t² + c₁t + c₀`, so `r(1) = c₂ + c₁ + c₀`, hence `c₁ = r(1) − c₂ − c₀ = (x₁+x₀)(y₁+y₀) − x₁y₁ − x₀y₀`. That is *exactly* Karatsuba's middle term, the product-of-sums minus the two corners. So Karatsuba is the three-point evaluate/interpolate scheme with points `{0, ∞, 1}`, and its "share one product across the cross terms" trick is just what interpolation does at the middle point. The unification holds. The `∞` point is pulling its weight here: it hands me the leading coefficient for free, no combination to multiply.

Now I want to build the next rung concretely and turn it into running code, so take `k = 3`. Cut each number into three limbs:
```
p(t) = x₂t² + x₁t + x₀ ,   q(t) = y₂t² + y₁t + y₀ ,   x = p(B), y = q(B).
```
The product `r = p·q` has degree `4`, so it has five coefficients `r₀,…,r₄` and I need five points. I want the points to make both the evaluations and the interpolation cheap, so I want small finite points, including one opposite pair, and I'll spend two of the five on the freebies `0` and `∞`. Take
```
{ 0, 1, -1, -2, ∞ } .
```
Why these and not, say, `{0,1,2,3,4}`? Because the evaluations `p(s)` must be cheap — ideally just additions and tiny shifts, no real multiplication — and they must not blow up the size of the operands I then recurse on. At `s = 1` I add the limbs; at `s = -1` I alternate signs, which lets the even and odd parts separate cleanly; at `s = -2` I get `x₀ − 2x₁ + 4x₂`, where `2x₁` is a shift and `4x₂` is a shift. Using small magnitudes keeps `p(s)` about the size of a single limb (`n/3` digits) so the recursive products stay the right size; a point like `s = 4` would make `4²·x₂ = 16x₂` and inflate things. The point `-2` gives one more distinct small row without giving up the cheap shifts. So evaluate `p` and `q` at the five points:
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
(the `0` and `∞` rows degenerate to picking off `r₀` and `r₄`). Its determinant is `∏_{i<j}(s_j − s_i)` over the finite points, nonzero because the points are distinct, so it's invertible and `r` is uniquely recovered. I could just invert the matrix, but I should be careful about *how*, for two reasons: I don't want fractions piling up, and the order of operations decides whether the divisions stay exact. So solve it by hand the cheap way, eliminating with the freebies first.

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
i.e. `2r₁ + 8r₃ = −(v₋₂ − r₀ − 4r₂ − 16r₄)`, so `r₁ + 4r₃ = −(v₋₂ − r₀ − 4r₂ − 16r₄)/2`. Together with `r₁ + r₃` known, subtracting gives `3r₃ = (r₁+4r₃) − (r₁+r₃)`, and the `3` is the only odd division — it's exact because `r₃` is an integer (the product polynomial has integer coefficients, since `p,q` do). Then `r₁ = (r₁+r₃) − r₃`.

That works, and it tells me the structure: a couple of subtractions, a few divisions by `2`, and a single division by `3`. But the order matters for keeping every intermediate an exact integer, so let me pin down a clean operation sequence and then verify it symbolically rather than trust my hand-algebra. The sequence I'll use, recovering in an order that reuses partial results:
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
Each division (`/3`, then two `/2`) lands on an exact integer; no fractions survive. Let me expand the temporary variables against a generic `r(t)=c₀+c₁t+c₂t²+c₃t³+c₄t⁴` so the signs cannot hide. The first temporary `r₃` is `(v₋₂−v₁)/3 = -c₁+c₂-3c₃+5c₄`; the first temporary `r₁` is `(v₁−v₋₁)/2 = c₁+c₃`; and the first temporary `r₂` is `v₋₁−v₀ = -c₁+c₂-c₃+c₄`. Then the second `r₃` becomes `((-c₁+c₂-c₃+c₄)-(-c₁+c₂-3c₃+5c₄))/2 + 2c₄ = c₃`; the second `r₂` becomes `(-c₁+c₂-c₃+c₄)+(c₁+c₃)-c₄ = c₂`; and the final `r₁` becomes `(c₁+c₃)-c₃ = c₁`. The `/3` numerator is `3(-c₁+c₂-3c₃+5c₄)`, the first `/2` numerator is `2(c₁+c₃)`, and the second `/2` numerator is `2c₃-4c₄`, so the exact divisions are visible coefficient by coefficient.

Why does the division stay exact in general, not just in this hand-tuned order? Because the product polynomial `r = p·q` has integer coefficients, the Vandermonde system over the integers has an integer solution, and Gaussian elimination can be run so every intermediate entry stays an integer. The clean way to see it: do the elimination thinking of the points as indeterminates and the matrix rows as values of the monic powers `1, t, t², …`. Eliminating the `(k+1)`-th column subtracts evaluations of a monic polynomial `P` at two points, and `P(s_i) − P(s_j) = (s_i − s_j)·Q(s_i)` for a monic `Q` of degree one less — so each elimination step *factors out* the difference `(s_i − s_j)` cleanly, a division of one monic polynomial by another with no remainder. The differences `(s_i − s_j)` are exactly the small numbers like `3` and `2` showing up above. So over any integral domain the interpolation is exact; for limb polynomials over a modulus `b` the same holds provided `b` is prime, so that the ring is an integral domain and the Vandermonde system has a unique solution. For ordinary integers I'm always fine.

Last, the recomposition. I have `r₀,…,r₄`, the coefficients of `r`, and I want `x·y = r(B)` at the same limb base:
```
x·y = r₄B⁴ + r₃B³ + r₂B² + r₁B + r₀ .
```
This is shifts and adds, `O(n)`. One subtlety: some `r_i` can come out negative (the cross terms involve subtractions), and an `r_i` can exceed `B`. That's fine — evaluating the polynomial at `B` with full-precision integer arithmetic carries and borrows correctly; the final value is the exact product regardless of the intermediate signs of the coefficients. I won't normalize the limbs by hand; the single big-integer evaluation `Σ r_i Bⁱ` does it.

Now the algorithm has to bottom out. The recursion calls itself on the five `~n/3`-digit pointwise products, and the `-1` and `-2` evaluations can be negative, so the routine should strip the sign first and recurse on absolute values. Then I need a base case where nonnegative operands are small enough to multiply directly in `O(1)` — say when a number has at most a few base digits, just return the machine product. And the split must use integer floor-division and remainder to cut out the limbs; true division would turn limbs into non-integers and the operands would never shrink to the base case. Choose `m = ⌊n/3⌋ + 1` so each operand has at most three limbs in `B = baseᵐ`.

```python
THRESHOLD = 3   # operands with at most a few base digits multiply directly

def exact_div(value, divisor):
    quotient, remainder = divmod(value, divisor)
    if remainder:
        raise ArithmeticError("interpolation division was not exact")
    return quotient

def toom3(x, y, base=10):
    if base <= 1:
        raise ValueError("base must be greater than 1")

    if x < 0 or y < 0:
        sign = -1 if (x < 0) ^ (y < 0) else 1
        return sign * toom3(abs(x), abs(y), base)

    # base case: small enough that the direct product is O(1)
    if x < base ** THRESHOLD or y < base ** THRESHOLD:
        return x * y

    # limb size m so each operand has at most 3 limbs in B = base**m
    n = max(len(str(x)), len(str(y)))
    m = n // 3 + 1
    B = base ** m

    # cut into 3 limbs: x = x2*B^2 + x1*B + x0  (the polynomial p; q likewise)
    # integer floor-division/remainder — true division would never reach the base case
    x0, x1, x2 = x % B, (x // B) % B, x // (B * B)
    y0, y1, y2 = y % B, (y // B) % B, y // (B * B)

    # evaluate p, q at 0, 1, -1, -2, inf  (adds and small shifts, no multiply)
    px0, py0 = x0, y0
    px1, py1 = x0 + x1 + x2, y0 + y1 + y2
    pxm1, pym1 = x0 - x1 + x2, y0 - y1 + y2
    pxm2, pym2 = x0 - 2*x1 + 4*x2, y0 - 2*y1 + 4*y2
    pxinf, pyinf = x2, y2

    # the FIVE recursive multiplications — the whole saving (5, not 9):
    v0   = toom3(px0,   py0,   base)   # r(0)   = x0*y0
    v1   = toom3(px1,   py1,   base)   # r(1)
    vm1  = toom3(pxm1,  pym1,  base)   # r(-1)
    vm2  = toom3(pxm2,  pym2,  base)   # r(-2)
    vinf = toom3(pxinf, pyinf, base)   # r(inf) = x2*y2

    # interpolation: recover r0..r4 from the five values.
    # every division (/3, /2) is exact because r = p*q has integer coefficients.
    r0 = v0
    r4 = vinf
    r3 = exact_div(vm2 - v1, 3)
    r1 = exact_div(v1 - vm1, 2)
    r2 = vm1 - v0
    r3 = exact_div(r2 - r3, 2) + 2 * r4
    r2 = r2 + r1 - r4
    r1 = r1 - r3
    coeffs = [r0, r1, r2, r3, r4]

    # recompose r(B) = sum r_i * B^i  (shifts + adds, O(n)); handles negative
    # and oversized coefficients automatically via big-integer carries/borrows
    result = 0
    for c in reversed(coeffs):
        result = result * B + c
    return result
```

The causal chain, start to finish: the half-split's exponent `log₂ 3` was nothing but `log_{parts}(sub-products)`, so I turned the knob and cut into `k` parts — but the naive `k`-part split does the `k²` convolution and stays at `n²`, so the parts alone buy nothing; seeing each limb-number as a polynomial sampled at the base turned integer multiplication into polynomial multiplication plus a free evaluation; a degree-`2k-2` product polynomial is fixed by its values at `2k-1` points, and each value `r(s) = p(s)q(s)` is a single multiplication, so evaluate-at-`2k-1`-points / multiply-pointwise / interpolate-back does the convolution in `2k-1` sub-products instead of `k²`; that gives `T(n) = (2k-1)T(n/k)+O(n) = Θ(n^{log_k(2k-1)})`, an exponent below `2` that falls toward `1` as `k` grows, with Karatsuba reappearing as the three-point `k=2` case; and concretely at `k=3`, five points `{0,1,-1,-2,∞}`, five recursive products, a small exact interpolation, and a shift-and-add recomposition give `Θ(n^{log₃ 5}) ≈ Θ(n^{1.465})`.
