We need to multiply two $n$-digit integers exactly, and the only thing we count is the asymptotic number of elementary digit operations as $n$ grows. Schoolbook multiplication forms every digit-against-digit partial product and costs $\Theta(n^2)$; because every pair of digits genuinely interacts in the answer, that quadratic cost felt forced for a long time. The half-split method of Karatsuba and Ofman (1962) broke it: cutting each operand at the middle, $x = x_1 B^m + x_0$ and $y = y_1 B^m + y_0$ with $m = n/2$, the obvious expansion wants four half-size products, but the middle coefficient needs only the *sum* $x_1 y_0 + x_0 y_1$, and that sum is recovered as $(x_1+x_0)(y_1+y_0) - x_1 y_1 - x_0 y_0$ from one extra product minus the two corners already computed. Three half-size products give $T(n) = 3T(n/2) + O(n) = \Theta(n^{\log_2 3}) \approx n^{1.585}$. The trouble is that $\log_2 3$ is a strange place to stop. It is nothing but $\log_2$ of the number of sub-products, where the base $2$ is the number of parts; the whole exponent is $\log_{(\text{parts})}(\text{sub-products})$. That begs us to turn the knob: cut into $k > 2$ parts and see whether the exponent falls. But the naive $k$-part split warns us not to be optimistic — cutting into $k$ limbs and computing the convolution coefficients $c_\ell = \sum_{i+j=\ell} x_i y_j$ directly costs one sub-product per pair $(i,j)$, i.e. $k^2$ sub-multiplications, so $T(n) = k^2 T(n/k) + O(n)$ and $\log_k(k^2) = 2$, right back to $\Theta(n^2)$. More parts by themselves buy nothing. Any win must come from doing the convolution in fewer than $k^2$ sub-products, the way the half-split cut four down to three.

I propose Toom–Cook multiplication: split each operand into $k$ limbs, but compute the limb convolution by evaluation, pointwise multiplication, and interpolation rather than by brute force. The structural observation that unlocks everything is that a number cut into limbs is a polynomial sampled at the limb base. Writing $B = b^m$ and
$$p(t) = x_{k-1}t^{k-1} + \cdots + x_1 t + x_0, \qquad q(t) = y_{k-1}t^{k-1} + \cdots + y_1 t + y_0,$$
we have $x = p(B)$ and $y = q(B)$, so $x\cdot y = p(B)\,q(B) = (p\cdot q)(B)$. Integer multiplication has become *polynomial* multiplication of two degree-$(k-1)$ polynomials, followed by a single evaluation of the product at $t = B$, which is just shifting each coefficient left by $\ell\cdot m$ digits and adding — $O(n)$, free. All the cost lives in finding the coefficients of $r := p\cdot q$. Now, a polynomial is pinned down not only by its coefficients but equally by its *values*: a polynomial of degree $d$ is determined by its values at any $d+1$ distinct points. Our product $r$ has degree $2k-2$, so it is fixed by its values at $2k-1$ distinct points, and the value at a point $s$ is
$$r(s) = p(s)\cdot q(s),$$
a single number times a single number. This is the load-bearing move: evaluate $p$ and $q$ at $2k-1$ points (cheap linear combinations of the limbs, adds and small shifts, $O(n)$), do *one* multiplication $p(s)q(s)$ per point to obtain $r(s)$, and then interpolate the $2k-1$ coefficients of $r$ back from its $2k-1$ values by solving the Vandermonde linear system the points define. The product-of-two-numbers structure of $r(s)=p(s)q(s)$ is exactly what makes each value cost only one multiply, so the convolution that needed $k^2$ sub-products now needs $2k-1$ of them, each on operands of size $n/k$. The recurrence becomes
$$T(n) = (2k-1)\,T(n/k) + O(n) \;\Rightarrow\; T(n) = \Theta\!\big(n^{\log_k(2k-1)}\big).$$
Since $2k-1 < k^2$ for every $k \ge 2$, the exponent is always below $2$, and as $k\to\infty$, $\log_k(2k-1) = 1 + \log_k 2 - o(1) \to 1$, so multiplication runs in $n^{1+\varepsilon}$ for any $\varepsilon>0$. This answers whether $1.585$ is special: it is not, it is merely the $k=2$ rung of a ladder. And the half-split is precisely that rung — with points $\{0, \infty, 1\}$, $r(0)=x_0 y_0$ is the constant coefficient, $r(\infty)$ (the leading-coefficient "value at infinity", the limit of $r(t)/t^d$) is $x_1 y_1$, and the middle coefficient comes out of interpolation as $r(1) - r(\infty) - r(0) = (x_1+x_0)(y_1+y_0) - x_1 y_1 - x_0 y_0$, exactly Karatsuba's trick. The "share one product across the cross terms" idea is just what interpolation does at the middle point, and the $\infty$ point earns its keep by handing over the leading coefficient with no combination to form.

The case I build concretely is $k=3$ (Toom-3), giving $\Theta(n^{\log_3 5}) \approx n^{1.465}$. Cut into three limbs, $p(t)=x_2 t^2 + x_1 t + x_0$ and $q$ likewise; the product $r$ has degree $4$ and five coefficients $r_0,\dots,r_4$, so I need five points. I take $\{0, 1, -1, -2, \infty\}$, and the choice matters. The evaluations $p(s)$ must be cheap and, crucially, must not inflate the size of the operands I recurse on. At $s=1$ I just add the limbs; at $s=-1$ the alternating signs separate the even and odd parts cleanly; at $s=-2$ I get $x_0 - 2x_1 + 4x_2$, where the $2$ and $4$ are small shifts. Keeping the magnitudes small holds $p(s)$ to about one limb ($n/3$ digits) so the recursive products stay the right size; a point like $s=4$ would make $16x_2$ and bloat the operands. The two freebies $0$ and $\infty$ cost no multiplication at all and read off $r_0$ and $r_4$ directly. So I evaluate
$$p(0)=x_0,\quad p(1)=x_0+x_1+x_2,\quad p(-1)=x_0-x_1+x_2,\quad p(-2)=x_0-2x_1+4x_2,\quad p(\infty)=x_2,$$
identically for $q$, and form the five pointwise products $v_0,v_1,v_{-1},v_{-2},v_\infty$ — the only real multiplications, the five I recurse on, in place of the nine of the convolution. Recovering the coefficients is a linear system whose matrix is the Vandermonde matrix of the points; its determinant $\prod_{i<j}(s_j-s_i)$ over the finite points is nonzero because the points are distinct, so $r$ is uniquely recovered. I solve it the cheap way, eliminating with the freebies first: $r_0=v_0$ and $r_4=v_\infty$ come immediately; subtracting those, the $+1$ and $-1$ rows add and subtract to separate even from odd parts, giving $r_2$ and the sum $r_1+r_3$; the fifth point $-2$ supplies the one extra equation needed to split $r_1$ from $r_3$. Concretely the ordered sequence is
$$r_0=v_0,\quad r_4=v_\infty,\quad r_3=\frac{v_{-2}-v_1}{3},\quad r_1=\frac{v_1-v_{-1}}{2},\quad r_2=v_{-1}-v_0,$$
$$r_3=\frac{r_2-r_3}{2}+2v_\infty,\quad r_2=r_2+r_1-v_\infty,\quad r_1=r_1-r_3,$$
reusing partial results so only a single $/3$ and two $/2$ divisions appear. Each division lands on an exact integer — this is essential, and it is not luck. Because $r=p\cdot q$ has integer coefficients, the Vandermonde system over $\mathbb{Z}$ has an integer solution, and the elimination can be run so every intermediate stays integral: each step subtracts evaluations of a monic polynomial $P$ at two points, and $P(s_i)-P(s_j) = (s_i-s_j)\,Q(s_i)$ for a monic $Q$ of degree one less, so the point-difference factors out cleanly with no remainder. Those differences are exactly the small numbers $3$ and $2$ that show up as divisors. (For limb polynomials reduced modulo $b$ the same holds provided $b$ is prime, so the ring is an integral domain and the system stays uniquely solvable; over the ordinary integers we are always safe.) Finally I recompose $x\cdot y = r_4 B^4 + r_3 B^3 + r_2 B^2 + r_1 B + r_0$ by shifts and adds; some $r_\ell$ can be negative or exceed $B$, but evaluating the polynomial at $B$ with full-precision integer arithmetic carries and borrows correctly, so no manual normalization is needed. The recursion bottoms out by stripping signs and recursing on absolute values (the $-1$ and $-2$ evaluations can be negative), with a base case that multiplies directly once an operand is small, and a limb size $m = \lfloor n/3 \rfloor + 1$ chosen via integer floor-division and remainder so the limbs are genuine integers and the operands actually shrink toward the base case.

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

    # cut into 3 limbs (the polynomials p, q); integer floor-div/remainder only
    x0, x1, x2 = x % B, (x // B) % B, x // (B * B)
    y0, y1, y2 = y % B, (y // B) % B, y // (B * B)

    # evaluate p, q at 0, 1, -1, -2, inf  (adds + small shifts, no multiply)
    px1, py1 = x0 + x1 + x2,        y0 + y1 + y2
    pxm1, pym1 = x0 - x1 + x2,      y0 - y1 + y2
    pxm2, pym2 = x0 - 2*x1 + 4*x2,  y0 - 2*y1 + 4*y2

    # the FIVE recursive multiplications (5 instead of 9)
    v0   = toom3(x0,   y0,   base)
    v1   = toom3(px1,  py1,  base)
    vm1  = toom3(pxm1, pym1, base)
    vm2  = toom3(pxm2, pym2, base)
    vinf = toom3(x2,   y2,   base)

    # interpolation (exact: r = p*q has integer coefficients)
    r0 = v0
    r4 = vinf
    r3 = exact_div(vm2 - v1, 3)
    r1 = exact_div(v1 - vm1, 2)
    r2 = vm1 - v0
    r3 = exact_div(r2 - r3, 2) + 2 * r4
    r2 = r2 + r1 - r4
    r1 = r1 - r3
    coeffs = [r0, r1, r2, r3, r4]

    # recompose r(B) = sum r_i * B^i  (shifts + adds, O(n))
    result = 0
    for c in reversed(coeffs):
        result = result * B + c
    return result


if __name__ == "__main__":
    import random
    for _ in range(5000):
        a = random.randint(-10 ** random.randint(0, 40), 10 ** random.randint(0, 40))
        b = random.randint(-10 ** random.randint(0, 40), 10 ** random.randint(0, 40))
        assert toom3(a, b) == a * b
    assert toom3(123456, 654321) == 80779853376
    print("all toom3 tests passed")
```
