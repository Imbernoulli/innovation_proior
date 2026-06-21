I would call this method Vieta jumping, or equivalently root-flipping infinite descent. It is the standard technique for the following problem: let $a$ and $b$ be positive integers such that $ab+1$ divides $a^2+b^2$; prove that the quotient $q=(a^2+b^2)/(ab+1)$ is always a perfect square. My first reaction is skepticism, because a divisibility condition between two free parameters usually lets the quotient be almost anything. So before trying to prove the claim I compute. The small admissible pairs, writing the smaller coordinate first, give $(1,1)\to1$, $(2,8)\to4$, $(3,27)\to9$, $(8,30)\to4$, $(27,240)\to9$, $(30,112)\to4$. Every quotient is $1$, $4$, $9$, never $2$, never $3$, never a non-square. The rigidity is absolute, which means it must come from the algebra rather than from coincidence.

The natural first tool for a sum of two squares is the Gaussian integers, using $a^2+b^2=(a+bi)(a-bi)$ and reading off which primes are $1$ modulo $4$. I try this and it stalls. The divisor $ab+1$ is a mixed quadratic in both variables; it does not factor over $\mathbb{Z}[i]$ in any way that communicates with the factorization of $a^2+b^2$, and the prime-by-prime bookkeeping never yields a statement about the quotient. The factorization of $a^2+b^2$ is real, but it is the wrong handle, because the real constraint is a relation between two integers lying on a curve. So I set Gaussian integers aside and work directly with the curve.

The key move is to clear the denominator and view the divisibility as a single equation. If $ab+1$ divides $a^2+b^2$ with quotient $q$, then
$$a^2+b^2=q(ab+1),$$
which rearranges to the symmetric conic
$$a^2-q\,ab+b^2-q=0.$$
Now I fix the value of $q$ and read this as a quadratic in one variable, say $a$, with $b$ held constant. That gives
$$x^2-(qb)\,x+(b^2-q)=0,$$
and my given $a$ is one root of this quadratic. A quadratic has two roots, and the second root $a'$ is delivered for free by Vieta's relations. The sum of the roots equals the coefficient $qb$ and the product equals the constant term $b^2-q$, so the same number $a'$ satisfies both
$$a'=qb-a\quad\text{and}\quad a'=\frac{b^2-q}{a}.$$
Each expression does different work. The first shows that $a'$ is an integer, because it is an integer minus an integer times an integer. The second shows that $a'$ is controlled in size. And because $a'$ is the other root of the quadratic built from the conic, the pair $(a',b)$ satisfies the same conic and therefore has the same quotient $q$. In one step I have manufactured a new integer solution with an unchanged quotient simply by flipping one coordinate to its companion root.

This is why Vieta jumping succeeds where a bare infinite-descent argument would struggle. Ordinary infinite descent only gives the contradiction skeleton; it has no built-in engine for producing a smaller solution. The second root of the quadratic is that engine. The sum-and-product relations hand me a smaller candidate automatically.

To turn the flip into a genuine descent I need the new pair to be strictly smaller and to stay in the first quadrant. Assume $a>b\ge1$ and replace $(a,b)$ by $(a',b)$ with $a'=qb-a$. First, $a'$ is an integer, already established. Second, the decrease follows from the product expression: $a'=(b^2-q)/a<b^2/a<b<a$, using $q>0$ for the first strict inequality and $b<a$ for the second. Thus $a'<b<a$, so the coordinate sum strictly drops. Third, and this is the subtle point, the sign of $a'$ is forced to be non-negative. The formula $(b^2-q)/a$ can look as if it might be negative when $q>b^2$, but the conic rules this out. The flipped pair satisfies $a'^2+b^2=q(a'b+1)$; the left side is a sum of squares with $b\ge1$, hence strictly positive, and $q>0$, so $a'b+1>0$, that is, $a'b>-1$. With $b\ge1$ this forces $a'\ge0$. A sum of two squares simply cannot equal a positive multiple of a non-positive number. Therefore the flip lands a strictly smaller non-negative integer solution with the same $q$.

Because the coordinate sum is a strictly decreasing sequence of non-negative integers, well-ordering forbids an infinite descent. The process must bottom out at a pair with one coordinate equal to zero. At such a base pair $(r,0)$ the quotient is $q=(r^2+0)/(0+1)=r^2$, manifestly a perfect square. Since $q$ never changed along the descent, the original quotient is that same square. This is exactly the structure visible in the small examples: for each square value of $q$, the admissible pairs are consecutive terms of one sequence governed by $T_n=q\,T_{n-1}-T_{n-2}$. For $q=4$ this gives $0,2,8,30,112,418,\dots$; for $q=9$ it gives $0,3,27,240,\dots$. Each flip is one backward step of that recurrence, walking the solution down to its base $(r,0)$.

The same argument can be packaged as a minimal-counterexample proof. Suppose some non-square positive integer $q$ were realized by an admissible pair. Among all pairs with that quotient, choose one $(A,B)$ with $A+B$ minimal and $A\ge B$. Since $q$ is non-square, $B$ cannot be zero, because $B=0$ would give $q=A^2$. So $B\ge1$. Flip $A$ to the companion root $x_2=qB-A=(B^2-q)/A$. The integer property and the sign argument give $x_2\ge0$, and because $q$ is non-square we have $x_2\ne0$; otherwise $B^2=q$ would make $q$ a square. Hence $x_2$ is a positive integer. The size estimate gives $x_2<B^2/A\le A$, so $x_2+B<A+B$. But $(x_2,B)$ is an admissible pair with the same $q$ and a strictly smaller coordinate sum, contradicting minimality. Therefore no non-square $q$ can occur.

Geometrically, for $q>2$ the equation $x^2+y^2-q\,xy-q=0$ is a hyperbola symmetric about the line $y=x$, with two branches. A first-quadrant lattice point $(x,y)$ with $x<y$ is sent by the companion-root relation to $(x,qx-y)$ on the other branch; the sign argument keeps this point in the first quadrant. Reflecting across $y=x$ gives $(qx-y,x)$ back on the original branch, with strictly smaller coordinates because $qx-y=(x^2-q)/y<x<y$. Repeating this flip-and-reflect slide marches the lattice point down the hyperbola until it reaches the axis at $(0,y)$, and substituting $x=0$ gives $q=y^2$. The algebraic descent and the geometric picture are the same proof.

What makes the method work in one sentence is this: the divisibility condition, once written as $a^2-q\,ab+b^2-q=0$, is quadratic in each variable separately, so every solution has a second integer root with the same $q$. Choosing the flip that decreases the solution turns that second root into a descent, and the descent terminates exactly where the quotient is forced to be a square. I therefore propose the canonical name Vieta jumping for this technique.

```python
import math

def is_square(n):
    r = int(math.isqrt(n))
    return r * r == n

def admissible_pairs(limit):
    """List unordered admissible pairs (a,b) with 0 <= a <= b <= limit."""
    found = []
    for a in range(limit + 1):
        for b in range(a, limit + 1):
            denom = a * b + 1
            numer = a * a + b * b
            if numer % denom == 0:
                found.append((a, b, numer // denom))
    return found

def vieta_descent(a, b):
    """Descend (a,b) with a >= b to a base pair (r,0), returning the path and q."""
    assert a >= b
    q = (a * a + b * b) // (a * b + 1)
    path = [(a, b)]
    while b != 0:
        a, b = b, q * b - a
        path.append((a, b))
    return path, q

# Brute-force check: every quotient up to 300 is a perfect square.
pairs = admissible_pairs(300)
assert all(is_square(q) for _, _, q in pairs)
print(f"Found {len(pairs)} admissible pairs with 0 <= a <= b <= 300.")
print(f"Largest quotient seen: {max((q for _, _, q in pairs), default=0)}.")

# Illustrate the two classical solution families.
for r in (2, 3):
    q = r * r
    seq = [0, r]
    for _ in range(6):
        seq.append(q * seq[-1] - seq[-2])
    print(f"q={q} family: {seq[:8]}")

# Trace a concrete descent.
path, q = vieta_descent(112, 30)
print(f"Descent for (112,30): {path}, q={q}")
path, q = vieta_descent(240, 27)
print(f"Descent for (240,27): {path}, q={q}")
```
