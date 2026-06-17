# Vieta Jumping (Root-Flipping Infinite Descent)

## The problem it solves

Let $a,b$ be positive integers such that $ab+1$ divides $a^2+b^2$. Then

$$q \;=\; \frac{a^2+b^2}{ab+1}$$

is the square of an integer.

More than the single result, the technique is a general pattern for quadratic
Diophantine equations: keep all but one variable fixed, use Vieta's relations to find the
other root of the resulting monic quadratic, and then use well-ordering when that
replacement decreases a chosen size.

## The key idea

Clear the denominator and view the relation as a conic,

$$a^2+b^2=q(ab+1)\quad\Longleftrightarrow\quad a^2-q\,ab+b^2-q=0,$$

then **fix $q$ and read it as a quadratic in a single variable**. For fixed $b$,

$$x^2-(qb)\,x+(b^2-q)=0$$

has the given $a$ as one root; by Vieta's relations its **second root** is

$$a'=qb-a=\frac{b^2-q}{a}.$$

The first expression shows $a'$ is an *integer*; because it is the other root of the same
quadratic, $(a',b)$ satisfies the same conic and hence has the same $q$. Replacing the
larger coordinate makes $a'<b$, so the solution strictly decreases while preserving $q$.
By well-ordering this descent cannot continue forever; it terminates at a solution with a
zero coordinate, where $q=r^2/(0+1)=r^2$ is manifestly a perfect square. Since $q$ is
unchanged along the descent, the original $q$ is that square.

## The decisive lemma

Let $q$ be a fixed positive integer and let $(a,b)$, with $a>b\ge1$, satisfy
$a^2-q\,ab+b^2-q=0$. Put $a'=qb-a$. Then:

1. **$a'$ is an integer** — it equals $qb-a$ (sum of the two roots is the coefficient
   $qb$).
2. **$a'\ge0$** — also $a'=(b^2-q)/a$ (product of the roots is $b^2-q$). The pair
   $(a',b)$ satisfies $a'^2+b^2=q(a'b+1)$; the left side is positive and $q>0$, so
   $a'b+1>0$, i.e. $a'b>-1$; with $b\ge1$ this forces $a'\ge0$.
3. **$a'<b$ (strict decrease)** — from
   $a'=(b^2-q)/a<b^2/a<b<a$ using $q>0$ and $b<a$. In particular
   $a'<b^2/a\le a$, but the stronger conclusion is $a'<b<a$.

Thus $(b,a')$ is an admissible pair with the same $q$ and strictly smaller coordinate
sum.

## Theorem and proof

**Theorem.** If $a,b$ are positive integers and $(ab+1)\mid(a^2+b^2)$, then
$q=(a^2+b^2)/(ab+1)$ is a perfect square.

**Proof (minimal-counterexample).**
Suppose for contradiction that some non-square positive integer $q$ is realized by an
admissible pair. Among all pairs $(a,b)$ of non-negative integers with
$(a^2+b^2)/(ab+1)=q$, choose $(A,B)$ minimizing $A+B$, and assume $A\ge B$. Since $q$ is
realized at all, a minimum exists by well-ordering. Also $B\ne0$, since $B=0$ would give
$q=A^2$ immediately; hence $B\ge1$.

Fix $B$ and regard $A$ as a root of

$$x^2-(qB)\,x+(B^2-q)=0.$$

Let $x_2$ be the other root. By Vieta's relations,

$$x_2=qB-A=\frac{B^2-q}{A}.$$

- From $x_2=qB-A$, $x_2$ is an **integer**.
- From $x_2=(B^2-q)/A$ and $q$ **non-square**, $x_2\ne0$ (else $B^2=q$).
- $(x_2,B)$ satisfies $x_2^2+B^2=q(x_2B+1)$; the left side is positive and $q>0$, so
  $x_2B+1>0$, hence $x_2>-1/B\ge-1$, giving $x_2\ge0$. With $x_2\ne0$, $x_2$ is a
  **positive integer**.
- $A\ge B$ gives $x_2=\dfrac{B^2-q}{A}<\dfrac{B^2}{A}\le A$, so $x_2<A$ and
  $x_2+B<A+B$.

Then $(x_2,B)$ is an admissible pair with the same $q$ but a strictly smaller coordinate
sum — contradicting the minimality of $A+B$. Hence no non-square $q$ occurs; $q$ is
always a perfect square. $\blacksquare$

**Proof (direct constant descent — no contradiction).**
If a coordinate is $0$ then $q$ is the square of the surviving coordinate (base case). If $a=b$
then $a^2+1\mid 2a^2$ forces $a^2+1\mid 2$, so $a=b=1$ and $q=1$. Otherwise take
$a>b\ge1$ and replace $(a,b)$ by $(qb-a,\,b)$; by the lemma this is a non-negative
integer pair with the **same** $q$ and strictly smaller coordinate sum. The sum is a
strictly decreasing sequence of non-negative integers, so the descent reaches a pair with
one coordinate $0$ after finitely many steps, where $q=r^2$. As $q$ is unchanged
throughout, the original $q=r^2$. $\blacksquare$

## Worked descent (illustration)

For $q=4$, the admissible pairs are consecutive terms of $0,2,8,30,112,418,\dots$
($T_n=4T_{n-1}-T_{n-2}$). Replacing the larger coordinate by the companion root,
$(a,b)\mapsto(qb-a,b)$, descends:

$$(112,30)\to(30,8)\to(8,2)\to(2,0),$$

each step keeping $q=4$ (e.g. $4\cdot30-112=8$, $4\cdot8-30=2$, $4\cdot2-8=0$), and at
the base $(2,0)$ one reads $q=2^2=4$. The $q=9$ family
$0,3,27,240,\dots$ ($T_n=9T_{n-1}-T_{n-2}$) descends

$$(240,27)\to(27,3)\to(3,0),$$

and at the base $(3,0)$ one reads $q=3^2=9$.

## Geometric form

Fix $q>2$; then $x^2+y^2-q\,xy-q=0$ is a hyperbola $H$, symmetric about $y=x$, with two
branches. A first-quadrant lattice point $(x,y)$ with $x<y$ is sent by the companion-root
relation to $(x,\,y')=(x,\,qx-y)$ on the other branch; the same sign argument gives
$y'\ge0$. Reflecting across $y=x$ gives $(y',x)$ on the original branch, with strictly
smaller coordinates because $y'=(x^2-q)/y<x<y$. The $x$-coordinates form a decreasing
sequence of non-negative integers, so the process terminates at a point $(0,y)$ on $H$;
substituting $x=0$ gives $q=y^2$. The two algebraic proofs above are this lattice-point
descent down the hyperbola to its axis.

## Why it works — the mechanism in one line

The relation is a binary quadratic form; fixing the quotient $q$ makes it a quadratic in
each variable, so every solution has a *second root* that is again an integer solution of
the same $q$. Choosing the flip that decreases the solution turns "the other root" into a
descent, and the descent bottoms out exactly where $q$ is forced to be a square. Descent
supplies the contradiction skeleton; Vieta's relations supply the engine that builds the
smaller solution — which is what generic infinite descent was always missing.
