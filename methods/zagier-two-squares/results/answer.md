# Two-Squares Involution Proof

## The theorem

**Theorem.** Every prime $p \equiv 1 \pmod 4$ is a sum of two squares: there exist integers $a, b$ with $p = a^2 + b^2$.

## The key idea

For an involution $f$ on a finite set $S$, the orbits have size $1$ (fixed points) or $2$, so

$$|S| \equiv \#\{s \in S : f(s) = s\} \pmod 2.$$

Apply this to the finite set

$$S = \{(x,y,z) \in \mathbb{Z}_{>0}^3 : x^2 + 4yz = p\}.$$

Two involutions act on $S$:

- the **swap** $\sigma(x,y,z) = (x,z,y)$, whose fixed points have $y = z$ and hence give $p = x^2 + 4y^2 = x^2 + (2y)^2$ — a sum of two squares;
- a **second involution $\tau$** (below) with exactly one fixed point, so $|S|$ is odd.

Since $|S|$ is odd, $\sigma$ has an odd number of fixed points, in particular at least one; that fixed point is the desired representation. (Non-constructive: it proves $a,b$ exist without producing them.)

## The involution

$$
\tau(x,y,z) = \begin{cases}
(x + 2z,\ z,\ y - x - z) & \text{if } x < y - z, \\[2pt]
(2y - x,\ y,\ x - y + z) & \text{if } y - z < x < 2y, \\[2pt]
(x - 2y,\ x - y + z,\ y) & \text{if } x > 2y.
\end{cases}
$$

## The complete proof

**Finiteness.** From $x^2 + 4yz = p$ with $x,y,z \ge 1$ we get $x \le \sqrt p$ and $y,z \le p/4$, so $S$ is finite. It is nonempty: $(1,1,\tfrac{p-1}{4}) \in S$ since $p \equiv 1 \pmod 4$.

**Each branch preserves $x^2 + 4yz = p$.**
- Branch 1: $(x+2z)^2 + 4z(y-x-z) = x^2 + 4xz + 4z^2 + 4yz - 4xz - 4z^2 = x^2 + 4yz$.
- Branch 2: $(2y-x)^2 + 4y(x-y+z) = 4y^2 - 4xy + x^2 + 4xy - 4y^2 + 4yz = x^2 + 4yz$.
- Branch 3: $(x-2y)^2 + 4(x-y+z)y = x^2 - 4xy + 4y^2 + 4xy - 4y^2 + 4yz = x^2 + 4yz$.

**Each branch maps positive triples to positive triples.**
- Branch 1 (needs $x < y - z$): $x+2z>0$, $z>0$, and $y-x-z>0 \iff x<y-z$.
- Branch 2 (needs $y - z < x < 2y$): $2y-x>0 \iff x<2y$; $y>0$; $x-y+z>0 \iff x>y-z$.
- Branch 3 (needs $x > 2y$): $x-2y>0 \iff x>2y$; $x-y+z>0$ since $x>2y\ge y>0$; $y>0$.

**The boundaries $x = y-z$ and $x = 2y$ never occur on $S$.**
- If $x = 2y$ then $p = 4y^2 + 4yz = 4y(y+z)$, so $4 \mid p$, impossible because $p \equiv 1 \pmod 4$ is an odd prime.
- If $x = y-z$ then $y = x+z$ and $p = x^2 + 4(x+z)z = (x+2z)^2$, a nontrivial perfect square, impossible for a prime.

Since $y,z>0$, we have $y-z<2y$. Hence every triple in $S$ satisfies exactly one of $x<y-z$, $y-z<x<2y$, $x>2y$; the three branches partition $S$, so $\tau$ is well-defined.

**$\tau$ is an involution.** Branches 1 and 3 are mutually inverse. The image of a branch-1 triple is $(x+2z, z, y-x-z)$, which satisfies $x' = x+2z > 2z = 2y'$, so it lies in the $x>2y$ region, and applying branch 3 returns
$$(x+2z) - 2z = x,\quad (x+2z) - z + (y-x-z) = y,\quad z,$$
recovering $(x,y,z)$. Conversely, the image of a branch-3 triple is $(x-2y, x-y+z, y)$, which satisfies
$$x' = x-2y < x-2y+z = y' - z',$$
so it lies in the $x<y-z$ region, and applying branch 1 returns
$$(x-2y)+2y=x,\quad y,\quad (x-y+z)-(x-2y)-y=z.$$
Branch 2 is self-inverse on the middle region: its image $(2y-x, y, x-y+z)$ again satisfies $y-z<x'<2y$ (since $x'=2y-x<2y$ as $x>0$, and $x' = 2y-x > 2y-x-z = y'-z'$ as $z>0$), and
$$(2y-x,\,y,\,x-y+z) \longmapsto \big(2y-(2y-x),\ y,\ (2y-x)-y+(x-y+z)\big) = (x,y,z).$$
Thus $\tau \circ \tau = \mathrm{id}$.

**$\tau$ has exactly one fixed point.** A fixed point cannot lie in branch 1 or 3 because those branches swap the two outer regions. In branch 2, $\tau(x,y,z)=(x,y,z)$ forces $2y - x = x$, so $x = y$, and then the third coordinate is automatically fixed. Substituting $x=y$ into the equation for $S$ gives $p = x^2 + 4xz = x(x+4z)$. Since $p$ is prime and $x + 4z > x \ge 1$, the smaller factor must be $x=1$, hence $p = 1 + 4z$ and $z = \tfrac{p-1}{4}$. This is a positive integer because $p \equiv 1 \pmod 4$, and $\big(1,1,\tfrac{p-1}{4}\big)$ indeed satisfies $x^2+4yz=p$. The unique fixed point is $\big(1,1,\tfrac{p-1}{4}\big)$.

**Conclusion.** By the parity principle $|S| \equiv 1 \pmod 2$, so $|S|$ is odd. Then $\sigma(x,y,z)=(x,z,y)$ has an odd (hence positive) number of fixed points; any such fixed point has $y=z$, giving $p = x^2 + (2y)^2$. $\blacksquare$

## Windmill picture

Each triple $(x,y,z)$ is a **windmill**: a central $x \times x$ square with four $y \times z$ rectangular arms attached pinwheel-fashion, the side of length $y$ lying along the central square, total area $x^2 + 4yz = p$. The two outer branches of $\tau$ re-cut the windmill by growing the square when it is too small ($x < y-z$) and trimming it when it is too large ($x > 2y$), which is precisely the completing-the-square trade $x' = x \pm 2(\cdot)$. The middle branch $x \mapsto 2y - x$ is self-dual. The swap $\sigma$ flips each arm's two side lengths; its fixed points $y = z$ are the windmills whose four arms are squares of side $y$, giving $p = x^2 + (2y)^2$.
