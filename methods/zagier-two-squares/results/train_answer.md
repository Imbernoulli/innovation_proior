We want to prove that every prime $p \equiv 1 \pmod 4$ is a sum of two squares, $p = a^2 + b^2$. The standard route does this in two separable stages, and the friction between those stages is exactly what I want to remove. First one shows that $-1$ is a quadratic residue: $(\mathbb Z/p)^\times$ is cyclic of order $p-1$, and since $4 \mid p-1$ there is an element of order $4$ whose square is $-1$, so $m^2 \equiv -1 \pmod p$ for some $m$. Then, separately, one converts that congruence into a representation — by Thue's pigeonhole lemma on the values $u - mv$, or by Fermat's infinite descent starting from $m^2 + 1 = kp$ and pushing $k$ down to $1$. Both stages are real labor of two different flavors (a group-theoretic existence statement and an inequality-chasing descent), and they don't communicate: the residue $m$ is conjured by one machine and fed to another. What I want instead is for the existence of $a$ and $b$ to fall out of a single structural fact, with the quadratic-residue content absorbed into that fact rather than imported as a separate lemma.

The tool that lets one prove a fixed point *exists* without exhibiting it is the parity principle for involutions. If $f \colon S \to S$ satisfies $f \circ f = \mathrm{id}$ on a finite set $S$, then $S$ decomposes into orbits of size $2$ (pairs $\{s, f(s)\}$) and size $1$ (the fixed points), so the cardinality and the number of fixed points share a parity: $$|S| \equiv \#\{s \in S : f(s) = s\} \pmod 2.$$ If I can build a set whose desired objects are the fixed points of some involution, and independently pin the parity of $|S|$, then parity *forces* a fixed point into existence. So I propose to prove the two-squares theorem by what I will call the windmill involution argument: realize "$p$ is a sum of two squares" as a fixed-point statement for one involution, and supply the needed parity of $|S|$ with a *second*, cleverly chosen involution that has exactly one fixed point.

The construction is this. I parametrize a sum of two squares so that one of the squares is even: writing the second square as $(2y)^2 = 4y^2$ and reading $4y^2$ as the diagonal $y=z$ of $4yz$, a sum of two squares $p = x^2 + (2y)^2$ becomes the diagonal case of the three-variable equation $x^2 + 4yz = p$. So I work on the finite set $$S = \{(x,y,z) \in \mathbb Z_{>0}^3 : x^2 + 4yz = p\},$$ which is finite because $x \le \sqrt p$ and $y, z \le p/4$, and nonempty because $p \equiv 1 \pmod 4$ makes $(1, 1, \tfrac{p-1}{4})$ a genuine element. On $S$ the dead-obvious involution is the swap $$\sigma(x,y,z) = (x,z,y),$$ whose fixed points are exactly the triples with $y = z$, and each such fixed point gives $p = x^2 + 4y^2 = x^2 + (2y)^2$ — a real sum of two squares. This is why the factor $4$ rather than $1$ was the right choice: the diagonal of $x^2 + 4yz$ lands on $x^2 + (\text{even})^2$, which reads directly as a sum of two squares, and (as it turns out) the $4$ also makes the second, harder involution algebraically clean. The whole theorem has now collapsed to a single combinatorial claim: if $|S|$ is *odd*, then $\sigma$ has an odd, hence positive, number of fixed points, so a representation exists. Notice that neither $p \equiv 1 \pmod 4$ nor primality has been used yet — that content must enter exactly when I prove $|S|$ is odd, which tells me precisely where the real input lives.

To force $|S|$ odd I exhibit a *second* involution $\tau$ on $S$ with exactly one fixed point; then $|S| \equiv \#\mathrm{Fix}(\tau) = 1 \pmod 2$. The map $\tau$ is found by completing-the-square surgeries that preserve the conserved quantity $x^2 + 4yz$ while staying in the positive orthant. Shifting $x' = x + 2z$ overshoots $x^2$ by $4z(x+z)$, which is repaired by keeping $y' = z$ and setting $z' = y - x - z$, giving the branch $(x,y,z) \mapsto (x+2z,\, z,\, y-x-z)$, legal exactly where $y - x - z > 0$, i.e. $x < y - z$. This single rule is not its own inverse, so it must be paired with the rule that undoes it: inverting it algebraically yields $(x,y,z) \mapsto (x - 2y,\, x - y + z,\, y)$, and one checks the image of the first branch always satisfies $x' = x + 2z > 2z = 2y'$, so the first branch maps into the region $x > 2y$ and the third maps back, making them mutual inverses. The remaining middle region $y - z < x < 2y$ must map to itself; reflecting $x$ about $y$ via $x' = 2y - x$ (compensated by $z' = x - y + z$) gives the self-inverse branch $(x,y,z) \mapsto (2y - x,\, y,\, x - y + z)$, and the two defining inequalities of the middle region are exactly the positivity conditions $2y - x > 0$ and $x - y + z > 0$, so the region is tailor-made for this rule. Assembling the three branches gives a $\tau$ that preserves $x^2 + 4yz = p$ everywhere, sends positive triples to positive triples, swaps the two outer regions, and is an involution on the middle region.

For $\tau$ to be a well-defined involution on all of $S$ the two boundary cases must be empty, and this is exactly where primality enters. If $x = 2y$ then $p = 4y^2 + 4yz = 4y(y+z)$, so $4 \mid p$, impossible for the odd prime $p \equiv 1 \pmod 4$. If $x = y - z$ then $y = x + z$ and $p = x^2 + 4(x+z)z = (x + 2z)^2$, a nontrivial perfect square, impossible for a prime. With both boundaries excluded and $y - z < 2y$ always, the three open regions partition $S$, so $\tau$ is well-defined and $\tau \circ \tau = \mathrm{id}$. Its fixed points cannot lie in the outer branches (those move a point between the two outer regions), so a fixed point lies in the middle, where $2y - x = x$ forces $x = y$; then $p = x^2 + 4xz = x(x + 4z)$, and primality with $x + 4z > x \ge 1$ forces $x = 1$ and $z = \tfrac{p-1}{4}$, a positive integer precisely because $p \equiv 1 \pmod 4$. So $\tau$ has the unique fixed point $(1, 1, \tfrac{p-1}{4})$, $|S|$ is odd, and $\sigma$ therefore has a fixed point $(x,y,y)$ giving $p = x^2 + (2y)^2$. What makes this work is that any two involutions on the same finite set have fixed-point sets of equal parity, both equal to $|S| \bmod 2$: the involution $\tau$ has a fixed-point set I can count exactly, while $\sigma$ has the fixed points I want but cannot count directly, and matching their parities transfers existence from the one I can count to the one I want. The quadratic-residue statement never has to be split off; it is hidden inside this parity transfer. The argument is non-constructive — it proves some fixed point of $\sigma$ exists without locating it.

The picture that makes $\tau$ feel inevitable reads each triple $(x,y,z)$ as a windmill: a central $x \times x$ square with four $y \times z$ rectangular arms attached pinwheel-fashion, the side of length $y$ lying along the square, total area $x^2 + 4yz = p$. The two outer branches re-cut the windmill to the largest central square the same four arms can frame — growing the square when it is too small and trimming it when it is too large, which is exactly the $x' = x \pm 2(\cdot)$ trade — and the middle branch $x \mapsto 2y - x$ is the self-dual case where the square is already maximal. The swap $\sigma$ flips each arm's two side lengths; its fixed points $y = z$ are the windmills whose arms are squares, reassembling into $x^2 + (2y)^2$. The unit windmill $(1,1,\tfrac{p-1}{4})$ is the one shape $\tau$ pins, and parity guarantees some other, square-armed windmill exists for $\sigma$ to pin.

**Theorem.** Every prime $p \equiv 1 \pmod 4$ is a sum of two squares: there exist integers $a, b$ with $p = a^2 + b^2$.

For an involution $f$ on a finite set $S$, the orbits have size $1$ (fixed points) or $2$, so

$$|S| \equiv \#\{s \in S : f(s) = s\} \pmod 2.$$

Apply this to the finite set

$$S = \{(x,y,z) \in \mathbb{Z}_{>0}^3 : x^2 + 4yz = p\}.$$

Two involutions act on $S$:

- the **swap** $\sigma(x,y,z) = (x,z,y)$, whose fixed points have $y = z$ and hence give $p = x^2 + 4y^2 = x^2 + (2y)^2$ — a sum of two squares;
- a **second involution $\tau$** (below) with exactly one fixed point, so $|S|$ is odd.

Since $|S|$ is odd, $\sigma$ has an odd number of fixed points, in particular at least one; that fixed point is the desired representation. (Non-constructive: it proves $a,b$ exist without producing them.)

$$
\tau(x,y,z) = \begin{cases}
(x + 2z,\ z,\ y - x - z) & \text{if } x < y - z, \\[2pt]
(2y - x,\ y,\ x - y + z) & \text{if } y - z < x < 2y, \\[2pt]
(x - 2y,\ x - y + z,\ y) & \text{if } x > 2y.
\end{cases}
$$

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
