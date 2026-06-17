# Boreico's sign-isolating SOS solution to IMO 2005 Problem 3

## The problem

Let $x, y, z > 0$ satisfy $xyz \ge 1$. Prove that
$$\frac{x^5 - x^2}{x^5 + y^2 + z^2} + \frac{y^5 - y^2}{y^5 + z^2 + x^2} + \frac{z^5 - z^2}{z^5 + x^2 + y^2} \ge 0.$$

The summands are not individually nonnegative ($x^5 - x^2 = x^2(x^3-1) < 0$ when $x < 1$), and the
expression is not homogeneous, so the difficulty is to control the cancellation between the positive
(large-variable) and negative (small-variable) terms and to use the inequality constraint $xyz \ge 1$
at the right spot.

## The key idea

Bound each fraction below by a fraction with the **common denominator** $x^2 + y^2 + z^2$, chosen so
that the three lower bounds add up to a manifest sum of squares. The per-term bound is

$$\boxed{\;\frac{x^5 - x^2}{x^5 + y^2 + z^2} \;\ge\; \frac{x^2 - yz}{x^2 + y^2 + z^2}.\;}$$

It is proved by routing through an intermediate fraction with the *same numerator* $x^5-x^2$ over the
straightened denominator $x^3(x^2+y^2+z^2)$: the difference of the original fraction and this
intermediate one factors as a perfect square over positives (the "sum of squares with signs" move),
which is where the sign of the changing numerator is isolated and neutralized; and the drop from the
intermediate fraction to $\frac{x^2-yz}{x^2+y^2+z^2}$ is exactly where $xyz \ge 1$ is consumed (via
$\tfrac1x \le yz$). Summing the three cyclic copies turns the right-hand sides into
$\tfrac12\sum(x-y)^2$ over $x^2+y^2+z^2 \ge 0$.

## The proof

**Lemma (per-term bound).** For $x,y,z>0$ with $xyz \ge 1$,
$$\frac{x^5-x^2}{x^5+y^2+z^2} \ge \frac{x^2 - yz}{x^2+y^2+z^2}.$$

*Proof.* First, the unconditional identity (direct computation)
$$\frac{x^5-x^2}{x^5+y^2+z^2} - \frac{x^5-x^2}{x^3(x^2+y^2+z^2)}
= \frac{(x^3-1)^2\,x^2\,(y^2+z^2)}{\big(x^5+y^2+z^2\big)\,\big(x^3(x^2+y^2+z^2)\big)} \ge 0,$$
since every factor is nonnegative ($(x^3-1)^2$ is a square, $x^2, y^2+z^2 \ge 0$, both denominators
$> 0$). Hence
$$\frac{x^5-x^2}{x^5+y^2+z^2} \ge \frac{x^5-x^2}{x^3(x^2+y^2+z^2)}
= \frac{x^2 - \tfrac1x}{x^2+y^2+z^2}.$$
Finally, $xyz \ge 1$ and $x>0$ give $\tfrac1x \le yz$, so $x^2 - \tfrac1x \ge x^2 - yz$, and dividing by
the positive $x^2+y^2+z^2$,
$$\frac{x^2 - \tfrac1x}{x^2+y^2+z^2} \ge \frac{x^2 - yz}{x^2+y^2+z^2}. \qquad\blacksquare$$

(Equivalently, the last step replaces the constant $1$ by $xyz$:
$x^5 - x^2 \ge x^5 - x^2\,(xyz) = x^3(x^2-yz)$, and $\dfrac{x^3(x^2-yz)}{x^3(x^2+y^2+z^2)} = \dfrac{x^2-yz}{x^2+y^2+z^2}$.)

**Theorem.** Summing the lemma over the three cyclic copies, all with the common denominator
$x^2+y^2+z^2$,
$$\sum_{\mathrm{cyc}} \frac{x^5-x^2}{x^5+y^2+z^2}
\ge \frac{(x^2-yz)+(y^2-zx)+(z^2-xy)}{x^2+y^2+z^2}
= \frac{\tfrac12\big[(x-y)^2+(y-z)^2+(z-x)^2\big]}{x^2+y^2+z^2} \ge 0,$$
using $x^2+y^2+z^2 - (xy+yz+zx) = \tfrac12\big[(x-y)^2+(y-z)^2+(z-x)^2\big]$. Equality holds iff
$x=y=z=1$. $\blacksquare$

## Why each piece

- **Common denominator $x^2+y^2+z^2$.** Forces the three per-term bounds to add without cross terms,
  and is the denominator that already lives inside every fraction once rewritten.
- **Target numerator $x^2-yz$.** Chosen so that $\sum(x^2-yz) = \sum x^2 - \sum yz = \tfrac12\sum(x-y)^2$
  is the canonical nonnegative symmetric certificate; the $yz$ also pairs with $\tfrac1x$ through the
  constraint.
- **Intermediate denominator $x^3(x^2+y^2+z^2)$.** The bridge that shares the factor $x^3$ with the
  numerator $x^2(x^3-1)$ *and* carries the target common denominator, so the difference of the two
  fractions collapses to a single perfect square $x^2(x^3-1)^2(y^2+z^2)$ over positives. Going directly
  to $\frac{x^2-yz}{x^2+y^2+z^2}$ in one step is messy and is in fact false without $xyz \ge 1$;
  splitting into an unconditional algebraic step plus one constraint step keeps both clean.
- **Where $xyz \ge 1$ is used.** Exactly once, in the final $\tfrac1x \le yz$ drop — consistent with
  $xyz = 1$ being the tight case.

All four identities hold exactly, so the chain
$\displaystyle\sum_{\mathrm{cyc}}\frac{x^5-x^2}{x^5+y^2+z^2}
\ge \sum_{\mathrm{cyc}}\frac{x^2-yz}{x^2+y^2+z^2}
= \frac{\tfrac12\sum(x-y)^2}{x^2+y^2+z^2} \ge 0$ is verified.
