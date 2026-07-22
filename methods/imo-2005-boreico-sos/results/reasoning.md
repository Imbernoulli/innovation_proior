Let me look at this thing properly before I touch it. Three positive reals, the only handle is $xyz \ge 1$, and I want

$$\frac{x^5 - x^2}{x^5 + y^2 + z^2} + \frac{y^5 - y^2}{y^5 + z^2 + x^2} + \frac{z^5 - z^2}{z^5 + x^2 + y^2} \ge 0.$$

First instinct: is each term nonnegative, so I can just add three nonnegative things and be done? No. The numerator $x^5 - x^2 = x^2(x^3 - 1)$ is negative whenever $x < 1$. And nothing stops a variable from being below $1$ — take $x = 1/2$, then $y$ and $z$ large enough that $xyz \ge 1$ still holds, and that first term is genuinely negative. So this isn't a sum of nonnegatives; it's a tug-of-war. The terms for the big variables (above $1$) are positive, the terms for the small variables are negative, and I have to show the positives win — any proof that doesn't grapple with that cancellation is missing the point.

Second instinct: the expression isn't homogeneous. Look at one fraction — the numerator has a degree-$5$ piece and a degree-$2$ piece, and the denominator likewise mixes $x^5$ with $y^2 + z^2$. So I can't just scale $(x,y,z) \mapsto (tx, ty, tz)$ to normalize the constraint away; scaling changes the value. And the constraint is an *inequality*, $xyz \ge 1$, not $xyz = 1$. Where does an inequality constraint even enter a sum of fractions? I'll come back to that — it's clearly the crux of where the proof's "juice" is.

Let me try the obvious global reduction first and see if it leads anywhere clean. Each term is $\frac{x^5 - x^2}{x^5+y^2+z^2}$; add $1$ to it and I get $\frac{x^5 - x^2 + x^5 + y^2 + z^2}{x^5+y^2+z^2}$ — ugh, that doesn't simplify. Add $1$ the other way: $1 - \frac{x^5-x^2}{x^5+y^2+z^2} = \frac{x^2 + y^2 + z^2}{x^5 + y^2 + z^2}$. *That* is clean. So the inequality $\sum \frac{x^5-x^2}{x^5+y^2+z^2} \ge 0$ is the same as $\sum\left(1 - \frac{x^2+y^2+z^2}{x^5+y^2+z^2}\right) \ge 0$, i.e.

$$\sum_{\mathrm{cyc}} \frac{x^2+y^2+z^2}{x^5+y^2+z^2} \le 3, \qquad\text{i.e.}\qquad \sum_{\mathrm{cyc}}\frac{1}{x^5+y^2+z^2} \le \frac{3}{x^2+y^2+z^2}.$$

OK, this is a known road. From here you reach for Cauchy–Schwarz to bound the denominator $x^5 + y^2 + z^2$ from below — you pick a companion sum $yz + y^2 + z^2$ and write $(x^5+y^2+z^2)(yz+y^2+z^2) \ge (\sqrt{x^5 \cdot yz} + y^2 + z^2)^2$, then notice $x^5 \cdot yz = x^4 \cdot xyz \ge x^4$ so the bracket is $\ge x^2 + y^2 + z^2$, and it all closes. It works. But I don't love it: it makes me pull the companion vector $yz + y^2 + z^2$ out of a hat, and the negative-term drama I identified at the start has been swept inside a global sum — I never *see* the small variable's term get dominated, it just comes out in the wash. And the other standard road, clear all denominators and grind it into a symmetric polynomial and beat it with Muirhead, is a page of monomials and tells me nothing. I want the cancellation to be visible. Let me try to find a proof that handles each term on its own terms.

So: a per-term bound. I want, for each variable, some lower bound $\frac{x^5-x^2}{x^5+y^2+z^2} \ge (\text{something simpler})$, where the three "somethings" add up to something I can see is $\ge 0$. For the three to add cleanly they had better share a common denominator. What common denominator? The most symmetric quantity floating around is $x^2 + y^2 + z^2$ — it appears in all three fractions once you stare at them, and it's symmetric so it survives the cyclic sum. So let me hope for bounds of the form

$$\frac{x^5-x^2}{x^5+y^2+z^2} \ge \frac{N(x,y,z)}{x^2+y^2+z^2}$$

with some numerator $N$, such that $N(x,y,z) + N(y,z,x) + N(z,x,y) \ge 0$ is transparent.

Now what should $N$ be? Work backward from the shape I want the sum to land on. The cleanest nonnegative symmetric thing I know with this denominator is the identity $x^2+y^2+z^2 - (xy+yz+zx) = \tfrac12\sum(x-y)^2 \ge 0$. That's *the* canonical "difference equals a sum of squares" certificate. So the dream is that the three numerators sum to exactly $x^2+y^2+z^2-(xy+yz+zx)$. That happens if I can take $N(x,y,z) = x^2 - yz$: then $\sum (x^2 - yz) = \sum x^2 - \sum yz = \tfrac12\sum(x-y)^2 \ge 0$. Beautiful — and $x^2 - yz$ is the right "degree" feel, and notice $yz$ is exactly the thing that pairs with $1/x$ through the constraint, which is encouraging given I still need to use $xyz \ge 1$ somewhere. So let me set my target as the per-term inequality

$$\frac{x^5-x^2}{x^5+y^2+z^2} \;\overset{?}{\ge}\; \frac{x^2 - yz}{x^2+y^2+z^2}.$$

If this is true for each variable, summing gives $\sum \frac{x^5-x^2}{x^5+y^2+z^2} \ge \frac{\frac12\sum(x-y)^2}{x^2+y^2+z^2} \ge 0$ and I'm completely done. So everything reduces to proving this one inequality. At $x=y=z=1$ both sides are $0$ — tight at the equality point, as it must be. And as $x\to\infty$ with $y,z$ fixed, LHS $\to \frac{x^5}{x^5} = 1$ and RHS $\to \frac{x^2}{x^2} = 1$ too — consistent.

Now, is it actually true? Let me cross-multiply and see. I want
$$(x^5 - x^2)(x^2+y^2+z^2) \ge (x^2 - yz)(x^5 + y^2 + z^2).$$
Both denominators are positive so the direction is preserved. Let me expand the difference LHS $-$ RHS and pray it factors nicely... actually, wait. Before I grind a quartic-in-disguise, let me feel out whether this can even hold *without* the constraint. Try $x = y = z = t$ with $t$ small, so $xyz = t^3 < 1$ — the constraint is violated, so the inequality is *allowed* to fail here, and if it fails I'll know the constraint is load-bearing in this per-term step. LHS $= \frac{t^5 - t^2}{t^5 + 2t^2}= \frac{t^3-1}{t^3+2}$, RHS $= \frac{t^2 - t^2}{3t^2} = 0$. For small $t$, LHS $= \frac{t^3-1}{t^3+2} < 0 = $ RHS. So the per-term inequality is *false* when $xyz < 1$. Good — that tells me $xyz \ge 1$ has to be used inside this step, and it tells me a direct cross-multiplication will produce something that is only nonnegative after I feed in the constraint. That's a warning: don't expect a clean unconditional factorization from brute force; the constraint is tangled in.

Let me think about how to *insert* the constraint surgically rather than hope it falls out of an expansion. The two places $xyz \ge 1$ likes to appear: as $1/x \le yz$, or by replacing a bare $1$ with $xyz$. Look at my target numerators. On the left I have $x^5 - x^2 = x^2(x^3-1)$. On the right I want $x^2 - yz$. The $yz$ is begging to come from $1/x$: if I had $x^2 - 1/x$ I could push it down to $x^2 - yz$ using $1/x \le yz$, since $-1/x \ge -yz$. So maybe the natural intermediate target is $\frac{x^2 - 1/x}{x^2+y^2+z^2}$, and that last drop to $x^2 - yz$ is *exactly* where, and the *only* place where, the constraint gets spent. Let me split the job:

(A) an unconditional step: $\dfrac{x^5-x^2}{x^5+y^2+z^2} \ge \dfrac{x^2 - 1/x}{x^2+y^2+z^2}$, pure algebra, no constraint;

(B) a constraint step: $\dfrac{x^2 - 1/x}{x^2+y^2+z^2} \ge \dfrac{x^2 - yz}{x^2+y^2+z^2}$, which is just $-1/x \ge -yz$, i.e. $yz \ge 1/x$, i.e. $xyz \ge 1$.

Step (B) is immediate and I can see exactly where the constraint lives — perfect, that's the surgical insertion I wanted. So the whole problem is now step (A). Notice $\frac{x^2 - 1/x}{x^2+y^2+z^2}$ — let me rewrite that numerator over the denominator I'd actually want to manipulate. $x^2 - \frac1x = \frac{x^3 - 1}{x}$. And here's a thought: $\frac{x^2-1/x}{x^2+y^2+z^2} = \frac{x^3-1}{x(x^2+y^2+z^2)} = \frac{x^2(x^3-1)}{x^3(x^2+y^2+z^2)} = \frac{x^5 - x^2}{x^3(x^2+y^2+z^2)}$. 

That's a lovely coincidence — the *same numerator* $x^5 - x^2$ as on the left, just over a different denominator $x^3(x^2+y^2+z^2)$ instead of $x^5+y^2+z^2$. So step (A) is asking me to compare one fraction with another fraction that has the identical numerator. That's the cleanest possible kind of comparison: same top, and I just need to know which bottom is "better." Step (A) becomes

$$\frac{x^5-x^2}{x^5+y^2+z^2} \;\ge\; \frac{x^5-x^2}{x^3(x^2+y^2+z^2)}.$$

Now I can really see what's going on. Write $D_1 = x^5+y^2+z^2$ and $D_2 = x^3(x^2+y^2+z^2) = x^5 + x^3y^2 + x^3z^2$. Same numerator $x^5 - x^2$ over $D_1$ vs over $D_2$. If the numerator were always positive I'd just want the smaller denominator; but the numerator changes sign (negative for $x<1$), so a naive "smaller denominator wins" is wrong — exactly the sign subtlety from the very start, now localized into this one comparison. I can't shortcut it; I have to actually compute the difference and let the signs sort themselves out. Let me subtract:

$$\frac{x^5-x^2}{D_1} - \frac{x^5-x^2}{D_2} = (x^5 - x^2)\cdot\frac{D_2 - D_1}{D_1 D_2}.$$

$D_2 - D_1 = (x^5 + x^3 y^2 + x^3 z^2) - (x^5 + y^2 + z^2) = (x^3 - 1)(y^2 + z^2)$. And $x^5 - x^2 = x^2(x^3-1)$. So the numerator of the whole difference is

$$x^2(x^3-1)\cdot(x^3-1)(y^2+z^2) = x^2 (x^3-1)^2 (y^2+z^2).$$

The two $(x^3-1)$ factors — one from the numerator $x^5-x^2$, one from $D_2 - D_1$ — multiply into $(x^3-1)^2$, a perfect square, and the sign problem evaporates. The difference is

$$\frac{x^5-x^2}{x^5+y^2+z^2} - \frac{x^5-x^2}{x^3(x^2+y^2+z^2)} = \frac{x^2\,(x^3-1)^2\,(y^2+z^2)}{(x^5+y^2+z^2)\cdot x^3(x^2+y^2+z^2)},$$

and every factor on the right is manifestly nonnegative — $x^2 \ge 0$, $(x^3-1)^2 \ge 0$, $y^2 + z^2 > 0$, both denominators positive. So the difference is $\ge 0$, which is exactly step (A). And it's *unconditional* — I never touched $xyz \ge 1$ here, consistent with my plan that the constraint only enters in (B). The whole reason it works is that the awkward denominator $x^5 + y^2 + z^2$ got swapped for the "straightened" $x^3(x^2+y^2+z^2)$, which shares the factor $x^3$ with the numerator $x^2(x^3-1)$ and carries the common denominator $x^2+y^2+z^2$ I wanted all along — and that swap is precisely what makes the cross term collapse to a square. (Note $(x^3-1)^2 = (x-1)^2(x^2+x+1)^2$, so it still vanishes exactly at $x=1$, the equality point, as a square must; the extra $x^2+x+1$ factor is just forced by the degree gap between $x^5$ and $x^3$.)

Putting (A) and (B) together: from (A), $\frac{x^5-x^2}{x^5+y^2+z^2} \ge \frac{x^5-x^2}{x^3(x^2+y^2+z^2)} = \frac{x^2 - 1/x}{x^2+y^2+z^2}$. From (B), since $xyz \ge 1$ gives $\frac1x \le yz$, hence $x^2 - \frac1x \ge x^2 - yz$, so $\frac{x^2-1/x}{x^2+y^2+z^2} \ge \frac{x^2-yz}{x^2+y^2+z^2}$. Chaining,

$$\frac{x^5-x^2}{x^5+y^2+z^2} \ge \frac{x^2 - yz}{x^2 + y^2 + z^2}.$$

There's a way to reach (B) without ever introducing the fraction $1/x$: replace the constant $1$ directly by $xyz$ inside the numerator $x^5-x^2$. Since $xyz \ge 1$ and $x^2 \ge 0$, $x^2\cdot xyz \ge x^2$, so $x^5 - x^2\cdot xyz \le x^5-x^2$, i.e. $x^5-x^2 \ge x^5 - x^2\cdot xyz = x^3(x^2-yz)$. Dividing by the positive $x^3(x^2+y^2+z^2)$ gives $\frac{x^5-x^2}{x^3(x^2+y^2+z^2)} \ge \frac{x^3(x^2-yz)}{x^3(x^2+y^2+z^2)} = \frac{x^2-yz}{x^2+y^2+z^2}$ directly — the same bound, reached without passing through $1/x$, confirming the constraint is spent in exactly one place.

Now sum the three cyclic copies. Each lower bound has the *same* denominator $x^2+y^2+z^2$, which is the entire reason I aimed for that target:

$$\sum_{\mathrm{cyc}} \frac{x^5-x^2}{x^5+y^2+z^2} \ge \frac{(x^2-yz) + (y^2-zx) + (z^2-xy)}{x^2+y^2+z^2}.$$

And the numerator is $x^2+y^2+z^2 - (xy+yz+zx) = \tfrac12\big[(x-y)^2 + (y-z)^2 + (z-x)^2\big]$, a sum of squares. So

$$\sum_{\mathrm{cyc}} \frac{x^5-x^2}{x^5+y^2+z^2} \ge \frac{\tfrac12\big[(x-y)^2+(y-z)^2+(z-x)^2\big]}{x^2+y^2+z^2} \ge 0.$$

Done. And I can read the equality case straight off: every step is an equality exactly when each $(x^3-1)^2(y^2+z^2)$ vanishes and each $(x-y)^2$ vanishes and $1/x = yz$ — that forces $x=y=z$ and $x^3=1$, i.e. $x=y=z=1$, which indeed makes every original term $0$.
