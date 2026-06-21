I start with the smallest obstruction, because the whole question is too large if I look at the full space at once. I have a linear functional `f` on a subspace `M`, and I know it is controlled there. In a normed space the control is `|f(x)|<=C||x||`; in the more flexible form it is `f(x)<=p(x)` for a sublinear `p`. I want to keep that control after I define values outside `M`. Choosing arbitrary values on a basis is useless, because the first arbitrary choice can make the estimate fail on a vector like `x+t z`. The missing value cannot be free. It has to be squeezed by all old vectors at once.

So I add only one new direction. Take `z notin M`. Any linear extension to `M+R z` has only one unknown number: `c=F(z)`, and then necessarily `F(x+t z)=f(x)+t c`. The domination condition becomes

`f(x)+t c <= p(x+t z)`

for every `x in M` and every real `t`. If `t>0`, divide by `t` and rewrite `x/t` as an old vector. I get

`c <= p(y+z)-f(y)` for every `y in M`.

If `t<0`, the same inequality reverses after division, and the condition becomes

`c >= f(x)-p(x-z)` for every `x in M`.

Thus the single number `c` has to lie between two families of bounds:

`sup_{x in M} (f(x)-p(x-z)) <= c <= inf_{y in M} (p(y+z)-f(y))`.

At first this looks like I have only renamed the difficulty: infinitely many lower bounds and infinitely many upper bounds. But now the whole problem is concentrated into one question: why is this interval nonempty? Pick any old `x,y in M`. Since `f` is linear and dominated by `p`,

`f(x)+f(y)=f(x+y)<=p(x+y)`.

But `x+y=(x-z)+(y+z)`, so subadditivity gives

`p(x+y)<=p(x-z)+p(y+z)`.

Therefore

`f(x)-p(x-z)<=p(y+z)-f(y)`.

Every lower bound is below every upper bound. That is the entire one-dimensional miracle. Sublinearity is not a cosmetic hypothesis; it is exactly the condition that makes the old constraints compatible when I slide one new line through the old subspace.

Now I choose any `c` in that interval and define `F(x+t z)=f(x)+t c`. This is well-defined because `z` is not in `M`, so `x+t z` has a unique representation. The inequalities I just derived give `F(u)<=p(u)` for every `u` in the enlarged subspace. If `p` is a seminorm or a norm and the original hypothesis is `|f|<=p`, then the one-sided statement is enough: apply it also to `-u`, and `-F(u)=F(-u)<=p(-u)=p(u)`, so `|F(u)|<=p(u)`. In the normed case this preserves the same norm bound. Nothing has been spent; I have paid for the new value with exactly the slack sublinearity provides.

This settles one new dimension, but it does not settle the space. If `E` were finite-dimensional over `M`, I could repeat the step finitely many times. If it were countably generated, I could try a sequence. But the theorem is supposed to ignore separability and completeness, so I cannot pretend the missing directions arrive in a list. I need a way to say: extend as far as possible, and then prove "as far as possible" means all the way.

I collect every partial success. Let `A` be the set of pairs `(N,g)` where `N` is a subspace containing `M`, `g:N->R` is linear, `g` extends `f`, and `g<=p` on `N`. Order these pairs by extension: `(N,g) <= (N',g')` if `N subset N'` and `g'` agrees with `g` on `N`. This family is not empty because `(M,f)` is in it. If I have a chain of such extensions, I take the union of the domains and define the functional by whichever member of the chain contains the point. The chain condition makes this definition unambiguous, linear, and still dominated by `p`. So every chain has an upper bound.

Zorn's lemma now gives a maximal dominated extension `(G,g)`. This is the nonconstructive leap. It is not telling me how to name the values; it is telling me that the local compatibility I proved can be pushed to a maximal consistent object even when the space has no manageable enumeration. The proof would be much weaker without this step, because the one-dimensional lemma alone only says every unfinished extension can move one step farther.

Now maximality has only one possible meaning. If `G` is not all of `E`, choose `z notin G`. The one-dimensional interval argument applies to `g` on `G`, so I can extend `g` to `G+R z` without losing domination. That strictly enlarges the maximal pair, impossible. Hence `G=E`. The global functional exists because every attempted stopping point can be defeated by the same one-dimensional calculation.

The same object has a geometric face. A dominated linear functional is a hyperplane in disguise: its kernel, or one of its level sets, is a flat separator. To separate a point from a subspace in a normed space, I take a closed subspace `M` and a point `z notin M`. The distance `d=dist(z,M)` is positive. Define a functional on `M+R z` by `f(m+t z)=t d`. This is linear, vanishes on `M`, and sends `z` to `d`. For `m in M` and `t != 0`,

`||m+t z||=|t| ||z+m/t|| >= |t| d`,

so `|f(m+t z)|=|t|d<=||m+t z||`. The extension theorem gives `F` on all of `E` with `||F||<=1`, `F|_M=0`, and `F(z)=d`. If I want the normalized separator, divide by `d`: a continuous functional is zero on `M` and equals `1` at `z`. The abstract extension has become a hyperplane that detects a point no vector construction could isolate.

For an open convex set the same thought is even clearer. A convex neighborhood of the origin carries its own gauge, the Minkowski functional. If an open convex set `A` is disjoint from a subspace `S`, choose a point and form a balanced convex neighborhood around the origin from translations of the cone generated by `A`; its gauge is a seminorm. On the span of `S` and one chosen outside point, define a functional that is zero on `S` and one on that point. The gauge inequality is arranged exactly so this functional is dominated. Extending it gives a global linear functional, and the kernel is a hyperplane containing `S` that misses the convex set. The analytic theorem is therefore not merely an estimate about functions. It says convex geometry in an arbitrary linear space is still visible through real-valued linear probes.

The discovery path has become precise. The theorem is not a mechanical recombination of basis extension and boundedness. A basis extension ignores the estimate; a pure estimate has no values outside the subspace. The new way of thinking is to treat the domination constraint as the primary object. Sublinearity turns all constraints imposed by a new direction into a nonempty interval for one scalar. Zorn's lemma turns the ability to enlarge by one direction into a maximal extension on the whole space. The resulting functional is then reinterpreted geometrically as a separating hyperplane. The proof works because it refuses to choose a whole basis first; it chooses one admissible value at a time, under the convex inequality that makes every future line remain controlled.
