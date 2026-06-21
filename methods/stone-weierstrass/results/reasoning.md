I start with the old theorem, but I try not to stare at its surface. On an interval, every continuous function can be uniformly approximated by polynomials. The familiar way to think about that is constructive: build Bernstein polynomials, or smooth with a kernel, or use the coordinate on the line. That way of thinking explains `[a,b]`, but it traps the result inside `[a,b]`. If I move to a compact space with no preferred coordinate, there is no monomial `x^n`, no equally spaced mesh, no order relation to exploit. So if the interval theorem is hiding a more general principle, the principle cannot be "use these particular polynomials." It has to be "this kind of family is already large enough."

What kind of family are polynomials, before they are polynomials? They form a real vector space of continuous functions, they contain constants, and they are closed under pointwise multiplication. So they are an algebra inside `C([a,b], R)`. They also distinguish points: if `x != y`, then the coordinate function itself takes different values at `x` and `y`. That last property is clearly necessary for density. If every function in my family has `u(x)=u(y)`, then every uniform limit also has equal values at `x` and `y`; I can never approximate a continuous function that separates those two points. Constants fail for exactly this reason. The closed algebra `{u in C(X): u(x)=u(y)}` fails for exactly this reason. So point separation is not decorative. It is the first obstruction visible to the uniform norm.

The question is whether there is any obstruction after that. This is where the usual constructive reflex gets in the way. If I try to write down approximants directly, I keep asking for coordinates and formulas I do not have. Instead I should look at a hypothetical closed counterexample. Let `A` be my algebra, and let `B` be its uniform closure in `C(X, R)`. The closure is still a real algebra: addition, scalar multiplication, and multiplication pass to uniform limits because multiplication is continuous under the uniform norm on bounded functions. It still contains constants and still separates points. If I can prove every closed algebra with these properties is already all of `C(X, R)`, I am done.

So what does a closed real algebra know how to do? It knows polynomials in its own functions. If `u in B` and `q` is a real polynomial, then `q(u) in B`, because `q(u)` is built from constants, sums, scalar multiples, and products. That sounds small until I remember the one scalar function from the old theorem that really matters: `t -> |t|`. Since `X` is compact, `u(X)` is contained in some interval `[-M,M]`. The classical Weierstrass theorem gives polynomials `q_n` that converge uniformly to `|t|` on `[-M,M]`. Then `q_n(u)` lies in `B` for every `n`, and `q_n(u)` converges uniformly on `X` to `|u|`. Because `B` is closed, `|u|` lies in `B`.

That is the first real change in viewpoint. I am not using Weierstrass to approximate the target function on `X`. I am using it once, in one real variable, to give every closed real algebra an internal absolute-value operation. The concrete interval theorem becomes a piece of algebraic functional calculus.

Once `B` contains absolute values, it contains pointwise maxima and minima. Indeed,

`max(u,v) = (u+v+|u-v|)/2`

and

`min(u,v) = (u+v-|u-v|)/2`.

So `B` is not just an algebra anymore; it is a lattice of continuous functions. This is the missing nonlinear operation. Linear combinations can tilt and shift functions, products can create polynomial expressions, but max and min let me paste upper and lower envelopes without leaving the closed algebra. That is exactly the kind of operation a coordinate-free proof needs.

Now I bring in point separation and ask what it can buy locally. Take an arbitrary target `f in C(X, R)` and two points `x,y in X`. If `x=y`, the constant function `f(x)` matches `f` at that point. If `x != y`, separation gives `h in B` with `h(x) != h(y)`. Since constants and scalar multiples are in `B`, the affine expression

`g_{xy}(z) = f(x) + (f(y)-f(x)) (h(z)-h(x))/(h(y)-h(x))`

also lies in `B`, and it satisfies `g_{xy}(x)=f(x)` and `g_{xy}(y)=f(y)`. So the algebra can interpolate the target at any pair of points. This is weaker than uniform approximation, but it is exactly the local brick I need. The old constructive proofs try to build one global approximant all at once. Here I can build many two-point witnesses and then use the lattice operations to assemble them.

Fix `x`. For each `y`, I have a function `g_{xy}` matching `f` at `x` and at `y`. Since `g_{xy}(y)=f(y)` and both functions are continuous, there is a neighborhood `V_y` of `y` on which

`g_{xy}(z) > f(z) - epsilon`

or, if I prefer non-strict notation, `g_{xy}(z) >= f(z)-epsilon` after shrinking the tolerance at the start. The neighborhoods `V_y` cover `X`. Compactness now enters in its most concrete form: finitely many of them, say `V_{y_1},...,V_{y_m}`, cover `X`. Take the pointwise maximum

`G_x = max(g_{x y_1},...,g_{x y_m})`.

Because `B` is a lattice, `G_x in B`. It has two properties. First, it lies above `f-epsilon` everywhere, because every point belongs to some `V_{y_j}` and the corresponding `g_{x y_j}` is high enough there. Second, at the fixed point `x`, all the functions `g_{x y_j}` equal `f(x)`, so their maximum also equals `f(x)`. I have built, for this one `x`, a global lower support: `G_x >= f-epsilon` on all of `X`, while `G_x(x)=f(x)`.

That is only half the squeeze. Since `G_x(x)=f(x)` and both functions are continuous, there is a neighborhood `U_x` of `x` on which

`G_x(z) < f(z) + epsilon`.

The neighborhoods `U_x` cover `X`, so compactness gives a finite subcover `U_{x_1},...,U_{x_k}`. Now take the pointwise minimum

`G = min(G_{x_1},...,G_{x_k})`.

Again `G in B`. Since every `G_{x_i}` is everywhere at least `f-epsilon`, their minimum is also at least `f-epsilon`. For the upper bound, take any point `z`. It lies in some `U_{x_i}`, and there `G_{x_i}(z) <= f(z)+epsilon`; because `G(z)` is the minimum of all the `G_{x_j}(z)`, it is no larger than that particular one. Therefore

`f(z)-epsilon <= G(z) <= f(z)+epsilon`

for every `z in X`. The uniform error is at most `epsilon`. Since `epsilon` was arbitrary and `G` lies in the closed algebra `B`, the target `f` lies in `B`. The target was arbitrary, so `B=C(X,R)`.

The proof now looks inevitable, but the inevitability only appears after changing the question. The old interval theorem asks, "Which polynomial sequence approximates this function?" The abstract proof asks, "What operations must any closed algebra already contain?" That shift is why the result is not a mechanical repackaging of Weierstrass. The central object is not a special formula for approximants; it is the closure of a function algebra. Once I inspect that closure, the one-dimensional approximation theorem is demoted to a tool for manufacturing `|u|`, and `|u|` unlocks max and min. Max and min unlock finite patching. Finite patching turns point separation from a necessary obstruction into a complete criterion.

This also explains why the complex case needs extra care. A complex algebra can separate points and still carry a hidden analytic rigidity: on the closed disk, functions continuous on the disk and holomorphic in the interior form a closed proper algebra that separates points. The real argument used `|u|`, `max`, and `min`, all order operations. Complex functions have no pointwise order compatible with the algebra. If the complex algebra is closed under conjugation, then real and imaginary parts of its functions stay inside the algebra, so the real theorem can be applied to the self-adjoint real-valued part. Then the real and imaginary parts of an arbitrary continuous complex function can be approximated separately and recombined. Without conjugation, separation alone misses the analytic obstruction.

So the final statement is clean. For compact Hausdorff `X`, a real subalgebra of `C(X,R)` that contains constants and separates points is uniformly dense in `C(X,R)`. For complex-valued functions, the same conclusion holds when the subalgebra also contains conjugates. The path to the theorem is the scientist's move: stop treating approximation as a search for the right explicit polynomial, and instead look for the invariant obstruction inside closed function algebras. The answer is that in the real compact setting there is only one obstruction, failure to distinguish points; the algebra and lattice structure remove the rest.
