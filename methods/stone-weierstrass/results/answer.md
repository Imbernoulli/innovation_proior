# Stone-Weierstrass Theorem

## Statement

Let `X` be a compact Hausdorff space, and let `A` be a real subalgebra of `C(X,R)` under pointwise operations. Suppose `A` contains the constant functions and separates points: for every `x != y` in `X`, there is `h in A` with `h(x) != h(y)`. Then `A` is dense in `C(X,R)` in the uniform norm.

Equivalently, for every `f in C(X,R)` and every `epsilon > 0`, there exists `g in A` such that

`||f-g||_infty < epsilon`.

For complex-valued functions, the corresponding statement holds for complex subalgebras of `C(X,C)` that contain constants, separate points, and are closed under complex conjugation.

## Proof

Let `B` be the uniform closure of `A` in `C(X,R)`. It is enough to show `B=C(X,R)`. The closure `B` is still a real algebra, contains constants, and separates points.

First, `B` is closed under absolute values. If `u in B`, compactness gives `|u(x)| <= M` on `X` for some `M`. By the classical Weierstrass approximation theorem on `[-M,M]`, there are real polynomials `p_n` converging uniformly to `|t|` on that interval. Since `B` is an algebra containing constants, `p_n(u) in B`; since `B` is closed, `|u| in B`.

Therefore `B` is a lattice:

`max(u,v) = (u+v+|u-v|)/2 in B`,

`min(u,v) = (u+v-|u-v|)/2 in B`.

Now fix `f in C(X,R)` and `epsilon > 0`, and put `delta=epsilon/2`. For each pair `x,y in X`, construct `g_{xy} in B` with

`g_{xy}(x)=f(x)` and `g_{xy}(y)=f(y)`.

If `x=y`, take the constant function `f(x)`. If `x != y`, choose `h in B` with `h(x) != h(y)` and set

`g_{xy}(z)=f(x)+(f(y)-f(x))(h(z)-h(x))/(h(y)-h(x))`.

Fix `x`. For each `y`, continuity gives a neighborhood `V_y` of `y` such that

`g_{xy}(z) >= f(z)-delta`

on `V_y`. Finitely many `V_{y_1},...,V_{y_m}` cover `X`. Define

`G_x=max(g_{x y_1},...,g_{x y_m})`.

Then `G_x in B`, `G_x >= f-delta` on `X`, and `G_x(x)=f(x)`.

Continuity gives a neighborhood `U_x` of `x` such that

`G_x(z) <= f(z)+delta`

on `U_x`. Finitely many `U_{x_1},...,U_{x_k}` cover `X`. Define

`G=min(G_{x_1},...,G_{x_k})`.

Then `G in B`. Since each `G_{x_i} >= f-delta`, also `G >= f-delta`. For any `z in X`, choose `i` with `z in U_{x_i}`; then

`G(z) <= G_{x_i}(z) <= f(z)+delta`.

Thus `|G(z)-f(z)| <= delta < epsilon` for all `z`, so `f` is in the uniform closure of `B`. Because `B` is closed, `f in B`. Since `f` was arbitrary, `B=C(X,R)`, hence `A` is dense.

For the complex version, let `A` be a complex subalgebra closed under conjugation. Its real-valued self-adjoint part contains constants and separates points: if `a(x) != a(y)`, either `Re a` or `Im a` separates them, and both lie in `A` by conjugation. The real theorem approximates the real and imaginary parts of any `F in C(X,C)`, and recombining the two approximants gives uniform approximation of `F`.
