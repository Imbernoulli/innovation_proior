# Hahn-Banach extension and separation

## Analytic theorem

Let `E` be a real vector space, let `p:E->R` be sublinear:

`p(x+y)<=p(x)+p(y)`, and `p(alpha x)=alpha p(x)` for `alpha>=0`.

Let `M` be a linear subspace of `E`, and let `f:M->R` be linear with

`f(x)<=p(x)` for every `x in M`.

Then there is a linear functional `F:E->R` extending `f` such that

`F(x)<=p(x)` for every `x in E`.

If `p` is a seminorm and `|f(x)|<=p(x)` on `M`, then `F` may be chosen with `|F(x)|<=p(x)` on `E`. In particular, if `E` is normed and `f` is bounded on `M`, then `f` extends to `E` with the same operator norm.

## Proof

First prove the one-step extension lemma. Suppose `z notin M`. Any extension to `M+R z` must have the form

`F(x+t z)=f(x)+t c`.

To keep `F<=p`, the scalar `c` must satisfy

`f(x)-p(x-z) <= c <= p(y+z)-f(y)` for all `x,y in M`.

This interval is nonempty. Indeed,

`f(x)+f(y)=f(x+y)<=p(x+y)=p((x-z)+(y+z))<=p(x-z)+p(y+z)`,

so `f(x)-p(x-z)<=p(y+z)-f(y)` for every pair `x,y`. Choose `c` in the interval and define `F(x+t z)=f(x)+t c`. For `t>0`, the upper bound on `c` gives `F(x+t z)<=p(x+t z)`. For `t<0`, the lower bound gives the same inequality after dividing by `t`; for `t=0`, it is the original domination of `f`. Thus the extension to `M+R z` is dominated by `p`.

Now let `A` be the set of all dominated extensions `(N,g)`, where `M subset N subset E`, `g:N->R` is linear, `g|_M=f`, and `g<=p` on `N`. Order `A` by extension. If `{(N_i,g_i)}` is a chain, then `N*=union_i N_i` is a subspace and the formula `g*(x)=g_i(x)` when `x in N_i` is well-defined, linear, extends `f`, and remains dominated by `p`. Hence every chain has an upper bound.

By Zorn's lemma there is a maximal dominated extension `(G,g)`. If `G != E`, choose `z in E\G`. The one-step lemma extends `g` to `G+R z`, contradicting maximality. Therefore `G=E`, and `g` is the required `F`.

For the seminorm version, apply the one-sided theorem to `f<=p`. Since `p(-x)=p(x)`, the extension satisfies `-F(x)=F(-x)<=p(-x)=p(x)`, hence `|F(x)|<=p(x)`.

## Norm-preserving corollary

Let `E` be a normed real vector space, `M subset E` a subspace, and `f:M->R` continuous. Put `C=||f||`. Then `|f(x)|<=C||x||`, so the seminorm version with `p(x)=C||x||` gives an extension `F:E->R` with `|F(x)|<=C||x||`. Thus `||F||<=||f||`; the reverse inequality is automatic because `F` extends `f`, so `||F||=||f||`.

## Separation corollary

Let `E` be a normed real vector space, let `M` be a closed subspace, and let `z notin M`. Set `d=dist(z,M)>0`. On `M+R z`, define

`f(m+t z)=t d`.

For `t != 0`,

`||m+t z||=|t| ||z+m/t|| >= |t| d = |f(m+t z)|`,

and the case `t=0` is trivial. Extend `f` to `F:E->R` with `||F||<=1`. Then `F(m)=0` for `m in M` and `F(z)=d`. The kernel of `F` is a closed hyperplane containing `M` but not containing `z`; after dividing by `d`, one obtains a continuous linear separator with value `1` at `z`.

This is the geometric content of the theorem: controlled extension of a functional is the same mechanism that produces separating hyperplanes for convex linear geometry.
