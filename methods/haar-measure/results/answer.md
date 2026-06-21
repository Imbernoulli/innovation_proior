# Haar measure

## Theorem

Let `G` be a locally compact Hausdorff topological group. There exists a nonzero regular Borel measure `mu` on `G`, finite on compact sets and positive on nonempty open sets, such that

`mu(xE) = mu(E)`

for every `x in G` and every Borel set `E`. If `nu` is another such left-invariant regular Borel measure, then there is a constant `c > 0` such that

`nu = c mu`.

Thus a locally compact group carries a canonical left-invariant notion of volume, unique up to the choice of unit.

## Proof artifact

Work first with functions. Let `C_c(G)` be the space of continuous compactly supported functions, and let

`(L_x f)(t) = f(x^{-1}t)`.

For nonnegative `f,g in C_c(G)` with `g` nonzero, define a covering gauge

`[f:g] = inf sum_i c_i`,

where the infimum is over all finite families `c_i > 0`, `x_i in G` satisfying

`f <= sum_i c_i L_{x_i}g`.

The quantity is finite because `g` is positive on some neighborhood and the compact set `supp(f)` is covered by finitely many left translates of that neighborhood. It is left invariant in the first argument:

`[L_y f:g] = [f:g]`.

Fix `f_0 in C_c(G)^+`, `f_0 != 0`, and normalize

`I_g(f) = [f:g] / [f_0:g]`.

As the support of `g` is restricted to smaller neighborhoods of the identity, compactness and uniform continuity force the normalized gauges to become additive on nonnegative test functions in the limit. A cluster point gives a positive linear functional

`I:C_c(G) -> C`

with `I(f_0)=1` and

`I(L_x f)=I(f)`

for all `x in G`, `f in C_c(G)`.

By the Riesz representation theorem for locally compact Hausdorff spaces, there is a regular Borel measure `mu`, finite on compact sets, such that

`I(f)=int_G f dmu`

for every `f in C_c(G)`. Invariance of `I` under every `L_x` implies invariance of `mu` under every left translation. The measure is nonzero because `I(f_0)=1`, and nonempty open sets have positive mass by testing against nonzero nonnegative compactly supported functions inside them.

For uniqueness, let `mu` and `nu` be two nonzero regular left-invariant Borel measures. Their integrals over `C_c(G)` are positive left-invariant linear functionals. The same compact-covering comparison used above shows that, after fixing a nonzero `f_0 >= 0`,

`int f dnu / int f_0 dnu = int f dmu / int f_0 dmu`

for every nonnegative `f in C_c(G)`. Hence

`int f dnu = c int f dmu`

for all `f in C_c(G)`, where

`c = (int f_0 dnu)/(int f_0 dmu)`.

Regularity then implies `nu=c mu` on Borel sets.

## Modular function caveat

For a left-invariant measure `mu` and `a in G`, the right translate

`E -> mu(Ea)`

is again a left-invariant regular Borel measure. By uniqueness it must be a scalar multiple of `mu`. With the convention

`mu(Ea) = Delta(a)^{-1} mu(E)`,

the scalar defines a continuous homomorphism

`Delta:G -> (0,infty)`.

This `Delta` is the modular function. The group is unimodular exactly when `Delta(a)=1` for all `a`, in which case a left Haar measure is also right invariant. In general the canonical guarantee is left-invariant integration; two-sided invariance is an additional property.
