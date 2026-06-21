# The positive Riesz representation theorem

Let `X` be a compact Hausdorff space, and let `C(X, R)` be the real Banach space of continuous functions on `X` with the uniform norm. If

`I:C(X, R)->R`

is a positive linear functional, meaning `f>=0` implies `I(f)>=0`, then there is a unique finite regular Borel measure `mu` on `X` such that

`I(f) = integral_X f dmu`

for every `f in C(X, R)`. Moreover,

`||I|| = I(1) = mu(X)`.

Conversely, every finite regular Borel measure on `X` defines a positive continuous linear functional by integration.

## Proof

First, positivity implies continuity. For every `f in C(X, R)`,

`-||f||_infty 1 <= f <= ||f||_infty 1`.

Applying `I` gives

`|I(f)| <= I(1)||f||_infty`.

Hence `||I|| <= I(1)`, and since `||1||_infty=1`, the reverse inequality `I(1)<=||I||` is immediate. Thus `||I||=I(1)`.

Define the mass of an open set `U subset X` by

`mu(U) = sup { I(f) : f in C(X, R), 0 <= f <= 1, supp(f) subset U }`.

For an arbitrary set `E subset X`, define the outer value

`mu*(E) = inf { mu(U) : E subset U, U open }`.

The usual Urysohn and compactness argument gives that `mu*` is an outer measure and that every open set is `mu*`-measurable. The key points are these:

- If a continuous `f` is supported in `union_n U_n`, compactness of `supp(f)` reduces the cover to finitely many `U_n`; a finite partition of unity subordinate to that cover decomposes `f` into continuous nonnegative pieces supported in the individual `U_n`. Positivity and linearity give countable subadditivity.
- If `K subset U` with `K` compact and `U` open, Urysohn's lemma gives `g in C(X, R)` with `1_K <= g <= 1_U`. These cutoffs separate disjoint regions well enough to prove Caratheodory measurability of open sets.

Restrict `mu*` to the Borel sets and write the resulting Borel measure as `mu`. It is finite because `mu(X)=I(1)`.

The construction is regular. Outer regularity is built into the definition:

`mu(E) = inf { mu(U) : E subset U, U open }`

for every Borel set `E`. For open `U`,

`mu(U) = sup { mu(K) : K subset U, K compact }`.

Indeed, every admissible continuous test inside `U` has compact support `K subset U`, and for every open `V superset K` the same test is admissible for `V`, so `I(f)<=mu(K)`. The reverse inequality is monotonicity. Since `X` is compact and `mu(X)<infty`, inner regularity for all Borel sets follows by applying outer regularity to complements: if `E` is Borel and `epsilon>0`, choose open `V superset X\E` with `mu(V)<=mu(X\E)+epsilon`; then `K=X\V` is compact, `K subset E`, and `mu(K)>=mu(E)-epsilon`.

It remains to show that `I` is integration against `mu`. For `f>=0`, partition the interval `[0, ||f||_infty]` into subintervals of mesh at most `epsilon`. The lower and upper level sets of `f` give simple functions that differ from `f` by at most `epsilon`. Regularity and Urysohn cutoffs replace the indicators of these level sets by continuous functions squeezed between compact inner sets and open outer sets. Positivity then squeezes `I(f)` between the corresponding lower and upper simple integrals, while the measure integral is squeezed by the same bounds. Letting the mesh and regularity errors go to zero yields

`I(f) = integral_X f dmu`

for all nonnegative continuous `f`. For a general real-valued continuous `f`, write `f=f_+-f_-` and use linearity.

Finally, uniqueness follows from the same continuous approximation of indicators. If `mu` and `nu` are finite regular Borel measures such that

`integral f dmu = integral f dnu`

for every `f in C(X, R)`, let `K subset U` with `K` compact and `U` open. Choose `g` with `1_K <= g <= 1_U`. Then

`mu(K) <= integral g dmu = integral g dnu <= nu(U)`.

Taking the infimum over open `U superset K` gives `mu(K)<=nu(K)`, and symmetry gives equality on compact sets. Inner regularity gives equality on open sets, and outer regularity gives equality on all Borel sets. Thus the representing measure is unique.

The theorem says precisely that a positive continuous linear functional on continuous functions is not merely an abstract element of the dual space. It is integration against a unique regular Borel measure, and that measure is recovered by testing the space with continuous functions.
