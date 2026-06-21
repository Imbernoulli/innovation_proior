## Research question

A continuous linear functional on `C(X, R)` is an abstract rule: it takes a continuous real-valued function on a space `X` and returns a number. In finite-dimensional linear algebra, a linear functional is a dot product with a vector. In analysis, the analogous question is sharper: when the input objects are functions, is a positive functional secretly integration against a spatial object?

The setting is a compact Hausdorff space `X` and the Banach space `C(X, R)` with the uniform norm. A functional `I:C(X, R)->R` is positive if `f>=0` pointwise implies `I(f)>=0`. Positivity is much stronger than a norm estimate: it says the functional respects the order structure of functions. The problem is to determine whether that order-respecting rule comes from a unique regular Borel measure `mu` satisfying

`I(f) = integral_X f dmu`

for every continuous `f`.

The difficulty is that a measure is normally defined on sets, while `I` only sees continuous functions. Indicator functions of rough sets are not continuous. A solution has to recover set mass using only continuous tests, explain why open and compact approximation are the right replacements for indicators, and prove that the resulting set function is countably additive and regular.

## Background

The space `C(X, R)` carries both linear structure and order. If `I` is positive, then `-||f||_infty 1 <= f <= ||f||_infty 1`, so positivity forces `|I(f)| <= I(1)||f||_infty`. Thus positivity already implies continuity and gives `||I||=I(1)` when `I` is nonzero. This is the first sign that an order condition can encode analytic boundedness.

Measures provide the obvious examples. If `mu` is a finite regular Borel measure on `X`, then `f |-> integral f dmu` is positive and continuous. The hard direction is the converse: starting only from `I`, define the mass of a set without applying `I` to its indicator.

Compact Hausdorff spaces supply the topological separation needed to approximate indicators. Urysohn-type bump functions allow continuous functions `f` with `1_K <= f <= 1_U` whenever `K` is compact, `U` is open, and `K subset U`. These functions do not equal indicators, but positivity makes inequalities between functions meaningful after applying `I`.

Regularity is the natural language for this recovery. Outer regularity says a set is measured from open neighborhoods:

`mu(E) = inf { mu(U) : E subset U, U open }`.

Inner regularity says it is measured from compact subsets:

`mu(E) = sup { mu(K) : K subset E, K compact }`.

Together they say that rough measurable sets are determined by topologically tame sets. This matches the fact that continuous functions can be made to live between compact interiors and open exteriors, but not on arbitrary pointwise boundaries.

## Baselines

- **Point evaluation.** For each `x in X`, the rule `I(f)=f(x)` is positive and continuous. It behaves like a unit mass at one point. Gap: point evaluations explain atoms but not spread-out mass or arbitrary positive functionals.

- **Finite weighted sums.** Rules of the form `I(f)=sum_i a_i f(x_i)` with `a_i>=0` are positive and continuous. They correspond to finite atomic measures. Gap: many natural functionals average over continua and cannot be reduced to finitely many samples.

- **Riemann and Riemann-Stieltjes integration on intervals.** On `[a,b]`, continuous functions can be integrated against length or against a function of bounded variation. Gap: the interval order and partition structure are special; a compact Hausdorff space may have no coordinates, no intervals, and no canonical mesh.

- **Hahn-Banach duality.** General bounded linear functionals can be extended and separated by abstract linear methods. Gap: those methods preserve norm bounds but do not reveal a positive set function or explain why continuous tests should determine measurable geometry.

- **Finitely additive set functions.** One can try to assign values to sets directly and integrate simple functions. Gap: without countable additivity and regularity, different set functions can agree on continuous tests, and pathological boundary behavior is invisible to `C(X, R)`.

- **Daniell-style integration.** One can start from a positive functional on a lattice of functions and extend it to a measure. Gap: the theorem still has to identify which sets are recovered and why topological regularity is forced by the continuous-function domain.

## Evaluation settings

The artifact is a theorem and proof. The main setting is a compact Hausdorff space `X`, the real Banach lattice `C(X, R)`, and a positive continuous linear functional `I`.

The proof should recover a finite regular Borel measure `mu`, prove uniqueness, and show `I(f)=integral f dmu` for every continuous `f`. It should also record the norm relation `||I||=mu(X)=I(1)`.

Stress cases include point masses, finite atomic measures, Lebesgue measure on a compact interval, probability measures on compact metric spaces, measures concentrated on closed nowhere dense sets, and sets whose boundaries cannot be detected by a single continuous function.

The natural failure modes are also important. If positivity is removed, a signed measure or Jordan decomposition is needed. If regularity is removed, nonunique Borel measures can induce the same values on continuous functions. If compactness or local compactness is removed, continuous compact support and finiteness on compact sets become part of the correct statement.

## Code framework

The artifact is a theorem and proof, not a computational method. The proof scaffold is:

1. Use positivity to turn order inequalities in `C(X, R)` into norm control.
2. Define the mass of an open set by taking the supremum of `I(f)` over continuous `0<=f<=1` whose support is contained in that open set.
3. Extend the set function to arbitrary sets by outer approximation with open sets.
4. Use Urysohn bump functions and compactness to prove countable additivity on Borel sets and inner/outer regularity.
5. Recover `I(f)` by approximating continuous functions through level-set simple functions and continuous cutoffs.
6. Prove uniqueness by approximating indicators of compact and open sets with continuous functions, then using regularity.
