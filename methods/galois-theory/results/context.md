# Context: the conditions for solubility of equations by radicals

## Research question

Given a polynomial equation
$$x^n + a_1 x^{n-1} + a_2 x^{n-2} + \cdots + a_n = 0,$$
can its roots always be written as a *formula in radicals* — a finite expression built from the coefficients $a_i$ using addition, subtraction, multiplication, division, and the extraction of $m$-th roots? For $n=2,3,4$ the answer is a famous yes. For $n=5$ it has been open for two and a half centuries. The question is twofold: (i) **for which individual equations** does such a formula exist, and (ii) **is there a single general formula** valid for every equation of degree $n$, the way the quadratic formula handles every quadratic? The lower-degree solutions were each found by a different clever substitution. What decides, given an equation, whether the search for a radical formula will succeed?

## Background

**The classical solutions.** The quadratic was solved in antiquity by completing the square. The cubic $x^3+px+q=0$ was solved by del Ferro, Tartaglia, and Cardano (*Ars Magna*, 1545): set $x=u+v$, impose $3uv=-p$, and $u^3,v^3$ are the two roots of a quadratic (the *resolvent*), giving
$$x = \sqrt[3]{-\tfrac q2 + \sqrt{\tfrac{q^2}4+\tfrac{p^3}{27}}} + \sqrt[3]{-\tfrac q2 - \sqrt{\tfrac{q^2}4+\tfrac{p^3}{27}}}.$$
The quartic was solved by Ferrari (published by Cardano): introduce a parameter, complete a square on both sides, and the condition for the right-hand side to be a perfect square is a *resolvent cubic*. In each case a higher equation is solved by descending through an auxiliary equation of lower degree.

**Lagrange's unification (1770–71).** Lagrange asked what these tricks have in common. His answer: each works by forming a function of the roots whose values, as the roots are permuted, are *fewer than $n!$*, and which therefore satisfies an equation of that lower degree. For the cubic with roots $x_1,x_2,x_3$ and $\omega$ a primitive cube root of unity, the *Lagrange resolvent*
$$t = (x_1 + \omega x_2 + \omega^2 x_3)^3$$
takes only $2$ distinct values as the $6$ permutations of the three roots are applied (the even permutations fix it, the odd ones swap it with its partner), so $t$ is a root of a quadratic — the resolvent quadratic. The quartic similarly yields a function taking $3$ values, hence a resolvent cubic. The pattern that makes a degree drop is a function whose stabilizer under permutation is large. Applied to the quintic, the natural resolvent function takes $6$ distinct values: the auxiliary equation has degree $6$. The construction locates the matter in the *permutations of the roots* and the number of values a function takes under them.

**Permutations of the roots as the carrier of the difficulty.** Lagrange's resolvents made the count "how many values does a function of the roots take under permutations" the governing quantity. Cauchy (1815) developed the calculus of permutations (substitutions) systematically — composition, cycles, the result that the number of distinct values of a function of $n$ letters divides $n!$ and obeys strong divisibility constraints. So: the roots themselves are inaccessible, but the *symmetry among them* is a finite combinatorial object one can compute with, and the classical solvability of $n\le 4$ is a fact about that symmetry.

**The notion of "rational" is relative to what is adjoined.** A polynomial is reducible if it factors over the coefficient field; irreducible otherwise. Reducibility depends on which quantities are treated as known. Adjoining a quantity (declaring it known) can make an irreducible polynomial reducible — for example, when a root of one of Gauss's cyclotomic auxiliary equations $\frac{x^n-1}{x-1}=0$ ($n$ prime) is adjoined, that equation splits into factors. So "rational" means *expressible as a rational function of the coefficients together with whatever has been adjoined*, and solving an equation is the act of enlarging the field of known quantities, one radical at a time, until the roots themselves become rational. Gauss (1801) had shown that the cyclotomic equation is always solvable by radicals, by exploiting the cyclic structure of the powers of a primitive root.

## Baselines

**Cardano–Ferrari resolvent method (cubic, quartic).** Core idea: descend to a lower-degree auxiliary equation by forming an algebraic combination of the roots that the symmetric functions can reach. A collection of degree-specific substitutions yielding the explicit cubic and quartic formulas.

**Lagrange resolvents (1770).** Core idea: every classical solution is a resolvent whose degree equals the number of distinct values of a root-function under permutation; relate solvability to permutation groups of the roots. The mechanism: choose $\rho = x_1 + \omega x_2 + \cdots + \omega^{n-1} x_n$ with $\omega$ a primitive $n$-th root of unity; its $n$-th power has a stabilizer in $S_n$, and the orbit size is the resolvent degree. For $n\le 4$ the resolvent degree is below $n$; for $n=5$ it is $6$.

**Ruffini's impossibility argument (1799).** Core idea: assuming a radical solution for $n=5$, study the permutations that the radical expression admits and derive a contradiction; he worked with the action of permutations on the tower of radical extensions and used divisibility properties of the order of permutation groups. The argument ran to roughly 500 pages and assumed that the auxiliary radicals in a solution can be replaced by natural irrationalities attached to the roots, with the needed roots of unity included.

**Abel's impossibility proof (1824–26).** Core idea: suppose the general quintic is solvable; build the field by adjoining radicals of prime exponent one at a time; at the step where the polynomial first becomes reducible, the adjoined radical $\rho=\sqrt[p]{\eta}$ has relative degree exactly $p$ over the previous field, and once the relevant roots of unity are present its conjugates are $\rho,\varepsilon\rho,\varepsilon^2\rho,\dots$; tracking the resulting factorizations and the permutations they impose on the five roots produces a contradiction. Abel also established the natural-irrationalities reduction: the auxiliary algebraic quantities can be taken from the root field together with the needed roots of unity. The mechanism is an induction over the radical tower with prime-exponent steps and conjugate-radical bookkeeping; it concerns the *general* quintic.

**Gauss's cyclotomic solution (1801).** Core idea: the equation $\frac{x^p-1}{x-1}=0$ for prime $p$ is solvable by radicals because the $p-1$ primitive roots are the powers of one of them, and their permutation structure is cyclic of order $p-1$; nesting "periods" of the cyclic structure gives the radical formula.

## Evaluation settings

The natural objects against which a theory of solubility would be tested are the classical worked cases and the resistant ones, all of which predate any general criterion: the general quadratic, cubic, and quartic (which must come out *solvable*, recovering the Cardano–Ferrari formulas as special cases); the general quintic and higher (the standing open challenge); Gauss's cyclotomic equations $\frac{x^p-1}{x-1}=0$, $p$ prime (known solvable, the cyclic prototype); the binomial equations $x^n - a = 0$ (solvable once roots of unity are available); and the modular equations of elliptic-function theory (a hard family of specific equations whose solvability is in question). The yardsticks are: does the criterion certify each classical solution as solvable, does it draw the line at degree $5$, and can it decide an *individual* equation (for instance one of prime degree) rather than only the generic one. The arithmetic tools available for the test are factorization over $\mathbb{Q}$, the Eisenstein-type irreducibility checks, root-counting via the intermediate value theorem to locate real versus complex roots, and Cauchy's permutation-divisibility constraints.

## Code framework

```
# Inputs and primitives available at the start:
#   - a base field F of currently-known quantities: rational functions of the
#     coefficients together with adjoined quantities.
#   - a polynomial f(x) in F[x], assumed without repeated roots.
#   - permutations of the roots (substitutions, after Lagrange/Cauchy); the count of
#     distinct values a function of the roots takes under them divides n!.
#   - Gauss: x^p - 1 and binomial x^n - a are handled once nth roots of unity are present.

def base_field_of_known_quantities(coeffs, adjoined):
    # rational functions of the coefficients and the adjoined quantities
    pass  # the relative notion of "rational"

def criterion_for_solvability_by_radicals(f, F):
    # TODO: a necessary-and-sufficient condition, defined purely from f and F,
    #       that decides whether f is solvable by radicals.
    pass

def verdict_for_the_general_quintic():
    # TODO: apply the criterion to the general degree-5 equation.
    pass

# A single witness equation, to be checked once the criterion exists:
def witness_quintic():
    # an explicit degree-5 polynomial over Q to test the verdict against.
    pass
```
