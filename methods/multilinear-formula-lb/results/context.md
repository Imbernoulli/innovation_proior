# Context: lower bounds for arithmetic formulas computing the permanent and determinant

## Research question

Fix a field F. An *arithmetic formula* over input variables x_1,...,x_m is a binary tree whose edges point toward the root; each leaf is labelled by an input variable or a field constant, and each internal node is a `+` gate or a `×` gate. The formula computes a polynomial in F[x_1,...,x_m] in the obvious way, and its *size* |φ| is the number of nodes. The permanent and the determinant of an n×n matrix of variables X = (x_{i,j}) are the two most-studied polynomials in this model. The smallest known formula for the determinant has size n^{O(log n)}; the smallest known formula for the permanent has size O(n²·2ⁿ). Whether *polynomial-size* formulas exist for either is a central open problem.

The brutal obstacle is that no super-polynomial size lower bound is known for general arithmetic formulas for **any** explicit polynomial. The best general lower bound for the permanent or determinant is Ω(n³). A solution worth having must therefore either crack the general problem (out of reach) or isolate a *natural restricted class* of formulas — one that the obvious, hand-written formulas for the permanent and determinant actually live in — and prove a super-polynomial lower bound there. The goal is an unconditional super-polynomial lower bound for such a class, over an arbitrary field, for the permanent and the determinant.

## Background

**The model and its complexity measures.** A formula is a *tree* (each gate feeds exactly one parent), so it cannot reuse intermediate results the way a *circuit* (a DAG) can. The determinant is computable by polynomial-size *circuits* (it is in the class VP, via Gaussian-elimination-style or Csanky/Berkowitz parallel algorithms, which also give it circuits of depth O(log² n), hence formulas of size n^{O(log n)}). The permanent is believed to require exponential-size circuits (VNP-completeness, Valiant). Formula size sits between these and is poorly understood.

**Multilinearity.** A polynomial is *multilinear* if every variable has degree at most one in every monomial. Both the permanent ∑_{σ∈S_n} ∏_i x_{i,σ(i)} and the determinant ∑_σ sgn(σ) ∏_i x_{i,σ(i)} are multilinear in the entries x_{i,j}. The standard, human-written formulas for them — cofactor/Laplace expansion, Ryser-type expansions — never raise an intermediate sub-expression to a power and then cancel it: every *sub-formula* of these formulas already computes a multilinear polynomial. Call a formula *multilinear* when the polynomial computed at every one of its gates is multilinear. This is a genuine restriction: a multilinear formula is forbidden from using a high intermediate power of a variable that is supposed to cancel out before the end. For many multilinear target functions, a non-multilinear formula is highly counter-intuitive, so the multilinear restriction captures "the natural way to compute a multilinear polynomial."

**What was known about lower bounds (the diagnostic landscape).**
- For *general* formulas: no super-polynomial bound for any explicit function; for the permanent/determinant the record is Ω(n³).
- For *multilinear* formulas specifically: **nothing** is known — not even for constant depth. The only related fact is an exponential lower bound for a *weaker* model, *set-multilinear* formulas of constant depth, where the variables are pre-partitioned into sets (e.g., the rows of the matrix) and every monomial of every gate uses exactly one variable from each set it depends on.
- For other restricted models, super-polynomial / exponential bounds were known: non-commutative formulas; depth-3 circuits over finite fields; monotone circuits.

**The recurring failure mode that frames the problem.** Every successful lower bound in this area assigns to a polynomial a *complexity measure* — a number, low for everything a small formula can compute and high for the target — and bounds how fast that measure can grow across `+` and `×` gates. The art is choosing a measure that (i) grows slowly along a small formula and (ii) is provably large for the permanent and determinant. The measures used for the restricted successes above either are tailored to non-commutativity, or to bounded depth, or are simply too large on multilinear formulas to separate them from the permanent/determinant. Finding a measure that stays small along arbitrary-depth multilinear formulas yet is large for the permanent and determinant is exactly the open problem.

## Baselines

**Nisan's non-commutative lower bound (Nisan 1991).** In the *non-commutative* world, where x y ≠ y x, Nisan introduced a coefficient matrix. For a homogeneous degree-r polynomial f, and a split point k, let M_k(f) be the n^k × n^{r−k} matrix whose ((i_1,...,i_k),(j_1,...,j_{r−k})) entry is the coefficient of the monomial x_{i_1}···x_{i_k}·x_{j_1}···x_{j_{r−k}}. Because every non-commutative monomial of degree r factors *uniquely* into a degree-k prefix times a degree-(r−k) suffix, this matrix is well defined and behaves cleanly: if an arithmetic branching program for f has t vertices in layer k, then writing f = ∑_{i=1}^t h_i g_i as a sum of t rank-one outer products gives rank(M_k(f)) ≤ t. A polynomial whose M_k has rank 2^{Ω(n)} therefore needs exponential-size non-commutative formulas, and Nisan exhibits such polynomials (including the non-commutative permanent and determinant). **Limitation.** The whole construction rests on non-commutativity: unique factorization of monomials is what makes M_k well defined and what makes rank multiply across products. In the commutative world the analogous matrix is no longer controlled by formula size — a formula of linear size can make it full rank — so the method does not transfer to commutative (let alone multilinear) formulas as stated.

**The partial-derivatives method (Nisan–Wigderson 1997).** For a polynomial f, let Dim(f) be the dimension of the vector space spanned by *all* partial derivatives of f, of all orders (including f itself). This measure is sub-additive across `+` (Dim(g+h) ≤ Dim(g)+Dim(h)) and sub-multiplicative across `×` (Dim(g·h) ≤ Dim(g)·Dim(h)), so it can be bounded along a circuit, and it is provably small for homogeneous depth-3 circuits and for constant-depth set-multilinear formulas, yielding lower bounds there. This is also the work in which multilinear formulas were formally defined. **Limitation.** For multilinear formulas, the dimension of the span of all partial derivatives can be enormous — even for a formula of *linear* size — and can be far larger than the corresponding dimension for the permanent or determinant. So this measure, applied directly, cannot separate small multilinear formulas from the permanent/determinant: it is large on both. A different measure is needed for the multilinear setting.

**Kalorkoti's lower bound (Kalorkoti 1985).** For general arithmetic formulas, Kalorkoti proved Ω(n³) for the determinant using an algebraic analogue of Nechiporuk's argument: partition the variables into blocks X_1,...,X_t and lower-bound the formula size by ∑_i td_{X_i}(f), where td is a transcendence-degree-based measure that is sub-additive over both `+` and `×`. **Limitation.** For any partition of the n² determinant variables, ∑_i td_{X_i}(DET) ≤ n⁴, so this method is provably incapable of proving a super-polynomial bound; it tops out at a fixed polynomial.

**Random restrictions (Furst–Saxe–Sipser; Håstad).** In Boolean circuit complexity, randomly fixing most input variables ("restriction") simplifies a shallow circuit while keeping a hard function hard, which is how AC⁰ lower bounds are proved. **Limitation / what it leaves open in the algebraic setting.** A restriction useful here must preserve the multilinear structure and must interact with whatever coefficient/rank measure is used — an off-the-shelf Boolean restriction does neither, so adapting the restriction idea to arithmetic multilinear formulas is not automatic.

## Evaluation settings

This is a theoretical question; the "yardstick" is what kind of statement counts as progress and how it is judged.
- **Target polynomials:** the permanent and the determinant of an n×n symbolic matrix X = (x_{i,j})_{i,j∈[n]}, n² variables.
- **Model:** multilinear arithmetic formulas (binary `+`/`×` trees, every gate computing a multilinear polynomial), over an arbitrary field F.
- **Quantity measured:** formula size |φ| as a function of n; the aim is a super-polynomial lower bound, i.e. n^{ω(1)}.
- **Standard of rigor:** an unconditional proof, holding over every field, for both polynomials. The natural form of the argument is: define a complexity measure, show it grows slowly along any small multilinear formula, and show it is large for the permanent and determinant.
- **Reference points already in the literature** against which any new bound is read: the Ω(n³) general-formula bound (Kalorkoti), the existence of n^{O(log n)} formulas for the determinant (so any lower bound must be sub-n^{O(log n)} for det), and the prior exponential bounds for the weaker constant-depth set-multilinear and non-commutative models.

## Code framework

This is a pure complexity-theory result; the artifact is a theorem and its proof, not software. The "scaffold" is the set of pre-existing structural primitives the argument will be assembled from. Stated abstractly, the objects already available are:

```
# A formula is a binary tree with + and × gates; leaves are variables or constants.
# For a node v: φ_v is the sub-formula at v; X_v is the set of variables occurring in φ_v.

def computes_multilinear(formula) -> bool:
    # every gate's polynomial has each variable to power <= 1
    ...

# A complexity measure: a number assigned to the polynomial at a node, meant to grow
# slowly along a small formula and be large for the target polynomial.
def measure(node):
    # TODO: the quantity we will attach to each node
    pass

# How the measure behaves across the two gate types — the inequalities the
# induction will run on.
def measure_plus(v1, v2):
    # TODO: bound for measure at a + gate in terms of the sons
    pass

def measure_times(v1, v2):
    # TODO: bound for measure at a × gate in terms of the sons
    pass

# A way to transform / simplify the variables before measuring, of the kind used in
# restriction-based arguments; it must keep the formula multilinear.
def transform_variables(formula):
    # TODO: the operation applied to the inputs before the measure is read off
    pass

# The lower-bound argument: assume a small formula, derive a contradiction with the
# measure's value on the target polynomial.
def lower_bound(target_polynomial):
    # TODO: combine the measure's slow growth with its large value on the target
    pass
```

The two known growth facts that any measure must respect are the elementary `+`/`×` behaviour the induction will use; the `# TODO` slots are the choice of measure, the choice of pre-processing transform, and the way the two are combined into a contradiction.
