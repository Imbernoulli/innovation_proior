# Context

## Research question

In 1959 Kadison and Singer asked a question that arose in the $C^*$-algebra
formalization of quantum mechanics: does every pure state on the abelian von
Neumann algebra $\mathbb{D}$ of bounded diagonal operators on $\ell_2$ extend
*uniquely* to a pure state on $B(\ell_2)$, the algebra of all bounded operators?
A pure state is an extreme point of the convex set of states (positive
unit-preserving linear functionals); on $\mathbb{D}$ the pure states are the
limits of coordinate evaluations. Restriction always gives *at least one*
extension; the question is whether it is *unique*. Kadison and Singer suspected
the answer was negative but did not formally conjecture either way.

Over the following decades a positive answer was shown to be equivalent to a long
list of concrete, finite, combinatorial-linear-algebraic statements about
decomposing matrices and vectors into well-behaved pieces — statements in operator
theory, frame theory, discrepancy theory, and signal processing. So the abstract
operator-algebra question became a question about a clean, finite-dimensional
phenomenon. The goal is to settle one of these equivalent finite statements *with
a universal constant* — a bound on the number of pieces that does **not** grow
with the dimension.

## Background

**The paving reformulation.** Twenty years after the original question, Anderson
showed Kadison-Singer is equivalent to a statement about *paving* finite matrices,
which by an elementary compactness argument extends to operators on $\ell_2$.
A matrix $T$ is **$(r,\epsilon)$-paved** if there exist coordinate projections
$P_1,\dots,P_r$ (each picking out a subset of basis coordinates) with
$\sum_i P_i = I$ and $\|P_i T P_i\| \le \epsilon\|T\|$ for every $i$. The Anderson
paving conjecture asks: for every $\epsilon>0$ there is an $r$ (independent of the
size $n$) so that every zero-diagonal self-adjoint $n\times n$ matrix can be
$(r,\epsilon)$-paved. The diagonal of $T$ is what survives a projection
$\sum_i P_i T P_i$; subtracting it off, the conjecture says any zero-diagonal $T$
can be split into a bounded number of coordinate blocks each of which has small
norm relative to $T$. It is known that one cannot do better than $r \approx
1/\epsilon^2$.

**Weaver's discrepancy reformulation $KS_2$.** Weaver recast a projection-paving
conjecture of Akemann and Anderson as a discrepancy statement, and showed it
implies Kadison-Singer. In the amended form: there are universal constants
$\eta\ge 2$ and $\theta>0$ so that whenever $w_1,\dots,w_m\in\mathbb{C}^d$ satisfy
$\|w_i\|\le 1$ and
$$\sum_{i=1}^m |\langle u, w_i\rangle|^2 = \eta \quad\text{for every unit vector } u,$$
there is a partition $S_1,S_2$ of $\{1,\dots,m\}$ with
$\sum_{i\in S_j}|\langle u, w_i\rangle|^2 \le \eta-\theta$ for every unit $u$ and
each $j$. The hypothesis says the $w_i$ form a tight frame of small vectors
(an isotropic position scaled by $\eta$); the conclusion asks to split the frame
into two parts each of which is *strictly* below the full frame in every
direction. Equivalently, writing $A=\sum_i v_iv_i^* = I$ with $v_i = w_i/\sqrt\eta$
small, one wants a two-coloring with each color's $\sum_{i\in S_j} v_iv_i^*$
bounded away from $I$.

**The common shape: a sum of rank-one matrices.** Both reformulations, and the
Ramanujan-graph existence problem from the same circle of ideas, reduce to a
single phenomenon: a sum $\sum_i v_iv_i^*$ of rank-one positive semidefinite
matrices, where one wants to make a *combinatorial choice* (a subset, a partition,
a $\pm$ signing) so the resulting sum has controlled operator norm. Concretely,
the target object is independent random vectors $v_1,\dots,v_m\in\mathbb{C}^d$ with
$$\sum_i \mathbb{E}\, v_iv_i^* = I, \qquad \mathbb{E}\,\|v_i\|^2 \le \epsilon,$$
and the question is how small $\big\|\sum_i v_iv_i^*\big\|$ can be made over the
support of the distribution.

**What was known about such sums.** Sums of independent random rank-one matrices
are exactly what matrix concentration was built for. The matrix Chernoff /
Ahlswede-Winter inequality and Rudelson's isotropic-position bound give, in this
setting,
$$\Big\|\sum_i v_iv_i^*\Big\| \le C(\epsilon)\cdot \log n$$
with high probability.

**Tools for the roots of polynomials.** A self-adjoint positive semidefinite
matrix $M$ has $\|M\|$ equal to its largest eigenvalue, which is the largest root
of its characteristic polynomial $\chi[M](x)=\det(xI-M)$. So a norm bound is a
statement about the largest root of a polynomial. The classical theory of where
roots of *real-rooted* univariate polynomials lie, and how they move under simple
operations, was available:
- **Interlacing.** A real-rooted $g$ of degree $n-1$ *interlaces* a real-rooted
  $f$ of degree $n$ if their roots alternate. A family $f_1,\dots,f_k$ has a
  *common interlacing* if a single $g$ interlaces all of them. It was known
  (Dedieu; Fell; Chudnovsky-Seymour) that $f_1,\dots,f_k$ of equal degree with
  positive leading coefficients have a common interlacing **iff** every convex
  combination $\sum_i \lambda_i f_i$ is real-rooted.
- **Real stable polynomials.** A multivariate $p\in\mathbb{C}[z_1,\dots,z_m]$ is
  *stable* if $p\ne 0$ whenever every $z_i$ has strictly positive imaginary part;
  *real stable* if additionally its coefficients are real. A univariate real
  stable polynomial is exactly a real-rooted one. Borcea and Brändén built a
  detailed theory of which linear operators preserve stability, and observed that
  for positive semidefinite $A_1,\dots,A_m$ the polynomial $\det(\sum_i z_iA_i)$
  is real stable. Operators of the form $1-\partial_{z_i}$ preserve real stability
  (Lieb-Sokal; Borcea-Brändén), and setting a variable to a real number preserves
  it.
- **Determinantal representation of bivariate stable polynomials.**
  Helton-Vinnikov (with the Lax-conjecture resolution of Lewis-Parrilo-Ramana)
  showed every bivariate real stable polynomial of degree $d$ equals
  $\pm\det(z_1 A + z_2 B + C)$ for $d\times d$ positive semidefinite $A,B$ and
  Hermitian $C$.
- **Rank-one update identities.** The matrix determinant lemma
  $\det(A+uv^*)=\det(A)(1+v^*A^{-1}u)$ and Jacobi's formula
  $\partial_t\det(A+tB)=\det(A)\,\mathrm{Tr}(A^{-1}B)$ describe how a determinant
  changes under a rank-one perturbation.
- **Capacity / van der Waerden.** Gurvits, reproving and generalizing the van der
  Waerden conjecture on permanents, studied the "capacity"
  $\mathrm{Cap}(p)=\inf_{x>0} p(x)/(x_1\cdots x_n)$ of a stable homogeneous
  polynomial and lower-bounded the coefficient extracted by
  $\partial^n/\partial x_1\cdots\partial x_n$ acting on $\det(\sum_i x_iA_i)$ —
  the *mixed discriminant*. His proof used differential operators of derivative
  type together with elementary inequalities, with no appeal to combinatorial
  structure.

**The barrier method for roots under rank-one updates.** For the related problem
of sparsifying a graph — approximating $\sum_i v_iv_i^* = I$ by a *reweighted
sub-sum* $\sum_i s_iv_iv_i^*$ with few nonzero $s_i$ and bounded condition number —
a deterministic technique was developed (Batson-Spielman-Srivastava). One builds
the sum one rank-one term at a time, maintaining two "barriers" above and below the
spectrum and a **potential function**
$$\Phi^u(A) = \mathrm{Tr}(uI-A)^{-1} = \sum_i \frac{1}{u-\lambda_i}$$
that blows up as any eigenvalue $\lambda_i$ approaches the upper barrier $u$. Using
the Sherman-Morrison expansion of $(uI-A-tvv^*)^{-1}$, one shows there is always a
vector and a weight to add that lets both barriers move forward by a fixed amount
without increasing the potential; iterating drives the condition number to
$\frac{d+1+2\sqrt d}{d+1-2\sqrt d}$. The same paper noted, "purely for
motivational purposes," that *adding the average vector* applies the operator
$I-\tfrac1m\frac{d}{dx}$ to the characteristic polynomial, so that starting from
$x^n$ and iterating produces $(I-\tfrac1m\frac{d}{dx})^k x^n$, a scaled associated
Laguerre polynomial whose extreme roots have the very ratio
$\frac{d+1+2\sqrt d}{d+1-2\sqrt d}$. This was an algorithmic, weighted result; it
controlled a *reweighted* sum, not a subset or a partition, and its potential
machinery was univariate.

**A primitive special case that already shows the shape.** When the random vectors
$v_1,\dots,v_m$ are *identically distributed* and isotropic ($\mathbb{E}\,vv^*=cI$),
everything collapses to one variable. Cauchy's interlacing theorem says
$\chi[A](x)$ interlaces $\chi[A+vv^*](x)$, so rank-one updates produce common
interlacings; and the expected update is exactly a differential operator,
$\mathbb{E}\,\chi[A+vv^*](x) = (I-c\tfrac{d}{dx})\chi[A](x)$ (matrix determinant
lemma plus $\mathbb{E}\,v^*(xI-A)^{-1}v = c\,\mathrm{Tr}(xI-A)^{-1}$). Since
$(1-cD)$ preserves real-rootedness (via the rational function $1-c\sum_i
\frac1{x-\lambda_i}$ and the intermediate value theorem) and preserves common
interlacings, the expected characteristic polynomials form a family one can run a
first-moment argument through, and the resulting expected polynomial is a Laguerre
polynomial whose roots are classically known. This isotropic case recovers
Bourgain-Tzafriri's restricted invertibility in the isotropic position. It is
*entirely univariate* — every covariance is a multiple of the identity, so all the
operators commute and act on a single polynomial.

## Baselines

**Matrix concentration (Ahlswede-Winter; Rudelson; matrix Chernoff).** These
bound the operator norm of a sum of independent random matrices by controlling
matrix moments / Laplace transforms. In the present setting they yield
$\|\sum_i v_iv_i^*\| \le C(\epsilon)\log n$ with high probability. Core idea:
$\mathbb{E}\,\mathrm{Tr}\,e^{\theta \sum_i v_iv_i^*}$ factorizes well enough to give
a tail bound on the top eigenvalue.

**Bourgain-Tzafriri restricted invertibility / partial Kadison-Singer results.**
Under stronger hypotheses than the full conjecture, various authors
(Bourgain-Tzafriri; Berman et al.; Paulsen; Baranov-Dyakonov; Lawton; Akemann
et al.; Popa) obtained partial pavings or partitions. The strongest of these gave
constants on the order of $1/\log n$.

**The deterministic barrier / sparsification method.** Builds a *reweighted*
sub-sum $\sum_i s_iv_iv_i^*$ with $s_i\ge 0$, few nonzero, condition number
$\le \frac{d+1+2\sqrt d}{d+1-2\sqrt d}$, via the upper/lower potential functions
and barrier shifts described above.

**The interlacing-families existence principle.** A collection of real-rooted
polynomials indexed by a product set $S_1\times\cdots\times S_m$ is an
*interlacing family* if, at every level, the children of any partial assignment
have a common interlacing. Such a family always contains a polynomial whose largest
root is at most the largest root of the sum of the whole family — proved by
repeatedly applying the common-interlacing lemma down the tree. This converts
"the average is good" into "some leaf is good," a first-moment existence statement.

## Evaluation settings

This is a pure existence theorem in matrix analysis / operator theory; the natural
yardstick is mathematical, not empirical.

- **The statement to certify.** A bound of the form: for independent random
  $v_i\in\mathbb{C}^d$ with $\sum_i\mathbb{E}\,v_iv_i^*=I$ and
  $\mathbb{E}\,\|v_i\|^2\le\epsilon$, with *nonzero probability*
  $\|\sum_i v_iv_i^*\|$ is below an explicit, dimension-free threshold. Success is
  measured by (i) the threshold being a universal constant (no $n$, no $d$), (ii)
  the guarantee being existence (nonzero probability), not high probability.
- **The equivalences it must feed.** Weaver's $KS_2$ with explicit universal
  $\eta,\theta$; Anderson paving with an explicit $r(\epsilon)$ independent of $n$;
  Akemann-Anderson projection paving; and through them Kadison-Singer itself.
- **Sharpness benchmarks.** Two known objects calibrate the right constant. The
  associated Laguerre polynomials, whose extreme-root locations are classical, give
  the ratio $\frac{d+1+2\sqrt d}{d+1-2\sqrt d}$ in the isotropic case. The matching
  polynomial of a graph (Heilmann-Lieb), whose largest root is $\le 2\sqrt{d-1}$,
  calibrates the regular case; and the Alon-Boppana / Feng-Li bounds show $2\sqrt{d-1}$
  cannot be beaten. A correct general bound should agree, to first order, with these.
- **Lower bound on the paving constant.** It is known that paving cannot use fewer
  than $r\approx 1/\epsilon^2$ pieces, so the achieved $r(\epsilon)$ is judged
  against that floor.

## Code framework

This is a theorem with a complete proof, not a computational method. There is no
training loop, dataset, or numerical artifact to fill in. The "scaffold" is the
logical skeleton the argument will occupy: a chain of lemmas that turns a norm
question into a question about the largest root of an expected characteristic
polynomial, and a bound on that root. The pieces below are the already-available
primitives and the empty slots the proof must fill; they correspond
piece-for-piece to the final theorem statements and proofs.

```
# ---- already-available primitives (stated above in Background) ----
# norm_to_maxroot:  ||M|| = largest root of chi[M](x), for M PSD self-adjoint.
# matrix_determinant_lemma:  det(A + u v*) = det(A) (1 + v* A^{-1} u)
# jacobi:  d/dt det(A + tB) = det(A) Tr(A^{-1} B)
# common_interlacing_iff_real_rooted:  {f_i} common interlacing  <=>  all convex
#     combos sum_i lambda_i f_i are real rooted        (Dedieu / Fell / C-S)
# real_stable_seed:  A_i PSD  =>  det(sum_i z_i A_i) is real stable   (Borcea-Branden)
# stability_preserved:  (1 - d/dz_i) preserves real stability; setting z_i real does too
# bivariate_det_rep:  bivariate real stable p = +- det(z1 A + z2 B + C), A,B PSD, C Herm.
# interlacing_family_principle:  an interlacing family contains a leaf whose
#     largest root <= largest root of the sum of the family.

# ---- the object under study ----
def random_rank_one_sum(v):          # v_1..v_m independent, finite support, in C^d
    # sum_i E[v_i v_i*] = I ,  E ||v_i||^2 <= eps
    return sum(v_i @ v_i.conj().T for v_i in v)

# ---- empty slots the proof must fill ----

def expected_char_poly(A):           # A_i = E[v_i v_i*]
    # TODO: express E[ chi[ sum_i v_i v_i* ](x) ] in closed form as an object we
    #       can analyze.  (what is it, and why is it tractable?)
    pass

def outcome_polynomials(v):
    # the per-outcome characteristic polynomials, indexed by the support product set
    # TODO: the structural property of this family that lets a first-moment
    #       existence argument go through
    pass

def root_bound(A, eps):              # sum_i A_i = I, Tr(A_i) <= eps
    # TODO: an upper bound on the largest root of expected_char_poly(A),
    #       dimension-free, that the construction will establish
    pass

def main_theorem(v, eps):
    # TODO: assemble outcome_polynomials + expected_char_poly + root_bound into
    #       "with nonzero probability, || random_rank_one_sum(v) || <= <bound>"
    pass

def weaver_and_paving(...):
    # TODO: derive the partition / paving consequences (with explicit universal
    #       constants) from main_theorem
    pass
```
