# Context: positive correlation of monotone quantities on ordered configuration spaces

## Research question

Across statistical mechanics and percolation theory, the same qualitative fact keeps surfacing:
"good events help each other." In a ferromagnet, knowing one spin is up makes its neighbours more
likely up. In a percolation model, knowing one region is well-connected makes another region more
likely to be connected too. Made precise, the claim is a **correlation inequality**: for two
quantities f and g that are both *increasing* in the underlying configuration,

  ⟨fg⟩ − ⟨f⟩⟨g⟩ ≥ 0,   equivalently  E[fg] ≥ E[f]E[g],

where ⟨·⟩ is the thermal/probabilistic average. Several special cases of this had been proven, each
bolted to the specific structure of its model — the signs of the interaction constants in an Ising
Hamiltonian, the restriction to a special class of observables, the independence of percolation edges.
The precise problem is: **strip away the physics and find the minimal structural hypotheses — on the
order in which configurations live, and on the measure that weights them — that *force* increasing
quantities to be positively correlated.** A solution would have to (i) identify the right abstract
arena (what kind of ordered set), (ii) state one clean condition on the measure that captures
"the measure does not anti-associate the coordinates," and (iii) prove the inequality from those two
hypotheses alone, recovering the known special cases as instances.

## Background

**The order structure of a configuration space.** A configuration of a physical system on a finite
site set X is a subset R ⊆ X (occupied sites) or, equivalently, a point of the Boolean cube
{0,1}^X; for spins it is a point of a product of two-point chains. These spaces carry a natural
**partial order** (R ≤ S iff R ⊆ S, coordinatewise ≤ for products), and they are **lattices**: any
two elements x, y have a least upper bound x∨y (union / coordinatewise max) and a greatest lower
bound x∧y (intersection / coordinatewise min). They are moreover **distributive** —
x∧(y∨z) = (x∧y)∨(x∧z) — because they are built from subsets/products. Birkhoff's lattice theory
(Birkhoff, *Lattice Theory*, 1967) supplies the structural fact that pins this down: **every finite
distributive lattice of length n is isomorphic to a sublattice of the subsets of an n-element set.**
So "finite distributive lattice" is the coordinatewise order structure closed under meet and join,
not an arbitrary partial order.

A real function f on such a poset is **increasing** if x ≤ y ⇒ f(x) ≤ f(y) (decreasing if the reverse).
An **increasing event** is an up-set A (x ∈ A and x ≤ y ⇒ y ∈ A); its indicator is an increasing
function. "Being connected" / "having a crossing" / "having large degree-sum" are increasing;
"being planar" / "being triangle-free" are decreasing. The covariance ⟨fg⟩ − ⟨f⟩⟨g⟩ measures whether
two such quantities tend to be large together.

**The totally ordered base fact.** When the ground set Γ is *totally* ordered, positive correlation of
increasing functions is elementary and holds for *any* positive measure μ. Writing
⟨f⟩ = Z⁻¹ Σ_x μ(x) f(x) with Z = Σ_x μ(x), a direct symmetrization gives

  ⟨fg⟩ − ⟨f⟩⟨g⟩ = (2Z²)⁻¹ Σ_{x,y} μ(x) μ(y) (f(x) − f(y)) (g(x) − g(y)).

Total order means every pair x, y is comparable, so f and g move the same way across the pair and each
summand (f(x) − f(y))(g(x) − g(y)) ≥ 0. This is the discrete Chebyshev sum / rearrangement inequality.
The whole difficulty is that real configuration spaces are *partially*, not totally, ordered.

**The measure matters in the partial case.** Unlike the totally ordered case, where positive
correlation held for *any* positive measure, on a genuine partial order not every positive μ makes
increasing functions correlate: a measure can in principle conspire with the incomparable pairs and
anti-correlate two increasing functions. So a general statement must isolate some property of μ —
beyond mere positivity — that rules this out. The product (independent-coordinate) measure
μ(x) = ∏_i μ_i(x_i), under which Harris already established positive correlation, is the distinguished
reference point of "no induced dependence between coordinates."

**The motivating prior inequalities (the phenomena to be unified).** By 1970 there were two distinct
streams of correlation inequalities, observed in different models and proved by different means:
(i) Griffiths-type inequalities for Ising ferromagnets, where positive correlation of spin observables
was a known consequence of ferromagnetic (sign-definite) couplings; and (ii) Harris's percolation
inequality, where increasing events were observed to be positively correlated under the independent
edge measure, and used to bound the critical probability. That these two looked unrelated yet said the
"same" thing — increasing quantities cluster — is the empirical fact that motivates seeking one
structural explanation.

## Baselines

**Griffiths' inequalities (Griffiths 1967; Kelly & Sherman 1968; Ginibre 1969, 1970).** For an Ising
ferromagnet with Hamiltonian H = − Σ_R J(R) σ_R (σ_R = ∏_{r∈R} σ_r) and **non-negative** couplings
J(R) ≥ 0, Griffiths' second inequality states that any two observables f, g in a suitable class (the
convex cone generated by the spin products σ_R) satisfy ⟨fg⟩ − ⟨f⟩⟨g⟩ ≥ 0. Ginibre generalized the
proof technique to a wider class of spin systems. **Limitations:** the result is tied to the sign
condition J(R) ≥ 0 on *all* interactions, it restricts the magnetic field (one-body term), and the
admissible observables are the special cone generated by the σ_R — not, e.g., arbitrary increasing
functions of the occupation numbers. The inequality is real but its hypotheses are entangled with the
Hamiltonian, not with the bare order structure.

**Harris's percolation inequality (Harris 1960, "A lower bound for the critical probability in a
certain percolation process," Lemma 4.1).** In independent bond percolation on the square lattice
(each edge open with probability p, independently), Harris proved that any two **increasing events**
are positively correlated, P(A ∩ B) ≥ P(A) P(B), and used it (with a clever argument that the
probability of an infinite open cluster is 0 at p = 1/2) to show the critical probability p_c ≥ 1/2.
Equivalently, for increasing functions of independent variables, E[fg] ≥ E[f]E[g]. **Limitations:** it
is proved only for the **product (independent) measure** and is stated for the percolation setting; it
gives no handle on dependent measures (Ising at non-trivial temperature, random-cluster with κ ≠ 1),
where the very interactions that make the model interesting destroy independence. It also "drew less
attention than it deserved," sitting apart from the Griffiths stream as if a different phenomenon.

**The random-cluster measure (Kasteleyn & Fortuin 1969).** A single measure on subsets R of the edge
set, μ(R) = κ^{c(R)} ∏_{r∈R} p_r ∏_{s∉R} q_s, where c(R) is the number of connected components of the
open subgraph on the fixed vertex set, including isolated vertices. For κ = 1 it is the independent
percolation measure (Harris's setting); for κ = 2 its
normalizing constant equals (up to a trivial factor) the Ising partition function, so Ising
correlations are recovered as expectations under it. **Limitations / opportunity:** this already
*unifies* percolation and Ising as one measure, strongly suggesting their correlation inequalities are
one fact — but a correlation inequality had been proven for this measure only in special cases, with no
single abstract statement covering dependent cases such as κ > 1 and arbitrary increasing observables.

## Evaluation settings

The natural arenas in which such an inequality would be tested and used (all pre-existing structures):
the **finite distributive lattice** P(X) of subsets of a finite set X under inclusion, and products of
finite chains; the **Ising / lattice-gas model** on a finite site set with a many-body Hamiltonian and
an (inhomogeneous) external field, where the targets are spin-spin correlations
⟨σ_r σ_s⟩ − ⟨σ_r⟩⟨σ_s⟩ in arbitrary field; the **independent bond/site percolation model** on a finite
subgraph of a lattice, where the targets are correlations of increasing connection events
(crossings, the origin connecting to a region); and the **random-cluster model** interpolating between
them. The relevant "metric" is qualitative: whether the covariance of two increasing functions has the
predicted sign. The yardstick prior results are Griffiths' second inequality and Harris's lemma — any
general statement must reproduce both.

## Code framework

The result is a theorem; the only computational artifact is an optional brute-force *check* on a small
lattice, which we scaffold here with the model-specific condition left as the slot to fill. The
primitives that already exist: enumerate a finite poset, define a positive measure and two monotone
functions on it, compute a normalized average and a covariance.

```python
from itertools import combinations

GROUND = ("a", "b", "c")
LATTICE = [frozenset(s) for k in range(len(GROUND) + 1)
           for s in combinations(GROUND, k)]      # P(X) under inclusion

def meet(x, y): return x & y                       # greatest lower bound
def join(x, y): return x | y                       # least upper bound

def increasing(values):                            # monotone non-decreasing under <=
    for x in LATTICE:
        for y in LATTICE:
            if x <= y and values[x] > values[y] + 1e-12:
                return False
    return True

def average(mu, h):
    Z = sum(mu[x] for x in LATTICE)
    return sum(mu[x] * h[x] for x in LATTICE) / Z

def covariance(mu, f, g):                          # <fg> - <f><g>
    fg = {x: f[x] * g[x] for x in LATTICE}
    return average(mu, fg) - average(mu, f) * average(mu, g)

def measure_condition_holds(mu):
    # TODO: the condition on the measure that guarantees increasing f, g
    #       are positively correlated.
    pass

# Desired property:
#   measure_condition_holds(mu) and increasing(f) and increasing(g)
#       => covariance(mu, f, g) >= 0
```
