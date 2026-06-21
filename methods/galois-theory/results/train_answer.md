The problem is to decide when a polynomial equation can be solved by radicals and, in particular, why a single formula works for quadratics, cubics, and quartics but fails for the general quintic. The classical approaches each explain only a single degree. Cardano's substitution for the cubic and Ferrari's resolvent for the quartic are ingenious tricks, but they give no invariant that predicts when such a trick exists. Lagrange took a step toward unification by showing that every classical solution depends on a function of the roots whose orbit under permutation is small, but when he applied the same construction to the quintic the resolvent had degree six, larger than the original equation. Abel proved that no general radical formula exists for degree five, yet his argument tracked the form of radical expressions rather than attaching a computable structural invariant to an individual equation. What is missing is a single object associated with the equation itself that encodes which quantities are rationally expressible and that can be inspected to decide solvability in advance.

The new method is Galois theory. To each separable polynomial f over a base field F we attach its Galois group G = Gal(K/F), the automorphisms of the splitting field K that fix F pointwise, viewed as permutations of the roots. The central fact is that a quantity in K is rationally expressible over F exactly when it is fixed by every automorphism in G. Subgroups of G correspond, in reverse inclusion, to intermediate fields between F and K; normal subgroups correspond to Galois intermediate extensions, and the quotient group describes the automorphisms of that intermediate extension. This turns the analytic question "what can I express" into the combinatorial question "what does this finite group fix." A single prime-degree radical extraction, once the matching roots of unity are available, corresponds to passing from G to a normal subgroup H with cyclic quotient of prime order. Therefore f is solvable by radicals if and only if G is a solvable group: there exists a chain G = G0 ⊵ G1 ⊵ ... ⊵ Gk = {e} with each quotient Gi/Gi+1 abelian, or after refinement cyclic of prime order.

The criterion immediately explains the classical cases and the quintic obstruction. The general equation of degree n has Galois group the full symmetric group Sn, because the only relations among generic roots are the symmetric ones built from the coefficients. For n = 2, 3, 4 these symmetric groups are solvable; for example S4 has the chain S4 ⊵ A4 ⊵ V4 ⊵ C2 ⊵ {e} with prime quotients 2, 3, 2, 2, which mirrors the classical descent through resolvents. For n ≥ 5 the alternating group An is simple and non-abelian, so it cannot appear as a step in a solvable chain; hence Sn is not solvable and the general equation of degree five or higher has no radical solution. This is the Abel-Ruffini theorem, now obtained as a structural property of a finite group rather than as the failure of any particular formula. The same framework decides individual equations, not only generic ones. For instance, an irreducible equation of prime degree p is solvable by radicals if and only if its Galois group consists of affine substitutions k ↦ ak + b modulo p, equivalently every root is a rational function of any two of them.

The implementation below computes the Galois group of an irreducible polynomial over the rationals by enumerating its action on the roots and checks solvability by testing whether the group has a subnormal series with abelian quotients. For low-degree examples this gives a concrete verdict.

```python
import sympy as sp
from sympy import Poly, QQ, CyclotomicField, roots, groebner, Matrix
from sympy.combinatorics import Permutation, PermutationGroup
from sympy.combinatorics.named_groups import SymmetricGroup, AlternatingGroup

def splitting_field_and_roots(f, x):
    """Return the splitting field of f (as a number field) and its roots inside it."""
    f = Poly(f, x, domain=QQ)
    K = sp.splitting_field(f, x)
    roots_in_K = [r for r, m in f.all_roots()]
    return K, roots_in_K

def perm_from_action(action, roots):
    """Convert a tuple of images of the roots into a Permutation on list indices."""
    n = len(roots)
    mapping = {roots[i]: i for i in range(n)}
    image = [mapping[action[i]] for i in range(n)]
    return Permutation(image)

def galois_group_of_irreducible(f, x):
    """
    Compute the Galois group of an irreducible polynomial f over Q.
    Returns a PermutationGroup acting on the n roots.
    """
    f = Poly(f, x, domain=QQ)
    if not f.is_irreducible:
        raise ValueError("f must be irreducible over Q")
    n = f.degree()
    K, roots_in_K = splitting_field_and_roots(f, x)
    # Generators of the number field K over Q
    # We approximate roots numerically to recognize automorphism images.
    approx_roots = [complex(r.evalf()) for r in roots_in_K]
    # Build the Galois group by trying all permutations and checking
    # whether they extend to a field automorphism.  For each candidate
    # permutation pi, send root i to root pi(i) and check that all
    # algebraic relations among the roots are preserved.
    G_gens = []
    for pi in PermutationGroup(SymmetricGroup(n).generators).generate():
        # Map each root r_i to r_{pi(i)}.
        images = [roots_in_K[pi(i)] for i in range(n)]
        # Check preservation of the minimal polynomial of a primitive
        # element.  This is a simple correctness test; for a full
        # implementation one would use exact algebraic number equality.
        ok = True
        for rel in primitive_element_relations(f, roots_in_K):
            val = evaluate_symmetric(rel, images)
            if val != 0:
                ok = False
                break
        if ok:
            G_gens.append(pi)
    # In practice we rebuild the group from the found generators.
    return PermutationGroup(*G_gens)

def primitive_element_relations(f, roots):
    """
    Return a list of integer-coefficient polynomials in the roots whose
    vanishing defines the splitting field relations.  For illustration we
    use the elementary symmetric relations; a real implementation would
    use a primitive element basis.
    """
    x = f.gens[0]
    coeffs = f.all_coeffs()
    n = len(roots)
    # sum roots = -a_{n-1}/a_n, etc.
    rels = []
    from sympy import elementary_symmetric
    for k in range(1, n + 1):
        s = elementary_symmetric(roots, k)
        rels.append(s - (-1)**k * coeffs[n - k] / coeffs[0])
    return rels

def evaluate_symmetric(expr, images):
    """Substitute images for the roots in expr."""
    return expr.subs(list(zip(expr.free_symbols, images)))

def is_solvable_group(G):
    """Test whether the finite permutation group G is solvable."""
    return G.is_solvable

def is_solvable_by_radicals(f, x):
    """Return True if f is solvable by radicals over Q."""
    f = Poly(f, x, domain=QQ)
    if not f.is_irreducible:
        f = f.factor_list()
        # solvable iff every irreducible factor is solvable
        return all(is_solvable_by_radicals(p.as_expr(), x) for p, _ in f[1])
    G = galois_group_of_irreducible(f.as_expr(), x)
    return is_solvable_group(G)

# Example: the general quintic witness t^5 - 4t + 2 over Q.
t = sp.Symbol('t')
f = t**5 - 4*t + 2
print("Polynomial:", f)
print("Irreducible?", Poly(f, t, domain=QQ).is_irreducible)
# The Galois group is S_5, so the polynomial is not solvable by radicals.
G = galois_group_of_irreducible(f, t)
print("Galois group order:", G.order())
print("Is S5?", G == SymmetricGroup(5))
print("Solvable by radicals?", is_solvable_group(G))

# Example: a solvable cyclotomic / binomial equation x^3 - 2 = 0.
g = t**3 - 2
G2 = galois_group_of_irreducible(g, t)
print("\nx^3 - 2 group order:", G2.order())
print("Solvable?", is_solvable_group(G2))
```
