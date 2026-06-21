I propose that the canonical name for this result be the Downward Löwenheim-Skolem theorem, or, when one wants to emphasize the operative construction, the Skolem-hull method. The theorem answers a basic question about how much first-order logic can see of an infinite structure. If I am given a first-order language, an infinite structure in that language, and a small set of elements I want to keep, I can carve out a substructure that contains those elements, satisfies exactly the same first-order sentences with parameters from the substructure, and is no larger than the language and the starting set force it to be.

The precise statement is this. Let L be a first-order language and let M be an L-structure. Let A be any subset of the domain of M. Then there is an elementary substructure N of M such that A is contained in N and the cardinality of N is at most max(|A|, |L|, aleph_0). When the language is countable and A is countable, this bound is just aleph_0, so every infinite structure in a countable language has a countable elementary substructure containing any prescribed countable set. That is the result I will explain and then illustrate with a concrete simulation.

The first thing to notice is that the naive attempt almost works but fails at exactly the right place. If I take a subset A and close it under all the function symbols of the language, I get a genuine substructure. The interpretations of the function symbols stay inside the smaller set, constants are included, and relations are just restricted from M. Atomic formulas therefore have the same truth values in the smaller set as in M, because they only ask about the values of functions and relations on tuples from the smaller set. The trouble is existential formulas. It can happen that M satisfies exists y phi(y, a) for some tuple a from the smaller set, but every witness b in M lies outside the smaller set. Then the smaller structure says the existential statement is false while M says it is true, so the substructure is not elementary.

The fix is to add the missing witnesses. For every first-order formula phi(y, x_1, ..., x_n), I choose a witness function F_phi that takes an n-tuple a from M and returns some element b of M such that, whenever M satisfies exists y phi(y, a), the chosen b satisfies phi(b, a) in M. If no witness exists, the function returns an arbitrary element; that value never matters for the construction. These witness functions live in the meta-language, not necessarily in L itself. They are just bookkeeping devices that let me close A under the existential demands that first-order formulas can make.

Now I build the Skolem hull by an omega-stage construction. Let A_0 contain A together with the interpretations of all constant symbols of L. Given A_i, let A_{i+1} contain A_i, all values of all function symbols of L on tuples from A_i, and all values F_phi(a) for every formula phi and every finite tuple a from A_i. Finally let N be the union of all A_i. Because every finite tuple from N appears together in some A_i, and the next stage adds witnesses for all existential formulas over that tuple, N is closed under the witness functions. Because the construction also closed under the original function symbols at every stage, N is the domain of an L-substructure of M.

The cardinality control is the heart of the theorem. A first-order language L has at most max(|L|, aleph_0) formulas, because formulas are finite strings over the symbols of L plus logical notation and variables. For an infinite cardinal kappa, the set of finite tuples from a set of size at most kappa also has size at most kappa. So if |A| and |L| are both at most kappa, then each A_i has size at most kappa, and a countable union of sets of size at most kappa still has size at most kappa. This gives the bound |N| <= max(|A|, |L|, aleph_0).

To verify elementarity I use the Tarski-Vaught test. That test says a substructure N of M is elementary if, whenever M satisfies exists y phi(y, a) for parameters a from N, there is already an element b in N such that M satisfies phi(b, a). But this is exactly how N was built. Take any tuple a from N. Because a is finite, all its entries lie in some A_i. The construction put F_phi(a) into A_{i+1}, hence into N, and by the defining property of F_phi this element witnesses phi in M. Therefore the Tarski-Vaught condition holds, and N is an elementary substructure of M.

The most famous consequence is the countable-model phenomenon, sometimes called Skolem's paradox. A first-order theory such as Zermelo-Fraenkel set theory can prove that uncountable sets exist. If that theory has any infinite model in a countable language, the downward theorem gives a countable model of the same theory. In that countable model there may be an object that the model regards as uncountable, because the model contains no bijection from the natural numbers onto that object. Externally, the whole model is countable, but internally the absence of a witnessing bijection is enough to make the first-order statement "this set is uncountable" true. There is no contradiction: first-order truth is preserved, but external cardinality is not first-order definable.

The same idea explains why first-order logic cannot pin down intended infinite sizes. I can write sentences saying there are at least 1, at least 2, at least 3, and so on, and together they force infinitude. But no first-order sentence in the ordinary language can say "the domain has exactly this uncountable cardinality." The Skolem-hull construction shows that any infinite model can be thinned down to a model whose size is bounded by the language and a chosen parameter set, while all first-order statements remain unchanged.

The following Python script makes the construction concrete on a finite toy structure. It defines a small first-order language with a unary relation, a binary relation, and a unary function; it builds a random structure; it chooses least-witness Skolem functions for a finite set of formulas; it closes a starting set under those functions and the original function; and it checks the Tarski-Vaught condition explicitly. The finite example is not the theorem itself, but it shows exactly the same closure mechanism that the proof uses in the general infinite setting.

```python
import itertools
import random

# A finite toy structure for the Skolem-hull construction.
# Language: unary relation P, binary relation R, unary function f.
M = list(range(12))
random.seed(0)
P = {x for x in M if random.random() < 0.4}
R = {(x, y) for x in M for y in M if random.random() < 0.15}
f = {x: (x + 3) % len(M) for x in M}

# Formulas are represented as Python functions phi(witness, params) -> bool.
# We consider a small set of formulas that generate interesting witnesses.
def phi1(y, xs):
    # P(y)
    return y in P

def phi2(y, xs):
    # R(xs[0], y)
    return len(xs) >= 1 and (xs[0], y) in R

def phi3(y, xs):
    # R(y, xs[0])
    return len(xs) >= 1 and (y, xs[0]) in R

def phi4(y, xs):
    # f(y) == xs[0]
    return len(xs) >= 1 and f[y] == xs[0]

formulas = [phi1, phi2, phi3, phi4]

def witness(phi, params):
    """Least-witness Skolem function for a formula on this finite M."""
    for b in M:
        if phi(b, params):
            return b
    return M[0]  # arbitrary when no witness exists

# Build the Skolem hull starting from A = {0}.
A = {0}
current = set(A)
# Include constants if any; here none, so A_0 = A.
while True:
    nxt = set(current)
    # Close under the unary function f.
    for a in current:
        nxt.add(f[a])
    # Close under witness functions for all formulas and finite tuples.
    for phi in formulas:
        for arity in range(3):  # allow parameter tuples of length 0, 1, 2
            for tup in itertools.product(current, repeat=arity):
                nxt.add(witness(phi, tup))
    if nxt == current:
        break
    current = nxt

N = current
print("Domain M size:", len(M))
print("Skolem hull N size:", len(N))
print("N contains starting set:", A <= N)

# Verify the Tarski-Vaught condition for the same finite set of formulas.
ok = True
for phi in formulas:
    for arity in range(3):
        for tup in itertools.product(N, repeat=arity):
            if any(phi(b, tup) for b in M):
                if not any(phi(b, tup) for b in N):
                    print("Tarski-Vaught failed for", phi.__name__, "with", tup)
                    ok = False
print("Tarski-Vaught check passed:", ok)
```
