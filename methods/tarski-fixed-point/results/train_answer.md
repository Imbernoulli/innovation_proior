I will work through the canonical fixed-point theorem that lives entirely in order theory: the Knaster–Tarski fixed-point theorem, often called Tarski’s fixed-point theorem. The setting is a complete lattice, a partially ordered set in which every subset has a least upper bound and a greatest lower bound. I write the order as \(\leq\). Completeness means that for every subset \(S\) there is a join \(\bigvee S\) and a meet \(\bigwedge S\). In particular, the empty subset has a join, which is the bottom element \(\bot\), and a meet, which is the top element \(\top\). The map I consider is any monotone self-map \(f : L \to L\), meaning that whenever \(x \leq y\) I have \(f(x) \leq f(y)\). No metric, no topology, no compactness, no continuity, and certainly no contraction constant is assumed. Monotonicity alone will be enough to produce a rich structure of fixed points.

The first thing I want is a least fixed point. Instead of iterating \(f\) from the bottom, which would require chain continuity to guarantee that the limit is preserved, I collect all of the one-sided stable bounds at once. Define the set of pre-fixed points, or upper stable bounds, as \(P = \{x \in L \mid f(x) \leq x\}\). This set is never empty because the top element satisfies \(f(\top) \leq \top\) trivially. Because \(L\) is complete, I can form the meet \(a = \bigwedge P\). I now argue that \(a\) is itself a fixed point. Take any \(p \in P\). Since \(a \leq p\), monotonicity gives \(f(a) \leq f(p)\), and because \(p \in P\) I also have \(f(p) \leq p\). Therefore \(f(a) \leq p\) for every \(p \in P\). That makes \(f(a)\) a lower bound of \(P\). But \(a\) is the greatest lower bound of \(P\), so \(f(a) \leq a\). This inequality already puts \(a\) back into \(P\). Since \(a\) is the meet of all elements of \(P\), and \(f(a)\) is one of those elements, I also have \(a \leq f(a)\). Both inequalities together force \(f(a) = a\). Every fixed point belongs to \(P\), because equality implies \(f(z) \leq z\), so \(a\) is below every fixed point. Thus \(a\) is the least fixed point, \(\text{lfp}(f)\).

The greatest fixed point comes from the dual construction. Define \(Q = \{x \in L \mid x \leq f(x)\}\), the set of post-fixed points or lower stable bounds. This set is nonempty because \(\bot \in Q\). Let \(b = \bigvee Q\). For any \(q \in Q\), monotonicity gives \(f(q) \leq f(b)\), and because \(q \leq f(q)\) I obtain \(q \leq f(b)\). Hence \(f(b)\) is an upper bound of \(Q\). Since \(b\) is the least upper bound, \(b \leq f(b)\). This places \(b\) in \(Q\). But \(b\) is also the join of all elements of \(Q\), and \(f(b) \in Q\), so \(f(b) \leq b\). The two inequalities collapse to \(f(b) = b\). Every fixed point lies in \(Q\), so every fixed point is below \(b\). Therefore \(b\) is the greatest fixed point, \(\text{gfp}(f)\).

So far I have existence and two canonical extremal fixed points. The deeper statement is that the entire set of fixed points, \(\text{Fix}(f)\), is itself a complete lattice under the inherited order. That means every subset of fixed points has a join and a meet inside \(\text{Fix}(f)\), not merely inside \(L\). The join in \(L\) of a family of fixed points need not be fixed, so I have to close it with the same one-sided argument, but now inside an interval sublattice.

Take any subset \(A \subseteq \text{Fix}(f)\) and let \(s = \bigvee_L A\). For each \(x \in A\), I have \(x = f(x) \leq f(s)\), because \(x \leq s\). Therefore \(f(s)\) is an upper bound of \(A\), and the least upper bound satisfies \(s \leq f(s)\). This means the interval \([s, \top]\) is stable under \(f\): if \(s \leq y \leq \top\), then \(s \leq f(s) \leq f(y) \leq \top\). The interval \([s, \top]\) is a complete lattice in its own right. Applying the least-fixed-point construction inside this interval gives a fixed point \(j \in [s, \top]\) that is the least fixed point above \(s\). It is above every element of \(A\), and if \(u\) is any fixed point above every element of \(A\), then \(s \leq u\), so \(u\) lies in \([s, \top]\); since \(j\) is the least fixed point there, \(j \leq u\). Thus \(j\) is exactly the join of \(A\) inside \(\text{Fix}(f)\). The meet construction is dual: if \(t = \bigwedge_L A\), then \(f(t) \leq t\), the interval \([\bot, t]\) is stable under \(f\), and the greatest-fixed-point construction inside that interval yields the meet of \(A\) inside \(\text{Fix}(f)\). Empty families are included automatically: the join of the empty family in \(\text{Fix}(f)\) is the least fixed point, and the meet of the empty family is the greatest fixed point.

The canonical names for the extremal fixed points are therefore \(\text{lfp}(f) = \bigwedge \{x \in L \mid f(x) \leq x\}\) and \(\text{gfp}(f) = \bigvee \{x \in L \mid x \leq f(x)\}\). The least fixed point is the inductive solution constrained by all upper stable bounds, and it is the one typically used to give meaning to recursive definitions in programming-language semantics. The greatest fixed point is the coinductive solution supported by all lower stable bounds, and it appears naturally in settings such as games with strategic complementarities, where monotone best responses can have multiple equilibria ordered from lowest to highest. The theorem explains why multiple fixed points are not accidental: they are organized into a complete lattice by the same order data that produced them.

The proof uses nothing beyond completeness and monotonicity. Contraction arguments would have asked for distances and shrinking constants and would have usually produced uniqueness, which is neither available nor desired here. Topological arguments would have asked for compactness, convexity, and continuity, none of which are part of the order-theoretic data. The complete lattice supplies global joins and meets, and monotonicity keeps the collections of one-sided stable bounds closed under \(f\) until the inequalities collapse to equality. Order alone carries the whole structure.

The Python illustration below verifies the theorem on a small finite complete lattice, the power set of \(\{0, 1, 2\}\) ordered by inclusion. The code defines a monotone map with several fixed points, computes the least and greatest fixed points using the Knaster–Tarski formulas, checks that they really are fixed, and checks that the fixed-point set is closed under joins and meets.

```python
from itertools import chain, combinations

# Power set of {0, 1, 2} as a complete lattice under inclusion.
universe = {0, 1, 2}
all_sets = [frozenset(s) for s in chain.from_iterable(combinations(universe, r) for r in range(4))]

def f(S):
    # A deliberately non-linear monotone map with several fixed points.
    # Fixed points are exactly the sets that contain 0 and contain 2 whenever they contain 1.
    return S | {0} | ({2} if 1 in S else set())

# Verify monotonicity on the whole lattice: A <= B implies f(A) <= f(B).
assert all(
    not (A <= B) or f(A) <= f(B)
    for A in all_sets for B in all_sets
)

P = [S for S in all_sets if f(S) <= S]   # pre-fixed points
Q = [S for S in all_sets if S <= f(S)]   # post-fixed points

lfp = frozenset.intersection(*P)
gfp = frozenset.union(*Q)

assert f(lfp) == lfp, "lfp is fixed"
assert f(gfp) == gfp, "gfp is fixed"

fixed = [S for S in all_sets if f(S) == S]
assert lfp in fixed and gfp in fixed
assert all(lfp <= s for s in fixed), "lfp is least"
assert all(s <= gfp for s in fixed), "gfp is greatest"

# Verify that the fixed-point set is closed under joins and meets in itself.
def meet_in_fixed(A):
    return frozenset.intersection(*A)

def join_in_fixed(A):
    return frozenset.union(*A)

# For every family of fixed points, their meet and join in the lattice are already fixed.
for r in range(len(fixed) + 1):
    for family in combinations(fixed, r):
        m = meet_in_fixed(family) if family else gfp
        j = join_in_fixed(family) if family else lfp
        assert f(m) == m
        assert f(j) == j

print("lfp =", sorted(lfp))
print("gfp =", sorted(gfp))
print("all fixed points:", [sorted(s) for s in fixed])
print("Knaster-Tarski verification passed.")
```

Running this script confirms that the least fixed point is \(\{0\}\), the greatest fixed point is \(\{0, 1, 2\}\), and the fixed-point set is exactly the family of sets containing \(0\) and containing \(2\) whenever they contain \(1\). It also confirms that arbitrary intersections and unions of fixed points remain fixed, which is the finite manifestation of the theorem’s claim that \(\text{Fix}(f)\) is a complete lattice. The Knaster–Tarski fixed-point theorem is the right statement whenever order, rather than metric or topological structure, is the primitive that governs a self-map.
