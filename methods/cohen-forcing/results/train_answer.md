The continuum hypothesis asks whether there is an infinite set of real numbers whose cardinality is strictly between that of the natural numbers and the full power set of the naturals. By the early 1960s this had become a question about models of set theory: if a statement can be made true in one model of ZFC and false in another, then it cannot be decided by ZFC alone. Kurt Gödel had already supplied one half of the answer by constructing the inner model L, the constructible universe, in which the axiom of choice and the generalized continuum hypothesis both hold. That showed ZFC cannot disprove the continuum hypothesis, because any contradiction with CH derived inside ZFC could be translated into a contradiction in ZFC alone. But an inner model only removes sets; it cannot manufacture the extra subsets of omega that would make the continuum larger. So the missing direction required an outer construction, and the naive idea of simply adjoining new subsets from the outside was uncontrolled. A new subset chosen externally can encode facts about the original model that the original model cannot audit, and closing under definability after such an addition gives no guarantee that Replacement, Power Set, or the intended cardinal structure survive.

What is needed is a way to enlarge the model while keeping every truth of the enlarged model accountable to finite information already present in the ground model. The method that achieves this is Cohen forcing. It begins with a countable transitive ground model M of ZFC and a partial order P of finite conditions. For adding many new reals, a condition is a finite partial function from aleph_tau^M times omega into 2, making finitely many decisions of the form n belongs to a_delta or not. These conditions are ordered by extension, so stronger conditions commit to more bits. Instead of choosing the final new subsets directly, one builds a filter G inside P that meets every dense subset of P that lies in M. Because M is countable externally, its dense requirements can be enumerated and satisfied one by one, producing a generic filter whose union defines new binary sequences a_delta. The new model M[G] is not built by vague set-theoretic closure but by evaluating P-names from M through G. A P-name is a set of pairs of earlier names and conditions, so every object in the extension comes with a finite certificate from the ground model.

The forcing relation, written p forces phi, is defined by recursion on formulas inside M. A condition forces an existential statement when it already commits to a named witness, and it forces a universal statement when no stronger condition can force a counterexample. The central forcing theorem then says that a statement phi is true in M[G] exactly when some condition in G forces phi. This pulls every semantic assertion about the extension back to a finite syntactic fact in M. With the forcing theorem in hand, one proves the ZFC axioms in M[G] directly rather than assuming them. Replacement follows because definable functions over a fixed name have witnesses that can be bounded below a fixed generated rank. Power Set follows from the analysis of incompatible conditions and closure of index sets. Choice follows from the definable well-order of names carried over from a constructible ground model. Finally, the construction preserves the old cardinals needed to measure the continuum, and with tau equal to 2 one obtains a model of ZFC in which 2 to the aleph_0 equals aleph_2, so the continuum hypothesis fails. Together with Gödel's constructible model, this establishes the independence of CH from ZFC.

```python
from itertools import count

class Condition:
    """A finite partial decision function (real_index, n) -> bool."""
    def __init__(self, decisions=None):
        self.d = dict(decisions) if decisions else {}

    def extends(self, other):
        """True if self contains every decision made by other."""
        return all(self.d.get(k) == v for k, v in other.d.items())

    def add(self, key, value):
        if key in self.d and self.d[key] != value:
            raise ValueError("incompatible decisions")
        new = Condition(self.d)
        new.d[key] = value
        return new


def distinctness_dense(i, j, bits):
    """Dense set forcing a_i and a_j to differ below bits."""
    def D(cond):
        for k in range(bits):
            vi = cond.d.get((i, k))
            vj = cond.d.get((j, k))
            if vi is not None and vj is not None and vi != vj:
                return cond
        for k in range(bits):
            vi = cond.d.get((i, k))
            vj = cond.d.get((j, k))
            if vi is None and vj is None:
                return cond.add((i, k), True).add((j, k), False)
            if vi is None:
                return cond.add((i, k), not vj)
            if vj is None:
                return cond.add((j, k), not vi)
        raise RuntimeError("cannot separate without flipping a decided bit")
    return D


def totality_dense(i, n):
    """Dense set deciding bit n of real i."""
    def D(cond):
        if (i, n) in cond.d:
            return cond
        return cond.add((i, n), True)
    return D


def build_generic(num_reals, bits):
    """Meet distinctness requirements first, then totality requirements."""
    dense = []
    for i in range(num_reals):
        for j in range(i):
            dense.append(distinctness_dense(i, j, bits))
    for i in range(num_reals):
        for n in range(bits):
            dense.append(totality_dense(i, n))
    p = Condition()
    for D in dense:
        p = D(p)
    return p


def evaluate_name(name, cond):
    """Evaluate a P-name under a condition that forces all needed facts."""
    seen = set()

    def val(n):
        nid = id(n)
        if nid in seen:
            raise RecursionError("ill-founded name")
        seen.add(nid)
        result = set()
        for member, c in n:
            if cond.extends(c):
                result.add(val(member) if isinstance(member, set) else member)
        seen.discard(nid)
        return frozenset(result)

    return val(name)


def real_name(i, bits):
    """Name for the i-th added Cohen real as a subset of {0,...,bits-1}."""
    return {(n, Condition({(i, n): True})) for n in range(bits)}


if __name__ == "__main__":
    NUM_REALS, BITS = 3, 8
    generic = build_generic(NUM_REALS, BITS)
    reals = [evaluate_name(real_name(i, BITS), generic) for i in range(NUM_REALS)]

    for i, r in enumerate(reals):
        print(f"real {i}: {sorted(r)}")

    for i in range(NUM_REALS):
        for j in range(i + 1, NUM_REALS):
            assert reals[i] != reals[j], "generic reals must be distinct"
    print("All generic reals are distinct.")
```
