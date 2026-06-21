The question is whether every infinite set can be put into one-to-one correspondence with the positive integers. Taking "same size" seriously as the existence of a bijection, many apparently larger collections turn out to be listable: the rationals can be enumerated by sweeping through finite diagonals of numerator-denominator pairs, and the real algebraic numbers can be enumerated by ordering integer polynomials by a finite-at-each-level "height" and listing their roots. These positive listing techniques work well for showing that a set is countable, but they give no way to prove the negative statement that no list can exist. The problem needs a method that, given an arbitrary claimed enumeration, produces a concrete witness missing from the list.

Earlier attempts to trap such a witness lean heavily on the structure of the real line. One can squeeze nested intervals around the listed reals and invoke completeness of the real numbers to extract a limiting missed point. That argument is correct, but it depends on order, intervals, monotone convergence, and a case analysis of whether the nested intervals collapse to a point or end with positive length. It is also tied to the continuum: it says nothing about whether a non-numeric set might resist enumeration. What is wanted is a purely combinatorial move that works for any indexed family where each listed object exposes a reversible binary choice at its own index.

The new method is Cantor's diagonal argument. Its core operation is to take any supposed list of objects, view it as an array whose rows are the listed objects and whose columns are indexed by the same positions, read down the main diagonal, and reverse every entry along that diagonal. The resulting object differs from the n-th listed object in the n-th coordinate, so it cannot equal any of them. Because the construction only requires that each listed object have a readable, flip-able binary value at its corresponding index, the same move applies both to infinite binary sequences and, in a slightly more abstract form, to subsets of an arbitrary set.

For concreteness, consider the set of all infinite sequences over two symbols, written as m and w. Suppose someone claims to have listed all such sequences as E_1, E_2, E_3, ..., where E_μ = (a_{μ,1}, a_{μ,2}, a_{μ,3}, ...). Build a new sequence E_0 = (b_1, b_2, b_3, ...) by setting b_ν equal to the opposite symbol of a_{ν,ν}. Then for each ν, b_ν differs from the diagonal entry a_{ν,ν}. If E_0 were equal to some listed E_μ, it would have to agree with E_μ in every coordinate, including coordinate μ; but by construction b_μ differs from a_{μ,μ}. So E_0 is not in the list. Since the list was arbitrary, the set of all two-symbol sequences is uncountable. The reals in an interval follow by encoding each two-symbol sequence as a decimal using only the digits 1 and 2, which is injective because the first differing position dominates the tail of the expansion.

The same move lifts cleanly to a fully general theorem about set size. For any set X, let P(X) be its power set, the set of all subsets of X. The map sending each element x to the singleton {x} injects X into P(X), so X is no larger than its power set. To show it is strictly smaller, take any function f from X to P(X); for each x, f(x) is a subset of X. Form the anti-diagonal subset T = { x ∈ X : x ∉ f(x) }. For any fixed x, if x is in T then the definition of T says x is not in f(x), so T and f(x) disagree at x; if x is not in T then x must be in f(x), so they again disagree at x. Thus T cannot equal f(x) for any x, which means f is not surjective. No bijection from X to P(X) exists, and therefore |X| < |P(X)|. This applies to every set, so there is no largest cardinality.

```python
from typing import List, Callable, Set, TypeVar

# --- Concrete diagonal argument on binary sequences ---

def flip(symbol: str) -> str:
    """Return the other symbol in the two-symbol alphabet {m, w}."""
    return "w" if symbol == "m" else "m"

def diagonal_counterexample(claimed_list: List[List[str]]) -> List[str]:
    """
    Given a finite or infinite list of infinite two-symbol sequences,
    return a new sequence that differs from the n-th listed sequence
    at position n.
    """
    result: List[str] = []
    for index, row in enumerate(claimed_list):
        # Ensure every row is long enough to read its diagonal entry.
        while len(row) <= index:
            row.append("m")  # pad with a default symbol if needed
        result.append(flip(row[index]))
    return result

def is_missing(witness: List[str], claimed_list: List[List[str]]) -> bool:
    """
    Verify that the diagonal witness differs from every row of the
    claimed list at the corresponding diagonal position.
    """
    for i, row in enumerate(claimed_list):
        if i >= len(witness):
            break
        if len(row) <= i:
            return True
        if witness[i] == row[i]:
            return False
    return True

# Runnable finite demo: a claimed "list" of binary sequences.
claimed_finite = [
    ["m", "m", "m"],
    ["w", "w", "w"],
    ["m", "w", "m"],
]

witness = diagonal_counterexample(claimed_finite)
print("Witness:", witness)
print("Differs from every row:", is_missing(witness, claimed_finite))

# --- General power-set argument ---

T = TypeVar('T')

def power_set_counterexample(
    X: Set[T],
    f: Callable[[T], Set[T]]
) -> Set[T]:
    """
    Given any function f : X -> P(X), return the anti-diagonal subset
    T = { x in X : x not in f(x) }, which is missed by f.
    """
    return {x for x in X if x not in f(x)}

def is_missed(T: Set[T], X: Set[T], f: Callable[[T], Set[T]]) -> bool:
    """Verify that T is not equal to f(x) for any x in X."""
    return all(T != f(x) for x in X)

# Example: let X = {1, 2, 3} and define an arbitrary f : X -> P(X).
X = {1, 2, 3}
f = lambda x: {1} if x == 1 else ({1, 2} if x == 2 else set())

T = power_set_counterexample(X, f)
print("Anti-diagonal subset T:", T)
print("T is missed by f:", is_missed(T, X, f))
```
