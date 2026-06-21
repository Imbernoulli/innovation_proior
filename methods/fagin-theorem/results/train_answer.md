The problem is to describe the complexity class NP without starting from Turing machines, tape heads, or polynomial clocks. A class defined by machines is useful for algorithms, but it hides the logical shape of the properties being recognized. A graph, a database instance, or any finite relational structure should be the real object of study, and the answer should not depend on how the elements happen to be named.

The natural first attempt is to use classical spectra: the set of cardinalities of finite models of a first-order sentence. But spectra recognize a set of numbers, not a class of structures. Because the binary notation for a cardinality n has length only log n, a brute-force search over relations on n elements looks exponential in the input length. That mismatch pushes spectra toward nondeterministic exponential time, not ordinary NP. What is needed is a setting where the input already carries the polynomial-sized relation tables of a finite structure, so that guessing an extra finite relation costs only polynomially many bits.

The right bridge is Fagin's theorem. It says that, for every nonempty finite vocabulary and every isomorphism-closed class of finite structures over that vocabulary, the class is a generalized spectrum if and only if its standard string encodings are in NP. In short, NP equals existential second-order definability: NP = SO∃.

A defining sentence has the form ∃R1 ... ∃Rk φ, where R1 through Rk are auxiliary relation variables of fixed arity and φ is a first-order sentence over the original vocabulary together with these new relations. The existential second-order quantifiers are the logical counterpart of nondeterministic guessing: the machine's certificate is replaced by a finite relational witness living on the same universe as the input structure. The first-order part φ is the logical counterpart of polynomial-time local verification: it can express that the guessed relations satisfy the required constraints.

The forward direction is immediate. Given such a sentence, a nondeterministic polynomial-time algorithm guesses the interpretation tables for R1,...,Rk and then evaluates φ on the expanded structure. Because the arities and the formula are fixed, the guessed tables have size polynomial in the input structure, and first-order evaluation is polynomial.

The reverse direction is the deeper half. Take a nondeterministic Turing machine running in time n^k on inputs of size n. Represent the accepting computation as finite relations over the input universe. Use k-tuples of elements to index time instants and tape positions, so the tableau has polynomially many cells. Existentially quantify relations that record, for each cell at each time, the symbol written there and, when the head is present, the current state. Another relation can record the nondeterministic choices if that simplifies the transition check. The first-order formula then asserts that the first row encodes the input, that every cell has a unique content, that each row follows from the previous one by the machine's local transition rules, and that an accepting state appears at the final time. Because an arbitrary finite structure need not come with an order, the auxiliary witness also guesses a linear order on the universe; this is just bookkeeping for tuple indexing and does not add computational power beyond the second-order existential guess.

The theorem therefore identifies the certificate of an NP problem with finite relational structure. Nondeterminism becomes existential quantification over relations, and polynomial verification becomes first-order local checking. The empty-vocabulary case remains the classical spectrum problem and stays at the nondeterministic exponential-time scale, but for any real input vocabulary the match is exact.

```python
import itertools

class Structure:
    def __init__(self, universe, relations):
        self.universe = list(universe)
        self.relations = {name: set(tups) for name, tups in relations.items()}

def evaluate(structure, assignment, formula):
    if formula[0] == 'forall':
        _, var, sub = formula
        return all(evaluate(structure, {**assignment, var: a}, sub)
                   for a in structure.universe)
    if formula[0] == 'exists':
        _, var, sub = formula
        return any(evaluate(structure, {**assignment, var: a}, sub)
                   for a in structure.universe)
    if formula[0] == 'and':
        return all(evaluate(structure, assignment, f) for f in formula[1:])
    if formula[0] == 'or':
        return any(evaluate(structure, assignment, f) for f in formula[1:])
    if formula[0] == 'not':
        return not evaluate(structure, assignment, formula[1])
    if formula[0] == 'implies':
        return (not evaluate(structure, assignment, formula[1])) \
               or evaluate(structure, assignment, formula[2])
    # atomic relation
    rel, *vars = formula
    tup = tuple(assignment[v] for v in vars)
    return tup in structure.relations.get(rel, set())

def guess_relation(universe, arity):
    elems = list(universe)
    domains = [elems] * arity
    all_tups = list(itertools.product(*domains))
    for bits in itertools.product([False, True], repeat=len(all_tups)):
        yield {tup for tup, b in zip(all_tups, bits) if b}

def solve_so_exists(structure, sentence):
    """
    sentence = [(R1, arity1), ..., (Rk, arityk), phi]
    Nondeterministically guess the auxiliary relations and evaluate phi.
    """
    guesses = [guess_relation(structure.universe, arity)
               for _, arity in sentence[:-1]]
    for combo in itertools.product(*guesses):
        expanded = Structure(structure.universe, dict(structure.relations))
        for (name, _), rel in zip(sentence[:-1], combo):
            expanded.relations[name] = rel
        if evaluate(expanded, {}, sentence[-1]):
            return True, expanded.relations
    return False, None

# Example: 3-colorability as an SO-exists sentence over graphs.
one_color = ('or', ('R', 'x'), ('or', ('G', 'x'), ('B', 'x')))
distinct = ('and',
            ('not', ('and', ('R', 'x'), ('G', 'x'))),
            ('not', ('and', ('R', 'x'), ('B', 'x'))),
            ('not', ('and', ('G', 'x'), ('B', 'x'))))
proper = ('and',
          ('implies', ('and', ('E', 'x', 'y'), ('R', 'x')), ('not', ('R', 'y'))),
          ('implies', ('and', ('E', 'x', 'y'), ('G', 'x')), ('not', ('G', 'y'))),
          ('implies', ('and', ('E', 'x', 'y'), ('B', 'x')), ('not', ('B', 'y'))))
phi = ('forall', 'x',
       ('and', one_color, distinct,
        ('forall', 'y', proper)))
three_colorable = [('R', 1), ('G', 1), ('B', 1), phi]

# A triangle is not 3-colorable with the usual meaning? It is (3 colors for 3 vertices).
# A 4-cycle is 2-colorable, hence 3-colorable.
graph = Structure({0, 1, 2, 3}, {'E': {(0, 1), (1, 2), (2, 3), (3, 0)}})
accepted, witness = solve_so_exists(graph, three_colorable)
print("4-cycle 3-colorable?", accepted)
```
