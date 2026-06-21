The problem is whether a single formal system rich enough for ordinary arithmetic can be both consistent and complete, and whether it can prove its own consistency. Hilbert's program in the late 1920s demanded exactly this: a finitary consistency proof together with the belief that the standard formal systems of the day, such as Principia Mathematica, Zermelo–Fraenkel set theory, and Peano arithmetic, already decided every arithmetical statement. Existing approaches all left the question open. Direct finitary consistency proofs made progress only for fragments of arithmetic and said nothing about completeness. The model-theoretic completeness route succeeded for pure first-order logic but failed for arithmetic because arithmetic has many non-standard models, so the existence of a model does not force every sentence to be decided. A truth-definitional reduction of consistency collapsed because an arithmetical truth predicate reproduces the Liar paradox through diagonalization. And structural stratification, as in Principia Mathematica's ramified type hierarchy, blocked some object-level self-reference but placed no constraint on statements about the system's own syntax, which is where the real action turns out to be.

The method that resolves the question is Gödel's incompleteness theorems. The decisive move is to replace the undefinable semantic notion of truth with the expressible syntactic notion of provability. Truth obeys the rigid equivalence that a sentence is true exactly when the statement that it is true holds, which makes diagonalization explode into the Liar. Provability is different: a proof is a finite combinatorial object, so the relation "x is a proof of y" is decidable by inspection, and "y is provable" is simply one existential quantifier over that decidable relation. This asymmetry is the hinge of the whole argument.

The first step is the arithmetization of syntax. Assign numerical codes to basic symbols, code variables by prime powers, and code a string of symbols by prime powers again, so that a sequence of symbol-codes maps to a single natural number whose prime exponents are the symbols in order. A proof, being a finite sequence of formulas, is coded one level up in the same way. Unique factorization makes decoding effective. Next, one shows that the syntactic relations are primitive recursive: formula-hood, substitution, axiom-hood, immediate consequence, and the proof relation x B y are all decidable by bounded search over finite objects. Provability Bew(y) is the only non-primitive-recursive relation in the list, being an unbounded existential over the decidable proof relation, and that is exactly the gap where incompleteness lives.

The second step is representability: for every primitive recursive relation R there is a formula r of the system such that if R holds on numerals then r with those numerals substituted is provable, and if R fails then its negation is provable. This is proved by induction on the build-up of R, mirroring each computation step by a formal derivation from the arithmetical axioms. With the proof relation representable, the diagonal construction can be applied to provability rather than truth. Take Q(x,y) to mean "x is not a proof of the formula obtained by substituting the numeral of y into the y-th one-variable formula." This is primitive recursive, so it is represented by some formula q. Generalize q on x to get p, then substitute p's own numeral into p. The resulting sentence G asserts its own unprovability. Consistency alone makes G unprovable: a proof of G would yield a specific proof number n, and the representability machinery would make the system prove both the instance that n proves G and the instance that it does not, contradicting consistency. ω-consistency, a slightly stronger condition that forbids proving every numeral instance of a property while denying the universal statement, makes the negation of G unprovable as well.

Because the entire argument that consistency implies the unprovability of G can be carried out inside the system, the system proves "Consistency implies G." If it could also prove its own consistency, it would prove G, which it cannot. Therefore a consistent system strong enough for arithmetic cannot prove its own consistency. The undecidable sentence is not a semantic trick: it is a statement about ordinary whole numbers, since every primitive recursive relation can be expressed purely with addition, multiplication, equality, and quantifiers over the naturals.

```python
"""
Toy demonstration of Gödel numbering and the diagonal construction
underlying Gödel's first incompleteness theorem.
"""

# Symbol codes for a tiny formal language.
SYMBOLS = {'0': 1, 'S': 2, '=': 3, 'x': 4, '~': 5}

# First few primes for prime-power coding of symbol sequences.
PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]


def godel_number(s):
    """Encode a short string of symbols as a product of prime powers."""
    out = 1
    for i, ch in enumerate(s):
        out *= PRIMES[i] ** SYMBOLS[ch]
    return out


def decode_number(n):
    """Recover the sequence of symbol codes from its prime-power code."""
    seq = []
    for p in PRIMES:
        if n == 1:
            break
        e = 0
        while n % p == 0:
            n //= p
            e += 1
        seq.append(e)
    return seq


def decode_formula(n):
    """Recover the original formula string from its Gödel number."""
    inv = {v: k for k, v in SYMBOLS.items()}
    return ''.join(inv[c] for c in decode_number(n) if c)


def numeral(n):
    """Return the syntactic numeral for n: n copies of S followed by 0."""
    return 'S' * n + '0'


def substitute(formula, var, term):
    """Replace every occurrence of var in formula by term."""
    return formula.replace(var, term)


def diagonal(formula, var):
    """
    Diagonal construction: substitute the numeral of the formula's own
    Gödel number into the formula at var.  The resulting sentence speaks
    about its own code.
    """
    return substitute(formula, var, numeral(godel_number(formula)))


# A one-variable formula pattern.  In a real system this predicate on x
# would be the arithmetized statement "~Prov(x)"; here we keep the
# language tiny and let "~x=0" play the same structural role.
pattern = '~x=0'

# The Gödel sentence: it asserts "~(numeral of my own code)=0", i.e.
# in the intended interpretation it says "I am not provable".
G = diagonal(pattern, 'x')

# Round-trip check on a small formula.
test = 'Sx=0'
print("Round-trip:", test, "->", godel_number(test), "->", decode_formula(godel_number(test)))

print("Pattern:", pattern)
print("Pattern code:", godel_number(pattern))
print("G begins with:", G[:60])
print("G contains this many symbols:", len(G))
```
