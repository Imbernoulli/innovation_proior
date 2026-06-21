The question is whether a strong, effectively axiomatized formal system for arithmetic can be complete and can also certify its own consistency from the inside. Hilbert's program hoped the answer was yes: every arithmetical truth should be derivable, and consistency should be provable by finitary means. The baselines all miss the target. The liar paradox shows that self-reference plus a truth predicate can lead to contradiction, but it gives no arithmetical sentence and no theorem about formal systems. External metamathematical reasoning can describe proofs from the outside, yet it leaves the system itself unable to speak about its own syntax. Ordinary diagonalization, such as Cantor's, constructs objects that differ from every member of a list, but it does not turn a theory's proof predicate back on itself. What is missing is a way to make arithmetic talk about formulas, proofs, and provability while remaining entirely inside arithmetic.

The method that closes this gap is Gödel's incompleteness theorems. The first move is the arithmetization of syntax. Every symbol, every finite string of symbols, and every finite sequence of strings is assigned a natural number. With a careful choice of coding, the syntactic operations that make up proofs become primitive recursive relations on numbers. Because a sufficiently strong arithmetic theory can represent primitive recursive relations, the external statement "p codes a proof in T of the formula whose code is n" becomes an arithmetical formula Proof_T(p, n). Provability is then expressed as Prov_T(n), meaning there exists a p such that Proof_T(p, n) holds. The theory has not introduced a truth predicate; it has introduced a precise, mechanical proxy for its own derivability relation.

Once the proof predicate is internalized, the fixed-point lemma supplies the self-reference. For any arithmetical formula A(x) with one free variable, the lemma constructs a sentence S such that T proves S if and only if A holds of the Gödel number of S. In other words, T proves S <-> A(code(S)). Choosing A(x) to be not Prov_T(x) yields the Gödel sentence G_T. Formally, G_T asserts that no natural number codes a proof of the formula whose code is code(G_T). Informally, it says "I am not provable in T." If T were to prove G_T, it would prove a sentence asserting the nonexistence of that very proof; under the relevant consistency assumptions, this is impossible. Therefore T does not prove G_T. With enough soundness about proof facts, G_T is true in the intended model.

The second incompleteness theorem pushes the same idea further. The theory can formulate its own consistency statement Con(T), usually as the claim that no number codes a proof of a contradiction. In the metatheory one shows that T can prove Con(T) -> G_T. If T also proved Con(T), then T would prove G_T, contradicting the first theorem. Hence a consistent sufficiently strong T cannot prove its own consistency. The whole construction is not a dressed-up liar paradox: it avoids semantic truth, relies on the finitary, checkable nature of proof, and produces a specific arithmetical sentence that the system leaves undecided.

The code below illustrates the pipeline in a toy setting. It defines a tiny formal language, encodes expressions as Gödel numbers, implements substitution, builds the fixed point for the formula "x is not provable," and checks a small proof predicate. The toy theory is too weak to be a real instance of Gödel's theorem, but it makes the arithmetization and self-reference concrete and executable.

```python
import math
from itertools import product

# ---------- Pairing function for Gödel numbering ----------
def pair(x, y):
    """Cantor pairing function."""
    return (x + y) * (x + y + 1) // 2 + y

def unpair(z):
    """Inverse Cantor pairing."""
    w = int((math.isqrt(8 * z + 1) - 1) // 2)
    t = w * (w + 1) // 2
    y = z - t
    x = w - y
    return x, y

# ---------- Syntax (nested tuples) ----------
# ('var', n), ('num', n), ('zero',), ('succ', e), ('eq', e1, e2),
# ('not', e), ('and', e1, e2), ('forall', n, e), ('provable', e),
# ('sub', formula_expr, var_index, term_expr).

def encode(expr):
    """Encode an expression as a Gödel number."""
    tag = expr[0]
    if tag == 'var':
        return pair(1, expr[1])
    if tag == 'num':
        return pair(2, expr[1])
    if tag == 'zero':
        return pair(3, 0)
    if tag == 'succ':
        return pair(4, encode(expr[1]))
    if tag == 'eq':
        return pair(5, pair(encode(expr[1]), encode(expr[2])))
    if tag == 'not':
        return pair(6, encode(expr[1]))
    if tag == 'and':
        return pair(7, pair(encode(expr[1]), encode(expr[2])))
    if tag == 'forall':
        return pair(8, pair(expr[1], encode(expr[2])))
    if tag == 'provable':
        return pair(9, encode(expr[1]))
    if tag == 'sub':
        return pair(10, pair(encode(expr[1]), pair(expr[2], encode(expr[3]))))
    raise ValueError(f"unknown tag {tag}")

def decode(n):
    """Decode a Gödel number back into an expression."""
    tag, rest = unpair(n)
    if tag == 1:
        return ('var', rest)
    if tag == 2:
        return ('num', rest)
    if tag == 3:
        return ('zero',)
    if tag == 4:
        return ('succ', decode(rest))
    if tag == 5:
        a, b = unpair(rest)
        return ('eq', decode(a), decode(b))
    if tag == 6:
        return ('not', decode(rest))
    if tag == 7:
        a, b = unpair(rest)
        return ('and', decode(a), decode(b))
    if tag == 8:
        v, body = unpair(rest)
        return ('forall', v, decode(body))
    if tag == 9:
        return ('provable', decode(rest))
    if tag == 10:
        f, rest2 = unpair(rest)
        v, t = unpair(rest2)
        return ('sub', decode(f), v, decode(t))
    raise ValueError(f"unknown tag {tag}")

# ---------- Substitution (meta-level) ----------
def substitute(expr, var, term):
    """Substitute term for every free occurrence of var in expr."""
    tag = expr[0]
    if tag == 'var':
        return term if expr[1] == var else expr
    if tag in ('num', 'zero'):
        return expr
    if tag == 'succ':
        return ('succ', substitute(expr[1], var, term))
    if tag == 'eq':
        return ('eq', substitute(expr[1], var, term), substitute(expr[2], var, term))
    if tag == 'not':
        return ('not', substitute(expr[1], var, term))
    if tag == 'and':
        return ('and', substitute(expr[1], var, term), substitute(expr[2], var, term))
    if tag == 'forall':
        if expr[1] == var:
            return expr
        return ('forall', expr[1], substitute(expr[2], var, term))
    if tag == 'provable':
        return ('provable', substitute(expr[1], var, term))
    if tag == 'sub':
        return ('sub', substitute(expr[1], var, term), expr[2], substitute(expr[3], var, term))
    raise ValueError(f"unknown tag {tag}")

def substitute_code(code, var, term):
    """Substitute term into the formula with the given code and return the new code."""
    return encode(substitute(decode(code), var, term))

def eval_sub(sub_expr):
    """Evaluate an object-level substitution term ('sub', formula_expr, var, term_expr)."""
    if sub_expr[0] != 'sub':
        raise ValueError("expected sub term")
    formula_code = sub_expr[1][1]  # formula_expr must be ('num', code)
    var = sub_expr[2]
    term_expr = sub_expr[3]
    return substitute_code(formula_code, var, term_expr)

# ---------- Toy proof predicate ----------
# In a real system this would be a primitive recursive predicate verified by T.
# Here we use a tiny finite set of "axioms" and a bounded search for derivations.
AXIOMS = {
    # A harmless tautology: 0 = 0
    encode(('eq', ('zero',), ('zero',))),
}

def modus_ponens(a, b):
    """If a is A and b is (A -> C), return C. Our toy -> is encoded as (not (and A (not C)))."""
    da = decode(a)
    db = decode(b)
    if db[0] == 'not' and db[1][0] == 'and' and encode(db[1][1]) == a and db[1][2][0] == 'not':
        return encode(db[1][2][1])
    return None

def is_proof(derivation, goal):
    """Check whether derivation, a list of formula codes, is a proof of goal."""
    if derivation[-1] != goal:
        return False
    seen = set()
    for code in derivation:
        if code in AXIOMS:
            seen.add(code)
            continue
        derived = False
        for earlier in list(seen):
            c = modus_ponens(earlier, code)
            if c is not None and c == code:
                derived = True
                break
            c = modus_ponens(code, earlier)
            if c is not None and c == code:
                derived = True
                break
        if not derived:
            return False
        seen.add(code)
    return True

def provable(goal, max_len=6):
    """Bounded search for a proof of goal."""
    universe = list(AXIOMS)
    for length in range(1, max_len + 1):
        for deriv in product(universe, repeat=length):
            if is_proof(list(deriv), goal):
                return True
    return False

# ---------- Fixed-point construction ----------
x = 0  # variable in A(x)
y = 1  # variable in B(y)

# A(x) = not Provable(x)
A = ('not', ('provable', ('var', x)))

# B(y) = A[ sub(y, y) / x ]
# The inner sub(y, y) is the object-level substitution of numeral y into formula y.
B = substitute(A, x, ('sub', ('var', y), y, ('var', y)))

code_B = encode(B)

# Diagonal step: G = B(code_B)
G = substitute(B, y, ('num', code_B))
code_G = encode(G)

# Evaluate the inner substitution to see what G actually claims.
inner_sub = G[1][1]  # ('sub', ('num', code_B), y, ('num', code_B))
claimed_code = eval_sub(inner_sub)

print("Code of B:", code_B)
print("Gödel sentence code:", code_G)
print("G claims this code is unprovable:", claimed_code)
print("Claimed code equals actual code?", claimed_code == code_G)
print("Is G provable in the toy theory?", provable(code_G))
print("Equivalent assertion: not provable(code(G)) =", not provable(code_G))
```
