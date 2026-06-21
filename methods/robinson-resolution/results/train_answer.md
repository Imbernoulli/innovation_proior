I propose the canonical name Robinson resolution for this method. Robinson resolution is the refutation calculus that reduces first-order logical proof search to repeated application of a single inference rule—resolution—made first-order by most general unification. The essential idea is to take the axioms of a theory, add the negation of the statement we want to prove, normalize the whole collection into a set of clauses, and then search for a way to derive the empty clause. When the empty clause appears, the original assumption set is unsatisfiable, so the negated goal cannot hold and the goal must be true.

The first representational move is conversion to clause form. Implications and quantifiers are eliminated or standardized: implications are rewritten with disjunction and negation, negations are pushed inward, variables are renamed apart across clauses, and existential quantifiers are replaced by Skolem functions. The result is a conjunction of clauses, each clause being a disjunction of literals. Universal quantifiers become implicit, because every variable in a clause is understood to be universally quantified. This normalization strips away most of the surface variety of first-order syntax and leaves a uniform data structure: a set of disjunctions of atomic literals and their negations.

The second move is refutation. Rather than trying to build a direct proof of the goal from the axioms, Robinson resolution asks whether the axioms together with the negated goal can lead to a contradiction. This reformulation is powerful because it removes the need to guess helpful intermediate lemmas or to choose among many introduction and elimination rules. The entire task is reduced to driving the clause set to inconsistency.

The single inference rule is resolution. In the propositional case, if one clause contains a literal A and another clause contains its negation not A, resolving the two clauses produces the disjunction of the remaining literals. The first-order case is the same except that the complementary literals may not be textually identical. One clause may contain P(x) and another may contain not P(a). Before resolution can occur, the terms must be matched. That matching is done by unification.

Unification computes a substitution that makes two first-order terms identical whenever such a substitution exists. Robinson resolution uses the most general unifier, the substitution that makes the terms match while committing to as little as possible. For example, P(x) and P(f(y)) unify with x := f(y), which is more general than any ground substitution such as x := f(a). Keeping the substitution general preserves future proof options. The resolution step therefore takes two clauses C or L and D or not L', finds the most general unifier theta of L and L', and derives (C or D)theta. Factoring can also collapse redundant unifiable literals inside a single clause.

The proof search is then a saturation loop. A prover repeatedly selects clauses, attempts to resolve pairs, adds any new resolvents back to the clause set, and halts successfully when it derives the empty clause. The loop is not a decision procedure: first-order validity is only semi-decidable, so a fair search will eventually find a refutation for any unsatisfiable clause set, but it may run forever when no refutation exists. The completeness theorem is what makes the method a calculus rather than a heuristic. If a theorem is valid, there exists a resolution derivation of the empty clause from the appropriate clause set.

Robinson resolution had a broad influence on artificial intelligence and automated reasoning. It converted the diffuse activity of logical deduction into a concrete symbolic engine. Instead of encoding many domain-specific rules of inference, a system could normalize its knowledge into clauses and rely on one general inference loop. Modus ponens, case analysis, syllogistic chaining, and quantifier instantiation all become special cases of resolution after clause conversion. The practical engineering problem shifted toward search control: which clauses to keep, which pairs to resolve next, how to index terms, and how to limit combinatorial growth. Later refinements such as paramodulation and superposition extended the same framework to handle equality efficiently.

A small Python illustration captures the central operations. It defines unification for simple first-order terms and performs a single resolution proof of the classic syllogism: Socrates is a man, all men are mortal, therefore Socrates is mortal. The code converts the problem into clauses, resolves the clause for Socrates being a man with the clause for all men being mortal, and then resolves the result with the negated goal to obtain the empty clause.

```python
def unify(x, y, theta=None):
    if theta is None:
        theta = {}
    # Variables are strings whose first character is uppercase.
    if isinstance(x, str) and x[0].isupper():
        return unify_var(x, y, theta)
    if isinstance(y, str) and y[0].isupper():
        return unify_var(y, x, theta)
    if isinstance(x, tuple) and isinstance(y, tuple):
        if x[0] != y[0] or len(x) != len(y):
            return None
        for xi, yi in zip(x[1:], y[1:]):
            theta = unify(xi, yi, theta)
            if theta is None:
                return None
        return theta
    return theta if x == y else None

def unify_var(var, term, theta):
    if var in theta:
        return unify(theta[var], term, theta)
    if var == term:
        return theta
    if occurs(var, term, theta):
        return None
    theta[var] = term
    return theta

def occurs(var, term, theta):
    if var == term:
        return True
    if isinstance(term, str) and term[0].isupper() and term in theta:
        return occurs(var, theta[term], theta)
    if isinstance(term, tuple):
        return any(occurs(var, arg, theta) for arg in term[1:])
    return False

def apply(term, theta):
    if isinstance(term, str) and term[0].isupper():
        return apply(theta.get(term, term), theta) if term in theta else term
    if isinstance(term, tuple):
        return (term[0],) + tuple(apply(arg, theta) for arg in term[1:])
    return term

def resolve(clause1, clause2):
    """Resolve two clauses represented as sets of signed literals.
    Positive literal: ('+', predicate_term)
    Negative literal: ('-', predicate_term)
    """
    for s1, p1 in clause1:
        for s2, p2 in clause2:
            if s1 == s2:
                continue
            theta = unify(p1, p2)
            if theta is None:
                continue
            resolvent = set()
            for s, p in clause1:
                if (s, p) != (s1, p1):
                    resolvent.add((s, apply(p, theta)))
            for s, p in clause2:
                if (s, p) != (s2, p2):
                    resolvent.add((s, apply(p, theta)))
            return resolvent, theta
    return None, None

# Syllogism: Man(socrates); if Man(x) then Mortal(x); prove Mortal(socrates).
Man = lambda t: ('Man', t)
Mortal = lambda t: ('Mortal', t)
Socrates = 'socrates'
X = 'X'

clause1 = {('+', Man(Socrates))}
clause2 = {('-', Man(X)), ('+', Mortal(X))}
goal_neg = {('-', Mortal(Socrates))}

r1, theta1 = resolve(clause1, clause2)
print('Resolvent 1:', r1, 'substitution:', theta1)

r2, theta2 = resolve(r1, goal_neg)
print('Resolvent 2:', r2, 'substitution:', theta2)
assert r2 == set(), 'Expected empty clause: refutation is complete'
```

The derivation starts from the clause Man(socrates) and the implication clause not Man(X) or Mortal(X). Resolving these clauses unifies X with socrates and yields Mortal(socrates). Resolving that resolvent with the negated goal not Mortal(socrates) then yields the empty clause, completing the refutation. This is exactly the chain Robinson resolution performs at machine scale, using only clause form, unification, and resolution.
