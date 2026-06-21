I propose the canonical name "Church-Rosser theorem for lambda-calculus beta-reduction" for this result. In my final treatment I present it as the statement that beta-reduction is a confluent rewrite relation: whenever a lambda term M can be reduced in zero or more steps to N1 and also to N2, there exists a common term P such that both N1 and N2 can be further reduced to P. This single property organizes the entire computational geometry of the lambda calculus and justifies the separation between reduction strategy and equational meaning.

The lambda calculus models computation through three syntactic constructs: variables, abstractions written λx.M, and applications written (M N). Beta-reduction is the rule (λx.M) N → M[x := N], which substitutes the argument for the bound variable in the function body. Because a term may contain many redexes at the same time and because reducing one redex can create, duplicate, or reshape others, the same starting term admits many different reduction sequences. A redex is simply a subterm of the form (λx.M) N, and a reduction strategy is nothing more than a rule for choosing which redex to contract next.

Church-Rosser theorem says that this branching is not catastrophic. Formally, if M →* N1 and M →* N2 under beta-reduction, then there is some P with N1 →* P and N2 →* P. The theorem is often pictured as a diamond: two edges leave M, but the tips can be completed into a four-sided figure whose bottom vertex is P. In fact the one-step beta relation does not itself satisfy a simple diamond in the lambda calculus, because contracting one redex can duplicate or disable another. Standard proofs therefore introduce an auxiliary parallel reduction relation that contracts many redexes at once and does satisfy a diamond property, then transfer the diamond back to the transitive closure of ordinary beta-reduction.

A term in normal form has no remaining redexes and therefore cannot be reduced further. If a term M reached two distinct normal forms A and B, Church-Rosser would supply a common reduct P of A and B; but A and B are irreducible, so P must equal both A and B up to alpha-renaming of bound variables. Hence normal forms, when they exist, are unique. This is why different interpreters or proof strategies may take different routes and still agree on the final answer. Different reduction orders may produce different intermediate terms and may consume different numbers of steps, yet they cannot manufacture incompatible normal forms.

It is important that Church-Rosser is not a termination theorem. Some terms, such as the self-applicator (λx.x x)(λx.x x), admit infinite reduction sequences. Some strategies may reach a normal form while others loop forever, even when a normal form exists. The theorem only promises that any two finite prefixes of reduction can eventually be reconciled; it does not promise that every path stops. This distinction between confluence and strong normalization is what keeps the result compatible with non-terminating programs and with lazy evaluation, where many subterms are never reduced at all.

Conceptually, Church-Rosser replaces a linear notion of execution with a spatial one. A term is a node, each beta-step is a directed edge, and a reduction strategy is merely a path through this graph. Confluence is a global geometric invariant: the graph may fork, but forks are not permanent. This is why the lambda calculus can serve as a foundation for functional programming languages and proof normalization: the meaning of an expression is fixed by the confluent rewrite structure, while an evaluator remains free to choose call-by-name, call-by-value, lazy, or eager strategies. The theorem therefore separates the accidental details of how a result is obtained from the mathematical fact of what the result is.

To make the geometry concrete, I include a small Python illustration that defines lambda terms, performs capture-avoiding substitution, enumerates one-step beta reducts, and checks whether two reducts can still reach a common term within a bounded search depth. The chosen example is (λx.x x)((λy.y) z), which has two distinct one-step reducts. The script verifies that those reducts can be joined again, giving a miniature demonstration of the Church-Rosser property.

```python
from dataclasses import dataclass
from typing import Set

@dataclass(frozen=True)
class Var:
    name: str
    def __repr__(self): return self.name

@dataclass(frozen=True)
class Lam:
    var: str
    body: object
    def __repr__(self): return f"(λ{self.var}.{self.body})"

@dataclass(frozen=True)
class App:
    fn: object
    arg: object
    def __repr__(self): return f"({self.fn} {self.arg})"

def free_vars(t) -> Set[str]:
    if isinstance(t, Var): return {t.name}
    if isinstance(t, Lam): return free_vars(t.body) - {t.var}
    if isinstance(t, App): return free_vars(t.fn) | free_vars(t.arg)
    raise TypeError(type(t))

def fresh(used: Set[str], base: str = "x") -> str:
    i = 0
    while f"{base}{i}" in used:
        i += 1
    return f"{base}{i}"

def subst(t, var: str, value):
    if isinstance(t, Var):
        return value if t.name == var else t
    if isinstance(t, Lam):
        if t.var == var:
            return t
        if t.var not in free_vars(value):
            return Lam(t.var, subst(t.body, var, value))
        new_var = fresh(free_vars(t.body) | free_vars(value) | {t.var}, t.var)
        renamed_body = subst(t.body, t.var, Var(new_var))
        return Lam(new_var, subst(renamed_body, var, value))
    if isinstance(t, App):
        return App(subst(t.fn, var, value), subst(t.arg, var, value))
    raise TypeError(type(t))

def beta_steps(t) -> list:
    result = []
    if isinstance(t, App) and isinstance(t.fn, Lam):
        result.append(subst(t.fn.body, t.fn.var, t.arg))
    if isinstance(t, App):
        result.extend(App(r, t.arg) for r in beta_steps(t.fn))
        result.extend(App(t.fn, r) for r in beta_steps(t.arg))
    if isinstance(t, Lam):
        result.extend(Lam(t.var, r) for r in beta_steps(t.body))
    return result

def closure(term, max_depth: int = 6) -> Set[object]:
    seen = {term}
    frontier = [term]
    for _ in range(max_depth):
        next_frontier = []
        for u in frontier:
            for v in beta_steps(u):
                if v not in seen:
                    seen.add(v)
                    next_frontier.append(v)
        frontier = next_frontier
        if not frontier:
            break
    return seen

def can_join(a, b, max_depth: int = 6) -> bool:
    return bool(closure(a, max_depth) & closure(b, max_depth))

if __name__ == "__main__":
    #(λx. x x) ((λy. y) z) has two redexes: outer and inner.
    z = Var("z")
    inner = App(Lam("y", Var("y")), z)
    term = App(Lam("x", App(Var("x"), Var("x"))), inner)
    one = beta_steps(term)
    print(f"Term: {term}")
    print("One-step reducts:")
    for r in one:
        print(" ", r)
    print("Pairwise joinable within depth limit:")
    for i, a in enumerate(one):
        for b in one[i + 1:]:
            print(f"  {a}  <->  {b} : {can_join(a, b)}")
```

I therefore summarize the Church-Rosser theorem as the confluence guarantee for beta-reduction in the lambda calculus. It tells us that reduction paths may fork but never diverge irreversibly, that normal forms are unique when they exist, and that the semantic meaning of a lambda term is independent of the particular reduction strategy used to evaluate it. The theorem does not ensure termination, nor does it favor any single evaluation order; instead it imposes a global geometric discipline on the rewrite graph, turning a collection of local substitution steps into a coherent theory of computation. This discipline is what allows the lambda calculus to serve simultaneously as a programming language foundation, a model of computability, and a syntax for proofs.
