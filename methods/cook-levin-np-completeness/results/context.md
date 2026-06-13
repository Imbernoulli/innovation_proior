# Context

## Research question

A large and growing list of problems resists every algorithm anyone can find. Deciding whether a
propositional formula is a tautology (or, dually, whether one is satisfiable); deciding whether one
graph is isomorphic to a subgraph of another; deciding whether a graph has a Hamiltonian circuit;
deciding whether a system of linear inequalities has a 0/1 solution; finding a minimal disjunctive
normal form for a partial Boolean function. For each of these the only methods known are essentially
a search over exponentially many candidates — try every truth assignment, every embedding, every
subset — and the worst-case running time grows exponentially in the length of the input. Years of
separate effort on each problem have produced no polynomial-time algorithm for any of them, and no
proof that none exists.

The precise question is twofold. First: is there a notion of "tractable" sharp enough to separate
the problems we can solve well from the ones we cannot? Second, and deeper: are these scattered hard
problems hard *for the same reason*, or is each one its own isolated obstacle? If their difficulty
could be shown to be a single shared difficulty — if one could exhibit one problem onto which all the
others (and indeed every problem of this guess-and-check shape) provably collapse — then the question
"can any of them be solved efficiently?" would become a single question about that one problem. A
solution would have to do two things at once: pin down the right class of "search" problems, and
locate a single member of it that is at least as hard as every other.

## Background

**Polynomial time as the line between feasible and infeasible.** Cobham (1964), "The intrinsic
computational difficulty of functions," isolated the class of functions computable in time bounded
by a polynomial in the input length and argued it is the robust, machine-independent notion of
"feasibly computable." Independently, Edmonds (1965), in "Paths, Trees, and Flowers" — which gave the
first polynomial-time algorithm for maximum matching in general graphs — drew the same line
explicitly, calling a polynomial-bounded algorithm a "good" algorithm and contrasting it with the
"finite but exponential" brute-force search, conjecturing that for some problems no good algorithm
exists. This is the prevailing wisdom by the late 1960s: polynomial time = tractable; exponential
search = the thing to escape. It gives the yardstick, but not yet any way to tie problems together.

**Turing machines and nondeterminism.** A (deterministic) Turing machine is a finite control reading
and writing a tape; its computation on input w is a sequence of configurations, each obtained from
the previous one by a single move of the transition function δ — and a single move touches only the
scanned cell, the control state, and the head position (which shifts by one). A *nondeterministic*
Turing machine (the TM analogue of Rabin–Scott 1959 nondeterministic finite automata) has a δ that
returns a *set* of allowed moves; it accepts w if *some* sequence of choices leads to an accepting
state. Equivalently, an NTM that runs in polynomial time accepts w exactly when there exists a short
certificate — the lucky sequence of guesses, of length polynomial in |w| — that a deterministic
machine could check, step by step, in polynomial time. Nondeterministic poly-time recognition is thus
the same as "there is a poly-length witness, poly-time checkable." That a nondeterministic TM is no
more powerful than a deterministic one for *what* it can compute (via dovetailing / exhaustive
simulation of all branches) was standard; the open question was entirely about *how fast*.

**The "search" / perebor tradition.** A parallel line of thought, strong in the Soviet school
(Kolmogorov's seminar; Yablonsky 1959 on the difficulty of synthesizing minimal contact circuits;
Trakhtenbrot's work on optimal computation; Yu. Zhuravlev on Boolean minimization), framed exactly
these problems as *perebor* — exhaustive enumeration — problems: "given x, find a y of length
comparable to x's such that a simple checkable condition A(x,y) holds." Minimizing a Boolean function,
finding a proof of bounded length, deciding graph isomorphism — all of this shape. The belief, argued
for but never proved, was that no method essentially faster than brute-force enumeration exists; e.g.
it had not even been shown that finding a mathematical proof is harder than verifying one.

**Theorem-proving procedures and Davis–Putnam.** Mechanical theorem proving was a live field with
dozens of competing procedures and no basis for comparing them. The Davis–Putnam procedure (1960)
decides whether a formula in conjunctive normal form is satisfiable; its dual decides whether a
formula in disjunctive normal form is a tautology. No polynomial bound was known for it, and equally
no family of examples was known forcing it to take more than polynomial time. The propositional
decision problem — tautology, equivalently satisfiability — sat at the center of this field as the
simplest object whose complexity was genuinely open. For the predicate calculus, by contrast, the
Herbrand theorem reduces validity to truth-functional inconsistency of conjunctions of substitution
instances, but the satisfiability problem there is undecidable (no recursive bound on the number of
instances needed), so the *finite*, propositional case is where a complexity theory could get traction.

**The diagnostic facts already on the table.** Each candidate hard problem had been *observed* to
admit only exponential methods; tautology-checking, subgraph isomorphism, graph isomorphism,
satisfiability, Boolean-function minimization were each, separately, resistant. Corneil–Gotlieb (1970)
had a heuristic for graph isomorphism with a conjecture that, if true, would put it in polynomial
time — a reminder that the boundary was genuinely unknown and that these problems were being studied
one at a time. The pattern — many independent walls, all looking like exhaustive search, none broken
— is the phenomenon that demands explanation.

## Baselines

The prior art is not a competing algorithm to beat but the conceptual apparatus a unification would
have to be built from, and the problem-specific methods it would have to subsume.

- **Cobham's class of polynomial-time functions (1964).** Core idea: the functions computable within
  a polynomial time bound form a natural, model-independent class (Cobham's L), the right formalization
  of "feasible." Math: closure properties characterizing the class by bounded recursion on notation.
  Gap it leaves: it defines the *target* (efficient) but says nothing about which concrete problems do
  or do not fall in it, nor any relation *between* problems.

- **Edmonds' "good algorithms" (1965).** Core idea: a polynomial-bounded algorithm is "good"; the
  blossom algorithm is the first good algorithm for maximum matching, computing a maximum matching by
  growing alternating trees and shrinking odd "blossoms" so that augmenting paths can be found in
  polynomial time. Gap: it establishes that *some* hard-looking combinatorial problems are secretly
  good, and conjectures that others are not — but offers no tool to *prove* a problem is hard, and no
  link tying the suspected-hard problems to one another.

- **Nondeterministic recognition (Rabin–Scott 1959, lifted to TMs).** Core idea: allow the machine to
  guess; accept if some branch accepts. Math: δ returns a set of moves; acceptance = ∃ accepting
  branch. Gap: nondeterminism is known not to add computational *power* (deterministic simulation by
  dovetailing recognizes the same languages), but the simulation is exponential, and whether
  nondeterministic *polynomial* time exceeds deterministic polynomial time is wide open. It supplies
  the right class of problems to study but, by itself, no internal structure.

- **The Davis–Putnam satisfiability procedure (1960).** Core idea: decide CNF satisfiability by
  iterated elimination (one-literal rule, pure-literal rule, resolution on a variable); the dual
  decides DNF tautology. Math: repeated variable elimination preserving (un)satisfiability. Gap: no
  polynomial bound proved and none refuted; its true complexity — and thus the complexity of the
  propositional decision problem it solves — is exactly the open question. It hands over the central
  candidate problem (satisfiability / tautology) without resolving where it sits.

- **Per-problem search methods (subgraph isomorphism, Hamiltonian circuit, 0/1 integer programs,
  Boolean minimization, graph isomorphism).** Core idea in each case: backtracking search over the
  exponential candidate space, with cleverness (pruning, heuristics, e.g. Corneil–Gotlieb for graph
  isomorphism) but no polynomial guarantee. Gap: each is attacked in isolation; success on one carries
  no implication for any other; none is known to be in polynomial time and none is known not to be.

## Evaluation settings

The natural yardsticks here are not benchmark datasets but the formal objects a result would be stated
over. The model of computation is the (deterministic or nondeterministic) one-tape Turing machine over
a fixed finite alphabet, with running time measured as a function of input length n; inputs are
strings encoding the relevant combinatorial objects (graphs as adjacency-matrix strings over {0,1,*},
formulas as strings over a propositional alphabet with atoms in binary, integers in binary/m-adic
notation). The complexity measure is worst-case time; the class P is the languages recognized by
deterministic poly-time TMs, the class NP the languages recognized by nondeterministic poly-time TMs.
The relation of interest is polynomial-time reducibility — a poly-time map f with x in A iff f(x) in B
(or its oracle/query-machine variant) — under which membership in P is inherited downward and which is
transitive. The candidate problems against which any structural claim is measured are the standard
ones above (tautology / satisfiability; subgraph and graph isomorphism; primality; set cover;
graph homomorphism; bounded-length matrix/tiling extension). The success criterion for a unification
is structural, not numerical: exhibit a single language to which every language in the class
poly-time-reduces.

## Code framework

The useful programming objects are minimal: a nondeterministic machine description and a Boolean
formula builder. With these in hand one can experiment with whatever map between the two a
structural argument might call for.

```python
class NTM:
    # states, tape alphabet Gamma (incl. blank), input alphabet,
    # transition relation delta: (state, symbol) -> set of (state, symbol, move in {-1,+1}),
    # start state q0, accept state q_accept, polynomial step_bound(n).
    def __init__(self, states, gamma, delta, q0, q_accept, blank, step_bound):
        ...

class CNF:
    """A propositional formula in conjunctive normal form: a list of clauses,
    each clause a list of literals (+var / -var). Supports .is_satisfiable()
    by a standard exhaustive or Davis-Putnam-style checker."""
    def __init__(self):
        self.clauses = []
    def add_clause(self, literals):
        self.clauses.append(literals)
    def is_satisfiable(self):
        ...

def reduce(M: NTM, w: str) -> CNF:
    """Map a machine M and input w to a formula instance."""
    # TODO: build the formula
    pass
```
