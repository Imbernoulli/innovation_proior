I think of computational hardness as a question about transfer. A huge number of finite search tasks share a common profile: a proposed solution can be checked quickly, yet no fast general method for finding one is known. Satisfiability, graph coloring, Hamiltonian paths, scheduling, and countless others all have this shape. The question is whether they share a hidden common core, in the sense that a truly efficient algorithm for one of them would make all of them easy.

I start with the ingredients already on the table. Turing machines give a precise model of local computation through configurations: state, tape, and head position change only in bounded ways at each step. Computability theory, however, is too coarse; it separates decidable from undecidable but does not distinguish a polynomial-time computation from an exponential exhaustive search. Combinatorial optimization had identified many natural problems where brute force seems unavoidable, yet a list of hard-looking examples is not a theory unless the hardness can be carried from one problem to another. Proof theory supplied propositional formulas and decision procedures, but those formulas were studied as objects to prove or refute, not as a compiled trace of bounded computation. What is missing is a way to align feasibility, nondeterministic search, and finite syntax through every translation.

The decisive shift comes from taking polynomial time seriously as the boundary of feasible computation, following Edmonds and Cobham. This changes the notion of a reduction: a translation between problems must itself be computable in polynomial time, otherwise it could hide the entire difficulty. It also identifies the right source class, polynomially bounded nondeterministic computation, or equivalently the set of languages for which a short certificate can be verified deterministically in polynomial time. This class, now called NP, does not promise an efficient way to find the certificate; it only promises that one exists and can be checked quickly. The common shape is a bounded verifier, an input, and a witness whose length is polynomial in the input size.

The Cook-Levin theorem is the statement that this entire class reduces to a single fixed syntactic problem. In its modern form, SAT, the problem of deciding whether a propositional formula has a satisfying assignment, is NP-complete. SAT is in NP because a proposed assignment can be evaluated in linear time. The deep part is the other direction: every language accepted by a nondeterministic Turing machine in polynomial time can be transformed into a SAT instance in polynomial time. Cook's 1971 paper proved this using P-reducibility to tautology and DNF tautology; Levin's 1973 paper independently proved the analogous universal search result. Their common achievement is to show that SAT is a universal receiving language for efficiently checkable search.

To build the reduction, I fix a nondeterministic machine M, an input x, and a polynomial time bound T = p(|x|). Only polynomially many time steps and tape cells can matter, so I introduce Boolean variables for a T-step tableau. Some variables represent the machine state at each time, others represent the head position, and others represent the tape symbol in each cell at each time. I then add clauses that force these variables to describe a real accepting computation. The first row must encode the initial configuration. At every time there must be exactly one state, exactly one head position, and exactly one symbol per cell. Each adjacent pair of rows must respect M's transition relation, cells away from the head must keep their symbol, and an accepting state must occur at some time.

This construction stays polynomial because a Turing-machine step changes only bounded local information. Each transition condition is a constant-size formula, and I need only polynomially many copies of it, one for each relevant time and cell. A satisfying assignment corresponds exactly to an accepting branch of M, and any accepting branch fills the tableau and satisfies the formula. Therefore solving SAT would solve the original nondeterministic polynomial-time problem after only polynomial overhead. The encoding is not a semantic simulation of the machine; it is a static compilation of every possible accepting history into local constraints.

The insight is not simply that machines can be encoded in logic. It is that polynomially bounded nondeterministic computation can be viewed as a locally checkable finite history, and that efficient reducibility can turn this history into a universal hardness statement. Turing supplied configurations and locality, Edmonds and Cobham supplied polynomial time as the feasibility boundary, and proof theory supplied formula syntax; Cook and Levin combined them into a complexity-theoretic comparison principle. Once SAT is known to be NP-complete, hardness spreads quickly through Karp's list of reductions, carrying the difficulty to graph coloring, clique, Hamiltonian paths, scheduling, and many other natural problems.

For the canonical name, I use the Cook-Levin theorem on NP-completeness, also called the Cook-Levin NP-completeness theorem. It is the foundational result that establishes SAT as the first universal problem for efficient search and provides the template for proving hardness across computer science.

```python
def cook_levin_cnf(M, x, T, blank='B'):
    """
    Encode acceptance of a single-tape nondeterministic TM M on input x
    within T steps as a CNF satisfiability problem.
    M = (states, alphabet, transitions, q0, accept).
    transitions[(q, a)] is a list of (q_next, write, move) with move in {-1,0,1}.
    Returns (clauses, n_vars).
    """
    states, alphabet, transitions, q0, accept = M
    cells = list(range(-T, T + 1))

    nxt = 1
    def fresh():
        nonlocal nxt
        v = nxt
        nxt += 1
        return v

    S = {q: [fresh() for _ in range(T + 1)] for q in states}
    H = {i: [fresh() for _ in range(T + 1)] for i in cells}
    Tape = {i: {a: [fresh() for _ in range(T + 1)] for a in alphabet} for i in cells}

    clauses = []

    def exactly_one(lits):
        clauses.append(list(lits))
        for i in range(len(lits)):
            for j in range(i + 1, len(lits)):
                clauses.append([-lits[i], -lits[j]])

    # Initial configuration.
    clauses.append([S[q0][0]])
    for q in states:
        if q != q0:
            clauses.append([-S[q][0]])
    for i in cells:
        clauses.append([H[i][0]] if i == 0 else [-H[i][0]])
        for a in alphabet:
            if 0 <= i < len(x):
                clauses.append([Tape[i][a][0]] if a == x[i] else [-Tape[i][a][0]])
            else:
                clauses.append([Tape[i][a][0]] if a == blank else [-Tape[i][a][0]])

    # Head position is bounded by elapsed time.
    for t in range(T + 1):
        for i in cells:
            if abs(i) > t:
                clauses.append([-H[i][t]])

    # Exactly one state, head position, and tape symbol per cell at each time.
    for t in range(T + 1):
        exactly_one([S[q][t] for q in states])
        exactly_one([H[i][t] for i in cells])
        for i in cells:
            exactly_one([Tape[i][a][t] for a in alphabet])

    # Cells away from the head keep their symbol.
    for t in range(T):
        for i in cells:
            if abs(i) > t:
                continue
            for j in cells:
                if j != i:
                    for sym in alphabet:
                        clauses.append([-H[i][t], -Tape[j][sym][t], Tape[j][sym][t + 1]])

    # Local transition constraints.
    for t in range(T):
        for i in cells:
            if abs(i) > t:
                continue
            for q in states:
                for a in alphabet:
                    cond = [-S[q][t], -H[i][t], -Tape[i][a][t]]
                    branches = transitions.get((q, a), [])
                    if not branches:
                        clauses.append(cond)
                        continue
                    choice_lits = [fresh() for _ in branches]
                    clauses.append(cond + choice_lits)
                    for k in range(len(choice_lits)):
                        for j in range(k + 1, len(choice_lits)):
                            clauses.append(cond + [-choice_lits[k], -choice_lits[j]])
                    for k, (qp, b, d) in enumerate(branches):
                        c = choice_lits[k]
                        clauses.append([-c, S[qp][t + 1]])
                        clauses.append([-c, H[i + d][t + 1]])
                        clauses.append([-c, Tape[i][b][t + 1]])

    # Accept state must occur at some time.
    clauses.append([S[accept][t] for t in range(T + 1)])

    return clauses, nxt - 1


def dpll(clauses, assignment=None):
    """A tiny DPLL SAT solver; returns True if the CNF is satisfiable."""
    if assignment is None:
        assignment = set()

    # Simplify under the current partial assignment.
    simplified = []
    for clause in clauses:
        sat = False
        new_clause = []
        for lit in clause:
            if lit in assignment:
                sat = True
                break
            if -lit not in assignment:
                new_clause.append(lit)
        if sat:
            continue
        if not new_clause:
            return False
        simplified.append(new_clause)
    clauses = simplified

    # Unit propagation.
    while True:
        unit = None
        for clause in clauses:
            if len(clause) == 1:
                unit = clause[0]
                break
        if unit is None:
            break
        if -unit in assignment:
            return False
        if unit in assignment:
            continue
        assignment.add(unit)
        new_clauses = []
        for clause in clauses:
            if unit in clause:
                continue
            new_clause = [l for l in clause if -unit != l]
            if not new_clause:
                return False
            new_clauses.append(new_clause)
        clauses = new_clauses

    if not clauses:
        return True

    # Choose a literal and branch.
    var = abs(clauses[0][0])
    return dpll(clauses, assignment | {var}) or dpll(clauses, assignment | {-var})


# Example: nondeterministic machine that accepts iff the input contains a '1'.
blank = 'B'
states = {'q0', 'qacc', 'qrej'}
alphabet = {'0', '1', blank}
transitions = {
    ('q0', '0'): [('q0', '0', 1)],
    ('q0', '1'): [('qacc', '1', 0), ('q0', '1', 1)],
    ('q0', blank): [('qrej', blank, 0)],
    ('qacc', '0'): [('qacc', '0', 0)],
    ('qacc', '1'): [('qacc', '1', 0)],
    ('qacc', blank): [('qacc', blank, 0)],
    ('qrej', '0'): [('qrej', '0', 0)],
    ('qrej', '1'): [('qrej', '1', 0)],
    ('qrej', blank): [('qrej', blank, 0)],
}
M = (states, alphabet, transitions, 'q0', 'qacc')

for w in ['01', '00']:
    clauses, n = cook_levin_cnf(M, w, T=3)
    print(f'input {w}: SAT={dpll(clauses)}')
```
