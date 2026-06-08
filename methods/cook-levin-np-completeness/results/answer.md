# The Cook–Levin Theorem: SAT is NP-complete

## The problem it solves

Many natural decision problems — Boolean satisfiability, subgraph isomorphism, Hamiltonian circuit,
0/1 integer programming, and so on — admit only exponential brute-force search and had each been
studied in isolation. They share a form: "given x, is there a short witness y passing a fast check
A(x,y)?" This is exactly the class **NP** = languages accepted by a nondeterministic Turing machine in
polynomial time. The question is whether these problems are hard for a *single shared reason*, made
precise as: is there one problem in NP to which every problem in NP polynomial-time reduces?

## The key idea

Yes — and **Boolean satisfiability (SAT)** is such a problem. The entire computation of a
nondeterministic poly-time machine M on an input w can be transcribed as a propositional formula
φ_{M,w} that is satisfiable iff M has an accepting computation on w. Lay the computation out as a
time × tape **tableau**; because one transition-function step changes only the scanned cell, the state,
and the head position, the global condition "this is a valid accepting computation" becomes a
**polynomial-size conjunction of constant-size local checks**. A satisfying assignment *is* an
accepting computation. Hence every NP language reduces to SAT in polynomial time, and since SAT itself
is in NP, SAT is **NP-complete**: SAT ∈ P ⇔ P = NP. (Discovered independently by Cook in 1971 via the
theorem-proving/tautology route, and by Levin in 1973 via the *perebor* / universal-search route.)

## The theorem and its proof

**Theorem.** SAT is NP-complete.

**Definition.** A language B is NP-complete if (i) B ∈ NP and (ii) for every A ∈ NP, A ≤_P B (A
polynomial-time many-one reduces to B).

### Part 1: SAT ∈ NP

Given a formula φ, a satisfying truth assignment is a certificate of length ≤ |φ|. A deterministic
Turing machine evaluates the formula tree (or every clause, for CNF input) under the assignment in time
linear in |φ|. So a nondeterministic machine guesses the assignment and verifies it in polynomial time:
SAT ∈ NP. ∎

### Part 2: every A ∈ NP reduces to SAT in polynomial time

Let A ∈ NP be decided by a nondeterministic Turing machine M (tape alphabet Γ, states Q, start state
q₀, accept state q_accept, blank ␣) within polynomial time p(n). Pad the bound to T = O(n^k) with
T ≥ p(n)+n+2. From w with |w| = n construct a formula φ_{M,w}.

**Tableau.** A table with T time rows and columns 0,…,T+1. Columns 0 and T+1 hold a fixed boundary
symbol #. Row i is a configuration string: tape symbols, plus one state marker immediately before the
symbol scanned by the head. The first row is #, q₀, w₁,…,w_n, blanks, #; an accepting computation (one
branch of M's nondeterminism) fills the table.

**Variables.** x_{i,j,σ} for 1 ≤ i ≤ T, 0 ≤ j ≤ T+1, and σ ∈ Γ ∪ Q ∪ {#}, with the meaning "position
(i,j) contains σ." A truth assignment is a way of filling every table position. Count: O(n^{2k})
variables.

**Formula.** φ_{M,w} = φ_cell ∧ φ_start ∧ φ_move ∧ φ_accept.

- φ_cell — exactly one symbol per table position:
  ⋀_{1≤i≤T,0≤j≤T+1} [ (⋁_{σ∈Γ∪Q∪{#}} x_{i,j,σ}) ∧ ⋀_{σ≠τ} (¬x_{i,j,σ} ∨ ¬x_{i,j,τ}) ].

- φ_start — first row is the start configuration on w, and boundaries are fixed:
  (⋀_{i=1}^{T} (x_{i,0,#} ∧ x_{i,T+1,#})) ∧ x_{1,1,q₀} ∧ x_{1,2,w₁} ∧ … ∧
  x_{1,n+1,w_n} ∧ x_{1,n+2,␣} ∧ … ∧ x_{1,T,␣}.

- φ_move — every 2×3 window is legal (consistent with δ). For a window with top positions (a,b,c) and
  bottom positions (d,e,f), "legal" means it can occur in two consecutive rows of a correct M-computation
  (read off δ once, as a constant-size list, with # immovable):
  ⋀_{1≤i<T,1≤j≤T} ⋁_{(a,b,c,d,e,f) legal}
     ( x_{i,j-1,a} ∧ x_{i,j,b} ∧ x_{i,j+1,c} ∧ x_{i+1,j-1,d} ∧ x_{i+1,j,e} ∧ x_{i+1,j+1,f} ).

- φ_accept — the accept state appears somewhere:
  ⋁_{1≤i≤T,0≤j≤T+1} x_{i,j,q_accept}.

**Locality lemma.** The first row has exactly one state marker. If row i has one state marker, then
every top neighborhood away from it forces copying, and the overlapping windows around it force one
consistent δ-permitted rewrite/state-shift choice. Boundary windows keep # fixed. By induction, each
row is a legal δ-successor of the previous one, so the local checks enforce the global "valid
computation" condition.

**Correctness.** φ_{M,w} is satisfiable ⇔ the table is a well-formed (φ_cell), correctly-started
(φ_start), legal-stepping (φ_move + lemma), accepting (φ_accept) computation of M on w ⇔ M accepts w
⇔ w ∈ A.

**Polynomial time.** O(T^2) = O(n^{2k}) variables; φ_cell and φ_move each contribute O(T^2)
constant-size pieces; φ_start, O(T) unit clauses; φ_accept, one O(T^2) disjunction. |φ_{M,w}| =
O(n^{2k}) (up to log factors), mechanically written from (M, w) in time polynomial in n. So A ≤_P
SAT. ∎

Since A ∈ NP was arbitrary, every NP language reduces to SAT; with Part 1, SAT is NP-complete.

**Dual / theorem-proving form.** Equivalently, over symbols P^i_{s,t} ("square s holds σ_i at step t"),
Q^i_t ("state q_i at step t"), S_{s,t} ("square s scanned at time t"), build a CNF formula
A(w) = B ∧ C ∧ D ∧ E ∧ F ∧ G ∧ H ∧ I (B/C/D: uniqueness of scanned square / symbol / state; E: start;
F/G/H: correct update; I: acceptance), where the transition group is
G^t_{i,j} = ⋀_{s=1}^{T} (¬Q^i_t ∨ ¬S_{s,t} ∨ ¬P^j_{s,t} ∨ Q^k_{t+1}). The displayed clause is the
single-instruction form: if δ is set-valued, use a constant-size disjunction over permitted q_k's or
split choices into separate states, with F and H carrying the same choice. A(w) is satisfiable iff M
accepts w; ¬A(w) in DNF is a tautology iff w ∉ A, giving the complementary theorem-proving decision
form.

### Robustness: 3SAT is NP-complete

SAT ≤_P 3SAT, preserving satisfiability (not logical equivalence). Introduce a fresh variable z for
each connective of φ, with width-≤3 clauses tying z to its inputs —
z ⟺ (u ∧ v): (¬z∨u)(¬z∨v)(z∨¬u∨¬v);  z ⟺ (u ∨ v): (z∨¬u)(z∨¬v)(¬z∨u∨v) —
force the root variable true, and split any wide clause (ℓ₁∨…∨ℓ_m) into a chain
(ℓ₁∨ℓ₂∨z₁)(¬z₁∨ℓ₃∨z₂)…(¬z_{m-3}∨ℓ_{m-1}∨ℓ_m). Linear blowup; satisfiability preserved. Hence 3SAT is
NP-complete too.

### Corollary

SAT ∈ P ⇔ P = NP. A polynomial algorithm for SAT would give one for every problem in NP; conversely if
any NP problem lacks a polynomial algorithm, so does SAT. The difficulty of all of NP is concentrated
in this one problem.

## Optional: the reduction made concrete

A small program witnesses that the reduction is mechanical — given a nondeterministic machine and an
input, it emits the CNF tableau. (SAT itself is not solved fast; it is the object being studied.)

```python
def encode_computation(M, w):
    """Emit CNF phi over vars x[(i,j,sigma)] that is satisfiable iff M has an
    accepting computation on w within a padded T = O(len(w)**k) rows."""
    n = len(w)
    T = M.step_bound(n) + n + 2                   # time rows and interior columns
    cols = T + 2                                  # boundary columns 0 and T+1
    boundary = getattr(M, "boundary", "#")
    symbols = set(M.gamma) | set(M.states) | {boundary}
    cnf = CNF()
    var = {}                                      # (i,j,sigma) -> fresh int id

    next_id = 1
    def new_var():
        nonlocal next_id
        out = next_id
        next_id += 1
        return out

    def V(i, j, s):
        if (i, j, s) not in var:
            var[(i, j, s)] = new_var()
        return var[(i, j, s)]

    # phi_cell: exactly one symbol per table position
    S = list(symbols)
    for i in range(T):
        for j in range(cols):
            cnf.add_clause([V(i, j, s) for s in S])                    # >= 1
            for a in range(len(S)):
                for b in range(a + 1, len(S)):
                    cnf.add_clause([-V(i, j, S[a]), -V(i, j, S[b])])   # <= 1

    # phi_start: fixed boundaries and first row = #, q0, w_1..w_n, blanks, #
    for i in range(T):
        cnf.add_clause([V(i, 0, boundary)])
        cnf.add_clause([V(i, cols - 1, boundary)])
    cnf.add_clause([V(0, 1, M.q0)])
    for j, ch in enumerate(w):
        cnf.add_clause([V(0, j + 2, ch)])
    for j in range(n + 2, cols - 1):
        cnf.add_clause([V(0, j, M.blank)])

    # phi_move: every 2x3 window centered on a nonboundary column is legal
    legal = M.legal_windows(boundary=boundary)    # constant-size list from delta
    for i in range(0, T - 1):
        for j in range(1, cols - 1):
            # OR over legal windows; each disjunct is an AND of 6 vars.
            # Encode via fresh selector z_w per legal window:
            #   z_w -> (each of the 6 cell-literals)   [width-2 clauses]
            #   (OR of z_w)                            [at least one legal window]
            zs = []
            for (a, b, c, d, e, f) in legal:
                z = new_var()
                zs.append(z)
                cells = [(i, j-1, a), (i, j, b), (i, j+1, c),
                         (i+1, j-1, d), (i+1, j, e), (i+1, j+1, f)]
                for (ci, cj, cs) in cells:
                    cnf.add_clause([-z, V(ci, cj, cs)])
            cnf.add_clause(zs)

    # phi_accept: accept state appears somewhere
    cnf.add_clause([V(i, j, M.q_accept) for i in range(T) for j in range(cols)])
    return cnf
```
