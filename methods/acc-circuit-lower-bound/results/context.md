# Context: Proving a Circuit Lower Bound Against ACC

## Research question

Nonuniform circuit families can recognize undecidable languages by using a different circuit for each input length, so the interesting question is whether some explicit, reasonably powerful class is too hard for small, restricted circuits.

The target class is **ACC**: constant-depth, polynomial-size circuits over unbounded-fan-in AND, OR, NOT, and **MOD_m** gates for any constant m > 1, where a MOD_m gate outputs 1 iff m divides the sum of its inputs. (AC0[m] denotes ACC with a single modulus m.)

The concrete goal is to prove that some explicit language in NEXP, E^NP, or EXP^NP has **no** polynomial-size nonuniform ACC circuit family. As a measure of where things stand, it is not even known whether EXP^NP can be computed by depth-three circuits made only of MOD_6 gates.

## Background

**The bottom-up program.** Lower bounds for AC0 and AC0[p] (p prime) use combinatorial restrictions and low-degree polynomial approximation over finite fields. For composite moduli m, there is no field over which to approximate MOD_m.

**Barriers.** A proof separating NEXP from ACC interacts with the standard barriers:
- *Relativization:* there are oracles A with NEXP^A ⊆ ACC^A.
- *Algebrization:* algebraic oracles can also be consistent with NEXP ⊆ ACC.
- *Natural proofs:* a constructive, large combinatorial property useful against a class that may contain pseudorandom functions would break those pseudorandom functions.

**Known ACC structure.** ACC circuits have a depth-two normal form. A **SYM⁺ circuit** is a symmetric gate at the output fed by ANDs of input variables. Yao and Beigel–Tarui showed that every size-s depth-d ACC circuit can be converted into an equivalent deterministic SYM⁺ circuit of quasipolynomial size with ANDs of polylogarithmic fan-in; Green et al. showed the symmetric output can be taken to be the middle bit of the sum of the AND outputs. The transformation is constructive, so a subexponential ACC circuit yields a subexponential SYM⁺ circuit.

**Compression and witnesses.**
- *Efficient Cook–Levin for NEXP:* every L ∈ NTIME[2^n] reduces to SUCCINCT 3SAT, described by a circuit with n + O(log n) inputs; each clause of the exponentially large 3-CNF is locally computable.
- *Easy witnesses:* if NEXP has small circuits, then every satisfiable succinct 3SAT instance has a small circuit whose truth table is a satisfying assignment.

**Karp–Lipton–Meyer connection.** If Circuit-SAT is in P, then EXP is not contained in P/poly: one can guess a circuit encoding an exponential computation history, verify it inside Σ₂, collapse Σ₂ using P = NP, and diagonalize against the time hierarchy. General Circuit-SAT algorithms save only subexponential factors over exhaustive search.

**Circuit satisfiability for restricted classes.** Randomized algorithms faster than 2^n are known for AC0 and for formulas. For ACC circuits, the algorithm known is exhaustive search.

**Tools for bulk circuit evaluation.** The zeta/Möbius transform on subsets of [n] can be computed by Yates's dynamic program in O(2^n poly(n)) time. Coppersmith's rectangular matrix multiplication multiplies an N × N^{.1} matrix by an N^{.1} × N matrix in O(N² polylog N) arithmetic operations.

## Baselines

- **Bottom-up restricted lower bounds (AC0, AC0[p]).** Use combinatorial restrictions and low-degree polynomial approximation over finite fields to prove explicit functions hard for small circuits. These techniques apply to AC0[p] for prime p.

- **Razborov–Rudich naturalization.** Many combinatorial lower-bound arguments automatically yield a natural property: a constructive, large combinatorial property of the hard function's truth table.

- **Karp–Lipton–Meyer (P = NP ⇒ EXP ⊄ P/poly).** Under a strong algorithmic assumption, guess a circuit encoding an exponential computation history and verify it inside Σ₂, then collapse Σ₂ and diagonalize against the time hierarchy. The verification handles unrestricted circuits.

- **Uniform ACC lower bounds.** Combine the SYM⁺ normal form with a counting argument to show the Permanent is not computable by small uniform ACC circuits; the argument uses an efficiently computable connection language.

- **Easy-witness / derandomization line.** If NEXP has small circuits, then satisfiable succinct instances have succinct witnesses and derandomization collapses; a conditional structural implication.

- **Exhaustive search for circuit satisfiability.** Evaluate the circuit on every assignment, the 2^n baseline; for ACC this is the running time known.

## Evaluation settings

This is a theorem-proving task; the "evaluation" is the precise target and the yardsticks an argument must meet.

- *Object to separate:* a single explicit language in NEXP (or E^NP / EXP^NP). The canonical choice is SUCCINCT 3SAT, since a separation for it lifts to all of NEXP by completeness.
- *Strength to beat:* nonuniform ACC of polynomial size; ideally the bound survives quasipolynomial size, an exponential size–depth tradeoff for E^NP, or slightly nonconstant depth.
- *Contradiction engine:* the nondeterministic time hierarchy theorem, which forbids NTIME[2^n] ⊆ NTIME[o(2^n)].
- *Barriers:* a valid proof must not relativize, algebrize, or yield a natural property.
- *Quality of any auxiliary algorithm:* running time on n-input subexponential-size depth-d ACC circuits, measured against the 2^n exhaustive baseline; whether it works for arbitrary nonuniform circuits.

## Code framework

The result is a theorem and proof, so the "code framework" is the inventory of formal primitives already available and the empty slots an argument must fill. Concretely, the proof will instantiate a nondeterministic procedure that decides an arbitrary NTIME[2^n] language faster than 2^n, and it will rely on an auxiliary routine for analyzing circuits from the restricted class.

```python
# ---- Primitives already on the shelf ----

def nondet_time_hierarchy_violated(L):
    """If some L in NTIME[2^n] is decided in NTIME[o(2^n)], that is a contradiction.
    Seiferas-Fischer-Meyer 1978; Zak 1983."""
    ...

def efficient_cook_levin(x):
    """Reduce x (for any L in NTIME[2^n]) to a SUCCINCT 3SAT circuit C_x with
    n + O(log n) inputs and poly(n) size, each clause locally computable.
    Cook 1988; Robson 1991; Fortnow-Lipton-van Melkebeek-Viglas 2005."""
    ...

def easy_witness_exists(C_x):
    """If the large class has small circuits, a satisfiable succinct instance has a small
    witness circuit W encoding a satisfying assignment.
    Impagliazzo-Kabanets-Wigderson 2002."""
    ...

def acc_to_sym_plus(circuit):
    """Rewrite a depth-d size-s ACC circuit as an equivalent symmetric-of-ANDs (SYM+)
    circuit of quasipolynomial size, ANDs of polylog fan-in.
    Yao 1990; Beigel-Tarui 1994; Allender-Gore 1991; Green et al. 1995."""
    ...

def zeta_transform(f_table):
    """Yates 1937 dynamic program; O(2^n poly(n))."""
    ...

def coppersmith_rect_matmul(A, B):
    """N x N^.1 by N^.1 x N in O(N^2 polylog N) time. Coppersmith 1982."""
    ...

# ---- The slots an argument must fill ----

def analyze_restricted_circuit(C):
    """An auxiliary routine over circuits from the restricted class, to be designed.
    # TODO: the procedure we will design here.
    """
    raise NotImplementedError

def lower_bound_argument():
    """Derive a contradiction with nondet_time_hierarchy_violated, under the assumption
    that the large class has small restricted circuits.
    # TODO: the chain of reductions we will construct here.
    """
    raise NotImplementedError
```

The content of the discovery is what goes inside those two `# TODO` slots.
