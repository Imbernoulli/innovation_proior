# Context

## Research question

Two questions, which around 1991 looked unrelated, and a suspicion that they were the same question.

**Q1 — Can we prove that optimization problems are hard to *approximate*?** The theory of
NP-completeness (Cook 1971, Karp 1972, Levin 1973) tells us that exactly solving SAT, clique,
MAX-3SAT, vertex cover, TSP, etc. is intractable unless P = NP. But for a practitioner the relevant
question is *approximation*: can we get within a constant factor of the optimum in polynomial time?
Here the picture was a mess with no organizing principle. Some problems have a fully polynomial-time
approximation scheme (knapsack); some have a PTAS (makespan minimization); some have only a
constant-factor algorithm (metric TSP, vertex cover, MAX-CUT); and unrestricted TSP has no
constant-factor algorithm unless P = NP (Sahni–Gonzalez 1976). For clique and MAX-3SAT *nothing*
was known: clique might or might not be approximable within a constant factor, and there was no
technique to settle it. The goal: a *general method to prove inapproximability* — to show "no
polynomial-time algorithm approximates problem Π within factor c unless P = NP." The obstacle is that
a standard Karp reduction preserves the *exact* optimum but is brittle near the optimum: a solution
that is merely close-to-optimal in the target can pull back to a close-to-optimal solution of the
source, so any "gap" between yes-instances and no-instances dissolves. What is needed is a
**gap-producing reduction** — one that maps yes-instances to instances of large optimum and
no-instances to instances of provably small optimum, with the gap surviving — and there was no
general source of such gaps.

**Q2 — How weak can a proof-verifier be and still check all of NP?** A second, internal-to-complexity
question. NP, by definition, is the class whose membership proofs a deterministic poly-time verifier
reads *in full*. A decade of work on randomized, interactive proof verification kept revealing that
verifiers far weaker in their *access to the proof* can still check surprisingly large classes. The
sharp form of the question: is there a *fixed constant* number of proof bits — read at random
locations using only O(log n) random bits — that suffices to check every NP statement, accepting true
ones always and rejecting false ones with probability ≥ ½?

Why it matters: if Q2 has a "yes" with a small enough query count, and if (as suspected) Q2 and Q1 are
linked, then a single result would simultaneously give a brand-new, robust notion of "proof" and an
all-purpose engine for proving inapproximability.

## Background

**The static notion of a proof, and its fragility.** Classically a proof is a sequential certificate:
to be convinced you read every symbol, and a single corrupted symbol can make a false statement look
proved. This is the NP verifier of Cook–Karp–Levin: for L ∈ NP there is a poly-time M with
x ∈ L ⇒ ∃π, M(x,π) accepts, and x ∉ L ⇒ ∀π, M(x,π) rejects, with |π| = poly(|x|), and M reads all of π.

**Interactive and probabilistic proofs (the ladder that kept climbing).** Goldwasser–Micali–Rackoff
1985 and Babai 1985 let the verifier be a randomized poly-time machine talking to an all-powerful but
untrusted prover (IP, AM). Lund–Fortnow–Karloff–Nisan 1990 introduced **arithmetization** and the
**sum-check protocol** and showed the permanent / #P and the polynomial hierarchy have interactive
proofs; Shamir 1992 pushed this to **IP = PSPACE**. Ben-Or–Goldwasser–Kilian–Wigderson 1988 defined
**multi-prover interactive proofs (MIP)**: a verifier talking to two or more non-communicating
provers, gaining a consistency-checking power (re-ask a second prover a random subset of the queries).
Fortnow–Rompel–Sipser 1988 gave the oracle reformulation of MIP: a poly-time verifier with random
access to a fixed (possibly exponentially long) proof string — i.e. a *probabilistically checkable
proof*. Babai–Fortnow–Lund 1991 proved the startling **MIP = NEXP**: every nondeterministic
exponential-time language has such proofs, equivalently NEXP = PCP(poly, poly). The proof of MIP=NEXP
runs: a NEXP version of the Cook–Levin theorem turns "x ∈ L" into satisfiability of an
exponentially-large 3-CNF Φ_x whose clause structure is given by a poly-time circuit; arithmetize Φ_x
into a low-degree polynomial P_x over a finite field; the verifier runs a sum-check over the Boolean
cube plus a multilinearity (low-degree) test on the prover's claimed assignment, reading only a few
field elements. The crucial *engine* facts:
- **Schwartz–Zippel / DeMillo–Lipton 1978–80**: two distinct degree-d polynomials on F^m agree on at
  most a d/|F| fraction of points; so a single random evaluation detects a wrong low-degree polynomial
  with probability 1 − d/|F|, and a degree-d polynomial is the *unique* one closest to a function that
  is, say, < 1/4-far from low-degree (uniqueness of the low-degree extension).
- **Program checking / self-testing** (Blum–Luby–Rubinfeld 1990, Rubinfeld–Sudan): a function can be
  tested for being (close to) low-degree by restricting it to a random line and checking the
  restriction is a low-degree univariate — a *local* test that makes the encoded object self-correcting.

**Scaling NEXP=PCP(poly,poly) down toward NP.** NEXP is the exponential analogue of NP, so researchers
asked whether MIP=NEXP could be "scaled down" by a logarithm to say something about NP.
Babai–Fortnow–Levin–Szegedy 1991 introduced **transparent (holographic) proofs** checkable in
polylogarithmic time once the input is encoded with an error-correcting code, giving
NP ⊆ PCP(polylog n, polylog n). Feige–Goldwasser–Lovász–Safra–Szegedy 1991 (FGLSS) gave a similar but
more query-efficient verifier, NP ⊆ PCP(log n·log log n, log n·log log n), implicitly defining the
parameters "number of random bits" and "number of proof bits read" as the quantities to minimize.

**The gap connection (FGLSS), the diagnostic that fused Q1 and Q2.** FGLSS made the decisive
observation linking proof checking to approximation. From an (r, q)-restricted verifier build a graph:
vertices are pairs (random string ρ, a local "view" = an accepting setting of the q queried bits for
ρ); connect two vertices if their views are mutually consistent (no proof location is assigned two
different values). If a random string has at most a accepting local views, the graph has at most
2^r·a vertices; for q queried bits, a ≤ 2^q. A clique in this graph is exactly a single proof string
consistent across many random strings. If x is a yes-instance, the honest proof gives an accepting view
for *every* ρ, so the graph has a clique of size 2^{r}; if x is a no-instance with soundness error s,
every proof is accepted on at most an s-fraction of ρ, so the largest clique is at most s·2^{r}. The
clique-size **gap** is the verifier's **soundness gap**. Hence: a low-query PCP for NP *manufactures* a
hardness-of-approximation gap for clique, and the smaller the query count, the sharper the
inapproximability factor. With the then-known NP ⊆ PCP(log n·log log n, log n·log log n) this already
showed clique cannot be approximated within any constant unless NP ⊆ DTIME(n^{O(log log n)}). The
barrier to a clean "P = NP" conclusion, and to inapproximability of MAX-3SAT (not just clique), was that
the query count was *not constant*.

**The state of play, stated as a target.** PCP(0, poly n) = NP (the static verifier);
PCP(poly n, poly n) = MIP = NEXP. NP sits at randomness 0. The open question: does NP have an *exact*
PCP characterization at randomness O(log n) — and how few proof bits can the verifier read? If NP were
contained in PCP(o(log n), o(log n)), the FGLSS reduction would shrink clique instances to sublinear
size and iteration would put clique, hence NP, in P. So O(log n) randomness is the natural floor. The
query count was the prize.

## Baselines

- **The static NP verifier (Cook 1971 / Karp 1972 / Levin 1973).** x ∈ L ⇒ ∃π accepted; x ∉ L ⇒ all π
  rejected; verifier reads *all* of π. Limitation: reads the whole proof, deterministic, gives zero
  soundness gap, hence no inapproximability leverage.
- **IP = PSPACE (LFKN 1990 sum-check; Shamir 1992).** Randomized verifier + one all-powerful prover,
  many rounds; arithmetize a quantified Boolean formula and run sum-check. Limitation: a single prover
  can adapt across rounds, the verifier reads prover messages "in plaintext," and the class is PSPACE —
  far larger than NP, and the protocol is interactive/multi-round, not a fixed proof read at O(1) spots.
- **MIP = NEXP (Babai–Fortnow–Lund 1991), i.e. NEXP = PCP(poly, poly).** Two non-communicating provers
  give the consistency leverage; NEXP-Cook-Levin + arithmetization + sum-check + multilinearity test.
  Limitation: poly(n) randomness and poly(n) queries; it characterizes NEXP, not NP. The whole research
  program is to scale the *engine* down by a logarithm while crushing the query count to O(1).
- **Transparent / holographic proofs (Babai–Fortnow–Levin–Szegedy 1991): NP ⊆ PCP(polylog, polylog).**
  First scale-down to NP, checkable in polylog time on an encoded input. Limitation: both randomness
  *and* query count are polylog — randomness above O(log n) and queries far above O(1), so the FGLSS gap
  is sub-constant.
- **FGLSS 1991: NP ⊆ PCP(log n·log log n, log n·log log n), and the PCP→clique gap reduction.** The
  query-efficient verifier plus the graph construction turning soundness into a clique gap. Limitations:
  (i) the randomness is log n·log log n, not O(log n), so the conclusion is NP ⊆ DTIME(n^{O(log log n)}),
  not NP = P; (ii) the query count is non-constant, so the gap (hence inapproximability factor) is not a
  constant — it gives only N^{1/poly log} for clique and *nothing* for MAX-3SAT, vertex cover, etc.
- **Approximation-preserving reductions and MAX SNP (Papadimitriou–Yannakakis 1991).** A syntactic
  class (defined via second-order logic à la Fagin 1974) of optimization problems closed under an
  approximation-preserving reduction, with complete problems MAX-3SAT, MAX-CUT, vertex cover, metric
  TSP, etc.: either all MAX SNP-hard problems have a PTAS, or none do. Limitation: it *organizes* the
  question — reduce one MAX SNP problem out of PTAS and they all go — but supplies no way to actually
  prove any of them lacks a PTAS. It is the empty frame waiting for a gap.

## Evaluation settings

This is a complexity-theoretic result; the "evaluation" is the set of complexity classes, complete
problems, and the parameters by which a verifier is measured.
- **Complexity classes / yardsticks:** NP, P, NEXP, PSPACE; the class PCP(r(n), q(n)); the
  randomness r(n) and query count q(n) of a verifier; the soundness error (probability of accepting a
  false statement) and completeness (probability of accepting a true one, taken to be 1 here).
- **Canonical NP-complete targets:** SAT / 3-SAT, CKT-SAT (algebraic-circuit satisfiability over GF(2),
  NP-complete via 3-CNF→circuit), MAX-3SAT (maximize the number of simultaneously satisfied clauses of
  a 3-CNF), clique / independent set, and the MAX SNP-complete problems (MAX-CUT, vertex cover, metric
  TSP, Steiner tree, shortest superstring).
- **Algebraic toolkit measured against:** finite fields F = GF(q); the family F_m^{(d)} of m-variate
  degree-d polynomials over F; the relative distance Δ(f,g) = |{x : f(x)≠g(x)}|/|F|^m; "δ-close to
  low-degree"; the low-degree test (restriction to a line/curve); the sum-check protocol.
- **The reduction quality measured:** the clique-gap ratio between yes/no instances (FGLSS), the
  MAX-3SAT satisfiable-fraction gap (1 vs. 1−ε), and the size blow-up of the reduction (must be
  polynomial, e.g. 2^r times the number of accepting local views in FGLSS, which keeps r at O(log n)
  when the local view bound is constant).

## Code framework

This is a pure-complexity result; the "code" is the abstract machinery. The available pieces are the
static NP verifier, the algebraic primitives (finite fields, low-degree polynomials, the
Schwartz–Zippel distance bound, the line/curve low-degree test, sum-check), and the FGLSS-style
reduction template. The open slots are the verifier whose randomness is O(log n) and whose query count
is O(1), and the associated gap reduction.

```python
# ---- Available primitives ----------------------------------------------------

def static_NP_verifier(x, pi):
    # Cook-Levin: reads ALL of pi, deterministic. No soundness gap.
    return decide_in_poly_time(x, pi)            # accept / reject

class FiniteField:                                # F = GF(q), |F| = q
    ...

def schwartz_zippel_distance(d, F):
    # two distinct degree-d polys on F^m agree on <= d/|F| fraction of points
    return d / len(F)

def low_degree_test(oracle_f, m, d, F):
    # restrict f to a random line/curve; check the restriction is a low-degree
    # univariate by reading O(d) values. Makes f self-correcting.
    pass

def sum_check(P, cube, claimed_value):
    # verify  sum_{cube} P == claimed_value  by stripping one variable per round
    pass

def arithmetize_3CNF(phi):
    # AND/OR/NOT -> field (x*y, 1-x, sums over {0,1}); 3-CNF -> low-degree poly
    pass

# ---- Open slots --------------------------------------------------------------

class Verifier:
    """An (r(n), q(n))-restricted verifier: uses r(n) random bits, reads q(n)
       proof symbols, accepts true x with prob 1, rejects false x with prob<=1/2."""
    def run(self, x, proof_oracle, randomness):
        # TODO: build the (log n, O(1)) verifier for an NP-complete language.
        #   - encode the witness as a LOW-DEGREE object (arithmetization)  [TODO]
        #   - check it locally (low-degree test + an assignment/consistency
        #     test + query aggregation) reading only O(1) symbols           [TODO]
        #   - get the answer size down to O(1) by PROOF COMPOSITION         [TODO]
        pass

def pcp_to_gap_reduction(x, V):
    """Turn the verifier V into a hardness-of-approximation gap instance."""
    # TODO: for each random string r, the q queried bits + accept-predicate
    #       become one constraint / a few 3-CNF clauses; aggregate over all r.
    #   yes-instance  -> gap-free (all constraints satisfiable)
    #   no-instance   -> >= constant fraction of constraints unsatisfiable
    pass

def gap_to_pcp_verifier(reduction):
    """The converse: a gap instance gives a verifier that samples one constraint."""
    # TODO: compute the constraint system; expect proof = assignment; pick a
    #       random constraint with O(log n) bits; query its O(1) variables.
    pass
```
