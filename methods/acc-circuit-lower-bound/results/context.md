# Context: Proving a Circuit Lower Bound Against ACC

## Research question

A nonuniform circuit family is allowed a different circuit for each input length, so it can
recognize even undecidable languages by sheer size. The interesting regime is when the circuits
are *restricted* — bounded depth, polynomial size, a fixed gate basis. The driving question of
this area is whether some explicit, "feasible-ish" complexity class provably *cannot* be
simulated by such restricted circuits.

The specific frontier is the class **ACC**: constant-depth, polynomial-size circuits over the
basis AND, OR, NOT (unbounded fan-in) together with **MOD_m gates** for an arbitrary constant
m > 1, where a MOD_m gate outputs 1 iff m divides the sum of its inputs. Writing AC0[m] for the
class with a single modulus m, ACC is the union over all m of AC0[m].

The concrete goal: exhibit a language in some large but explicitly-defined class — the natural
candidates being nondeterministic exponential time NEXP = ∪_k NTIME[2^{n^k}], or its relatives
like E^NP (2^{O(n)} time with an NP oracle), or EXP^NP — and prove that it has **no** nonuniform
ACC circuit family of polynomial size. This matters because it would be the first unconditional
lower bound breaking through a wall that has stood for over two decades, and because the gap
between what we can rule out and what we believe is true here is, frankly, embarrassing: at
present it is not even known whether EXP^NP can be computed by *depth-three* circuits made of
**only MOD_6 gates**. A solution would have to produce a single, explicit hard language and an
argument that survives the formal barriers described below.

## Background

**The bottom-up program and where it stalled.** In the early 1980s the hope was: prove lower
bounds for very restricted circuits, then gradually lift the restrictions until you reach
unrestricted P/poly and thereby P ≠ NP. The program had real early successes.

- *AC0 (no MOD gates).* Furst–Saxe–Sipser [1984] and independently Ajtai [1983] proved that
  the parity of n bits has no polynomial-size constant-depth AND/OR/NOT circuit. Yao [1985]
  pushed this to exponential size and Håstad [1986] proved essentially optimal AC0 lower bounds
  for parity via the switching lemma.
- *AC0[p], p prime.* Granting AC0 a parity (MOD_2) gate for free was the natural next step.
  Razborov [1987] proved an exponential lower bound for computing MAJORITY with AND/OR/NOT/MOD_2
  circuits; Smolensky [1987] proved exponential lower bounds for computing MOD_q with AC0[p]
  circuits, for distinct primes p, q, via approximation of small circuits by low-degree
  polynomials over F_p.
- *ACC = AC0[m], m composite.* Barrington [1989] flagged this as the next step. **Here progress
  stopped.** The polynomial-approximation method that conquered AC0[p] does not extend to
  composite m (there is no field to work over). After 1987 the bottom-up program produced no
  strong ACC lower bound at all, despite the conjecture that even MAJORITY needs superpolynomial
  ACC. The frustration is sharpened by the fact that AC0[p] is provably very weak for every
  prime p, yet AC0[6] (granting both MOD_2 and MOD_3, since MOD_6 = MOD_2 ∧ MOD_3 and each of
  MOD_2, MOD_3 is a MOD_6 of duplicated inputs) resisted every attempt.

**The weakened question.** Unable to find a simple function hard for weak circuits, the field
flipped the quantifiers: take a *complicated* function (something in NEXP) and try to rule out
*weak* circuits for it. A line beginning with Nisan–Wigderson [1994] (with Babai–Fortnow–Nisan–
Wigderson [1993], Impagliazzo–Kabanets–Wigderson [2002], Klivans–van Melkebeek [2002]) showed
that even NEXP ⊄ P/poly would have derandomization consequences — so such a lower bound would
already require substantially new ideas. Even NEXP ⊄ ACC was wide open.

**The formal barriers — any proof must thread these needles.**

- *Relativization* [Baker–Gill–Solovay 1975]: there are oracles A with respect to which
  NEXP^A ⊆ ACC^A (and even with a stronger containment). Any argument that goes through unchanged
  when an oracle is added to every machine and circuit therefore *cannot* separate NEXP from ACC.
  Pure diagonalization relativizes: it proves the uniform separation NEXP ≠ NP, but it is
  powerless against a nonuniform class.
- *Algebrization* [Aaronson–Wigderson 2009]: the natural algebraic extension of relativization
  is likewise insufficient — there are algebraic oracles consistent with NEXP ⊆ ACC.
- *Natural proofs* [Razborov–Rudich 1997]: a "natural" combinatorial property — one that is
  constructive and large, satisfied by a random function — that is useful against a circuit class
  C powerful enough to compute pseudorandom functions would, by definition, break those PRFs.
  This blocks most bottom-up combinatorial lower-bound techniques against strong classes.

**Structural facts about ACC that are known and available.** A line of work descending from
Toda's theorem has shown that low-depth ACC circuits have a surprising depth-two normal form.
Define a **SYM⁺ circuit** to be a depth-two circuit that computes some symmetric function at the
output gate, applied to ANDs of input variables on the bottom layer. Yao [1990] showed every
size-s ACC circuit can be simulated by a *probabilistic* SYM⁺ circuit of size s^{O(log^c s)}
(c depending on the depth), with ANDs of poly(log s) fan-in; Beigel–Tarui [1994] removed the
probabilistic condition (deterministic SYM⁺ of quasipolynomial size); Allender–Gore [1991]
established a uniform version and used it for a uniform lower bound on the Permanent; Green et
al. [1995] showed the symmetric function can be taken to be the specific "middle bit of the sum
of the inputs" function. The transformation is constructive: composite moduli are split via the
Chinese Remainder Theorem into prime-power moduli, prime-power MOD gates are handled with the
**modulus-amplifying polynomials** of Beigel–Tarui, AND/OR gates are randomized à la
Valiant–Vazirani and then derandomized, and the layers of MOD gates are successively absorbed
into the top symmetric gate. The blowup is only quasipolynomial, so for circuits of subexponential
size the resulting SYM⁺ circuit is still subexponential.

**Compression and witnesses.** Two further known facts about how exponentially-long objects can
be represented succinctly:

- *Efficient Cook–Levin for NEXP.* Papadimitriou–Yannakakis [1986] showed that for every natural
  NP-complete problem P, the SUCCINCT version (given a circuit C, decide whether its 2^n-bit
  truth table is a yes-instance of P) is NEXP-complete. SUCCINCT 3SAT is the canonical example.
  Moreover, the standard Cook–Levin tableau is extremely regular, and the sharpened reductions of
  Cook [1988], Robson [1991], and Fortnow–Lipton–van Melkebeek–Viglas [2005] turn any
  L ∈ NTIME[2^n] into a SUCCINCT 3SAT instance whose describing circuit has only n + O(log n)
  inputs — each clause of the decompressed 3-CNF being computable from its index in
  polylogarithmic time.
- *Easy witnesses.* Impagliazzo–Kabanets–Wigderson [2002] proved that if NEXP ⊆ P/poly, then not
  only does SUCCINCT 3SAT have small circuits, but every satisfiable succinct instance has a
  *succinct satisfying assignment*: a small circuit W whose truth table is a satisfying assignment
  to the decompressed formula. The proof mixes hardness-versus-randomness with diagonalization.

**The Karp–Lipton–Meyer seed.** Karp–Lipton [1980], crediting Meyer, observed a connection in the
*other* direction between fast algorithms and circuit lower bounds: if Circuit-SAT ∈ P (i.e.
P = NP), then EXP ⊄ P/poly — because under P = NP one can collapse the Σ₂ verification of a
guessed circuit encoding an exponential computation history into P and contradict the time
hierarchy. The same argument tolerates a "half-exponential"-time Circuit-SAT algorithm
(f with f(f(n^k)^k) ≤ 2^{n/2}). But half-exponential-time Circuit-SAT is far beyond reach: the
best general CNF-SAT and Circuit-SAT algorithms (Monien–Speckenmeyer [1985] and successors;
Schuler [2005]; Calabro–Impagliazzo–Paturi [2006]) save only a subexponential factor over 2^n,
and the Exponential Time Hypothesis [Impagliazzo–Paturi 2001] posits even 3SAT needs 2^{Ω(n)}.

**The complexity of circuit satisfiability for restricted classes.** For each circuit class C one
can define C-SAT: given a circuit from C, is it satisfiable? Very little is known about its
worst-case time complexity. For depth-d AC0 circuits, randomized algorithms running in
2^{n − Ω(n/(log s)^{d−1})} time are known (Calabro–Impagliazzo–Paturi; Impagliazzo–Matthews–Paturi);
for formulas over AND/OR/NOT, Santhanam [2010] gives 2^{n − n/c^k}-time algorithms. For ACC
circuits, no algorithm beating the 2^n exhaustive search was known.

**Tools for evaluating circuits in bulk.** Two classical primitives are on the shelf. The
zeta/Möbius transform of a function on subsets of [n] can be computed by Yates's 1937 dynamic
program in O(2^n · poly(n)) time. And Coppersmith's [1982] rectangular matrix multiplication
multiplies an N × N^{.1} matrix by an N^{.1} × N matrix in O(N² log² N) arithmetic operations —
near-optimal because the inner dimension is polynomially smaller than the outer ones; the
algorithm is explicit and runs on a multitape Turing machine in O(N² · poly(log N)) time over
small fields.

## Baselines

These are the prior methods and results a new ACC lower bound would be measured against or would
have to get past.

- **Bottom-up restricted lower bounds (AC0, AC0[p]).** *Idea/math:* combinatorial restrictions
  (switching lemma) and low-degree polynomial approximation over F_p. *Gap:* these techniques are
  understood to stop at AC0[p] for prime p; for composite modulus m there is no field to support
  the polynomial-approximation argument, and after 1987 the approach produced no ACC lower bound.
  The restrictions have been relaxed only very gradually, and ACC has been the standing wall.

- **Razborov–Rudich naturalization of the bottom-up program.** *Idea:* most combinatorial
  lower-bound arguments yield a natural property. *Gap:* against a class that may contain
  pseudorandom functions this is self-defeating, so a successful argument against a strong class
  must not produce an obviously natural property — it cannot look like the AC0/AC0[p] proofs.

- **Karp–Lipton–Meyer (P = NP ⇒ EXP ⊄ P/poly).** *Idea/math:* under a strong algorithmic
  assumption, simulate every exponential-time computation by guessing a circuit for its history
  and verifying locally inside Σ₂, then collapse Σ₂ via the assumption and diagonalize against the
  time hierarchy. *Gap:* the assumption needed is P = NP, or at best a half-exponential-time
  Circuit-SAT algorithm — wildly stronger than anything achievable, so the implication is vacuous
  as a route to an unconditional bound. The verification step is also for *unrestricted* circuits,
  so it says nothing class-specific.

- **Uniform ACC lower bounds (Allender–Gore; Allender).** *Idea/math:* the SYM⁺ normal form plus
  a counting argument shows the Permanent has no subexponential *uniform* ACC circuits (later:
  no polynomial uniform TC0). *Gap:* the argument requires uniformity (an efficiently computable
  connection language). The difference between uniform and nonuniform may be enormous — it is
  clear that P ≠ NEXP, yet open whether NEXP ⊆ P/poly — so a uniform bound does not touch the
  nonuniform frontier that is actually wanted.

- **Easy-witness / derandomization line (IKW 2002 and relatives).** *Idea/math:* NEXP ⊆ P/poly
  forces succinct satisfying assignments and collapses derandomization. *Gap:* on its own this is
  a conditional structural statement; it converts one hardness assumption into another but does
  not, by itself, produce an unconditional lower bound against any restricted class.

- **Exhaustive search for circuit satisfiability.** *Idea:* try all 2^n assignments, evaluating
  the circuit each time. *Gap:* this is the trivial 2^n · poly(s) baseline; for ACC nothing better
  was known, and it is not obvious that the SYM⁺ structure buys anything algorithmically.

## Evaluation settings

This is a theorem-proving program, so the "evaluation" is the precise mathematical target and the
yardsticks an argument is held to, not an empirical benchmark.

- *The object to separate:* a single explicit language in NEXP (or E^NP / EXP^NP). The cleanest
  target is the NEXP-complete problem SUCCINCT 3SAT (given a circuit C, is the 3-CNF whose
  2^n-bit description is the truth table of C satisfiable?), since a separation for it lifts to the
  whole class by completeness.
- *The strength to beat:* nonuniform ACC of polynomial size — and ideally the bound should be
  robust to strengthening, e.g. to quasipolynomial size, or to an exponential size–depth tradeoff
  for the larger class E^NP, or to slightly nonconstant depth.
- *The contradiction engine:* the nondeterministic time hierarchy theorem
  [Seiferas–Fischer–Meyer 1978; Žák 1983], which forbids NTIME[2^n] ⊆ NTIME[o(2^n)]. By the
  robustness of nondeterministic time across machine models [Gurevich–Shelah 1989], it suffices to
  collapse NTIME[2^n] into NTIME[o(2^n/n^k)] on a random-access machine.
- *The barriers any candidate proof must visibly clear:* it must not relativize (there are oracles
  with NEXP^A ⊆ ACC^A), must not algebrize, and should not yield a natural property in the
  Razborov–Rudich sense.
- *Quality measures for the auxiliary algorithm, should one be involved:* its running time on
  n-input, subexponential-size depth-d ACC circuits, measured against the 2^n exhaustive-search
  baseline; how much faster than 2^n it must be (a savings of a 1/n^{ω(1)} factor versus a savings
  of a 2^{Ω(n^δ)} factor are very different targets); and whether it works for arbitrary
  (nonuniform) input circuits, not just uniform ones.

## Code framework

This discovery lands as a theorem and proof, not as software, so the "code framework" is the
inventory of formal primitives already on the shelf and the empty slots an argument would fill in.
Concretely, the proof of contradiction will instantiate a nondeterministic procedure that, given
an input to an arbitrary NTIME[2^n] language, tries to decide it faster than 2^n; and it will rely
on an auxiliary routine for analyzing circuits from the restricted class. The pre-method scaffold:

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
