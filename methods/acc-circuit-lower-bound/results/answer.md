# Non-Uniform ACC Circuit Lower Bounds — The Algorithmic Method

## Problem

ACC is the class of constant-depth, polynomial-size circuit families over unbounded fan-in
AND, OR, NOT, and MOD_m gates for an arbitrary constant m > 1 (a MOD_m gate outputs 1 iff m
divides the sum of its inputs); equivalently ACC = ∪_m AC0[m]. Strong lower bounds had stalled
at AC0[p] for prime p (no field exists for composite m), to the point that it was unknown whether
even EXP^NP could be computed by depth-three circuits of only MOD_6 gates. The goal is an
unconditional nonuniform lower bound against ACC for an explicit large class — one that survives
relativization, algebrization, and natural-proofs barriers.

## Key idea (the algorithmic method)

Invert the usual relationship between algorithms and lower bounds: a **satisfiability algorithm
for a circuit class C that beats exhaustive search by even a tiny factor implies a circuit lower
bound against C**. Concretely, if C-Circuit-SAT on n-input circuits can be solved a 1/n^{ω(1)}
factor faster than the trivial 2^n, then NEXP ⊄ C. The proof guesses C-circuit "representatives"
of exponentially long objects (a satisfying-assignment witness, and a C-equivalent of the
Cook–Levin circuit) and verifies them with that faster SAT algorithm, collapsing NTIME[2^n] into
NTIME[o(2^n)] and contradicting the nondeterministic time hierarchy. The "spinning" half uses only
that C ⊇ AC0 and C is closed under composition; the class-specific work is confined to building the
fast SAT algorithm, which for ACC is delivered via the SYM⁺ normal form.

## Main theorems

**Theorem 1 (lower bound for NEXP).** NTIME[2^n] — hence NEXP — does not have nonuniform ACC
circuits of polynomial size. The bound extends to quasipolynomial and even sub-third-exponential
size, and down to grotesque classes just above NTIME[2^n].

**Theorem 2 (exponential size–depth tradeoff for E^NP).** For every d and m there is a δ > 0 and a
language in E^NP (= TIME[2^{O(n)}] with an NP oracle) that has no nonuniform depth-d ACC circuits of
size 2^{n^δ} with MOD_m gates.

**Theorem 3 (metatheorem).** Let C be any circuit class containing AC0 and closed under composition
(e.g. ACC, TC0, NC1, formulas, P/poly). There is a k > 0 such that, if C-Circuit-SAT on n-variable,
n^c-size circuits can be solved in O(2^n / n^k) time for every c, then NTIME[2^n] has no nonuniform
polynomial-size C-circuits.

## Proof

### Part A — Spinning SAT algorithms into lower bounds (uses only AC0 ⊆ C, composition)

**Ingredient 1 (efficient Cook–Levin for NEXP).** SUCCINCT 3SAT (given a circuit C, is the 3-CNF
whose 2^n-bit truth table is T(C) satisfiable?) is NEXP-complete (Papadimitriou–Yannakakis). The
sharpened reductions (Cook 1988; Robson 1991; Fortnow–Lipton–van Melkebeek–Viglas 2005) give, for
every L ∈ NTIME[2^n] and input x of length n, a poly(n)-time-computable circuit C_x with
**n + O(log n) inputs** and poly(n) size such that x ∈ L iff F_{C_x} is satisfiable.

**Ingredient 2 (time lower bound).** If SUCCINCT 3SAT had a nondeterministic algorithm running in
2^{n − ω(log n)} time on n-input poly-size circuits, then via Ingredient 1 every L ∈ NTIME[2^n]
would be in NTIME[2^{(n+O(log n)) − ω(log n)} poly(n)] = NTIME[o(2^n)], contradicting the
nondeterministic time hierarchy theorem (Seiferas–Fischer–Meyer 1978; Žák 1983). The tight input
count n + O(log n) is essential: with 3n + O(log n) inputs one would need a 2^{n/3}-beating
algorithm, far harder.

**Ingredient 3 (easy witnesses).** If NEXP ⊆ C ⊆ P/poly, then (Impagliazzo–Kabanets–Wigderson 2002)
every satisfiable succinct instance C_x has a small witness circuit W_x whose truth table is a
satisfying assignment to F_{C_x}; since P ⊆ C makes Circuit-Value ∈ C, W_x can be taken in C. For
E^NP, the witness comes directly: binary-search the lex-least satisfying assignment with the NP
oracle (an E^NP function), giving an S(3n)-size C witness if E^NP ⊆ C of size S.

**The verification reduces to two SAT calls.** Assume for contradiction NEXP ⊆ ACC (so P ⊆ ACC).
For L ∈ NTIME[2^n] and input x, the following nondeterministic algorithm decides x ∈ L:

```
SATALG5(x):
    C_x ← efficient Cook–Levin circuit for x        # n + O(log n) inputs, poly(n) size, unrestricted
    guess ACC circuits D(x,·), E(x,·,·)             # gate-connection and gate-value circuits of C_x
                                                    #   (exist since they compute poly-time functions, P ⊆ ACC)
    verify D directly: evaluate D(x,·) on all |C_x| gate indices, check it matches C_x  # poly time
    build VALUE(i,j) from D, E:                      # local gate-consistency check
        (j1,j2,g) ← D(x,j);  v1,v2,v ← E(x,i,j1),E(x,i,j2),E(x,i,j)
        VALUE = 0  iff  E's claim at gate j on input i is consistent with type g
    if VALUE is satisfiable:  reject                 # one ACC-SAT call: certifies E correct
    x' ← E(·, j*)   where j* = output gate of C_x    # ACC circuit equivalent to C_x
    guess ACC witness W'_x                           # encodes a satisfying assignment (Ingredient 3)
    build D'(i) from x' and 3 copies of W'_x:        # iterate clauses by feeding index i to x'
        x'(i) prints clause i's 3 (index, sign) pairs; W'_x reads off the 3 assigned values
        D'(i) = 0  iff  clause i is satisfied by the assignment encoded by W'_x
    accept iff D' is unsatisfiable                   # one ACC-SAT call: certifies W'_x satisfies F
```

Correctness: `D` is checked directly (base case, no SAT call, stops any regress). Given correct `D`,
`VALUE` is unsatisfiable iff `E` reports the true value of every gate of C_x on every input, so the
single ACC-SAT call certifies `E`, whence `x' = E(·, j*)` is a genuine ACC equivalent of C_x. The
witness `W'_x` exists iff x ∈ L (Ingredient 3), and `D'` is unsatisfiable iff `W'_x` encodes a
satisfying assignment to F_{C_x} iff x ∈ L. Every circuit handed to ACC-SAT is ACC (this needs only
AC0 ⊆ C and closure under composition, so all copies/compositions stay in the class), has
n + O(log n) inputs, and quasipolynomial size.

If ACC-SAT runs in O(2^n / n^k) time, SATALG5 runs in O(2^n / n^k) time, so
NTIME[2^n] ⊆ NTIME[o(2^n)] — contradiction. Hence a faster ACC-SAT algorithm ⇒ NEXP ⊄ ACC. ∎ (Part A)

### Part B — A faster-than-2^n ACC satisfiability algorithm (uses ACC structure)

**Ingredient 4 (SYM⁺ normal form).** A SYM⁺ circuit is a symmetric function of ANDs of (non-negated)
variables. Every depth-d, size-s ACC circuit with MOD_m gates has an equivalent SYM⁺ circuit of size
s^{O(log^{f(d,m)} s)} with ANDs of O(log^{f(d,m)} s) fan-in, where f(d,m) ≤ m^{O(d)}, constructible in
that much time (Yao 1990; Beigel–Tarui 1994; Allender–Gore 1991; Green et al. 1995). Construction:
make the circuit a tree; randomize AND/OR into MOD_p of polylog-ANDs (Valiant–Vazirani); split
composite moduli by CRT and reduce MOD_{p^e} to constant-fan-in ANDs of MOD_p; derandomize to a
MAJORITY top gate; then push each MOD_p layer into the top symmetric gate using the modulus-amplifying
polynomial
  P_k(x) = 1 − (1 − x)^k ( Σ_{i=0}^{k−1} C(k+i−1, i) x^i ),
which satisfies P_k(x) ≡ 0 mod p^k for x ≡ 0 mod p and ≡ 1 mod p^k for x ≡ 1 mod p; with
Q_k(x) = P_k(x^{p−1}) and Fermat, 1 − Q_k(Σ y_i) ≡ 1 mod p^k iff p | Σ y_i. The final symmetric
function is F(v) = MAJORITY((…((v mod p_1^{k_1}) mod p_2^{k_2})… mod p_{d'}^{k_{d'}}).

**Ingredient 5 (Evaluation Lemma).** A SYM⁺ circuit of size s ≤ 2^{0.1n} on n inputs, with a
symmetric function evaluable in poly(s) time, can be evaluated on **all** 2^n assignments in
(2^n + poly(s)) · poly(n) time. Proof (dynamic programming via the zeta transform): index each AND_j
by its variable subset G_j ⊆ [n]; let f(S) = #{ j : G_j = S } (built in O(2^n + s·poly(n))). The
number of ANDs satisfied by the assignment T is g(T) = Σ_{S⊆T} f(S), the zeta transform of f, computed
by Yates's 1937 DP: g_0 = f and g_i(T) = g_{i−1}(T) + g_{i−1}(T\{i}) if i∈T else g_{i−1}(T), for
i = 1…n; induction gives g_n = g, in O(2^n·poly(n)). Apply F to each count. (Two alternative proofs:
Coppersmith's rectangular matrix multiplication — N×N^{.1} by N^{.1}×N in O(N²log²N), splitting inputs
into two halves to count satisfied ANDs per pair; and coefficient-to-point multilinear evaluation, a
multidimensional FFT, O(2^n·poly(n)) via R(2^n) = 2R(2^{n−1}) + O(2^n poly n).)

**The algorithm (k-blowup to get below 2^n).**

```
ACCSAT(C):                                  # C: depth-d ACC, MOD_m, n inputs, size s = 2^{n^ε}
    k ← n^{1/(2 f(d,m))}
    C' ← OR over all 2^k settings of a k-subset of inputs, with that subset substituted into C
            # C': n − k free inputs, size ≤ 2^k · s, still ACC (one more depth level), sat iff C sat
    (F, {AND_j}) ← SYM⁺ normal form of C'   # Ingredient 4; K = 2^{O(k f(d,m) + log^{f(d,m)} s)} ANDs
    evaluate on all 2^{n−k} assignments via the zeta-transform DP   # Ingredient 5
    output "satisfiable" iff F(count) = 1 for some assignment
```

For s ≤ 2^{n^{o(1)}} and k = n^{1/(2f(d,m))}, the monomial count is K ≤ 2^{n^{2/3}}, and the running
time is O(2^{n−k}·poly(n) + K) = 2^{n − Ω(n^{1/(2f(d,m))})}.

**Theorem 4 (ACC-SAT).** For every d, m there is ε ∈ (0,1) such that satisfiability of depth-d ACC
circuits with MOD_m gates, n inputs, and 2^{n^ε} size is decidable in 2^{n − Ω(n^δ)} time for some
δ > ε depending only on d and m. ∎ (Part B)

### Combine

Theorem 4 gives ACC-SAT in 2^{n − Ω(n^δ)} ≤ O(2^n / n^k) time for every k. Part A turns this into
NEXP ⊄ ACC (Theorem 1) and, with the E^NP witness, the exponential size–depth tradeoff (Theorem 2).
Since Part A used only AC0 ⊆ C and closure under composition, it is the metatheorem (Theorem 3): a
faster C-SAT algorithm yields NTIME[2^n] ⊄ poly C for any robust C. Improved exponential-time
satisfiability algorithms are the sole remaining barrier to lower bounds against TC0, P/poly, and NP.

### Why the barriers are cleared

- **Natural proofs:** the proof uses completeness of SUCCINCT 3SAT and diagonalization (the time
  hierarchy), not a combinatorial property of the hard function; no P-natural property is produced.
- **Relativization / algebrization:** the only non-black-box step is the ACC-SAT algorithm, which
  exploits the SYM⁺ *structure* of ACC; all known faster-than-2^n SAT algorithms break when an oracle
  or algebraic extension is added. There are oracles A with NEXP^A ⊆ ACC^A, and the proof correctly
  does not relativize because it reaches inside the circuit.
