# Context

## Research Question

By the early 1990s, two pressures point in the same direction without yet looking identical.

The first comes from approximation. Cook, Levin, and Karp have made exact satisfiability, clique, vertex cover, and many other decision problems into one hardness world. But an exact reduction is brittle near the optimum. A theory of inapproximability needs reductions that preserve a numerical gap: satisfiable instances should map to optimum 1, while unsatisfiable instances should map to optimum bounded away from 1.

The second comes from proof verification. A deterministic verifier for NP reads the whole witness. Interactive proofs show that randomness and algebra can verify much more surprising claims, and multi-prover proofs show that consistency checks can replace direct trust. The question is whether ordinary NP witnesses can be rewritten so that a randomized verifier inspects only a few carefully chosen locations while still rejecting false claims reliably.

## Background

Classical NP verification is global. For a language `L`, membership means there is a polynomial-size witness `pi` such that a deterministic polynomial-time machine accepts `(x, pi)`, and every witness is rejected when `x` is not in `L`. This is powerful enough for exact decision reductions but offers no local sampling guarantee.

Interactive proof systems change the verifier. LFKN arithmetize Boolean claims into low-degree polynomial statements and use random challenges to check sums over exponentially many points. Shamir extends this line to `IP = PSPACE`. BFL then shows that two noncommunicating provers give `MIP = NEXP`, using arithmetization plus multilinearity testing. In oracle form, this says an exponentially long proof-like object can be checked by randomized local access.

Transparent-proof work scales the idea toward NP. BFLS show that computations can be checked in polylogarithmic Monte Carlo time when the theorem candidate is encoded. FGLSS then connects low-access verification to clique approximation by turning accepting local views into vertices and consistency into edges. The soundness gap of the verifier becomes a numerical gap in clique size.

## Baselines

- Static NP witness: exact, deterministic, and complete; the verifier reads the whole witness.
- LFKN sum-check: a global polynomial-sum claim can be checked through random univariate consistency checks; the protocol is interactive and operates over a fixed proof string with adaptive queries.
- BFL multi-prover verification: `NEXP` gets efficient randomized proof systems through arithmetization and multilinearity testing, operating at the exponential scale.
- BFLS transparent proofs: computations can be checked in polylogarithmic time after error-correcting encoding.
- FGLSS clique construction: proof-checking soundness becomes an approximation gap by turning accepting local views into vertices and consistency into edges.
- Algebraic self-testing: BLR and Rubinfeld-Sudan show how local tests can certify global algebraic structure.

## Evaluation Settings

The object is a decision-language verifier with random access to a proof string. The parameters are:

- random bits used by the verifier;
- proof bits or proof symbols read;
- completeness, taken as accepting true inputs with probability 1;
- soundness, taken as accepting false inputs with probability at most 1/2;
- proof length, which must remain polynomial in the input size.

The associated optimization yardsticks are gap constraint systems, MAX-3SAT, and clique. For a constraint system `C`, `UNSAT(C)` is the minimum fraction of violated constraints over all assignments. A useful reduction distinguishes `UNSAT(C) = 0` from `UNSAT(C) >= alpha` for a fixed positive constant `alpha`.

The algebraic yardsticks are finite fields, low-degree polynomial tables, relative Hamming distance, line and curve restrictions, and the fact that distinct degree-d polynomials over a sufficiently large field disagree on most points.

## Available Ingredients

The pieces on the table are a Cook-Levin style NP-complete relation, arithmetization over finite fields, sum-check reasoning, local tests for linearity and low degree, and the verifier-to-gap conversion of FGLSS.
