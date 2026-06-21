# Context: proving circuit lower bounds, and why the program might be stuck

## Research question

The goal is to prove that some explicit Boolean function — SAT, or another problem in NP — has no polynomial-size Boolean circuits. A proof of `NP ⊄ P/poly` would imply `P ≠ NP`, and it would do so in the stronger, nonuniform setting that escapes the recursion-theoretic obstacles. So the precise target is: take a sequence of explicit functions `{g_n}`, `g_n : {0,1}^n → {0,1}`, and show that for every constant `k`, for all large `n`, the circuit size of `g_n` exceeds `n^k`.

The pain point is that this program has stalled. A decade of combinatorial circuit lower bounds produced strong results for very weak circuit classes and then went silent. The natural next steps — climbing from constant-depth circuits with a single modular gate up toward general polynomial-size circuits — have produced essentially nothing. The question is *why*: is there a structural reason these techniques cannot climb the ladder, analogous to the way an earlier generation of impossibility arguments identified a structural reason that diagonalization could not separate `P` from `NP`?

A solution would have to do one of two things: either supply a technique that provably differs in kind from everything tried so far, or characterize precisely what all the existing techniques have in common and show that this common feature is exactly what blocks them.

## Background

**Relativization.** Baker, Gill, and Solovay (1975) constructed oracles `A`, `B` with `P^A = NP^A` and `P^B ≠ NP^B`. Any proof technique that goes through unchanged when every machine is handed the same oracle — diagonalization and simulation are the canonical examples — cannot settle `P` vs `NP`, because it would have to give the same verdict relative to both oracles. This closed the recursion-theoretic toolbox and pushed the field toward combinatorial circuit complexity, which is not obviously subject to relativization.

**The combinatorial wave (the techniques whose reach is in question).** Through the 1980s a sequence of genuinely new, non-relativizing techniques appeared. They share a flavor: pick a complexity measure or structural feature of a Boolean function, show that small circuits force that feature to be "tame," and exhibit an explicit function whose feature is "wild." The function's wildness then certifies it is not computable by small circuits. The salient instances:

- *Random restrictions / the switching lemma.* Furst–Saxe–Sipser (1984), independently Ajtai (1983), and quantitatively sharpened by Yao (1985) and Håstad (1986), proved PARITY `∉ AC0` (constant-depth, unbounded-fanin AND/OR circuits). A random restriction of the variables almost surely collapses a small `AC0` circuit to a constant, while parity stays sensitive under any restriction that leaves enough variables free.

- *The polynomial / approximation method.* Razborov (1987) and Smolensky (1987) handled `AC0[p]` (constant-depth circuits also allowed MOD-`p` gates, `p` prime). The idea: every depth-`d`, size-`S` circuit over `F_3` can be approximated by a polynomial of degree `(2t)^d` agreeing on at least a `1 − S/2^t` fraction of inputs, yet no low-degree polynomial over `F_3` agrees with parity on more than `49/50` of inputs — contradiction unless `S` is large. Smolensky's bound is `SIZE ≥ 2^{Ω(n^{1/2d})}`.

- *Discrepancy for threshold circuits.* Hajnal–Maass–Pudlák–Szegedy–Turán (1987) showed the inner-product (a Hadamard) function needs exponential-size depth-2 threshold (`TC0`) circuits, because any matrix of low discrepancy is hard for such circuits and Hadamard matrices have low discrepancy.

- *Perceptrons, formula size, switching-and-rectifier networks.* Aspnes–Beigel–Furst–Rudich (1994) bounded the degree of a perceptron approximating parity; Andreev and then Håstad (1993) drove the formula-size lower bound for an explicit function toward `n^3` via the shrinkage exponent under random restrictions; Razborov (1990) gave size bounds for switching-and-rectifier networks via a MINIMUM COVER characterization.

**The recurring shape, stated as facts about the proofs.** Two features keep reappearing across this list, and they are *observable* properties of the proofs as written, knowable before any new theory about them exists:

1. Each proof, when read carefully, identifies a property `C_n` of the *truth table* of a function (high sensitivity; large rank of an associated matrix; high approximate degree; low discrepancy; large minimum cover) and shows that the explicit hard function has `C_n` while small-circuit functions lack it. Deciding `C_n` from the `2^n`-bit truth table is, in every case, doable in time polynomial in `2^n` — usually in a low class: the parity property is decidable in `AC0`, the rank property in `NC2`, the perceptron degree by linear programming in `P`, the discrepancy property in `TC0`.

2. The explicit function is "hard because it looks generic": a counting argument inside each proof shows that a *random* function has the very property `C_n` being exploited, with probability close to one. The proofs do not seize on an idiosyncrasy of parity or inner-product; they exhibit a property shared by almost all functions.

**Pseudorandomness and one-way functions.** A parallel line built cryptographic primitives from complexity assumptions. A one-way function family `f_n : {0,1}^n → {0,1}^{p(n)}` is computable in polynomial time but hard to invert: for every polynomial-time `A` and polynomial `q`, `Pr_x[f_n(A(f_n(x))) = f_n(x)] < 1/q(n)` for large `n`. From a one-way function one builds a pseudorandom generator, and Blum and Micali (1984) gave a concrete generator whose security rests on the discrete logarithm: for a prime `p` and generator `g`, the predicate `B_{p,g}(x) = 1` iff `log_g x ≤ (p−1)/2` is a hard bit, and discrete log is random-self-reducible, so its worst-case and average-case hardness coincide; the standard generator based on it is believed `2^{n^{1/3}}`-hard. Goldreich, Goldwasser, and Micali (1986) then converted any length-doubling pseudorandom generator `G : {0,1}^k → {0,1}^{2k}` into a *pseudorandom function*: writing `G(s) = (G_0(s), G_1(s))` for the first and last `k` bits, define for a seed `s` and an `m`-bit input `x = x_1 … x_m` a walk down a binary tree of depth `m`, where bit `x_i` selects the left or right `k`-bit half at level `i`, and output the first bit of the reached `k`-bit label. A pseudorandom function is one no efficient adversary can distinguish from a uniformly random function even given query access; its security is proved by a hybrid argument over the levels of the tree, each step charged to one invocation of `G`.

**The state of play.** Lower bounds reach `AC0` and `AC0[p]` and stop. They have not come near `NC1`, let alone `P`. Diagonalization is barred by relativization; the combinatorial methods are non-relativizing and were supposed to carry the program upward, but they have not. There is, so far, no account of why.

## Baselines

The prior methods a new account of this stall must reckon with — each with its core mechanism and the gap it leaves:

- **Relativization (Baker–Gill–Solovay 1975).** *Mechanism:* exhibit oracles flipping the answer, so any oracle-invariant technique is powerless. *Gap:* it speaks only to relativizing techniques. The entire combinatorial wave above is non-relativizing — it manipulates the actual circuit, not a black box — so relativization says nothing about why those methods stall. A separate explanation is needed, and BGS does not even hint at its shape.

- **Random-restriction lower bounds (Furst–Saxe–Sipser 1984; Yao 1985; Håstad 1986).** *Mechanism:* small `AC0` circuits die under random restrictions; parity survives. *Gap:* the method is glued to `AC0`. It provably cannot reach `AC0[p]` (modular gates survive restrictions), so something strictly stronger is required at the next rung — already a hint that each rung needs a richer property, but no theory of why the climb terminates.

- **The approximation / polynomial method (Razborov 1987; Smolensky 1987).** *Mechanism:* approximate `AC0[p]` circuits by low-degree polynomials over `F_p`; show the hard function is far from all of them. *Gap:* it is tied to prime-power moduli; `AC0[m]` for composite `m`, and `TC0`, are already out of reach. The certifying content — "the associated matrix has large rank" — is decidable quickly (in `NC2`), which is a clue, not yet a theorem.

- **Discrepancy bounds for threshold circuits (Hajnal et al. 1987).** *Mechanism:* low-discrepancy matrices are hard for depth-2 threshold circuits; Hadamard matrices have low discrepancy. *Gap:* depth-2 only; and again the certifying property (low discrepancy) is `TC0`-decidable and holds for almost all matrices, so the method's own structure is generic in a way no one has yet diagnosed.

- **Counting / diagonalization arguments (folklore).** *Mechanism:* most functions are hard (a counting fact); or build a hard function by diagonalizing against all small circuits. *Gap:* counting tells you a random function is hard but hands you no *explicit* hard function and no efficient way to certify hardness; diagonalization gives an explicit function but is relativizing. Neither has produced a circuit lower bound near `P`. Whatever distinguishes these from the combinatorial methods is exactly the structural question at issue.

## Evaluation settings

The natural yardsticks already in place, against which any claim about the reach of these techniques would be measured:

- **The class hierarchy.** `AC0 ⊆ AC0[p] ⊆ AC0[m] ⊆ ACC0 ⊆ TC0 ⊆ NC1 ⊆ P/poly`, with quasi-polynomial-size variants. The relevant question for any technique is the highest class against which it can prove a lower bound.
- **Explicit target functions.** PARITY and MOD-`q` (for `AC0`, `AC0[p]`); inner-product / Hadamard (for threshold); Andreev's function (for formula size); discrete logarithm's hard bit `B_{p,g}` (as a candidate hard, random-self-reducible function).
- **The complexity measures used as certificates.** Average sensitivity / Fourier mass on high-degree coefficients (random restrictions); rank of a linear map of the truth table (approximation method); approximate / weak degree (perceptrons); matrix discrepancy (threshold); minimum cover size (switching-and-rectifier).
- **The cryptographic yardstick.** The conjectured hardness of pseudorandom generators and functions: the discrete-log generator at `2^{n^{1/3}}`, and the general belief that for any polynomial `p(n)` there are seeded function families on `n` bits requiring `2^{p(n)}` time to distinguish from uniform — measured by the largest circuit a generator can fool.


