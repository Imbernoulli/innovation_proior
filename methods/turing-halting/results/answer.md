# The unsolvability of the Entscheidungsproblem, via the Turing machine

## The problem it solves

Hilbert's **Entscheidungsproblem** asks for a definite general method that decides, for any formula of first-order logic, whether it is provable (equivalently, by Gödel's completeness theorem, valid). To prove that *no* such method exists, one must first make "effective method" precise — a class to quantify over. The result does both: it defines computation as an abstract machine and then proves that no machine can decide provability.

## The key idea

1. **Define "effective method" by the abstract machine.** Analyze a human computer reducing every step to its simplest parts. Finiteness is forced at each point — finitely many internal states (else two are confusable), a finite symbol alphabet (else two are arbitrarily close), a one-dimensional tape (the second dimension is inessential), a bounded scanned window, at most one symbol changed and a bounded shift of attention per step, and behaviour fixed by the (scanned symbol, state) pair. This yields the **a-machine**: a finite control of m-configurations q₁…q_R, a tape over a finite alphabet S₀(blank), S₁,…, and a transition table. Its claim to capture *all* effective computation is the finiteness analysis itself, not fiat.

2. **Encode and enumerate machines.** Standardize each table line to "q_i S_j S_k {L/R/N} q_l." Encode q_i → D A^i, S_j → D C^j, join lines with ";" — the **standard description (S.D.)** over {A,C,D,L,R,N,;}. Map A→1,C→2,D→3,L→4,R→5,N→6,;→7 to get the **description number (D.N.)**, an integer. Every computable sequence has a D.N.; finite descriptions are countable, so the computable sequences are **countable**.

3. **Universal machine.** A single machine U, given the S.D. of any M on its tape, simulates M and, if M is circle-free, prints M's sequence — because program (the S.D.) and data (the running complete configurations) are both strings of letters U can mark, compare, copy, and update. Computation is closed under "run a description."

4. **Diagonalize against the decider.** A machine is **circle-free** if it prints figures forever (well-behaved), **circular** if it stops producing figures. The naïve Cantor diagonal on the computable sequences contradicts their countability; the false step is the assumption that one can decide which descriptions are circle-free. Aiming the diagonal at *that* decider proves it cannot exist.

5. **Reduce to logic.** Convert "circle-free undecidable" → "ever-prints-0 undecidable" → "provability undecidable," via a first-order formula Un(M) whose provability is equivalent to M printing 0.

## The theorems and proofs

### Theorem A (undecidability of circle-free)
*No machine, given a candidate description number, decides whether it is the D.N. of a circle-free machine.*

**Proof.** Suppose a total decider **D** exists: given any candidate description number, it halts with "s" if it is the D.N. of a circle-free machine and "u" otherwise. Fuse D with the universal machine U into a machine **H** that emits the *unflipped* diagonal figure φ_n(n). H runs in sections; in section N it forms N and runs D on N:
- if N is unsatisfactory (D says "u"), set R(N)=R(N−1) and continue;
- if N is satisfactory (D says "s"), set R(N)=R(N−1)+1; then N is the R(N)-th satisfactory number, so the machine with D.N. N is the R(N)-th circle-free machine. H uses U to compute its first R(N) figures and writes the R(N)-th as the next figure of β′.

Each section ends in finite time (D always halts; a circle-free machine's R(N)-th figure is reached in finite time), and there are infinitely many satisfactory numbers (simple periodic printers already supply them), so **H is circle-free**. Let K = D.N.(H). In section K, H tests K:
- D(K) = "u" contradicts H being circle-free;
- D(K) = "s" makes K satisfactory, so R(K)=R(K−1)+1 and H is the R(K)-th circle-free machine in the satisfactory-number ordering. H must compute its own R(K)-th figure before it can emit it — the instruction becomes "compute the first R(K) figures of H and write the R(K)-th," which can never be reached, so **H stalls — is circular** — contradicting both "circle-free" and the verdict "s."

Both verdicts are impossible; D cannot exist. ∎

The unflipped diagonal is crucial: self-application turns the verdict "runs forever" into "stalls," so the contradiction uses the polarity of circle-free itself rather than only the flipped-digit equation β = 1 − φ_n(n).

### Theorem B (ever-prints-0 is undecidable)
*No machine decides, from an arbitrary M's S.D., whether M ever prints 0.*

**Proof.** Suppose a decider **E** exists. From M build M_k by following M while replacing its first k printed 0's with a marked non-figure 0̄. Then "M prints 0 infinitely often" ⇔ "every M_k eventually prints 0": finitely many zeros are exhausted by some k, while infinitely many leave a later zero for every k.

Use E to build one auxiliary machine X_M. It successively writes SD(M₁), SD(M₂),…; after each one it runs E, and X_M prints 0 exactly when E says the tested M_k never prints 0. Since E is total, each stage ends. Now run E once on X_M. If X_M ever prints 0, some M_k never prints 0, so M has finitely many zeros. If X_M never prints 0, every M_k still eventually prints 0, so M prints 0 infinitely often. Thus E decides "M prints 0 infinitely often," and the same construction decides "M prints 1 infinitely often." A machine is circle-free iff it prints 0 infinitely often or 1 infinitely often, so E would decide circle-free — contradicting Theorem A. ∎

The identification "general process" ≡ "machine that determines" is licensed by the machine capturing all effective methods; that is what makes Theorems A–B bear on Hilbert.

### Theorem C (the Entscheidungsproblem is unsolvable)
*No machine decides whether an arbitrary first-order formula 𝔄 is provable in the functional calculus K.*

**Construction.** Predicates encode M's run: R_{S_l}(x,y) = "in complete configuration x, square y bears S_l"; I(x,y) = "in complete configuration x, y is scanned"; K_{q_m}(x) = "in complete configuration x, the m-config is q_m"; F(x,y) = "y is the immediate successor of x," a single successor relation used for both next configurations and neighboring tape squares, with N(x) and the number axioms. Each instruction "q_i S_j → print S_k, move L, q_l" becomes a sentence Inst{…}: for all x,y with successor configuration x′ and left-neighbour y′, if config x has q_i scanning S_j on y, then config x′ has S_k on y, scans y′, has m-config q_l, **and every other square is unchanged** (the frame conjunct). The R and N move cases are parallel: scan the right successor, or keep the same scanned square. Conjoin all Inst's into Des(M). Let A_M(u) be N(u), the successor axioms, the blank initial tape (∀y)R_{S₀}(u,y), I(u,u), K_{q₁}(u), and Des(M). Set

> Un(M) := (∃u)A_M(u) → (∃s)(∃t)[N(s) & N(t) & R_{S₁}(s,t)],

which under the intended reading says "M, run from its start, eventually prints 0."

**Lemma 1.** *If M ever prints 0, Un(M) is provable.* Let CC_n encode the actual n-th complete configuration of M at instant u^{(n)}, and CF_n := A_M(u) & F^{(n)} → CC_n. CF₀ is provable because A_M(u) states the initial blank tape, scanned square u, and m-config q₁. For CF_n → CF_{n+1}, the unique instruction matching the scanned symbol and m-config in CC_n is in Des(M). If it moves left, Inst uses F(y′,y) to scan the left neighbour; if it moves right, it uses F(y,y′); if it does not move, the scanned square remains y. In each case Inst writes the new symbol/state and the frame conjunct carries every other square unchanged, yielding CC_{n+1}. By induction every CF_n is provable. If 0 appears at instant N on square K, CC_N contains R_{S₁}(u^{(N)},u^{(K)}); with N′=max(N,K), the successor chain yields N(s), N(t), and ∃s∃t[N(s)&N(t)&R_{S₁}(s,t)], so Un(M) is provable. ∎

**Lemma 2.** *If Un(M) is provable, M ever prints 0.* Under the intended interpretation, the antecedent (∃u)A_M(u) is true: M has its blank initial configuration and follows Des(M). Soundness of K makes the provable Un(M) true in this interpretation, so the consequent is true: 0 appears somewhere on M's tape. ∎

**Proof of C.** By Lemmas 1–2, Un(M) is provable iff M ever prints 0. A decider for provability, applied to the formulas Un(M), would decide "M ever prints 0" — impossible by Theorem B. Hence provability in K is undecidable: the Entscheidungsproblem has no solution. ∎

**Prenex sharpening.** Un(M) is expressible with all quantifiers in front, of the shape (u)(∃x)(w)(∃u₁)…(∃u_n)𝔅 with u and w universal, 𝔅 quantifier-free, and n = 6; unimportant modifications reduce n to 5. Thus undecidability already holds for a small, fixed quantifier-prefix class.

## Distinction from Gödel
Gödel's incompleteness: within a fixed system there exist sentences 𝔄 with neither 𝔄 nor ¬𝔄 provable, and the system cannot prove its own consistency. This result is different in kind: there is no general *method* deciding, for arbitrary 𝔄, whether 𝔄 is provable. (Had first-order logic been complete in the sense "for each 𝔄, 𝔄 or ¬𝔄 provable," the proof-enumerator would decide provability — running until 𝔄 or ¬𝔄 appears, with consistency of K making ¬𝔄 a verdict of unprovability. It is the absence of that completeness that leaves room for undecidability, but the proof runs through the machine and the diagonal, not through Gödel's sentence.)

## The landing artifact
The artifact is not code but the theorem chain above: **effective computation = the abstract machine** (justified by the human-computer finiteness analysis), made self-referential by **encoding machines as integers** and run by a **single universal machine**, **diagonalized** to prove no machine decides circle-free, then **reduced through "ever prints 0" to provability** to prove the Entscheidungsproblem unsolvable. The small machine printing 010101… and the description-number encoding serve only to keep the abstraction concrete; the construction of D and H is the proof, not a program to run.
