# Context

## Research question

Saying a language `L` is in NP is the same as saying `L` has an efficient *proof system*: a prover
who possesses unlimited computing power writes down a string `α` (a "proof"), and a verifier, running
in time polynomial in the length of the input `x`, checks `x` and `α` and is convinced that `x ∈ L`.
This is the classical notion of "communicating a proof," and it works: to prove a propositional
formula `F` is satisfiable, send a satisfying assignment `I`; to prove a graph is Hamiltonian, send a
Hamiltonian tour.

But these proofs *over-deliver*. After seeing the satisfying assignment `I`, the verifier knows far
more than the single bit "`F ∈ SAT`" — it now holds an actual satisfying assignment, something it
might not have been able to find on its own in polynomial time. The Hamiltonian tour is a tour, not
just the fact of Hamiltonicity. The proof leaks its *witness*.

The question is whether a proof can convey *only* the truth of the statement and nothing more. To even
ask it precisely requires answering a prior question: what does "nothing more" mean for a verifier that
is a computational device? It is not obvious what the right yardstick for "knowledge" is here — a
verifier truthfully told a single bit has clearly suffered no harm, yet some notions of "information"
would count that bit, while others credit an unbounded verifier with knowing every logical consequence
already. The applications are cryptographic: in a protocol one party often must convince another of
some fact ("this number is a product of two primes," "I followed the protocol honestly") without
handing over the secret that makes the fact true.

## Background

**Proof systems and the NP yardstick.** The complexity-theoretic notion of an efficient proof
(Cook 1971; Cook–Reckhow on the relative efficiency of propositional proof systems) fixes the
prevailing picture: an efficient proof of membership is a short certificate the verifier checks in
polynomial time. NP is exactly the class of languages with such proof systems. The certificate is, by
construction, re-checkable by the verifier — which is precisely why it cannot be hidden: anything the
verifier must re-check, it ends up holding.

**Probabilistic, interactive generalizations of "proof."** Several lines loosen the NP picture by
adding randomness and dialogue. Babai's *Arthur–Merlin games* let an all-powerful Merlin convince a
randomized Arthur, but Arthur's messages are restricted to *public* random coins that Merlin sees.
Babai–Szemerédi exhibit "matrix group" problems with such proof systems that are not known to be in
NP. A separate, more flexible idea — letting the verifier flip coins and *keep them secret*, sending
the prover only some function `f(R)` of its randomness — appears implicitly in Blum's
coin-flipping-by-telephone protocol and in Goldwasser–Micali's protocols below. Whether public coins
are enough for recognizing languages is a different question from whether a verifier's hidden
randomness can keep an interaction from leaking a witness.

**Security definitions for semantic-secure encryption.** Goldwasser–Micali (*Probabilistic
Encryption*, JCSS 1984) defined security of an encryption scheme. They reject the notion that security
means "the adversary cannot recover the whole message" and instead demand *semantic security*:
**whatever an eavesdropper can compute about the cleartext given the ciphertext, it can also compute
without the ciphertext.** Formally this rests on *polynomial security* — for any two messages an
efficient adversary picks, it cannot distinguish their encryptions: for every poly-size circuit (the
"line-tapper") the probability of outputting 1 on an encryption of `m₁` versus `m₂` differs by less
than `1/poly`. Security is thus phrased as the *indistinguishability of two probability distributions*
to a bounded judge. Yao's work on trapdoor functions and pseudorandomness uses the same
distribution-indistinguishability lens.

**The number theory the protocols will stand on.** For an integer `x`, let `Z*_x = {y : 1 ≤ y < x,
gcd(x,y) = 1}`; membership is testable in polynomial time. `y ∈ Z*_x` is a *quadratic residue* (QR)
mod `x` if `y ≡ w² (mod x)` for some `w ∈ Z*_x`, else a *quadratic nonresidue* (QNR). Define the
predicate `Q_x(y) = 0` if `y` is a QR and `1` otherwise.
- `y` is a QR mod `x` iff it is a QR modulo every prime factor of `x`.
- Given the prime factorization of `x`, `Q_x(y)` is computable in polynomial time.
- The *Jacobi symbol* `(y/x) = ∏ (y/p_i)^{a_i}` over the prime power factorization is computable in
  polynomial time *without* the factorization (Euclidean-style). If `(y/x) = −1`, then `y` is a QNR
  and `Q_x(y) = 1` — easy. The hard case is `(y/x) = +1`: deciding `Q_x(y)` there is the *quadratic
  residuosity problem* (QRP), for which the best known algorithm is to factor `x` first, and which is
  conjectured to be as hard as factoring.
- `Q_x` is multiplicative on `Z*_x`: if `Q_x(y) = Q_x(z) = 0` then `Q_x(yz) = 0`; if `Q_x(y) ≠ Q_x(z)`
  then `Q_x(yz) = 1`. Every quadratic residue has the same number of square roots mod `x`,
  independent of which residue.

Define the languages
`QR = {(x,y) : y is a QR mod x}` and `QNR = {(x,y) : y ∈ Z*_x, (y/x) = +1, Q_x(y) = 1}`.
Both lie in `NP ∩ co-NP` (a square root witnesses residue membership; the factorization of `x` decides
either side), yet
neither is known to be decidable in probabilistic polynomial time. They are the natural test cases for
a proof system that hides its witness, because the obvious NP proof — handing over `x`'s factorization
— would give away exactly the secret whose hardness everything rests on.

**An early "don't reveal the factorization" protocol.** Blum's protocol for the language `BL` of
certain Blum integers, and Goldwasser–Micali's protocols for `GM1` (integers with exactly two prime
divisors) and `GM2`, already let a prover demonstrate membership *without sending the factorization*;
the verifier's secret challenges keep the prover from cheating. Under the assumption that factoring is
hard, these protocols are conjectured not to give away the factorization.

## Baselines

- **NP proof systems (Cook 1971).** Core idea: `x ∈ L` iff there is a short certificate `α` such that
  a deterministic poly-time verifier accepts `(x, α)`. The whole of `L`'s difficulty is pushed into
  finding `α`; checking is easy. For SAT the verifier ends up holding a satisfying assignment; for QR,
  the natural certificate is a square root or the factorization.

- **Arthur–Merlin games (Babai); Babai–Szemerédi.** Core idea: an interactive proof in which an
  all-powerful Merlin and a randomized Arthur alternate, but every Arthur message is a fresh public
  random string Merlin gets to see (*public coins*). This already recognizes languages
  (matrix-group nonmembership/order) not known to be in NP, so interaction + randomness adds
  recognizing power beyond static proofs.

- **Blum coin-flipping / `GM1`, `GM2` membership protocols (Blum; Goldwasser–Micali).** Core idea:
  interactive protocols by which a prover convinces a verifier that an integer `n` has a special
  multiplicative structure (e.g. `n ∈ BL`, or `n` has exactly two prime factors) without sending `n`'s
  factorization; the verifier's secret challenges keep the prover from cheating. Under the hardness of
  factoring, these are believed not to reveal the factorization.

- **Semantic / polynomial security for encryption (Goldwasser–Micali 1984).** Core idea: phrase "the
  adversary learns nothing" as "whatever it computes from the ciphertext it could compute without it,"
  and make that rigorous via indistinguishability of distributions to a bounded judge (poly-size
  circuit), with `|Pr[C(E(m₁))=1] − Pr[C(E(m₂))=1]| < 1/poly`. This is a definition for *encryption*,
  a one-way transfer of a single hidden message, with a fixed adversary (a passive line-tapper).

## Evaluation settings

The yardstick is mathematical, not empirical: a proposed proof system for a language `L` is judged by
whether it can be *proven* to satisfy three properties.
- **Completeness:** for `x ∈ L`, the honest prover makes the honest verifier accept with overwhelming
  probability — e.g. acceptance probability at least `1 − |x|^{-k}` for every constant `k` and all
  sufficiently long `x`.
- **Soundness:** for `x ∉ L`, *no* prover strategy (of arbitrary computational power) makes the
  verifier accept with more than negligible probability — at most `|x|^{-k}` for every `k`.
- **Releases no knowledge:** the property whose definition is the open problem — there is no agreed
  way yet to say what, or how much, the verifier learns beyond `x ∈ L`, nor to measure it.
  The natural candidate languages on which to demonstrate it are `QR` and `QNR` — in `NP ∩ co-NP` but
  not known to be in probabilistic polynomial time. Graph isomorphism is a related candidate
  (believed not NP-complete, not known in P).
