Start with the obstruction. A NEXP computation has an exponentially large witness or computation
history. A polynomial-time verifier cannot read that history, and a normal NP certificate model has no
place to put it. If the verifier only samples a few entries from a claimed exponential table, a single
prover can answer each sample as if it came from a different convenient table. Local plausibility alone
does not imply global consistency.

The multi-prover idea changes exactly that point. The provers may agree on a strategy beforehand, but
after the verifier sends them questions they are isolated. This makes every question double as a
commitment test. Ask prover A for a local view of the alleged global proof: values on a line, a plane, a
small constraint neighborhood, or a low-degree restriction. Ask prover B for one randomly chosen point or
subview that should overlap with A's answer. Prover A does not know which part will be checked; prover B
does not see A's local story. If they agree with high probability, their answers cannot be arbitrary
per-query fabrications. They must behave as though they are reading from one common global object.

This is the sense in which "isolating provers creates constraints." The verifier has not made the
provers less powerful computationally; it has removed their ability to coordinate after seeing the
random challenges. That missing communication turns overlap checks into binding constraints. A single
prover can always maintain a locally consistent fiction adapted to the transcript. Two separated provers
must maintain compatible fictions across unpredictable overlaps, and high compatibility is strong enough
to reconstruct or reason about an underlying global proof.

Once there is a way to enforce a global proof oracle, the NEXP lower bound becomes plausible. A NEXP
machine on input `x` has an exponentially long accepting tableau if `x` is in the language. By a
Cook-Levin style transformation at exponential scale, this tableau can be represented as an
exponentially large constraint system whose local constraints are succinctly computable from `x`. The
honest provers behave as oracles for an encoding of the satisfying assignment to this system.

The encoding is algebraic. Instead of treating the huge witness as a raw bit string, map it to a
low-degree extension over a finite field. Low-degree polynomials have two useful properties: they spread
errors out, and they can be checked locally. If two low-degree polynomials differ, Schwartz-Zippel style
reasoning says they disagree on many random points. If a function is not close to low-degree, line and
multilinearity tests catch it with noticeable probability. These are PCP-like ideas: make the proof
redundant enough that local samples reveal global faults.

The verifier then checks two kinds of statements. First, the provers' answers must be restrictions of
one low-degree object. That is the consistency and multilinearity part, enforced through cross-checks
between isolated provers. Second, the low-degree object must encode an accepting computation. That is
the arithmetic constraint part: sample a random local constraint, arithmetize its satisfaction condition,
and check the relevant values supplied by the provers. If the encoded tableau is valid, honest provers
pass. If no accepting tableau exists, then either the provers are inconsistent with any single low-degree
encoding, or they are consistent with an encoding that violates a non-negligible fraction of the
arithmetized constraints. The verifier catches the first case by overlap and low-degree tests, and the
second by random constraint checks.

The key contrast with NP is that the verifier never receives a polynomial-size certificate summarizing
the exponential computation. Instead, interaction plus isolation lets the verifier use the provers as
random-access access points to an exponentially long encoded proof. The proof is too large to read, but
its local consistency can be tested probabilistically because the provers cannot adapt jointly to the
same hidden random check. This is the mechanism by which interactive proof verification expands from
checking NP witnesses to checking nondeterministic exponential-time computations.

The upper bound, `MIP ⊆ NEXP`, is conceptually easier. A nondeterministic exponential-time machine can
guess complete prover strategies for all polynomial-length questions and transcripts, then compute the
verifier's acceptance probability by enumerating its exponentially many random choices and possible
message histories. Since the verifier itself runs in polynomial time, all questions, answers, and
transcripts have polynomial length, so the full strategy table has exponential size. That fits inside
NEXP. Thus multi-prover interaction is powerful enough to reach NEXP, but not beyond it.

Putting both directions together gives `MIP = NEXP`. The memorable insight is not just "more provers
are stronger." It is that noncommunication turns separate provers into consistency witnesses for a huge
object. Cross-checking makes a hidden exponential proof behave like a locally testable codeword; PCP and
arithmetization make the huge computation locally and probabilistically checkable; the verifier's
polynomial conversation is enough because every sampled inconsistency represents many global
inconsistencies.
