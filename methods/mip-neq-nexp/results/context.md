# Context

## Research question

Can a polynomial-time randomized verifier be convinced of a statement whose ordinary witness and
ordinary verification are exponentially large?

For NP, the verifier receives one polynomial-size certificate and checks it directly. For NEXP, the
witness may have exponential length, and the deterministic check may inspect an exponential computation
tableau. A polynomial-time verifier cannot even read the object it is meant to verify. The question
is whether interaction with several all-powerful provers changes that limit.

## Background

**MIP model.** A multi-prover interactive proof has a probabilistic polynomial-time verifier and two or
more computationally unbounded provers. The provers may coordinate before the protocol starts, but after
the verifier sends questions they cannot communicate. Completeness means honest provers make the
verifier accept true instances with high probability; soundness means no joint cheating strategy can
make the verifier accept false instances except with bounded probability.

**NEXP.** A language is in NEXP if membership can be certified by a nondeterministic computation running
in `2^poly(n)` time. Equivalently, there is an exponentially long witness and an exponential-time
deterministic verifier. A NEXP-complete problem can be expressed as a succinctly described, exponentially
large constraint system: there is a huge assignment, and every local constraint in a huge computation
tableau must be satisfied.

**Arithmetization and PCP perspective.** The LFKN and Shamir line of work showed that Boolean
computations can be converted into low-degree polynomial identities over finite fields, and that random
field evaluations expose false low-degree claims. The PCP viewpoint says a verifier can read only a few
random locations of a proof if the proof is encoded so local inconsistency is spread across many
locations. MIP combines both ideas: use algebraic encodings for the exponential proof, then use isolated
provers to enforce local consistency without reading the proof.

## Baselines

- **NP certificates.** A single static witness works when the witness is polynomial length and the
  verifier can inspect enough of it directly.
- **Single-prover IP.** Arithmetization and sum-check let one prover convince a polynomial-time verifier
  of PSPACE statements, but a single prover can adapt all later answers to the transcript.
- **Naive spot-checking.** Sampling a few cells of an exponentially large tableau is one approach when
  local views are taken independently.
- **Multi-prover cross-checking.** The verifier can ask one prover for a structured local object, such as
  values on a line, plane, or small neighborhood, and ask another prover for one hidden overlap. Because
  neither prover knows exactly how the other will be checked, the protocol places constraints on a
  global consistency condition.

## Evaluation settings

The theorem is evaluated in terms of proof-system power, not experiments.

- **Completeness and soundness:** true NEXP instances have honest strategies accepted with high
  probability, while false instances are rejected with constant probability against every noncommunicating
  cheating strategy; error can be reduced by repetition.
- **Verifier resources:** polynomial time, polynomial randomness, polynomial communication, and a
  constant or polynomially bounded number of rounds depending on the formulation.
- **Prover resources:** computationally unbounded, allowed pre-agreement, but no post-question
  communication.
- **Canonical checked object:** a succinct exponential constraint system or computation tableau encoded
  as a low-degree extension over a finite field.
- **Soundness tools:** low-degree testing, multilinearity testing, Schwartz-Zippel style random
  evaluation bounds, and consistency tests between overlapping prover answers.
