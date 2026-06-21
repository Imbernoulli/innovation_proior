# MIP = NEXP

## Result

**MIP = NEXP:** the languages with multi-prover interactive proofs are exactly the languages decidable
by nondeterministic exponential-time machines.

The surprising direction is `NEXP ⊆ MIP`. A polynomial-time verifier can check an exponentially large
computation by interrogating multiple all-powerful provers who cannot communicate after seeing their
questions.

## Core insight

The unique insight is that **isolated provers manufacture constraints**.

With one prover, a verifier asking for a few locations of a huge proof only gets a story adapted to the
current transcript. With two noncommunicating provers, the verifier can ask one prover for a local view
of a huge encoded computation and ask the other for a hidden overlapping part. If they agree under many
random overlaps, their answers are forced to look like restrictions of one global object. The verifier
has converted noncommunication into a consistency test.

That is why MIP reaches beyond NP-style verification. NP checks a short witness. MIP checks an
exponentially long witness indirectly: the witness is encoded as a low-degree / PCP-like object, and the
verifier samples local constraints while cross-checking the provers' overlapping answers.

## Protocol Shape

1. Convert the NEXP computation into a succinctly described exponential constraint system.
2. Encode the alleged accepting tableau as a low-degree extension over a finite field.
3. Ask one prover for a local algebraic view and another prover for a random overlap.
4. Reject on disagreement, failed low-degree or multilinearity tests, or a violated sampled constraint.
5. Accept if all sampled local checks are consistent.

Completeness is straightforward: honest provers answer from the same valid encoded tableau. Soundness
comes from the split: cheating provers either fail to define one consistent global low-degree object, or
that object violates the computation constraints and is caught by random local tests.

## Why Exactly NEXP

`NEXP ⊆ MIP` follows because the verifier can probabilistically check a succinct exponential tableau via
cross-checked local views. `MIP ⊆ NEXP` follows because an exponential-time nondeterministic machine can
guess the provers' full strategies and enumerate the verifier's random choices to evaluate acceptance.

So the added power is precise: multiple isolated provers let interaction scale from checking polynomial
NP witnesses to checking exponential NEXP witnesses, but the full strategy space is still only
exponential.
