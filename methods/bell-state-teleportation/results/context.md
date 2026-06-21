## Research question

Can an unknown qubit be transferred from a sender to a receiver when the two parties share entanglement beforehand but have no quantum channel at the time of transfer, communicating only over a classical channel? The target phenomenon is not remote state description and not duplication. It is exact state relocation: the sender's original state is destroyed and the receiver's system becomes the state.

## Background

The no-cloning theorem sets the central constraint. A device that copies every unknown state would have to act linearly on superpositions, but the linear output for a superposition is not the tensor square of that superposition. This bears on measure-and-reprepare schemes for arbitrary nonorthogonal states and on any protocol that leaves the sender with an intact original while the receiver obtains a perfect copy.

EPR correlations are the available nonclassical resource. A shared maximally entangled pair has no local description that carries an unknown state in advance: each party's reduced state is maximally mixed. The pair is a nonlocal resource because a joint operation on the sender's input and her half of the pair can change which state the receiver's half is conditionally assigned to.

Classical information is also available. Entanglement alone does not transmit a chosen message or a usable unknown state, because the receiver's local statistics before any classical message are independent of the sender's input. A protocol thus has two kinds of resource to work with: a consumed nonclassical correlation, and ordinary classical bits.

## Baselines

- Direct quantum transmission: physically send the particle or swap the state into another carrier and send that carrier. This transfers the state by moving a quantum system between the parties.
- Classical measurement and re-preparation: measure the input, send the result, and prepare a new system. This reproduces a known orthogonal set from the recorded outcome.
- Universal copying before transmission: clone the unknown state, keep one copy, and send another. This is forbidden by linearity.
- EPR correlation alone: a shared pair gives correlated local outcomes between the two parties.
- Separate local measurements with classical communication: local operations exchange ordinary outcome data over the classical channel.

## Evaluation settings

The method is evaluated as an exact theoretical protocol for a single qubit

`|psi> = alpha|0> + beta|1>`, with `|alpha|^2 + |beta|^2 = 1`.

The protocol should satisfy four algebraic checks. First, each measurement outcome occurs with probability `1/4`, independent of `alpha` and `beta`. Second, conditioned on the outcome, the receiver's qubit differs from `|psi>` only by a known Pauli operation. Third, averaging over the unknown outcome leaves the receiver's local state maximally mixed, so no faster-than-light signaling is possible. Fourth, after the sender's measurement, the original input system is no longer an independent copy of `|psi>`.

No empirical performance claims are needed here. Teleportation experiments are outside this bounded draft; the artifact only needs the protocol, the derivation identity, and the causal/no-cloning checks.

## Code framework

No implementation is required. The formal scaffold is the three-system state space:

`H_1` for the unknown input at the sender, `H_2` for the sender's half of a shared EPR pair, and `H_3` for the receiver's half.

The proof target is a decomposition of

`|psi>_1 |R>_23`

into an orthonormal joint basis on systems `1,2`, where every basis outcome leaves system `3` in a known unitary transform of `|psi>`. The classical message names that basis outcome. The receiver's correction inverts the corresponding transform. The correctness proof is therefore a basis-expansion identity plus a no-signaling trace check, not a simulation or code path.
