## Research question

Can an unknown qubit be transferred from a sender to a receiver without copying it, when the sender and receiver have no direct quantum channel at the time of transfer? The target phenomenon is not remote state description and not duplication. It is exact state relocation: the sender's original state is destroyed, the receiver's system becomes the state, and the only online communication is a classical two-bit message conditioned on a joint measurement in the right basis.

## Background

The no-cloning theorem sets the central constraint. A device that copies every unknown state would have to act linearly on superpositions, but the linear output for a superposition is not the tensor square of that superposition. This blocks a measure-and-reprepare story for arbitrary nonorthogonal states and also blocks any protocol that leaves the sender with an intact original while the receiver obtains a perfect copy.

EPR correlations provide the central resource. A shared maximally entangled pair has no local description that carries the unknown state in advance: each party's reduced state is maximally mixed. The pair is nevertheless a nonlocal resource because a joint operation on the sender's input and her half of the pair can change which state the receiver's half is conditionally assigned to.

Classical information is still required. Entanglement alone cannot transmit a chosen message or a usable unknown state, because the receiver's local statistics before the classical message are independent of the sender's input. The protocol must therefore separate two roles: a consumed nonclassical correlation that supplies the channel for quantum coherence, and ordinary classical bits that identify which correction the receiver must apply.

## Baselines

- Direct quantum transmission: physically send the particle or swap the state into another carrier and send that carrier. This transfers the state but does not explain how a state can be relocated using only pre-shared entanglement plus later classical communication.
- Classical measurement and re-preparation: measure the input, send the result, and prepare a new system. This works only for a known orthogonal set. It fails for an arbitrary unknown qubit because a single copy cannot reveal the amplitudes without disturbance.
- Universal copying before transmission: clone the unknown state, keep one copy, and send another. This is forbidden by linearity and would also create superluminal signaling problems in EPR settings.
- EPR correlation without the sender's joint measurement: a shared pair gives correlated outcomes but does not by itself encode the chosen unknown state at the receiver.
- Separate local measurements with classical communication: local operations can exchange ordinary outcome data, but they do not replace the specific joint measurement needed to convert the sender's unknown state into a known correction on the receiver's half of the entangled pair.

## Evaluation settings

The method is evaluated as an exact theoretical protocol for a single qubit

`|psi> = alpha|0> + beta|1>`, with `|alpha|^2 + |beta|^2 = 1`.

The protocol should satisfy four algebraic checks. First, each measurement outcome occurs with probability `1/4`, independent of `alpha` and `beta`. Second, conditioned on the two-bit outcome, the receiver's qubit differs from `|psi>` only by a known Pauli operation. Third, averaging over the unknown outcome leaves the receiver's local state maximally mixed, so no faster-than-light signaling is possible. Fourth, after the sender's joint measurement, the original input system is no longer an independent copy of `|psi>`.

No empirical performance claims are needed here. Later teleportation experiments are outside this bounded draft; the artifact only needs the protocol, the derivation identity, and the causal/no-cloning checks.

## Code framework

No implementation is required. The formal scaffold is the three-system state space:

`H_1` for the unknown input at the sender, `H_2` for the sender's half of a shared EPR pair, and `H_3` for the receiver's half.

The proof target is a decomposition of

`|psi>_1 |R>_23`

into an orthonormal joint basis on systems `1,2`, where every basis outcome leaves system `3` in a known unitary transform of `|psi>`. The classical message names that basis outcome. The receiver's correction inverts the corresponding transform. The correctness proof is therefore a basis-expansion identity plus a no-signaling trace check, not a simulation or code path.
