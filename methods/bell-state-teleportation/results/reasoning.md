I start from the obstruction rather than from the trick. If Alice has a single system in an unknown qubit state `|psi> = alpha|0> + beta|1>`, I cannot let her learn enough classical data to reconstruct it elsewhere. A measurement that identifies an arbitrary pair of amplitudes is not available on one copy, and a universal copier is ruled out by linearity. If the state reaches Bob exactly, the original at Alice must stop being an independent copy. The problem is therefore not how to copy an unknown state across space, but how to move it while consuming some other resource.

I then look at what an EPR pair can and cannot provide. A shared maximally entangled pair does not contain Alice's unknown state before the protocol begins. Bob's reduced state is maximally mixed, so he has no local signal and no hidden copy of `alpha` and `beta`. That is a useful constraint, not a defect, because it keeps the protocol compatible with relativity. The pair can still be useful if Alice performs an operation that involves both the unknown qubit and her half of the pair. The nonlocal resource is already distributed, and the online part can remain classical.

I need the right joint measurement. A separate measurement of Alice's input in the computational basis would reveal only a basis value, erase the phase relation, and leave Bob with a state that is not a reversible transform of the original superposition. I therefore ask for a joint basis on Alice's two qubits whose outcomes are equally likely and whose conditional effect is to place the unknown coefficients on Bob's qubit, up to a known unitary. The Bell basis has exactly this shape because its vectors pair Alice's two systems maximally rather than treating either one as a standalone carrier of the state.

I choose the shared resource

`|Phi+>_23 = (|00>_23 + |11>_23) / sqrt(2)`

and I use the Bell basis on systems `1,2`:

`|Phi+> = (|00> + |11>) / sqrt(2)`, `|Phi-> = (|00> - |11>) / sqrt(2)`, `|Psi+> = (|01> + |10>) / sqrt(2)`, and `|Psi-> = (|01> - |10>) / sqrt(2)`.

When I expand the three-qubit state in that basis, the whole mechanism appears:

`|psi>_1 |Phi+>_23 = 1/2[ |Phi+>_12 |psi>_3 + |Phi->_12 Z|psi>_3 + |Psi+>_12 X|psi>_3 + |Psi->_12 XZ|psi>_3 ]`.

This identity is the protocol. Alice measures qubits `1,2` in the Bell basis. The measurement destroys the original free input state because qubits `1,2` become one of the Bell states. At the same time, Bob's qubit becomes one of four possible states: `|psi>`, `Z|psi>`, `X|psi>`, or `XZ|psi>`. The outcome probabilities are all `1/4`, so Alice's two-bit result carries no readable description of `alpha` and `beta`; it only tells Bob which inverse Pauli correction is needed.

I now see why exactly two classical bits matter. There are four Bell outcomes, and Bob cannot choose the correction without knowing which one occurred. Before the message arrives, his state is the average over all four possibilities:

`(rho + Z rho Z + X rho X + XZ rho ZX) / 4 = I/2`.

This average is independent of `rho = |psi><psi|`, so Bob cannot detect Alice's input or Alice's outcome by local measurement. The classical channel is therefore not decorative. It is the causal key that turns a locally useless conditional state into the intended state, and it prevents the pre-shared entanglement from becoming a faster-than-light communication channel.

The resulting method has the desired tension resolved in the right direction. The unknown state is never copied, since Alice's input is absorbed into a Bell-basis measurement and loses its separate identity. The EPR pair is consumed, since its entanglement is spent to make Bob's qubit the conditional carrier of the unknown coefficients. The two classical bits are ordinary, copyable information, but they contain only the correction label. Once Bob receives them, he applies the matching Pauli inverse and obtains `|psi>` exactly.
