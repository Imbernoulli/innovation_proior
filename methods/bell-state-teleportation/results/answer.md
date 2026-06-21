Bell-State Teleportation.

Goal: transfer an unknown qubit

`|psi>_1 = alpha|0>_1 + beta|1>_1`

from Alice to Bob without copying it, using one shared maximally entangled pair and two classical bits.

## Resources

Alice holds the unknown qubit `1` and her half `2` of an EPR pair. Bob holds qubit `3`.

Use the shared pair

`|Phi+>_23 = (|00>_23 + |11>_23) / sqrt(2)`.

Alice and Bob also have a classical channel from Alice to Bob. No quantum system is sent during the transfer step.

## Bell Basis

Alice measures qubits `1,2` in the Bell basis:

`|Phi+> = (|00> + |11>) / sqrt(2)`

`|Phi-> = (|00> - |11>) / sqrt(2)`

`|Psi+> = (|01> + |10>) / sqrt(2)`

`|Psi-> = (|01> - |10>) / sqrt(2)`

The key expansion is

`|psi>_1 |Phi+>_23 = 1/2[ |Phi+>_12 |psi>_3 + |Phi->_12 Z|psi>_3 + |Psi+>_12 X|psi>_3 + |Psi->_12 XZ|psi>_3 ]`.

This is an exact identity for every `alpha,beta`.

## Protocol

1. Entanglement distribution: before teleportation, Alice and Bob share `|Phi+>_23`. The pair contains no information about the later unknown state.
2. Joint measurement: Alice performs a Bell-basis measurement on the unknown qubit `1` and her entangled qubit `2`.
3. Classical communication: Alice sends the two-bit label of her Bell outcome to Bob.
4. Correction: Bob applies the Pauli correction determined by Alice's two-bit label.

## Correction Table

| Alice outcome | Bob's conditional state | Alice sends | Bob applies |
| --- | --- | --- | --- |
| `|Phi+>_12` | `|psi>_3` | `00` | `I` |
| `|Phi->_12` | `Z|psi>_3` | `01` | `Z` |
| `|Psi+>_12` | `X|psi>_3` | `10` | `X` |
| `|Psi->_12` | `XZ|psi>_3` | `11` | `ZX` |

The last row is insensitive to global phase conventions. Applying `ZX` to `XZ|psi>` returns `|psi>` up to the physically irrelevant phase of Pauli products.

## Correctness Checks

Each Bell outcome has probability `1/4`, because every term in the expansion has amplitude factor `1/2` and each conditional Bob state is normalized. The outcome distribution is independent of `alpha` and `beta`.

After Alice's Bell measurement, qubits `1,2` are projected into a Bell state. The original state is not left behind at Alice. The protocol therefore transfers the state rather than cloning it.

Before Bob receives Alice's two-bit message, his local state is the average over the four possible conditional states:

`(rho + Z rho Z + X rho X + XZ rho ZX) / 4 = I/2`,

where `rho = |psi><psi|`. This state is independent of the unknown input, so the protocol cannot transmit information faster than the classical message.

The shared EPR pair is consumed. The final usable state is Bob's corrected qubit `|psi>_3`; the entanglement resource and Alice's original input have both been spent.

## Relation To Bennett's Singlet Convention

Bennett et al. present the original protocol using a singlet EPR pair and a Bell measurement on Alice's two particles. That convention changes signs and associates the four outcomes with rotations about different axes, but it is locally equivalent to the `|Phi+>` convention above. In both forms, the essential structure is the same: Bell-basis measurement, two classical bits, Pauli correction, destroyed sender state, reconstructed receiver state.
