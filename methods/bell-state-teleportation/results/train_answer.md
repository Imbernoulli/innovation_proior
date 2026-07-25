The problem is to move an unknown qubit state from a sender to a receiver without physically sending the particle and without creating a copy. The central obstruction is the no-cloning theorem: any linear quantum operation that would produce two independent copies of an arbitrary state is impossible, because the linear image of a superposition is not the tensor square of that superposition. At the same time, a single copy of an unknown qubit cannot be measured to reveal its amplitudes without disturbing them, so a classical measure-and-reprepare strategy works only for a fixed orthogonal set of inputs. The task is therefore not to duplicate or describe the state, but to relocate it by consuming some pre-shared resource.

Existing ideas fall short in different ways. Direct quantum transmission simply sends the carrier, which does not explain how relocation can happen when no quantum channel is available at transfer time. A universal copier is forbidden by linearity and would also enable superluminal signaling in entanglement settings. A shared EPR pair by itself does not place the unknown state at the receiver, because the receiver's reduced state is maximally mixed and independent of the sender's input. Separate local measurements on the two halves of the pair only exchange ordinary correlation data; they do not perform the specific joint operation that transfers the unknown coefficients onto the receiver's qubit. What is missing is a measurement basis that treats the sender's unknown qubit and her half of the entangled pair as a single system, so that the measurement outcome tells the receiver which local correction makes his qubit match the original.

The method that resolves this is Bell-State Teleportation. Alice holds the unknown qubit and one qubit of a maximally entangled pair shared with Bob; Bob holds the other qubit. Alice measures her two qubits in the Bell basis, whose four states maximally entangle the two systems. Because the total three-qubit state can be rewritten as a uniform superposition over Bell outcomes, each outcome places Bob's qubit in the original state up to a known Pauli operator. Alice sends Bob the two-bit label of her Bell outcome, and Bob applies the matching Pauli inverse. The original qubit is destroyed in Alice's joint measurement, the shared entanglement is consumed, and Bob's qubit becomes the original state exactly.

The algebraic identity that makes the protocol work is the expansion of the initial state |psi>_1 |Phi+>_23 in the Bell basis of systems 1 and 2. Writing |psi> = alpha|0> + beta|1> and |Phi+> = (|00> + |11>)/sqrt(2), one obtains |psi>_1 |Phi+>_23 = (1/2)[ |Phi+>_12 |psi>_3 + |Phi->_12 Z|psi>_3 + |Psi+>_12 X|psi>_3 + |Psi->_12 XZ|psi>_3 ]. Each term has amplitude 1/2, so every outcome occurs with probability 1/4 independent of alpha and beta. Bob's conditional state is therefore one of |psi>, Z|psi>, X|psi>, or XZ|psi>, and the correction is respectively the identity I, a Pauli-Z, a Pauli-X, or ZX applied to his qubit. Before receiving the two classical bits, Bob's local state averages to the maximally mixed state I/2, so no information about Alice's input is available locally and no faster-than-light signaling occurs. The two classical bits are ordinary and copyable, but they contain only the correction label, not a description of the amplitudes.

The protocol, stated in full, is this. Alice and Bob begin by sharing $|\Phi^+\rangle_{23} = (|00\rangle_{23} + |11\rangle_{23})/\sqrt{2}$, distributed before the unknown state $|\psi\rangle_1$ ever exists at Alice's side. Alice performs a single projective measurement on qubits $1,2$ jointly, in the Bell basis $\{|\Phi^+\rangle,|\Phi^-\rangle,|\Psi^+\rangle,|\Psi^-\rangle\}$, and obtains one of four outcomes, each with probability exactly $1/4$ regardless of $\alpha,\beta$. She sends the two-bit label of that outcome to Bob over the classical channel, and Bob applies the corresponding correction to his half of the pair:

$$
U(\text{outcome}) =
\begin{cases}
I & \text{outcome } |\Phi^+\rangle_{12},\ \text{bits } 00\\[2pt]
Z & \text{outcome } |\Phi^-\rangle_{12},\ \text{bits } 01\\[2pt]
X & \text{outcome } |\Psi^+\rangle_{12},\ \text{bits } 10\\[2pt]
ZX & \text{outcome } |\Psi^-\rangle_{12},\ \text{bits } 11
\end{cases}
$$

After this one correction, Bob's qubit is $|\psi\rangle_3$ exactly, for every $\alpha,\beta$ — there is no residual error and no free parameter left to tune. The certificate that this is genuine relocation and not a hidden signaling channel is the state Bob holds before the two bits arrive, averaged over the four equally likely outcomes:

$$
\frac{1}{4}\Big(\rho + Z\rho Z + X\rho X + XZ\,\rho\,ZX\Big) = \frac{I}{2}, \qquad \rho = |\psi\rangle\langle\psi|,
$$

which equals the maximally mixed state for every input, so nothing about $\alpha,\beta$ is available to Bob until the classical bits complete the correction. Alice's original qubit is simultaneously gone, projected into whichever Bell state her measurement selected: the unknown state has been relocated, not duplicated, and the no-cloning theorem is never in tension with the protocol because no full description of $\alpha,\beta$ is ever carried by anything, quantum or classical — only a two-bit correction label is sent, and the correction itself is what completes the transfer.
