The problem is to move an unknown qubit state from a sender to a receiver without physically sending the particle and without creating a copy. The central obstruction is the no-cloning theorem: any linear quantum operation that would produce two independent copies of an arbitrary state is impossible, because the linear image of a superposition is not the tensor square of that superposition. At the same time, a single copy of an unknown qubit cannot be measured to reveal its amplitudes without disturbing them, so a classical measure-and-reprepare strategy works only for a fixed orthogonal set of inputs. The task is therefore not to duplicate or describe the state, but to relocate it by consuming some pre-shared resource.

Existing ideas fall short in different ways. Direct quantum transmission simply sends the carrier, which does not explain how relocation can happen when no quantum channel is available at transfer time. A universal copier is forbidden by linearity and would also enable superluminal signaling in entanglement settings. A shared EPR pair by itself does not place the unknown state at the receiver, because the receiver's reduced state is maximally mixed and independent of the sender's input. Separate local measurements on the two halves of the pair only exchange ordinary correlation data; they do not perform the specific joint operation that transfers the unknown coefficients onto the receiver's qubit. What is missing is a measurement basis that treats the sender's unknown qubit and her half of the entangled pair as a single system, so that the measurement outcome tells the receiver which local correction makes his qubit match the original.

The method that resolves this is Bell-State Teleportation. Alice holds the unknown qubit and one qubit of a maximally entangled pair shared with Bob; Bob holds the other qubit. Alice measures her two qubits in the Bell basis, whose four states maximally entangle the two systems. Because the total three-qubit state can be rewritten as a uniform superposition over Bell outcomes, each outcome places Bob's qubit in the original state up to a known Pauli operator. Alice sends Bob the two-bit label of her Bell outcome, and Bob applies the matching Pauli inverse. The original qubit is destroyed in Alice's joint measurement, the shared entanglement is consumed, and Bob's qubit becomes the original state exactly.

The algebraic identity that makes the protocol work is the expansion of the initial state |psi>_1 |Phi+>_23 in the Bell basis of systems 1 and 2. Writing |psi> = alpha|0> + beta|1> and |Phi+> = (|00> + |11>)/sqrt(2), one obtains |psi>_1 |Phi+>_23 = (1/2)[ |Phi+>_12 |psi>_3 + |Phi->_12 Z|psi>_3 + |Psi+>_12 X|psi>_3 + |Psi->_12 XZ|psi>_3 ]. Each term has amplitude 1/2, so every outcome occurs with probability 1/4 independent of alpha and beta. Bob's conditional state is therefore one of |psi>, Z|psi>, X|psi>, or XZ|psi>, and the correction is respectively the identity I, a Pauli-Z, a Pauli-X, or ZX applied to his qubit. Before receiving the two classical bits, Bob's local state averages to the maximally mixed state I/2, so no information about Alice's input is available locally and no faster-than-light signaling occurs. The two classical bits are ordinary and copyable, but they contain only the correction label, not a description of the amplitudes.

```python
import numpy as np

# Pauli matrices
I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)

# Bell basis on two qubits (order: Phi+, Phi-, Psi+, Psi-)
bell = [
    np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2),   # |Phi+>
    np.array([1, 0, 0, -1], dtype=complex) / np.sqrt(2),  # |Phi->
    np.array([0, 1, 1, 0], dtype=complex) / np.sqrt(2),   # |Psi+>
    np.array([0, 1, -1, 0], dtype=complex) / np.sqrt(2),  # |Psi->
]

# Correction operators Bob applies for each Bell outcome
corrections = [I, Z, X, Z @ X]

def ket(*bits):
    """Return computational-basis state |b1 b2 ...>."""
    state = np.zeros(2 ** len(bits), dtype=complex)
    idx = sum(b << (len(bits) - 1 - i) for i, b in enumerate(bits))
    state[idx] = 1.0
    return state

def teleport(psi):
    """Simulate one shot of Bell-state teleportation of single-qubit state psi."""
    # Shared EPR pair |Phi+> between Alice (qubit 2) and Bob (qubit 3)
    phi_plus = (ket(0, 0) + ket(1, 1)) / np.sqrt(2)

    # Total state: |psi>_1 * |Phi+>_23, reshaped as (input, Alice, Bob)
    psi_total = np.kron(psi, phi_plus)
    psi_tensor = psi_total.reshape(2, 2, 2)

    # Alice measures qubits 1 and 2 in the Bell basis.
    # For each outcome, compute Bob's unnormalized conditional state.
    bob_states = []
    probs = []
    for b in bell:
        projected = np.tensordot(b.conj().reshape(2, 2), psi_tensor, axes=([0, 1], [0, 1]))
        norm = np.linalg.norm(projected)
        probs.append(norm ** 2)
        bob_states.append(projected / norm if norm > 0 else projected)

    probs = np.array(probs, dtype=float)
    probs /= probs.sum()
    outcome = np.random.choice(4, p=probs)

    # Alice sends the two-bit outcome to Bob; Bob applies the matching correction
    bob_corrected = corrections[outcome] @ bob_states[outcome]

    # Verify Bob's state equals the original up to a global phase
    fidelity = abs(np.vdot(psi, bob_corrected)) ** 2
    return outcome, fidelity

# Example: teleport an arbitrary pure state
if __name__ == "__main__":
    theta = 1.2
    phi = 0.7
    psi = np.cos(theta / 2) * ket(0) + np.exp(1j * phi) * np.sin(theta / 2) * ket(1)

    outcome, fid = teleport(psi)
    print(f"Bell outcome index: {outcome}")
    print(f"Teleportation fidelity: {fid:.10f}")

    # Check the four conditional Bob states against the expansion identity
    phi_plus = (ket(0, 0) + ket(1, 1)) / np.sqrt(2)
    psi_tensor = np.kron(psi, phi_plus).reshape(2, 2, 2)
    print("\nConditional Bob states before correction:")
    for o in range(4):
        cond = np.tensordot(bell[o].conj().reshape(2, 2), psi_tensor, axes=([0, 1], [0, 1]))
        print(f"  outcome {o}: {cond}")
```
