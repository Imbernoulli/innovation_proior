I will present the No-Cloning Theorem as a fundamental constraint on quantum information processing and then give a short numerical illustration that makes the linearity contradiction concrete.

The canonical name for this result is the No-Cloning Theorem. It states that no fixed quantum operation can perfectly copy an arbitrary unknown pure quantum state. More precisely, there is no unitary operator U such that, for every normalized pure state |ψ⟩ and a fixed blank state |b⟩, we have U(|ψ⟩ ⊗ |b⟩) = |ψ⟩ ⊗ |ψ⟩. This is not a statement that individual states can never be duplicated; known orthogonal basis states can be copied by controlled operations. The theorem rules out a universal copier that would work for all possible input states at once.

The intuition behind the theorem comes from the tension between two features of quantum mechanics: linearity and inner-product preservation. Quantum evolution of a closed system is described by a unitary operator, which is linear and preserves the inner product between state vectors. A cloning map, however, would have to take a state |ψ⟩ and produce two copies of it, |ψ⟩ ⊗ |ψ⟩. This map is not linear in |ψ⟩ because the amplitudes get multiplied with themselves. Therefore it cannot be implemented by any linear unitary evolution.

To see the contradiction through inner products, suppose a unitary U could clone two different states |ψ⟩ and |φ⟩. Then we would have U(|ψ⟩ ⊗ |b⟩) = |ψ⟩ ⊗ |ψ⟩ and U(|φ⟩ ⊗ |b⟩) = |φ⟩ ⊗ |φ⟩. Because U is unitary, the inner product of the inputs must equal the inner product of the outputs. The input inner product is ⟨ψ|φ⟩ · ⟨b|b⟩ = ⟨ψ|φ⟩. The output inner product is ⟨ψ|φ⟩². So we would need ⟨ψ|φ⟩ = ⟨ψ|φ⟩². The only solutions are ⟨ψ|φ⟩ = 0, meaning the states are orthogonal, or ⟨ψ|φ⟩ = 1, meaning the states are identical. Any two distinct non-orthogonal states would give a contradiction. Hence a universal copier cannot exist.

The same conclusion follows directly from linearity. Imagine a device that successfully clones the computational basis states: |0⟩|b⟩ becomes |0⟩|0⟩ and |1⟩|b⟩ becomes |1⟩|1⟩. By linearity, its action on a superposition a|0⟩ + b|1⟩ must be a|00⟩ + b|11⟩. But two genuine copies of the superposition would be (a|0⟩ + b|1⟩) ⊗ (a|0⟩ + b|1⟩) = a²|00⟩ + ab|01⟩ + ab|10⟩ + b²|11⟩. These two expressions are generally different; the cross-terms |01⟩ and |10⟩ are missing from the linear output. The cloning operation is quadratic in the amplitudes, while unitary evolution is strictly linear.

This distinction matters because classical information can be copied freely. A classical bit is a label drawn from a set of perfectly distinguishable states, which can be read out and rewritten without changing the original. In quantum mechanics, only orthogonal states are perfectly distinguishable, and only orthogonal states can be cloned by a fixed procedure. A general quantum state carries information in its complex amplitudes and in the non-orthogonal overlaps with other states. That continuous information cannot be extracted non-destructively and broadcast to a second system.

The No-Cloning Theorem is therefore not a technological limitation that better engineering might overcome. It is a structural consequence of how quantum states are described and how they evolve. It is also why measurement cannot be used as a workaround: measuring an unknown non-orthogonal state yields at best partial information and disturbs the original, so measure-and-prepare is not a universal copying strategy.

One might wonder whether extra auxiliary states or environmental degrees of freedom could rescue cloning. Suppose the output includes an apparatus state |A_ψ⟩, so the copier produces |ψ⟩|ψ⟩|A_ψ⟩. The same inner-product argument gives ⟨ψ|φ⟩ = ⟨ψ|φ⟩² ⟨A_ψ|A_φ⟩. When 0 < |⟨ψ|φ⟩| < 1, the right-hand side has magnitude at most |⟨ψ|φ⟩|², which is strictly smaller than |⟨ψ|φ⟩|. The contradiction remains, showing that hiding information in an ancilla or environment does not help.

The practical importance of the theorem extends to quantum cryptography and quantum communication. Because an eavesdropper cannot clone an unknown quantum state, protocols such as quantum key distribution can detect interception by the disturbance it causes. The theorem also shapes the design of quantum error correction, which must protect quantum information without ever making unauthorized copies of an unknown logical state.

To make the central linearity check concrete, here is a small Python script that compares what a linear cloning device would produce for a superposition with what a true two-copy state would be.

```python
import numpy as np

# Computational basis states for one qubit
ket0 = np.array([1.0, 0.0])
ket1 = np.array([0.0, 1.0])

# Two-qubit basis states
ket00 = np.kron(ket0, ket0)
ket01 = np.kron(ket0, ket1)
ket10 = np.kron(ket1, ket0)
ket11 = np.kron(ket1, ket1)

# Choose a non-trivial superposition, e.g. a = sqrt(1/3), b = sqrt(2/3)
a = np.sqrt(1.0 / 3.0)
b = np.sqrt(2.0 / 3.0)
psi = a * ket0 + b * ket1

# What a linear device would output if it clones |0> -> |00> and |1> -> |11>
linear_output = a * ket00 + b * ket11

# What a true two-copy state of psi would be
true_copy = np.kron(psi, psi)

# Compare the two vectors
overlap = np.vdot(linear_output, true_copy)
linear_norm = np.linalg.norm(linear_output)
true_norm = np.linalg.norm(true_copy)

print("Amplitudes of linear output:", linear_output)
print("Amplitudes of true two-copy state:", true_copy)
print("Overlap between the two states:", overlap)
print("Norm of linear output:", linear_norm)
print("Norm of true copy:", true_norm)
print("Are they equal up to numerical precision?", np.allclose(linear_output, true_copy))
```

Running this script shows that the linear output and the true two-copy state are not the same. The linear output retains the original amplitudes a and b on the |00⟩ and |11⟩ components, while the true copy spreads the squared amplitudes a², b² and cross-amplitudes ab across all four basis states. For a = √(1/3) and b = √(2/3), the linear output is √(1/3)|00⟩ + √(2/3)|11⟩, whereas the genuine two-copy state is (1/3)|00⟩ + (√2/3)|01⟩ + (√2/3)|10⟩ + (2/3)|11⟩. The overlap between them is (a³ + b³), which is strictly less than one for any non-trivial superposition. This numerical check mirrors the general proof: a unitary that clones the basis states is forced by linearity to produce a state different from the desired universal clone.

In summary, the No-Cloning Theorem tells us that arbitrary unknown quantum states cannot be duplicated by any fixed physical process. The obstacle is built into the linear, inner-product-preserving character of quantum mechanics. Orthogonal or identical states remain copyable, but the continuous set of all pure quantum states cannot be universally cloned. This result marks a deep departure from classical information theory and underpins many of the security and communication primitives unique to quantum information science.
