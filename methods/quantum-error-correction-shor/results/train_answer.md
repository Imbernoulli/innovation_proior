I will describe the Shor nine-qubit quantum error-correcting code, explain why it solves the problem of protecting an unknown qubit, and then verify the construction with a compact Python simulation. The canonical name I use is Shor's code.

The first difficulty in quantum memory is that the information we want to protect is not classical. A logical qubit is an unknown superposition alpha |0> + beta |1>, and both amplitudes matter, including their relative phase. If I tried to protect it the classical way by making copies and voting, quantum mechanics would stop me. A copying device that sends |0> to |00> and |1> to |11> sends alpha |0> + beta |1> to alpha |00> + beta |11>, which is not two independent copies of the unknown state. Worse, comparing copies would require measurements that collapse the superposition and reveal the very information I am trying to preserve. So redundancy must be achieved without cloning.

The second difficulty is that quantum errors are continuous. A qubit can suffer an arbitrary rotation, not just a bit flip. Any single-qubit noise operator can be expanded in the Pauli basis as a linear combination of the identity I, the bit-flip X, the phase-flip Z, and the combined bit-and-phase flip Y. If a code can diagnose and reverse each of the discrete Pauli faults coherently, then linearity of quantum mechanics automatically extends the recovery to arbitrary single-qubit noise under the single-fault assumption. This is why Pauli-stabilizer thinking is so powerful.

The third difficulty is that the diagnostic itself must not look at the encoded information. I cannot measure whether the stored state is closer to |0_L> or |1_L>, because that would project the unknown superposition onto one of those basis states. Instead I must measure only properties that reveal what disturbance occurred, leaving the logical amplitudes untouched. These are called syndrome measurements.

Shor's code builds the protection in two layers, each borrowed from the classical repetition idea but adapted to the quantum setting. The first layer protects against bit flips. I encode |0> as |000> and |1> as |111>. A single bit flip produces a state where one qubit disagrees with the other two. Measuring the parity operators Z1 Z2 and Z2 Z3 inside the block tells me which qubit flipped without revealing whether the block was encoding |0> or |1>. If Z1 Z2 = +1 and Z2 Z3 = -1, the third qubit is the odd one out, so I apply X to qubit three. This is the same majority-vote logic as the classical repetition code, but the parity checks are implemented as quantum measurements whose outcomes do not distinguish the logical basis.

The second layer protects against phase flips. Because the Hadamard transform exchanges bit flips and phase flips, I use the conjugate repetition code: |0> is encoded as |+++> and |1> as three copies of the minus state, where |+> = (|0> + |1>)/sqrt(2) and |-> = (|0> - |1>)/sqrt(2). A phase flip changes |+> to |-> and vice versa, and parity checks in this basis reveal which qubit suffered the phase error. This code protects phase information but is vulnerable to bit flips, exactly the opposite of the first layer.

Shor's construction concatenates these two layers. First I encode the logical qubit into three qubits using the phase-flip repetition code. Then I encode each of those three qubits into three more qubits using the bit-flip repetition code. The result uses nine physical qubits. Explicitly, the logical basis states are

|0_L> = (1/sqrt(8)) (|000> + |111>) tensor (|000> + |111>) tensor (|000> + |111>),
|1_L> = (1/sqrt(8)) (|000> - |111>) tensor (|000> - |111>) tensor (|000> - |111>).

An arbitrary logical state is alpha |0_L> + beta |1_L>. Notice that the amplitudes alpha and beta are not copied into nine independent qubits. The nine physical qubits form a highly entangled state, and the two-dimensional subspace spanned by |0_L> and |1_L> carries the protected information. This is redundancy by entanglement, not cloning.

The syndrome measurements are organized into two groups. Inside each block of three physical qubits I measure the bit-flip parities Z1 Z2 and Z2 Z3, giving six observables in total for the three blocks. Across the three blocks I measure the phase-flip parities X1 X2 X3 X4 X5 X6 and X4 X5 X6 X7 X8 X9. The bit-flip syndromes locate the physical qubit, if any, that experienced an X error. The phase-flip syndromes locate the block, if any, that experienced a Z error. Because the phase code is repetition across blocks, I only need to know which of the three outer blocks was flipped, and then I apply Z to any one qubit inside that block. If both a bit flip and a phase flip occurred, the two syndromes appear together and I apply the corresponding X and Z, which corrects a Y error up to a global phase.

Why does measuring these syndromes not destroy the logical state? Each syndrome operator commutes with the logical Pauli operators that act on the encoded qubit, and each stabilizes the code subspace. Therefore the measurement outcome depends only on the error, not on alpha and beta. After the syndrome is obtained, the recovery operator is a tensor product of Pauli operators conditioned on the syndrome. Because the same syndrome corresponds to matching error components on |0_L> and |1_L>, the relative amplitudes alpha and beta remain intact. The environment has been disentangled from the logical information.

The code is not a complete engineering solution, and it makes important assumptions. It relies on the single-fault model: at most one of the nine physical qubits suffers an error between rounds of correction. It also assumes that the syndrome measurement and recovery operations themselves are perfect, which is not true in practice and motivates fault-tolerant gadgets. Two-qubit correlated errors can defeat the code, and the nine-qubit overhead is large compared with later codes such as the Steane or surface codes. Nevertheless, Shor's code is decisive as a proof of principle: it shows that quantum information can be protected without being copied or read.

The Python program below simulates the essential logic using density-matrix-free state vectors. It defines the nine-qubit logical states, applies a random single-qubit Pauli error, measures the syndromes classically from the corrupted state, and applies the inferred recovery. The final fidelity with the original encoded state is printed. The simulation is small enough to run on a laptop but captures the same syndrome table used by the actual code.

```python
import numpy as np

ket0 = np.array([1.0, 0.0])
ket1 = np.array([0.0, 1.0])

I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
Y = 1j * X @ Z

H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)

def tensor(*ops):
    M = ops[0]
    for op in ops[1:]:
        M = np.kron(M, op)
    return M

def bit_state(bits):
    v = ket0 if bits[0] == 0 else ket1
    for b in bits[1:]:
        v = np.kron(v, ket0 if b == 0 else ket1)
    return v

# Build the repetition cat state |000> + |111> and its phase-flipped cousin.
cat = (bit_state([0, 0, 0]) + bit_state([1, 1, 1])) / np.sqrt(2)
cat_minus = (bit_state([0, 0, 0]) - bit_state([1, 1, 1])) / np.sqrt(2)

zero_L = np.kron(np.kron(cat, cat), cat)
one_L = np.kron(np.kron(cat_minus, cat_minus), cat_minus)

alpha = 0.6 + 0.1j
beta = np.sqrt(1 - abs(alpha)**2)
psi_L = alpha * zero_L + beta * one_L

# Apply a random single-qubit Pauli error.
np.random.seed(0)
q = np.random.randint(0, 9)
pauli_list = [X, Z, Y]
P = pauli_list[np.random.randint(0, 3)]
ops = [I] * 9
ops[q] = P
error_op = tensor(*ops)
corrupted = error_op @ psi_L

# Syndrome measurements.
def pauli_string(ops):
    return tensor(*ops)

syndrome_bit = []
for block_start in [0, 3, 6]:
    z12 = pauli_string([I]*block_start + [Z, Z, I] + [I]*(6 - block_start))
    z23 = pauli_string([I]*block_start + [I, Z, Z] + [I]*(6 - block_start))
    syndrome_bit.append(int(np.round(np.vdot(corrupted, z12 @ corrupted).real)))
    syndrome_bit.append(int(np.round(np.vdot(corrupted, z23 @ corrupted).real)))

x_left = pauli_string([X, X, X, X, X, X, I, I, I])
x_right = pauli_string([I, I, I, X, X, X, X, X, X])
phase_syndrome = [
    int(np.round(np.vdot(corrupted, x_left @ corrupted).real)),
    int(np.round(np.vdot(corrupted, x_right @ corrupted).real))
]

# Decode syndromes to recover.
def bit_flip_qubit(syndrome_bit):
    for b in range(3):
        s1, s2 = syndrome_bit[2*b], syndrome_bit[2*b+1]
        if s1 == -1 and s2 == -1:
            return 3*b
        if s1 == -1 and s2 == 1:
            return 3*b + 1
        if s1 == 1 and s2 == -1:
            return 3*b + 2
    return None

bit_q = bit_flip_qubit(syndrome_bit)
if phase_syndrome == [1, 1]:
    phase_block = None
elif phase_syndrome == [-1, 1]:
    phase_block = 0
elif phase_syndrome == [-1, -1]:
    phase_block = 1
else:
    phase_block = 2

recover_ops = [I] * 9
if bit_q is not None:
    recover_ops[bit_q] = X
if phase_block is not None:
    recover_ops[3*phase_block] = Z
recovery = tensor(*recover_ops)
recovered = recovery @ corrupted

fidelity = abs(np.vdot(psi_L, recovered))**2
print(f"Random error on qubit {q} with Pauli {'XYZ'[pauli_list.index(P)]}")
print(f"Detected bit-flip qubit: {bit_q}, detected phase-flip block: {phase_block}")
print(f"Recovery fidelity: {fidelity:.6f}")
```
