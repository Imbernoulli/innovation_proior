# Shor Quantum Error-Correcting Code

Encode one qubit `|psi> = alpha |0> + beta |1>` as

```text
|0_L> = (1/sqrt(8)) (|000> + |111>) (|000> + |111>) (|000> + |111>)
|1_L> = (1/sqrt(8)) (|000> - |111>) (|000> - |111>) (|000> - |111>)
|psi_L> = alpha |0_L> + beta |1_L>
```

This is redundancy by entanglement, not cloning. The amplitudes `alpha` and `beta` are stored in the two-dimensional code subspace.

## Syndrome Measurements

Measure bit-flip syndromes inside each block of three:

```text
Z1 Z2, Z2 Z3, Z4 Z5, Z5 Z6, Z7 Z8, Z8 Z9
```

Measure phase-flip syndromes across the three blocks:

```text
X1 X2 X3 X4 X5 X6
X4 X5 X6 X7 X8 X9
```

The outcomes identify the error sector without distinguishing `|0_L>` from `|1_L>`.

## Recovery

If the `Z Z` checks locate a single bit flip, apply `X` to the indicated qubit. If the block-sign checks locate a phase-flipped block, apply `Z` to any one qubit in that block. If both syndromes appear, apply both corrections, which corrects a `Y` error up to global phase.

Thus the code corrects any single-qubit `X`, `Z`, or `Y` error on the nine physical qubits. Since one-qubit Pauli operators span arbitrary one-qubit errors, the same syndrome-and-recovery procedure corrects arbitrary single-qubit noise in the ideal single-fault model.

## Claim

Quantum error correction works here because the recovery learns what happened to the encoded block, not what unknown state was stored. The logical information remains hidden in the code subspace while local faults are converted into measurable syndromes.
