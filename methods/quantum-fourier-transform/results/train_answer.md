The discrete Fourier transform is the workhorse behind spectral analysis, signal processing, and countless solvers for linear systems, yet its conventional formulation treats the input as a classical list of N complex numbers and returns another classical list. A quantum computer, however, stores information as amplitudes attached to computational basis states, and the natural question is whether a Fourier transform can be performed directly on those amplitudes without ever materializing the full vector. The Quantum Fourier Transform, or QFT, is the operation that answers this question. It is a unitary transformation that maps an n-qubit state with amplitudes x_k into a superposition whose amplitudes are proportional to the discrete Fourier coefficients of the input sequence. Because the result remains a quantum state, the QFT exposes its answer only through the amplitudes of a superposition, not as an explicit output vector, and extracting every classical coefficient would still require an exponential number of measurements. Its power lies elsewhere: when the input has a compact description or when only partial information about the spectrum is needed, the QFT can be implemented with far fewer resources than a classical FFT.

Formally, the QFT acts on the N = 2^n basis states |k> with k = 0, ..., N-1 by sending each |k> to a uniform superposition over all basis states weighted by roots of unity. Concretely, the action is |k> -> (1/sqrt(N)) sum_{j=0}^{N-1} e^{2 pi i j k / N} |j>. By linearity, an arbitrary input state sum_k x_k |k> is mapped to sum_j y_j |j> where y_j = (1/sqrt(N)) sum_k e^{2 pi i j k / N} x_k. Up to the conventional normalization convention, this is exactly the discrete Fourier transform applied to the amplitude vector. The factor of 1/sqrt(N) preserves the l2 norm, so the transformation is unitary and therefore physically realizable as a quantum circuit. Reversing the sign in the exponent gives the inverse QFT, which is equally useful and is implemented by running the same circuit backwards.

The reason the QFT is efficient on a quantum computer is that it factorizes beautifully into one- and two-qubit gates. The key identity is the product representation of the output state for a basis input |k>. Writing k in binary as k_1 k_2 ... k_n, the QFT produces a tensor product of n single-qubit states, each of the form (1/sqrt(2))(|0> + e^{2 pi i 0.k_l k_{l+1} ... k_n}|1>), where the notation 0.k_l k_{l+1} ... k_n means the binary fraction k_l / 2 + k_{l+1} / 4 + ... + k_n / 2^{n-l+1}. This factorization immediately suggests a circuit. Start with the most significant qubit k_1 and apply a Hadamard gate, which creates (1/sqrt(2))(|0> + e^{2 pi i 0.k_1}|1>). Then apply controlled phase rotations conditioned on the remaining qubits k_2, k_3, ..., k_n to add the bits 0.k_1 k_2, 0.k_1 k_2 k_3, and so on into the relative phase. Repeat this process for each qubit, and finish by reversing the qubit order with swap gates so that the least significant output appears on the least significant wire. The complete circuit uses n Hadamard gates and n(n-1)/2 controlled rotations, for a total gate count of O(n^2), which is exponential in the number of qubits but only quadratic in the number of bits needed to index the input. This is the source of the exponential speedup over the classical FFT when the input is given implicitly and the output is used coherently.

The controlled rotations are conventionally named R_m and add a phase of e^{2 pi i / 2^m} when the control qubit is |1>. In the standard QFT circuit, R_2 adds a phase of pi/2, R_3 adds pi/4, and so on. Because the phase becomes exponentially small for large m, approximate versions of the QFT are often used in practice that drop the smallest rotations while still giving accurate results for many applications. This approximate QFT further reduces the gate count and is one of the reasons the transform remains feasible on near-term devices.

The importance of the QFT extends well beyond being a fast Fourier transform for amplitudes. It is the core subroutine in quantum phase estimation, where it converts an eigenvalue encoded in the phase of a controlled unitary into a computational basis state that can be measured. Quantum phase estimation in turn underpins Shor's factoring algorithm, quantum algorithms for solving linear systems, quantum simulation, and many other quantum speedups. In each of these settings, the QFT is applied not to read out a spectrum but to manipulate interference so that the desired answer appears with high probability upon measurement.

The QFT also illustrates a broader lesson about quantum algorithms. It does not provide a generic way to compute all Fourier coefficients classically faster than the FFT, because reading out all N coefficients would require N measurements and would destroy the quantum speedup. Instead, it provides a way to access global, periodic, or structured properties of the amplitude vector in a way that would be hard to simulate classically. The canonical name for this transformation is the Quantum Fourier Transform, and its circuit realization is one of the most elegant constructions in quantum computing.

```python
import numpy as np

def qft_matrix(n):
    """Return the exact n-qubit Quantum Fourier Transform matrix U[j, k]."""
    N = 2 ** n
    k = np.arange(N).reshape(-1, 1)
    j = np.arange(N).reshape(1, -1)
    return np.exp(2j * np.pi * k * j / N) / np.sqrt(N)

def inverse_qft_matrix(n):
    """The inverse QFT is the conjugate transpose of the QFT."""
    return qft_matrix(n).conj().T

def verify_qft(n=4):
    """Verify unitarity, inverse relationship, FFT consistency, and period detection."""
    N = 2 ** n
    U = qft_matrix(n)
    U_inv = inverse_qft_matrix(n)

    # 1. Unitarity: U U^dagger = U^dagger U = I.
    assert np.allclose(U @ U.conj().T, np.eye(N)), "QFT is not unitary"

    # 2. Inverse QFT genuinely inverts the transform.
    rng = np.random.default_rng(0)
    x = rng.normal(size=N) + 1j * rng.normal(size=N)
    x /= np.linalg.norm(x)
    assert np.allclose(U_inv @ (U @ x), x), "Inverse QFT does not invert QFT"

    # 3. QFT agrees with numpy's inverse FFT up to normalization.
    y_qft = U @ x
    y_fft = np.fft.ifft(x) * np.sqrt(N)
    assert np.allclose(y_qft, y_fft), "QFT does not match normalized inverse FFT"

    # 4. Period detection: a periodic amplitude vector produces concentrated peaks.
    period = 3
    amplitudes = np.zeros(N, dtype=complex)
    amplitudes[::period] = 1.0
    amplitudes /= np.linalg.norm(amplitudes)
    spectrum = np.abs(U @ amplitudes) ** 2
    peak_positions = np.where(spectrum > 0.1)[0]
    print(f"n={n}, N={N}, period={period}, spectral peaks at j={peak_positions}")
    return True

if __name__ == "__main__":
    print("QFT verification passed:", verify_qft(4))
```
