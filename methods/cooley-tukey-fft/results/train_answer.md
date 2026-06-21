The discrete Fourier transform is the workhorse for extracting frequency content from sampled signals, but its direct definition as a matrix-vector product costs N² complex multiplications. For the million-point records that arise in seismology, spectroscopy, and three-dimensional solid-state calculations, that quadratic cost is the difference between a feasible computation and one that would consume weeks of machine time. The matrix entries are not arbitrary numbers: every one of them is a power of the single primitive root of unity W = e^{-2πi/N}. That redundancy means most of the arithmetic in the direct method is recomputation, and the right way to exploit it is to express a long transform in terms of genuinely shorter transforms rather than merely grouping equal trigonometric values.

Earlier speedups did not change the order of growth. Term-grouping schemes collect coefficients that multiply the same sine or cosine and add them before multiplying, which saves a constant factor but still leaves the count at Θ(N²). Runge's doubling rule and the Danielson–Lanczos identity did split a length-N transform into two length-N/2 transforms, but they were presented as special halving or doubling devices rather than a uniform machine procedure for general composite lengths. What was missing was a single recursive factorization, with explicit twiddle factors and index bookkeeping, that a computer could run for any highly composite N.

The method is the Cooley–Tukey Fast Fourier Transform. It takes the root-of-unity structure seriously and factors the transform size. For the radix-2 case that covers power-of-two lengths, split the input by parity into even-indexed and odd-indexed subsequences. Because W² is the primitive root of unity for length N/2, the even half becomes a length-N/2 DFT E and the odd half becomes another length-N/2 DFT O, with a single cross factor W^j that splices them together. The half-transforms are periodic with period N/2, and because W^{N/2} equals -1, one twiddle multiply W^j·O(j) yields two outputs at once: X(j) = E(j) + W^j·O(j) and X(j+N/2) = E(j) - W^j·O(j). This add-and-subtract pair is the butterfly. Recursing on E and O gives the cost recurrence T(N) = 2T(N/2) + O(N), which solves to T(N) = O(N log N). At a million points the reduction is by a factor of roughly fifty thousand.

The recursion has a clean iterative form. Repeatedly splitting by parity routes each input sample according to its bits read least-significant-first, so the sample that ends up at leaf position p is the input whose index is the bit-reversal of p. If the working array is first permuted into bit-reversed order, the leaves are adjacent one-point transforms and the result can be built bottom-up with log₂ N passes of in-place butterflies. Each pass doubles the transform length, computes the root of unity for that block size once, and propagates a running twiddle through the butterflies by multiplication. This is the standard in-place radix-2 FFT.

The same factorization generalizes beyond radix 2. For N = r₁r₂, group inputs by residue modulo r₁ and split outputs compatibly; the inner stage computes r₁ DFTs of length r₂, the outer stage computes r₂ DFTs of length r₁, and the cross term W^{j₀k₀} is the twiddle that couples them. Recursing to the prime factorization N = p₁p₂⋯pₗ gives cost N(p₁ + p₂ + ⋯ + pₗ). When every prime is 2 this is 2N log₂ N; when N is prime the factorization has no smaller pieces and the method falls back to direct Θ(N²) evaluation. For scientific applications that can tolerate a longer transform, zero-padding to a composite length is the usual escape.

```python
import cmath

def root_of_unity(N, sign=-1):
    # W = exp(sign * 2πi / N); sign=-1 for forward, +1 for inverse
    return cmath.exp(sign * 2j * cmath.pi / N)

def dft_direct(A, sign=-1):
    # Baseline O(N^2) transform; also the trusted small-N oracle.
    N = len(A)
    W = root_of_unity(N, sign)
    return [sum(A[k] * W ** (j * k) for k in range(N)) for j in range(N)]

def _validate_length(n):
    if n < 1 or n & (n - 1):
        raise ValueError("length must be a power of two")

def transform_recursive(A, sign=-1):
    # Recursive radix-2 decimation-in-time transform.
    N = len(A)
    _validate_length(N)
    if N == 1:
        return [A[0]]
    E = transform_recursive(A[0::2], sign)  # even-indexed subsequence
    O = transform_recursive(A[1::2], sign)  # odd-indexed subsequence
    X = [0j] * N
    W = root_of_unity(N, sign)
    for j in range(N // 2):
        t = W ** j * O[j]                # twiddle W^j times O(j)
        X[j] = E[j] + t                  # X(j)   = E(j) + W^j O(j)
        X[j + N // 2] = E[j] - t         # X(j+N/2) = E(j) - W^j O(j)
    return X

def _reorder_for_iteration(A):
    # Route input to leaf order: position p receives input at index bitrev(p).
    A = list(A)
    n = len(A)
    _validate_length(n)
    bits = n.bit_length() - 1
    for p in range(n):
        q = int(format(p, '0{}b'.format(bits))[::-1], 2)
        if q > p:                        # reversal is an involution
            A[p], A[q] = A[q], A[p]
    return A

def transform_iterative(A, sign=-1):
    # In-place radix-2 FFT on a bit-reversed working copy.
    A = _reorder_for_iteration(A)
    n = len(A)
    L = 2
    while L <= n:
        wL = root_of_unity(L, sign)      # root of unity for length-L blocks
        for start in range(0, n, L):
            w = 1 + 0j
            for r in range(L // 2):
                u = A[start + r]
                t = w * A[start + r + L // 2]
                A[start + r] = u + t     # butterfly: E + W^r O
                A[start + r + L // 2] = u - t  # butterfly: E - W^r O
                w *= wL
        L <<= 1
    return A

def inverse_transform(X):
    # Inverse: conjugate root and 1/N normalization.
    n = len(X)
    return [v / n for v in transform_iterative(X, sign=+1)]
```
