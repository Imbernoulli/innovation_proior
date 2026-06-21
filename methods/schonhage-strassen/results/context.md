# Context: fast multiplication of very large integers

## Research question

Given two integers each about n bits long, compute their product exactly. The schoolbook
method costs Θ(n²) bit operations. The question is how far below Θ(n²) one can push the *bit*
complexity of an exact integer product, ideally toward something near-linear, while keeping the
arithmetic exact (no floating-point round-off) so the answer is provably correct to the last bit.

## Background

**Multiplication is a convolution.** Write an integer in base 2^w as a digit sequence,
a = Σ_i a_i (2^w)^i. The product of two such numbers has digits c_k = Σ_{i+j=k} a_i b_j before
carry propagation — exactly the (acyclic) convolution of the two digit sequences. So "multiply two
n-bit numbers" is "convolve two length-N sequences and then propagate carries," and the whole game
becomes: how cheaply can a convolution be computed.

**The convolution theorem and the DFT.** The discrete Fourier transform diagonalizes convolution:
if â = DFT(a) and b̂ = DFT(b), then DFT(a∗b) = â · b̂ pointwise, so a∗b = IDFT(â · b̂). A length-N
DFT done naively is Θ(N²), but the Cooley–Tukey Fast Fourier Transform (Cooley & Tukey, 1965)
computes it in Θ(N log N) operations by recursively splitting even/odd indexed terms, with the
recurrence T(N) = 2T(N/2) + Θ(N). The FFT needs a primitive N-th root of unity ω (ω^N = 1, and
ω^r ≠ 1 for 0 < r < N) and the invertibility of N inside whatever ring the coefficients live in.

**Roots of unity in finite rings.** A root of unity need not be the complex number e^{2πi/N}. In a
finite ring Z/mZ, an element g is a primitive N-th root of unity if g^N ≡ 1 (mod m) and no smaller
power is 1. A transform built on such a g is a *Number-Theoretic Transform* (NTT). The FFT needs only that
ω^N = 1, that ω^r − 1 is invertible for 0 < r < N, and that N is invertible in the ring; any
commutative ring with such an element supports the transform.

**Cyclic vs. negacyclic convolution.** A length-N cyclic convolution computes c mod (x^N − 1); a
*negacyclic* (negative-wrapped) convolution computes c mod (x^N + 1), where the wraparound
coefficient k+N is subtracted from coefficient k rather than added. These are the two natural
length-N convolutions a transform on N points can produce.

**The divide-and-conquer ancestry, as motivating empirical fact.** Splitting an integer into d+1
parts and treating it as a degree-d polynomial is a known, measured way to cut core-multiplication
counts: two d+1-digit numbers need (d+1)² core multiplications by the schoolbook layout, but
clever recombination needs far fewer — the count is what these methods are judged on, and the
schoolbook tableau makes the (d+1)² count concrete (a 4×4 product visibly performs 16 core
multiplications, an 8×8 performs 64).

## Baselines

**Schoolbook / long multiplication.** c_k = Σ_{i+j=k} a_i b_j directly, then carry. Θ(N²) core
multiplications for length-N operands. Exact and simple.

**Karatsuba (Karatsuba & Ofman, 1962).** Split each number into two halves, a = a₁B + a₀,
b = b₁B + b₀. Naively ab = a₁b₁B² + (a₁b₀ + a₀b₁)B + a₀b₀ needs four half-products; Karatsuba's
identity a₁b₀ + a₀b₁ = (a₁ + a₀)(b₁ + b₀) − a₁b₁ − a₀b₀ trades one multiplication for additions,
leaving three half-products. Recursing gives T(n) = 3T(n/2) + Θ(n) = Θ(n^{log₂3}) ≈ Θ(n^{1.585}).

**Toom–Cook (Toom, 1963; Cook, 1966).** Generalize: split into k+1 parts, regard each operand as a
degree-k polynomial, *evaluate* both at 2k+1 points, multiply pointwise (2k+1 smaller products),
and *interpolate* the product polynomial. Toom-3 gives 5 sub-products of n/3-size operands,
T(n) = 5T(n/3) + Θ(n) = Θ(n^{log₃5}) ≈ Θ(n^{1.465}). As k → ∞ the exponent approaches 1, but the
interpolation work and the constant factor grow with k.

**Floating-point FFT multiplication.** Apply the Cooley–Tukey FFT over C with ω = e^{2πi/N} to get
the convolution in Θ(N log N) operations. Round-off in floating-point roots of unity means the
recovered integer coefficients must be guarded with extra precision, and the error analysis requires
care.

## Evaluation settings

The natural yardstick is bit complexity: the number of bit operations to multiply two n-bit
integers, measured as a function of n, and the crossover n at which a method overtakes the
schoolbook, Karatsuba, and Toom–Cook methods in practice. Operands range from a few machine words
(where schoolbook wins) up through thousands and millions of bits (where asymptotically faster
methods take over). A core operation is a multiplication of two machine-word integers; additions,
shifts, and carries are counted in bit operations. Correctness is exact equality with the true
product, validated on integers spanning many bit-lengths.

## Code framework

The primitives that already exist: arbitrary-precision integers with shift/add/mod, and the
Cooley–Tukey FFT recurrence over a ring that supplies a root of unity. We can lay out the harness
into which a fast method will be filled.

```python
# Arbitrary-precision integers, bit-shifts, add, and modular reduction already exist,
# along with a Cooley-Tukey length-N transform recurrence over a ring that supplies a
# root of unity.

def multiply(a, b):
    # TODO: fill in an exact fast multiplication.
    pass
```
