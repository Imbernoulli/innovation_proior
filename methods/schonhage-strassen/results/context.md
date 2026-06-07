# Context: fast multiplication of very large integers

## Research question

Given two integers each about n bits long, compute their product exactly. The schoolbook
method costs Θ(n²) bit operations, and for the regimes that matter — thousands to millions of
digits in primality testing, π/e computations, cryptography, computer algebra — that quadratic
cost is the binding constraint. The question is how far below Θ(n²) one can push the *bit*
complexity of an exact integer product, ideally toward something near-linear, while keeping the
arithmetic exact (no floating-point round-off) so the answer is provably correct to the last bit.

A solution must do three things at once: (1) reduce the number of expensive "core" multiplications
far below the n² the schoolbook method performs; (2) stay exact — integer arithmetic only; and
(3) compose with itself, so the same idea can be applied at every scale rather than bottoming out.

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
power is 1. A transform built on such a g is a *Number-Theoretic Transform* (NTT). Of special
interest are Fermat-type moduli m = 2^n + 1: in Z/(2^n+1)Z one has 2^n ≡ −1, hence 2^{2n} ≡ 1, and
2 has order exactly 2n. If a transform length N divides 2n, then ω = 2^{2n/N} has order N. If
N divides n, then θ = 2^{n/N} satisfies θ^N = −1 and supplies the 2N-th root needed for a
negacyclic transform. Powers of 2 are the cheapest possible ring elements to multiply by —
multiplication by 2^s is a left shift, and the wraparound past 2^n folds back as a sign flip because
2^n ≡ −1.

**Cyclic vs. negacyclic convolution.** A length-N cyclic convolution computes c mod (x^N − 1); at
x = 2^M that is a product mod 2^{MN} − 1 (a Mersenne modulus). A *negacyclic* (negative-wrapped)
convolution computes c mod (x^N + 1); at x = 2^M that is a product mod 2^{MN} + 1 (a Fermat
modulus). The negacyclic version uses the odd powers of a 2N-th root θ: θ, θ^3, ..., θ^{2N−1},
the roots of x^N + 1. Those are exactly the evaluations needed for the negative-wrapped product;
the even powers belong to x^N − 1.

**The divide-and-conquer ancestry, as motivating empirical fact.** Splitting an integer into d+1
parts and treating it as a degree-d polynomial is a known, measured way to cut core-multiplication
counts: two d+1-digit numbers need (d+1)² core multiplications by the schoolbook layout, but
clever recombination needs far fewer — the count is what these methods are judged on, and the
schoolbook tableau makes the (d+1)² count concrete (a 4×4 product visibly performs 16 core
multiplications, an 8×8 performs 64). The open gap, observed across these schemes, is that the
exponent stays strictly above 1 and the hidden constant grows as the split widens.

## Baselines

**Schoolbook / long multiplication.** c_k = Σ_{i+j=k} a_i b_j directly, then carry. Θ(N²) core
multiplications for length-N operands. Exact and simple; the quadratic cost is the gap.

**Karatsuba (Karatsuba & Ofman, 1962).** Split each number into two halves, a = a₁B + a₀,
b = b₁B + b₀. Naively ab = a₁b₁B² + (a₁b₀ + a₀b₁)B + a₀b₀ needs four half-products; Karatsuba's
identity a₁b₀ + a₀b₁ = (a₁ + a₀)(b₁ + b₀) − a₁b₁ − a₀b₀ trades one multiplication for additions,
leaving three half-products. Recursing gives T(n) = 3T(n/2) + Θ(n) = Θ(n^{log₂3}) ≈ Θ(n^{1.585}).
Gap: the exponent is still well above 1.

**Toom–Cook (Toom, 1963; Cook, 1966).** Generalize: split into k+1 parts, regard each operand as a
degree-k polynomial, *evaluate* both at 2k+1 points, multiply pointwise (2k+1 smaller products),
and *interpolate* the product polynomial. Toom-3 gives 5 sub-products of n/3-size operands,
T(n) = 5T(n/3) + Θ(n) = Θ(n^{log₃5}) ≈ Θ(n^{1.465}). As k → ∞ the exponent → 1, but the
interpolation work and the constant factor blow up with k, so no fixed k reaches near-linear.
Gap: still polynomial with exponent > 1; evaluation/interpolation overhead caps the gain.

**Floating-point FFT multiplication.** Apply the Cooley–Tukey FFT over C with ω = e^{2πi/N} to get
the convolution in Θ(N log N) operations. Gap: it is *inexact* — round-off in floating-point roots
of unity means the recovered integer coefficients must be guarded with extra precision, and the
error analysis is delicate; this disqualifies it as an exact method without care.

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
# Arbitrary-precision integers, bit-shifts, add, and modular reduction already exist.

def reduce_mod(x, n):
    # TODO: choose a modulus where exact transform arithmetic is possible.
    pass

def shift_mod(x, exponent, n):
    # TODO: decide when multiplying by a transform root can be implemented as a shift.
    pass

def ntt(values, n, inverse=False):
    # Cooley-Tukey length-N transform over a ring.
    # TODO: supply the root of unity, the inverse root, and the inverse scale.
    pass

def pointwise_mul(x, y, n):
    # TODO: decide whether these coefficient products are base-case products
    #       or smaller products of the same modular shape.
    pass

def multiply_mod(a, b, K, M, n):
    # TODO: split into pieces, transform, multiply pointwise, invert, and reassemble.
    pass

def multiply(a, b):
    # TODO: choose the split parameters and call multiply_mod so the final product is exact.
    pass
```
