I propose the canonical name Schönhage–Strassen integer multiplication for this method. The goal is to multiply two very large integers exactly while using far fewer bit operations than the quadratic schoolbook approach, and to do it without any floating-point round-off that would jeopardize the last bit of the answer. The insight that makes this possible is that integer multiplication is really polynomial convolution in disguise. If I write an integer a as a sum of M-bit digits a_i times powers of 2^M, and do the same for b, then the product digits before carries are c_k equal to the sum over all i and j with i plus j equal to k of a_i times b_j. That is exactly the convolution of the two digit sequences. The remaining work, propagating carries, is only linear in the number of digits. So the hard part of multiplication is convolution, and the hard part of convolution can be diagonalized by a Fourier transform.

The convolution theorem tells me that the discrete Fourier transform turns convolution into pointwise multiplication. If I compute the DFT of each digit sequence, multiply the transformed values point by point, and apply the inverse DFT, I recover the convolution. The fast Fourier transform computes each DFT in N log N operations by recursively splitting even and odd indexed terms and recombining them with butterfly operations. The naive use of this idea runs the FFT over the complex numbers, where the root of unity is a transcendental complex exponential. That works in principle, but floating-point arithmetic introduces round-off in every twiddle factor, and guarding against that error requires extra precision and delicate analysis. I want exactness instead, so I need a setting where the root of unity is represented exactly.

The key observation is that the FFT does not actually need complex numbers. It only needs a commutative ring containing a primitive N-th root of unity omega, meaning omega raised to N is one and omega raised to r minus one is invertible for every r between zero and N, and it needs N itself to be invertible in the ring so that the inverse transform can divide by N. If I can find such a ring inside the integers modulo some carefully chosen number, every butterfly becomes an exact integer operation. The natural choice is a Fermat-type ring, the integers modulo 2^n plus one. In this ring, 2^n is congruent to minus one, so 2 raised to 2n is congruent to one, and 2 has order exactly 2n. If the transform length N divides 2n, then omega equal to 2 raised to 2n over N is a primitive N-th root of unity. Even better, omega is a power of two. Multiplying a residue by a power of two is just a bit shift, possibly followed by a reduction modulo 2^n plus one. That reduction is cheap because bits that spill past position n wrap back with a sign flip, exploiting the fact that 2^n is minus one. Every twiddle factor, every weight, and even the division by N in the inverse transform become shifts. The transform itself becomes multiplication-free.

There is still a subtlety. A plain length-N transform computes a cyclic convolution, which corresponds to multiplication modulo x^N minus one, meaning coefficient k plus N wraps around and adds to coefficient k. For recursive nesting it is more convenient to work modulo x^N plus one, the negacyclic convolution, because then the result at x equal to 2^M is a product modulo 2^{MN} plus one, which has the same Fermat shape as the ring itself. I can obtain the negacyclic convolution from the cyclic one by weighting the input digits. If I multiply digit i by theta raised to i before the transform, where theta equals 2 raised to n over N, then theta raised to N equals 2^n, which is minus one in the ring. The wraparound term at position k plus N picks up an extra factor of theta^N, which is minus one, so after unweighting by theta raised to minus k the wrapped contribution is subtracted rather than added. This converts the cyclic convolution into the negacyclic one. The weights and unweights are again powers of two, so they cost only shifts.

The algorithm now has a clean recursive shape. To multiply two numbers modulo 2^q plus one, I choose a transform length T and split each operand into T pieces of about half the ring width in bits. I weight the pieces, run the forward shift-only NTT, multiply the N resulting coefficient pairs pointwise modulo the current Fermat ring, run the inverse NTT, unweight, and reassemble with carries. The pointwise products are themselves multiplications modulo a smaller Fermat number, so they are smaller instances of the same problem. I recurse until the operands are small enough that a base-case multiplication is faster than setting up another transform. Choosing the inner size to be about the square root of the outer size makes the recursion depth logarithmic in the logarithm of the input size, because q shrinks to q^{1/2}, then q^{1/4}, and so on. Each level performs an N log N transform on numbers whose total bit length is proportional to the current input size, so each level costs order q log q. Multiplying by the logarithmic number of levels gives an overall bit complexity of order n log n log log n, all in exact integer arithmetic.

At the top level, where I want the full integer product rather than a residue, I zero-pad by filling only the lower half of the coefficient array and choosing the padded length so that the Fermat modulus exceeds the true product. The negacyclic wraparound then lands in zero coefficients and disappears, leaving the honest convolution followed by ordinary carry propagation. This is the complete Schönhage–Strassen pipeline: convolution through an exact number-theoretic transform over a Fermat ring, multiplication-free butterflies because the root of unity is a power of two, negacyclic weighting to make the subproblems self-similar, and recursion on those subproblems until they become small.

The following Python program illustrates the core mechanism. It implements arithmetic modulo 2^n plus one, a shift-based cyclic NTT, the negacyclic weighting that turns it into a Schönhage–Strassen style convolution, and a top-level wrapper that zero-pads to obtain the exact full product. It then verifies the result against Python's built-in big integers on several random inputs. The implementation is not tuned for speed, but it faithfully executes the mathematical pipeline that gives the n log n log log n exact bound.

```python
import random

def reduce_mod(x, n):
    return x % ((1 << n) + 1)

def shift_mod(x, exponent, n):
    # multiply by 2^exponent in Z/(2^n+1); exponent is modulo 2n.
    return reduce_mod(x << (exponent % (2 * n)), n)

def ntt(a, n, inverse=False):
    N = len(a)
    assert N & (N - 1) == 0 and (2 * n) % N == 0
    # bit-reversal permutation
    j = 0
    for i in range(1, N):
        bit = N >> 1
        while j & bit:
            j ^= bit
            bit >>= 1
        j ^= bit
        if i < j:
            a[i], a[j] = a[j], a[i]
    length = 2
    while length <= N:
        step = (2 * n) // length
        if inverse:
            step = -step
        for start in range(0, N, length):
            twiddle = 0
            for i in range(length // 2):
                u = a[start + i]
                v = shift_mod(a[start + i + length // 2], twiddle, n)
                a[start + i] = reduce_mod(u + v, n)
                a[start + i + length // 2] = reduce_mod(u - v, n)
                twiddle += step
        length <<= 1
    if inverse:
        k = N.bit_length() - 1
        for i in range(N):
            a[i] = shift_mod(a[i], -k, n)
    return a

def schonhage_strassen_mod(a, b, K, M, n):
    # Compute a*b mod (2^(K*M)+1) using a length-K negacyclic NTT over Z/(2^n+1).
    out_n = K * M
    out_mod = (1 << out_n) + 1
    a %= out_mod
    b %= out_mod
    if a == (1 << out_n) or b == (1 << out_n):
        return (a * b) % out_mod
    pa = [(a >> (M * i)) & ((1 << M) - 1) for i in range(K)]
    pb = [(b >> (M * i)) & ((1 << M) - 1) for i in range(K)]
    theta_step = n // K
    A = [shift_mod(pa[i], theta_step * i, n) for i in range(K)]
    B = [shift_mod(pb[i], theta_step * i, n) for i in range(K)]
    ntt(A, n)
    ntt(B, n)
    C = [reduce_mod(A[i] * B[i], n) for i in range(K)]
    ntt(C, n, inverse=True)
    C = [shift_mod(C[i], -theta_step * i, n) for i in range(K)]
    res = 0
    for i in range(K):
        ci = C[i]
        if ci > (1 << (n - 1)):
            ci -= (1 << n) + 1
        res += ci << (M * i)
    return res % out_mod

def schonhage_strassen_full(a, b):
    if a == 0 or b == 0:
        return 0
    sign = -1 if (a < 0) ^ (b < 0) else 1
    a, b = abs(a), abs(b)
    S = max(a.bit_length(), b.bit_length())
    # zero-pad: data lives in lower K/2 pieces, MN >= 2S.
    k = max(2, (2 * S).bit_length().bit_length() + 2)
    K = 1 << k
    M = (S + (K // 2) - 1) // (K // 2)
    n = 2 * M + K.bit_length()
    n = ((n + K - 1) // K) * K  # ensure K divides n
    return sign * schonhage_strassen_mod(a, b, K, M, n)

if __name__ == "__main__":
    random.seed(0)
    for bits in [64, 256, 1024, 4096, 16384]:
        a = random.getrandbits(bits)
        b = random.getrandbits(bits)
        b_neg = -b
        assert schonhage_strassen_full(a, b) == a * b
        assert schonhage_strassen_full(a, b_neg) == a * b_neg
        print(f"ok: {bits}-bit operands")
    print("All exact-multiplication checks passed.")
```

This program demonstrates why the Schönhage–Strassen algorithm is historically important: it was the first practical method to break the quadratic barrier for exact integer multiplication, reaching a worst-case bit complexity of order n log n log log n while keeping every arithmetic step exact. The modern view is that multiplication is convolution, convolution is diagonalized by Fourier transforms, and the FFT becomes both exact and cheap once the transform is performed over a ring where the root of unity is itself a shift, namely a power of two in the Fermat ring Z modulo 2^n plus one.
