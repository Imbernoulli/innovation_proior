I want to multiply two enormous integers — n bits each, where n is in the millions — and I want it to be fast and exact. Schoolbook is Θ(n²): for length-N digit sequences I do N² little products. Karatsuba shaves that to n^{1.585} by computing the two cross terms a₁b₀+a₀b₁ with one multiply instead of two, recursing; Toom-Cook generalizes it — cut into k+1 pieces, treat each number as a degree-k polynomial, evaluate at 2k+1 points, multiply pointwise, interpolate — and pushes the exponent down toward 1 as k grows. But the exponent never *reaches* 1, and the hidden constant explodes as k grows, because the evaluation-and-interpolation machinery and the recombination overhead grow with the number of pieces. So Toom-Cook is a sequence of methods crawling toward exponent 1 but never arriving. I want to actually arrive — near n log n, not n^{1+ε}.

Let me strip the problem down to what it really is. If I write a = Σ_i a_i 2^{wi} and b = Σ_j b_j 2^{wj}, with each a_i, b_j a w-bit digit, then ab = Σ_k (Σ_{i+j=k} a_i b_j) 2^{wk}, *before* I worry about carries. So the digit sequence of the product, pre-carry, is c_k = Σ_{i+j=k} a_i b_j. Stare at that. That is exactly the convolution of the two digit sequences. The carries are a cheap linear cleanup afterward. So the whole problem is: convolve two length-N sequences cheaply, then propagate carries. The cost of multiplication *is* the cost of convolution.

And there's a classical fact about convolution I keep circling back to: it diagonalizes under the Fourier transform. The convolution theorem. If I take the DFT of each sequence, â = DFT(a), b̂ = DFT(b), then the DFT of the convolution is the *pointwise* product â · b̂. So a∗b = IDFT(â·b̂). The reason that's exciting: the convolution itself is the Θ(N²) operation, but the DFT — which turns it into N independent pointwise multiplications — can be computed in Θ(N log N) by the Fast Fourier Transform. Cooley and Tukey: split the transform into its even-indexed and odd-indexed halves, each of which is a half-size transform, glue with a "butterfly," recurse; T(N) = 2T(N/2) + Θ(N) = Θ(N log N). So three FFTs (two forward, one inverse) plus N pointwise products. If those pointwise products were free, I'd have an Θ(N log N) multiplication. They're not free, but hold that thought.

The first attempt is obvious: use that transform sandwich directly and see where the arithmetic breaks.

The obvious way is the complex FFT: ω = e^{2πi/N}, do everything over the complex numbers. Let me actually push this through, because even if it is inexact it tells me exactly where the round-off bites. In the classical parameterization, I split the N-bit inputs into 2^{r-1} nonzero chunks of l bits each and zero-pad to a length-2^r cyclic transform, so the honest acyclic convolution fits. A coefficient c_k is a sum of at most 2^{r-1} products of l-bit chunks, hence |c_k| < 2^{2l+r}. To recover c_k by rounding, the final error must be below 1/2. The transform size and the coefficient magnitudes amplify s-bit fixed-point errors, and the safe guard condition is

  s >= 2l + 2r + lg N + const.

Now count the cost if a complex butterfly multiplication at this precision is reduced to integer multiplications supplied by the previous method V_m. The real operands have Θ(l+r) useful bits plus guard room, and a complex multiplication costs a constant number of such real multiplications, so

  M_{m+1}(N) = O(2^r((1+r)^2 + r M_m(3(l+r)))).

If I call this transform exponent n instead of r, those two bounds read s >= 2l + 2n + lg N + const and M_{m+1}(N)=O(2^n((1+n)^2+n M_m(3(l+n)))). The (1+r)^2 term is the fixed-precision shift/add/rounding work around each transform position, and the r factor is the number of FFT stages that use these guarded coefficient multiplications. With the first FFT construction this gives M_1(N)=O(N(lg N)^2). Feeding that bound back into the same recurrence gives M_2(N)=O(N lg N (lg lg N)^2). One more nesting lets the remaining squared iterated-log factor be absorbed into (lg lg N)^ε for any fixed ε>0, so three nestings give O(N lg N (lg lg N)^{1+ε}). The convolution route works and gets essentially to N lg N, except for the one ugly word: round-off. The whole apparatus rests on guard bits and an error bound. I want exact. Floating-point roots of unity are not exact. I need to kill the round-off at the source, and if I can, shave away the last (lg lg N)^ε too.

Where does the round-off come from? Only from ω being a transcendental complex number. But look again at what the FFT actually demands of ω: it needs ω^N = 1, it needs ω^r − 1 to be invertible for 0 < r < N (so the geometric-series cancellations that make the transform invert actually work), and it needs N itself to be invertible (for the 1/N in the inverse transform). That's *all* it needs. Nothing says ω must be a complex number. It can be any element of any commutative ring that happens to have an N-th root of unity of the right kind. So: pick a finite ring. Then arithmetic is exact integer-mod-something arithmetic, no round-off at all. This is a number-theoretic transform.

Which ring? I need a primitive N-th root of unity living in it. The naive choice — a prime field Z/pZ where p ≡ 1 mod N has a root of unity — works for exactness but the root of unity is some ugly residue, and multiplying by it in the butterflies is a full multiplication mod p, which is *itself* an expensive operation. I'd be paying a real multiply at every one of the N log N butterfly steps. That can't be how I get below n². I need a ring where the root of unity is not just exact but *cheap to multiply by*.

What's the cheapest thing to multiply a binary integer by? A power of two. Multiplying by 2^s is a left shift — for free in bit-operation count, it just relabels bit positions. So the dream root of unity is a power of two. When is a power of two a root of unity? I need 2^s, raised to N, to be ≡ 1 in my ring. Equivalently I need the ring to be Z modulo something that makes a power of 2 have finite order. And there's an obvious candidate sitting right there: work modulo 2^n + 1. Because then 2^n ≡ −1, so 2^{2n} ≡ 1, and 2 has order exactly 2n in this ring. If the transform length N divides 2n, then ω = 2^{2n/N} has order N, since the order of 2^a is 2n/gcd(2n,a) and here a = 2n/N. So ω is a primitive N-th root of unity, and it is a power of two. Every twiddle in the FFT becomes "shift left by some multiple of 2n/N bits, then reduce mod 2^n+1." And the reduction itself is cheap: any bits that spill past position n fold back with a sign flip, because 2^n ≡ −1. Shift and fold. No multiplications in the transform at all — the FFT becomes *multiplication-free*. The only genuine multiplications left in the whole convolution are the N pointwise products of residues. That's the crux of the cheapness: I moved all the work into N pointwise products and made the surrounding transform cost only shifts and adds.

Let me sanity-check the root of unity on a tiny case to make sure I'm not fooling myself. Take n = 8, ring Z/(2^8+1) = Z/257. Then 2^8 = 256 ≡ −1, so 2^{16} ≡ 1, 2 has order 16. For an 8-point transform, N = 8 divides 16, ω = 2^{16/8} = 2² = 4, and indeed 4^8 = 2^{16} ≡ 1 mod 257, and 4^4 = 2^8 = 256 ≡ −1 ≠ 1, so 4 is a *primitive* 8th root. Good — the construction is real.

Now, the size problem, because I haven't been careful and it's about to bite. If I try to transform all n bits as a single block — N ≈ n points — then I need N | 2n with N ≈ n, fine, but each "coefficient" is a residue mod 2^n+1, an n-bit number, and the N pointwise products are products of n-bit numbers, and there are N ≈ n of them... that's Θ(n) multiplications each of n-bit numbers, which is Θ(n²·something). I've gone nowhere; in fact the additions alone, Θ(n) of them each Θ(n) bits across Θ(log n) stages, already cost Θ(n² log n). Worse than schoolbook. The single-big-block NTT is a trap.

So I must chunk. Don't make N ≈ n; make N modest and the *pieces* big. Split the S-bit input into N = 2^k pieces of M = S/N bits each, and regard the integer as a polynomial evaluated at x = 2^M: a = A(2^M) with A(x) = Σ_{i<N} a_i x^i, a_i an M-bit digit. Same for b. Now the convolution is the product of two degree-(N−1) polynomials, length N, and the pointwise products are on numbers of size about the ring width — call it n bits — which I get to choose, and which is roughly 2M, not the whole input. The transform length N and the inner size M are now two separate knobs I can tune against each other. That's the structural freedom Toom-Cook didn't have: there, the number of pieces and the recombination cost were locked together by interpolation; here the FFT decouples them.

How big does the ring 2^n+1 need to be — how large is n — so the pointwise products are computed without overflow? The coefficients of the product polynomial are c_k = Σ_{i+j=k} a_i b_j. Each a_i, b_j is an M-bit number, so each product is < 2^{2M}, and there are at most N of them in a c_k, so c_k < N·2^{2M}. For the ring Z/(2^n+1) to hold each c_k faithfully — to recover the true integer coefficient, not its residue — I need 2^n+1 > N·2^{2M}, i.e. n ≥ 2M + log₂N + 1. So pick n a bit more than 2M and I'm safe. And I'll want N | 2n for the root of unity; arrange n to be a multiple of N and that's automatic.

Now the kind of convolution matters. The FFT over Z/(2^n+1) with ω = 2^{2n/N} naturally computes a *cyclic* convolution — it gives me the product polynomial mod (x^N − 1), because the transform lives on the group Z_N and convolution there wraps around. Mod (x^N − 1) means coefficient k and coefficient k+N get added together. If I want the *full* product polynomial (degree up to 2N−2) I'd have to zero-pad: use length 2N, fill only the bottom N coefficients, leave the top N as zeros, so the wraparound lands harmlessly in the zeros. That works for the top level and gives me the honest full product. Fine.

But here's the thing I want for the recursion. Those N pointwise products — products of n-bit residues — are themselves multiplications of largish integers. I want to compute *them* by the same method, recursively. For the recursion to nest cleanly, the inner problem should have the *same shape* as the outer one: "multiply two numbers modulo 2^{something}+1." Look at what the pointwise products actually are: I'm multiplying residues that live in Z/(2^n+1). So if my method's natural output were "the product modulo 2^{MN}+1," then the inner multiply-mod-2^n+1 is just a smaller instance of the exact same routine, with no zero-padding waste. The levels would fit together like nested dolls, every level being "multiply mod a Fermat-type modulus."

Can I make the transform produce a product mod (x^N + 1) instead of mod (x^N − 1)? That's the *negacyclic* convolution — negative-wrapped, where coefficient k+N gets *subtracted* from coefficient k rather than added. At x = 2^M that gives me ab mod (2^{MN} + 1) — exactly the Fermat-type modulus that matches the ring, exactly what the recursion wants. So I want the negacyclic, not the cyclic, convolution.

What does negacyclic require? Mod (x^N + 1) instead of mod (x^N − 1). In the polynomial ring, x^N ≡ −1. So I need a transform whose root of unity ω satisfies ω^N = −1 — a *2N-th* root of unity, not an N-th root. Do I have one as a power of two? Yes: θ = 2^{n/N}. Check: θ^N = 2^n ≡ −1 in Z/(2^n+1). Perfect — θ is a 2N-th root of unity and still a power of two. (This needs N | n, slightly stronger than the N | 2n I had before; arrange n to be a multiple of N and both hold.)

How do I actually fold θ into the machinery? I could rebuild the FFT around a 2N-th root, but there's a cleaner trick. Take the plain length-N cyclic NTT (root ω = θ² = 2^{2n/N}) that I already have, and *weight* the inputs before transforming: replace a_i by θ^i a_i and b_j by θ^j b_j. Then the cyclic convolution of the weighted sequences, unweighted afterward by θ^{−k}, equals the negacyclic convolution of the originals. Why does that work? Watch the wraparound term. In the cyclic convolution a coefficient at position k+N folds onto position k unchanged. With the weights, that term carries a factor θ^{(k+N)} = θ^k · θ^N = θ^k · (−1); after I divide the whole position-k result by θ^k to unweight, the wrapped term keeps the extra factor −1 relative to the unwrapped term. So the wrap that *added* now effectively *subtracts* — precisely mod (x^N + 1). The weighting converts cyclic into negacyclic. This is the discrete weighted transform.

And the weights are free in the same way the twiddles are: θ^i = 2^{(n/N)i} is a power of two, so weighting is a shift. Unweighting by θ^{−k} = 2^{−(n/N)k} is also a shift (a negative shift is a shift the other way, mod the order 2n). The whole negacyclic apparatus costs only shifts on top of the cyclic NTT.

Let me make this concrete on a small worked instance to be sure the weighting is right, the way I'd actually trust it. Suppose I'm doing a negacyclic length-8 convolution in the ring mod 2^16 + 1, so n = 16, N = 8, θ = 2^{n/N} = 2^2 = 4 — wait, let me recompute, θ = 2^{16/8} = 2^2 = 4, and ω = θ² = 2^4 = 16. The weights θ^i for i = 0..7 are 4^0, 4^1, …, 4^7 = 1, 4, 16, 64, 256, 1024, 4096, 16384, each mod (2^16+1). I multiply a_i by these shifts, run the ordinary 8-point NTT with ω = 16, multiply pointwise, inverse-NTT, divide each output position k by θ^k, and reduce mod 2^16+1. What comes back is A(x)B(x) mod (x^8+1) with coefficients in the ring; if I take x = 2^2, the same negative wrap is multiplication modulo 2^{16}+1. Good — the bookkeeping closes.

There's a further halving hiding in the structure that I'd be foolish to leave on the table. If I computed the negative-wrapped product by first embedding it in a length-2N cyclic transform, I would evaluate at all 2N powers of a primitive 2N-th root and do 2N pointwise products. But my inputs are honest integer-coefficient polynomials and I only want the factor modulo x^N+1. The roots of x^N+1 are the odd powers θ, θ^3, ..., θ^{2N-1}; the even powers are roots of x^N−1 and describe the other CRT factor. So the even-indexed values are not needed for this product at all. The weighting identity is exactly the compact length-N way to evaluate at those odd powers, because θω^j = θ(θ²)^j = θ^{2j+1}. I keep only the odd-indexed products, N of them instead of the 2N that a zero-padded cyclic transform would have used. This is not an asymptotic miracle, but it halves the expensive recursive multiplications at every level.

There's one more thing the inverse transform needs: division by N. The inverse NTT formula has a 1/N factor. Is that cheap and exact here? N = 2^k, so 1/N = 2^{−k}. And 2 is invertible mod 2^n+1: since 2·2^{2n−1} = 2^{2n} ≡ 1, we have 2^{−1} ≡ 2^{2n−1}. So 1/N = 2^{−k} ≡ 2^{k(2n−1)} mod (2^n+1) — yet another power of two, yet another shift. Everything in this transform, forward and inverse, twiddles and weights and the 1/N, is a shift. The only real arithmetic is adds (with the cheap mod-2^n+1 fold) and the N pointwise products. That's the whole point and it's now airtight.

So here is the algorithm taking shape. To multiply two S-bit numbers a, b: pick N = 2^k pieces and choose M so the data fit in the lower half when I want a full product; choose a ring width n ≥ 2M + log₂N + 1 with N | n; split a, b into M-bit pieces; weight by θ^i; forward-NTT both by shifts and adds; multiply the N pairs of coefficients pointwise mod 2^n+1; inverse-NTT; unweight by θ^{−k}; reassemble Σ c_k 2^{Mk}, letting carries propagate, and reduce mod the appropriate Fermat modulus. For the top level, where I want the honest full product and not a residue, I zero-pad by filling only the lower N/2 pieces and choose MN at least twice the input bit length, so the negative wrap lands in zeros and 2^{MN}+1 is larger than the true product.

Now recurse. The N pointwise products are multiplications of n-bit numbers mod 2^n+1 — the exact same problem, smaller. Apply the method to each, choosing the inner parameters N', M' with M'·N' = n so they nest. Keep recursing until n drops below a threshold where a native machine multiply (or Karatsuba/Toom) is simply faster than setting up another transform; there, stop and multiply directly.

Let me work out the cost in the modular form, because that is the clean recursive object. Suppose the current task is multiplication modulo 2^q+1. Choose q0 = Θ(q^{1/2}) and T = 2q/q0, arranged so T divides q0. Split each input into T chunks of q0/2 bits. Then x = 2^{q0/2} satisfies x^T = 2^q ≡ −1 modulo 2^q+1, so the task becomes multiplication in R[x]/(x^T+1), where R = Z/(2^{q0}+1)Z. This is where I have to keep two correctness stories separate. At the top level, if I want actual integer coefficients back, I need the safety width n ≥ 2M + log₂N + 1. Inside the recursive modular problem I am not trying to recover large integer coefficients in each slot; the coefficients are residues in R, and the pointwise products are exactly products modulo 2^{q0}+1.

The transform over R touches T coefficients, each q0 bits wide, through O(log T) stages; since Tq0 = 2q, the shift/add work is O(q log q). The pointwise stage has exactly T products in R, because I evaluate only at the odd roots of x^T+1, and each product is a smaller modular multiplication of size q0. So if F(q) is the cost of multiplication modulo 2^q+1,

  F(q) <= (2q/q0) F(q0) + O(q log q),      q0 = Θ(q^{1/2}).

Now divide by q log q. The recursive term contributes (2q/q0)·q0·log q0 = 2q·(1/2 log q + O(1)) = O(q log q), so each level adds only a constant amount to F(q)/(q log q). The size falls as q -> q^{1/2} -> q^{1/4} -> ..., and it reaches the base case when q^{1/2^d} is constant, i.e. when 2^d = Θ(log q). Thus the recursion depth is Θ(log log q), and

  F(q) = O(q log q log log q).

There it is. The log q is the FFT; the log log q is the number of exact modular levels before the pieces become base-case size. And every step is exact integer arithmetic, because the root of unity is a power of two in a Fermat ring and nothing ever leaves the integers. That's the whole thing: multiplication is convolution; convolution is a pointwise product between number-theoretic transforms over Z/(2^n+1) with 2 as the root of unity, so the transform is just shifts; the negacyclic weighting makes the result land mod a Fermat modulus so the pointwise products are smaller instances of the same problem; recurse, and the recursion depth log log n times the FFT cost n log n gives n log n log log n.

(One free optimization I notice while I'm here: in Z/(2^n+1) with n divisible by 4, the element 2^{3n/4} − 2^{n/4} squares to 2 — call it √2 — and it has order 4n instead of 2n, since its square has order 2n. That lets the transform length N divide 4n instead of 2n, doubling the available length for the same ring width, and multiplying by √2 costs just two shifts and a subtract. A handy way to squeeze the parameters, though the asymptotics don't need it.)

Let me write it down in code in the structure I just derived — split, weight, forward NTT by shifts, recursive pointwise multiply, inverse NTT, unweight, carry — and check it against the schoolbook product.

```python
# Arithmetic in R = Z/(2^n + 1)Z.  In R: 2^n == -1, so 2 has order 2n.
def reduce_mod(x, n):
    return x % ((1 << n) + 1)

def shift_mod(x, exponent, n):
    # multiply by 2^exponent in R; exponent is taken modulo the order 2n.
    return reduce_mod(x << (exponent % (2 * n)), n)

def ntt(a, n, inverse=False):
    # length-N cyclic NTT, omega = 2^(2n/N); every '* w^j' is a shift.
    N = len(a)
    assert N & (N - 1) == 0 and (2 * n) % N == 0
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
        step = (2 * n) // length                       # root for this stage
        if inverse:
            step = -step                               # inverse root
        for start in range(0, N, length):
            twiddle = 0
            for i in range(length // 2):
                u = a[start + i]
                v = shift_mod(a[start + i + length // 2], twiddle, n)
                a[start + i]               = reduce_mod(u + v, n)   # butterfly
                a[start + i + length // 2] = reduce_mod(u - v, n)
                twiddle += step
        length <<= 1
    if inverse:                                          # 1/N = 2^{-log2 N}
        k = N.bit_length() - 1
        for i in range(N):
            a[i] = shift_mod(a[i], -k, n)
    return a

def multiply_mod(a, b, K, M, n):
    # a*b mod (2^(K*M)+1): K=2^k pieces of M bits, ring 2^n+1; needs K|n, n>=2M+log2 K+1.
    out_n = K * M
    out_mod = (1 << out_n) + 1
    a %= out_mod
    b %= out_mod
    if a == (1 << out_n) or b == (1 << out_n):
        return (a * b) % out_mod                       # the single residue that does not fit in out_n bits
    pa = [(a >> (M * i)) & ((1 << M) - 1) for i in range(K)]
    pb = [(b >> (M * i)) & ((1 << M) - 1) for i in range(K)]
    theta_step = n // K                                # theta = 2^theta_step, theta^K = 2^n == -1
    A = [shift_mod(pa[i], theta_step * i, n) for i in range(K)]   # negacyclic weights
    B = [shift_mod(pb[i], theta_step * i, n) for i in range(K)]
    ntt(A, n); ntt(B, n)
    C = [pointwise_mul(A[i], B[i], n) for i in range(K)]      # recursive pointwise products
    ntt(C, n, inverse=True)
    C = [shift_mod(C[i], -theta_step * i, n) for i in range(K)]  # unweight
    res = 0
    for i in range(K):
        ci = C[i]
        if ci > (1 << (n - 1)): ci -= (1 << n) + 1     # negacyclic coeffs may be negative
        res += ci << (M * i)                           # reassembly; Python integer addition carries
    return res % out_mod

RECURSE_BITS = 1 << 14
def pointwise_mul(x, y, n):
    if n <= RECURSE_BITS:
        return reduce_mod(x * y, n)                   # base case: stop recursing
    if x == (1 << n) or y == (1 << n):
        return reduce_mod(x * y, n)                   # residue -1 does not fit in n split bits
    k = max(2, n.bit_length() // 2)
    K = 1 << k                                        # recurse: same routine, smaller n
    while K > 2 and n % K:
        k -= 1
        K = 1 << k
    if n % K:
        return reduce_mod(x * y, n)
    M = n // K
    np = 2 * M + K.bit_length()                        # 2M + log2(K) + 1
    np = ((np + K - 1) // K) * K                       # round so K | n'
    if np >= n:
        return reduce_mod(x * y, n)
    return multiply_mod(x, y, K, M, np) % ((1 << n) + 1)

def multiply(a, b):
    # full product via zero-padded negacyclic transform (data fills lower K/2 pieces).
    if a == 0 or b == 0:
        return 0
    sign = -1 if (a < 0) ^ (b < 0) else 1
    a, b = abs(a), abs(b)
    S = max(a.bit_length(), b.bit_length())
    k = max(2, (2 * S).bit_length().bit_length() + 2)
    K = 1 << k
    M = (S + (K // 2) - 1) // (K // 2)
    np = 2 * M + K.bit_length()
    np = ((np + K - 1) // K) * K
    return sign * multiply_mod(a, b, K, M, np)
```

So the causal chain, end to end: a product of two integers is the convolution of their digit sequences; a convolution is a pointwise product sandwiched between Fourier transforms, and the FFT makes those transforms Θ(N log N); doing the transform over the Fermat ring Z/(2^n+1) instead of over C makes it exact and — because 2^n ≡ −1 means 2 (and its powers θ = 2^{n/N}, ω = 2^{2n/N}) are roots of unity that are powers of two — makes every twiddle, weight, and 1/N a mere bit-shift, so the transform is multiplication-free and the only genuine multiplications are the N pointwise products; weighting the inputs by θ^i turns the cyclic transform into a negacyclic one, so the output lands mod a Fermat modulus and the inner products are smaller instances of the very same problem; recursing on them, with piece sizes chosen so the exponent decays geometrically, bottoms out after Θ(log log n) levels, and Θ(log log n) levels times the Θ(n log n) transform per level gives the exact-arithmetic upper bound O(n log n log log n).
