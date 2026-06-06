OK, let me think this through from scratch. I want to multiply two enormous integers — n bits each, where n is in the millions — and I want it to be fast and exact. Schoolbook is Θ(n²): for length-N digit sequences I do N² little products. Karatsuba shaves that to n^{1.585} by computing the two cross terms a₁b₀+a₀b₁ with one multiply instead of two, recursing; Toom-Cook generalizes it — cut into k+1 pieces, treat each number as a degree-k polynomial, evaluate at 2k+1 points, multiply pointwise, interpolate — and pushes the exponent down toward 1 as k grows. But the exponent never *reaches* 1, and the hidden constant explodes as k grows, because the evaluation-and-interpolation machinery and the recombination overhead grow with the number of pieces. So Toom-Cook is a sequence of methods crawling toward exponent 1 but never arriving. I want to actually arrive — near n log n, not n^{1+ε}.

Let me strip the problem down to what it really is. If I write a = Σ_i a_i 2^{wi} and b = Σ_j b_j 2^{wj}, with each a_i, b_j a w-bit digit, then ab = Σ_k (Σ_{i+j=k} a_i b_j) 2^{wk}, *before* I worry about carries. So the digit sequence of the product, pre-carry, is c_k = Σ_{i+j=k} a_i b_j. Stare at that. That is exactly the convolution of the two digit sequences. The carries are a cheap linear cleanup afterward. So the whole problem is: convolve two length-N sequences cheaply, then propagate carries. The cost of multiplication *is* the cost of convolution.

And there's a classical fact about convolution I keep circling back to: it diagonalizes under the Fourier transform. The convolution theorem. If I take the DFT of each sequence, â = DFT(a), b̂ = DFT(b), then the DFT of the convolution is the *pointwise* product â · b̂. So a∗b = IDFT(â·b̂). The reason that's exciting: the convolution itself is the Θ(N²) operation, but the DFT — which turns it into N independent pointwise multiplications — can be computed in Θ(N log N) by the Fast Fourier Transform. Cooley and Tukey: split the transform into its even-indexed and odd-indexed halves, each of which is a half-size transform, glue with a "butterfly," recurse; T(N) = 2T(N/2) + Θ(N) = Θ(N log N). So three FFTs (two forward, one inverse) plus N pointwise products. If those pointwise products were free, I'd have an Θ(N log N) multiplication. They're not free, but hold that thought.

So this is the plan: multiplication = convolution = pointwise product sandwiched between transforms. Let me try it the obvious way first and watch it break.

The obvious way is the complex FFT: ω = e^{2πi/N}, do everything over the complex numbers. And immediately — wall. These are e^{2πi/N}, irrational, I have to represent them in floating point, and every butterfly multiplies by them. The convolution coefficients c_k are integers, but I'll recover them as floating-point numbers polluted by round-off, and to be *sure* I round to the right integer I need enough guard bits that the error analysis becomes the whole problem. I wanted exact. Floating-point roots of unity are not exact. I need to kill the round-off at the source.

Where does the round-off come from? Only from ω being a transcendental complex number. But look again at what the FFT actually demands of ω: it needs ω^N = 1, it needs ω^r − 1 to be invertible for 0 < r < N (so the geometric-series cancellations that make the transform invert actually work), and it needs N itself to be invertible (for the 1/N in the inverse transform). That's *all* it needs. Nothing says ω must be a complex number. It can be any element of any commutative ring that happens to have an N-th root of unity of the right kind. So: pick a finite ring. Then arithmetic is exact integer-mod-something arithmetic, no round-off at all. This is a number-theoretic transform.

Which ring? I need a primitive N-th root of unity living in it. The naive choice — a prime field Z/pZ where p ≡ 1 mod N has a root of unity — works for exactness but the root of unity is some ugly residue, and multiplying by it in the butterflies is a full multiplication mod p, which is *itself* an expensive operation. I'd be paying a real multiply at every one of the N log N butterfly steps. That can't be how I get below n². I need a ring where the root of unity is not just exact but *cheap to multiply by*.

What's the cheapest thing to multiply a binary integer by? A power of two. Multiplying by 2^s is a left shift — for free in bit-operation count, it just relabels bit positions. So the dream root of unity is a power of two. When is a power of two a root of unity? I need 2^s, raised to N, to be ≡ 1 in my ring. Equivalently I need the ring to be Z modulo something that makes a power of 2 have finite order. And there's an obvious candidate sitting right there: work modulo 2^n + 1. Because then 2^n ≡ −1, so 2^{2n} ≡ 1, and 2 has order exactly 2n in this ring. If I make the transform length N divide 2n, then ω = 2^{2n/N} satisfies ω^N = 2^{2n} = 1 — a primitive N-th root of unity, and it's a power of two. Every twiddle in the FFT becomes "shift left by some multiple of 2n/N bits, then reduce mod 2^n+1." And the reduction itself is cheap: any bits that spill past position n fold back with a sign flip, because 2^n ≡ −1. Shift and fold. No multiplications in the transform at all — the FFT becomes *multiplication-free*. The only genuine multiplications left in the whole convolution are the N pointwise products of residues. That's the crux of the cheapness: I moved all the work into N pointwise products and made the surrounding transform cost only shifts and adds.

Let me sanity-check the root of unity on a tiny case to make sure I'm not fooling myself. Take n = 8, ring Z/(2^8+1) = Z/257. Then 2^8 = 256 ≡ −1, so 2^{16} ≡ 1, 2 has order 16. For an 8-point transform, N = 8 divides 16, ω = 2^{16/8} = 2² = 4, and indeed 4^8 = 2^{16} ≡ 1 mod 257, and 4^4 = 2^8 = 256 ≡ −1 ≠ 1, so 4 is a *primitive* 8th root. Good — the construction is real.

Now, the size problem, because I haven't been careful and it's about to bite. If I try to transform all n bits as a single block — N ≈ n points — then I need N | 2n with N ≈ n, fine, but each "coefficient" is a residue mod 2^n+1, an n-bit number, and the N pointwise products are products of n-bit numbers, and there are N ≈ n of them... that's Θ(n) multiplications each of n-bit numbers, which is Θ(n²·something). I've gone nowhere; in fact the additions alone, Θ(n) of them each Θ(n) bits across Θ(log n) stages, already cost Θ(n² log n). Worse than schoolbook. The single-big-block NTT is a trap.

So I must chunk. Don't make N ≈ n; make N modest and the *pieces* big. Split the S-bit input into N = 2^k pieces of M = S/N bits each, and regard the integer as a polynomial evaluated at x = 2^M: a = A(2^M) with A(x) = Σ_{i<N} a_i x^i, a_i an M-bit digit. Same for b. Now the convolution is the product of two degree-(N−1) polynomials, length N, and the pointwise products are on numbers of size about the ring width — call it n bits — which I get to choose, and which is roughly 2M, not the whole input. The transform length N and the inner size M are now two separate knobs I can tune against each other. That's the structural freedom Toom-Cook didn't have: there, the number of pieces and the recombination cost were locked together by interpolation; here the FFT decouples them.

How big does the ring 2^n+1 need to be — how large is n — so the pointwise products are computed without overflow? The coefficients of the product polynomial are c_k = Σ_{i+j=k} a_i b_j. Each a_i, b_j is an M-bit number, so each product is < 2^{2M}, and there are at most N of them in a c_k, so c_k < N·2^{2M}. For the ring Z/(2^n+1) to hold each c_k faithfully — to recover the true integer coefficient, not its residue — I need 2^n+1 > N·2^{2M}, i.e. n ≥ 2M + log₂N + 1. So pick n a bit more than 2M and I'm safe. And I'll want N | 2n for the root of unity; arrange n to be a multiple of N and that's automatic.

Now a subtlety about *which* convolution I'm computing, and this is where the truly clever part lives. The FFT over Z/(2^n+1) with ω = 2^{2n/N} naturally computes a *cyclic* convolution — it gives me the product polynomial mod (x^N − 1), because the transform lives on the group Z_N and convolution there wraps around. Mod (x^N − 1) means coefficient k and coefficient k+N get added together. If I want the *full* product polynomial (degree up to 2N−2) I'd have to zero-pad: use length 2N, fill only the bottom N coefficients, leave the top N as zeros, so the wraparound lands harmlessly in the zeros. That works for the top level and gives me the honest full product. Fine.

But here's the thing I want for the recursion. Those N pointwise products — products of n-bit residues — are themselves multiplications of largish integers. I want to compute *them* by the same method, recursively. For the recursion to nest cleanly, the inner problem should have the *same shape* as the outer one: "multiply two numbers modulo 2^{something}+1." Look at what the pointwise products actually are: I'm multiplying residues that live in Z/(2^n+1). So if my method's natural output were "the product modulo 2^{MN}+1," then the inner multiply-mod-2^n+1 is just a smaller instance of the exact same routine, with no zero-padding waste. The levels would fit together like nested dolls, every level being "multiply mod a Fermat-type modulus."

Can I make the transform produce a product mod (x^N + 1) instead of mod (x^N − 1)? That's the *negacyclic* convolution — negative-wrapped, where coefficient k+N gets *subtracted* from coefficient k rather than added. At x = 2^M that gives me ab mod (2^{MN} + 1) — exactly the Fermat-type modulus that matches the ring, exactly what the recursion wants. So I want the negacyclic, not the cyclic, convolution.

What does negacyclic require? Mod (x^N + 1) instead of mod (x^N − 1). In the polynomial ring, x^N ≡ −1. So I need a transform whose root of unity ω satisfies ω^N = −1 — a *2N-th* root of unity, not an N-th root. Do I have one as a power of two? Yes: θ = 2^{n/N}. Check: θ^N = 2^n ≡ −1 in Z/(2^n+1). Perfect — θ is a 2N-th root of unity and still a power of two. (This needs N | n, slightly stronger than the N | 2n I had before; arrange n to be a multiple of N and both hold.)

How do I actually fold θ into the machinery? I could rebuild the FFT around a 2N-th root, but there's a cleaner trick. Take the plain length-N cyclic NTT (root ω = θ² = 2^{2n/N}) that I already have, and *weight* the inputs before transforming: replace a_i by θ^i a_i and b_j by θ^j b_j. Then the cyclic convolution of the weighted sequences, unweighted afterward by θ^{−k}, equals the negacyclic convolution of the originals. Why does that work? Watch the wraparound term. In the cyclic convolution a coefficient at position k+N folds onto position k unchanged. With the weights, that term carries a factor θ^{(k+N)} = θ^k · θ^N = θ^k · (−1); after I divide the whole position-k result by θ^k to unweight, the wrapped term keeps the extra factor −1 relative to the unwrapped term. So the wrap that *added* now effectively *subtracts* — precisely mod (x^N + 1). The weighting converts cyclic into negacyclic. This is the discrete weighted transform.

And the weights are free in the same way the twiddles are: θ^i = 2^{(n/N)i} is a power of two, so weighting is a shift. Unweighting by θ^{−k} = 2^{−(n/N)k} is also a shift (a negative shift is a shift the other way, mod the order 2n). The whole negacyclic apparatus costs only shifts on top of the cyclic NTT.

Let me make this concrete on a small worked instance to be sure the weighting is right, the way I'd actually trust it. Suppose I'm doing a negacyclic length-8 convolution in the ring mod 2^16 + 1, so n = 16, N = 8, θ = 2^{n/N} = 2^2 = 4 — wait, let me recompute, θ = 2^{16/8} = 2^2 = 4, and ω = θ² = 2^4 = 16. The weights θ^i for i = 0..7 are 4^0, 4^1, …, 4^7 = 1, 4, 16, 64, 256, 1024, 4096, 16384, each mod (2^16+1). I multiply a_i by these (shifts), run the ordinary 8-point NTT with ω = 16 (more shifts), multiply pointwise, inverse-NTT, then divide each output position k by θ^k (the inverse weights), and reduce mod 2^16+1. The result is ab mod (2^16+1). The negative wrap is automatic. Good — the bookkeeping closes.

There's one more thing the inverse transform needs: division by N. The inverse NTT formula has a 1/N factor. Is that cheap and exact here? N = 2^k, so 1/N = 2^{−k}. And 2 is invertible mod 2^n+1: since 2·2^{2n−1} = 2^{2n} ≡ 1, we have 2^{−1} ≡ 2^{2n−1}. So 1/N = 2^{−k} ≡ 2^{k(2n−1)} mod (2^n+1) — yet another power of two, yet another shift. Everything in this transform, forward and inverse, twiddles and weights and the 1/N, is a shift. The only real arithmetic is adds (with the cheap mod-2^n+1 fold) and the N pointwise products. That's the whole point and it's now airtight.

So here is the algorithm taking shape. To multiply two S-bit numbers a, b: pick N = 2^k pieces, M ≈ ... bits each, a ring width n ≥ 2M + log₂N + 1 with N | n; split a, b into pieces; weight by θ^i; forward-NTT both (shifts and adds); multiply the N pairs of coefficients pointwise *mod 2^n+1*; inverse-NTT; unweight by θ^{−k}; reassemble Σ c_k 2^{Mk} with carry propagation, reduced mod the appropriate Fermat modulus. For the top level, where I want the honest full product and not a residue, I either zero-pad (fill only N/2 pieces, so the negacyclic/cyclic wrap lands in zeros and I read off the full acyclic product) or pick N large enough that 2^{MN}+1 exceeds the true product so reducing mod it changes nothing.

Now recurse. The N pointwise products are multiplications of n-bit numbers mod 2^n+1 — the exact same problem, smaller. Apply the method to each, choosing the inner parameters N', M' with M'·N' = n so they nest. Keep recursing until n drops below a threshold where a native machine multiply (or Karatsuba/Toom) is simply faster than setting up another transform; there, stop and multiply directly.

Let me work out the cost and find the right way to choose the piece size, because this is where the famous bound comes from. Let T(n) be the cost of multiplying n-bit numbers. One level does: forward and inverse transforms — Θ(N log N) ring operations, each ring operation a shift/add on n/N... let me be careful and count in bits. With N pieces and ring width Θ(M) ≈ Θ(n/N)... actually it's cleanest to track it as: the transforms touch N coefficients across log N stages, each coefficient Θ(M)-bit, so Θ(N · M · log N) = Θ(n log N) bit operations for the shifts-and-adds, plus the carry reassembly Θ(n + (n/M)·log n). And then N pointwise products each of size Θ(M) recursing. Writing ℓ for the piece size, the recurrence is

  T(n) = (n/ℓ)·T(Θ(ℓ)) + Θ(n log(n/ℓ)).

Now, how should I choose ℓ? Suppose the inner instances have size Θ(ℓ) = Bℓ for some constant B > 1 (the ring width is a constant factor bigger than the piece, because n ≥ ~2M). Unrolling the recursion:

  T(n) = n log(n/ℓ₁) + (n/ℓ₁)·Bℓ₁·log(Bℓ₁/ℓ₂) + (n/ℓ₁)(Bℓ₁/ℓ₂)·Bℓ₂·log(Bℓ₂/ℓ₃) + …
       = n[ log(n/ℓ₁) + B log(Bℓ₁/ℓ₂) + B² log(Bℓ₂/ℓ₃) + … ].

I want to choose the sequence ℓ₁, ℓ₂, … to keep this sum small. Try ℓᵢ = n^{(1/B)^i} — geometric decay of the *exponent*. Substitute log(n/ℓ₁) = (1 − 1/B) log n, and log(Bℓᵢ/ℓ_{i+1}) = log B + ((1/B)^i − (1/B)^{i+1}) log n. The log-B terms sum to (log B)(1 + B + B² + …) over the levels, and the log-n terms give n·(1−1/B)·log n·(1 + B·(1/B) + B²·(1/B)² + …) = n log n · (number of levels worth of (1−1/B)). The key quantity is the number of levels: ℓᵢ shrinks to a constant when (1/B)^i log n ≈ 1, i.e. when B^i ≈ log n, i.e. at i = Θ(log log n) levels. So the dominant transform cost Θ(n log n) is paid Θ(log log n) times:

  T(n) = Θ(n log n log log n).

There it is. The log n is the FFT; the log log n is the number of times I recurse before the pieces become constant-size. And every step is exact integer arithmetic, because the root of unity is a power of two in a Fermat ring and nothing ever leaves the integers. That's the whole thing: multiplication is convolution; convolution is a pointwise product between number-theoretic transforms over Z/(2^n+1) with 2 as the root of unity, so the transform is just shifts; the negacyclic weighting makes the result land mod a Fermat modulus so the pointwise products are smaller instances of the same problem; recurse, and the recursion depth log log n times the FFT cost n log n gives n log n log log n.

(One free optimization I notice while I'm here: in Z/(2^n+1) with n even, the element 2^{3n/4} − 2^{n/4} squares to 2 — call it √2 — and it's a root of unity of order 4n instead of 2n, since (√2)^{4n} = 2^{2n} ≡ 1 and no smaller power is 1. That lets the transform length N divide 4n instead of 2n, doubling the available length for the same ring width, and multiplying by √2 costs just two shifts and a subtract. A handy way to squeeze the parameters, though the asymptotics don't need it.)

Let me write it down as code, in the structure a real implementation uses — split, weight, forward NTT by shifts, recursive pointwise multiply, inverse NTT, unweight, carry — and check it against the schoolbook product.

```python
# Arithmetic in R = Z/(2^n + 1)Z.  In R: 2^n == -1, so 2 has order 2n.
def reduce_mod(x, n):
    return x % ((1 << n) + 1)

def shift_mod(x, s, n):
    # multiply by 2^s in R == left-shift then fold (2^n == -1).  This IS the twiddle.
    return reduce_mod(x << (s % (2 * n)), n)

def ntt(a, n, inverse=False):
    # length-N cyclic NTT, omega = 2^(2n/N); every '* w^j' is a shift.
    N = len(a)
    # bit-reversal permutation
    j = 0
    for i in range(1, N):
        bit = N >> 1
        while j & bit: j ^= bit; bit >>= 1
        j ^= bit
        if i < j: a[i], a[j] = a[j], a[i]
    length = 2
    while length <= N:
        step = (2 * n) // length                    # omega for this stage
        if inverse: step = (2 * n) - step            # omega^{-1} = 2^(2n-step)
        for start in range(0, N, length):
            w = 0
            for i in range(length // 2):
                u = a[start + i]
                v = shift_mod(a[start + i + length // 2], w, n)
                a[start + i]               = reduce_mod(u + v, n)   # butterfly
                a[start + i + length // 2] = reduce_mod(u - v, n)
                w += step
        length <<= 1
    if inverse:                                       # 1/N = 2^{-k} = 2^{k(2n-1)}
        k = N.bit_length() - 1
        for i in range(N): a[i] = shift_mod(a[i], k * (2 * n - 1), n)
    return a

def ss_mul_mod(a, b, K, M, n):
    # a*b mod (2^(K*M)+1): K=2^k pieces of M bits, ring 2^n+1; needs K|n, n>=2M+log2 K+1.
    pa = [(a >> (M * i)) & ((1 << M) - 1) for i in range(K)]
    pb = [(b >> (M * i)) & ((1 << M) - 1) for i in range(K)]
    theta = n // K                                    # 2K-th root: theta^K = 2^n == -1
    A = [shift_mod(pa[i], theta * i, n) for i in range(K)]   # negacyclic weights
    B = [shift_mod(pb[i], theta * i, n) for i in range(K)]
    ntt(A, n); ntt(B, n)
    C = [pointwise_mul(A[i], B[i], n) for i in range(K)]      # the only real products
    ntt(C, n, inverse=True)
    C = [shift_mod(C[i], (2 * n) - (theta * i) % (2 * n), n) for i in range(K)]  # unweight
    KM, res = K * M, 0
    for i in range(K):
        ci = C[i]
        if ci > (1 << (n - 1)): ci -= (1 << n) + 1     # negacyclic coeffs may be negative
        res += ci << (M * i)
    return res % ((1 << KM) + 1)

RECURSE_BITS = 1 << 14
def pointwise_mul(x, y, n):
    if n <= RECURSE_BITS:
        return reduce_mod(x * y, n)                   # base case: stop recursing
    k = max(2, n.bit_length() // 2); K = 1 << k       # recurse: same routine, smaller n
    while n % K: k -= 1; K = 1 << k
    M = n // K
    np = 2 * M + K.bit_length() + 1
    np = ((np + K - 1) // K) * K                       # round so K | n'
    return ss_mul_mod(x, y, K, M, np) % ((1 << n) + 1)

def ssa_multiply(a, b):
    # full product via zero-padded negacyclic transform (data fills lower K/2 pieces).
    if a == 0 or b == 0: return 0
    S = max(a.bit_length(), b.bit_length())
    k = max(2, (2 * S).bit_length().bit_length() + 2); K = 1 << k
    M = (S + (K // 2) - 1) // (K // 2)
    np = 2 * M + K.bit_length() + 1
    np = ((np + K - 1) // K) * K
    return ss_mul_mod(a, b, K, M, np)
```

So the causal chain, end to end: a product of two integers is the convolution of their digit sequences; a convolution is a pointwise product sandwiched between Fourier transforms, and the FFT makes those transforms Θ(N log N); doing the transform over the Fermat ring Z/(2^n+1) instead of over C makes it exact and — because 2^n ≡ −1 means 2 (and its powers θ = 2^{n/N}, ω = 2^{2n/N}) are roots of unity that are powers of two — makes every twiddle, weight, and 1/N a mere bit-shift, so the transform is multiplication-free and the only genuine multiplications are the N pointwise products; weighting the inputs by θ^i turns the cyclic transform into a negacyclic one, so the output lands mod a Fermat modulus and the inner products are smaller instances of the very same problem; recursing on them, with piece sizes chosen so the exponent decays geometrically, bottoms out after Θ(log log n) levels, and Θ(log log n) levels times the Θ(n log n) transform per level gives the exact-arithmetic bound Θ(n log n log log n).
