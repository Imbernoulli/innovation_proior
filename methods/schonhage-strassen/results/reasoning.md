I want to multiply two enormous integers — n bits each, where n is in the millions — and I want it to be fast and exact. Schoolbook is Θ(n²): for length-N digit sequences I do N² little products. Karatsuba shaves that to n^{1.585} by computing the two cross terms a₁b₀+a₀b₁ with one multiply instead of two, recursing; Toom-Cook generalizes it — cut into k+1 pieces, treat each number as a degree-k polynomial, evaluate at 2k+1 points, multiply pointwise, interpolate — and pushes the exponent down toward 1 as k grows. But the exponent never *reaches* 1, and the hidden constant explodes as k grows, because the evaluation-and-interpolation machinery and the recombination overhead grow with the number of pieces. So Toom-Cook is a sequence of methods crawling toward exponent 1 but never arriving. I want to actually arrive — near n log n, not n^{1+ε}.

Let me strip the problem down to what it really is. If I write a = Σ_i a_i 2^{wi} and b = Σ_j b_j 2^{wj}, with each a_i, b_j a w-bit digit, then ab = Σ_k (Σ_{i+j=k} a_i b_j) 2^{wk}, *before* I worry about carries. So the digit sequence of the product, pre-carry, is c_k = Σ_{i+j=k} a_i b_j. Stare at that. That is exactly the convolution of the two digit sequences. The carries are a cheap linear cleanup afterward. So the whole problem is: convolve two length-N sequences cheaply, then propagate carries. The cost of multiplication *is* the cost of convolution.

And there's a classical fact about convolution I keep circling back to: it diagonalizes under the Fourier transform. The convolution theorem. If I take the DFT of each sequence, â = DFT(a), b̂ = DFT(b), then the DFT of the convolution is the *pointwise* product â · b̂. So a∗b = IDFT(â·b̂). The reason that's exciting: the convolution itself is the Θ(N²) operation, but the DFT — which turns it into N independent pointwise multiplications — can be computed in Θ(N log N) by the Fast Fourier Transform. Cooley and Tukey: split the transform into its even-indexed and odd-indexed halves, each of which is a half-size transform, glue with a "butterfly," recurse; T(N) = 2T(N/2) + Θ(N) = Θ(N log N). So three FFTs (two forward, one inverse) plus N pointwise products. If those pointwise products were free, I'd have an Θ(N log N) multiplication. They're not free, but hold that thought.

The first attempt is obvious: use that transform sandwich directly and see where the arithmetic breaks.

The obvious way is the complex FFT: ω = e^{2πi/N}, do everything over the complex numbers. Let me actually push this through, because even if it is inexact it tells me exactly where the round-off bites. In the classical parameterization, I split the N-bit inputs into 2^{r-1} nonzero chunks of l bits each and zero-pad to a length-2^r cyclic transform, so the honest acyclic convolution fits. A coefficient c_k is a sum of at most 2^{r-1} products of l-bit chunks, hence |c_k| < 2^{2l+r}. To recover c_k by rounding, the final error must be below 1/2. The transform size and the coefficient magnitudes amplify s-bit fixed-point errors, and the safe guard condition is

  s >= 2l + 2r + lg N + const.

Now count the cost if a complex butterfly multiplication at this precision is reduced to integer multiplications supplied by the previous method V_m. The real operands have Θ(l+r) useful bits plus guard room, and a complex multiplication costs a constant number of such real multiplications, so

  M_{m+1}(N) = O(2^r((1+r)^2 + r M_m(3(l+r)))).

If I call this transform exponent n instead of r, those two bounds read s >= 2l + 2n + lg N + const and M_{m+1}(N)=O(2^n((1+n)^2+n M_m(3(l+n)))). The (1+r)^2 term is the fixed-precision shift/add/rounding work around each transform position, and the r factor is the number of FFT stages that use these guarded coefficient multiplications. With the first FFT construction this gives M_1(N)=O(N(lg N)^2). Feeding that bound back into the same recurrence gives M_2(N)=O(N lg N (lg lg N)^2). One more nesting lets the remaining squared iterated-log factor be absorbed into (lg lg N)^ε for any fixed ε>0, so three nestings give O(N lg N (lg lg N)^{1+ε}). So the convolution route does work and does get essentially to N lg N — but with one ugly word hanging over the whole thing: round-off. The entire apparatus rests on guard bits and an error bound that I have to *trust*. I want exact, correct-to-the-last-bit. Floating-point roots of unity are not exact. So this is not yet the answer; if I'm going to keep the convolution structure I have to kill the round-off at its source, and ideally that same fix would also shave away the last (lg lg N)^ε.

Where does the round-off come from? Only from ω being a transcendental complex number. But look again at what the FFT actually demands of ω: it needs ω^N = 1, it needs ω^r − 1 to be invertible for 0 < r < N (so the geometric-series cancellations that make the transform invert actually work), and it needs N itself to be invertible (for the 1/N in the inverse transform). That's *all* it needs. Nothing in that list says ω must be a complex number. It can be any element of any commutative ring that happens to have an N-th root of unity of the right kind. So: pick a finite ring. Then arithmetic is exact integer-mod-something arithmetic, no round-off at all. This is a number-theoretic transform.

Which ring, though? I need a primitive N-th root of unity living in it. The naive choice is a prime field Z/pZ with p ≡ 1 mod N, which does have a root of unity — exact, no round-off. But the root of unity there is some ugly residue, and multiplying by it in the butterflies is a full multiplication mod p, which is *itself* an expensive operation. I'd be paying a real multiply at every one of the N log N butterfly steps. Let me sanity-check whether that's actually fatal before discarding it: if each twiddle multiply costs a multiplication of two ~(log p)-bit numbers, and there are N log N twiddles, the transform alone costs N log N · (cost of one mod-p multiply). The whole point of the FFT was to push all the cost into N pointwise products; here I've smeared a multiply across every butterfly too, so I've put Θ(N log N) multiplications *back into the transform*. That cannot get me below n². So the prime-field NTT is exact but defeats its own purpose. I need a ring where the root of unity is not just exact but *cheap to multiply by*.

What's the cheapest thing to multiply a binary integer by? A power of two. Multiplying by 2^s is a left shift — essentially free in bit-operation count, it just relabels bit positions. So the question becomes: is there a ring whose root of unity is a power of two? When is a power of two a root of unity? I need 2^s, raised to N, to be ≡ 1 in my ring. Equivalently I need the ring to be Z modulo something that makes a power of 2 have finite order. There's a candidate sitting right there: work modulo 2^n + 1. Then 2^n ≡ −1, so 2^{2n} ≡ 1, and 2 has order exactly 2n in this ring. If the transform length N divides 2n, then ω = 2^{2n/N} has order N, since the order of 2^a is 2n/gcd(2n,a) and here a = 2n/N. So ω is a primitive N-th root of unity, and it is a power of two. Every twiddle in the FFT becomes "shift left by some multiple of 2n/N bits, then reduce mod 2^n+1." And the reduction is also cheap: any bits that spill past position n fold back with a sign flip, because 2^n ≡ −1. Shift and fold. If that all holds, the FFT has *no multiplications in it at all*, and the only genuine multiplications left in the whole convolution are the N pointwise products of residues.

I should not take that on faith — I've twice now reasoned myself into a parameter regime that quietly broke, so let me actually compute the root of unity on a tiny case. Take n = 8, ring Z/(2^8+1) = Z/257. Then 2^8 = 256, and 256 mod 257 is −1, so 2^{16} ≡ 1 and 2 has order 16. For an 8-point transform, N = 8 divides 16, so ω = 2^{16/8} = 2² = 4. I need ω^8 ≡ 1 and ω^4 ≠ 1 (primitivity). Computing: 4^8 = 2^{16} = 65536, and 65536 mod 257 — well 257·255 = 65535, so 65536 ≡ 1. Good. And 4^4 = 2^8 = 256 ≡ −1, which is not 1, so the order of 4 is genuinely 8, not a proper divisor. So 4 is a primitive 8th root of unity mod 257, and it's a power of two. The construction is real, at least here.

Now, the size problem, because I haven't been careful about magnitudes and it's about to bite. If I try to transform all n bits as a single block — N ≈ n points — then I need N | 2n with N ≈ n, fine, but each "coefficient" is a residue mod 2^n+1, an n-bit number, and the N ≈ n pointwise products are products of n-bit numbers. That's Θ(n) multiplications each of n-bit numbers, which is Θ(n²)·something. I've gone nowhere; in fact the additions alone, Θ(n) of them each Θ(n) bits across Θ(log n) stages, already cost Θ(n² log n) — worse than schoolbook. So the single-big-block NTT is a trap: making the ring as wide as the whole input destroys the win. The transform length and the coefficient width have to be decoupled.

So I must chunk. Don't make N ≈ n; make N modest and the *pieces* big. Split the S-bit input into N = 2^k pieces of M = S/N bits each, and regard the integer as a polynomial evaluated at x = 2^M: a = A(2^M) with A(x) = Σ_{i<N} a_i x^i, a_i an M-bit digit. Same for b. Now the convolution is the product of two degree-(N−1) polynomials, length N, and the pointwise products are on numbers of size about the ring width — call it n bits — which I get to choose, and which is roughly 2M, not the whole input. The transform length N and the inner size M are now two separate knobs I can tune against each other. That's the structural freedom Toom-Cook didn't have: there, the number of pieces and the recombination cost were locked together by interpolation; here the FFT decouples them.

How big does the ring 2^n+1 need to be — how large is n — so the pointwise products are computed without overflow? The coefficients of the product polynomial are c_k = Σ_{i+j=k} a_i b_j. Each a_i, b_j is an M-bit number, so each product is < 2^{2M}, and there are at most N of them in a c_k, so c_k < N·2^{2M}. For the ring Z/(2^n+1) to hold each c_k faithfully — to recover the true integer coefficient, not its residue — I need 2^n+1 > N·2^{2M}, i.e. n ≥ 2M + log₂N + 1. So pick n a bit more than 2M and I'm safe. And I'll want N | 2n for the root of unity; arrange n to be a multiple of N and that's automatic.

Now the kind of convolution matters. The FFT over Z/(2^n+1) with ω = 2^{2n/N} naturally computes a *cyclic* convolution — it gives me the product polynomial mod (x^N − 1), because the transform lives on the group Z_N and convolution there wraps around. Mod (x^N − 1) means coefficient k and coefficient k+N get added together. If I want the *full* product polynomial (degree up to 2N−2) I'd have to zero-pad: use length 2N, fill only the bottom N coefficients, leave the top N as zeros, so the wraparound lands harmlessly in the zeros. That works for the top level and gives me the honest full product. Fine.

But here's the thing I want for the recursion. Those N pointwise products — products of n-bit residues — are themselves multiplications of largish integers. I want to compute *them* by the same method, recursively. For the recursion to nest cleanly, the inner problem should have the *same shape* as the outer one: "multiply two numbers modulo 2^{something}+1." Look at what the pointwise products actually are: I'm multiplying residues that live in Z/(2^n+1). So if my method's natural output were "the product modulo 2^{MN}+1," then the inner multiply-mod-2^n+1 is just a smaller instance of the exact same routine, with no zero-padding waste. The levels would fit together like nested dolls, every level being "multiply mod a Fermat-type modulus." That's a strong reason to *prefer* the modulus-shaped output if I can get it.

Can I make the transform produce a product mod (x^N + 1) instead of mod (x^N − 1)? That's the *negacyclic* convolution — negative-wrapped, where coefficient k+N gets *subtracted* from coefficient k rather than added. At x = 2^M that gives me ab mod (2^{MN} + 1) — exactly the Fermat-type modulus that matches the ring. So the recursion wants the negacyclic, not the cyclic, convolution. Now I have to actually build it.

What does negacyclic require? Mod (x^N + 1) instead of mod (x^N − 1). In the polynomial ring, x^N ≡ −1. So I need a transform whose root of unity ω satisfies ω^N = −1 — a *2N-th* root of unity, not an N-th root. Do I have one as a power of two? Try θ = 2^{n/N}. Then θ^N = 2^n, and 2^n ≡ −1 in Z/(2^n+1). So θ^N = −1: θ is a 2N-th root of unity and still a power of two. (This needs N | n, slightly stronger than the N | 2n I had before; arrange n to be a multiple of N and both hold.)

How do I actually fold θ into the machinery? I could rebuild the FFT around a 2N-th root, but there's a cleaner trick I half-remember and want to check rather than just invoke. Take the plain length-N cyclic NTT (root ω = θ² = 2^{2n/N}) that I already have, and *weight* the inputs before transforming: replace a_i by θ^i a_i and b_j by θ^j b_j. The claim is that the cyclic convolution of the weighted sequences, unweighted afterward by θ^{−k}, equals the negacyclic convolution of the originals. Let me see *why* it should, by watching the wraparound term, before I trust it. In the cyclic convolution a coefficient at position k+N folds onto position k unchanged. With the weights, that wrapped term carries a factor θ^{(k+N)} = θ^k · θ^N = θ^k · (−1); after I divide the whole position-k result by θ^k to unweight, the wrapped contribution keeps an extra −1 relative to the unwrapped one. So the wrap that *added* now effectively *subtracts* — which is precisely the negacyclic rule mod (x^N + 1). That's the discrete weighted transform, and the argument is suggestive, but a sign error in the wrap-vs-unwrap bookkeeping is exactly the kind of thing this argument could hide. I'm going to run a full small instance through the actual transform and compare against a negacyclic convolution I compute independently by hand.

But first: the weights have to be as cheap as the twiddles or the whole gain evaporates. θ^i = 2^{(n/N)i} is a power of two, so weighting is a shift. Unweighting by θ^{−k} = 2^{−(n/N)k} is also a shift (a negative shift is a shift the other way, taken mod the order 2n). So the negacyclic apparatus costs only shifts on top of the cyclic NTT — provided the bookkeeping is actually correct.

Let me do that check concretely. Take N = 4 and the ring Z/(2^16+1) = Z/65537, so n = 16, θ = 2^{n/N} = 2^4 = 16, ω = θ² = 2^8 = 256. First confirm θ is the right kind of root: θ^N = 16^4 = 2^16 = 65536 ≡ −1 mod 65537. Good. Now pick two sequences small enough to convolve by hand: a = [1,2,3,4], b = [5,6,7,8] (these are the coefficient digits; their actual size is irrelevant to the bookkeeping check). The honest full polynomial product A(x)B(x) has coefficients

  full = [5, 16, 34, 60, 61, 52, 32]

(e.g. the x³ coefficient is 1·8+2·7+3·6+4·5 = 8+14+18+20 = 60). The negacyclic reduction mod (x⁴+1) subtracts the wrapped half: c_k = full[k] − full[k+4], giving

  c₀ = 5 − 61 = −56,  c₁ = 16 − 52 = −36,  c₂ = 34 − 32 = 2,  c₃ = 60 − 0 = 60,

so the negacyclic convolution I'm aiming for is [−56, −36, 2, 60]. Now run the weighted-transform pipeline: weight a_i by θ^i to get [1, 2·16, 3·256, 4·4096] = [1, 32, 768, 16384] mod 65537 (and b likewise), forward-NTT both with ω = 256, multiply pointwise mod 65537, inverse-NTT, then unweight position k by θ^{−k}. Carrying that through (the residues > 2^{15} read as negative), the output comes out as **[−56, −36, 2, 60]** — identical to the by-hand negacyclic result. So the weighting trick really does convert cyclic into negacyclic, and the unweight-by-θ^{−k} sign bookkeeping is right; the −1 from θ^N is landing on exactly the wrapped terms and nowhere else. Now I trust it.

There's a further halving hiding in the structure that I'd be foolish to leave on the table. If I computed the negative-wrapped product by first embedding it in a length-2N cyclic transform, I would evaluate at all 2N powers of a primitive 2N-th root and do 2N pointwise products. But my inputs are honest integer-coefficient polynomials and I only want the factor modulo x^N+1. The roots of x^N+1 are the odd powers θ, θ³, ..., θ^{2N−1}; the even powers are roots of x^N−1 and describe the other CRT factor. So the even-indexed values are not needed for this product at all. The weighting identity is exactly the compact length-N way to evaluate at those odd powers, because θω^j = θ(θ²)^j = θ^{2j+1}. I keep only the odd-indexed products, N of them instead of the 2N a zero-padded cyclic transform would have used. Not an asymptotic miracle, but it halves the expensive recursive multiplications at every level.

There's one more thing the inverse transform needs: division by N. The inverse NTT formula has a 1/N factor. Is that cheap and exact here? N = 2^k, so 1/N = 2^{−k}, and I need 2 to be invertible mod 2^n+1. From 2·2^{2n−1} = 2^{2n} ≡ 1, the inverse of 2 is 2^{2n−1}. Let me confirm on a case rather than wave at it: n = 8, modulus 257, 2^{2n−1} = 2^{15} = 32768; 32768 mod 257 is 32768 − 127·257 = 32768 − 32639 = 129; and 2·129 = 258 ≡ 1 mod 257. So 2^{−1} ≡ 129 there, exactly as 2^{2n−1} predicts. Hence 1/N = 2^{−k} ≡ 2^{k(2n−1)} mod (2^n+1) — yet another power of two, yet another shift. So everything in this transform — forward and inverse, twiddles and weights and the 1/N — is a shift. The only real arithmetic is adds (with the cheap mod-2^n+1 fold) and the N pointwise products. That is the lever the whole method turns on, and now I've checked each moving part rather than asserted it.

So here is the algorithm taking shape. To multiply two S-bit numbers a, b: pick N = 2^k pieces and choose M so the data fit in the lower half when I want a full product; choose a ring width n ≥ 2M + log₂N + 1 with N | n; split a, b into M-bit pieces; weight by θ^i; forward-NTT both by shifts and adds; multiply the N pairs of coefficients pointwise mod 2^n+1; inverse-NTT; unweight by θ^{−k}; reassemble Σ c_k 2^{Mk}, letting carries propagate, and reduce mod the appropriate Fermat modulus. For the top level, where I want the honest full product and not a residue, I zero-pad by filling only the lower N/2 pieces and choose MN at least twice the input bit length, so the negative wrap lands in zeros and 2^{MN}+1 is larger than the true product.

Now recurse. The N pointwise products are multiplications of n-bit numbers mod 2^n+1 — the exact same problem, smaller. Apply the method to each, choosing the inner parameters N', M' with M'·N' = n so they nest. Keep recursing until n drops below a threshold where a native machine multiply (or Karatsuba/Toom) is simply faster than setting up another transform; there, stop and multiply directly.

Let me work out the cost in the modular form, because that is the clean recursive object. Suppose the current task is multiplication modulo 2^q+1. Choose q0 = Θ(q^{1/2}) and T = 2q/q0, arranged so T divides q0. Split each input into T chunks of q0/2 bits. Then x = 2^{q0/2} satisfies x^T = 2^q ≡ −1 modulo 2^q+1, so the task becomes multiplication in R[x]/(x^T+1), where R = Z/(2^{q0}+1)Z. Here I have to keep two correctness stories separate. At the top level, if I want actual integer coefficients back, I need the safety width n ≥ 2M + log₂N + 1. Inside the recursive modular problem I am not trying to recover large integer coefficients in each slot; the coefficients are residues in R, and the pointwise products are exactly products modulo 2^{q0}+1.

The transform over R touches T coefficients, each q0 bits wide, through O(log T) stages; since Tq0 = 2q, the shift/add work is O(q log q). The pointwise stage has exactly T products in R, because I evaluate only at the odd roots of x^T+1, and each product is a smaller modular multiplication of size q0. So if F(q) is the cost of multiplication modulo 2^q+1,

  F(q) <= (2q/q0) F(q0) + O(q log q),      q0 = Θ(q^{1/2}).

Now divide by q log q and check the per-level contribution explicitly. The recursive term is (2q/q0)·F(q0); writing F(q0) ≈ q0 log q0 inside it, that's (2q/q0)·q0 log q0 = 2q log q0, and dividing by q log q gives 2 log q0 / log q = 2·(½ log q)/log q = 1. So each level adds exactly a constant (about 1) to F(q)/(q log q), not a growing amount — the √-step is what makes that ratio constant rather than logarithmic. The size falls q → q^{1/2} → q^{1/4} → …, reaching the base case when q^{1/2^d} is constant, i.e. when 2^d = Θ(log q). So the depth is Θ(log log q): for q = 2^16 the √-iteration takes 3 steps and log₂log₂q = log₂16 = 4; for q = 2^256 it takes 7 steps and log₂log₂q = log₂256 = 8 — the depth tracks log log q as claimed. Depth Θ(log log q) times the constant-per-level contribution gives

  F(q) = O(q log q log log q).

The log q is the FFT; the log log q is the number of exact modular levels before the pieces become base-case size. And every step is exact integer arithmetic, because the root of unity is a power of two in a Fermat ring and nothing ever leaves the integers — which is the property I set out to get and have now traced through each piece of machinery.

(One free optimization I notice while I'm here. In Z/(2^n+1) with n divisible by 4, take s = 2^{3n/4} − 2^{n/4}. Then s² = 2^{3n/2} − 2·2^n + 2^{n/2}; using 2^n ≡ −1 this is −2^{n/2} − 2(−1) + 2^{n/2} = 2, so s is a square root of 2 in the ring. Let me check on n = 8, modulus 257: s = 2^6 − 2^2 = 64 − 4 = 60, and 60² = 3600 = 14·257 + 2 = 3598 + 2, so 60² ≡ 2 mod 257. Since s² = 2 has order 2n, s itself has order 4n — confirm: s^{2n} = 2^n ≡ −1 ≠ 1, so the order is not 2n but 4n. That lets the transform length N divide 4n instead of 2n, doubling the available length for the same ring width, and multiplying by √2 costs just two shifts and a subtract. A handy way to squeeze the parameters, though the asymptotics don't need it.)

Let me write it down in code in the structure I just derived — split, weight, forward NTT by shifts, recursive pointwise multiply, inverse NTT, unweight, carry — and check it against the schoolbook product across many bit-lengths.

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

I ran this against the schoolbook product `a*b` on a few hundred random pairs with bit-lengths scattered from 1 up through several thousand, with random signs, and then on single pairs at 10⁴, 5·10⁴, and 10⁵ bits: every product matched exactly. So the negacyclic weighting, the shift-only transform, the recursive pointwise multiply, and the signed-coefficient reassembly all compose correctly — the small [−56,−36,2,60] check scales up to the full routine.

So the causal chain, end to end: a product of two integers is the convolution of their digit sequences; a convolution is a pointwise product sandwiched between Fourier transforms, and the FFT makes those transforms Θ(N log N); doing the transform over the Fermat ring Z/(2^n+1) instead of over C makes it exact and — because 2^n ≡ −1 means 2 (and its powers θ = 2^{n/N}, ω = 2^{2n/N}) are roots of unity that are powers of two — makes every twiddle, weight, and 1/N a mere bit-shift, so the transform is multiplication-free and the only genuine multiplications are the N pointwise products; weighting the inputs by θ^i turns the cyclic transform into a negacyclic one (checked digit-by-digit on N=4), so the output lands mod a Fermat modulus and the inner products are smaller instances of the very same problem; recursing on them, with piece sizes chosen so the exponent decays geometrically, bottoms out after Θ(log log n) levels, and Θ(log log n) levels times the Θ(n log n) transform per level gives the exact-arithmetic upper bound O(n log n log log n).
