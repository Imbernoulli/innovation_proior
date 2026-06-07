# Synthesis — Schönhage–Strassen integer multiplication

## The problem (pre-method)
Multiply two n-bit integers. Schoolbook: O(n^2) bit ops. Karatsuba (1962): split each into
2 halves, 3 half-products via (a0+a1)(b0+b1)=a0b0+a1b1+(a0b1+a1b0), recurse ⇒ O(n^{log2 3})=O(n^1.585).
Toom-Cook (Toom 1963, Cook 1966): split into k+1 parts, view as degree-k polynomials, evaluate at
2k+1 points, pointwise multiply, interpolate ⇒ O(n^{1+ε}) with ε→0 as k grows but constant blows up.
Pain point: every "split into d parts" scheme multiplies two degree-d polynomials with Θ(d^2)
coefficient products (Karatsuba/Toom shave that to ~2d+1 *evaluations* but interpolation cost +
the per-coefficient recursion keeps the exponent above 1). We want the multiplication of two
length-N digit sequences — i.e. their CONVOLUTION — done in near-linear coefficient operations.

## Leap 1: multiplication = convolution = pointwise product after a transform
Write a = Σ a_i x^i, b = Σ b_j x^j at x = 2^w (w = word/piece bit-size). Product
c = Σ_k (Σ_{i+j=k} a_i b_j) x^k. The coefficient sequence c_k is the (acyclic) convolution a*b.
Convolution theorem: a DFT diagonalizes convolution. DFT(a*b) = DFT(a) · DFT(b) pointwise;
so c = IDFT( DFT(a)·DFT(b) ). With an FFT, the transform is O(N log N) ring operations instead
of O(N^2). This is the key structural move: turn the Θ(N^2) coefficient products into N
*pointwise* products plus three near-linear transforms.

## Wall 1: floating-point DFT is inexact
Classic FFT uses ω = e^{2πi/N} ∈ C ⇒ rounding error; for exact integer results we'd need huge
precision, and the error analysis is painful. We want EXACT arithmetic.

## Leap 2: do the FFT in a finite ring — a Number-Theoretic Transform (NTT)
We only need a ring R with a primitive N-th root of unity ω (ω^N=1, ω^r-1 invertible for 0<r<N)
and N invertible (for the 1/N in the inverse). Then the same butterfly works, exactly, over R.
Choose R = Z/(2^n + 1)Z. In this ring 2^n ≡ -1, so 2^{2n} ≡ 1: 2 has order 2n. Hence if N=2^k | 2n,
ω = 2^{2n/N} is a primitive N-th root of unity — AND IT IS A POWER OF TWO.

## Leap 3 (the cheapness): the root of unity is 2, so transform multiplications are bit-shifts
In the FFT every twiddle multiplication is "× ω^j" = "× 2^{(2n/N)j}" = a left shift by that many
bits, reduced mod 2^n+1 (and 2^n ≡ -1 makes the wrap-around a cheap negate-and-carry). So the
N log N ring "multiplications" of the FFT cost only O(n) bit ops EACH as shifts, never a real
multiply. The only genuine multiplications left are the N pointwise products of n-bit residues.
This is what makes the transform "cheap": multiplication-free except at the pointwise stage.

## Wall 2: a single big NTT is not enough / piece sizing
If we transform all n bits as one block we'd need a ring with an n-th root of unity of simple form,
which forces 2^n+1 huge and the inner products n-bit — no gain. Fix: chunk. Split the S-bit inputs
into N = 2^k pieces of M = S/N bits each (treat as a degree-(N-1) polynomial in x=2^M). Now the
pieces are small; the transform length is N; the inner pointwise products are on Θ(M)-bit numbers.

## Leap 4: negacyclic (negative-wrapped) convolution gives mod 2^n+1 "for free" and enables recursion
Cyclic convolution of length N gives c mod (x^N - 1) i.e. result mod 2^{MN}-1 (Mersenne). We instead
want C(x)=A(x)B(x) mod (x^N + 1) — the *negacyclic* convolution — because then evaluating at x=2^M
gives ab mod (2^{MN}+1), a Fermat-type modulus that matches the ring used recursively. To realize
mod (x^N+1) we need ω with ω^N = -1, i.e. a 2N-th root of unity: 2^{n/N} (since (2^{n/N})^N = 2^n ≡ -1).
Equivalently, apply a weight θ^i = (2^{n/N})^i to a_i (and b_i) before a plain length-N cyclic NTT,
then unweight by θ^{-i} after the inverse. The weighting turns the cyclic transform into the
negacyclic one (Discrete Weighted Transform, DWT). Why bother: the pointwise products are then taken
mod 2^n+1, exactly the shape that lets SSA call ITSELF on those inner products with no zero-padding —
the levels nest perfectly (every level is "multiply mod 2^something+1").

## Parameter constraints (must hold for correctness)
- Pieces: N = 2^k pieces, M = S/N bits each (S = bit length, pad to power-of-two-times-M).
- Inner ring modulus 2^n+1: the convolution coefficients c_k = Σ a_i b_j (over i+j≡k) are sums of at
  most N products of M-bit numbers, so c_k < N·2^{2M}, i.e. need n ≥ 2M + log2(N) + 1 so residues fit
  in Z/(2^n+1)Z without overflow. (Kruppa: n ≥ 2M + log2(#) + 1.)
- Root-of-unity divisibility: need N | 2n (cyclic ω = 2^{2n/N}) and, for the weight, N | n
  (θ = 2^{n/N}). So choose n a multiple of N.
- Inverse needs N^{-1} mod 2^n+1; since N=2^k and 2 is a unit (2·2^{...}≡±1), N^{-1}=2^{n_? } reduces
  to a shift/negate too. Concretely 1/2 ≡ 2^{n-1}·(something) — handled as a shift mod 2^n+1.

## Recursion + complexity
Top level: to get the FULL product (not mod anything) use an acyclic convolution (zero-pad so only
N/2 pieces carry data) OR pick N large enough that ab < 2^{MN}+1 and use negacyclic mod 2^{MN}+1 with
MN ≥ 2S. The N pointwise products are multiplications of Θ(n)=Θ(M)-bit numbers → recurse with SSA
(choosing inner N', M' with M'N' = n). Recurrence (Filmus' form, ℓ ≈ piece size):
  T(n) = (n/ℓ) · T(Θ(ℓ)) + Θ(n log(n/ℓ)).
Choosing ℓ_i = n^{(1/B)^i} (B>1 the constant in Θ(ℓ)=Bℓ), the recursion depth is Θ(log log n)
(stops when ℓ becomes constant, i.e. B^i ≈ log n), and each level contributes Θ(n log n) transform
cost ⇒ total Θ(n log n log log n). The log log n is the number of recursion levels.

## √2 trick (optimization, optional)
In Z/(2^n+1)Z, when n is even, √2 = 2^{3n/4} - 2^{n/4} satisfies (√2)^2 ≡ 2, giving a root of unity of
order 4n instead of 2n — lets you double the transform length (N | 4n) for the same ring, ~ a free
factor, multiplication by √2 is two shifts and an add. (Kruppa, Gallot.)

## Design-decision → why table
- Ring 2^n+1 (not 2^n-1): 2^n ≡ -1 gives a 2N-th root (needed for negacyclic), Fermat modulus matches
  recursion; Mersenne 2^n-1 only gives cyclic (order n) and mod 2^n-1 result.
- ω a power of 2: makes every FFT twiddle a bit-shift → transform is multiplication-free (the crux).
- Negacyclic instead of acyclic at inner levels: result already mod 2^n+1, so recursion nests with no
  padding; weighting cost is O(N) shifts, cheap.
- Split into 2^k pieces: power-of-two length ⇒ radix-2 FFT; balances transform length N vs inner size M.
- n ≥ 2M+log2 N+1: prevents coefficient overflow in the ring (the convolution sums must fit).
- Recurse on pointwise products vs base-case schoolbook/Toom: below a cutoff the asymptotics lose to
  the recursion overhead, so stop and use Toom/Karatsuba/schoolbook.

## Canonical implementation reference
- GMP `mpn_mul_fft` (mpn/generic/mul_fft.c) — production SSA. Structure: split, forward fft (shifts),
  pointwise mul (recursive), inverse fft, carry/normalize. Our illustrative Python mirrors this
  structure: split → weight → NTT (shifts) → pointwise (recursive SSA / base mul) → INTT → unweight →
  carry-propagate.
- Filmus notes (Technion) give the exact complexity recurrence and NTT-over-2^n+1 with ω=2.
- Kortekaas "SSA description" gives the worked negacyclic/DWT weighting example (mod 2^16+1, θ=2^2).
- Kruppa (INRIA) slides give the parameter constraints n ≥ 2M+log2(#)+1 and the recursive M'#'=n.
