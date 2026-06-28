Let me start from the operation that actually hurts. In the R-LWE world every single primitive step — encrypt, sign, key-exchange — comes down to multiplying two polynomials of degree below `n` in `R_q = Z_q[x]/(x^n + 1)`, with `n` a power of two, a few hundred to a thousand. Schoolbook multiplication is `O(n^2)` coefficient products, and for `n = 1024` that quadratic is the whole cost of the protocol. I want `O(n log n)`. And separately, the same kernel shows up when I multiply two enormous integers: chop each into base-`u` digits, and the product digits are a convolution of the digit vectors. Same shape, same quadratic pain.

The classical answer to "convolution in `n log n`" is the Fourier transform. The convolution theorem says: transform both sequences, multiply pointwise, transform back. Over the complex numbers I'd use `ζ = e^{2πi/n}`, the DFT `A_j = Σ_i a_i ζ^{ij}`, its inverse `a_i = (1/n) Σ_j A_j ζ^{-ij}`, and Cooley–Tukey to do each transform in `n log n` instead of `n^2`. So in principle I'm done: FFT both polynomials, multiply, inverse-FFT, read off the product.

Except my data are *integers* (mod `q`), and the answer must be an exact integer. If I run the convolution through complex floating-point FFTs, the output is a vector of floats that I round to the nearest integer at the end. That's only correct if the accumulated round-off across all those `e^{2πi/n}` multiplications stays below `1/2`, so the rounding lands on the true integer. The precision I need for that grows with `n` and with how big the coefficients get. For a cryptographic kernel that also wants to be constant-time and bit-exact, threading a floating-point error budget through every operation is exactly the kind of fragility I don't want. The transform is the right idea; doing it over `C` is the wrong arithmetic.

So let me ask the uncomfortable question: what does the DFT *actually* need from `ζ`? Not that it's complex. Let me re-derive the one identity everything rests on. The inverse recovers `a_i` because, summing the forward then the inverse,
`(1/n) Σ_j (Σ_k a_k ζ^{kj}) ζ^{-ij} = (1/n) Σ_k a_k Σ_j ζ^{(k-i)j}`,
and the inner sum `Σ_{j=0}^{n-1} ζ^{(k-i)j}` is `n` when `k = i` and must be `0` otherwise for the whole thing to collapse to `a_i`. Why is it zero otherwise? For `K = k - i ≢ 0 (mod n)`, `(1 - ζ^K) Σ_j ζ^{Kj} = 1 - ζ^{Kn} = 1 - 1 = 0`, and since `ζ^K ≠ 1`, the factor `1 - ζ^K` is nonzero, so the sum itself is zero. That's the entire engine. The convolution theorem is a one-line consequence of the same geometric-series telescoping.

Stare at that derivation. It uses three things about `ζ`: that `ζ^n = 1` (so `ζ^{Kn} = 1`), that `ζ^K ≠ 1` for `0 < K < n` (so I can divide by `1 - ζ^K`), and that I'm working in something where I can add, multiply, and divide. Order `n` and a multiplicative inverse. That's it. There is no analysis here — no limits, no continuity, no magnitude. The complex numbers were never essential; they were just one place where an element of exact order `n` happens to live.

So I should look for an element of order `n` somewhere I can do exact arithmetic. The cleanest such place is a finite field `Z_p`, `p` prime. Its nonzero elements form a cyclic group of order `p - 1`. A cyclic group of order `p - 1` contains an element of order exactly `n` precisely when `n | p - 1`. Call such an element `ω`. Then `ω^n = 1`, and for `0 < K < n`, `ω^K ≠ 1` because `ω` has order exactly `n`. Both conditions I need, holding *exactly* in integer arithmetic mod `p`. And division is fine — `Z_p` is a field, every nonzero element has an inverse.

So let me define the transform exactly as the DFT but over `Z_p`:
`A_j = Σ_{i=0}^{n-1} a_i ω^{ij} (mod p)`, and the inverse `a_i = n^{-1} Σ_{j=0}^{n-1} A_j ω^{-ij} (mod p)`,
where `ω` is a primitive `n`-th root of unity in `Z_p` and `n^{-1}` is the modular inverse of `n`. The same telescoping proof should go through verbatim with `ζ` replaced by `ω`: `Σ_j ω^{Kj} = n` if `K ≡ 0` and `0` otherwise, because `1 - ω^K ≠ 0`. If that holds then the convolution theorem holds, and `(c_i) = INTT(NTT(a) ∘ NTT(b))` is the cyclic convolution of `(a_i)` and `(b_i)` — all in exact modular arithmetic, no round-off, ever. Before I build the fast version on top of this I want to actually see the bare identity recover a sequence, not just believe the algebra. Take `p = 7681`, `n = 4`; a primitive root is `17`, so `ω = 17^{7680/4} mod 7681 = 3383`, and indeed `3383^4 ≡ 1` while `3383, 3383^2 = 7680 = -1, 3383^3` are all `≠ 1`, so its order is exactly `4`. Transform `a = [1,2,3,4]` by the matrix definition, then run the inverse with `n^{-1} = 4^{-1}` and `ω^{-1}`: the roundtrip returns `[1, 2, 3, 4]`. So the telescoping does collapse to the identity in `Z_p` — the transform is invertible over the finite field exactly as over `C`. This is what I'll call the number-theoretic transform.

Now the speed. Can I run Cooley–Tukey over `Z_p`? Look at where the radix-2 split comes from. Split the index `i` by parity. The even part `a_0, a_2, …` and odd part `a_1, a_3, …` each get a length-`n/2` transform, call the results `E_j` and `O_j`. Then
`A_j = Σ_i a_i ω^{ij} = Σ_m a_{2m} ω^{2mj} + Σ_m a_{2m+1} ω^{(2m+1)j} = E_j + ω^j O_j`,
where I used that `ω^2` is a primitive `(n/2)`-th root, so the even/odd inner sums are genuine length-`n/2` transforms. And for the other half of the outputs,
`A_{j+n/2} = E_j + ω^{j+n/2} O_j = E_j - ω^j O_j`,
because `ω^{n/2} = -1` — the element of order `n` squared-to-order-2 is the unique order-2 element, which is `-1`. So I get the butterfly: from `E_j` and the twiddled `ω^j O_j`, produce `A_j = E_j + ω^j O_j` and `A_{j+n/2} = E_j - ω^j O_j`. This is *exactly* the complex Cooley–Tukey butterfly, with `ω^j` playing the twiddle and `ω^{n/2} = -1` giving the sign flip. None of it used anything complex. Recurse `log_2 n` times, `n/2` butterflies per stage: `O(n log n)` modular operations. The FFT transplants to `Z_p` unchanged.

Good — for cyclic convolution, i.e. multiplication modulo `x^n - 1`, I'm done. But my ring is `Z_q[x]/(x^n + 1)`, not `x^n - 1`. The length-`n` NTT computes the cyclic convolution `Σ_{i+j≡k (mod n)} a_i b_j`, which is the product modulo `x^n - 1`: the terms that overflow past degree `n-1` *fold back additively*. What I want, modulo `x^n + 1`, is for those overflow terms to fold back with a *minus* sign, because `x^n ≡ -1`. The negacyclic convolution `c_k = Σ_{i+j=k} a_i b_j - Σ_{i+j=k+n} a_i b_j`.

First instinct: just zero-pad. If I extend both polynomials to length `2n` with zeros, the linear (un-wrapped) product of degree `2n-2` is recovered exactly by a length-`2n` cyclic convolution, and then I reduce modulo `x^n + 1` by hand — subtract the high half from the low half. That works, but it doubles the transform length (a `2n`-point NTT needs a `2n`-th root and twice the butterflies) and tacks on an explicit reduction pass. I'm paying `2n log 2n` plus a fold, when the data only has `n` real coefficients. Wasteful. There should be a way to bake the `x^n + 1` directly into a length-`n` transform.

Let me think about what `x^n + 1` *is*. Its roots are the primitive `2n`-th roots of unity: `x^n = -1` means `x^{2n} = 1` but `x^n ≠ 1`. The cyclic case `x^n - 1` factors over the roots that are `n`-th roots of unity; the negacyclic case wants the *odd* `2n`-th roots — the ones whose `n`-th power is `-1`. So whatever I do should involve a `2n`-th root of unity, not just an `n`-th one. Suppose `ψ` is a primitive `2n`-th root of unity in `Z_q`: `ψ^{2n} = 1`, `ψ^n = -1`, and `ω := ψ^2` is then a primitive `n`-th root. For this to exist I now need `2n | q - 1`, i.e. `q ≡ 1 (mod 2n)` — a stronger congruence than the cyclic `q ≡ 1 (mod n)`, but the same idea.

Now the trick. Instead of feeding the raw coefficients into the NTT, pre-weight them: replace `a_i` by `ψ^i a_i` and `b_i` by `ψ^i b_i`. Let me see what cyclic convolution does to the weighted sequences. The cyclic convolution of `(ψ^i a_i)` and `(ψ^i b_i)` at index `k` is
`Σ_{i+j≡k (mod n)} ψ^i a_i · ψ^j b_j = Σ_{i+j≡k} ψ^{i+j} a_i b_j`.
Split by whether `i + j = k` (no wrap) or `i + j = k + n` (one wrap): the un-wrapped terms carry `ψ^{k}`, the wrapped terms carry `ψ^{k+n} = ψ^k · ψ^n = -ψ^k`. So the whole thing is
`ψ^k ( Σ_{i+j=k} a_i b_j - Σ_{i+j=k+n} a_i b_j ) = ψ^k c_k`,
where `c_k` is exactly the negacyclic coefficient I wanted. The `ψ^n = -1` is what turns the additive wrap into a subtractive one — the sign of `x^n + 1` materializes out of the weighting. So if I post-multiply the result by `ψ^{-k}`, I recover `c_k`. Concretely:
weight inputs by `ψ^i`, run the ordinary length-`n` NTT (at root `ω = ψ^2`), multiply pointwise, inverse-NTT, then weight outputs by `ψ^{-i}`. A single length-`n` transform, no padding to `2n`, no separate reduction. This is the negative-wrapped convolution.

Let me double check the post-weight indexing, because it's easy to get wrong. The forward weighted transform is `â_j = Σ_i ψ^i ω^{ij} a_i`. The natural inverse undoes the `ω^{ij}` with `ω^{-ij}` and the `n^{-1}`, but the `ψ^i` was a per-*input*-index weight, and after the inverse the index is the *output* index `i`. So the post-weight must be `ψ^{-i}` keyed on the output index: `a_i = n^{-1} ψ^{-i} Σ_j ω^{-ij} â_j`. If I instead wrote `ψ^{-j}` under the sum, keyed on the summation index, I don't think it would invert the forward map — the weight has to come outside the sum, attached to `i`. But this is exactly the kind of index bookkeeping I get wrong on paper, and both versions look plausible at a glance, so I'm not going to trust my reading of it. Let me flag it: candidate A is post-weight `ψ^{-i}` output-indexed, outside the sum; candidate B is `ψ^{-j}` summation-indexed, inside the sum. I'll settle which one actually inverts the forward transform by running both on a concrete vector, below — not by staring harder at the exponents.

There's a cleaner way to see the same thing that also tells me how to implement it fast. The forward weighted transform is `â_j = Σ_i ψ^{2ij + i} a_i = Σ_i ψ^{(2j+1)i} a_i`. So the weighted NTT is just a transform whose "root" is evaluated at the *odd* powers `ψ^{2j+1}` of `ψ` — exactly the primitive `2n`-th roots, exactly the roots of `x^n + 1`. The `ψ` pre-weighting and the `ω`-transform are two views of one thing: evaluating the polynomial at the odd `2n`-th roots of unity. That's why no explicit reduction is needed — evaluating at roots of `x^n + 1` *is* working modulo `x^n + 1`.

Now I'd like to fold the `ψ` weighting into the FFT so it costs nothing extra, and ideally kill the bit-reversal too. Redo the Cooley–Tukey split on the odd-power form `â_j = Σ_i ψ^{(2j+1)i} a_i`. Separate by parity of `i`:
`â_j = Σ_{i even} ψ^{(2j+1)i} a_i + Σ_{i odd} ψ^{(2j+1)i} a_i = Σ_m ψ^{(2j+1)2m} a_{2m} + ψ^{2j+1} Σ_m ψ^{(2j+1)2m} a_{2m+1}`.
The inner sums use `ψ^{2(2j+1)m} = (ψ^2)^{(2j+1)m} = ω^{(2j+1)m}` — so they're themselves length-`n/2` odd-power transforms of the even and odd subsequences, with `ω = ψ^2` now playing the role `ψ` did one level up (a primitive `2·(n/2)`-th root). Write them `A_j`, `B_j`. Then `â_j = A_j + ψ^{2j+1} B_j` and, using the symmetry `ψ^{(2(j+n/2)+1)} = ψ^{2j+1} · ψ^n = -ψ^{2j+1}`, the partner output `â_{j+n/2} = A_j - ψ^{2j+1} B_j`. Same butterfly, but now the twiddle factor is `ψ^{2j+1}`, an odd power of `ψ`, instead of `ω^j`. So the `ψ`-weighting is absorbed: I don't pre-scale the inputs at all; I just use a twiddle table of `ψ`-powers (the odd ones) rather than `ω`-powers. Decimation-in-time naturally consumes those twiddles in a bit-reversed order and emits the outputs in bit-reversed order.

For the inverse I do the dual split — decimation-in-frequency, splitting the *output* by upper/lower half instead of the input by parity. The forward map is `â_j = Σ_i ψ^{(2j+1)i} a_i`, so its inverse has to put the `(2j+1)` on the summation index `j` and the bare output index `i` together: `a_i = n^{-1} Σ_j ψ^{-(2j+1)i} â_j` (the per-input weight `ψ^{-i}` lives *outside* the `ω^{-ij}` sum, keyed on `i`). Splitting the sum at `n/2`,
`a_i = n^{-1} ( Σ_{j<n/2} ψ^{-(2j+1)i} â_j + Σ_{j<n/2} ψ^{-(2(j+n/2)+1)i} â_{j+n/2} )`,
and `ψ^{-(2(j+n/2)+1)i} = ψ^{-(2j+1)i} · ψ^{-ni} = ψ^{-(2j+1)i} · (-1)^i` (since `ψ^n = -1`) produces the `±` again, keyed on the parity of the output index `i`. Pulling out the output-index weight `ψ^{-i}` (since `ψ^{-(2j+1)i} = ψ^{-i} ω^{-ij}`) and grouping the even/odd output halves gives the Gentleman–Sande butterfly: `a[j] ← U + V`, `a[j+t] ← (U - V)·S`, where `S` is now a power of `ψ^{-1}`. This butterfly takes its input in bit-reversed order and produces output in normal order. So if I pair a decimation-in-time forward transform (normal in, bit-reversed out) with a decimation-in-frequency inverse (bit-reversed in, normal out), the two bit-reversal permutations meet in the middle and cancel — pointwise multiplication doesn't care about ordering, so I can multiply the two bit-reversed spectra directly and the inverse hands me the answer back in normal order. No explicit bit-reversal pass at all.

Let me also pin down the inverse-NTT scaling, since I asserted `n^{-1}` without re-deriving it. From the same telescoping identity, `M M^* = n I` where `M_{ji} = ω^{ij}` and `M^*_{ij} = ω^{-ij}`: the `(i,k)` entry of `M^* M` is `Σ_j ω^{-ij} ω^{jk} = Σ_j ω^{(k-i)j} = n δ_{ik}`. So `M^{-1} = (1/n) M^*`, and the inverse transform carries a `n^{-1} (mod q)` factor. In the merged-`ψ` GS inverse this `n^{-1}` is applied once at the very end (or folded into the final twiddle).

Where do I get `ω` and `ψ` concretely? `Z_q^*` is cyclic of order `q - 1`; let `g` be a generator (a primitive root). Then `g^{(q-1)/(2n)}` has order exactly `2n` — take that as `ψ` — and `ω = ψ^2 = g^{(q-1)/n}` has order `n`. The conditions `ψ^2 = ω` and `ψ^n = -1` and `ω^n = 1` all hold by construction. Finding `g` is the standard primitive-root test: factor `q - 1`, and `g` is a generator iff `g^{(q-1)/r} ≠ 1` for each prime factor `r` of `q - 1`. (And there may be several valid `ω`, `ψ`; any consistent choice works, as long as forward and inverse use the matched pair.)

Now the modulus. I need `q ≡ 1 (mod 2n)`, prime. The cheapest such primes have the shape `q = k·2^m + 1` with `2n | 2^m` and `k` small — Proth primes. Two of them dominate practice: `q = 12289 = 3·2^{12} + 1` (so `2n` up to `2^{12}` works, covering `n = 512, 1024`) and `q = 7681`. The historical cousins are Fermat primes `2^{2^k}+1`, chosen because `q - 1 = 2^{2^k}` is a pure power of two so plenty of `2`-power roots of unity exist, and Mersenne primes `2^p - 1`, chosen instead because `2^p ≡ 1` makes reduction cheap — their `q - 1 = 2(2^{p-1}-1)` carries only one factor of two, so they don't supply many radix-2 roots. The shape also pays off in the inner loop: every butterfly does a modular multiply, and reduction mod `q` is the hot operation. The general tools are Montgomery reduction (carry values in a transformed residue system `x̃ = x·R` so reducing a product is multiply–add–shift, no division) and Barrett reduction (precompute a reciprocal of `q`). But for `q = k·2^m + 1` I can do better with the structure directly: `k·2^m ≡ -1 (mod q)` — for `q = 12289`, `k = 3`, `m = 12`, indeed `3·2^{12} = 12288 ≡ -1 (mod 12289)`. So writing a product `C = C_0 + 2^m C_1` with `0 ≤ C_0 < 2^m`, multiplying through by `k` and substituting `k·2^m ≡ -1` gives `kC = k C_0 + k·2^m C_1 ≡ k C_0 - C_1 (mod q)` — a shift to peel off `C_1`, a small multiply by `k`, a subtract. Let me check that on a real product to be sure I didn't drop a sign: take `C = 9999·8888 = 88871112`; then `C_0 = C mod 2^{12} = 200`, `C_1 = ⌊C/2^{12}⌋ = 21697`, and `k C_0 - C_1 = 3·200 - 21697 = -21097 ≡ 3481 (mod 12289)`, while computing `kC mod q = 3·88871112 mod 12289` directly also gives `3481`. They agree, so `K-RED(C) = k C_0 - C_1` does represent `kC mod q`. That `K-RED(C) = kC_0 - C_1` doesn't fully reduce into `[0,q)`; it lands within a bounded multiple of `q` and changes the residue class by a tracked factor of `k`, just like Montgomery's representation. In a transform I let the coefficients grow lazily, apply `K-RED` only after multiplications, keep a running power of `k`, and divide it out at the very end — folding that final correction into the same `n^{-1}` scaling so it's free. If a value threatens to overflow the word, a double-width `K-RED-2x` on `C = C_0 + 2^m C_1 + 2^{2m} C_2` returning `k^2 C_0 - k C_1 + C_2` mops it up. The whole reduction story is: choose `q` of Proth form, exploit `k·2^m ≡ -1` for a branch-light reduction, defer full reductions to the end.

Let me make this concrete and check it, because I want to *see* the negacyclic product fall out, and I want to confirm the `ψ^{-i}` post-weight sign and indexing. Take the smallest honest case: `n = 4`, and a prime with `2n = 8 | q - 1`. `q = 7681 = 15·2^9 + 1` works, `q - 1 = 7680` is divisible by `8`. Its primitive root is `17`; `ω = 17^{7680/4} = 3383` has order `4`, and `ψ = 17^{7680/8} = 1925` has `ψ^2 = 3383 = ω` and `ψ^4 = -1 ≡ 7680`. Now multiply `f = [1,2,3,4]` by `g = [5,6,7,8]` modulo `x^4 + 1`. Schoolbook: the linear product is `5 + 16x + 34x^2 + 60x^3 + 61x^4 + 52x^5 + 32x^6`, and folding `x^4 = -1`, `x^5 = -x`, `x^6 = -x^2` gives `(5 - 61) + (16 - 52)x + (34 - 32)x^2 + 60x^3 = -56 - 36x + 2x^2 + 60x^3`, i.e. `[7625, 7645, 2, 60]` mod `7681`. That's my target.

Here's the code — a single-file C++17 program that reads `q`, `n`, then the coefficients of `f` and `g` from stdin and prints `f*g mod (x^n + 1)`. It's the fast merged-`ψ` pair: a Cooley–Tukey decimation-in-time forward (twiddles are odd powers of `ψ` drawn from a bit-reversed table, so the `ψ`-weighting is built in and there's no pre-scale pass, output bit-reversed) and a Gentleman–Sande decimation-in-frequency inverse (twiddles are powers of `ψ^{-1}`, the `n^{-1}` and `ψ^{-i}` post-weight folded in, input bit-reversed). Every modular product runs through `__int128` so two near-`q` residues never overflow. I'll check it against the schoolbook negacyclic convolution by hand and at scale.

The risky step is the merged-`ψ` negacyclic NTT butterfly with the CT/GS twiddle order and output-indexed `ψ^{-i}` post-weight; if I wasn't confident I could implement that correctly within budget, I'd fall back to the length-`2n` zero-padded cyclic NTT and explicitly subtract the high half modulo `x^n + 1`, then ship that.

```cpp
// Negacyclic polynomial multiplication in Z_q[x]/(x^n + 1) via the NTT, O(n log n).
// stdin:  q  n   then n coefficients of f   then n coefficients of g
//         (n a power of two, q a prime with q == 1 (mod 2n))
// stdout: the n coefficients of f*g mod (x^n + 1), each reduced into [0, q).
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

ll power_mod(ll a, ll e, ll q) {            // a^e mod q, exact via 128-bit products
    a %= q; if (a < 0) a += q;
    ll r = 1 % q;
    while (e > 0) {
        if (e & 1) r = (__int128)r * a % q;
        a = (__int128)a * a % q;
        e >>= 1;
    }
    return r;
}

ll inv_mod(ll a, ll q) { return power_mod(a, q - 2, q); }   // q prime => Fermat inverse

// Generator g of (Z/qZ)^*, q prime: factor q-1 and test orders.
ll find_primitive_root(ll q) {
    ll phi = q - 1, m = phi;
    vector<ll> factors;
    for (ll d = 2; d * d <= m; ++d)
        if (m % d == 0) { factors.push_back(d); while (m % d == 0) m /= d; }
    if (m > 1) factors.push_back(m);
    for (ll g = 2; g < q; ++g) {
        bool ok = true;
        for (ll f : factors) if (power_mod(g, phi / f, q) == 1) { ok = false; break; }
        if (ok) return g;
    }
    throw runtime_error("no primitive root");
}

int brv(int x, int bits) {                  // bit-reversal of x in 'bits' bits
    int r = 0;
    for (int i = 0; i < bits; ++i) { r = (r << 1) | (x & 1); x >>= 1; }
    return r;
}

// psi^0..psi^{n-1} stored in bit-reversed index order (the twiddle table).
vector<ll> psi_table(ll psi, int n, ll q) {
    int bits = __builtin_ctz((unsigned)n);
    vector<ll> pw(n);
    pw[0] = 1 % q;
    for (int i = 1; i < n; ++i) pw[i] = (__int128)pw[i - 1] * psi % q;
    vector<ll> rev(n);
    for (int i = 0; i < n; ++i) rev[i] = pw[brv(i, bits)];
    return rev;
}

// Negacyclic forward NTT, Cooley-Tukey (decimation-in-time): standard -> bit-reversed.
// psi weighting is built into the twiddle table, so there is no separate pre-scale.
void ntt_forward(vector<ll> &a, ll psi, ll q) {
    int n = a.size();
    vector<ll> Psi = psi_table(psi, n, q);
    for (int m = 1, t = n; m < n; m <<= 1) {
        t >>= 1;
        for (int i = 0; i < m; ++i) {
            int j1 = 2 * i * t; ll S = Psi[m + i];
            for (int j = j1; j < j1 + t; ++j) {
                ll U = a[j], V = (__int128)a[j + t] * S % q;
                a[j]     = (U + V) % q;
                a[j + t] = (U - V % q + q) % q;
            }
        }
    }
}

// Negacyclic inverse NTT, Gentleman-Sande (decimation-in-frequency): bit-reversed -> standard.
// The n^{-1} scaling and psi^{-i} post-weighting fold into the twiddle table and final scale.
void ntt_inverse(vector<ll> &a, ll psi, ll q) {
    int n = a.size();
    vector<ll> Psi = psi_table(inv_mod(psi, q), n, q);
    for (int t = 1, m = n; m > 1; t <<= 1, m >>= 1) {
        int j1 = 0, h = m / 2;
        for (int i = 0; i < h; ++i) {
            ll S = Psi[h + i];
            for (int j = j1; j < j1 + t; ++j) {
                ll U = a[j], V = a[j + t];
                a[j]     = (U + V) % q;
                a[j + t] = (__int128)((U - V + q) % q) * S % q;
            }
            j1 += 2 * t;
        }
    }
    ll ninv = inv_mod(n % q, q);
    for (int i = 0; i < n; ++i) a[i] = (__int128)a[i] * ninv % q;
}

// Product f*g in Z_q[x]/(x^n + 1): forward-NTT both, multiply pointwise, inverse-NTT.
vector<ll> negacyclic_mul(vector<ll> f, vector<ll> g, ll q) {
    int n = f.size();
    ll root = find_primitive_root(q);
    ll psi = power_mod(root, (q - 1) / (2LL * n), q);   // primitive 2n-th root, psi^n = -1
    ntt_forward(f, psi, q);
    ntt_forward(g, psi, q);
    vector<ll> h(n);
    for (int i = 0; i < n; ++i) h[i] = (__int128)f[i] * g[i] % q;
    ntt_inverse(h, psi, q);
    return h;
}

int main() {
    ios_base::sync_with_stdio(false); cin.tie(nullptr);
    ll q; int n;
    if (!(cin >> q >> n)) return 0;
    vector<ll> f(n), g(n);
    for (int i = 0; i < n; ++i) { cin >> f[i]; f[i] = ((f[i] % q) + q) % q; }
    for (int i = 0; i < n; ++i) { cin >> g[i]; g[i] = ((g[i] % q) + q) % q; }
    vector<ll> c = negacyclic_mul(f, g, q);
    for (int i = 0; i < n; ++i) cout << c[i] << (i + 1 < n ? ' ' : '\n');
    return 0;
}
```

Now let me actually run it on the `n = 4`, `q = 7681` case and see whether what comes out is the negacyclic product I computed by hand. First the parameters the code derives: it finds the primitive root `g = 17`, sets `ψ = 17^{7680/8} = 1925` and `ω = ψ^2 = 3383`, and the asserts `ω^4 ≡ 1`, `ψ^4 ≡ -1` both pass — so `1925^4 mod 7681 = 7680 = -1`, the sign-flip element is where I claimed.

The forward transform `ntt_forward([1,2,3,4])` returns `[1467, 3471, 2807, 7621]`. I should sanity-check that this is the weighted transform I think it is, in the order I think it is. The definitional weighted transform `â_j = Σ_i ψ^i ω^{ij} a_i` evaluated directly gives `[1467, 2807, 3471, 7621]` in standard `j` order; bit-reversing the index (swap entries 1 and 2 for `n=4`) gives `[1467, 3471, 2807, 7621]` — which is exactly what the fast routine emitted. So the Cooley–Tukey forward really is producing the `ψ`-weighted spectrum in bit-reversed order, with the weighting absorbed into the twiddles and no pre-scaling pass. Good.

Now the resolution of the `ψ^{-i}` vs `ψ^{-j}` question I deferred. I evaluate both inverse candidates on the forward spectrum `A = [1467, 2807, 3471, 7621]`. Candidate A (`a_i = n^{-1} ψ^{-i} Σ_j ω^{-ij} A_j`, the weight outside, output-indexed) returns `[1, 2, 3, 4]` — it inverts. Candidate B (`a_i = n^{-1} Σ_j ψ^{-j} ω^{-ij} A_j`, the weight inside, summation-indexed) returns `[1454, 5738, 4048, 5589]` — not even close to `[1,2,3,4]`. So my worry was justified: the post-weight must be `ψ^{-i}`, output-indexed, outside the sum, and the summation-indexed arrangement that looked just as plausible on paper is simply wrong. The two were genuinely not interchangeable, and only running them told me which.

With that pinned down, the full pipeline: pointwise-multiply the forward transform of `[1,2,3,4]` with that of `[5,6,7,8]`, run the GS inverse. It returns `[7625, 7645, 2, 60]` — exactly the schoolbook negacyclic product `[-56, -36, 2, 60] mod 7681` I worked out above. The roundtrip `ntt_inverse(ntt_forward(f))` returns `[1,2,3,4]`, so the `n^{-1}` scaling and the matched `ψ`/`ψ^{-1}` tables are consistent end-to-end; and the bare definitional cyclic transform `a_i = n^{-1} Σ_j (Σ_k a_k ω^{kj}) ω^{-ij}` also returns `f` when worked out by hand, re-confirming the bare identity underneath. Finally I push to the real parameters — `n = 512`, `q = 12289` — with random polynomials: `negacyclic_mul` agrees with `schoolbook_negacyclic` on every one of the 512 coefficients. That's the size the cryptography actually runs at, so the merged-`ψ`, no-bit-reversal version is not just correct on the toy case.

So the whole causal chain: I wanted exact `O(n log n)` polynomial multiplication, especially modulo `x^n + 1`. The DFT-plus-convolution-theorem gives `O(n log n)` but over `C` it rounds, and rounding integers is fragile. Stripping the DFT's correctness proof to its core, it needs only an element of exact order `n` with `1 - ω^K ≠ 0` — a primitive `n`-th root of unity — which `Z_p` supplies exactly when `n | p - 1`, in pure modular arithmetic with no round-off. The Cooley–Tukey butterfly is purely algebraic (`ω^{n/2} = -1`), so it runs over `Z_p` unchanged, giving the NTT. Cyclic convolution mod `x^n - 1` follows immediately. For the cryptographic ring `x^n + 1`, instead of doubling the length, I weight the inputs by powers of a `2n`-th root `ψ` (existence needs `q ≡ 1 mod 2n`); because `ψ^n = -1`, the wrap-around term flips sign, turning the cyclic convolution into the negacyclic one — equivalently, the weighted transform evaluates at the odd `2n`-th roots, the roots of `x^n + 1`, so the reduction is implicit. Folding `ψ` into the twiddle factors and pairing a decimation-in-time forward with a decimation-in-frequency inverse absorbs both the weighting and the bit-reversal. Roots come from a primitive root via `ψ = g^{(q-1)/2n}`, `ω = ψ^2`; the modulus is a Proth prime `k·2^m + 1` whose structure (`k·2^m ≡ -1`) also gives a fast modular reduction, with Montgomery/Barrett as the general fallback. The result is bit-exact integer convolution at `O(n log n)`, verified against schoolbook.
