Let me start from the thing that actually bothers me, which is not factoring — it's that I have a machine in front of me, a quantum machine, that on paper can hold `2^n` complex amplitudes at once, and so far nobody has gotten a single useful thing out of it that a classical computer with a coin couldn't already do. Deutsch and Jozsa, Bernstein and Vazirani — they get separations, but either the separation is exact and fragile or the problem is so contrived it feels like cheating. The whole field is sitting on this enormous Hilbert space and producing toys. So the real question is: what does a quantum computer naturally do well, and is there a problem people care about whose structure matches it?

And I keep coming back to the one piece of machinery that genuinely impressed me, which is Simon's. He has a black box `f` on `n`-bit strings with a promise: `f(x) = f(y)` exactly when `x ⊕ y` is `0` or some hidden `s`. Classically you are stuck — to learn `s` you have to find a collision, and that's `~2^{n/2}` queries, birthday-style, no way around it. His quantum routine does it in `O(n)`. Let me make sure I actually understand *why* it works, because the "why" is the only thing that will transfer. He puts the first register into a uniform superposition over all `x`, computes `f(x)` into a second register, and then he *measures the second register*. The first register collapses onto exactly the set of `x` that produce the observed `f`-value, which by the promise is a single coset `{x_0, x_0 ⊕ s}`. So now he is holding `(1/√2)(|x_0⟩ + |x_0 ⊕ s⟩)` and he knows it's a translate of the two-element subgroup `{0, s}`, he just doesn't know `x_0` or `s`. Measuring now would give a random `x_0` or `x_0 ⊕ s` — useless, the offset `x_0` is random noise. So he applies a Fourier transform over the binary vector space `(Z_2)^n`, which is just a Hadamard on every wire, and the magic is that the random offset `x_0` only contributes a *global phase* (or a sign per output), so it washes out of the probabilities, while the *period* `s` survives in the constraint that the only `y` you can observe satisfy `y · s = 0`. Repeat, gather `n−1` independent linear equations over `GF(2)`, solve, done.

Strip away the `(Z_2)^n` and what's left is general. A quantum computer is good at one very specific thing: take a function with a *hidden periodicity*, put its whole domain in superposition, and use a *Fourier transform* to convert that hidden period into something you can read off by measurement — because the Fourier transform is exactly the basis in which a shift becomes a phase, so the unknown shift cancels and the period concentrates the amplitude. The interference does the work: amplitudes for the "wrong" outcomes cancel, amplitudes consistent with the period add.

The instant I phrase it that way, I want to ask: what other problems are secretly period-finding? Simon's period lives in `(Z_2)^n` and is extracted with a Fourier transform over that group. What if the periodicity I care about lives somewhere else — in the integers, in a cyclic group?

Discrete logarithm jumps out first. Fix a prime `p` and a generator `g` of the cyclic group mod `p`; given `x`, find `r` with `g^r ≡ x (mod p)`. Stare at the two-variable function `f(a, b) = g^a x^{-b} (mod p)`. When does `f(a,b) = f(a', b')`? We need `g^{a-a'} = x^{b-b'} = g^{r(b-b')}`, i.e. `a - a' ≡ r(b-b') (mod p-1)`. So `f` is constant exactly along the lines `(a, b) → (a + r t, b + t)`: it has a hidden *period vector* `(r, 1)` in the lattice `Z_{p-1} × Z_{p-1}`, and `r` — the thing I want — is literally the slope of that period. This is the same shape as Simon: a function constant on cosets of a hidden subgroup, where the subgroup encodes the answer. Only now the group is cyclic of order `p-1`, not `(Z_2)^n`. So Simon's Hadamard won't do. I need the honest Fourier transform over a *cyclic* group, the discrete Fourier transform: the one that sends `|a⟩` to `q^{-1/2} Σ_c exp(2πi a c / q) |c⟩`. That's the object whose interference detects integer periods. If I can build *that* unitary efficiently, I can run Simon's strategy in `Z_q` and read off `r`.

Two problems immediately. First, can I even build the DFT as a quantum circuit in polynomial time? Classically the FFT does length `q` in `O(q log q)` operations by recursively splitting on the factorization of `q`. On `q = 2^m` amplitudes, `q log q` is exponential in `m` if I count it naively — but I'm not manipulating `q` numbers one by one, I'm manipulating `m` qubits, and the radix-2 structure is controlled by bit positions. So maybe the quantum version collapses to a pattern of one-bit operations and pairwise phases, only `O(m²)` gates. Let me hold that thought; it smells right because the FFT is already a product of sparse, structured factors.

Second, and worse: in discrete log my period `r` and the modulus `p-1` are genuine arbitrary integers. In Simon's case the period `s` was a subgroup and `f` was *exactly* constant on cosets, perfectly aligned with the group. Here, when I do order-finding the bad version of this will bite: my period `r` will *not* divide my transform length `q`. The interference won't be perfectly sharp; it'll be smeared. I'm going to have to deal with a Fourier peak that sits at a non-integer location and bound how much amplitude leaks. Let me park that too — it's the real technical fight — and first nail down whether the whole plan even reaches factoring.

Let me try to push discrete log through to feel out the machinery, and start with a case I can control. Suppose `p - 1` is *smooth* — all its prime factors are small. Then I can do the Fourier transform over `Z_{p-1}` exactly by factoring `p-1` into small primes and doing a small DFT for each, the classical mixed-radix FFT, and everything lands on exact group elements with no smearing. And in fact in this smooth case I can even check my answer against a classical computation, because discrete log with smooth `p-1` is already easy classically (Pohlig–Hellman). Good — that means the smooth case is a *sanity harness*, not a real win, but it lets me debug the quantum interference where I know the right answer. In that exact setting, the Simon strategy on `f(a,b) = g^a x^{-b}` makes the measured Fourier labels obey the slope relation that contains `r`. The transform really is converting the hidden period vector into a measurable phase relation. Now I believe the engine.

So now generalize: forget `p-1` being smooth, and forget discrete log being the target. The engine is *order-finding*: given a function whose values repeat with some unknown period `r`, find `r`. Discrete log was just one dress. What's the most valuable thing whose hardness is secretly an order?

Factoring. And the bridge already exists — I don't have to invent it. It's Miller's reduction: factoring an odd `N` reduces, with randomization, to computing the *order* of a random element. Let me re-derive it so I trust every case, because the whole plan dies if the reduction is leaky. Pick a random `x` with `1 < x < N`. If `gcd(x, N) ≠ 1` I've already stumbled onto a factor, lovely, but that's vanishingly rare for large `N`, so assume `x` is a unit. Let `r` be its order, the least `r` with `x^r ≡ 1 (mod N)`. Suppose `r` is even. Then
`x^r - 1 = (x^{r/2} - 1)(x^{r/2} + 1) ≡ 0 (mod N)`,
so `N` divides the product `(x^{r/2}-1)(x^{r/2}+1)`. Now `x^{r/2} ≢ 1 (mod N)` because `r` is the *least* exponent giving `1` and `r/2 < r`, so `N` does not divide the left factor. If additionally `x^{r/2} ≢ -1 (mod N)`, then `N` doesn't divide the right factor either. So `N` divides the product but neither factor — which means `N`'s prime factors are split between the two factors, and therefore `gcd(x^{r/2} - 1, N)` is a *nontrivial* divisor of `N`. (And `gcd(x^{r/2} + 1, N)` gives the complementary one.) The reduction fails only in two cases: `r` odd (no `r/2`), or `x^{r/2} ≡ -1 (mod N)`.

I need those two failures to be rare over random `x`, or the reduction is worthless. Write `N = ∏_{i=1}^{k} p_i^{a_i}` with `k` distinct odd primes. By the Chinese remainder theorem, choosing `x` at random mod `N` is the same as independently choosing `x_i` at random mod each `p_i^{a_i}`. Let `r_i` be the order of `x` mod `p_i^{a_i}`; then `r = lcm(r_i)`. Look at the largest power of `2` dividing each `r_i`, call its exponent `e_i`. The bad cases happen exactly when all the `e_i` are *equal*: if every `e_i = 0` then every `r_i` is odd so `r` is odd; if every `e_i` equals some common `e ≥ 1`, then `x^{r/2} ≡ -1 (mod p_i^{a_i})` for each `i`, hence `x^{r/2} ≡ -1 (mod N)`. If the `e_i` are not all equal, let `E` be the largest one. For every factor with `e_i < E`, the exponent `r/2` is still a multiple of `r_i`, so `x^{r/2} ≡ +1` there; for every factor with `e_i = E`, the exponent `r/2` is an odd multiple of `r_i/2`, so cyclicity gives the unique element of order `2`, namely `-1`. The residues are mixed, so `x^{r/2}` is neither `+1` nor `-1` modulo `N`, and the gcd bites. The probability bound comes from cyclicity: each `(Z/p_i^{a_i})*` is cyclic for an odd prime power, so for a random `x_i` the probability that `e_i` takes any one specified value is at most `1/2` — in a cyclic group of order `2^t m` with `m` odd and `t ≥ 1`, the elements with any fixed 2-adic valuation of their order form at most half the group. The `e_i` are independent across `i` by CRT. So the chance that all `k` of them agree with, say, the first one, is at most `(1/2)^{k-1}`. Hence the procedure yields a factor with probability at least `1 - 1/2^{k-1}`. For `N` with at least two distinct odd prime factors that's at least `1/2` — a constant. (Prime powers and even `N` I peel off classically beforehand.) The reduction is solid.

So the entire problem is now: **find the order `r` of `x` modulo `N`**, i.e. the period of `a → x^a (mod N)`. If I can do that on the quantum machine in polynomial time, factoring falls. Everything rides on building the order-finder, which is the cyclic-Fourier period-finder I already half-believe in.

Let me build it concretely. I want a transform length `q`. How big? I'll come back and choose `q` from a resolution requirement, but let me carry it symbolically and take it a power of `2` so the FFT-style circuit is clean.

First, put the first register into the uniform superposition over `a ∈ {0, …, q-1}`. That's just a Hadamard on each of the `m = log_2 q` qubits, since `H |0⟩ = (|0⟩+|1⟩)/√2` and tensoring gives `q^{-1/2} Σ_a |a⟩`. State:
`q^{-1/2} Σ_{a=0}^{q-1} |a⟩ |0⟩`.

Second, compute `x^a (mod N)` into the second register. This is modular exponentiation, and it has to be reversible. Keeping `a` in the first register makes the map `(a, 0) → (a, x^a mod N)` reversible, because the input `a` is preserved. Concretely I precompute the constants `x^{2^i} mod N` classically, and then `power := power · x^{2^i} mod N` controlled on bit `a_i` — repeated squaring, but with the squarings done at circuit-design time because `x` is fixed and only `a` is in superposition. Multiplication mod `N` I build from repeated controlled addition mod `N`, and I make *that* reversible by the standard Bennett dance: compute `b → bc mod N` into fresh space, then erase the input `b` by running the inverse multiplication (by `c^{-1} mod N`, which exists since I only ever multiply by units) backwards. Longhand multiplication of `O(log N)`-bit numbers is `O((log N)²)` gates, and `O(log N)` controlled multiplications give `O((log N)³)` total time and `O(log N)` space. The point is just: this is a reversible classical subroutine, polynomial, leaves me in
`q^{-1/2} Σ_{a=0}^{q-1} |a⟩ |x^a mod N⟩`.

Now the structure I'm counting on is visible: the second register's value `x^a mod N` depends only on `a mod r`, because `x` has order `r`. So the second register is *periodic* in `a` with period `r`. This is exactly a hidden-period function — Simon's situation, but the period `r` is an integer in the cyclic index group, not an XOR-shift.

Third, the Fourier transform on the first register:
`|a⟩ → q^{-1/2} Σ_{c=0}^{q-1} exp(2πi a c / q) |c⟩`.
Apply it and the joint state becomes
`(1/q) Σ_{a=0}^{q-1} Σ_{c=0}^{q-1} exp(2πi a c / q) |c⟩ |x^a mod N⟩`.

Fourth, measure. It's enough to measure `|c⟩`, but to see the structure let me imagine measuring both registers and getting `|c, x^k mod N⟩` for some `0 ≤ k < r`. What's the probability? I have to sum the amplitude over *every* `a` that produced this same second-register value — and `x^a ≡ x^k (mod N)` exactly when `a ≡ k (mod r)`. So the surviving `a` are `a = b r + k` for `b = 0, 1, …, ⌊(q - k - 1)/r⌋`. The amplitude is
`(1/q) Σ_{a ≡ k (mod r)} exp(2πi a c / q) = (1/q) Σ_{b} exp(2πi (br + k) c / q)`,
and the probability is the squared modulus. The `exp(2πi k c / q)` factor is common to every term, pull it out, it has modulus `1`, drop it. Left with
`P_k(c) = | (1/q) Σ_{b=0}^{⌊(q-k-1)/r⌋} exp(2πi b r c / q) |²`.

It's a geometric sum of unit phasors stepping by angle `2π r c / q` per term. If that step angle is near a multiple of `2π`, all the phasors point the same way and the sum is huge — order `q/r` terms each near `+1`. If the step angle is far from a multiple of `2π`, the phasors fan out around the circle and cancel. So the probability concentrates on those `c` for which `rc/q` is near an integer — i.e. `c` near a multiple of `q/r`. That's the period, read out as the *location of the Fourier peaks*. Beautiful: the order `r` shows up as `c ≈ (integer) · q/r`.

Let me make "near a multiple of `2π`" quantitative, because `r` does not divide `q` and that's the smearing I worried about. Let `{rc}_q` be the residue of `rc` modulo `q` chosen in the symmetric range `-q/2 < {rc}_q ≤ q/2`. Replacing `rc` by `{rc}_q` in the exponent only adds multiples of `2π`, so
`P_k(c) = | (1/q) Σ_{b=0}^{⌊(q-k-1)/r⌋} exp(2πi b {rc}_q / q) |²`.
Now I want a lower bound when `{rc}_q` is small. The summand `exp(2πi b {rc}_q / q)` is a slowly rotating phasor in `b`; compare the sum with the integral
`(1/q) ∫_0^{⌊(q-k-1)/r⌋} exp(2πi b {rc}_q / q) db`,
with an error that is `O((⌊(q-k-1)/r⌋ / q)·|exp(2πi {rc}_q/q) - 1|)`; when `|{rc}_q| ≤ r/2` this error is `O(1/q)`. Substitute `u = r b / q`, `db = (q/r) du`; the upper limit becomes `(r/q)⌊(q-k-1)/r⌋`, which since `k < r` is within `O(1/q)` of `1`, so replacing it by `1` costs another `O(1/q)`:
`(1/r) ∫_0^1 exp(2πi ({rc}_q / r) u) du`.
This integral is the standard "sinc" envelope. Its modulus is minimized, over `{rc}_q / r ∈ [-1/2, 1/2]`, at the endpoints `± 1/2`, where
`| (1/r) ∫_0^1 exp(± π i u) du | = (1/r) · |(e^{±πi} - 1)/(±πi)| = (1/r)·(2/π) = 2/(π r)`.
So whenever `|{rc}_q| ≤ r/2`, the main term in the amplitude has modulus at least `2/(πr)`, and the actual amplitude differs from it by only `O(1/q)`. Since `q ≥ N²` and `r < N`, that error is much smaller than `1/r` for large `N`, so the probability has asymptotic lower bound `4/(π² r²)` and in particular is at least `1/(3r²)` for sufficiently large `N`. Each such good pair `(c, x^k)` is hit with probability at least `1/(3r²)`, regardless of `k`.

The measurement is useful only if I can turn the peak location back into the denominator `r`. The good `c` are exactly those with `|{rc}_q| ≤ r/2`, i.e. there's an integer `d` with `-r/2 ≤ rc - dq ≤ r/2`. Divide through by `rq`:
`| c/q - d/r | ≤ 1/(2q)`.
I *know* `c` and `q`; I *want* `r`. The fraction `c/q` is within `1/(2q)` of the unknown `d/r`. This is a Diophantine-approximation problem, and continued fractions are exactly the tool: the convergents of `c/q` are the best rational approximations, computed by the simple recurrence `p_n = a_n p_{n-1} + p_{n-2}`, `q_n = a_n q_{n-1} + q_{n-2}` from the continued-fraction digits `a_n = ⌊ξ_n⌋`. I just need to know that `d/r` is *the unique* fraction with denominator below `N` that close to `c/q` — otherwise I can't pin it down. Suppose two distinct reduced fractions `d/r` and `d'/r'` both had denominators `< N` and both within `1/(2q)` of `c/q`. Then they'd be within `1/q` of each other, but distinct fractions with denominators `< N` differ by at least `1/(r r') > 1/N²`. So I need `1/q ≤ 1/N²`, i.e. `q ≥ N²`. *That* is where the transform length comes from: I choose `q` a power of `2` with `N² ≤ q < 2N²`. With that choice, rounding `c/q` to its nearest fraction of denominator below `N` — which the continued-fraction expansion finds in polynomial time — returns `d/r` in lowest terms. If the reduced denominator is `r`, I'm done; in general I get `r` divided by `gcd(d, r)`, so I get `r` exactly when `d` happens to be coprime to `r`.

So how often does a run succeed? Two things must go right: I must land on a good pair `(c, x^k)` with probability at least `1/(3r²)` for that pair, and the corresponding multiple `d` must be coprime to `r` so the continued fraction returns the true `r` and not a proper divisor. Count the winners. For each `c` near a multiple of `q/r`, the integer `d` is the nearest multiple — and the values of `d` coprime to `r` number `φ(r)`, Euler's totient. Each such `d` corresponds to one good `c`. And the second register `x^k` can be any of `r` values, each pairing with these `c`'s. So there are `r · φ(r)` measurement outcomes `(c, x^k)` from which I recover `r`, each occurring with probability at least `1/(3r²)`. Total:
`P(recover r) ≥ r φ(r) · 1/(3r²) = φ(r) / (3r)`.
And `φ(r)/r` is not too small: by the classical bound `φ(r)/r > δ / log log r` for a constant `δ` (this is the statement that an integer doesn't have *too* many small prime factors). So each run succeeds with probability `≳ 1 / log log r`, and `O(log log r)` repetitions give success with high probability. A polynomial-time order-finder, hence — through Miller's reduction — a polynomial-time factorer. The whole thing runs in `O((log N)²)` Fourier gates plus the modular-exponentiation bottleneck `O((log N)³)` (or `O((log N)² log log N log log log N)` with Schönhage–Strassen), times the `O(log log r)` repetitions, plus polynomial classical post-processing.

I owe myself the one thing I waved at: that the cyclic Fourier transform on `m` qubits really is buildable in `O(m²)` gates, and that the circuit I write down actually computes `q^{-1/2} Σ_c exp(2πi a c/q)|c⟩`. Take `q = 2^m`, write `a = a_{m-1}…a_0` in binary. I'll use only two kinds of gate. A Hadamard on bit `j`,
`R_j: |0⟩ → (|0⟩+|1⟩)/√2,  |1⟩ → (|0⟩-|1⟩)/√2`,
and a controlled phase between bits `j < k`,
`S_{j,k}: |11⟩ → e^{iθ_{k-j}} |11⟩`, with `θ_{k-j} = π / 2^{k-j}`, all other basis states fixed.
Apply them in the order `R_{m-1} S_{m-2,m-1} R_{m-2} S_{m-3,m-1} S_{m-3,m-2} R_{m-3} … R_0`: the Hadamards in descending bit order, and between `R_{j+1}` and `R_j` all the controlled phases `S_{j,k}` for `k > j`. That's `m` Hadamards and `m(m-1)/2` phase gates, so `O(m²)`. Now check it gives the DFT phase. The Hadamards are the only gates that *flip* bit values; the `S_{j,k}` only add phases. So the amplitude on a path from `|a⟩` to a particular output `|b⟩` is determined entirely: the `m` Hadamards each contribute `1/√2`, multiplying to `(1/√2)^m = 1/√q` overall (with a sign `-` whenever `a_j = b_j = 1`, i.e. a phase `π a_j b_j`), and `S_{j,k}` contributes its phase `π/2^{k-j}` exactly when `a_j = 1` and `b_k = 1`. Total accumulated phase:
`Σ_{0≤j<m} π a_j b_j + Σ_{0≤j<k<m} (π/2^{k-j}) a_j b_k = Σ_{0≤j≤k<m} (π / 2^{k-j}) a_j b_k`,
folding the diagonal `j=k` term (where `π/2^0 = π`) into the double sum. The Hadamard/phase circuit, read naively, produces the output index in *bit-reversed* order: the bit that ends up as the most significant of the output came from the least significant input position, so the natural output `b` is the bit-reversal of the Fourier index `c`, meaning `b_k = c_{m-1-k}`. Substitute:
`Σ_{0≤j≤k<m} (π / 2^{k-j}) a_j c_{m-1-k}`.
Reindex `k ← m-1-k` to turn `c_{m-1-k}` into `c_k`; the constraint `j ≤ k` becomes `j + k ≤ m-1`, i.e. `j + k < m`, and `1/2^{k-j}` becomes `2^{j} 2^{k} / 2^{m}` after the substitution and a factor of `2`:
`Σ_{j+k<m} 2π (2^j 2^k / 2^m) a_j c_k`.
The terms with `j + k ≥ m` only add integer multiples of `2π` (because `2^j 2^k / 2^m = 2^{j+k-m}` is an integer there), so I can freely extend the sum to all `j, k < m` without changing the phase:
`Σ_{j,k=0}^{m-1} 2π (2^j 2^k / 2^m) a_j c_k = (2π / 2^m) (Σ_j 2^j a_j)(Σ_k 2^k c_k) = 2π a c / q`,
using `q = 2^m`, `a = Σ 2^j a_j`, `c = Σ 2^k c_k`, and the distributive law. That is exactly the DFT phase `exp(2πi a c / q)`, with the `1/√q` from the Hadamards out front. The circuit is the quantum Fourier transform. (Either reverse the output bits at the end, or just read them backwards — trivial.) And I notice the `S_{j,k}` for large `k-j` apply a phase `π/2^{k-j}` that is exponentially tiny; those are exactly the hard-to-implement gates physically, and they barely matter — dropping the tiniest ones gives an *approximate* Fourier transform that is still good enough to factor and uses far fewer gates. Good to know, and reassuring that the precision demand isn't catastrophic.

Let me also notice, now that it's built, how little I used. In the order-finder I never used anything about multiplication mod `N` except that `a → x^a` repeats with some period `r` and that I can iterate it cheaply. So the same machine finds the period of *any* function `f` on `{0,…,n-1}` whose `k`-th iterate is computable in time polynomial in `log n` and `log k` — the order of an element under any such map. The factoring application is just the most lucrative instance.

Factoring has no obvious handle, but Miller turns it into order-finding: choose random `x`, find the order `r` of `x mod N`, and `gcd(x^{r/2} ± 1, N)` splits `N` whenever `r` is even and `x^{r/2} ≢ -1` — which happens with probability at least `1 - 1/2^{k-1}` for `k` distinct odd prime factors, hence at least half once the prime-power case is removed. Order is a *period*: `r` is the period of `a → x^a mod N`. A quantum machine finds periods because, after putting the domain in superposition and computing the periodic function, a Fourier transform makes the amplitudes for `c ≈ (multiple of) q/r` interfere constructively and everything else cancel; with noticeable probability a measurement returns a `c` with `|c/q - d/r| ≤ 1/(2q)`. Choosing the transform length `q ≈ N²` makes `d/r` the *unique* low-denominator fraction that close to `c/q`, so a classical continued-fraction expansion recovers `r` when `d` is coprime to `r`; repeating `O(log log r)` times clears the `φ(r)/r` factor and gives `r` with high probability. The Fourier transform itself is `m` Hadamards plus `m(m-1)/2` controlled phases, with `m = log_2 q = O(log N)`, whose accumulated phases sum exactly to `2π ac/q`. Polynomial time, start to finish — and RSA's hardness assumption with it.

```python
import math, random
from collections import defaultdict
from fractions import Fraction

# ---- classical primitives (already standard) ----
def gcd(a, b):
    while b:
        a, b = b, a % b
    return abs(a)

def modexp(base, e, mod):
    r = 1; base %= mod
    while e:
        if e & 1: r = (r * base) % mod
        base = (base * base) % mod; e >>= 1
    return r

def _integer_nth_root(n, k):
    lo, hi = 1, 1 << ((n.bit_length() + k - 1) // k)
    while lo <= hi:
        mid = (lo + hi) // 2
        p = mid ** k
        if p == n:
            return mid
        if p < n:
            lo = mid + 1
        else:
            hi = mid - 1
    return hi

def _perfect_power_factor(N):
    for k in range(2, N.bit_length() + 1):
        root = _integer_nth_root(N, k)
        if root > 1 and root ** k == N:
            return root
    return None

# ---- the period-finder: the quantum subroutine, here on a classical
#      simulator so the logic can be checked on small N (e.g. N=15, N=91) ----
def find_period(x, N):
    q_bits = (N * N - 1).bit_length()     # q a power of 2 with N^2 <= q < 2 N^2
    q = 1 << q_bits
    # 1. uniform superposition over a in [0,q):   q^{-1/2} sum_a |a>|0>
    # 2. reversible modular exponentiation:        |a>|x^a mod N>
    #    (the function x^a mod N is periodic in a with period r = ord_N(x))
    # 3. quantum Fourier transform on the first register  ->  amplitude
    #    on |c> equals (1/q) sum_{a: x^a == x^k} exp(2 pi i a c / q)
    # 4. measure c; good pairs have probability at least 1/(3*r^2)
    c = _simulate_qft_measurement(x, N, q)
    # continued fractions give a candidate denominator; verify it is an order
    frac = Fraction(c, q).limit_denominator(N - 1)
    candidate = frac.denominator           # may be r / gcd(d, r), not r
    if candidate > 0 and modexp(x, candidate, N) == 1:
        return candidate
    return None

def _simulate_qft_measurement(x, N, q):
    # exact classical simulation of steps 1-4 for small N
    # group a's by x^a mod N (the entanglement with register 2),
    # then DFT within each group and sample |.|^2 over all c.
    groups = defaultdict(list)
    value = 1
    for a in range(q):
        groups[value].append(a)
        value = (value * x) % N
    probs = [0.0] * q
    for alist in groups.values():
        for c in range(q):
            s = sum(complex(math.cos(2*math.pi*a*c/q),
                            math.sin(2*math.pi*a*c/q)) for a in alist)
            probs[c] += abs(s / q) ** 2
    t = random.random(); acc = 0.0
    for c in range(q):
        acc += probs[c]
        if t <= acc: return c
    return q - 1

# ---- factoring shell: Miller's reduction around find_period ----
def factor(N):
    if N < 2:
        raise ValueError("N must be at least 2")
    if N % 2 == 0: return 2
    pp = _perfect_power_factor(N)
    if pp is not None:
        return pp
    while True:
        x = random.randrange(2, N)
        g = gcd(x, N)
        if g != 1:
            return g                       # lucky common factor
        r = find_period(x, N)
        if r is None or r % 2 != 0:
            continue                       # need even order
        y = modexp(x, r // 2, N)
        if y == 1 or y == N - 1:
            continue                       # r odd/proper, or x^{r/2} == -1 mod N
        f = gcd(y - 1, N)
        if 1 < f < N:
            return f                       # gcd(x^{r/2}-1, N) is a factor
```
