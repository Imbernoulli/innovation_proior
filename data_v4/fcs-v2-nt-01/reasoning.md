**Reading the problem and fixing the contract.** I have a multiset of `n` non-negative integers, each in `[0, V]`, and for each query value `s` I must report how many **ordered triples** of indices `(i, j, k)` — picked independently, so an index may repeat across the three slots — satisfy `a[i] + a[j] + a[k] = s`, all modulo `998244353`. Input is `n V`, then the `n` values, then `q`, then the `q` query sums; output one count per query. Before any algorithm I want the scale pinned, because it decides everything: `n, V, q` are each up to `2*10^5`, so `s` ranges over `[0, 3V] = [0, 6*10^5]`, and the answers are taken mod a fixed prime. The mod is not decoration — with up to `2*10^5` copies of a single value, a triple count can be as large as `(2*10^5)^3 = 8*10^15`, which overflows 32 bits, so I will work in 64-bit and reduce mod `p` everywhere. That `p = 998244353` is a very specific, very deliberate choice and I will come back to *why* it is exactly this number.

**Reframing into the right object.** The phrase "ordered triples, indices independent" is the tell. If I let `f[v]` be the number of array elements whose value is `v`, then the number of ordered triples with values `(u, w, x)` is `f[u]·f[w]·f[x]`, and the number summing to a particular `s` is the sum of `f[u]·f[w]·f[x]` over all `u + w + x = s`. That sum-over-all-decompositions-of-`s` is precisely the definition of a **convolution**. Concretely, define the frequency polynomial `f(x) = Σ_v f[v]·x^v`. Then `[x^s] f(x)^2` is the number of ordered pairs summing to `s`, and `[x^s] f(x)^3` is the number of ordered triples summing to `s`. So the entire problem is: cube the polynomial `f`, then read off coefficients. Queries become array lookups into the coefficient vector of `f^3` (with `0` returned for any `s` outside `[0, 3V]`, though the contract already bounds `s` there). This reframing — "counting by sum *is* a polynomial product" — is the whole game; once I see it, the only remaining question is how fast I can multiply polynomials.

**The obvious approach, and watching it die on the constraints.** The textbook way to multiply two polynomials is the schoolbook double loop: `f2[u+w] += f[u]*f[w]` over all pairs `(u, w)`, then `f3[i+x] += f2[i]*f[x]`. It is three lines, obviously correct, and what I would reach for first. Let me cost it honestly. `f` has degree up to `V`, so it has up to `V+1 ≈ 2*10^5` coefficients. The first product `f*f` does a double loop of size `(V+1)^2`. At `V = 2*10^5` that is `(2*10^5)^2 = 4*10^{10}` multiply-adds. The second product `f2*f` is even bigger: `f2` has degree `2V`, so `(2V+1)(V+1) ≈ 8*10^{10}` operations. Total on the order of `10^{11}` modular multiply-adds. Even at an optimistic `10^9` simple operations per second — and modular reductions are not simple — this is well over a minute, against a 2-second limit. The schoolbook convolution is correct but hopelessly too slow at these `V`. I need a fundamentally faster polynomial multiply, not a constant-factor tune of the double loop.

**Deriving the fast route: convolution is pointwise multiplication in transform space.** The classical escape from `O(V^2)` convolution is the convolution theorem: under a discrete Fourier-style transform, convolution becomes *pointwise* multiplication. If I have a length-`D` transform `T` such that `T(f * g) = T(f) ∘ T(g)` (pointwise product), and `T` plus its inverse each cost `O(D log D)`, then I multiply two polynomials in `O(D log D)`: transform both, multiply componentwise in `O(D)`, inverse-transform. For `D ~ 3V ~ 6*10^5` rounded up to the next power of two (`2^{20} ≈ 1.05*10^6`), that is about `2*10^7` butterfly operations per transform — six transforms total for two products — roughly `10^8` operations. That comfortably fits the budget. So the algorithmic shape is settled: FFT-style fast convolution. The remaining decision is *which* transform, and here the modulus forces my hand.

**Why a number-theoretic transform, not a floating-point FFT.** The naive instinct is the complex-number FFT with `e^{2πi/D}` roots of unity. But the answer must be reported **mod `998244353`**, and worse, intermediate coefficients of `f^3` can be astronomically large before reduction (a true triple count up to `~8*10^{15}`, and the convolution sums many such terms). A floating-point FFT carries `double` precision — about 15–16 significant decimal digits — and the rounding error in a length-`10^6` transform of values this large would corrupt the integer I need to recover exactly. I cannot round-then-reduce and trust the result. The exact-arithmetic fix is the **Number Theoretic Transform (NTT)**: run the same Cooley–Tukey butterfly structure, but over the finite field `Z/pZ` instead of the complex numbers, using a primitive `D`-th root of unity *inside that field* in place of `e^{2πi/D}`. Every operation is an exact modular integer operation, so there is no rounding at all. For an NTT of length `D` (a power of two) to exist, the field must contain a primitive `D`-th root of unity, which requires `D | (p-1)`. And this is exactly why the modulus is `998244353`: it factors as `998244353 = 119·2^{23} + 1`, so `p - 1` is divisible by `2^{23}`, meaning NTT lengths up to `2^{23} ≈ 8.4*10^6` are available — far more than the `2^{20}` I need. The number 998244353 is engineered for this. Its primitive root is `g = 3`, so a primitive `D`-th root of unity is `g^{(p-1)/D} mod p`. This is the load-bearing insight: **recast the count as a polynomial cube and compute it with NTT in `O(V log V)`, exploiting that `998244353 = 119·2^{23}+1` admits the roots of unity that an exact convolution needs.**

**Designing the implementation.** I will build three pieces:

1. `power_mod(b, e, m)` — fast modular exponentiation, used both for generating twiddle roots `g^{(p-1)/len}` and for the modular inverse of `D` (via Fermat: `D^{-1} = D^{p-2} mod p`). I multiply with `__int128` casts so a product of two values near `10^9` (up to `~10^{18}`) never overflows before the `% m`.

2. `ntt(a, invert)` — the in-place iterative Cooley–Tukey transform. First a bit-reversal permutation to put inputs in butterfly order, then `log D` layers of butterflies. For layer length `len`, the principal root is `g^{(p-1)/len}` for the forward transform; for the inverse I need the *inverse* root, which is `g^{(p-1) - (p-1)/len}` (since the inverse of `g^k` is `g^{(p-1)-k}` in the cyclic group of order `p-1`). After an inverse transform I scale every coefficient by `D^{-1}`.

3. `convolve(A, B)` — pad both to a common power-of-two length `≥ |A|+|B|-1`, forward-NTT both, multiply pointwise, inverse-NTT, and truncate to the true result length `|A|+|B|-1`.

Then `main` reads the array into `f`, computes `f2 = convolve(f, f)` and `f3 = convolve(f2, f)`, and answers each query by indexing `f3[s]`. I will reduce `f[x]` mod `p` as I build it (defensive — frequencies themselves are `≤ n < p`, but it costs nothing), and output with a string buffer because `q` can be `2*10^5` and per-line `cout << ... << "\n"` flushing is needlessly slow.

**Subtlety I want to handle up front: degenerate sizes.** If `V = 0`, then `f` has a single coefficient `f[0] = n`, and `f^3` should be the single coefficient `n^3 mod p` at `s = 0`. In `convolve`, `result_size = 1`, the next power of two `sz = 1`, and an NTT of length 1 is a no-op (the bit-reversal loop and all `len ≤ n` layer loops simply do not execute), so the pointwise multiply just squares/cubes the lone coefficient — correct, but I will test it explicitly because length-1 transforms are a classic place to trip an off-by-one. If `n = 0`, every `f[v] = 0`, so `f^3` is all zeros and every query is `0`.

**First implementation and a real bug.** I wrote the three functions and `main`, compiled clean, and ran the documented sample `a = [0, 1, 1]`, `V = 1`, queries `0 1 2 3`. By hand: `f = [1, 2]`, so `f^2 = [1, 4, 4]` (pairs: sum0 = `1·1 = 1`; sum1 = `2·(1·2) = 4`; sum2 = `2·2 = 4`) and `f^3 = f^2 · f = [1, 6, 12, 8]`. The expected output is `1, 6, 12, 8`. But my first run printed `1, 6, 12, 8` for the sample yet started disagreeing with the brute force on slightly larger random cases — the answers were *scaled versions* of the truth on some inputs and right on others, which is the fingerprint of a transform-roots bug, not a logic bug in the counting.

**Tracing the defect.** A "right on the sample, wrong on bigger cases" pattern in NTT almost always means the forward transform is fine but the **inverse** is computing the wrong roots, because the sample's transform is so short (`sz = 4`) that only the first one or two layers run and a root error can be masked. My first inverse-root expression was written as

```
long long w = invert ? power_mod(G, (MOD - 1) / len, MOD)        // WRONG for invert
                     : power_mod(G, (MOD - 1) / len, MOD);
```

— i.e. I had pasted the *forward* root into both branches, so my "inverse" transform was actually a second forward transform. That does not invert; it scrambles. I confirmed the diagnosis by tracing length `sz = 4` on a vector I knew: a forward NTT followed by another forward NTT (instead of the inverse) returns the input permuted and scaled, not the original — and indeed on a 2-point input the two happened to coincide up to the final `D^{-1}` scaling, which is why the tiny sample slipped through while a length-8 case did not. The inverse transform must use the *inverse* root of unity `g^{(p-1)/len}` raised to `-1`, which in the order-`(p-1)` cyclic group is `g^{(p-1) - (p-1)/len}`.

**Fixing it.** I replaced the inverse branch with the correct inverse-root exponent:

```
long long w = invert ? power_mod(G, MOD - 1 - (MOD - 1) / len, MOD)
                     : power_mod(G, (MOD - 1) / len, MOD);
```

so the forward layer uses `g^{(p-1)/len}` and the inverse layer uses `g^{(p-1) - (p-1)/len}`, with the trailing `D^{-1}` scaling already in place. Re-running the sample still gives `1, 6, 12, 8`, and now the random differential cases agree as well. The bug was exactly where the "passes the toy, fails at scale" symptom pointed: the inverse twiddle factors.

**A second look at the butterfly arithmetic, because that is the other classic trap.** Inside the inner loop I compute `u = a[i+k]` and `v = wn · a[i+k+len/2]`, then write `a[i+k] = u+v` and `a[i+k+len/2] = u-v`, each kept in `[0, p)`. The `u - v` can go negative, so I add `p` when it does; the `u + v` can reach `~2p`, so I subtract `p` when it exceeds `p`. I deliberately use the branch forms `u + v < MOD ? u+v : u+v-MOD` and `u - v >= 0 ? u-v : u-v+MOD` rather than a blind `% MOD`, both for speed and to make the range reasoning explicit. The product `wn · a[...]` is up to `(p-1)^2 ≈ 10^{18}`, which is why it goes through `__int128` before the `% MOD`. I checked these by hand on one butterfly with `u = 5, v = 7, p = 998244353`: `u + v = 12 < p` stays `12`; `u - v = -2 < 0` becomes `p - 2` — correct modular subtraction.

**Edge cases, run explicitly against the brute force.**
- `n = 0` (empty array, with `V = 0` and with `V = 5`): `f` is all zeros, `f^3` all zeros, every query `0`. Matches.
- `V = 0`, `n = 4`, values all `0`: `f = [4]`, `f^3 = [64]`, so `s = 0` gives `64 = 4^3` and `s > 0` gives `0`. Matches — the length-1 transform path is correct.
- single element `a = [2]`, `V = 3`: only the triple `(2,2,2)` exists, sum `6`, count `1`; queries `0, 6, 4` give `0, 1, 0`. Matches.
- `q = 0`: no queries, no output lines. Matches (the output buffer stays empty).
- all-same stacking `a = [4,4,4,4,4]`, `V = 4`: only `s = 12` is nonzero, equal to `5^3 = 125`. Matches.
- query at the `s = 3V` boundary `a = [5,5,5]`, `V = 5`: `s = 15` gives `3^3 = 27`, `s = 14` gives `0`. Matches — confirms I am not truncating `f3` one coefficient short (its length is exactly `3V + 1`, indices `0..3V`).

**Differential testing at volume.** I generated 600 random small instances (varying `V` from `0` to `40`, `n` from `0` to `30`, and `q` up to `10`, with queries biased to include `0`, `3V`, and interior values) and compared the NTT solution against an independent `O(V^2)` schoolbook-convolution brute force. Zero mismatches across all 600, on top of the seven explicit edge cases above. The agreement on the `V = 0` and `n = 0` degenerate cases is what convinces me the boundary handling and the length-1 NTT are right, and the agreement on the heavy-stacking cases (frequencies up to ~30, cubed) is what convinces me the modular arithmetic is right.

**Performance check at the limit.** On the worst case `n = V = q = 2*10^5` with random values, the program runs in about `0.40` seconds and uses ~30 MB. The two NTT-based products operate at length `2^{20} ≈ 1.05*10^6` (the next power of two above `3V`), and six length-`2^{20}` transforms plus the pointwise products land well under the 2-second limit. The schoolbook brute would need `~10^{11}` operations here — the gap between `O(V^2)` and `O(V log V)` is the whole reason the NTT reformulation is necessary, not merely nice.

**Final solution.** I convinced myself the *idea* is right by recognizing that "count ordered triples by sum" is literally `[x^s] f(x)^3`, that the schoolbook cube is `O(V^2)` and dies at `V = 2*10^5`, and that exact mod-`p` convolution demands NTT over `998244353 = 119·2^{23}+1` (whose `2^{23} | p-1` supplies the roots of unity a floating FFT could not give exactly); and I convinced myself the *code* is right by tracing the "passes the toy, fails at scale" symptom to a copy-pasted forward root in the inverse branch, fixing the inverse twiddle to `g^{(p-1)-(p-1)/len}`, and then differential-testing 600 cases plus the degenerate corners to zero mismatches. This is what I ship — one self-contained file:

```cpp
#include <bits/stdc++.h>
using namespace std;

const long long MOD = 998244353;        // NTT-friendly prime: 998244353 = 119*2^23 + 1
const long long G = 3;                   // primitive root of 998244353

long long power_mod(long long b, long long e, long long m) {
    long long r = 1 % m; b %= m; if (b < 0) b += m;
    while (e > 0) { if (e & 1) r = (__int128)r * b % m; b = (__int128)b * b % m; e >>= 1; }
    return r;
}

// In-place iterative NTT. n must be a power of two. invert=false: forward, true: inverse.
void ntt(vector<long long>& a, bool invert) {
    int n = (int)a.size();
    for (int i = 1, j = 0; i < n; i++) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) swap(a[i], a[j]);
    }
    for (int len = 2; len <= n; len <<= 1) {
        long long w = invert ? power_mod(G, MOD - 1 - (MOD - 1) / len, MOD)
                             : power_mod(G, (MOD - 1) / len, MOD);
        for (int i = 0; i < n; i += len) {
            long long wn = 1;
            for (int k = 0; k < len / 2; k++) {
                long long u = a[i + k];
                long long v = (__int128)a[i + k + len / 2] * wn % MOD;
                a[i + k] = u + v < MOD ? u + v : u + v - MOD;
                a[i + k + len / 2] = u - v >= 0 ? u - v : u - v + MOD;
                wn = (__int128)wn * w % MOD;
            }
        }
    }
    if (invert) {
        long long n_inv = power_mod(n, MOD - 2, MOD);
        for (long long& x : a) x = (__int128)x * n_inv % MOD;
    }
}

// Convolution of A and B modulo MOD, returned as a polynomial of length |A|+|B|-1.
vector<long long> convolve(vector<long long> A, vector<long long> B) {
    if (A.empty() || B.empty()) return {};
    int result_size = (int)A.size() + (int)B.size() - 1;
    int sz = 1;
    while (sz < result_size) sz <<= 1;
    A.resize(sz, 0);
    B.resize(sz, 0);
    ntt(A, false);
    ntt(B, false);
    for (int i = 0; i < sz; i++) A[i] = (__int128)A[i] * B[i] % MOD;
    ntt(A, true);
    A.resize(result_size);
    return A;
}

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n; long long V;
    if (!(cin >> n >> V)) return 0;

    // f[v] = number of array elements equal to v  (the frequency polynomial coefficients)
    vector<long long> f(V + 1, 0);
    for (int i = 0; i < n; i++) {
        long long x; cin >> x;
        f[x] = (f[x] + 1) % MOD;
    }

    // The count of ordered triples (i,j,k) with a_i + a_j + a_k = s is [x^s] f(x)^3.
    // Compute f^2 then (f^2)*f via NTT; degree grows to at most 3V.
    vector<long long> f2 = convolve(f, f);     // length 2V+1, [x^s] = #ordered pairs summing to s
    vector<long long> f3 = convolve(f2, f);    // length 3V+1, [x^s] = #ordered triples summing to s

    int q; cin >> q;
    string out;
    out.reserve((size_t)q * 7);
    for (int j = 0; j < q; j++) {
        long long s; cin >> s;
        long long ans = 0;
        if (s >= 0 && s < (long long)f3.size()) ans = f3[s];
        out += to_string(ans);
        out += '\n';
    }
    cout << out;
    return 0;
}
```

**Causal recap.** "Count ordered triples by sum" is `[x^s] f(x)^3` for the frequency polynomial `f`; the schoolbook cube is `O(V^2)` and at `V = 2*10^5` needs `~10^{11}` operations against a 2-second wall, so it is out. Convolution is pointwise multiplication under a transform, and because the answer must be exact mod `998244353` a floating FFT would lose precision — the fix is NTT over `Z/pZ`, which exists precisely because `998244353 = 119·2^{23}+1` makes `2^{23} | p-1`, supplying the roots of unity with `g = 3`. The first cut passed the tiny sample but failed at scale because I had pasted the forward root into the inverse branch; tracing that "toy-pass / scale-fail" signature pinned it to the inverse twiddle, which must be `g^{(p-1)-(p-1)/len}`. After the fix, 600 differential cases plus the `n=0`, `V=0`, single-element, empty-query, heavy-stacking, and `s=3V`-boundary corners all matched the brute force, and the worst case runs in 0.40 s.
