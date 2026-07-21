The size sets the ceiling: `n` up to `2*10^5` with a 2-second limit, so a per-candidate `O(n)`
scan over every `p` — the naive `O(n^2)` — is `4*10^10` operations and hopeless. Whatever I do
has to be near-linear, `O(n log n)` at the outside. And the wildcard is what makes the "obvious"
near-linear route treacherous, which is the real heart of this problem: `?` matches anything, so
the usual period machinery quietly breaks. I pin down the definition first, then find where the
shortcut fails, because the failure names the obstruction.

Period `p` means the filled string repeats a block of length `p`: the indices split into residue
classes mod `p`, and within a class every position must end up the same letter. A `?` is free and
adopts whatever its class settles on; a concrete letter pins its class. So `p` is a period iff
every residue class mod `p` contains at most one distinct concrete letter — every class
monochromatic. That is exact and obviously correct; the only question is testing it fast across
all `p`.

The tempting shortcut is the KMP border/period test with `?` treated as matching anything: `p` is
a period iff `s[i]` and `s[i+p]` match for every `i`, and then read the answer off as
`n - pi[n-1]` — three lines, `O(n)`. But the prefix function's correctness rests on equality being
transitive, and wildcard-match is not. The smallest string that exposes it is `s = b?a`. Test
`p = 1`: pairwise, `s[0]=b` vs `s[1]=?` matches, `s[1]=?` vs `s[2]=a` matches, both pass, so the
test declares `p=1` and reports `1`. But under `p=1` all three positions are one class and must
fill to a single letter; position 0 is `b`, position 2 is `a`, and the lone `?` cannot be both. No
fill works — the true answer is `3`. The chain `b`→`?`→`a` was allowed even though `b` and `a`
themselves clash. Wildcard matching is not transitive, and the pairwise/prefix test silently
assumes it is. That kills the fast route, and it tells me exactly what the correct test must
catch: two concrete letters that clash inside a class can sit at distance `2p`, `3p`, ... with
only `?`s between them (`b?a` is precisely this — `b` and `a` at distance 2 with class step 1).

So the complete condition is: `p` is a period iff for every distance `q` that is a multiple of `p`
(`q < n`), no index `i` has `s[i]` and `s[i+q]` both concrete and different. Re-check `b?a` at
`p=1`: the multiples are `q=1,2`; at `q=2`, `s[0]=b` and `s[2]=a` clash, so `p=1` is rejected.
Including all multiples is the transitive closure the pairwise scan was missing.

The leverage comes from separating that condition from `p`. Define, for each shift `q` in
`[1,n)`, `compatible[q]` = "no index `i` has `s[i]`, `s[i+q]` both concrete and different." This
depends only on `q`, not on `p`. The period test then reads: `p` is a period iff `compatible[q]`
holds for every multiple `q=p,2p,...` below `n`. Given the whole array `compatible[1..n-1]`,
finding the smallest period is a sieve — for `p=1,2,3,...` test `compatible[p], compatible[2p],
...`, and the first `p` passing all its multiples wins — at cost `sum_p n/p = O(n log n)`, the
harmonic series. Within budget. So everything reduces to one question: compute `compatible[q]`
for all `q` in `O(n log n)`.

That is exactly wildcard self-matching — does `s` agree with `s` shifted by `q` everywhere both
are concrete? The tool for all offsets at once is the FFT mismatch trick. Encode letters as
`a_i in {1..26}`, with `a_i=0` for `?`, and `act_i=[a_i!=0]`. The concrete-mismatch count at
shift `q` is `mismatch(q) = sum_i act_i*act_{i+q}*(a_i-a_{i+q})^2`: a pair contributes iff both
ends are concrete and the letters differ, so `compatible[q]` holds iff `mismatch(q)==0`. Since
`act_i a_i = a_i` (and `act_i a_i^2 = a_i^2`), expanding the square gives
`mismatch(q) = sum_i [a_i^2 act_{i+q} - 2 a_i a_{i+q} + act_i a_{i+q}^2]` — three
cross-correlations of fixed sequences, `(a^2,act)`, `(a,a)`, `(act,a^2)`, each one FFT
multiplication over all shifts. Three FFTs yield every `mismatch(q)`, then the harmonic sieve
reads off the answer. Total `O(n log n)`.

The one place to be careful is the correlation itself. I want `corr[q] = sum_i u[i] v[i+q]`, but
`ifft(fft(u) .* fft(v))` computes the convolution `sum_i u[i] v[k-i]`, which reverses one operand;
a correlation needs one factor conjugated in frequency space, `corr = ifft(conj(FU) .* FV)`. Getting
this backwards is silent when wrong, so I pin the formula with a two-element trace: for
`u=[1,2,3], v=[4,5,6]`, `corr[1]` should be `u[0]v[1]+u[1]v[2]=5+12=17`,
whereas the plain product gives the convolution value `u[0]v[1]+u[1]v[0]=5+8=13`. `17` is what I
need, so the conjugate-product form is the correct one. The cyclic wrap-around from padding to
`sz>=2n` lands in indices `q>=n`, which the sieve never reads, so it cannot contaminate the
answers.

Floating point is the remaining soft spot: `mismatch(q)` comes back as a `double` I round with
`llround`. A single term reaches at most `26^2 * n ≈ 1.4*10^8`, orders of magnitude below the
`~9*10^15` where `double` loses integer exactness, so the rounding is safe. That estimate settles
it, but I still confirm the reported `p` independently of the FFT: build the canonical letter of
each residue class mod `p`, reject if any class holds two different concrete letters, then tile
that block across the string and check with two rolling hashes (different moduli, and the tiled
string versus the `?`-filled string built from different index expressions) that they agree on
every concrete position. If this ever disagreed with the sieve, fall back to a direct per-class
scan over all `p` — it should never fire.

The edge cases fall out of the same structure. `n=1`: a single position is trivially period `1`,
handled before the FFT so there is no degenerate transform. All wildcards `?...?`: every
`mismatch(q)=0`, every shift compatible, sieve returns `1`. No valid period below `n` (e.g. `n`
distinct letters): every shift has a clash, nothing survives, and `ans` stays at its initial `n` —
the initialization is what encodes the always-valid `p=n` fallback.

The three FFT correlations and the harmonic sieve are each `O(n log n)`, the hash check one `O(n)`
pass, and memory is `O(n)`. At `n=2*10^5` this stays well inside the 2-second, 256-MB budget even
on the worst case — `n` distinct letters, where the answer is `n` and the sieve runs to the end.
The full self-contained C++ program is the answer.
