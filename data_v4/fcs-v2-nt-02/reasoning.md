I am counting colorings up to symmetry — `n` beads on a circle, `k` colors, two colorings identified
when some rotation or reflection carries one to the other, so the acting group is the dihedral group
`D_n` with its `2n` elements, and I want the orbit count modulo `p = 1000000007`. What separates this
from a textbook Burnside exercise is the scale printed in the constraints: `n` reaches `10^9` under a
1-second limit. That single number kills the naive method before I write it. I cannot afford to touch
each of the `n` rotations individually, so the rotation contribution has to fall out of the *structure*
of `n` — its divisors — rather than an enumeration. Everything below is organized around dodging that
`O(n)` wall.

Burnside (Cauchy–Frobenius) is the frame: the number of orbits equals the average number of fixed
colorings over the group, `answer = (1/(2n)) * (sum over rotations Fix + sum over reflections Fix)`.
The two halves are independent, and the whole difficulty is in evaluating each sum cheaply.

A rotation by `d` positions, read as a permutation of the beads, splits them into `gcd(d,n)` cycles of
length `n/gcd(d,n)`; a coloring survives it iff every cycle is monochromatic, so it fixes
`k^{gcd(d,n)}` colorings. The rotation sum is therefore `sum_{d=0}^{n-1} k^{gcd(d,n)}`. Correct and
standard — and, written this way, `O(n)` modular exponentiations, `10^9` of them, each power itself
~30 multiplies: on the order of `10^{10}` operations, hopeless in a second. This is the sum I have to
collapse.

The collapse: in `sum_{d=0}^{n-1} k^{gcd(d,n)}` the exponent `gcd(d,n)` only ever takes values that
**divide `n`**. So I sum over the divisors instead of the offsets, weighting each `k^g` by how many
`d` produce `gcd(d,n) = g`. Writing `d = g*t`, the condition `gcd(g*t, n) = g` is exactly
`gcd(t, n/g) = 1` with `t` ranging over `0..(n/g)-1`, and the count of such `t` is Euler's totient
`phi(n/g)` (the offset `d = 0` lands in `g = n` with `phi(1) = 1`, contributing `k^n` — the identity
fixing everything, as it must). Hence

```
rotation part = sum_{g | n} phi(n/g) * k^{g}.
```

Any `n` below `10^9` has at most ~1344 divisors, and I enumerate them in `O(sqrt n)` — about `31623`
candidates at the top of the range, pairing each `d <= sqrt n` with `n/d`. A billion-term sum becomes
a few thousand terms; the `O(n)` wall is gone. Re-indexing `g -> n/g` gives the equivalent
`sum_{g|n} phi(g) k^{n/g}`; I take the first form and, per divisor `d`, compute `phi(n/d) * k^d`.

I cannot sieve totients up to `10^9`, but `phi(m)` of a single `m` is cheap by trial division: divide
out each prime up to `sqrt m` and fold in `result -= result / q` per distinct prime `q`, `O(sqrt m)`
per call. Because `m = n/d` shrinks quickly as `d` grows, the total across all divisors sits well
under the crude `(#divisors) * sqrt n` bound.

The reflections are the error-prone half — their cycle count turns on the parity of `n` and on whether
the mirror axis passes through beads or through gaps, the classic spot for an off-by-one, so I count
cycles concretely. For `n` odd, every axis runs through one bead and the opposite gap: that bead is a
fixed point, the remaining `n-1` beads pair into `(n-1)/2` transpositions, giving
`1 + (n-1)/2 = (n+1)/2` cycles. All `n` axes are of this one type, so the odd reflection part is
`n * k^{(n+1)/2}`. For `n` even there are two axis types, `n/2` of each: through two opposite beads
(two fixed points plus `(n-2)/2` transpositions, `n/2 + 1` cycles, `k^{n/2+1}`), and through two
opposite gaps (all beads pair, `n/2` cycles, `k^{n/2}`). The even reflection part is
`(n/2)(k^{n/2+1} + k^{n/2})`.

These parity formulas are the most fragile piece, so I run the whole assembly on `n = 4, k = 2`, whose
count is `6`. Rotations: `phi(4)k^1 + phi(2)k^2 + phi(1)k^4 = 2*2 + 1*4 + 1*16 = 24`. Reflections
(even, `n/2 = 2`): `2*k^3 + 2*k^2 = 16 + 8 = 24`. Total `48`, over `2n = 8`, is `6` — the parity cases
land.

Dividing by `|G| = 2n` under a prime modulus means multiplying by `inverse(2n) = (2n)^{p-2} mod p`,
which exists iff `gcd(2n, p) = 1`. Here the fixed modulus earns its exact value: `p = 10^9 + 7` is odd
and strictly larger than every allowed `n <= 10^9`, so `p` divides neither `2` nor `n`, and the
inverse is defined on every input. Had the contract handed a prime below `10^9` — say `998244353` —
some valid `n` could be a multiple of it, the inverse would vanish, and the division would silently
break. That `p > n` is load-bearing, not cosmetic. Two smaller arithmetic points: `k` up to `10^9` I
reduce once as `kk = k % p` so every exponentiation runs on a reduced base; and `2*(n mod p)` reaches
~`2*10^9`, past 32 bits, so I keep `long long` throughout and multiply through `__int128` inside
`power_mod`.

The divisor loop has one real trap for this problem, because the tests include perfect-square `n` and
`n = 1`. Enumerating divisors by pairing each `d <= sqrt n` with `n/d` double-counts when the two
coincide — the middle divisor of a square (`d = sqrt n`), and `n = 1` where `d = n/d = 1`. Add both
terms unconditionally and a square counts its middle divisor twice: for `n = 1, k = 7` the rotation
sum should be `phi(1)*7 = 7` but comes out `14`. So I add the complement term only when `n/d != d`.
That is exactly the skip the `n = 4` check above wanted: `d = 1` contributes both `phi(4)*2 = 4` and
`phi(1)*2^4 = 16`, while `d = 2 = n/d` contributes only `phi(2)*4 = 4` and no complement — recovering
`rotSum = 24`.

A couple of degenerate inputs exercise the assembly rather than just the loop. `k = 1`: every power is
`1`, so the rotation part is `sum phi(n/g) = n` and the reflection part is `n` as well, total `2n` over
`2n` — one bracelet for any `n`, as it must be. `n = 2, k = 2`: rotations `phi(2)*2 + phi(1)*4 = 6`,
reflections (even, `n/2 = 1`) `2 + 4 = 6`, total `12/4 = 3` (the colorings 00, 01, 11). Cost is
dominated by the `~31623`-step `sqrt n` scan whether `n` is prime (two divisors) or divisor-rich
(~1344 near `10^9`), plus one `phi`/`power_mod` pair per divisor — inside the second either way.

For correctness I lean on two independent references. A brute force that, for tiny `n, k`,
canonicalizes each of the `k^n` colorings under all `n` rotations and all `n` reflections and counts
distinct canonical forms — pure orbit enumeration, no Burnside, no divisor sums — agrees on random
`(n, k)` with `n <= 12, k <= 7` and on the explicit edges. And a second reference that computes the
same Burnside answer the slow way, `sum_{d=0}^{n-1} k^{gcd(d,n)}` term by term, agrees with the
divisor-sum solution for `n` up to a few million across many `k`, including `k` near `p` — reaching
the large-`n` regime the small brute cannot touch and pinning down that the regrouping computes
exactly the same number.
