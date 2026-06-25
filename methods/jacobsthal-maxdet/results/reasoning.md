I want the `29 × 29` sign matrix whose determinant is as large as possible. The first thing to
fix is what "as large as possible" can even mean here, because the answer depends entirely on `29
mod 4`. If the rows of a `±1` matrix were mutually orthogonal, the Gram matrix `HHᵀ` would be
`29·I`, the determinant would be `29^{29/2}`, the Hadamard ceiling, and I would be done. But the
rows have length `29`, which is odd, and the inner product of two `±1` vectors of odd length is a
sum of an odd number of `±1`'s — an odd integer, never zero. So no two rows can be orthogonal;
`HHᵀ` can never be `29·I`; the Hadamard bound is unreachable at this order. That is not a
detail, it is the whole character of the problem: I am not hunting for a perfect orthogonal
design that exists and is hard to find, I am pushing against a ceiling that provably cannot be
touched, where the best `±1` matrices are *almost* orthogonal and the question is how close.

So let me ask what "almost orthogonal" should look like. If I cannot make the off-diagonal inner
products zero, I want them as small and as uniform as possible. The off-diagonal entries of `HHᵀ`
are odd, so the smallest they can be in magnitude is `±1`. The cleanest target I can imagine is a
Gram matrix where every diagonal entry is `29` and every off-diagonal entry is exactly `−1`:
`HHᵀ = 29·I − J + I = 30·I − J`, or written another way, `HHᵀ = (n+1)I − J` with `n = 29`. A
matrix whose rows pairwise overlap by exactly `−1` is as near to orthogonal as parity allows.
Is such a thing even realizable, or am I wishing for an object that does not exist? A matrix `C`
with zero diagonal and `±1` off-diagonal satisfying `CCᵀ = (n−1)I` is what the literature calls a
*conference matrix*, and conference matrices of order `n+1` exist exactly when `n+1 ≡ 2 (mod 4)`
and the right number theory cooperates. For `n = 29`, `n + 1 = 30 ≡ 2 (mod 4)`, and `29` is a
prime `≡ 1 (mod 4)`. Those are precisely the conditions under which Paley's quadratic-residue
construction produces one, so the almost-orthogonal structure I want is not a wish — I can build
it explicitly and read a `±1` matrix off it.

The engine is the Legendre symbol. For the prime `q = 29`, define `χ(a)` to be `+1` if `a` is a
nonzero quadratic residue mod `q`, `−1` if it is a non-residue, and `0` if `a ≡ 0`. Build the
`q × q` matrix `Q` with `Q_{ij} = χ(i − j)` — the Jacobsthal matrix. Before I trust any of its
properties I should pin down two things by hand. First, is `Q` symmetric? `Q_{ji} = χ(j − i) =
χ(−(i − j))`, so symmetry hinges on whether `χ(−a) = χ(a)`, i.e. whether `−1` is itself a
quadratic residue. For an odd prime, `−1` is a residue exactly when `q ≡ 1 (mod 4)`; `29 = 4·7 + 1`
qualifies, so I expect `χ(−1) = +1` and `Q = Qᵀ`. Let me not just expect it — list the nonzero
residues mod 29: `{1, 4, 5, 6, 7, 9, 13, 16, 20, 22, 23, 24, 25, 28}`, fourteen of them, and `28 ≡
−1` is in the list. So `χ(−1) = +1`, confirmed, and `Q` is symmetric.

Second, what is `QQᵀ`? The diagonal of `QQᵀ` is `∑_k χ(i−k)²`, a sum over the `q − 1` indices where
`i − k ≠ 0`, each term `1`, giving `q − 1 = 28`. The off-diagonal `(i,j)` entry is `∑_k χ(i−k)χ(j−k)`,
a character sum that I would normally have to argue collapses to `−1`. Rather than wave at "the
multiplicative structure makes it cancel," I build `Q` and multiply it out. The diagonal of `QQᵀ`
comes back uniformly `28`, and every single off-diagonal entry comes back `−1` — which is exactly
`29·I − J`, since that matrix has `29 − 1 = 28` on the diagonal and `−1` off it. The whole product
equals `qI − J` on the nose. So `Q` already realizes the
almost-orthogonal Gram structure I was reaching for. But `Q` is not a `±1` matrix — its diagonal
is all zeros (`χ(0) = 0`). I have to fill those `29` diagonal zeros with `±1` to get a legal sign
matrix.

Fill them with `+1`. Set `R = Q + I`. Now every entry is `±1` (the diagonal is `0 + 1 = 1`, the
off-diagonal is the `±1` of `Q` unchanged), so `R` is a legal output — I confirm the entry set of
`R` is exactly `{−1, +1}`. What did that do to the Gram matrix? `RRᵀ = (Q + I)(Q + I)ᵀ = QQᵀ + Q +
Qᵀ + I = (qI − J) + 2Q + I`, using `Q = Qᵀ`, so `RRᵀ = (q+1)I − J + 2Q`; multiplying `R` out and
comparing confirms the identity entry-for-entry. The cross term `2Q` keeps the rows from being as
clean as `Q` alone, but the diagonal is now `q + 1 = 30`, the right magnitude, and the structure
is still highly regular.

Now I need the determinant of `R`, and I want it as an exact integer with its prime factorization,
not a float — the score is keyed to powers of `2` and `7`, so a number like `1.8 × 10^20` tells me
nothing. The clean way is through the spectrum of the symmetric `Q`. Two structural facts narrow
it down. The rows of `Q` sum to `∑_a χ(a) = 0` (a prime has equally many residues and non-residues
among `1…q−1`), so the all-ones vector is an eigenvector with eigenvalue `0`. And from `QQᵀ = qI −
J`, on the subspace orthogonal to all-ones where `J` acts as `0`, `Q²` acts as `qI`, so every other
eigenvalue is `±√q`. That gives a candidate spectrum: one `0`, and the remaining `q − 1 = 28`
eigenvalues split as `±√29`. I do not want to assume the split is even, so I check it directly: the
eigenvalues of `Q` for `q = 29` come back as one `0`, fourteen at `+√29`, and fourteen at `−√29`.

With that spectrum, `det(R) = det(Q + I) = ∏(λ + 1) = (0+1)·(1+√29)^{14}·(1−√29)^{14} =
[(1+√29)(1−√29)]^{14} = (1 − 29)^{14} = (−28)^{14} = 28^{14}`. And `28 = 2²·7`, so `28^{14} =
2^{28}·7^{14}`. That is a determinant whose factorization is forced to be exactly the structural
`2^{28}` times an extra `7^{14}` — which means the multiplier is `m = 2^{28}·7^{14} / (2^{28}·7^{12})
= 7² = 49`. I check this against the harness's exact Bareiss elimination on the actual `29 × 29`
matrix rather than trusting the float arithmetic: `|det(R)| = 182059119829942534144`, and
`182059119829942534144 = 2^{28}·7^{14}` exactly, `m = 49.0`, with the float eigenvalue product
agreeing to all the digits that single precision can carry. So the multiplier is `49`, and the
score is `49/342 ≈ 0.1433`.

Before I rely on the spectral derivation I want to know it is not an artifact of `q = 29`, so I run
the same construction at the smaller primes `≡ 1 (mod 4)`. At `q = 5`: `Q` symmetric, `QQᵀ = 5I −
J`, eigenvalues one `0` and four of magnitude `√5`, `det(Q + I) = 16 = (1−5)² = (−4)² = 2⁴`,
matching `(1−q)^{(q−1)/2}` exactly. At `q = 13`: `det(Q + I) = 2985984 = (1−13)^6 = (−12)^6`, again
the closed form. The pattern `det(Q + I) = (1 − q)^{(q−1)/2}` holds in every case I can compute, and
at `q = 29` it gives `(−28)^{14} = 2^{28}·7^{14}` — the same number Bareiss returns. That is the
verification I wanted: the multiplier `49` is not read blindly off a determinant routine, it is the
value the spectrum predicts and the exact integer arithmetic confirms.

There is a choice I glossed: I filled the diagonal with `+1`, but `−1` was equally available,
giving `R' = Q − I`. Is the diagonal sign a lever worth pulling? Through the spectrum, `det(Q − I) =
∏(λ − 1) = (0−1)·(√29−1)^{14}·(−√29−1)^{14} = −[(√29)² − 1]^{14} = −(28)^{14}`, the same magnitude as
`det(Q + I)` with the opposite sign. The small cases agree: `det(Q − I)` comes back `−16` at `q = 5`
and `−2985984` at `q = 13`, exactly negating the `Q + I` values. So `|det(Q + I)| = |det(Q − I)|`;
the diagonal sign moves only an overall sign, never the magnitude I am scored on. The construction
has no free parameter to tune — which is what I should expect from a rigid symmetric design, and it
means I cannot squeeze anything more out of this object by tweaking it.

Now the honest appraisal of where this lands, because the point of starting here is to know the
floor. This `R = Q + I` is a *textbook* object: the natural symmetric `±1` matrix attached to the
prime `29`, fully determined, no search. Its determinant is exactly `2^{28}·7^{14}`, multiplier
`49`. Against the yardsticks — Barba ceiling at multiplier `369.94`, human record at `320` — a
multiplier of `49` is about an eighth of the record and a seventh of the provable ceiling. That is
modest, and I think it is modest for a structural reason: the symmetry that makes the matrix
elegant is the same symmetry that pins it to a single configuration, and the maximal-determinant
records at this order are known to come from *breaking* such symmetry under search. So I am not
proposing this as a strong answer. I am proposing it as the *right* starting point: it is
principled (it realizes the almost-orthogonal Gram structure the parity obstruction allows), it is
provably legal, it is parameter-free, and it gives a concrete, exactly-known baseline determinant
that every later, searched construction must beat. The multiplier came out to `49` — and that tells
me precisely how far there is to go: the entire distance from `49` to `320` has to be bought by
search that deliberately leaves the symmetry behind. The structure gets me onto the board; it will
not get me near the frontier, and any next step has to start spending the determinant I cannot
construct in closed form.
