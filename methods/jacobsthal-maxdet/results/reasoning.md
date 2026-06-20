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
This is not an arbitrary wish — it is the defining property of a known object. A matrix `C` with
zero diagonal and `±1` off-diagonal satisfying `CCᵀ = (n−1)I` is a *conference matrix*, and
conference matrices of order `n+1` exist exactly when `n+1 ≡ 2 (mod 4)` and the right number
theory cooperates. For `n = 29`, `n + 1 = 30 ≡ 2 (mod 4)`, and `29` is a prime `≡ 1 (mod 4)`, so
the Paley construction hands me a symmetric conference matrix of order `30` for free. The almost-
orthogonal structure I want is realizable; I just have to build it and read off a `±1` matrix
from it.

The engine is the Legendre symbol. For the prime `q = 29`, define `χ(a)` to be `+1` if `a` is a
nonzero quadratic residue mod `q`, `−1` if it is a non-residue, and `0` if `a ≡ 0`. Build the
`q × q` matrix `Q` with `Q_{ij} = χ(i − j)` — the Jacobsthal matrix. Two facts about it I can
lean on. First, because `q ≡ 1 (mod 4)`, `−1` is itself a quadratic residue, so `χ(−a) = χ(a)`,
which makes `Q` *symmetric*. Second, the multiplicative structure of the residues gives the
identity `QQᵀ = qI − J` exactly: the diagonal of `QQᵀ` is `q − 1` from the `q − 1` nonzero terms,
and every off-diagonal entry is `−1` because the character sums of differences cancel to exactly
that. So `Q` already has the almost-orthogonal Gram structure I was reaching for — but `Q` is not
a `±1` matrix, it has zeros on the diagonal. I have to fill those `29` diagonal zeros with `±1` to
get a legal sign matrix.

Fill them with `+1`. Set `R = Q + I`. Now every entry is `±1` (the diagonal is `0 + 1 = 1`, the
off-diagonal is the `±1` of `Q` unchanged), so `R` is a legal output. What did that do to the
Gram matrix? `RRᵀ = (Q + I)(Q + I)ᵀ = QQᵀ + Q + Qᵀ + I = (qI − J) + 2Q + I`, using `Q = Qᵀ`. The
cross term `2Q` keeps the rows from being as clean as `Q` alone, but the diagonal is now `q + 1 =
30`, the right magnitude, and the structure is still highly regular. The determinant of `R`
factors through the spectrum of `Q`. The eigenvalues of the symmetric `Q` are known in closed
form for the quadratic-residue construction: one eigenvalue tied to the all-ones direction, and
the rest split into two conjugate families with magnitude `√q`. Pushed through `R = Q + I`, the
determinant comes out to a clean integer with a heavy power-of-small-primes factorization. I do
not need to track every eigenvalue by hand; I will compute `det(R)` exactly with the harness's
Bareiss elimination and read the multiplier off.

There is a choice I glossed: I filled the diagonal with `+1`, but `−1` was equally available,
giving `R' = Q − I`. By the symmetry of the residues these two are determinant-twins —
`|det(Q + I)| = |det(Q − I)|` — because negating the identity shift corresponds to an overall
sign and conjugation that the quadratic-residue spectrum is symmetric under. So the diagonal sign
is not a lever worth pulling; both give the same magnitude. Good — that means the construction
has no free parameter to tune, which is exactly what I expect from a rigid symmetric design.

Now the honest appraisal of where this lands, because the point of starting here is to know the
floor. This `R = Q + I` is a *textbook* object: it is the natural symmetric `±1` matrix attached
to the prime `29`, fully determined, no search. Its determinant will be a specific multiplier of
`2^28 · 7^12`, and I expect it to be modest — the symmetry that makes it elegant is the same
symmetry that pins it to a single, rigid configuration, and the maximal-determinant records at
this order come precisely from *breaking* such symmetry under search. The Barba ceiling sits at
multiplier `369.94`; the human record at `320`; I would be surprised if a bare symmetric design
reached even a fifth of that. So I am not proposing this as a strong answer. I am proposing it as
the *right* starting point: it is principled (it realizes the almost-orthogonal Gram structure the
parity obstruction allows), it is guaranteed legal, it is parameter-free, and it gives a concrete
baseline determinant that every later, searched construction must beat. What I will watch in the
feedback is the multiplier — if it comes out around `49`, that is the symmetric-design floor, and
it tells me the entire remaining distance to the record (`49 → 320`) has to be bought by search
that deliberately leaves the symmetry behind. The structure gets me onto the board; it will not
get me near the frontier, and any next step has to start spending the determinant I cannot
construct in closed form.
