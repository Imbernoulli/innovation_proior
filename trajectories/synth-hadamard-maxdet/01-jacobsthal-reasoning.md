I want the `29 × 29` sign matrix whose determinant is as large as possible. The first thing to
fix is what "as large as possible" can even mean here, because the answer depends entirely on `29
mod 4`. If the rows of a `±1` matrix were mutually orthogonal, the Gram matrix `HHᵀ` would be
`29·I`, the determinant would be `29^{29/2}`, the Hadamard ceiling, and I would be done. Put a
number on that ceiling so I know what I am chasing: `29^{14.5} = 29^{14}·√29 ≈ 1.60 × 10^{21}`. It
is a big, clean target — and it is unreachable, which I can see directly rather than take on
faith. The rows have length `29`, which is odd. The inner product of two `±1` vectors of odd
length is a sum of an odd number of `±1`'s, and a sum of an odd count of odd numbers is odd — an
odd integer, never zero. So no two rows can be orthogonal; `HHᵀ` can never equal `29·I`; the
Hadamard bound is not merely hard, it is provably off the table at this order. That is not a
detail, it is the whole character of the problem: I am not hunting for a perfect orthogonal
design that exists and is hard to find, I am pushing against a ceiling that provably cannot be
touched, where the best `±1` matrices are *almost* orthogonal and the entire question is how
close.

So let me ask, concretely, what "almost orthogonal" should look like as a Gram matrix. If I cannot
make the off-diagonal inner products zero, I want them as small and as uniform as possible. The
off-diagonal entries of `HHᵀ` are odd, so the smallest they can be in magnitude is `±1`. The
cleanest target I can imagine is a Gram matrix where every diagonal entry is the row norm-squared
`29` and every off-diagonal entry is exactly `−1`: `HHᵀ = 29·I + (−1)(J − I) = 30·I − J`, i.e.
`HHᵀ = (n+1)I − J` with `n = 29`. It is worth pricing this ideal before I chase it, because the
price tells me whether the whole almost-orthogonal program is even worth much. The matrix
`30·I − J` has a spectrum I can read off: `J` is rank one with eigenvalue `29` on the all-ones
vector and `0` on its `28`-dimensional complement, so `30·I − J` has eigenvalue `30 − 29 = 1`
once and `30` twenty-eight times. Its determinant is `1 · 30^{28}`, so a matrix realizing it would
have `|det(H)| = √(30^{28}) = 30^{14}`. In multiplier units that is `30^{14} / (2^{28}·7^{12}) ≈
128.7`. So the perfectly-uniform `−1`-overlap ideal, *if a real sign matrix could hit it*, would
already be worth multiplier `~129` — a large fraction of the record `320` and far above where a
naive fill will land. That gap between the ideal Gram pattern and what I can actually construct is
the entire drama of this problem, and it is what the later rungs will have to buy back.

Can a `29 × 29` `±1` matrix realize `30I − J` exactly? A matrix `C` with zero diagonal and `±1`
off-diagonal satisfying `CCᵀ = (m−1)I` is a *conference matrix* of order `m`, and symmetric
conference matrices of order `m` exist when `m ≡ 2 (mod 4)` and the number theory cooperates. The
uniform pattern I want lives most naturally at order `m = 30`: `30 ≡ 2 (mod 4)`, and since `29` is
a prime `≡ 1 (mod 4)`, the Paley construction hands me a symmetric conference matrix of order `30`
for free. But my constructor must output a `29 × 29` matrix, not a `30 × 30` one, so I cannot
simply hand back the order-`30` conference matrix. What I can do is use its `29 × 29` engine — the
quadratic-residue core that sits inside the Paley design — and turn that into a legal sign matrix
of the right size, accepting whatever degradation the resizing costs. So the realizable object
will not hit the clean `30I − J`; the interesting question is how far short it falls.

One more structural fact belongs on the table before I build, because it explains the units the
record is quoted in and tells me what any construction is guaranteed to carry. The determinant of
*any* `29 × 29` `±1` matrix is divisible by `2^{28}`. The reason is a two-line row reduction:
subtract row `0` from each of the other `28` rows. This does not change the determinant, and every
entry of the modified rows is now a difference of two `±1`'s, hence in `{−2, 0, 2}` — divisible by
`2`. Factor a `2` out of each of the `28` altered rows and the determinant picks up `2^{28}` times
the determinant of an integer matrix. So `2^{28}` divides `|det(H)|` unconditionally; that is the
structural `2^{28}` in the score normalization, and it is why the interesting content of a
determinant lives in the *multiplier* `|det| / (2^{28}·7^{12})` rather than in the raw
twenty-one-digit integer. The extra `7^{12}` is not automatic — it appears only for the specially
structured designs — and part of what I want to learn from this first rung is whether the symmetric
Jacobsthal design carries that `7`-power cleanly.

The engine is the Legendre symbol. For the prime `q = 29`, define `χ(a)` to be `+1` if `a` is a
nonzero quadratic residue mod `q`, `−1` if it is a non-residue, and `0` if `a ≡ 0`. Build the
`q × q` matrix `Q` with `Q_{ij} = χ(i − j)` — the Jacobsthal matrix. Two facts about it I can
lean on, and I want to actually check them, not assert them. First, because `q ≡ 1 (mod 4)`, `−1`
is itself a quadratic residue (it is `28 ≡ −1`, and `−1` is a square mod a prime iff the prime is
`≡ 1 (mod 4)`), so `χ(−a) = χ(a)`, which makes `Q` *symmetric*. Second, the multiplicative
structure of the residues gives the identity `QQᵀ = qI − J` exactly. I can see the two pieces of
that: the diagonal of `QQᵀ` is the squared norm of a row of `Q`, and a row has `q − 1 = 28` nonzero
`±1` entries (the single `0` sits on the diagonal), so the diagonal is `28 = q − 1`; and every
off-diagonal entry is `−1`, and I want to pin that down rather than wave at it, because the whole
almost-orthogonal claim rests on it. The `(i, j)` off-diagonal of `QQᵀ` is `Σ_k χ(i−k)χ(j−k)`. Set
`t = i − j ≠ 0` and substitute so the sum reads `Σ_k χ(k)χ(k + t)`. The `k = 0` term is `0`, and
for `k ≠ 0` write `χ(k)χ(k+t) = χ(k)²·χ(1 + t/k) = χ(1 + t/k)` since `χ(k)² = 1`. As `k` ranges
over the `28` nonzero residues, `1 + t/k` ranges over every residue except `1`, so the sum is
`Σ_{a ≠ 1} χ(a) = (Σ_a χ(a)) − χ(1) = 0 − 1 = −1`. Exactly `−1`, at every off-diagonal — the
tightest uniform overlap parity allows. So `Q`
already carries the almost-orthogonal Gram structure I was reaching for — the `29 × 29` shadow of
that order-`30` conference matrix. But `Q` is not a `±1` matrix: it has `29` zeros on the diagonal.
I have to fill those zeros with `±1` to get a legal sign matrix, and that fill is exactly where the
degradation from the clean ideal will enter.

Fill them with `+1`. Set `R = Q + I`. Now every entry is `±1` (the diagonal is `0 + 1 = 1`, the
off-diagonal is the `±1` of `Q` unchanged), so `R` is a legal output. What did that fill cost the
Gram matrix? Compute it: `RRᵀ = (Q + I)(Q + I)ᵀ = QQᵀ + Q + Qᵀ + I = (qI − J) + 2Q + I`, using
`Q = Qᵀ`. So `RRᵀ = (q+1)I − J + 2Q = 30I − J + 2Q`. The diagonal is `30 − 1 + 2·0 = 29`, exactly
the row norm-squared I need, good. But the off-diagonal is no longer the uniform `−1` of the ideal:
it is `−1 + 2·Q_{ij}`, which is `+1` where `Q_{ij} = +1` and `−3` where `Q_{ij} = −1`. So the Gram
matrix of my realizable `R` has off-diagonals in `{−3, +1}` rather than the clean uniform `−1`.
That `2Q` cross term is precisely the price of squeezing the order-`30` conference structure into a
`29 × 29` sign matrix — and I can even audit its size: each row of `Q` has `14` residues and `14`
non-residues among its `28` off-diagonal positions, so each row of `G = RRᵀ` has `14` off-diagonal
entries equal to `+1` and `14` equal to `−3`. It is worth reading those Gram values back as row
overlaps, because that is the geometry the determinant actually sees. Two rows of `R`, each a `±1`
vector of length `29`, have inner product `29 − 2d` where `d` is the number of positions they
disagree in. An off-diagonal `+1` means `29 − 2d = 1`, i.e. `d = 14`; an off-diagonal `−3` means
`29 − 2d = −3`, i.e. `d = 16`. So every pair of rows of `R` disagrees in either `14` or `16` of
the `29` coordinates — never `15`, never near-orthogonal parity `14.5` (impossible), but a rigid
two-valued pattern locked by the residue structure. That two-valued rigidity is the visual of
"symmetric design": the rows are as balanced as the residues force them to be and not one flip
freer. The row sum of `G` is then `29 + 14·(1) + 14·(−3) =
29 + 14 − 42 = 1`, which is a clean consistency check: `G·𝟙 = (30I − J + 2Q)𝟙 = 30·𝟙 − 29·𝟙 +
2·0 = 𝟙`, since every row of `Q` sums to `Σ_a χ(a) = 0`. The all-ones vector is an eigenvector of
`G` with eigenvalue `1`. That single fact is going to pin down the whole determinant.

Because now I can get `det(R)` in closed form rather than waiting to measure it. `R = Q + I` is a
polynomial in the symmetric matrix `Q`, so its eigenvalues are `1 + λ` over the eigenvalues `λ` of
`Q`, and I know the spectrum of `Q` exactly. On the all-ones vector, `Q𝟙 = 0`, so `λ = 0` there.
On the `28`-dimensional complement, `J` acts as `0`, so `Q² = QQᵀ = qI − J` collapses to `Q² =
29·I`, meaning every eigenvalue there satisfies `λ² = 29`, i.e. `λ = ±√29`. How do the `±√29`
split? The trace pins it: `tr(Q) = 0` because the diagonal is all zeros, and the `λ = 0`
eigenvalue contributes nothing, so `a·(√29) + b·(−√29) = 0` with `a + b = 28`, forcing `a = b =
14`. So `Q` has eigenvalues `0` (once), `+√29` (`14` times), `−√29` (`14` times). Push these
through `R = Q + I`:

`det(R) = ∏(1 + λ) = (1 + 0) · (1 + √29)^{14} · (1 − √29)^{14} = 1 · [(1 + √29)(1 − √29)]^{14} =
(1 − 29)^{14} = (−28)^{14} = 28^{14}.`

So `|det(R)| = 28^{14}`. Factor it: `28 = 2²·7`, so `28^{14} = 2^{28}·7^{14}`, and the multiplier
is `2^{28}·7^{14} / (2^{28}·7^{12}) = 7² = 49`. That is not a guess I will confirm against the
evaluator — it is a prediction I am willing to be judged on: the symmetric Jacobsthal design at
`n = 29` has multiplier *exactly* `49`, `|det| = 2^{28}·7^{14}`. The exact Bareiss determinant in
the harness should return that integer to the digit, and if it returns anything else my spectral
accounting is wrong somewhere. The gap to my aspirational `~129` is now explicit and mechanical:
the `2Q` cross term that filling the diagonal forced on the Gram matrix is exactly what drags the
would-be `128.7` down to `49`. I am not near the ideal, let alone the record — but I know precisely
why, and the reason is a specific, named term I can point to.

The same argument gives a general closed form worth stating, because it lets me sanity-check the
`29` computation against a case small enough to trust by hand. For any prime `q ≡ 1 (mod 4)` the
Jacobsthal spectrum is `0` and `±√q` (each `(q−1)/2` times), so `det(Q + I) = (1)·[(1+√q)(1−√q)]^{(q−1)/2}
= (1 − q)^{(q−1)/2}`. At `q = 5` this is `(1 − 5)^{2} = (−4)^2 = 16`, and I can check that directly:
the `5 × 5` Jacobsthal `R` should have `|det| = 16`, which is `2^4 = 2^{q−1}`, the bare structural
factor with no extra prime — the `5 × 5` design is too small to carry a `7`-power. At `q = 13`,
`(1 − 13)^{6} = 12^6 = 2^{12}·3^6`. The formula `(1 − q)^{(q−1)/2}` is clean and monotone in the
right way, and plugging `q = 29` returns `(−28)^{14} = 28^{14} = 2^{28}·7^{14}` — so the `7^{14}`
is not an accident of `29`, it is `28 = 4·7` raised to the `14`-th power, and the multiplier
`7^{14}/7^{12} = 49` is exactly the "extra `7`-power" the score normalization anticipates, carried
cleanly by this particular design. The `q = 5` and `q = 13` checks give me confidence the `q = 29`
value is `49` and not a mis-simplification.

There is a choice I glossed: I filled the diagonal with `+1`, but `−1` was equally available,
giving `R' = Q − I`. The same spectral computation settles whether that matters. `R' = Q − I` has
eigenvalues `−1 + λ`, so `det(R') = (−1 + 0)(−1 + √29)^{14}(−1 − √29)^{14} = (−1)·[(√29 − 1)(−√29 −
1)]^{14}`, and `(√29 − 1)(−√29 − 1) = −(√29 − 1)(√29 + 1) = −(29 − 1) = −28`, so `det(R') = (−1)·
(−28)^{14} = −28^{14}`. The magnitude is identical: `|det(Q + I)| = |det(Q − I)| = 28^{14}`. So the
diagonal sign is a determinant-twin, not a lever — both fills give the same `49`. Good: that means
the construction has no free parameter to tune, which is exactly what I expect from a rigid
symmetric design. Any hope of improvement has to come from *outside* the family of quadratic-residue
fills.

Before I commit, let me make sure a cheaper or larger construction isn't sitting right next to
this one. Three alternatives are live. One, fill the diagonal with a *mixed* pattern of `±1`
instead of all `+1` or all `−1` — but any diagonal fill `R = Q + D` with `D` diagonal `±1` still
gives `RRᵀ = qI − J + 2Q·(sign coupling) + I` with off-diagonals of the same `{−3, 1}` flavor, and
worse, a mixed `D` breaks the symmetry that made the spectrum computable and generically *lowers*
`|det|` because it destroys the clean `Q² = 29I` collapse; there is no reason to expect a mixed
diagonal to beat the twin value `49`, and I would be trading a provable `49` for an unprincipled
gamble. Two, use a *circulant* or otherwise more structured `±1` matrix built directly from the
residues without the `Q + I` shift — but that is just a relabeling of the same quadratic-residue
object and lands on the same spectrum; relabeling a symmetric design does not change `|det|`.
Three, abandon structure and search over sign matrices directly — but that is a different kind of
move entirely, and the whole point of this rung is to establish the principled floor *before*
spending any search budget, so that I know what the search has to beat. None of the three improves
on `49` cheaply, and the first two are provably twins or degradations. So the forced construction
is the right thing to ship first.

I should be honest that "make the overlaps as small as possible" — the heuristic that pointed me at
the uniform `−1` pattern and hence at `Q` — is a proxy, and not exactly the quantity I am paid on.
The determinant of a diagonal-`29` Gram matrix with uniform off-diagonal `c`, `G = (29−c)I + cJ`,
is `(29 + 28c)(29 − c)^{28}`: the all-ones direction gives eigenvalue `29 + 28c`, the `28`-dimensional
complement gives `29 − c`. For `c = −1` this is `1 · 30^{28}`, my near-orthogonal ideal, multiplier
`~129`. But for `c = +1` it is `57 · 28^{28}`, giving `|det| = √57 · 28^{14}`, multiplier `49·√57 ≈
369.94` — which is exactly the Barba ceiling quoted for `n ≡ 1 (mod 4)`. So the determinant does
*not* prefer the smallest overlaps; the uniform `+1`-overlap Gram `28I + J` (every pair of rows
disagreeing in exactly `14` coordinates) beats the uniform `−1` Gram by a factor of `57` under the
determinant, and it is the true provable ceiling. My Jacobsthal `R` sits at the same diagonal `29`
but pays a split: `14` off-diagonals per row are the good Barba-like `+1` (`d = 14`), and `14` are
the costly `−3` (`d = 16`). Those `14` over-disagreeing pairs per row are, mechanically, the entire
deficit from `49` to `369.94`. That is a clarifying way to see why a rigid symmetric design cannot
be strong: the residues force half of each row's overlaps to the wrong value, and no closed-form
fill of a quadratic-residue matrix can unstick them. Pushing toward the ceiling means selectively
turning `−3` overlaps into `+1` — coordinated sign changes the residue structure will never make
on its own.

Now the honest appraisal of where this lands, because the point of starting here is to know the
floor. This `R = Q + I` is a textbook object: the natural symmetric `±1` matrix attached to the
prime `29`, fully determined, no search, multiplier provably `49`. The symmetry that makes it
elegant — that lets me write its determinant in closed form as `28^{14}` — is the very same
symmetry that pins it to a single rigid configuration, and the maximal-determinant records at this
order come precisely from *breaking* such symmetry under search. The Barba ceiling sits at
multiplier `49·√57 ≈ 369.94`; the human record at `320`; my aspirational uniform-overlap Gram
pattern would have been worth `~129`; and the rigid symmetric design I can actually construct
delivers `49`, less than a fifth of the record. So I am not proposing this as a strong answer. I am
proposing it as the *right* starting rung: it is principled (it realizes the largest-overlap Gram
structure the parity obstruction and the `29 × 29` size allow), it is guaranteed legal, it is
parameter-free, and it gives a concrete, exactly-predicted baseline determinant that every later,
searched rung must beat. What I will watch in the feedback is a single number: whether the
multiplier is exactly `49`, which on the frozen score `m/342` reads as `49/342 ≈ 0.1433` — the
only metric this task reports. If it is, that confirms both the closed form and the diagnosis — that
the entire remaining distance to the record (`49 → 320`) has to be bought by search that
deliberately leaves the symmetry behind, spending the determinant that the `2Q` cross term costs me
and that I cannot recover in any closed-form symmetric fill. The two-valued overlap rigidity makes
me suspect something sharper that I will want the feedback to test: that this point may be a
*strict local maximum* under the smallest move I have, a single sign flip. Each row's overlaps are
locked at exactly `14` entries of `+1` and `14` of `−3`, a balanced extreme, and nudging one entry
perturbs a whole column of those overlaps at once; from such a balanced configuration I would
expect every single-flip direction to be flat-or-down rather than up. If that suspicion holds — if
not one of the `29 × 29 = 841` single-entry flips raises `|det|` — then greedy hill-climbing dies
on its first step here, and the only way off this floor is a search willing to accept downhill
moves. The structure gets me onto the board; it will not get me near the frontier, and the next
rung has to start paying for the determinant I cannot construct.
