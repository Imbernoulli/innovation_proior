I want the `29 × 29` sign matrix whose determinant is as large as possible, and the first thing to
fix is what "as large as possible" can even mean, because it hinges on `29 mod 4`. If the rows of a
`±1` matrix were mutually orthogonal, `HHᵀ` would be `29·I`, the determinant `29^{29/2} =
29^{14}·√29 ≈ 1.60 × 10^{21}`, the Hadamard ceiling. But that ceiling is provably off the table: the
rows have odd length `29`, and the inner product of two `±1` vectors of odd length is a sum of an odd
number of `±1`'s, hence odd, never zero. No two rows can be orthogonal; `HHᵀ` can never equal
`29·I`. So the best `±1` matrices here are *almost* orthogonal, and the whole question is how close.

What should "almost orthogonal" look like as a Gram matrix? If the off-diagonal inner products
cannot be zero, I want them as small and uniform as possible; being odd, the smallest is `±1`. The
cleanest target is `HHᵀ = 29·I − (J − I) = 30·I − J = (n+1)I − J`, every pair of rows overlapping by
exactly `−1`. A zero-diagonal `±1` matrix `C` with `CCᵀ = (m−1)I` is a *conference matrix* of order
`m`, existing symmetrically when `m ≡ 2 (mod 4)`; the uniform pattern I want lives at order `m = 30`,
and since `29` is a prime `≡ 1 (mod 4)` the Paley construction hands me a symmetric conference matrix
of order `30` for free. But my constructor must output `29 × 29`, so I use its quadratic-residue core
and turn that into a legal sign matrix.

One structural fact fixes the units before I start. The determinant of *any* `29 × 29` `±1` matrix is
divisible by `2^{28}`: subtract row `0` from each other row (determinant unchanged), and every entry
of those `28` rows is now a difference of two `±1`'s, so in `{−2, 0, 2}`; factor a `2` out of each.
That is the `2^{28}` in the normalization, and it is why the interesting content lives in the
*multiplier* `|det| / (2^{28}·7^{12})`. The extra `7^{12}` is not automatic — it appears only for
specially structured designs — and whether this design carries it cleanly is part of what I want to
learn.

The engine is the Legendre symbol. For `q = 29` define `χ(a) = +1` on nonzero quadratic residues,
`−1` on non-residues, `0` at `0`, and build `Q_{ij} = χ(i − j)`, the Jacobsthal matrix. Because
`q ≡ 1 (mod 4)`, `−1` is itself a residue, so `χ(−a) = χ(a)` and `Q` is *symmetric*. And `QQᵀ = qI −
J` exactly: the diagonal is a row's `28` nonzero squared entries; the off-diagonal at `t = i − j ≠ 0`
is `Σ_k χ(k)χ(k + t) = Σ_{k≠0} χ(1 + t/k) = Σ_{a ≠ 1} χ(a) = −χ(1) = −1`. Exactly `−1` at every
off-diagonal, the tightest uniform overlap parity allows — `Q` already carries the almost-orthogonal
structure. But `Q` has `29` zeros on its diagonal; filling them with `±1` is where the degradation
enters.

Fill with `+1`: `R = Q + I`, legal. What did the fill cost the Gram matrix? `RRᵀ = QQᵀ + 2Q + I =
30I − J + 2Q`. The diagonal is `29`, the row norm I need. But the off-diagonal is `−1 + 2Q_{ij}`:
`+1` where `Q_{ij} = +1`, `−3` where `Q_{ij} = −1`, `14` of each per row. Read as row overlaps
`29 − 2d`, a `+1` means `d = 14` disagreements, a `−3` means `d = 16`. Every pair of rows disagrees
in `14` or `16` of `29` coordinates, a rigid two-valued pattern locked by the residue structure. That
`2Q` cross term is the price of squeezing order-`30` conference structure into a `29 × 29` sign
matrix.

Now `det(R)` in closed form. `R = Q + I` is a polynomial in the symmetric `Q`, so its eigenvalues are
`1 + λ`. On the all-ones vector `Q𝟙 = 0`. On the `28`-dimensional complement `J` acts as `0`, so
`Q² = qI − J` collapses to `Q² = 29I`, giving `λ = ±√29`, and `tr(Q) = 0` splits them `14`/`14`. So
`Q` has eigenvalues `0`, `+√29` (`×14`), `−√29` (`×14`), and

`det(R) = (1 + √29)^{14}(1 − √29)^{14} = (1 − 29)^{14} = (−28)^{14} = 28^{14}.`

Since `28 = 2²·7`, `28^{14} = 2^{28}·7^{14}`, multiplier `7^{14}/7^{12} = 49`. So this design has
multiplier *exactly* `49`, `|det| = 2^{28}·7^{14}`, and the `7`-power the normalization anticipates is
carried cleanly as `(4·7)^{14}` — the Bareiss determinant should return that integer to the digit.

The diagonal fill was a choice, but the twin settles it: `R' = Q − I` has eigenvalues `−1 + λ`, so
`det(R') = (−1)(−28)^{14} = −28^{14}`, identical magnitude. The diagonal sign is a determinant-twin,
not a lever — the construction has no free parameter to tune, exactly what a rigid symmetric design
should have. Any improvement has to come from outside the quadratic-residue family: a mixed `±1`
diagonal breaks the symmetry that made `Q² = 29I` collapse and generically lowers `|det|`, and a
circulant relabeling lands on the same spectrum.

I should be honest that "make the overlaps as small as possible" was a proxy, not the quantity I am
paid on. For a diagonal-`29` Gram matrix with uniform off-diagonal `c`, `det = (29 + 28c)(29 −
c)^{28}`. At `c = −1` that is `30^{28}`, the near-orthogonal ideal, multiplier `~129`; but at `c = +1`
it is `57·28^{28}`, giving `|det| = √57·28^{14}`, multiplier `49·√57 ≈ 369.94` — exactly the Barba
ceiling for `n ≡ 1 (mod 4)`. So the determinant does *not* prefer the smallest overlaps: the uniform
`+1`-overlap Gram (every pair disagreeing in exactly `14`) beats the uniform `−1` Gram by a factor of
`57` and is the true provable ceiling. My `R` sits at diagonal `29` but pays the split — `14`
Barba-like `+1` overlaps and `14` costly `−3` per row — and pushing toward the ceiling means
selectively turning `−3` overlaps into `+1`, coordinated sign changes the residue structure will
never make on its own.

So `R = Q + I` is the natural symmetric `±1` matrix attached to `29`, fully determined, no search,
multiplier provably `49` — less than a fifth of the record `320`. The very symmetry that lets me write
its determinant as `28^{14}` is what pins it to one rigid configuration; the records at this order
come from *breaking* such symmetry under search. I ship it as the right floor: principled, legal,
parameter-free, with an exactly-predicted baseline every later rung must beat. The two-valued overlap
rigidity makes me suspect this is a *strict* local maximum under single flips — each row's overlaps are
locked at a balanced `14`/`14` extreme, and nudging one entry perturbs a whole column of overlaps at
once — so I expect greedy hill-climbing to die on its first step, and the only way off this floor to be
a search willing to accept downhill moves. That is the next rung's job.
