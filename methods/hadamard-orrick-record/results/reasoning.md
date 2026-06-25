The rank-one rung did what I asked of it and then stopped where I should have expected. Making
each candidate flip free bought two orders of magnitude more steps and a handful of structured
restarts, and the multiplier climbed into the `180`s — squarely in the band where the best
reported program-evolution results for this order also sit. But it did not move toward the record,
and I want to be precise about why, because the reason feels structural, not a matter of more compute.

Single-entry flip annealing, however cheaply I can afford it, is a walk on one fixed landscape: the
graph of `±1` matrices where neighbors differ in one sign. The Jacobsthal design and all its
multiplier-relabelings are valleys in that landscape, and the search escapes them and finds nearby
higher ground — but the determinant is a brutally rugged function of `841` coupled signs, and the
gains that matter come from *coordinated* changes across many entries at once. A local-move chain
can only reach a coordinated configuration by passing through a long corridor of individually
neutral-or-worse moves, and the probability of threading that corridor by undirected annealing
falls off a cliff as the corridor lengthens. So the search saturates: it finds the best matrices
reachable by short coordinated sequences from a structured seed, and that is the low-`0.5`s band. No
schedule tweak, no extra restart, moves that wall, because the wall is the geometry of single-entry
moves, not the budget. I had been quietly hoping more steps would keep paying; they do not.

So if the answer is not a short walk from anything I can seed, where does it actually live? Let me go
back to what makes `n = 29` hard in the first place and see whether the obstruction points at a better
space to work in. The Hadamard bound `n^{n/2}` is reached only by `±1` matrices with orthogonal rows,
and at `n ≡ 1 (mod 4)` orthogonal rows are impossible: two `±1` rows of length `29` have inner product
`(#agree) − (#disagree)`, and since `#agree + #disagree = 29` is odd, the difference is odd too — never
zero. (`29 % 2 = 1`, so the parity of the inner product is fixed at odd; I'll lean on this in a moment.)
The honest accounting of "how close to orthogonal can `29` rows get" is therefore not about the entries
of `R` at all, it is about the *inner products between rows* — exactly the off-diagonal entries of
`G = R Rᵀ`. The diagonal of `G` is `29` (each row dotted with itself), and `det(R)² = det(G)`. That last
identity is the lever: the quantity I care about, `det(R)²`, is a function of `G` alone, and `G` lives in
a vastly smaller, far more rigid space — symmetric, integer, diagonal pinned to `29`, off-diagonals
constrained by the parity I just computed to be odd, and bounded in magnitude by how near-orthogonal a
`±1` design can be. Entry-flip annealing has been optimizing the right scalar in the wrong space the
whole time. The record is decided up in `G`, and the sign matrix is recovered afterward.

This reframes what "doing this rung honestly" means. There is no formula for the optimal `G` at `29`
that I can drop into a constructor; this is the regime the context warned about, where the answer is the
residue of decades of number theory and computer search rather than a closed form. The dedicated
maximal-determinant search — Solomon's, tabulated by Orrick and Brent — searches admissible Gram
matrices `G` (the symmetric designs whose off-diagonals obey those constraints), finds the one of largest
determinant, and only then decomposes it into a `±1` factor `R` with `R Rᵀ = G`. For `n = 29` that search
terminated on a specific conjectured-optimal `G`: Bruce Solomon found it on `6 July 2002`, with
`det(G) = (2^28 · 7^12 · 320)²`, so any factor `R` has `|det(R)| = 2^28 · 7^12 · 320` — multiplier `320`.
The honest top-of-ladder move is to bring one published `R` into the harness and check, with exact
arithmetic, that it really is that object — not to pretend a local search reached it.

But "import and trust" is not a check; I want to actually run the verdict on the matrix before I report
a number, because a transcription error in `841` signs would be invisible to the eye and would corrupt
everything downstream. So I take class `1` (the most symmetric representative, automorphism group size
`18`) from Brent's order-29 tabulation, load it, and compute.

First the cheap structural checks, the ones that decide whether this is even a Solomon Gram solution. The
shape is `(29, 29)` and every entry is in `{−1, +1}` — good, it is a sign matrix at all. Then `G = R Rᵀ`:
the diagonal comes back as `29` in every position, as it must for unit `±1` rows. The off-diagonals are
the real test, because the parity argument above says they must all be odd, and the Barba-permissible
design at this order should use only a small set of small odd values. Computing the multiset of the `812`
off-diagonal entries gives exactly three values — `−3`, `1`, and `5` — with multiplicities `8`, `786`, and
`18`. All odd, as the parity computation demanded; all within the permitted band; nothing like a `0` or an
even value that would betray a corrupted row. And `R Rᵀ = Rᵀ R` holds exactly — the normal-equation
condition, which a genuine factor of a symmetric design satisfies but a random sign matrix would not. So
the structure is intact: this is a real factor of the conjectured-optimal `G`, not a near-miss.

Now the one check that fixes the score, and the one I trust least until I see it, because it is the whole
point. I compute the determinant of `R` the way the evaluator does — fraction-free Bareiss elimination, no
`slogdet`, no float anywhere in the verdict — and it returns

```
|det(R)| = 1188957517256767569920.
```

Against the predicted record `2^28 · 7^12 · 320`, which is `1188957517256767569920` to the digit, this is
an exact match: `|det(R)|` is divisible by `2^28 · 7^12` and the quotient is exactly `320`, so the multiplier
is `320` on the nose and the score is `320 / 342 = 0.935673`. As a cross-check on the `det(R)² = det(G)`
identity that motivated the whole Gram reframing, I also took `det(G)` directly by Bareiss; it came back as
`1413619977841376753822343762242622088806400`, which is exactly `|det(R)|²` and exactly `(2^28 · 7^12 · 320)²`.
The identity I was leaning on holds numerically on this matrix, not just in principle — which is the
reassurance I wanted that I am reading the right object out of the right space.

I want to be clear about what this rung is and is not. It is not a search that out-climbed the previous
rung — local annealing genuinely plateaus near the machine-evolution band, and I am not going to dress
that plateau up as something it isn't. It is the deliberate import of the dedicated maximal-determinant
construction that the whole problem is about, reproduced from its primary source and verified here in the
exact arithmetic that scores it. The ladder's earlier rungs measure how far autonomous local search gets;
this final rung measures the gap that search leaves, by standing the verified human record next to it. The
two numbers belong on the same page precisely because the distance between them — multiplier `184.6`
reached by annealing versus `320` established by Gram-space search — is the real, still-open content of the
`n = 29` problem. And `320` itself is not the end of the road: the Barba ceiling for `n ≡ 1 (mod 4)` is
`(2n−1)^{1/2}(n−1)^{(n−1)/2}`, which at `n = 29` is `49·√57 ≈ 369.94` in multiplier terms — I checked the
arithmetic, `369.9418…` — and that ceiling is unmet, so even the verified record is only *conjectured*
optimal. The corridor from what we can search to what is provably best is still partly dark, and this rung
marks where the lit part currently ends.
