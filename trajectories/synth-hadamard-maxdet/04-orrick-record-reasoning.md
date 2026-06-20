The rank-one rung did what I asked of it and then stopped where I should have expected. Making
each candidate flip free bought two orders of magnitude more steps and a handful of structured
restarts, and the multiplier climbed into the `180`s — squarely in the band where the best
reported program-evolution results for this order also sit. But it did not move toward the record,
and I want to be precise about why, because the reason is structural, not a matter of more compute.

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

What actually produces the record is a different object entirely, and naming it is the whole point
of this rung. The maximal-determinant problem at `n ≡ 1 (mod 4)` is not solved over `±1` matrices
directly — it is solved over their *Gram matrices*. For a `29 × 29` `±1` matrix `R`, the record is
characterized by the structure of `G = R Rᵀ`: a symmetric integer matrix with `29` on the diagonal
and a constrained set of off-diagonal inner products, and the determinant of `R` is fixed by `G`
through `det(R)² = det(G)`. The dedicated search — Solomon's, tabulated by Orrick and Brent — does
not flip entries of `R` at all. It searches the far smaller, far more rigid space of *admissible
Gram matrices* `G`, the symmetric designs whose entries obey the residue constraints that the Barba
analysis permits, finds the `G` of largest determinant, and only then decomposes that `G` back into
a `±1` matrix `R` with `R Rᵀ = G`. The determinant is decided up in Gram space; the sign matrix is
recovered afterward. This is why entry-flip annealing cannot find it: it is optimizing the right
quantity in the wrong space, where the answer is not a short walk from anything it can seed from.

For `n = 29` this search has been done, and it terminated on a specific conjectured-optimal Gram
matrix. Bruce Solomon found it on `6 July 2002`; its determinant is `(2^28 · 7^12 · 320)²`, so any
`±1` factor `R` has `|det(R)| = 2^28 · 7^12 · 320` — multiplier exactly `320`. Will Orrick tabulated
it in the maximal-determinant database, and Brent's order-29 page publishes both the compressed Gram
matrix and, by randomised decomposition of that `G`, the explicit `±1` solutions `R` — `4918` known
Hadamard-equivalence classes of them. The record is not a formula I can derive in a constructor; it
is the output of that infrastructure, and the honest thing to do at the top of this ladder is to
*reproduce and verify* it rather than pretend a local search reached it.

So this rung does exactly that. I take one explicit representative `±1` matrix from Solomon's
solution set — class `1`, the most symmetric one, automorphism group size `18` — as published in
Brent's order-29 tabulation of Orrick's database, and I bring it into the harness as the constructor's
output. Then I check it honestly, with no float anywhere in the verdict. First the cheap structural
checks that confirm it is what it claims to be: every entry is exactly `±1`; `G = R Rᵀ` has `29` on
every diagonal and only the permitted off-diagonal values; and `R Rᵀ = Rᵀ R`, the normal-equation
condition that marks a genuine solution of the conjectured Gram matrix. Then the one check that
decides the score: the exact integer determinant by the same fraction-free Bareiss elimination the
evaluator uses, with no appeal to `slogdet` or any floating-point determinant. The reported number
is `|det(R)|` computed in exact integer arithmetic on a genuine sign matrix.

I want to be clear about what this rung is and is not. It is not a search that out-climbed the
previous rung — local annealing genuinely plateaus near the machine-evolution band, and I am not
going to dress that plateau up as something it isn't. It is the deliberate import of the dedicated
maximal-determinant construction that the whole problem is about, reproduced from its primary source
and verified exactly here. The ladder's earlier rungs measure how far autonomous local search gets;
this final rung measures the gap that search leaves, by standing the verified human record next to it.
The two numbers belong on the same page precisely because the distance between them — multiplier
`184.6` reached by annealing versus `320` established by Gram-space search — is the real, still-open
content of the `n = 29` problem. Above even `320` sits the Barba ceiling at multiplier `369.94`,
unmet, so the record itself is only *conjectured* optimal: the corridor from what we can search to
what is provably best is still partly dark, and this rung marks where the lit part currently ends.
