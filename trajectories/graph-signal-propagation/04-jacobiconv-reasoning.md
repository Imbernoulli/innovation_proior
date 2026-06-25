I start from the monomial filter because its numbers show both the win and the crack I want to pry
open. Dropping the constraint and starting flat pays off across the board: Texas sits at 0.9065 (from
ChebNetII's 0.8770), Cornell at 0.8705 (from 0.8470), and the homophilic graphs do *not* regress:
Cora reaches 0.8890 and Citeseer reaches 0.8020, both above ChebNetII's 0.8725 and 0.8004. So the
uniform-init, fast-coefficient, unconstrained monomial filter dominates both constrained rungs, and
the ladder's verdict so far is blunt: on these four graphs the constraint machinery the earlier rungs
were proud of holds them back, and the plainest learnable hop-mixer, freed of its priors, is strongest.
But two things in the GPR-GNN numbers tell me it is *not* the ceiling. First, the seed-to-seed variance
is the largest of any rung — Texas std across seeds ranges 0.0262 to 0.0471 (mean 0.0361), Cornell
0.0275 to 0.0434 — and one seed even drops a Cora run entirely. That spread is the fingerprint of a
hard optimization landscape: the filter reaches a good response *on average*, but where it lands
depends on the seed, which means the loss surface in the coefficients is poorly conditioned and the
optimizer is fighting it. Second, I choose the monomial basis knowing it is ill-conditioned — the
powers `P^k` go collinear as `k` grows — and I wave that off because "at `K=10` on these graphs it
works." The variance is the bill for that hand-wave arriving.

So the question for this rung is not "what other constraint" — I have learned constraints cost more
than they pay here. It is: **can I keep the unconstrained, sign-free, flat-start freedom that made
GPR-GNN win, but fix the conditioning that made it noisy?** And to answer that I have to look harder at
something I have been treating as obvious across all three rungs. Every rung I have run differs in
exactly one thing — the polynomial *basis*: Bernstein, then Chebyshev nodes, then monomial. And every
one of those bases is *complete*: by the polynomial-filter argument they all span the same degree-`K`
function space, so they have the *same expressive power*. Any filter the monomial basis can represent,
the Chebyshev and Bernstein bases can too, and vice versa. They are linear changes of variable for one
function space. Yet they reached visibly different accuracies — Bernstein soft on citation graphs,
Chebyshev stiff on WebKB, monomial best-but-noisy. If they can all express the same filters, the
difference between them cannot be *what* they represent. It has to be *how the optimizer reaches* the
good filter inside each. That reframes the whole ladder: the basis was never an expressiveness choice,
it was an **optimization** choice, and I have been picking bases by intuition instead of by which one
conditions the descent. The monomial variance is the loudest evidence yet that I picked wrong on
conditioning.

Let me make that precise, because if I can write down what determines "how hard is this basis to
optimize," I can choose the basis that minimizes it instead of guessing. Take the decoupled filter
`Z = sum_k γ_k g_k(L̂) X` and look at the optimization over the coefficients `γ` alone, near the
optimum, under a squared loss. This is convex in `γ`, so gradient descent's convergence rate is set by
the condition number `κ(H)` of the Hessian `H`. Compute `H`: its `(k₁,k₂)` entry is
`X^T g_{k₂}(L̂) g_{k₁}(L̂) X = sum_i g_{k₁}(λ_i) g_{k₂}(λ_i) X̂²_{λ_i}`, summing over the spectrum,
where `X̂²_{λ_i}` is the signal's energy at frequency `λ_i`. In the continuum,
`H_{k₁k₂} = ∫₀² g_{k₁}(λ) g_{k₂}(λ) f(λ) dλ`, where `f(λ)` is the **spectral density of the graph
signal** — how the signal's energy is distributed over frequency. So the Hessian is exactly the Gram
matrix of the basis functions under the inner product weighted by `f`. And the whole optimization
difficulty — the thing producing GPR-GNN's seed variance — is `κ(H)`. There it is, explicit: a basis
is easy to optimize precisely when its Gram matrix under the signal density is well-conditioned.

Now the punchline that selects the basis. `κ(H)` is smallest, equal to 1, exactly when `H = I` — when
the basis is **orthonormal under the density-weighted inner product** `⟨g,h⟩ = ∫ g h f dλ`. So the
optimal basis is not fixed; it is whichever orthogonal family has weight function equal to the graph's
signal density `f`. This explains every result on the ladder at once. The monomial basis I just used
is *provably not orthogonal under any weight function* — its Hessian is dense and ill-conditioned, the
worst case — which is precisely why GPR-GNN optimizes well *on average* (it is expressive) but with the
*highest variance* (the descent is poorly conditioned, so the seed decides where it lands). The
Bernstein basis is also non-orthogonal. The Chebyshev basis ChebNetII used *is* orthogonal — but only
under *one fixed* weight, `(1-x²)^{-1/2}`. If the graph's `f` happens to look like that weight,
Chebyshev is near-optimal; but `f` is *not* the same across these four graphs — homophilic Cora and
Citeseer have their signal energy at *low* frequency, heterophilic Texas and Cornell at *high*
frequency — so one fixed weight cannot match all four, and Chebyshev's mismatch on the WebKB graphs is
the stiffness I diagnosed two rungs ago. No fixed-weight basis can be optimal across graphs whose
densities differ. I need a basis whose weight I can *tune to match `f`* — and that is a basis the
ladder has not tried.

That requirement names it. Among the classical orthogonal polynomials, the **Jacobi** family
`P_k^{a,b}` is orthogonal under the weight `(1-x)^a (1+x)^b` on `[-1,1]`, with two free parameters
`a, b` that reshape the weight continuously. Chebyshev is the *single special case* `a=b=-1/2`,
Legendre is `a=b=0` — so Jacobi *contains* the basis ChebNetII used and adds two knobs to slide the
weight toward whichever end of the spectrum the graph's energy occupies. On the normalized adjacency
`Â = I − L`, whose spectrum is `[-1,1]`, low graph frequency `λ≈0` maps to `x≈1` and high `λ≈2` to
`x≈-1`. So for a homophilic graph (low-frequency energy) I push the Jacobi weight toward `x=1` by
making the exponent on `(1-x)` smaller relative to the exponent on `(1+x)`; for a heterophilic graph
(high-frequency) I push it toward `x=-1` by making the exponent on `(1+x)` smaller. Tuning `a, b`
*is* matching the weight to `f`, which is the `κ(H)→1` knob made an explicit hyperparameter. This is the move that
keeps everything GPR-GNN got right — unconstrained, sign-free, learned coefficients, flat uniform start
— while fixing the one thing it got wrong: it replaces the non-orthogonal monomial basis, whose bad
conditioning produced the seed variance, with the *adaptable* orthogonal basis that drives the Hessian
toward identity on each graph. I do not have to *compute* the optimal `a, b` (that needs the
eigendecomposition I cannot afford); I pick the flexible family and a sensible default, and the weight
sits close enough to `f` to condition the descent.

There is a deeper licence here too, one I should state because it changes what I am even building.
Across all three prior rungs I kept the nonlinear ReLU MLP encoder without questioning it. But a
spectral GNN is a filter, and I can ask whether a *linear* spectral GNN — drop the activation — is
already enough. The answer is yes under two conditions: the normalized Laplacian must have no repeated
eigenvalues, and the features must miss no frequency component. Repeated eigenvalues come from graph
symmetry; these benchmark graphs are irregular and fewer than 1% of eigenvalues are multiple, and no
frequency component is missing. ReLU's only spectral effect is to *mix* frequency components — the
repair for exactly those two obstructions — so on these graphs it fixes a disease the data does not
have. The principled JacobiConv is therefore a *linear* model, which is also what made the Hessian
argument above clean (the coefficient fit is convex). The harness I am editing keeps the ReLU between
its two linear layers, so in this task I land the Jacobi *propagation* into the existing decoupled
encoder rather than stripping it to a single linear layer — the basis change is where the leverage is,
and it carries over whether or not the encoder is linear.

Now I have to be honest about exactly what this task's edit surface lets me build, because the
principled JacobiConv has two refinements the harness cannot express, and I must land the version the
scaffold supports, not the full one. First, *individual filters per output channel*: the universality
argument is one-dimensional, and a single shared filter cannot produce an arbitrary multi-class
prediction, so the full method gives each output channel its own coefficient vector `α_{kl}`. But this
task's `CustomProp` owns a *single* shared coefficient vector `temp` of length `K+1`, applied
identically to every channel — there is no per-channel filter slot. Second, *polynomial coefficient
decomposition (PCD)*: the full method decomposes `α_{kl} = β_{kl} Π_i γ_i` with shared bounded ratios
to rescale the per-`k` magnitudes, which is what realizes the orthogonal basis's good conditioning in
practice. The harness exposes only the raw `temp[k]`, so there is no PCD here either. So what I am
actually landing is the **shared-coefficient, fixed-`(a,b)` Jacobi filter** — the orthogonal-basis core
of the method without the per-channel filters and without PCD. That is the honest delta over GPR-GNN:
*same* unconstrained learned coefficients, *same* uniform `1/(K+1)` start, *same* decoupled encoder —
but the monomial powers `P^k` are replaced by the Jacobi polynomials `P_k^{a,b}(Â)`, with `a=b=1` as
the symmetric default (which makes the recurrence's `θ'_k = (a²−b²)·(…)` term vanish, a basis neutral
between the two spectral ends).

The propagation itself stays linear in `K`, which I want on the record since cheapness has been a
running theme. I build `L` with the symmetric-normalized `get_laplacian`, shift to `L̃ = L − I = −Â`
by adding self-loops of weight `−1`, and run the Jacobi three-term recurrence: `P_0 = x`,
`P_1 = ((a+b+2)/2)·(L̃ x) + ((a-b)/2)·x`, and for `k ≥ 2`,
`P_k = (A_k + B_k L̃) P_{k-1} − C_k P_{k-2}` with the Jacobi coefficients `A_k, B_k, C_k` computed from
`a, b, k`. Each step is one `propagate` (one sparse mat-vec), so the whole filter is `K` propagations —
`O(K m d)`, the same order as GPR-GNN and ChebNetII, strictly cheaper than BernNet. One subtlety I have
to get right: running the recurrence on `L̃ = −Â` rather than `Â` evaluates `P_k(−x) = (−1)^k
P_k^{a,b}(x)` (the Jacobi reflection identity, symmetric when `a=b`), an alternating sign that the
*learned* `temp[k]` absorb completely — so the filter family is identical to the `Â` convention and I
lose nothing by using the operator the harness builds for free. The literal edit replaces the two class
bodies (`CustomProp` becomes the Jacobi layer with its `a, b` and recurrence, `CustomFilter` keeps the
standard encoder/readout and passes `a=b=1` into the propagation); the full scaffold module is in the
answer.

So the bar this rung must clear, against GPR-GNN's real numbers, and what I would validate. The whole
motivation was GPR-GNN's *conditioning*, not its means, so the first thing I expect is not a dramatic
mean jump but a *quieter* one: the Jacobi basis should match or modestly exceed GPR-GNN's accuracies —
Cora at or above 0.8890, Citeseer at or above 0.8020, Texas at or above 0.9065, Cornell at or above
0.8705 — while *shrinking the seed-to-seed variance* that betrayed the monomial basis's ill-conditioned
Hessian (GPR-GNN's Texas std spread of 0.0262–0.0471 and the dropped Cora run are exactly what a
well-conditioned orthogonal basis should tighten). The cleanest falsification would be the opposite
profile: if the shared-coefficient Jacobi filter — shorn of the per-channel filters and PCD the full
method relies on — lands *below* GPR-GNN on any graph, that would say the conditioning win does not
survive without PCD on this harness, and the monomial filter remains the right answer here. The honest
risk I name: with `a=b` fixed (no per-dataset tuning of the weight toward the graph's density) and no
PCD, the Jacobi advantage on this harness may be small — the orthogonality is only *exactly* matched to
`f` when `a, b` track each graph's density, and a single symmetric default leaves some conditioning on
the table. But even the untuned symmetric Jacobi basis is orthogonal under *a* weight, where the
monomial basis is orthogonal under *none*, so I expect the conditioning to improve and the variance to
fall — and that, more than a fractional accuracy gain, is the result that would justify the orthogonal
basis as the right ceiling for this ladder.
