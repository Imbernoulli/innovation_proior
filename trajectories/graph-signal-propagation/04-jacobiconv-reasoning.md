I start from the monomial filter because its numbers show both the win and the crack I want to pry open,
and the cleanest way to see both is to lay the three rungs side by side and do the arithmetic. Per graph,
the progression reads Cora 0.8554 → 0.8725 → 0.8890, Citeseer 0.7795 → 0.8004 → 0.8020, Texas 0.9093 →
0.8770 → 0.9065, Cornell 0.8443 → 0.8470 → 0.8705. Averaging the four datasets at each rung gives 0.8471
for BernNet, 0.8492 for ChebNetII, 0.8670 for the monomial filter. So the two increments are wildly
uneven: BernNet → ChebNetII bought **+0.0021** on average — a wash, the near-minimax interpolation raising
the citation graphs by about the same amount it dropped Texas — while ChebNetII → monomial bought
**+0.0178**, nearly an order of magnitude larger, and it did so by lifting the whole board at once: Texas
back up to 0.9065, Cornell up to 0.8705, *and* Cora and Citeseer above ChebNetII rather than below. The
ladder's verdict so far is therefore blunt, and it is not about which clever basis: the real progress came
from *dropping the constraint*, not from refining it. Freeing the coefficients — sign, magnitude, uniform
start — was the move that mattered, and the constrained rungs were holding the whole board down. So
whatever I do next, I am not going back to a constraint.

But two things in the monomial numbers tell me it is *not* the ceiling, and both live in the variance
tables rather than the means. The seed-to-seed spread is the largest of any rung: on Texas the three seeds
read 0.9098/0.9049/0.9049 (a spread of 0.0049 where ChebNetII's three Texas seeds were identical to four
digits), on Cornell 0.8738/0.8738/0.8639 (spread 0.0099), and one seed dropped a Cora run entirely. The
across-split std tells the same story from the other direction — the monomial filter's Texas std averages
0.0361 (ranging 0.0262 to 0.0471 across seeds) and Cornell 0.0337, the widest heterophilic cells of any
learnable filter, wider than ChebNetII's stiff 0.0246 on Texas. So freedom bought the mean and cost the
stability: the filter reaches a good response *on average*, but *where* it lands depends on the seed. That
is the fingerprint of a hard optimization landscape — a poorly-conditioned loss surface in the
coefficients that the optimizer is fighting, and the seed decides which basin it settles in. And I chose
the monomial basis knowing it is ill-conditioned — I measured its consecutive high powers sitting 2.9°
apart on the spectrum — and waved that off with "at `K=10` on these graphs it works." The variance is the
bill for that hand-wave arriving. So the question for this rung is not "what other constraint" — I have
learned constraints cost more than they pay here. It is: **can I keep the unconstrained, sign-free,
flat-start freedom that made the monomial filter win, but fix the conditioning that made it noisy?**

To answer that I have to look harder at something I have been treating as obvious across all three rungs.
Every rung I have run differs in exactly one thing — the polynomial *basis*: Bernstein, then Chebyshev
nodes, then monomial. And every one of those bases is *complete*: by the polynomial-filter argument they
all span the same degree-`K` function space, so they have the *same expressive power*. Any filter the
monomial basis can represent, the Chebyshev and Bernstein bases can too, and vice versa — they are linear
changes of variable for one function space. Yet they reached visibly different accuracies — Bernstein soft
on citation graphs, Chebyshev stiff on WebKB, monomial best-but-noisy. If they can all express the same
filters, the difference between them *cannot be what they represent*. It has to be *how the optimizer
reaches* the good filter inside each. That reframes the whole ladder: the basis was never an expressiveness
choice, it was an **optimization** choice, and I have been picking bases by intuition instead of by which
one conditions the descent. The monomial variance is the loudest evidence yet that I picked wrong on
conditioning.

Let me make that precise, because if I can write down what determines "how hard is this basis to
optimize," I can choose the basis that minimizes it instead of guessing. Take the decoupled filter `Z =
sum_k γ_k g_k(L̂) X` and look at the optimization over the coefficients `γ` alone, near the optimum, under
a squared loss. This is convex in `γ`, so gradient descent's convergence rate is set by the condition
number `κ(H)` of the Hessian `H`. Compute `H`: its `(k₁,k₂)` entry is `X^T g_{k₂}(L̂) g_{k₁}(L̂) X =
sum_i g_{k₁}(λ_i) g_{k₂}(λ_i) X̂²_{λ_i}`, summing over the spectrum, where `X̂²_{λ_i}` is the signal's
energy at frequency `λ_i`. In the continuum, `H_{k₁k₂} = ∫₀² g_{k₁}(λ) g_{k₂}(λ) f(λ) dλ`, where `f(λ)`
is the **spectral density of the graph signal** — how the signal's energy is distributed over frequency.
So the Hessian is exactly the Gram matrix of the basis functions under the inner product weighted by `f`.
And the whole optimization difficulty — the thing producing the monomial filter's seed variance — is
`κ(H)`. There it is, explicit: a basis is easy to optimize precisely when its Gram matrix under the signal
density is well-conditioned.

I want to see this bite with numbers before I trust it to choose a basis, so I will compute the monomial
Gram matrix under a simple density and read off its condition number. Take the flat density `f ≡ 1` on
`[-1,1]` (the benign case — nothing pathological about the signal). The monomial Gram is `H_{k₁k₂} =
∫_{-1}^1 x^{k₁+k₂} dx`, which is `2/(k₁+k₂+1)` when the sum is even and `0` when odd. For the degree-1
basis `{1, x}` that is `H = diag(2, 2/3)`, condition number `3`. For `{1, x, x²}` it becomes `[[2,0,2/3],
[0,2/3,0], [2/3,0,2/5]]`, whose eigenvalues are about `2.24, 0.667, 0.158` — condition number `14.1`. For
degree 3 it is already `67.6`, and it keeps climbing steeply with degree. So even under the most benign
density imaginable, the monomial basis's optimization Hessian is ill-conditioned and worsening fast with
`K` — and I am running `K=10`, far past degree 3. That is the mechanism of the seed variance, computed:
the descent crawls along the ill-conditioned directions and the seed picks which one it happens to make
progress on. Now the contrast that makes the punchline concrete: the Legendre polynomials `P_0=1, P_1=x,
P_2=(3x²-1)/2` are *orthogonal* under exactly this flat density — `∫_{-1}^1 P_0 P_2 dx = ∫(3x²-1)/2 dx =
(3·(2/3) - 2)/2 = 0` — so their Gram matrix is *diagonal*, condition number `1` after normalization.
Switching from the monomial basis to an orthogonal one, under the same density, takes `κ` from `14`
(degree 2) toward `1`. The reframe is now a computation, not a slogan.

That gives me the selection rule. `κ(H)` is smallest, equal to 1, exactly when `H = I` — when the basis is
**orthonormal under the density-weighted inner product** `⟨g,h⟩ = ∫ g h f dλ`. So the optimal basis is not
fixed; it is whichever orthogonal family has weight function equal to the graph's signal density `f`. This
explains every result on the ladder at once. The monomial basis I just used is *provably not orthogonal
under any weight function* — its Hessian is the dense, ill-conditioned Gram I just computed, the worst case
— which is precisely why it optimizes well *on average* (it is expressive) but with the *highest variance*
(the descent is poorly conditioned, so the seed decides where it lands). The Bernstein basis is also
non-orthogonal. The Chebyshev basis ChebNetII used *is* orthogonal — but only under *one fixed* weight,
`(1-x²)^{-1/2}`. If the graph's `f` happens to look like that weight, Chebyshev is near-optimal; but `f`
is *not* the same across these four graphs — homophilic Cora and Citeseer have their signal energy at
*low* frequency, heterophilic Texas and Cornell at *high* frequency — so one fixed weight cannot match all
four, and Chebyshev's mismatch on the WebKB graphs is the stiffness I diagnosed two rungs ago. No
fixed-weight basis can be optimal across graphs whose densities differ. I need a basis whose weight I can
*tune to match `f`* — and that is a basis the ladder has not tried.

That requirement names the family. Among the classical orthogonal polynomials, the **Jacobi** family
`P_k^{a,b}` is orthogonal under the weight `(1-x)^a (1+x)^b` on `[-1,1]`, with two free parameters `a, b`
that reshape the weight continuously. Chebyshev is the *single special case* `a=b=-1/2`, Legendre is
`a=b=0` — so Jacobi *contains* both bases the ladder already used and adds two knobs to slide the weight
toward whichever end of the spectrum the graph's energy occupies. I can see the sliding concretely by
evaluating the three weights near the endpoints: at `x=0.9` the Chebyshev weight `(1-x²)^{-1/2}` is `2.29`
— it *blows up* toward the spectral extremes; the Legendre weight `(1-x²)^0` is `1` — flat; and the
Jacobi weight `(1-x)(1+x) = 1-x²` at `a=b=1` is `0.19` — it *vanishes* toward the extremes and peaks at
the center. So the `a=b` axis slides the emphasis from the two ends (`a=b=-1/2`), through flat (`a=b=0`),
to the middle (`a=b=1`). On the normalized adjacency `Â = I − L`, whose spectrum is `[-1,1]`, low graph
frequency `λ≈0` maps to `x≈1` and high `λ≈2` to `x≈-1`. So for a homophilic graph (low-frequency energy)
I would push the Jacobi weight toward `x=1` by lowering the exponent on `(1-x)` relative to the one on
`(1+x)`; for a heterophilic graph (high-frequency) I would push it toward `x=-1` by lowering the exponent
on `(1+x)`. Tuning `a, b` *is* matching the weight to `f`, which is the `κ(H)→1` knob made an explicit
hyperparameter. This is the move that keeps everything the monomial filter got right — unconstrained,
sign-free, learned coefficients, flat uniform start — while fixing the one thing it got wrong: it replaces
the non-orthogonal monomial basis, whose bad conditioning produced the seed variance, with the *adaptable*
orthogonal basis that drives the Hessian toward identity. I do not have to *compute* the optimal `a, b`
(that needs the eigendecomposition I cannot afford); I pick the flexible family and a sensible default, and
the weight sits close enough to `f` to condition the descent.

There is a deeper licence here too, one I should state because it changes what I am even building. Across
all three prior rungs I kept the nonlinear ReLU MLP encoder without questioning it. But a spectral GNN is
a filter, and I can ask whether a *linear* spectral GNN — drop the activation — is already enough. The
answer is yes under two conditions: the normalized Laplacian must have no repeated eigenvalues, and the
features must miss no frequency component. Repeated eigenvalues come from graph symmetry; these benchmark
graphs are irregular and fewer than 1% of eigenvalues are multiple, and no frequency component is missing.
ReLU's only spectral effect is to *mix* frequency components — the repair for exactly those two
obstructions — so on these graphs it fixes a disease the data does not have. The principled version is
therefore a *linear* model, which is also what made the Hessian argument above clean: the coefficient fit
is convex only if the encoder does not fold nonlinearity into the spectrum. The harness I am editing keeps
the ReLU between its two linear layers, so in this task I land the Jacobi *propagation* into the existing
decoupled encoder rather than stripping it to a single linear layer — the basis change is where the
leverage is, and it carries over whether or not the encoder is linear.

Now I have to be honest about exactly what this task's edit surface lets me build, because the principled
method has two refinements the harness cannot express, and I must land the version the scaffold supports,
not the full one. First, *individual filters per output channel*: the universality argument is
one-dimensional, and a single shared filter cannot produce an arbitrary multi-class prediction, so the
full method gives each output channel its own coefficient vector `α_{kl}`. But this task's `CustomProp`
owns a *single* shared coefficient vector `temp` of length `K+1`, applied identically to every channel —
there is no per-channel filter slot. Second, *polynomial coefficient decomposition (PCD)*: the full method
decomposes `α_{kl} = β_{kl} Π_i γ_i` with shared bounded ratios to rescale the per-`k` magnitudes, which
is what realizes the orthogonal basis's good conditioning in practice. The harness exposes only the raw
`temp[k]`, so there is no PCD here either. So what I am actually landing is the **shared-coefficient,
fixed-`(a,b)` Jacobi filter** — the orthogonal-basis core of the method without the per-channel filters and
without PCD. That is the honest delta over the monomial rung: *same* unconstrained learned coefficients,
*same* uniform `1/(K+1)` start, *same* decoupled encoder — but the monomial powers `P^k` are replaced by
the Jacobi polynomials `P_k^{a,b}(Â)`, with `a=b=1` as the symmetric default. I choose `a=b=1` rather than
Chebyshev's `a=b=-1/2` or Legendre's `a=b=0` for two reasons I can now state precisely: it is symmetric, so
it commits to neither spectral end (the right stance when I am not tuning per-graph and the four graphs
split between low- and high-frequency energy), and the symmetry makes the recurrence's leading term vanish
— with `a=b`, `a²−b²=0`, so the coefficient `A_k = (2k+a+b-1)(a²−b²)/denom` is exactly zero at every `k`,
and the recurrence loses its asymmetric drift term, becoming a clean symmetric three-term recursion. And I
should confirm the family I am about to build is actually orthogonal under its claimed weight, not just
assert it, because the whole conditioning argument rests on that. The first three `a=b=1` polynomials are
`P_0=1`, `P_1=2x`, `P_2=3.75x²−0.75` (the ones I trace from the recurrence below). Under the weight `w =
1−x²`, the pairs `P_0,P_1` and `P_1,P_2` integrate to zero by parity alone (odd integrand on `[-1,1]`), so
the only non-trivial check is `P_0 ⊥ P_2`: `∫_{-1}^1 (3.75x²−0.75)(1−x²)\,dx = ∫_{-1}^1(−3.75x⁴ + 4.5x² −
0.75)\,dx = −3.75(2/5) + 4.5(2/3) − 0.75(2) = −1.5 + 3.0 − 1.5 = 0`. It vanishes exactly, so the first
three orders are mutually orthogonal under `(1−x²)` — the Gram matrix is diagonal at this end of the basis,
the `κ→1` property I am buying, confirmed by a computation rather than a citation.

The propagation itself stays linear in `K`, which I want on the record since cheapness has been a running
theme. I build `L` with the symmetric-normalized `get_laplacian`, shift to `L̃ = L − I = −Â` by adding
self-loops of weight `−1`, and run the Jacobi three-term recurrence: `P_0 = x`, `P_1 = ((a+b+2)/2)·(L̃ x)
+ ((a-b)/2)·x`, and for `k ≥ 2`, `P_k = (A_k + B_k L̃) P_{k-1} − C_k P_{k-2}` with the Jacobi coefficients
`A_k, B_k, C_k` computed in closed form from `a, b, k`. Each step is one `propagate` (one sparse mat-vec),
so the whole filter is `K` propagations — `O(K m d)`, the same order as the monomial and Chebyshev rungs,
strictly cheaper than BernNet, and with no DCT scalar transform at all: the `A_k, B_k, C_k` are three
scalar closed forms per order. I want to trace the recurrence at a=b=1 by hand for two orders, both to
confirm the coefficient formulas and to check the parity that the shift will exploit. At `k=1`, `c_0 =
(a-b)/2 = 0` and `c_1 = (a+b+2)/2 = 2`, so `P_1 = 2 L̃ x` — an odd polynomial `2x`. At `k=2`, `denom =
2·2·(2+2)·(4+2-2) = 64`, `A_2 = (5·0)/64 = 0` (the symmetric-default cancellation, confirmed), `B_2 =
(5·4·6)/64 = 1.875`, `C_2 = (2·2·2·6)/64 = 0.75`, giving `P_2 = 1.875 L̃(2 L̃ x) − 0.75 x = 3.75 L̃² x −
0.75 x` — an even polynomial `3.75x²−0.75`. So `P_1` is odd and `P_2` is even, which is the parity I need
for the one subtlety I have to get right: running the recurrence on `L̃ = −Â` rather than `Â` evaluates
`P_k(−x) = (−1)^k P_k^{a,b}(x)` — the Jacobi reflection identity, symmetric when `a=b` — so the operator I
build for free flips the odd orders' sign and leaves the even ones, and that alternating `(−1)^k` is
absorbed completely by the *learned* `temp[k]` (a sign flip on a free coefficient is a no-op on the family
it spans). The two hand-traced orders confirm it: `P_1(−x)=−P_1(x)`, `P_2(−x)=+P_2(x)`, exactly the
`(−1)^k` pattern. So the filter family is identical to the `Â` convention and I lose nothing by using the
operator the harness constructs. The literal edit replaces the two class bodies (`CustomProp` becomes the
Jacobi layer with its `a, b` and recurrence, `CustomFilter` keeps the standard encoder/readout and passes
`a=b=1` into the propagation); the full scaffold module is in the answer.

So the bar this rung must clear, against the monomial filter's real numbers, and what I would validate.
The whole motivation was *conditioning*, not the means, so the first thing I expect is not a dramatic mean
jump but a *quieter* one: the Jacobi basis should match or modestly exceed the monomial accuracies — Cora
at or above 0.8890, Citeseer at or above 0.8020, Texas at or above 0.9065, Cornell at or above 0.8705 —
while *shrinking the variance* that betrayed the monomial basis's ill-conditioned Hessian. Concretely, the
place to watch is the heterophilic std: the monomial filter's Texas std of 0.0361 (spanning 0.0262 to
0.0471 across seeds) and Cornell 0.0337 are exactly what a better-conditioned orthogonal basis should pull
in toward the tighter ~0.025 range the constrained rungs held there, and the dropped Cora run and the
seed-to-seed spreads (Texas 0.0049, Cornell 0.0099) should shrink toward the near-invariance a well-posed
descent produces. That — a falling std and vanished dropped runs at held-or-higher means — is the result
that would justify the orthogonal basis as the right ceiling for this ladder. The cleanest falsification
would be the opposite profile: if the shared-coefficient Jacobi filter — shorn of the per-channel filters
and PCD the full method relies on — lands *below* the monomial filter on any graph, that would say the
conditioning win does not survive without PCD on this harness, and the plain monomial filter remains the
right answer here. And I should name the honest risk that points that way: with `a=b` fixed at the
symmetric default (no per-dataset tuning of the weight toward each graph's density) and no PCD, the Jacobi
advantage on this harness may be *small* — the orthogonality is only *exactly* matched to `f` when `a, b`
track each graph's density, and a single symmetric weight, peaked at mid-frequency and vanishing at the
ends, leaves some conditioning on the table for graphs whose energy sits at an end. But even the untuned
symmetric Jacobi basis is orthogonal under *a* weight, where the monomial basis is orthogonal under *none*
and carries the `κ≈14`-and-climbing Gram I computed, so I expect the conditioning to improve and the
variance to fall — and that, more than a fractional accuracy gain, is the outcome I am after.
