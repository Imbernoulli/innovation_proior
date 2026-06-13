ChebNetII came in almost exactly where I bet it would on the homophilic graphs, and almost exactly
where I feared on the heterophilic ones — and the *way* it landed is the tell that reframes the whole
ladder. On Cora it climbed to 0.8725 and on Citeseer to 0.8004, both clearly above BernNet's 0.8554 and
0.7795: the linear-cost near-minimax interpolation bought back the smooth-low-pass resolution Bernstein
under-resolved at fixed `K`, exactly as predicted. The seed-to-seed variance collapsed to almost
nothing — Cora 0.8724/0.8726/0.8724, Citeseer within 0.0006 — which is the signature of tying the
coefficients to a stable interpolation problem rather than a free fit. So the homophilic story is a
clean win. But the heterophilic prediction also came true, and harder than I wanted: Texas *fell* to
0.8770, below BernNet's 0.9093, and Cornell sat flat at 0.8470. The inversion I flagged is real —
ChebNetII wins the citation graphs and loses Texas — and that means neither the Bernstein certificate
nor the Chebyshev interpolant is uniformly best. Now here is the thing that actually nags me: ChebNetII's
near-perfect seed-invariance on the citation graphs is not only a feature, it is a *symptom*. A filter
whose three seeds land within 0.0006 of each other on Cora is a filter that is barely moving off its
initialization — the optimizer is finding nearly the same response every time. That is wonderful when the
target is a single smooth low-pass, but it suggests the parameterization is *over-regularized*: the
ReLU-on-values plus the DCT plus the near-minimax node placement together pin the filter so tightly that
it cannot reach the sharper, possibly sign-changing response Texas needs — which is exactly why Texas
regressed. So my question for this rung is: can I keep the linear cost, but *loosen* the parameterization
so the filter can swing further toward the heterophilic responses, and recover the texas/cornell edge a
constrained basis gives up?

Let me reconsider what each rung actually constrained, because the pattern across the two failures points
somewhere specific. BernNet constrained the response to be *non-negative everywhere* (a global
certificate) and paid with quadratic cost and slow convergence. ChebNetII constrained the response to be
a *near-minimax interpolant of non-negative sampled values* and paid with a filter so stiff it under-fits
the heterophilic graphs. Both constraints were motivated by the fear that an unconstrained filter would
overfit — the original ChebNet's disease, free Chebyshev coefficients piling weight on high frequencies
and losing even to GCN. But look at where the two constrained filters actually *lose*: not on the large
citation graphs where overfitting capacity would be dangerous, but on Texas and Cornell — graphs with
**183 nodes**, where the heterophilic response is sharp and sign-changing, and where the constraint is
not protecting against overfitting so much as *forbidding the very response the graph needs*. The
non-negativity constraint, in particular, is suspect: a true heterophilic filter is high-pass, and a
high-pass response on the normalized spectrum can want to *alternate sign* across frequency — exactly
what a non-negative-everywhere or non-negative-at-nodes constraint cannot express. So the lesson the two
feedbacks are jointly teaching me is that the *constraint* — the thing both prior rungs were proud of —
is the thing capping heterophilic performance. The next move is to drop it: parameterize the filter so
the coefficients are *unconstrained* and free to take either sign, and trust a *different* mechanism than
a hard constraint to control overfitting.

That sounds like a step backward — it is the free-coefficient ChebNet disease all over again — unless I
change two things at once. First, the *basis*. The ChebNet failure was free coefficients in a basis
where high-`k` terms are high-frequency cosines, so unconstrained learning piles capacity exactly where
it overfits. But there is a basis where the natural, regularization-friendly default is *not* a wild
response: the **monomial** basis in the GCN-normalized adjacency, `h(P) = sum_k γ_k P^k` with `P =
D^{-1/2} A D^{-1/2}`. Here `P^k` is the `k`-hop propagation operator, and the coefficient `γ_k` is the
weight on information from `k` hops away — a directly interpretable quantity, not an abstract Chebyshev
amplitude. A node's label, on either a homophilic or a heterophilic graph, depends on its neighborhood
at a few hops; learning *how much* to trust each hop, with the sign free, is exactly the degree of
freedom Texas needs (it can learn to *subtract* the 1-hop average to build a contrast) and exactly what
the non-negative bases forbade. The monomial basis is the one the scaffold default already uses (frozen
at PPR); the move is to *unfreeze* it and let every hop-weight be learned, sign and all. Yes, the
monomial basis is ill-conditioned — the powers `P^k` become collinear as `k` grows — and that is a real
objection that motivated the *Chebyshev* rung in the first place. But conditioning is an
*approximation-theory* concern about fitting an arbitrary target to high precision; here `K=10` is
modest, the target responses are not pathological, and the empirical record on these exact graphs is
blunt: a learned monomial filter outperforms a free Chebyshev filter on the citation graphs despite the
conditioning. So I will accept the conditioning cost to buy the unconstrained, sign-free, hop-interpretable
coefficients.

Second, and this is what keeps the unconstrained filter from overfitting the way ChebNet did: the
*regularization* moves from a hard architectural constraint to a *soft* one — weight decay on the
coefficients, plus a careful learning-rate split, plus the right *initialization*. This is where I
deliberately depart from the textbook monomial filter, which initializes the hop-weights to the PPR
pattern `γ_k = α(1-α)^k` (a decaying low-pass, the same shape the scaffold default freezes). PPR
initialization bakes in the homophily prior — it starts the filter as a low-pass and asks the optimizer
to climb out of it toward high-pass on heterophilic graphs. On a 183-node heterophilic graph with little
data, that starting bias is a headwind: the optimizer may not travel far enough from a low-pass start to
reach the contrast filter Texas needs, which is one more way a homophily prior caps heterophilic
accuracy. So I initialize **uniformly**: every `γ_k = 1/(K+1)`, an equal-weight average over all hops `0`
through `K`. That is dataset-agnostic — it commits to neither low- nor high-pass, it is the flat,
unbiased start (the same neutrality logic that made all-ones the right init for the previous two rungs,
here expressed as the uniform hop-average) — and it lets the optimizer move freely toward whichever
response the labels demand, low-pass on Cora or sign-alternating on Texas, without first having to undo a
baked-in prior. This is the single most important departure of this rung: *uniform, not PPR,
initialization*, precisely so the heterophilic graphs are reachable.

The training-hyperparameter split is the other deliberate departure, and it is the soft regularization
that replaces the hard constraints. I give the *filter coefficients* the **same** learning rate as the
MLP — `0.05` for both, where the scaffold default would slow the propagation parameters to `0.01`.
ChebNetII's stiffness, I now think, was partly that its parameterization plus a slow propagation LR meant
the filter barely moved; here I *want* the unconstrained hop-weights to move fast enough to find the
heterophilic response, so I do not throttle them. To control the overfitting that fast, unconstrained,
ill-conditioned coefficients invite, I lean on weight decay where it belongs: a small `5e-4` weight decay
on the MLP encoder (the high-capacity part, where overfitting on a 183-node graph is the real danger),
and **zero** weight decay on the `K+1` filter coefficients themselves (decaying the filter toward zero
would bias it back toward an over-smooth response — the wrong prior). So the regularization budget is
spent on the encoder, not the filter, and the filter is left free and fast. Propagation dropout stays off
(`dprate=0.0`), consistent with the whole task's finding that propagation dropout hurts spectral filters
on heterophilic data.

The propagation itself is the cheapest of all three rungs and that is a virtue I should state plainly,
since BernNet's quadratic cost was a named weakness. The monomial filter is `hidden = γ_0 x; for k:
x ← P x; hidden += γ_{k+1} x` — `K` sparse mat-vecs of the GCN-normalized adjacency, accumulating the
learned hop-weights as I go. That is `O(K m d)`, linear in `K`, the same order as ChebNetII and strictly
cheaper than BernNet, with no DCT coefficient transform and no Laplacian shift — just the normalized
adjacency powers. So this rung is simultaneously the *least constrained* and the *cheapest*: I am betting
that on these graphs the right design is not a clever constrained basis but the plainest possible
learnable hop-mixer, freed of every prior, regularized softly, and started flat. The literal edit
replaces the two class bodies and sets the four `custom_*` training overrides; the full scaffold module
is in the answer.

Now the falsifiable expectations against ChebNetII's numbers, because this rung either justifies dropping
the constraint or it does not. The whole motivation is the heterophilic regression, so that is the first
thing I expect to fix: Texas should climb back off ChebNetII's 0.8770 toward 0.90 — recovering and
plausibly exceeding it, because the unconstrained sign-free coefficients with a uniform start can build
the contrast response that both non-negative bases forbade — and Cornell should rise off 0.8470 toward
0.87, since the same freedom helps the second WebKB graph. The risk is the mirror of ChebNetII's win: by
dropping the constraint and the near-minimax conditioning I might *give back* the homophilic gain, so
Cora and Citeseer are the graphs to watch for a regression. My honest prediction is that they will *not*
regress and may edge slightly higher — Cora toward 0.89, Citeseer holding near 0.80 — because the
uniform-init monomial filter is perfectly capable of learning a smooth low-pass on a clean citation
graph (a decaying `γ_k` is right there in its reach), and the soft weight decay on the encoder controls
overfitting without stiffening the filter. The second thing I expect is *more* seed-to-seed variance than
ChebNetII's near-zero — an unconstrained, fast-learning filter will not land in the identical place every
run — but I expect that variance to stay small and the *mean* to be higher across the board. If both hold
— heterophilic recovered (Texas back near 0.90, Cornell up) *and* homophilic held-or-improved (Cora and
Citeseer at or above ChebNetII) — then the unconstrained, uniform-init, fast-coefficient monomial filter
dominates both constrained rungs on this harness, and the ladder's verdict is that on these four graphs
the constraint machinery the prior rungs were built around was the thing holding them back. That is the
result I am chasing: the simplest filter, freed of its priors, as the strongest.
