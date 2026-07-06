ChebNetII came in almost exactly where I bet it would on the homophilic graphs, and almost exactly where
I feared on the heterophilic ones — and the *way* it landed is the tell that reframes the whole ladder.
On Cora it climbed to 0.8725 and on Citeseer to 0.8004, both clearly above BernNet's 0.8554 and 0.7795:
the linear-cost near-minimax interpolation bought back the smooth-low-pass resolution Bernstein
under-resolved at fixed `K`, exactly as predicted. And the seed-to-seed variance is the tell I flagged.
ChebNetII's three seeds on Cora land at 0.8724/0.8726/0.8724 — a spread of 0.0002 — where BernNet's three
Cora seeds spanned 0.8563/0.8552/0.8547, a spread of 0.0016, eight times wider; and on Texas ChebNetII
reads 0.8770/0.8770/0.8770, *identical to four digits across three seeds*. That is not merely stable, it
is a filter that finds the same response no matter where it starts, which is wonderful when the target is
a single smooth low-pass and a symptom when it is not. Because the heterophilic prediction also came true,
and harder than I wanted: Texas *fell* to 0.8770, below BernNet's 0.9093, and Cornell sat flat at 0.8470.

Now I want to do the arithmetic across the two rungs before I choose a direction, because a delta table
reads differently than two separate stories. ChebNetII minus BernNet, per graph: Cora +0.0171, Citeseer
+0.0209, Texas −0.0323, Cornell +0.0027. The two homophilic gains average about +0.019; the Texas loss
is −0.032, larger in magnitude than either gain; Cornell barely moved. Averaging the four datasets:
BernNet sits at 0.8471, ChebNetII at 0.8492 — a net improvement of **0.0021**. Two full rungs of
constraint machinery — a global non-negativity certificate, then a near-minimax interpolant of
non-negative sampled values — bought two-tenths of one percent on average, because the Texas regression
very nearly cancelled the citation-graph gains. That number is the thing that reorients me. It is not that
ChebNetII is bad; it is that the constraint I keep being proud of is buying almost nothing net, and it is
buying it by *trading* one region for another rather than lifting the whole board. So the question for
this rung is not "what other constraint." It is: can I keep the linear cost, but *loosen* the
parameterization so the filter can swing further toward the heterophilic responses, and recover the
Texas/Cornell edge a constrained basis gives up — without giving back the homophilic gain?

The across-split std tables add two cautions I want to carry into my predictions rather than discover
afterward. First, the improvement ChebNetII bought on Citeseer came with *more* split-to-split noise, not
less: BernNet's Citeseer std was 0.0056 — the tightest cell on the board — and ChebNetII's is 0.0149,
nearly three times wider, even as its Citeseer mean rose from 0.7795 to 0.8004. So the near-minimax
interpolation raised the Citeseer ceiling but made the filter more split-sensitive there, and both
constrained rungs stalled the Citeseer *mean* right around 0.80. That is a hint that Citeseer may be near
a genuine plateau — a clean six-class citation graph whose smooth low-pass is nearly saturated by any
adequately-resolving filter — so my honest expectation for this rung on Citeseer is *hold near 0.80*, not
a further clear climb; freedom in the basis is unlikely to break a ceiling that is about the graph, not
the constraint. Second, Cornell is the noisiest graph on the whole board — std 0.0408 for BernNet and
0.0462 for ChebNetII, the two largest cells anywhere — which is what 183 nodes split ten ways with a
sharp heterophilic signal will do regardless of the filter. So whatever this rung does to Cornell's
*mean*, I should expect its across-split std to stay the largest of the four, and its seed-to-seed spread
to widen too once I unthrottle the coefficients: an intrinsically high-variance target plus a fast,
unconstrained filter is a combination that will not land tightly. I will read a Cornell mean gain as the
signal and not be alarmed by a wide Cornell std, because the std is the graph's, not the method's.

Let me reconsider what each rung actually constrained, because the pattern across the two failures points
somewhere specific. BernNet constrained the response to be *non-negative everywhere* (a global
certificate) and paid with quadratic cost and slow convergence. ChebNetII constrained the response to be a
*near-minimax interpolant of non-negative sampled values* and paid with a filter so stiff — the Texas
seeds identical to four digits — that it under-fits the heterophilic graphs. Both constraints were
motivated by the fear that an unconstrained filter would overfit: the original ChebNet's disease, free
Chebyshev coefficients piling weight on high-frequency cosines and losing even to GCN, which is exactly
why a free-coefficient Chebyshev filter comes in last on these graphs. But look at where the two
constrained filters actually *lose*: not on the large citation graphs where overfitting capacity would be
dangerous, but on Texas and Cornell — graphs with **183 nodes**, roughly 110 training nodes after the
split, where the heterophilic response is sharp and sign-changing, and where the constraint is not
protecting against overfitting so much as *forbidding the very response the graph needs*. The
non-negativity constraint, in particular, is suspect, and I can pin down why with a concrete filter. A
true heterophilic filter is a high-pass, and a high-pass response on the normalized spectrum can want to
*change sign* across frequency — to actively negate a band, not merely to attenuate it. Take the simplest
signed monomial, `γ = (0, -1, 1)`, i.e. `h(P) = P^2 - P`, whose response is `h(μ) = μ^2 - μ = μ(μ - 1)`.
At the two ends it reads `h(1) = 0` (kill the smooth mode) and `h(-1) = 2` (double the roughest mode) — a
clean high-pass — but in the middle it dips *negative*: `h(0.5) = 0.25 - 0.5 = -0.25`. That negative lobe
across `μ ∈ (0,1)` is a response value below zero at frequencies a heterophilic graph may well want to
invert, and it is exactly what a non-negative-everywhere basis, or a ReLU-on-sampled-values basis, cannot
represent — ChebNetII's ReLU forces `h` non-negative at every interpolation node, so a response that dips
to `-0.25` anywhere near a node is simply outside its reach. So the lesson the two feedbacks are jointly
teaching me is that the *constraint* — the thing both prior rungs were proud of — is the thing capping
heterophilic performance. The next move is to drop it: parameterize the filter so the coefficients are
*unconstrained* and free to take either sign, and trust a *different* mechanism than a hard constraint to
control overfitting.

That sounds like a step backward — it is the free-coefficient ChebNet disease all over again — unless I
choose deliberately, so let me walk the options actually on the table rather than jump. Option one is to
loosen ChebNetII directly by removing its ReLU, learning the Chebyshev interpolation values (or
coefficients) freely. That is precisely the free-Chebyshev filter the record already condemns: in the
Chebyshev basis the high-`k` terms are high-frequency cosines, so unconstrained learning piles capacity
where it overfits, and the five knobs `w_6..w_10` have more than enough freedom to fit noise on ~110 WebKB
training nodes. Rejected — it is the very disease I am trying to route around. Option two is the `1/k`
reparameterization, damping the Chebyshev coefficients by a fixed schedule. It suppresses the
high-frequency overfitting, but a *fixed* `1/k` decay is dataset-blind: a graph whose true response is
genuinely sharp — a near-non-analytic high-pass, plausibly what Texas wants — would be over-damped by a
mandatory `1/k`, so it re-creates the stiffness I am trying to escape, just softer, and it still cannot go
sign-free-and-fast in an interpretable way. Rejected. Option three is to change the *basis* to one whose
natural, regularization-friendly default is *not* a wild response: the **monomial** basis in the
GCN-normalized adjacency, `h(P) = sum_k γ_k P^k` with `P = D^{-1/2} A D^{-1/2}`. Here `P^k` is the `k`-hop
propagation operator, and the coefficient `γ_k` is the weight on information from `k` hops away — a
directly interpretable quantity, not an abstract Chebyshev amplitude. A node's label, on either a
homophilic or a heterophilic graph, depends on its neighborhood at a few hops; learning *how much* to
trust each hop, with the sign free, is exactly the degree of freedom Texas needs — it can learn to
*subtract* the 1-hop average to build the contrast whose signed response I just computed — and exactly
what the non-negative bases forbade. The monomial basis is the one the scaffold default already uses
(frozen at PPR); the move is to *unfreeze* it and let every hop-weight be learned, sign and all.

I have to be honest that the monomial basis is the one I argued *against* two rungs ago: its powers `P^k`
go collinear as `k` grows — I measured consecutive high powers sitting `2.9°` apart on the spectrum — so
it is ill-conditioned, and that ill-conditioning is a real objection, the one that motivated the Chebyshev
rung in the first place. But conditioning is an *approximation-theory* concern about fitting an arbitrary
target to high precision; here `K=10` is modest, the target responses are not pathological, and the
empirical record on these exact graphs is blunt: a learned monomial filter outperforms a free Chebyshev
filter on the citation graphs *despite* the conditioning. So I will accept the conditioning cost to buy
the unconstrained, sign-free, hop-interpretable coefficients — I expect it to show up as somewhat noisier
runs than ChebNetII's near-frozen seeds, and I am willing to pay that in variance for the mean.

Second, and this is what keeps the unconstrained filter from overfitting the way ChebNet did: the
*regularization* moves from a hard architectural constraint to *soft* ones — the right initialization, a
learning-rate split, and weight decay placed where it belongs. Start with initialization, because it is
where I deliberately depart from the textbook monomial filter, which initializes the hop-weights to the
PPR pattern `γ_k = α(1-α)^k`. Written out at `α=0.1`, that is `0.100, 0.090, 0.081, ..., 0.035` — a
monotone-decaying, geometrically-falling profile, the numeric fingerprint of a low-pass, the same shape
the scaffold default freezes. PPR initialization bakes in the homophily prior: it starts the filter as a
low-pass and asks the optimizer to climb out of it toward a high-pass on heterophilic graphs. On a
183-node graph with ~110 training nodes, that starting bias is a headwind — the gradient must first undo a
committed decaying shape before it can build a contrast, and it may not travel far enough. So I initialize
**uniformly**: every `γ_k = 1/(K+1) = 1/11 ≈ 0.0909`, a flat, equal-weight average over all hops `0`
through `K`. I want to be precise about the sense in which this is "neutral," because I can compute the
two inits and they are neutral in different amounts along different axes. In *coefficient* space the PPR
vector is `(0.100, 0.090, 0.081, ..., 0.035)` — front-loaded, `γ_0` roughly three times `γ_10`, a
committed decaying shape — while the uniform vector is `(0.0909, ..., 0.0909)`, flat. To build a high-pass
the optimizer must lower the low hops and raise the high ones; from uniform every hop starts at the same
value so the gradient can push any coordinate up or down symmetrically, whereas from PPR the low hops
start *higher* (more to remove) and the high hops start *lower* (more to add), so PPR is strictly farther
from a high-pass in every coordinate that has to change. That coefficient-space symmetry is the axis that
matters for reachability. In *response* space the picture is subtler and I should not overclaim: the
uniform-init response is `h(μ) = (1/11)Σ_{k=0}^{10} μ^k`, which reads `h(1) = 1` at DC and `h(-1) =
(1/11)Σ(-1)^k = 1/11 = 0.0909` at the roughest mode — so it is not perfectly flat, it is a *mild*
low-pass, because low-order hops dominate the DC sum. Uniform is neutral in coefficient space, not in
response space. But PPR is committed on *both* axes at once — front-loaded coefficients *and* a sharper
low-pass response (`h(1) = 0.686`, `h(-1) = 0.069`, a lower and steeper roll-off) — so on the axis that
governs how freely gradient descent can travel, uniform is strictly the less-committed start. That is why
I take it: the optimizer can move toward whichever response the labels demand, low-pass on Cora or
sign-alternating on Texas, without first spending gradient to undo a baked-in prior. This is the
same neutrality logic that made all-ones the right init for the previous two rungs (there it produced the
flat all-pass response; here the uniform hop-average is its monomial expression), and it is the single
most important departure of this rung: *uniform, not PPR, initialization*, precisely so the heterophilic
graphs are reachable.

The training-hyperparameter split is the second deliberate departure, and it is the soft regularization
that replaces the hard constraints. I give the *filter coefficients* the **same** learning rate as the MLP
— `0.05` for both — where the scaffold default throttles the propagation parameters to `0.01`, a fifth of
the encoder rate. This matters more than it looks, and I think it is half of ChebNetII's stiffness story.
ChebNetII left the training overrides at the scaffold defaults, so its filter coefficients crawled at
`0.01` while the encoder moved at `0.05` — a stiff parameterization *and* a throttled learning rate
together mean the filter barely leaves its all-ones start, which is a mechanical explanation for Texas
landing on 0.8770 identically across three seeds: a filter that does not move cannot disagree with itself
between seeds. Here I *want* the unconstrained hop-weights to move fast enough to find the heterophilic
response, so I do not throttle them; I set `custom_prop_lr = 0.05` to match `custom_lr = 0.05`. To control
the overfitting that fast, unconstrained, ill-conditioned coefficients invite, I lean on weight decay
where it belongs, and the parameter counts tell me where that is. On Cora the encoder alone is about
`1433·64 + 64·7 ≈ 92,000` weights; on the WebKB graphs, with `1703` input features, `lin1` alone is
`1703·64 + 64 ≈ 109,000` weights — over a hundred thousand parameters trained against roughly `110`
labelled nodes, a massively over-parameterized encoder that is the real overfitting surface. The filter,
by contrast, is `K+1 = 11` parameters. So I spend the regularization budget on the encoder — `custom_wd =
5e-4` on `lin1/lin2` — and set **zero** weight decay on the filter coefficients (`custom_prop_wd = 0`):
decaying eleven hop-weights toward zero would shrink the filter toward the null response, biasing it back
toward over-smoothing — the wrong prior, and the exact homophily bias I removed from the initialization.
The regularization budget is spent on the hundred-thousand-parameter encoder, not on the eleven-parameter
filter, and the filter is left free and fast. Propagation dropout stays off (`dprate = 0.0`), consistent
with the whole task's finding that propagation dropout hurts spectral filters on heterophilic data by
deleting the very high-frequency contrast the filter is building.

The propagation itself is the cheapest of all three rungs and that is a virtue I should state plainly,
since BernNet's quadratic cost was a named weakness. The monomial filter is `hidden = γ_0 x; for k:
x ← P x; hidden += γ_{k+1} x` — `K` sparse mat-vecs of the GCN-normalized adjacency, accumulating the
learned hop-weights as I go. That is `O(K m d)`, linear in `K`, the same order as ChebNetII but with *no*
DCT coefficient transform and *no* Laplacian shift — ten propagations of the normalized adjacency and a
running sum, strictly the leanest forward pass on the ladder. Counting it against the two prior rungs makes
the leanness concrete: BernNet spent `K + K(K+1)/2 = 65` sparse mat-vecs per forward pass at `K=10` and was
quadratic; ChebNetII spent `10` mat-vecs plus a `(K+1)² = 121`-entry scalar DCT to turn interpolation
values into coefficients, plus a self-loop Laplacian shift to build `L̂`; this filter spends `10` mat-vecs
and *nothing else* — no coefficient transform, no shift, the coefficients are used raw as they are learned.
So dropping the constraint did not merely free the response, it also stripped the machinery: the least
constrained rung is also the cheapest, which is the opposite of the usual trade and worth noting because it
means the bet costs me nothing in compute to place. So this rung is simultaneously the *least
constrained* and the *cheapest*: I am betting that on these graphs the right design is not a clever
constrained basis but the plainest possible learnable hop-mixer, freed of every prior, regularized softly,
and started flat. The literal edit replaces the two class bodies and sets the four `custom_*` training
overrides; the full scaffold module is in the answer.

Now the falsifiable expectations against ChebNetII's numbers, because this rung either justifies dropping
the constraint or it does not. The whole motivation is the heterophilic regression, so that is the first
thing I expect to fix: Texas should climb back off ChebNetII's 0.8770 toward 0.90 — recovering and
plausibly exceeding it, because the unconstrained sign-free coefficients with a uniform start can build
the contrast response, the negative-lobed high-pass, that both non-negative bases forbade — and Cornell
should rise off 0.8470 toward 0.87, since the same freedom helps the second WebKB graph and the encoder
weight decay controls the over-parameterization there. The risk is the mirror of ChebNetII's win: by
dropping the constraint and the near-minimax conditioning I might *give back* the homophilic gain, so Cora
and Citeseer are the graphs to watch for a regression. My honest prediction is that they will *not*
regress and may edge slightly higher — Cora toward 0.89, Citeseer holding near 0.80 — because the
uniform-init monomial filter is perfectly capable of learning a smooth low-pass on a clean citation graph
(a decaying `γ_k` is right there in its reach, one gradient descent away from the flat start), and the
soft weight decay on the encoder controls overfitting without stiffening the filter. The second thing I
expect is the accepted cost of the trade: *more* seed-to-seed variance than ChebNetII's near-zero — an
unconstrained, fast-learning, ill-conditioned filter will not land in the identical place every run the
way a frozen one does, so I should expect the WebKB std in particular to widen from ChebNetII's stiff
tightness. That is the price of the freedom and I am paying it deliberately; what I am buying is a higher
*mean* across the board. If both hold — heterophilic recovered (Texas back near 0.90, Cornell up) *and*
homophilic held-or-improved (Cora and Citeseer at or above ChebNetII), even at somewhat higher variance —
then the unconstrained, uniform-init, fast-coefficient monomial filter dominates both constrained rungs on
this harness, and the ladder's verdict is that on these four graphs the constraint machinery the prior
rungs were built around was the thing holding them back. That is the result I am chasing: the simplest
filter, freed of its priors, as the strongest.
