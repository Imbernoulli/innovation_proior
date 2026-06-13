The Bernstein run told me where its certificate helps and where it costs, and it told me in the
asymmetry I predicted. On the heterophilic WebKB graphs the non-negative basis did its job: Texas landed
at 0.9093 mean, the *highest* texas of anything I will run, and Cornell at 0.8443, both tight across
seeds — the controllable non-negative bumps found the high-frequency response those graphs need, and the
non-negativity constraint kept the filter from overfitting the tiny 183-node graphs. But on the
homophilic citation graphs it is soft, exactly where I said the slow `K^{-1/2}` Bernstein convergence at
a fixed `K=10` would bite: Cora 0.8554 and Citeseer 0.7795 are the lowest homophilic numbers I will see
from a learnable filter. The Citeseer gap is the loud one — 0.7795 is a clear step below where a smooth
low-pass should land on a clean citation graph. So the diagnosis is sharp and it is *not* a constraint
problem and *not* an expressivity problem: Bernstein is expressive enough and its constraint is helping.
The problem is **approximation efficiency** — at the fixed `K=10` the Bernstein basis resolves a smooth
target response too slowly, and on top of that its `O(K^2 m d)` cost means I cannot just raise `K` to
buy resolution. I want the same near-best approximation quality with *linear* cost and *faster*
per-degree convergence, and the basis that is supposed to deliver exactly that is Chebyshev. So the
obvious next move is to swap the Bernstein basis for the Chebyshev basis in the same decoupled harness.
But there is a puzzle waiting on that path, and if I do not resolve it I will make Citeseer *worse*, not
better.

Here is the puzzle. The truncated Chebyshev expansion of a function is *near-minimax*: among all
degree-`K` polynomials its worst-case uniform error is within a small factor of the best possible. By
approximation theory it is the *good* basis — the whole reason numerical analysts abandoned the monomial
powers a century ago is that the Chebyshev basis is far better conditioned. So a naive expectation is
that dropping the Chebyshev basis into the harness — learn `w_k` freely, propagate
`sum_k w_k T_k(L_hat) f_θ(X)` — should beat both Bernstein and the monomial GPR filter. Yet the
empirical record on these citation graphs is the opposite: a free-coefficient Chebyshev filter comes in
*last* of the three bases, below the monomial GPR filter and below Bernstein. The near-minimax basis is
the worst performer. Either approximation theory is lying to me, or I am using Chebyshev wrong — and I
will bet on the latter, because the theory is about the *best* polynomial in the basis, and I am letting
gradient descent pick the coefficients freely, which is a different thing.

So let me think about what "learn `w_k` freely" actually does. Any continuous response on `[-1,1]` has a
Chebyshev expansion `h(x) = sum_k w_k T_k(x)`, and the high-`k` `T_k` are the high-frequency modes:
`T_k(x) = cos(k arccos x)`, literally a cosine of increasing frequency. So the large-`k` coefficients
control how much high-frequency wiggle the response has. Now the fact I keep underusing: there is a
constraint on those coefficients that any *well-behaved* filter must obey. If `h` is analytic on the
open interval — locally a convergent power series, the smooth kind of response a sensible filter should
be — then its Chebyshev coefficients must *decay*, asymptotically like `1/k^q` for some positive `q`.
The high-frequency coefficients of a smooth filter are *forced* to die off. A response whose large-`k`
Chebyshev coefficients do *not* decay is, by the contrapositive, non-analytic — a wild, oscillatory,
hard-to-approximate function. And free `w_k`, fit by gradient descent chasing training accuracy, have
*nothing* enforcing that decay: the optimizer is free to pile weight onto the large-`k` coefficients and
fit a jagged, high-frequency response that memorizes the labels and generalizes terribly. That is a
textbook overfitting machine — enormous capacity in exactly the high-frequency modes — and it is why the
free-Chebyshev filter loses despite the basis being near-minimax. This is the same disease that made the
original ChebNet *lose to its own first-order GCN special case* and get worse as `K` rose: GCN
physically cannot express high-frequency garbage, so on a mostly-homophilic citation graph its
inflexibility is a feature, while a free-coefficient high-`K` Chebyshev filter has ten extra
high-frequency knobs to overfit with. The basis was never the problem; the *missing coefficient
constraint* is.

So the cure is to force the decay — to tie the coefficients to a well-behaved response instead of
letting them roam. The crudest way is a `1/k` reparameterization: learn `w_k` but propagate with
`w_k/k` per order, mechanically pushing the spectrum toward the `1/k^q` shape the theorem demands. And
that does work — it jumps a free-Chebyshev filter past the monomial and the Bernstein ones. But it is a
hack, and it has the exact same wall the monomial basis had against the *previous* rung: I cannot extend
it to impose the non-negativity constraint that BernNet's whole design was built around. With abstract
expansion coefficients there is no transparent relation between `w_k` and the *value* `h(λ̂)`, so "keep
the response non-negative" is a hard global condition again, not a box constraint. I want the lesson
BernNet taught me — *constraints on the response should be constraints on the parameters* — without
BernNet's quadratic cost and slow convergence. That sentence is the thing to chase.

What makes response constraints turn into parameter constraints? If my trainable numbers *were* the
filter's values themselves, then "the filter is non-negative at the control points" would just be "these
numbers are non-negative" — a ReLU, done. So instead of parameterizing the *coefficients* of the
polynomial, parameterize the polynomial by the *values* it takes at a fixed set of points, and let the
coefficients be *derived* from those values. That is not a basis change; it is switching from expansion
to **interpolation**. Given values `γ_j` at `K+1` distinct nodes `x_0 < ... < x_K`, there is a unique
degree-`≤K` polynomial through them. Now the free parameters *are* `γ_j = h(x_j)`, the response values,
and a constraint on the response at a control point is a box constraint on a parameter. When the sampled
values come from a smooth response, the recovered coefficient vector is the interpolation coefficient
vector of *that smooth response*, so the decay comes from the response itself rather than from a
heuristic `1/k` penalty — and `γ_j ≥ 0` is exactly "non-negative response value at node `j`," the
BernNet-style constraint, now living at the parameter level.

Except interpolation has its own famous trap, and if I walk into it I have traded one disease for a
worse one: the **Runge phenomenon**. Sample at *equispaced* nodes and crank `K` up on a Runge-type
target, and the interpolant oscillates harder and harder near the interval ends, the high-degree error
diverging. Let me locate exactly where that comes from so I know what to do about it. The interpolation
error at any point is `R_K(λ̂) = h^{(K+1)}(ζ)/(K+1)! · π_{K+1}(λ̂)`, where `π_{K+1}(λ̂) = prod_k(λ̂ -
x_k)` is the *nodal polynomial* — the monic degree-`K+1` polynomial whose roots are my chosen nodes. The
`h^{(K+1)}/(K+1)!` factor is the function's business, untouchable. But `π_{K+1}` is *entirely mine* — it
is decided by where I put the nodes. With equispaced nodes `π_{K+1}` is small in the middle and swings
enormously near `±1`, so the error explodes at the boundaries. So the whole Runge problem reduces to a
clean question I control: *where do I place `K+1` nodes so that `max |prod_k(x - x_k)|` over `[-1,1]` is
as small as possible?*

And I know the answer to that. Among all monic degree-`K+1` polynomials on `[-1,1]`, the one with the
smallest possible maximum absolute value is the scaled Chebyshev polynomial `2^{-K} T_{K+1}`, with
uniform norm exactly `2^{-K}`. So if I choose my nodes to be the *roots of `T_{K+1}`*, the nodal
polynomial *is* that minimizer, `||π_{K+1}|| = 2^{-K}` — the smallest it can be, shrinking geometrically
in `K`. Those roots are the **Chebyshev nodes** `x_j = cos((j + 1/2)π/(K+1))`, `j = 0..K`, which cluster
near the endpoints — denser exactly where equispaced nodes let the error run wild, which is the
geometric reason they tame it. And this also delivers the approximation quality, not just a small nodal
polynomial: the Lebesgue constant of Chebyshev-node interpolation grows like `log K` (versus `~2^K` for
equispaced), so the interpolant is always within a `~log K` factor of the minimax polynomial. That is
the near-minimax guarantee, and it confirms the basis was never the problem — it is interpolation at the
*right nodes* that is near-best, and free-coefficient fitting that was wrong.

Now the coefficients in closed form, because the structure is the payoff. Write `x_j = cos θ_j` with
`θ_j = (j+1/2)π/(K+1)`, so `T_k(x_j) = cos(k θ_j)`. The Chebyshev polynomials are *discretely
orthogonal* at their own nodes: `sum_{j=0}^K T_m(x_j) T_l(x_j)` is `0` for `m≠l`, `(K+1)/2` for
`m=l≠0`, and `K+1` for `m=l=0` — the half-integer offset in `θ_j` makes the cross sums of unit roots
cancel, with the `m=l=0` case being *twice* the others. Plugging the interpolation conditions `P_K(x_j) =
γ_j` into the expansion `P_K = (c_0/2)T_0 + sum_{k≥1} c_k T_k` and inverting via that orthogonality gives
one normalization for every coefficient: `c_k = (2/(K+1)) sum_{j=0}^K γ_j T_k(x_j)`, with the constant
term applied as `c_0/2` to compensate `T_0`'s doubled discrete norm. This is exactly a discrete cosine
transform of the sampled values — not a fudge, the inverse of the orthogonality relation. So the recipe
is: treat the response values at the Chebyshev nodes as the trainable `γ_j`, take this near-DCT to get
the Chebyshev coefficients of the unique near-minimax interpolant, and propagate with the standard
three-term recurrence.

Two implementation choices close it, and both are the *opposite* trade-off from BernNet. First the
rescaling: Chebyshev lives on `[-1,1]` and the normalized-Laplacian spectrum lives on `[0,2]`. ChebNet
needed `λ_max` to set `L_hat = 2L/λ_max - I` — an extra eigen-computation — but I know `λ_max ≤ 2` a
priori, and using `λ_max = 2` gives the free shift `L_hat = L - I`: build `L` once with `get_laplacian`,
then add self-loops with weight `-1`. No spectrum estimate. Second the propagation: the Chebyshev
three-term recurrence `T_k(L_hat)x = 2 L_hat T_{k-1}(L_hat)x - T_{k-2}(L_hat)x`, carrying two running
vectors through one sparse mat-vec each, accumulating `out += c_k T_k(L_hat)x` started at `(c_0/2)T_0 +
c_1 T_1`. Forming the `K+1` coefficients is `O(K^2)` scalar work on the fixed `T_k(x_j)` table; the
propagation is `K` sparse mat-vecs, `O(K m d)`. So the forward pass is `O(K^2 + K m d)` — **linear in
`K`**, where BernNet was quadratic — and the per-degree convergence is faster too: Chebyshev
interpolation error scales like `ω(K^{-1}) log K` against Bernstein's `ω(K^{-1/2})`, so at the *same*
fixed `K=10` it resolves a smooth target more sharply. That is precisely the deficit I diagnosed on
Citeseer: I am buying back the resolution Bernstein left on the table, at lower cost.

The initialization mirrors BernNet's logic in the new parameterization. The parameters are response
*values* `γ_j = h(x_j)`, so the neutral start is every `γ_j = 1`, which interpolates to the constant
`h ≡ 1` — the all-pass / identity response, no a-priori low- or high-pass bias, the same flat start that
let the Bernstein filter find heterophilic responses. The ReLU on `γ` before the DCT keeps the
non-negativity constraint at the interpolation nodes — the BernNet lesson preserved, though I should be
honest that this is a *sampled-value* constraint (non-negative at the `K+1` nodes), not BernNet's global
positivity certificate between nodes. I keep the harness's propagation dropout at the default and the
training overrides at the scaffold values. The literal edit replaces the two class bodies; the full
scaffold module is in the answer.

So the falsifiable expectations against the Bernstein numbers. The whole motivation was Bernstein's
softness on the *homophilic* citation graphs, so that is where I expect to gain: Citeseer should climb
clearly off 0.7795 toward 0.80, and Cora off 0.8554 toward 0.87 — the linear-cost, faster-converging
near-minimax interpolation buying the smooth-low-pass resolution Bernstein under-resolved at fixed `K`. I
expect the per-seed variance to *shrink* relative to Bernstein's, because the Chebyshev-node
parameterization ties the coefficients to a stable interpolation problem instead of a free or slowly
converging fit — and indeed the place to watch is whether Cora/Citeseer come in nearly seed-invariant.
The genuine risk is on the *heterophilic* side, and I want it on the record: BernNet's global
non-negativity certificate may be doing real regularizing work on the tiny 183-node Texas and Cornell
graphs, and my weaker sampled-value constraint plus the higher-resolution basis could *overfit* there
where Bernstein did not. So my honest prediction is an inversion of Bernstein's profile: I expect
ChebNetII to *win the homophilic graphs* (Cora and Citeseer above Bernstein) while possibly *losing
texas* (Bernstein's 0.9093 is a high bar that its certificate earned on a tiny graph) and landing
roughly even on Cornell. If that inversion shows — homophilic up, heterophilic flat-to-down — then
neither the certificate nor the near-minimax interpolation is uniformly best, and the next rung's
question is already framed: is there a parameterization that keeps the linear cost and recovers the
heterophilic edge a constrained basis gives up, without the quadratic price?
