The mean told me, in numbers, exactly which assumption to attack. The floor landed at latitude-weighted
RMSE 353.50 on z500-3day, 2.6032 on t850-5day, 3.3991 on wind10m-7day — a working forecaster on all
three targets, not a degenerate one, which is the reassuring part: the scale-preserving uniform average
plugs into the pretrained ClimaX backbone cleanly and produces real forecasts. But it is a floor, and the
shape of these numbers points straight at its one structural defect. The mean weights every variable
identically at every location and in every atmospheric state. With `V = 48` variables, three of which are
near-inert static fields (land-sea mask, orography, latitude) and many of which are pressure-level fields
of very uneven relevance to a given target, "count all 48 equally" is almost certainly leaving signal on
the table. The 500 hPa geopotential forecast at 353.50 is the loudest tell: z500 is a large-magnitude
field, and its 3-day evolution is governed by the synoptic dynamical variables — geopotential and wind at
the mid-troposphere — far more than by the land-sea mask or the surface humidity. The mean dilutes those
informative fields by averaging in dozens of comparably-weighted-but-less-relevant ones. So the diagnosis
is precise and it is *not* a scale problem (the mean already preserves scale) and *not* a capacity problem
(the backbone is untouched): it is that the aggregation has no way to say *this variable matters more than
that one*. That is the degree of freedom I'll buy back now, as cheaply as possible, before reaching for
anything content-dependent.

Let me question the equal-weight assumption directly, because it is a free decision the mean made for
free. When I write `(1/V) Σ_v x_v`, I am asserting every variable contributes equally to the fused token.
The variables are heterogeneous by construction — different physical fields, different units, different
relevance — so there is no reason on earth they deserve the same say. And I don't have to argue this in
the abstract; the mean's own numbers are the evidence. If equal weighting were already optimal, no learned
re-weighting could beat it, and I'd be wasting parameters. The bet I'm making is that it is *not* optimal —
that there is a fixed, global "this field matters more" structure the uniform average is blind to. The
most direct fix imaginable: attach a weight `w_v` to each variable and combine by a weighted sum,
`O = Σ_v w_v x_v`, and let `w_v` be a learnable parameter trained by ordinary backprop. The gradient flows
straight through — `dO/dw_v = x_v` — so a variable that consistently helps reduce the latitude-weighted
loss has its weight pushed up, one that hurts pushed down. The network discovers per-variable importance
instead of me legislating uniformity.

How expressive should the weight be? It could be a scalar — one number per variable — or a length-`D`
vector (one per channel), or a full per-location tensor. The per-channel and per-location versions are
strictly more expressive, but the quantity I'm trying to decide is "how much does variable `v` matter
*overall*," and that is naturally one number per variable. With `V = 48` a scalar weighting is 48
parameters total — utterly negligible next to the ViT backbone, basically free. So I start with the
scalar-per-variable form: the cheapest knob that expresses exactly the thing the mean's numbers told me to
add. This is deliberately the *minimal* step up from the floor — I am not yet making the weighting depend
on the token contents or the location (that is the cross-attention rung above), I am only letting the
global per-variable split move off uniform. If even this minimal step can't beat the mean, then the
per-variable contributions really are uniform and the whole ladder above is in question.

Now let me actually try to train `O = Σ_v w_v x_v` with `w_v` free real numbers and feel where it breaks,
because the mean's one virtue — scale preservation, the thing that let it plug into the pretrained
backbone at all — is exactly what free weights would destroy. The map from `(w_1, …, w_V)` to `O` is
linear and homogeneous of degree one in the weights: scale every `w_v` by `c` and `O` scales by `c`.
Nothing pins the overall magnitude of `w`. So during fine-tuning there is an entire ray of weight settings
— all `c·w` — producing outputs that differ only by a global gain, and the optimizer can drift along it
however it likes. That is bad in two concrete ways, and the first is the one the mean's success warned me
about. The pretrained ClimaX backbone expects features on the single-token scale; the mean delivered
exactly that, which is why 353.50/2.6032/3.3991 are sensible numbers and not garbage. If `Σ_v w_v` wanders
up to 10 or down to 0.1, I have silently multiplied the fused token by 10 or 0.1 and yanked the pretrained
stack off the operating point the mean carefully respected — re-creating, by the back door, the very
scale-leak I rejected the bare sum for. The second is that there is no ceiling at all: `w_v` is a free
scalar, it can grow without bound, and an unbounded multiplicative gain in the middle of a deep fine-tuned
network is exactly the kind of thing that makes training unstable. I'd have handed the model
expressiveness and a way to blow itself up in the same line. Wall.

The failure points at the cure. I never cared about the *absolute* size of the weights — I cared about
*relative* contribution: how much variable `v` matters compared to the others. That is a ratio, not a
magnitude. The absolute scale of `w` is a nuisance degree of freedom, and it is precisely the thing that
ran away. So I quotient it out: fix the overall scale and let the network move only the relative split.
The cleanest way to fix the scale is to demand the weights sum to one; and if I also keep them
nonnegative, then `Σ_v w_v x_v` with `w_v ≥ 0`, `Σ_v w_v = 1` is a *convex combination* of the variable
tokens — a weighted average. That is exactly the well-behaved object the mean was, generalized: it lives
in the convex hull of the `x_v`, so it is on the same single-token scale as any one variable (the property
that makes the pretrained backbone happy is *preserved*, not gambled), it cannot introduce an arbitrary
gain, and when all `w_v` are equal it collapses back to the mean. The mean I just measured is the uniform
point of this very family; I'm letting the network pick any other point of the simplex. Bounding the
weights onto `{w ≥ 0, Σ w = 1}` removes the runaway scale and keeps every bit of the relative-importance
expressiveness. That's the constraint I need, and it is the right constraint *because* it keeps the one
thing that made the floor work.

Now the question is purely mechanical: how do I parameterize a point on the simplex with free,
unconstrained parameters, smoothly and differentiably, so plain Adam trains it with no projection step? I
do not want to carry `w_v ≥ 0, Σ w_v = 1` as hard constraints during fine-tuning — that means projecting
onto the simplex after every step, an ugly special case in the otherwise-vanilla ClimaX training loop. I
want raw parameters `a_v` living anywhere in `R^V`, and a fixed function that maps them onto the simplex
with gradients flowing through. Nonnegativity from an arbitrary real: the smooth positive map is the
exponential, `e^{a_v} > 0` always. Sum-to-one: divide by the total. Put them together,
`w_v = e^{a_v} / Σ_j e^{a_j}` — softmax. Staring at it, it does exactly and only what I asked: every `w_v`
is strictly positive, they sum to one by construction, so the output is always a valid point on the
simplex — a distribution over the 48 variables, which is a satisfying way to read it: `w_v` is the model's
estimate of how much variable `v` should carry. The raw `a_v` are unconstrained, so I register them as
ordinary parameters and let Adam move them freely; softmax does the projection every forward pass. It is
differentiable everywhere, and the gradient on `a_v` couples all the variables (the denominator depends on
all of them), which is correct — pushing one variable's share up necessarily pushes the others' shares
down, since they compete for a budget that sums to one. The constraint isn't a penalty bolted on; it is
baked into the functional form, which is why it composes cleanly with the frozen-recipe training loop.

Let me sanity-check the degenerate case against the run I'm trying to beat, because it determines where
fine-tuning starts. If all `a_v` are equal — in particular if I initialize them all to zero — then
`w_v = e^0 / (V·e^0) = 1/V` for every `v`. Uniform. So at initialization the softmax-weighted sum *is* the
plain mean over the 48 variables: the exact aggregator that produced 353.50/2.6032/3.3991. That is the
ideal place to start a fine-tune-from-pretrained run. I begin from the measured floor — the safe
equal-contribution prior that already works with the pretrained backbone — and training perturbs the `a_v`
off uniform only insofar as the latitude-weighted loss rewards it. There is no cold-start risk where the
aggregator begins in a wild corner of the simplex and corrupts the pretrained features; it begins at the
centroid, which is the baseline I am improving on, and climbs from there. (Softmax is shift-invariant, so
any constant init — zeros or ones — gives the same uniform start; the task's edit zero-initializes the raw
weights, which is what I want.) This is a strictly monotone-by-construction improvement story: at step 0 I
am the mean, and any move the optimizer makes is one it found reduces the loss.

I should place this against the rung above so I'm honest about what I am *not* yet buying. The fully
expressive aggregation is cross-attention: project each variable into query/key/value and let a query at
each location decide, per location and per atmospheric state, how much to read from each variable. That
captures *content-dependent* and *location-dependent* mixing — strictly more than I'm doing, because my
`w_v` is a single distribution shared across every location and every example, whereas attention
recomputes the mixing at every token. But that expressiveness drags in the QKV projection matrices
(parameters scaling with `D = 1024`) plus an attention computation at every one of the 512 locations — a
real tax on top of an already-large fine-tuned backbone, to answer a question — "how much does each
variable contribute" — that the mean's failure suggests may be largely answerable with a *global*
per-variable split. My 48 scalars are the minimal answer to that question: one learned mixing distribution
for the whole tensor, no projections, no attention matrix. If the global split is most of the signal, I
get most of the benefit at essentially none of the cost. That is the bet of this rung, and it is the right
middle point to test between the parameter-free mean below and the cross-attention above.

So the delta from step 1 is exact and minimal: where the mean computed `x.mean(dim=1)`, I now hold one
learnable length-`V` vector of raw weights, softmax it over the variable axis into a simplex distribution,
broadcast it against `[B, V, L, D]` so each variable token is scaled by its scalar weight, and sum over
the variable axis to collapse `V → 1`, leaving `[B, L, D]`. (The full scaffold module is in the answer.) I
keep only the softmax form — not the ReLU fast-normalized sibling that latency-bound detectors use, and
not a per-channel weight — because the task's edit surface and its fine-tuning budget call for the clean,
smooth, exactly-sums-to-one default, zero-initialized so it starts as the measured mean.

Let me close on the falsifiable expectations against the numbers I have. I expect this to *beat the mean
on every target*, and most clearly on z500-3day: that 353.50 is the floor's largest headroom, and the
500 hPa geopotential forecast is exactly where a non-uniform split — upweighting mid-tropospheric
geopotential and wind, downweighting the static surface constants and surface humidity — should help most,
so I expect the z500 RMSE to drop the most in relative terms. On t850-5day (2.6032) and wind10m-7day
(3.3991) I expect a smaller but real improvement, since those targets also have their own characteristic
informative subsets the uniform mean smears. The bet is falsifiable and sharp: if the learned weighted sum
*fails* to beat 353.50 / 2.6032 / 3.3991 on each target, then the per-variable contributions genuinely are
uniform — the mean was already optimal among global weightings — and the next rung's content-dependent
machinery would have to justify itself purely on *location*- and *state*-dependent mixing rather than on
fixing a bad global split. And I already suspect where even this rung will run out: the weighting it learns
is *fixed* — once trained, variable `v` gets the same share at every grid cell and in every atmospheric
state, no matter what its token actually says there. The right weighting at a synoptically active cell
surely differs from a quiescent one. So I expect this to beat the mean and then itself be beaten, by an
aggregator whose weights depend on the token contents — which is the cross-attention rung directly above.
