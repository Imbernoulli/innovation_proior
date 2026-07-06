I want a Mixture-of-Experts layer to do what it promises: route each token to a couple of
specialized experts out of many, so that capacity grows with the number of experts while the
per-token compute stays fixed. The router is the small gate that, for each token, scores the
experts and sends the token to its top two. The first thing I have to decide is what trains the
router, and the honest answer at this starting point is: nothing but the language-model loss. There
is no separate signal telling the router to use all the experts; it sees only the cross-entropy at
the output, backpropagated through whichever experts the tokens happened to be sent to. Before I add
any corrective term I want to know, precisely and quantitatively, what the unregularized router does,
because the entire ladder that follows is a sequence of answers to a question — how bad is the
imbalance, and how much does each fix cut it? — that is unanswerable without a measured floor to
compare against. So this rung is deliberately diagnostic. I add no balancing loss at all, the
editable slot returns a literal zero, and I read off what the router does when the only thing pulling
on it is prediction.

Let me think carefully about what that objective actually wants, because the whole problem of this
ladder is hiding in the answer. Suppose, early in training, the router by chance sends a few more
tokens to expert three than to the others — the kind of asymmetry that is guaranteed to exist at
initialization, since the router weights are random and no two experts start with exactly equal
pull. Expert three now receives a little more gradient than its neighbors, so its FFN trains slightly
faster, so its outputs become slightly more useful, so the cross-entropy improves a little more when
tokens are routed to it, so the router — which is trained to lower that very cross-entropy — learns
to raise the score it gives expert three and send it even more tokens. That is a closed loop, and its
sign is positive: usage feeds quality feeds routing feeds usage. It is worth being concrete about the
shape of this dynamic. If I track the usage fraction of the leading expert as a discrete map from one
optimizer step to the next, the loop makes the map's slope at the symmetric point greater than one —
a small lead is amplified rather than damped — which is exactly the condition for the uniform
allocation to be an *unstable* fixed point and the concentrated allocations at the corners of the
simplex to be the stable attractors. This is the same rich-get-richer mechanism that a Polya urn has,
where drawing a color makes that color likelier next time; the equilibrium is not the balanced middle
but a lopsided corner chosen by early noise. So I should not think of collapse as a rare pathology I
would have to provoke. It is the default destination of an MoE trained on cross-entropy alone, and
the starved experts — the ones early noise happened not to favor — receive almost no gradient, never
specialize, and become dead weight: parameters that cost memory and buy nothing, which is precisely
the capacity the MoE was built to add.

It helps to make that loop concrete with a back-of-the-envelope model, even a crude one, because the
crudeness still fixes the qualitative behavior. Suppose I summarize the router by a single number,
the usage share `x` of the current leading expert, and imagine that the extra quality an expert
accrues grows with the extra traffic it has already seen, so the router's tendency to send it still
more grows with `x`. A minimal caricature of one training step is then `x ← x + η·x·(1 − x)`: the
leader gains at a rate proportional both to its present lead `x` and to the head-room `1 − x` it has
yet to capture. Started a whisker above the symmetric share `1/N = 0.125`, this map creeps away from
balance, accelerates through the middle, and saturates near the top — the textbook logistic push-off
from an unstable equilibrium. I do not believe the true dynamics are this tidy, and the `η` here
stands in for a tangle of optimizer and data effects I have not measured; the only load the sketch
carries is that the *sign* of the feedback, not its exact form, already makes the symmetric
allocation a repeller. And the geometry it repels toward is set by `K = 2`: the router cannot pour
everything onto one expert, so the caricature's saturation is physically truncated at `x = 0.5`, and
the corners actually reachable are the two- and few-expert faces of the simplex, where a small live
set splits the traffic while the rest sit near zero. That is the specific shape of collapse I am
watching for — not one expert eating everything, but a handful monopolizing while the tail starves.

I should be precise about why the cross-entropy cannot see this as a problem, because the precision is
what tells me where the cure has to come from. The loss is a sum over tokens of how well the model
predicted each next token. Fix a single token: for its own contribution to the loss it does not care
whether the expert it used is also used by a million other tokens or by ten; it cares only that the
expert it landed on produced a good prediction for it. There is no term anywhere in that sum that
ranges over *experts* and asks whether the usage is spread out. Balance is a property of the whole
routing distribution — a statement about the histogram of assignments across experts — and the
per-token loss is structurally blind to properties of that histogram. So the router will optimize
exactly what it is told to optimize, which is prediction, and balance will be whatever falls out as a
side effect, which by the argument above is skew. Nothing opposes the skew from inside the objective.

There is a second half to the diagnosis that matters just as much: not only is cross-entropy blind to
collapse, it is barely *hurt* by it, and I should reason out why so that I know what to expect from
the CE number. With eight experts and top-two routing, even a badly collapsed router is still a
functioning model — each token is still processed by two experts of the full hidden width, and if the
live experts have absorbed the gradient the dead ones were denied, they can carry most of the
predictive load themselves. On an easy synthetic next-token task there is only so much distinct
structure to specialize on, so the marginal predictive value of the eighth expert over the second is
small, and losing it to starvation costs the cross-entropy only a second-order amount. That is the
uncomfortable truth the control is meant to expose: the model can be quite imbalanced and still
predict quite well, so the objective has no incentive to fix the imbalance. The disease is real but
nearly painless to the thing being optimized, which is exactly why an external term is needed.

It is worth stating what the *healthy* outcome would even look like, because every balancing loss on
the rungs above is an attempt to manufacture it, and if I mis-state the target I will mistake the
cure for the disease. A well-used MoE does not route uniformly token by token — that would defeat
the whole point of specialization — it routes each token sharply to the two experts most suited to
it, and yet across the corpus every expert still carries roughly its fair share, because the
variety of the data keeps all the specializations in demand at once. So the target is a conjunction,
not a single condition: sharp, content-dependent routing locally, flat usage globally. The control's
failure is that with no pressure the global histogram is free to be as skewed as the local decisions,
and the feedback loop guarantees it will be. The cure therefore has to inject exactly the missing
half of the conjunction — a term that watches the *aggregate* histogram and pushes it toward flat —
while disturbing the local per-token routing as little as it can, because that local sharpness is the
specialization I am trying to protect. That tension, between flattening the histogram and preserving
the sharpness, is the single axis every later rung moves along, and it is already visible here as the
reason the crude fixes will be the ones that flatten too much and pay for balance in cross-entropy.

Now I have to pin down the second measurement, the one this whole ladder is built around: the load
imbalance. I define it as the L1 deviation of the token allocation from uniform — take `f_i`, the
fraction of all the routed (token, slot) assignments that landed on expert `i`, and compute half the
sum over experts of the absolute gap between `f_i` and the uniform share `1/N`. Before trusting it I
want to check its range by hand, because a metric I cannot bound is a metric I cannot read. First, is
`f` even a distribution here? Each token contributes two assignments under top-two routing, so the
counts sum to twice the token total, but `f` divides by that same total count, so `Σ_i f_i = 1`
regardless — the top-K multiplicity cancels and the uniform reference `1/N` is still the right
comparison point. Good. At the balanced extreme every `f_i = 1/8` and the sum of gaps is zero, so
`L_imb = 0`. At the opposite extreme the context suggests the number approaches `1 − 1/N = 0.875`,
and I want to see where that comes from and whether it is actually reachable here. Putting all mass on
a single expert, `f = (1, 0, …, 0)`, gives `½(|1 − 0.125| + 7·0.125) = ½(0.875 + 0.875) = 0.875` —
so `0.875` is the *one-expert* collapse bound. But under top-two routing a single expert can never
hold more than half of the slots, because every token must pick two *distinct* experts; the tightest
collapse I can physically reach is all tokens choosing the same pair, `f = (0.5, 0.5, 0, …, 0)`, which
gives `½(2·0.375 + 6·0.125) = ½(0.75 + 0.75) = 0.75`. So the working ceiling for this configuration is
`0.75`, not `0.875`; the extra quarter is unreachable at `K = 2`. That distinction is not pedantry —
it fixes the scale on which I read every later number. To calibrate the middle of that scale, take a
conspicuous but not catastrophic skew: two experts running at double their fair share, `f_1 = f_2 =
0.25`, the remaining six splitting what is left, `0.5/6 ≈ 0.083` each. That works out to `½(2·0.125 +
6·0.042) = ½(0.25 + 0.25) = 0.25`. So even a clearly visible two-fold overload of a pair of experts
sits only a third of the way to the reachable ceiling. A `L_imb` around a tenth, then, would mean a
mild-to-moderate skew — enough to see, far from collapse — and that is roughly the regime I expect an
untended router to relax into: pulled toward a corner by the positive-feedback loop, but only for as
long and as hard as fifteen hundred steps of a small model allow, not all the way to the `0.75` wall.

There is a measurement subtlety I want to settle before I trust any imbalance number, this control's
included, because it sets the smallest value the metric can even resolve. `L_imb` is computed from
finite batches — the protocol averages it over twenty fresh held-out batches — and a finite sample
of routing decisions is noisy even when the underlying router is perfectly uniform. If the true
per-expert probability were exactly `1/N`, a batch of `T` tokens still yields only `M = 2T` hard
assignments spread over the experts, and those counts fluctuate like a multinomial: each `f_i` has a
standard deviation of about `√(p(1 − p)/M)` with `p = 1/N`. Because `L_imb` sums *absolute*
deviations, these fluctuations do not cancel the way a signed sum would; treating each `f_i` as
roughly Gaussian about `1/N`, the expected absolute gap is `σ·√(2/π)`, so the noise inflates `L_imb`
by about `(N/2)·√(2/π)·√(p(1 − p)/M)` — a strictly positive floor that falls only as `1/√M`.
Averaging over twenty batches sharpens the estimate of the *mean* `L_imb` but does not remove this
bias, because each individual batch is biased upward. The consequence for the whole ladder is that
"uniform" means "down at the sampling floor," not literally zero, and it would be a mistake to chase
an imbalance below what the batch size can resolve. It also means the control's `L_imb`, whatever it
comes in at, is only worth taking seriously as evidence of collapse if it sits *clearly above* that
floor — which, given everything above, I expect it comfortably to.

One last note on what the number means across runs. I train a single seed, `1234`, so the particular
experts that end up hot are an accident of that seed's initialization and early data order; a
different seed would anoint a different subset of experts as the winners. But the *magnitude* of the
imbalance is a property of the mechanism rather than the seed — whichever experts win the early
lottery, the same positive-feedback loop concentrates onto a comparably small live set and drives
`L_imb` into a comparable band. So I read the control's imbalance as a magnitude that would reproduce
across seeds even though the identity of the collapsed-onto experts would not, and that stability of
magnitude is exactly what lets a single-seed number serve as the fixed floor the ladder measures
against.

I should be explicit that this `L_imb` is a *measurement* and not a candidate training signal, because
the distinction is the seed of the next rung. It is built entirely from the hard count fractions
`f_i`, which come from the top-K `argmax` and carry no useful gradient — nudging the router weights a
hair does not change which experts are top-two until some token crosses a decision boundary, at which
point the count jumps discontinuously. So nothing assembled out of `f` alone can serve as the penalty
that trains the router; whatever cure I eventually add will have to reach the router through some
differentiable quantity and merely *correlate* with this count-based imbalance. To see how sharp that
obstacle is, picture the loss surface as I nudge a single router weight: for a while nothing at all
happens to the counts, because every token keeps the same top-two experts it already had, so `f` — and
any penalty built from `f` — is flat, gradient exactly zero. Then, at some threshold, one token's
third-place expert overtakes its second, the assignment flips, and a whole count jumps by one in a
step. A function that is piecewise constant with jumps is the worst possible thing to hand a
gradient-based optimizer: it is either telling the router nothing or telling it something unusable. So
the cure cannot *be* the imbalance; it has to be a smooth surrogate that lives on the differentiable
side of the router — its continuous probability mass — and only borrows the counts to point in the
right direction. That is a problem for the rung after this one. Here I only need `L_imb` as a clean,
bounded, model-agnostic readout of exactly the thing the cross-entropy cannot see.

The two numbers are combined into the single fitness the later rungs are judged on, `r = −(L_CE +
L_imb)`, and I want to understand its geometry now, because it governs how I read the whole ladder.
The cross-entropy of a language model on this task is a quantity of order a few nats — a perplexity
`exp(L_CE)` in the tens — while `L_imb` lives in the narrow interval `[0, 0.75]`. So in absolute terms
`r` is dominated by the cross-entropy. But I have just argued that the cross-entropy barely responds
to imbalance: collapse does not move it much, and a well-behaved balancing penalty, if I design one
that does not distort the router's predictions, should not move it much either. If `L_CE` sits nearly
frozen across every rung, then the term that actually *discriminates* between rungs in `r` is `L_imb`.
That reframes the entire exercise cleanly: the ladder is a contest to drive `L_imb` toward zero
subject to the hard side-constraint that `L_CE` not degrade. A fix that halves the imbalance while
holding CE flat is a genuine win of about `+0.06` in `r`; a fix that drives imbalance lower but pays
for it with even a modest rise in CE — the hollow victory of crushing the router into a dense model of
two experts' width — is designed by this fitness to lose, because a CE increase of a tenth of a nat
would swamp any plausible imbalance gain. That is the yardstick I will hold every later rung to: not
lowest imbalance alone, but lowest imbalance at unmoved cross-entropy.

I can put numbers on that sensitivity to keep myself honest later. Because `r` adds the two terms
directly, the exchange rate between them is one to one: an imbalance improvement of `Δ` is worth
exactly the same in `r` as a cross-entropy improvement of `Δ` nats. So if some later rung halves an
imbalance of, say, `0.12` down to `0.06`, that buys `+0.06` in `r` — and it would be handed straight
back the moment the cross-entropy rose by a mere `0.06` of a nat, which on a language model is a small
but entirely ordinary amount to lose by over-constraining the router. That is a sobering exchange
rate: it means the imbalance gains I can realistically hope for, on the order of hundredths, are
fragile against cross-entropy losses I could incur almost by accident. It is also the quantitative
reason the control matters so much. If I did not have its cross-entropy pinned down as the
undisturbed reference, I could not tell whether a later rung's balancing gain came for free or was
quietly financed by a CE regression of the same size, and the whole fitness comparison would be
uninterpretable.

The perplexity reading is the same information in a more intuitive unit, and I keep it precisely
because cross-entropy nats are hard to feel. Perplexity is `exp(L_CE)`, the effective branching
factor — the number of equally likely next tokens the model behaves as if it were choosing among. A
model that has learned real structure in the synthetic latent-topic task sits at a branching factor
well below the raw vocabulary size; a model that had learned nothing would sit near the vocabulary
size itself. So when I say I expect the control's cross-entropy to be "tolerable," I mean concretely
that its perplexity should land in the tens, clearly below the uninformed ceiling, which is the
signature of a model that is predicting competently — and, crucially, is doing so *while* its routing
histogram is skewed. That co-occurrence, competent prediction over a lopsided allocation, is the
whole diagnosis in one sentence, and reading it in perplexity rather than nats makes it harder to talk
myself out of: the model is genuinely working, and it is genuinely imbalanced, and the objective has
no complaint about the second fact.

So this rung commits to none of the cure. It adds no term, trains on cross-entropy alone, and I
measure exactly two things: the held-out cross-entropy, which I also read as perplexity, and the load
imbalance defined above. What I expect, and will treat as the falsifiable content of the control, is a
tolerable cross-entropy — collapse is not a catastrophe for the loss, by the second-order argument
above — sitting on top of an imbalance clearly above zero, in the moderate band well short of the
`0.75` ceiling, with the token allocation visibly skewed away from uniform. The load-bearing
prediction is the *joint* shape: an acceptable `L_CE` that barely notices the collapse, next to a bad
`L_imb` that does. If instead the control came back near-uniform on its own, the whole ladder would be
pointless and I would have to rethink whether unregularized routing really collapses at this scale;
and if the CE were wrecked by the collapse, the story that the objective is indifferent to balance
would be wrong. I expect neither. I expect the gap — a fine cross-entropy over a skewed histogram —
and that gap is the entire opening: it says the disease is real and intrinsic to unregularized
routing, and that any cure must come from *outside* the cross-entropy, as a term added by hand that
explicitly inspects the routing distribution and pushes it toward uniform. To keep the floor this
control establishes meaningful, the same tiny model, the same synthetic latent-topic data, and the
same optimizer are reused unchanged for every rung above, so that the loss is the only variable that
moves. This rung fixes the number every later loss must beat; the first attempt to beat it — a
penalty that pairs the count I cannot differentiate with the probability I can — is the next rung.
