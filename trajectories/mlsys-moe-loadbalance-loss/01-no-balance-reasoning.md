I want a Mixture-of-Experts layer to do what it promises: route each token to a couple of
specialized experts out of many, so that capacity grows with the number of experts while the
per-token compute stays fixed. The router is the small gate that, for each token, scores the
experts and sends the token to its top two. The first thing I have to decide is what trains the
router, and the honest answer at this starting point is: nothing but the language-model loss. There
is no separate signal telling the router to use all the experts; it sees only the cross-entropy at
the output, backpropagated through whichever experts the tokens happened to be sent to. Before I add
any corrective term I want to know, precisely and quantitatively, what the unregularized router does,
because the whole ladder that follows is a sequence of answers to one question — how bad is the
imbalance, and how much does each fix cut it? — that is unanswerable without a measured floor to
compare against. So this rung is deliberately diagnostic: I add no balancing loss, the editable slot
returns a literal zero, and I read off what the router does when the only thing pulling on it is
prediction.

Let me think about what that objective actually wants, because the whole problem is hiding in the
answer. Suppose, early in training, the router by chance sends a few more tokens to expert three than
to the others — the asymmetry is guaranteed at initialization, since the router weights are random.
Expert three now receives a little more gradient, so its FFN trains slightly faster, so its outputs
become more useful, so the cross-entropy improves a little more when tokens are routed to it, so the
router — trained to lower that very cross-entropy — learns to raise expert three's score and send it
even more tokens. That is a closed loop with positive sign: usage feeds quality feeds routing feeds
usage. It makes the uniform allocation an *unstable* fixed point and the concentrated allocations the
stable attractors — the same rich-get-richer mechanism a Polya urn has, where the equilibrium is not
the balanced middle but a lopsided corner chosen by early noise. So collapse is not a rare pathology
I would have to provoke; it is the default destination of an MoE trained on cross-entropy alone, and
the starved experts never specialize and become dead weight — precisely the capacity the MoE was
built to add. The geometry of that collapse is set by `K = 2`: the router cannot pour everything onto
one expert, because every token must pick two *distinct* experts, so the reachable corners are the
two- and few-expert faces of the simplex, where a small live set splits the traffic while the rest
sit near zero. That is the shape I am watching for — not one expert eating everything, but a handful
monopolizing while the tail starves.

I should be precise about why the cross-entropy cannot see this as a problem, because that tells me
where the cure has to come from. The loss is a sum over tokens of how well the model predicted each
next token. Fix a token: for its own contribution it does not care whether the expert it used is also
used by a million other tokens or by ten; it cares only that the expert it landed on produced a good
prediction. There is no term anywhere in that sum that ranges over *experts* and asks whether usage
is spread out. Balance is a property of the whole routing histogram, and the per-token loss is
structurally blind to properties of that histogram. So the router optimizes exactly what it is told
to — prediction — and balance is whatever falls out as a side effect, which by the argument above is
skew. Nothing opposes the skew from inside the objective.

There is a second half to the diagnosis that matters just as much: not only is cross-entropy blind to
collapse, it is barely *hurt* by it. With eight experts and top-two routing, even a badly collapsed
router is still a functioning model — each token is still processed by two experts of the full hidden
width, and if the live experts have absorbed the gradient the dead ones were denied, they carry most
of the predictive load. On an easy synthetic next-token task there is only so much distinct structure
to specialize on, so the marginal predictive value of the eighth expert over the second is small, and
losing it to starvation costs the cross-entropy only a second-order amount. That is the uncomfortable
truth the control is meant to expose: the model can be quite imbalanced and still predict quite well,
so the objective has no incentive to fix the imbalance. The disease is real but nearly painless to
the thing being optimized, which is exactly why an external term is needed.

It is worth stating what the *healthy* outcome even looks like, because every balancing loss above is
an attempt to manufacture it. A well-used MoE does not route uniformly token by token — that would
defeat specialization — it routes each token sharply to the two experts most suited to it, and yet
across the corpus every expert still carries roughly its fair share, because the variety of the data
keeps all the specializations in demand. So the target is a conjunction: sharp, content-dependent
routing locally, flat usage globally. The control's failure is that with no pressure the global
histogram is free to be as skewed as the local decisions, and the feedback loop guarantees it will
be. The cure has to inject exactly the missing half — a term that watches the *aggregate* histogram
and pushes it toward flat — while disturbing local per-token routing as little as it can, because
that local sharpness is the specialization I am protecting. That tension, between flattening the
histogram and preserving the sharpness, is the single axis every later rung moves along.

Now the second measurement, the one this ladder is built around. I define load imbalance as the L1
deviation of the token allocation from uniform: take `f_i`, the fraction of routed (token, slot)
assignments landing on expert `i`, and compute half the sum over experts of `|f_i − 1/N|`. Each token
contributes two assignments, but `f` divides by that same total count, so `Σ_i f_i = 1` and the
uniform reference `1/N` is still the right comparison point. At the balanced extreme every `f_i = 1/8`
and `L_imb = 0`. The context suggests a ceiling of `1 − 1/N = 0.875`, but that is the *one-expert*
collapse bound: `f = (1, 0, …)` gives `½(0.875 + 7·0.125) = 0.875`. Under top-two routing a single
expert can never hold more than half the slots, so the tightest collapse I can physically reach is
all tokens choosing the same pair, `f = (0.5, 0.5, 0, …)`, which gives `½(2·0.375 + 6·0.125) = 0.75`.
So the working ceiling here is `0.75`, not `0.875`, and that fixes the scale I read every later number
on. To calibrate the middle, a conspicuous two-fold overload of a pair — `f_1 = f_2 = 0.25`, the rest
splitting `0.5/6 ≈ 0.083` — works out to `½(2·0.125 + 6·0.042) = 0.25`. So even a clearly visible
two-fold overload sits only a third of the way to the reachable ceiling; an `L_imb` around a tenth
would mean a mild-to-moderate skew, and that is roughly the regime I expect an untended router to
relax into over fifteen hundred steps of a small model — pulled toward a corner, but not all the way
to the `0.75` wall.

One measurement subtlety sets the smallest value the metric can resolve. `L_imb` is computed from
finite batches, and a finite sample of routing decisions is noisy even under a perfectly uniform
router: with `M = 2T` hard assignments the counts fluctuate like a multinomial, each `f_i` with a
standard deviation of about `√(p(1 − p)/M)`, `p = 1/N`. Because `L_imb` sums *absolute* deviations
these do not cancel, so there is a strictly positive sampling floor that falls only as `1/√M`, and
averaging over twenty batches sharpens the estimate of the mean without removing the per-batch bias.
So "uniform" means "down at the sampling floor," not literally zero, and the control's `L_imb` is only
evidence of collapse if it sits *clearly above* that floor — which, given everything above, I expect
it comfortably to. I train a single seed, so the particular experts that end up hot are an accident
of that seed; but the *magnitude* of the imbalance is a property of the mechanism, since whichever
experts win the early lottery the same feedback loop concentrates onto a comparably small live set.
That stability of magnitude is what lets a single-seed number serve as the fixed floor.

I should be explicit that this `L_imb` is a *measurement* and not a candidate training signal, because
that distinction seeds the next rung. It is built entirely from the hard count fractions `f_i`, which
come from the top-K `argmax` and carry no useful gradient: nudge a router weight and, for a while,
every token keeps its same top-two experts, so `f` is flat with gradient exactly zero; then one
token's third-place expert overtakes its second and a count jumps by one. A function that is
piecewise-constant with jumps is the worst thing to hand a gradient-based optimizer. So the cure
cannot *be* the imbalance; it has to be a smooth surrogate living on the differentiable side of the
router — its continuous probability mass — that only borrows the counts to point in the right
direction. That is a problem for the next rung.

The two numbers combine into the fitness the later rungs are judged on, `r = −(L_CE + L_imb)`, and its
geometry governs how I read the ladder. Cross-entropy on this task is a few nats — perplexity in the
tens — while `L_imb` lives in `[0, 0.75]`, so in absolute terms `r` is dominated by cross-entropy. But
I have just argued cross-entropy barely responds to imbalance: collapse does not move it much, and a
well-designed penalty should not either. If `L_CE` sits nearly frozen across rungs, the term that
actually *discriminates* between them is `L_imb`. Because `r` adds the two directly, the exchange rate
is one to one: halving an imbalance of `0.12` to `0.06` buys `+0.06` in `r`, and it would be handed
straight back the moment cross-entropy rose by `0.06` of a nat — a small but entirely ordinary amount
to lose by over-constraining the router. That is the yardstick for every later rung: not lowest
imbalance alone, but lowest imbalance at unmoved cross-entropy. It is also the quantitative reason the
control matters — without its cross-entropy pinned down as the undisturbed reference, I could not tell
whether a later balancing gain came free or was quietly financed by a CE regression of the same size.

So this rung commits to none of the cure. It trains on cross-entropy alone and measures exactly two
things: the held-out cross-entropy (read also as perplexity) and the load imbalance. What I expect,
and treat as the falsifiable content, is a tolerable cross-entropy — a branching factor in the tens,
the signature of a competent model — sitting on top of an imbalance clearly above zero, in the
moderate band well short of the `0.75` ceiling. The load-bearing prediction is the *joint* shape: an
acceptable `L_CE` that barely notices the collapse, next to a bad `L_imb` that does. If the control
came back near-uniform on its own the whole ladder would be pointless; if the CE were wrecked by
collapse, the story that the objective is indifferent to balance would be wrong. I expect the gap, and
that gap is the entire opening: the disease is real and intrinsic to unregularized routing, and any
cure must come from *outside* the cross-entropy. The same tiny model, data, and optimizer are reused
unchanged for every rung, so the loss is the only variable that moves. The first attempt to beat this
floor — a penalty that pairs the count I cannot differentiate with the probability I can — is next.
