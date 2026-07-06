The Switch aux loss did what I designed it to do, and the numbers let me read the outcome precisely.
Imbalance fell from the control's `0.1286` to `0.0587`, a `2.19×` cut, while the cross-entropy went
from `3.7280` to `3.7281` — a change of `+0.0001`, which on a model at perplexity `41.6` is nothing,
inside any plausible run-to-run noise. So the fitness climbed from `−3.8566` to `−3.7868`, a gain of
`+0.0698`, and when I decompose it the story is clean: the imbalance term contributed `+0.0699` and
the cross-entropy term `−0.0001`, so essentially the entire fitness gain is a balance gain bought at
no cost in prediction. In perplexity the same flatness reads as `41.594 → 41.599`, a drift of five
thousandths of a unit of branching factor — the model is, to any reading I can trust, predicting
identically well before and after the penalty, only more evenly. The `f·P` surrogate steered the
router toward uniform exactly as the gradient analysis said it would. That is the good news, and it
is worth stating plainly before I complain.

Now the complaint, which I have to be honest about because the data did not hand it to me on a plate.
The previous rung predicted that the micro-batch scope would tax the cross-entropy by flattening
per-slice specialization, and I set up the decomposition precisely to catch that tax. It did not
show: the cross-entropy was flat to four decimals. So the empirical case that micro-batch scope is
hurting me is, at this scale, unmade — the tax is below the resolution of this run, just as I flagged
it might be, because four quarters of a small batch on a synthetic task are still fairly
representative of one another and of the whole. I could let that stop me, but I think that would be
the wrong read. The principled objection to micro-batch scope survives untouched: the penalty
demands uniformity of every slice, which is strictly stronger than the whole-batch uniformity the
metric rewards and strictly at odds with the per-slice specialization I want to keep, and the only
reason it is not costing me visibly here is that my slices are too alike to expose it. On any corpus
with real topical heterogeneity — the setting the loss is actually meant for — that gap opens up. So
I want to make the change that is principled even where it is not yet empirically forced, and I want
to make it as the smallest possible move, turning exactly one knob so that if anything shifts I know
what shifted it. The knob is the set of tokens over which the frequency `f_i` is computed.

Let me reason about why that single knob is the whole story. The property I actually care about is
corpus-level: across all the data the model trains on, every expert should pull roughly its fair
share so that no expert dies and the capacity is used. I emphatically do *not* care that one
micro-batch of two dozen sequences is internally uniform — in fact I want the opposite, because if a
micro-batch happens to be all one topic the right behavior is for that topic's experts to light up
and the rest to stay quiet on that slice. Specialization *is* per-slice skew that averages out to
balance over the corpus. The micro-batch penalty conflates the two: it sees a topic-slice's benign
skew and cannot distinguish it from the pathological skew of collapse, so it penalizes both. It is
enforcing the right constraint at the wrong scope. So the fix is to measure `f_i` over the global
batch — the union of all the micro-batches, synchronized — instead of each micro-batch alone, and to
keep everything else about the penalty exactly as it was: `α · N · Σ_i f_i P_i`, the same coefficient
`α = 10^{-2}`, the same multiply-by-`N` that holds the balanced optimum scale-free at `1`, the same
detached counts and differentiable probabilities, the same mean over the two layers. Only the scope
of `f` moves. Now the penalty asks a different question — across all this data, is the usage uniform?
— and it is silent about whether any individual slice is uniform. A micro-batch is free to be as
specialized as its content demands, so long as the specializations of different slices cover all the
experts when summed. The router keeps its learned structure and still satisfies the balance
constraint, which is exactly the freedom the micro-batch version denied it.

I can make the scope difference exact with an idealized example, because the idealization is the
cleanest possible statement of what I am buying. Imagine the batch splits into four equal slices,
each perfectly specialized: slice one routes every token to experts `{1,2}`, slice two to `{3,4}`,
slice three to `{5,6}`, slice four to `{7,8}`, with the router's probability on each slice peaked on
that slice's pair. This is the *ideal* MoE — every token sharply routed, every expert fully used
across the corpus — and its whole-batch histogram is exactly uniform, because each expert appears in
one slice at half-mass weighted a quarter, so mean usage is `1/8` across the board and the global
`L_imb` is `0`. Now score it two ways. The global penalty pools the tokens, sees uniform `f`, and
because the four per-slice peaks average to a uniform mean `P` as well, reads `N Σ_i f_i P_i = 1` —
the scale-free floor, nothing to correct. The micro-batch penalty scores each slice on its own: on
slice one, `f = P = (0.5, 0.5, 0, …)` gives `N Σ_i f_i P_i = 8·0.5 = 4`, and averaged over the four
slices the penalty reads `4`, four times the floor, with a gradient on every slice actively trying to
flatten that slice's peaked `P` back toward uniform. So on the identical, ideal router the
micro-batch loss reports a `4×` violation and pushes to destroy the specialization, while the global
loss reports no problem at all. That factor of four is the over-constraint made numerical: it is not
that the micro-batch loss is wrong about collapse, it is that it cannot tell this perfect router
apart from a collapsed one, and the global scope can. At this small scale my real slices are nowhere
near this idealized separation — which is exactly why the tax stayed invisible last rung — but the
mechanism I am correcting is precisely this one, sized down.

Widening the scope of `f` buys a second thing beyond preserving specialization, and it is worth
naming because it is independent of the topical argument: it denoises the weight. The counts are a
finite sample, so `f` is a multinomial estimate whose per-expert standard deviation scales as `1/√M`
in the number of assignments `M`. A micro-split holds a quarter of the batch's tokens, so its `f` is
roughly twice as noisy as the whole batch's. And `f` enters the penalty as the detached weight that
decides *where* the corrective pressure points, so a noisier `f` means the direction of the gradient
jitters more from one micro-split to the next — the penalty spends part of its pull chasing sampling
artifacts, correcting whichever expert happened to look hot in that small slice. Measuring `f` over
the global batch halves that noise and hands the gradient a steadier, better-aimed direction. So the
single knob does two things at once: it stops the penalty from fighting legitimate per-slice skew,
and it lowers the variance of the very count weights the penalty is built on. Both effects point the
same way, toward the global scope, which makes me more confident the change is right in principle even
where the small scale keeps its effect on the headline numbers modest.

I should give the micro-batch scope its due before I abandon it, because it did not become the
default by accident. In a real distributed MoE the experts live on different devices and each device
gives its experts a finite capacity — a cap on how many tokens one expert may process in a step — and
tokens over that cap are dropped. Per-device load balancing is, among other things, a guard against
that overflow: it keeps any one device's experts from being swamped and its tokens discarded, which
is a genuinely per-device property that a purely corpus-level constraint does not enforce. So the
micro-batch scope buys an operational safety the global scope gives up, and that is a real reason it
exists. But that justification simply does not apply to this reproduction. The contract here forbids
dropping tokens and imposes no capacity limit; there is one device and the batch is processed whole.
The only balance I am accountable for is the corpus-level histogram the metric measures, and against
that goal the per-device constraint is pure over-constraint with no compensating benefit. Recognizing
that is what lets me drop the micro-batch scope with a clear conscience: I am not discarding a
safeguard I need, I am discarding a safeguard for a hazard this setup does not have.

Before I settle on "the exact global batch" I want to consider the one real alternative to it, which
is to approximate the global count with a running average — an exponential moving average of recent
batches' `f`, a momentum on the counts — rather than the true batch-wide count. That is tempting in
the real distributed setting because it avoids a synchronization: each device could keep its own
decaying history instead of all-reducing the counts every step. But it has a defect I can name
without running it. The moving average mixes *stale* counts, gathered under an older router, with the
*current* probability `P` that the gradient flows through. The whole construction relies on `f`
pointing at which experts are hot *now* so that the pressure on `P` is aimed correctly; feeding it a
lagged `f` aims yesterday's correction at today's router, and near the balanced fixed point that lag
can turn the corrective step into an oscillation. In this single-process reproduction I have no
synchronization to avoid — there is one device and the training batch already *is* the global batch —
so the clean choice is simply to compute `f` on the full batch and skip the approximation entirely.
Concretely I realize the contrast by computing the previous rung's penalty on four micro-splits and
this rung's on the full batch: "uniformity demanded of each quarter" versus "uniformity demanded of
the whole." It is a faithful small-scale stand-in for the real per-device-versus-global distinction,
even if the absolute gap is muted here where the quarters are already representative — which, given
that the tax did not surface last rung, means I should predict the imbalance to land in the same good
band as the Switch loss and the cross-entropy to be no worse, rather than promise a dramatic
separation the scale cannot deliver. If global scope helped a lot here I would actually be suspicious
that something other than scope had changed.

There is a complementary lever I want to run alongside this rung, because it sits naturally beside a
balancing loss even though it is not a loss at all. It maintains a per-expert bias `b_i` used *only*
to break ties in the top-K selection — added to the routing scores for ranking, but excluded from the
gate weights that actually combine the experts — and nudges it once per step by a small constant in
the direction that cools overloaded experts and warms underloaded ones: with `c_i = N f_i` the
normalized load (so its mean is exactly `1` at any allocation, since `Σ_i f_i = 1`) and `c̄` that
mean, the update is `b_i ← b_i + u · sign(c̄ − c_i)` with `u = 10^{-3}`. The sign is right — for an
over-used expert `c_i > 1` so `c̄ − c_i < 0` and its bias falls, cooling it; for an under-used expert
the bias rises, warming it. Two design choices in that rule are worth understanding rather than
copying. First, it acts on selection only, not on the gate weights, and that is what keeps it
prediction-neutral: it changes *which* experts a token is routed to but then combines them with their
true, unbiased probabilities, so it does not corrupt the mixture the way biasing the combine weights
would, and therefore it should not cost cross-entropy. Second, it is bang-bang — a fixed step in the
direction of `sign`, not a step proportional to the imbalance — which decouples the controller's
speed from the size of the error and makes it robust, at the price of only ever moving `b` at rate
`u`. I can bound its authority: over `1200` steps a consistently one-signed bias drifts by at most
`u · 1200 = 1.2` in logit units, and since the router's raw logits come from a linear map on a
`64`-dimensional hidden and sit at a spread of order one, a bias of order one is enough to promote a
persistently cold expert into a token's top two. So the controller has real authority over the hard
selection, and because it balances the *counts directly* rather than through a smooth surrogate I
expect it, when added, to drive imbalance lower than the loss alone — more aggressively than any
gradient on `P` will. I will run the global-batch loss both alone and with this bias to see how much
the count-level controller buys on top.

The selection-only restriction deserves a closer look, because it is what makes the bias almost free
on cross-entropy, and the mechanism is specific. For each token the router picks its top-two experts
and then combines their outputs with weights equal to the softmax probabilities of the chosen
experts, renormalized over the two. There are two places I could inject `b`: into those combine
weights, or into the ranking scores that decide which two experts are chosen. If I biased the combine
weights I would be mixing the expert outputs with weights that are no longer the model's true
probabilities — directly corrupting the mixture the network learned, a first-order hit to the
prediction that would surface immediately in cross-entropy. Restricting `b` to the ranking leaves the
combine weights untouched: they stay the honest softmax over whichever two experts were chosen. So
the only way the bias can move the output at all is by occasionally swapping which expert sits in the
top two, and it swaps only experts that were already close in score — the promoted cold expert was a
near-miss for that token, so displacing the expert it edges out costs the prediction little. That is
how a controller with the authority to move counts hard can still leave cross-entropy essentially
where it was: it reroutes without ever re-weighting, and it reroutes only at the margin where two
experts were nearly tied to begin with.

It is worth thinking through where that bias controller comes to rest, because a bang-bang integral
controller does not converge to a point — it limit-cycles. Once the counts are balanced,
`sign(c̄ − c_i)` keeps flipping as the finite-batch fluctuations push each `c_i` just above or just
below the mean, so each `b_i` dithers by about `±u` from step to step around whatever offset it needed
to equalize the load. That residual dither has amplitude `u = 10^{-3}` in logit space, negligible
against a logit spread of order one, so it does not itself reintroduce imbalance; the controller
settles into balancing the counts to within the batch's own sampling noise — which, notably, is close
to the resolution floor I argued `L_imb` has anyway, so a count controller is the kind of mechanism
that can in principle press imbalance right down to that floor. And it cooperates with the loss rather
than fighting it: both want the under-used experts warmed, the loss by raising their `P` through a
gradient and the bias by raising their selection score through an integral of past load. The two act
on different surfaces — smooth probability versus hard ranking — toward the same allocation, so I
expect them to compose rather than interfere, which is why running the loss with the bias is a
sensible stack and not a contradiction.

One detail of the bias I should get right for the same reason the loss is averaged per layer rather
than pooled: the bias is maintained per layer too. The two MoE layers have independent routers, and
there is no reason the first layer's dead experts are the second's — each has its own histogram and
its own tail. So the controller keeps a separate bias vector for each layer, updated from that layer's
own counts, exactly as the loss computes `f·P` per layer. Pooling either the counts or the biases
across layers would let a well-spread layer statistically mask a collapsed one, the same masking
failure I flagged for the loss. Keeping both per-layer means each router is corrected against the
balance it individually produced — which is the only version of "balanced" that actually prevents
dead experts, since an expert dies in a specific layer, not in some cross-layer average.

On the fitness, the Switch rung sat at `r = −3.7868`. If the global loss alone lands imbalance in the
same band at unmoved cross-entropy, its `r` will be a whisker either side of that — same penalty
form, and I should not expect the scope change to move `r` much at this scale. The interesting number
is the bias variant. If the count controller drives imbalance down by even a factor of two or three
below the loss's band, and the selection-only design keeps cross-entropy flat as intended, then `r`
would climb appreciably past `−3.78` toward the mid `−3.75`s, and that gain would again be almost
entirely in the imbalance term. I will read the decomposition the same way as before: a fitness gain
that lives in `L_imb` while `L_CE` stands still is the honest kind; a gain financed by a
cross-entropy regression would mean the bias was leaking into the predictions after all, contradicting
the selection-only claim, and I would want to catch that rather than bank the headline `r`.

So the rung is: the same `f·P` penalty with `f` measured over the global batch rather than the
micro-batch, optionally augmented by the gradient-free selection bias. My falsifiable expectations
are that the loss alone lands imbalance in the Switch band or a hair better, at cross-entropy no
worse; and that adding the bias pushes imbalance meaningfully below the loss alone, since a direct
count controller balances counts harder than a smooth term can. But I want to carry forward, precisely,
the thing this rung does *not* fix, because it is the opening for whatever comes next. The global-batch
loss equalizes the *average* usage, and it has nothing special to say about the experts that have
fallen *well below* their fair share. I can see this in the same gradient I derived for the surrogate:
the push on expert `j` is `N P_j (f_j − m)`, whose magnitude scales with `P_j`, so a nearly-dead
expert with `P_j ≈ 0` gets a nearly-zero gradient even though its deficit `f_j − m` is largest — the
two small factors multiply and the rescue collapses exactly in the tail where it matters most. Let me
put a number on it for a router the global loss has *almost* balanced, with one expert dying: say
`P = f = (0.16, 0.15, 0.15, 0.15, 0.15, 0.12, 0.11, 0.01)`, so `m = Σ P_i^2 = 0.142`. The dying
expert at `0.01` has by far the largest deficit, `f_j − m = −0.132`, four times any other expert's;
yet its push is only `8·0.01·0.132 = 0.011`, while a merely mild-cold expert at `0.11`, whose deficit
is a quarter as large at `−0.032`, gets `8·0.11·0.032 = 0.028` — `2.7×` the pull on the expert that
needs it four times *less*. The prefactor `P_j` has inverted the priority: the smooth term works
hardest on the experts closest to recovery and barely touches the one actually dying. To the `f·P`
term an expert at a tenth of its fair share and one at nine-tenths of it are, if anything, ranked
backwards. And the count bias does not close this gap either, for a different
reason: it does balance the counts, but it gives the router *no gradient* about balance through `P` at
all — it fixes the symptom in the hard selection while leaving the router's own probabilities
uninstructed about the under-used experts. So neither mechanism specifically resurrects the dying
through the router's own probabilities. Balancing the mean is not the same as resurrecting the dying,
and that under-utilized tail — left on the table by the smooth global term and untouched-in-gradient
by the count bias — is the opening the next rung has to attack.
