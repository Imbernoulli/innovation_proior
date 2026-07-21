The Switch aux loss did what I designed it to do, and the numbers let me read the outcome precisely.
Imbalance fell from the control's `0.1286` to `0.0587`, a `2.19×` cut, while the cross-entropy went
from `3.7280` to `3.7281` — a change of `+0.0001`, nothing at perplexity `41.6`, inside any plausible
run-to-run noise. So fitness climbed from `−3.8566` to `−3.7868`, a gain of `+0.0698`, and the
decomposition is clean: the imbalance term contributed `+0.0699` and the cross-entropy `−0.0001`. The
`f·P` surrogate steered the router toward uniform exactly as the gradient analysis said it would, at no
cost in prediction.

Now the complaint, which the data did not hand me on a plate. The previous rung predicted the
micro-batch scope would tax the cross-entropy by flattening per-slice specialization, and I set up the
decomposition to catch that tax. It did not show: the cross-entropy was flat to four decimals. So the
empirical case that micro-batch scope is hurting me is, at this scale, unmade — the tax is below the
resolution of this run, just as I flagged it might be, because four quarters of a small batch on a
synthetic task are still fairly representative of one another. But the principled objection survives
untouched: the penalty demands uniformity of every slice, which is strictly stronger than the
whole-batch uniformity the metric rewards and strictly at odds with the per-slice specialization I want
to keep. On any corpus with real topical heterogeneity that gap opens up. So I make the change that is
principled even where it is not yet empirically forced, and I make it as the smallest possible move,
turning exactly one knob: the set of tokens over which `f_i` is computed.

Why that single knob is the whole story. The property I care about is corpus-level — across all the
data, every expert should pull roughly its fair share. I emphatically do *not* care that one
micro-batch is internally uniform; if a micro-batch happens to be all one topic, the right behavior is
for that topic's experts to light up and the rest to stay quiet. Specialization *is* per-slice skew
that averages out to balance over the corpus, and the micro-batch penalty conflates that benign skew
with the pathological skew of collapse. So the fix is to measure `f_i` over the global batch — the
union of all the micro-batches — and keep everything else exactly as it was: `α · N · Σ_i f_i P_i`, the
same `α = 10^{-2}`, the same multiply-by-`N`, the same detached counts and differentiable
probabilities, the same mean over layers. Only the scope of `f` moves. Now the penalty asks a different
question — across all this data, is usage uniform? — and is silent about whether any individual slice
is. A micro-batch is free to be as specialized as its content demands, so long as the specializations
of different slices cover all the experts when summed.

The scope difference is exact on an idealized example. Split the batch into four equal slices, each
perfectly specialized: slice one routes every token to experts `{1,2}`, slice two to `{3,4}`, and so on
through `{7,8}`, the router's probability on each slice peaked on that pair. This is the *ideal* MoE —
every token sharply routed, every expert fully used across the corpus — and its whole-batch histogram
is exactly uniform, so global `L_imb = 0`. The global penalty pools the tokens, sees uniform `f` and a
uniform mean `P`, and reads `N Σ_i f_i P_i = 1`, the scale-free floor, nothing to correct. The
micro-batch penalty scores each slice on its own: on slice one, `f = P = (0.5, 0.5, 0, …)` gives
`8·0.5 = 4`, four times the floor, with a gradient actively trying to flatten that slice's peaked `P`.
So on the identical, ideal router the micro-batch loss reports a `4×` violation and pushes to destroy
the specialization, while the global loss reports no problem at all. It cannot tell this perfect router
from a collapsed one, and the global scope can. My real slices are nowhere near this separation — which
is exactly why the tax stayed invisible — but the mechanism I am correcting is this one, sized down.
Widening the scope buys a second thing independent of the topical argument: `f` is a multinomial
estimate whose per-expert noise scales as `1/√M`, and a micro-split holds a quarter of the tokens, so
its `f` is roughly twice as noisy. Since `f` is the detached weight deciding *where* the pressure
points, a noisier `f` jitters the gradient direction from split to split; the global scope halves that
noise and hands the gradient a steadier aim.

I should give the micro-batch scope its due before abandoning it, because it did not become the default
by accident. In a real distributed MoE the experts live on different devices with a finite per-device
capacity, and tokens over the cap are dropped; per-device balancing guards against that overflow, a
genuinely per-device property a corpus-level constraint does not enforce. But that justification does
not apply here: the contract forbids dropping tokens, imposes no capacity limit, and processes one batch
on one device. So the per-device constraint is pure over-constraint with no compensating benefit, and I
drop it with a clear conscience. I also consider the one real alternative to the exact global count — an
exponential moving average of recent batches' `f`, tempting in a distributed setting because it avoids
a synchronization. But it mixes *stale* counts, gathered under an older router, with the *current* `P`
the gradient flows through; feeding a lagged `f` aims yesterday's correction at today's router, which
near the balanced fixed point can turn the corrective step into an oscillation. In this single-process
reproduction the training batch already *is* the global batch, so I compute `f` on the full batch and
skip the approximation. Concretely I realize the contrast by computing the previous rung's penalty on
four micro-splits and this rung's on the full batch — a faithful stand-in for the per-device-versus-
global distinction, even if the absolute gap is muted where the quarters are already representative. If
global scope helped a lot here I would be suspicious that something other than scope had changed.

There is a complementary lever I want to run alongside this rung, because it sits naturally beside a
balancing loss even though it is not a loss at all. It maintains a per-expert bias `b_i` used *only* to
break ties in the top-K selection — added to the routing scores for ranking, excluded from the gate
weights that actually combine the experts — and nudges it once per step in the direction that cools
overloaded experts and warms underloaded ones: with `c_i = N f_i` the normalized load (mean exactly `1`
at any allocation) and `c̄` its mean, `b_i ← b_i + u · sign(c̄ − c_i)`, `u = 10^{-3}`. The sign is right
— an over-used expert has `c_i > 1` so its bias falls; an under-used one rises. Two design choices are
worth understanding. First, it acts on selection only, not the gate weights: for each token the router
picks its top-two and combines their outputs with the softmax probabilities of the chosen experts. If I
biased the combine weights I would be mixing expert outputs with weights that are no longer the model's
true probabilities — a first-order hit to prediction that would surface immediately in cross-entropy.
Restricting `b` to the ranking leaves the combine weights honest, so the only way the bias moves the
output is by occasionally swapping which expert sits in the top two, and it swaps only experts already
close in score — the promoted cold expert was a near-miss, so displacing the expert it edges out costs
little. That is how a controller with the authority to move counts hard leaves cross-entropy where it
was: it reroutes without ever re-weighting, and only at the margin. Second, it is bang-bang — a fixed
step in the direction of `sign`, not proportional to the imbalance — which makes it robust at the price
of only ever moving `b` at rate `u`. I can bound its authority: over `1200` steps a consistently
one-signed bias drifts by at most `u · 1200 = 1.2` in logit units, and since the router's raw logits
sit at a spread of order one, a bias of order one is enough to promote a persistently cold expert into
a token's top two. So the controller has real authority over the hard selection, and because it
balances the *counts directly* rather than through a smooth surrogate I expect it, when added, to drive
imbalance lower than the loss alone.

A bang-bang integral controller does not converge to a point — it limit-cycles. Once the counts are
balanced, `sign(c̄ − c_i)` keeps flipping as finite-batch fluctuations push each `c_i` just above or
below the mean, so each `b_i` dithers by about `±u` around whatever offset it needed. That dither has
amplitude `10^{-3}` in logit space, negligible against a spread of order one, so it settles into
balancing the counts to within the batch's own sampling noise — close to the resolution floor `L_imb`
has anyway, so a count controller can in principle press imbalance right down to that floor. And it
cooperates with the loss rather than fighting it: both want the under-used experts warmed, the loss by
raising their `P` through a gradient and the bias by raising their selection score through an integral
of past load. The two act on different surfaces toward the same allocation, so I expect them to compose.
Like the loss, the bias is maintained per layer — the two routers have independent tails, an expert
dies in a specific layer, and pooling either counts or biases across layers would let a well-spread
layer mask a collapsed one. I will run the global-batch loss both alone and with this bias to see how
much the count-level controller buys on top.

On the fitness, the Switch rung sat at `−3.7868`. If the global loss alone lands imbalance in the same
band at unmoved cross-entropy, its `r` will be a whisker either side of that — same penalty form, and I
should not expect the scope change to move `r` much at this scale. The interesting number is the bias
variant: if the count controller drives imbalance down by a factor of two or three below the loss's
band while the selection-only design keeps cross-entropy flat, `r` climbs appreciably past `−3.78`
toward the mid `−3.75`s, again almost entirely in the imbalance term. A gain that lives in `L_imb` while
`L_CE` stands still is the honest kind; a gain financed by a cross-entropy regression would mean the
bias was leaking into predictions after all, contradicting the selection-only claim.

But I want to carry forward, precisely, the thing this rung does *not* fix, because it is the opening
for whatever comes next. The global-batch loss equalizes the *average* usage and has nothing special to
say about experts that have fallen *well below* their fair share. This is the same tail-weakness I
flagged in the gradient `N P_j(f_j − m)`: a nearly-dead expert with `P_j ≈ 0` gets a nearly-zero
gradient even though its deficit is largest. On a router the global loss has *almost* balanced with one
expert dying — say `P = f = (0.16, 0.15, 0.15, 0.15, 0.15, 0.12, 0.11, 0.01)`, `m = Σ P_i^2 = 0.142` —
the dying expert at `0.01` has by far the largest deficit `−0.132`, yet its push is only
`8·0.01·0.132 = 0.011`, while a merely mild-cold expert at `0.11` with a quarter the deficit gets
`8·0.11·0.032 = 0.028` — `2.7×` the pull on the expert that needs it far less. The prefactor `P_j` has
inverted the priority: the smooth term works hardest on the experts closest to recovery and barely
touches the one actually dying. And the count bias does not close this from the loss side either: it
balances the counts by shoving the hard selection around but gives the router *no gradient* about
balance through `P` at all. So neither mechanism resurrects the dying *through the router's own
probabilities*. Balancing the mean is not the same as resurrecting the dying, and that under-utilized
tail is the opening the next rung has to attack.
