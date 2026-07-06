Four point four one minutes now, 1530 steps, val_loss 3.2751, about 174 ms a step — the value embeddings
paid for their extra lookup time and then some. Reading the record: the step count fell 1750 → 1530 (13%),
the step_avg rose 160.89 → 174.08 ms (8%, the 12×-wider vte gather, right where I predicted), and the val_loss
even improved to 3.2751 with the margin out to 0.0049 — so the dedicated per-token value was a richer signal
than the shared v1, exactly the bet. On the product, 1530 × 174.08 ≈ 266k ms against the prior 282k — a ~6%
cut where a 13% step drop was partly eaten by the 8% step-time rise, the trade I signed up for. The architecture is now dense with little learned knobs: per-layer value
lambdas, embed-shortcut lambdas, U-net skip weights, a dedicated value table. With all of that in place I find
myself rereading the script for any *constant* I set once and never reconsidered, because a constant chosen
for one regime is exactly the kind of thing that's quietly wrong in another — every other knob on this model
is now learned or scheduled, and the un-revisited hard-coded numbers are where stale assumptions hide. Which constant, though — there are a few un-learned numbers left. The momentum endpoints (0.85, 0.95) and the
window endpoints (64, 1792) are constants, but they're already *scheduled* — I've thought about their regime
dependence and given them a trajectory. The LR schedule is tuned. What's left that is a single fixed number
set once, for a reason that was about a *different* regime than the one I'm in, and never revisited? One
jumps out: the tanh logit softcap. I put it in to keep any single logit from blowing up — `logits =
30*torch.tanh(logits/30)` — and I picked 30 the way you pick a safety bound, loose enough that it never
interferes with normal training. But "never interferes" should make me suspicious now. If a regularizer never
bites, it isn't regularizing; it's just sitting there, and a term that does nothing is either free insurance
or a wasted degree of freedom, and I'd like to know which.

Let me actually look at what a cap of 30 does in this regime, with numbers rather than intuition. tanh(z) is
essentially linear for small z — tanh(z) ≈ z − z³/3, so it's within a percent of z until |z| ≈ 0.3 — which
means `30*tanh(logit/30)` is essentially the identity until the raw logit gets to a meaningful fraction of
30. Put values through it. At a logit of 5: 30·tanh(5/30) = 30·tanh(0.167) = 30·0.1651 = 4.95, a 1%
reduction — invisible. At a logit of 10: 30·tanh(0.333) = 30·0.3215 = 9.64, a 3.6% reduction — barely there.
The cap only starts to *do* anything once logits climb toward 15 or 20, and it only hard-saturates near 30.
So the question becomes empirical: in a 1530-step run on this much data, how big do the logits actually get?
Let me estimate the logit magnitude the model needs. A decisive next-token prediction — say the model wants to
put probability ~0.5 on the correct token out of a 50304-way vocabulary — needs the winning logit to sit
roughly ln(0.5·V) ≈ ln(25000) ≈ 10 above the bulk of the field. Even a very confident p ≈ 0.9 prediction is a
logit margin of order ~12–13. So the logits this run actually produces to make its predictions live in the
range of roughly 10–15, not 30. This is a small-scale, short regime — tiny by the standards of where logit
softcap was born. Gemma 2 is a multi-trillion-token model with a head over a huge vocabulary, and a cap of 30
there is genuinely protecting against runaway logits that *do* occur over long training. Here I'm training for
minutes, the head is zero-initialized and starts from uniform logits, and the run ends long before logits
would plausibly approach 30. That zero-init matters for the argument: the logits don't *start* large and get
capped down, they start at exactly 0 (uniform softmax) and *grow* over training as the head sharpens. So the
question is how far they grow in ~1500 steps, and the answer from the prediction-margin estimate is "to about
10–15" — the model gets confident enough to make decisive predictions but has no reason and no time to push a
logit to 30, which would correspond to an absurd over-confidence (p ≈ 1 to ~13 decimal places) that
next-token prediction on real text never needs. So for almost the entire run, `30*tanh(logit/30)` ≈ `logit` —
the softcap is barely active. It's a guardrail at the edge of a cliff I never walk near.

So the honest reframing is: the cap isn't currently functioning as a regularizer at all, it's functioning
as the identity over the 10–15 range where my logits live. And that reframes what *lowering* it would do. If
I tighten the cap — say to 15 — `15*tanh(logit/15)` is linear only until logits reach a fraction of 15, and
it starts visibly bending much earlier. Put the same values through the tighter cap to see the difference. At
a logit of 10: 15·tanh(10/15) = 15·tanh(0.667) = 15·0.5829 = 8.74 — a 12.6% reduction, where the cap of 30
gave only 3.6%. At a logit of 5: 15·tanh(0.333) = 15·0.3215 = 4.82 — a 3.6% reduction, where 30 gave 1%. And at the top of the
band, a logit of 13 (a p ≈ 0.9-scale prediction): 15·tanh(13/15) = 15·tanh(0.867) = 15·0.700 = 10.5, a 19%
pull-down under cap 15, versus 30·tanh(13/30) = 30·tanh(0.433) = 30·0.408 = 12.25, only 5.8% under cap 30. So
across exactly the 5–15 band where the run's logits actually sit, the cap of 15 is meaningfully active — it's
squashing the large logits back, and the more so the larger they get — while the cap of 30 was nearly inert
throughout. A tighter cap is a stronger constraint
on the *shape* of the logit distribution: it forces the logits to be more compact, pulling the large ones back
toward the bounded range, which is precisely what a regularizer does — it imposes structure the network would
otherwise have to discover (or fail to discover) on its own.

And here's the thing about the small-scale regime specifically, which decides the *sign* of this trade.
When you have lots of data and lots of steps, the network can afford to learn all the structure itself;
imposing extra structure from outside just gets in its way and you'd rather let it be flexible — a tight cap
in that regime is a straitjacket that prevents the model from expressing the confident predictions the
abundant data supports. But when you're training for *minutes*, the network doesn't have time to learn
everything from scratch, and any correct structure you can *hand* it — for free, as a constraint — is
structure it doesn't have to spend its scarce steps discovering. A tighter logit bound is one such piece of
structure: keeping the logit distribution compact is, empirically, a good inductive bias for next-token
prediction, and I can name the mechanism through the softmax gradient. Cross-entropy's gradient with respect
to the correct-token logit is (p − 1): as an over-confident prediction drives p → 1, that gradient → 0, so the
model stops learning from tokens it's already (over-)sure about, while an over-confident *wrong* logit gets a
large gradient — the loss landscape near saturated logits is exactly where the signal is worst. Keeping logits
compact keeps predictions in the region where the softmax gradient is still informative. And the softcap acts
directly on that gradient: the derivative of C·tanh(logit/C) is sech²(logit/C), so the cap attenuates the
gradient it passes back on large logits. Compute it. At a logit of 10, cap 30 gives sech²(0.333) ≈ 0.90 —
a 10% attenuation, negligible; cap 15 gives sech²(0.667) ≈ 0.66 — a 34% attenuation, real. At a logit of 15,
cap 30 gives sech²(0.5) ≈ 0.79 (21%) while cap 15 gives sech²(1.0) ≈ 0.42 (58%). So the tighter cap doesn't
just squash the forward logit, it damps the gradient pressure that was inflating it in the first place, and it
does so precisely over the 10–15 band — a self-limiting feedback that keeps the distribution compact. Forcing
that via the cap means the optimizer can spend its gradient budget elsewhere. In
the data-rich regime a tight cap would probably hurt. In this data-starved, time-starved regime, the
straitjacket is a gift. The regime determines the sign of the trade, and I'm deep in the regime where extra
imposed structure tends to win — which is the same principle behind everything else I've done that helps
*because* the run is short (the curriculum, the gentle inits): when steps are scarce, free correct structure
beats learned-from-scratch flexibility.

There's a tension I should name, because lowering the cap to 15 is not obviously safe. If I tighten it too
far I'll distort the actual probabilities the model needs to express. A token that genuinely deserves a
logit margin of 20 over its competitors can't get it if the cap is 15 — the softmax can never become as
confident as the data wants, and that would *raise* the loss, not lower it. So the right cap is the loosest
one that still bites meaningfully across the run, not the tightest possible; I want it active over the 10–15
band but not so tight that it clips the genuinely-confident predictions at the top of that band. Let me weigh
the alternatives on that axis. A cap of 20 would be only slightly more active than 30 — at logit 10 it gives
15·... 20·tanh(0.5) = 20·0.462 = 9.24, a 7.6% reduction, more than 30's 3.6% but still leaving the 10–15 band
mostly linear — so it barely turns the regularizer on. A cap of 10 would bite hard but sit right *at* the
typical decisive-logit magnitude, clipping exactly the confident predictions I need to keep — a straitjacket
even for this regime. 15 is the halving: a real tightening (active across the 5–15 band, as the numbers show)
without collapsing onto the confident-prediction magnitude, so it regularizes the bulk of the distribution
while still letting the top-end logits reach ~15.

Let me verify the "15 is safe, 10 is not" claim with the confidence headroom it leaves, because that's the
whole distinction. The tanh output asymptotes to ±C, so the maximum capped logit is C, and the confidence the
model can express for the true token is set by its margin over the competing logits. A margin of C = 15 nats
means odds of e¹⁵ ≈ 3.3 million to 1 in favor of the true token over a competitor at the bulk — and against
the *entire* uniform 50304-way vocabulary that's 3.3M/50304 ≈ 65× headroom, so the true token still dominates
by a wide margin even in the worst case where every other token is a serious contender. So a cap of 15 does
*not* prevent the model from expressing any confidence it realistically needs; there's 65× slack. Now cap 10:
margin 10 gives e¹⁰ ≈ 22026 to 1, and 22026 < 50304 — *less* than the vocabulary size, so against a broad
field the true token no longer dominates, and a genuinely confident prediction would be clipped below the
confidence the data supports. That's the line: 15 keeps ample headroom over the full vocabulary while
regularizing the bulk; 10 crosses into clipping the confident predictions themselves. The arithmetic picks 15
as the loosest cap that still bites without distorting, and rejects 10 as over-tight — not by taste but by the
e^C-versus-vocabulary comparison. (There's a minor numerical bonus too: bounded logits keep the softmax's
exponentials in a comfortable range, so the cross-entropy stays well-conditioned — but that's a side benefit,
not the reason.) It's the natural first step down — moderate, testable, and
if it holds I can consider going lower later. If val_loss holds at or below the bar with the cap at 15, that's
direct evidence the network was leaving structure on the table that the cap can supply for free, and I'd
expect to be able to *drop steps* as a result — the imposed structure does some of the work the steps were
doing.

It's worth being explicit about *how* imposed structure turns into fewer steps, because "regularizer helps"
is not automatically "run is shorter." The tanh cap with the sech² gradient damping short-circuits the
logit-inflation feedback loop: without a biting cap, part of every late gradient goes into pushing the winning
logit higher (the softmax rewards confidence), and part of the optimizer's budget is spent inflating logits
that the model then has to keep in check — motion that doesn't reduce val_loss much because it's over-confidence
past the point of diminishing returns. With the cap biting over 10–15, that inflation is damped at the source,
so those gradient components are freed to go toward genuine loss reduction instead. The structure the cap
hands the model — "keep the logit distribution compact" — is thus not just a passive constraint; it redirects
gradient budget from a low-value activity (inflating already-confident logits) to loss reduction, which is
exactly what "does the work the steps were doing" means concretely.

There's a meta-point here that's really the reason I looked at the softcap at all. This ladder has been
relentlessly *shortening* the run — from 6200 steps at the Muon record down to 1530 now, a factor of four —
so the regime has drifted a long way from wherever any borrowed constant was originally set. The softcap of 30
came from a large-scale, long-training setting; my run is now shorter and smaller than that setting by orders
of magnitude, so a constant calibrated there has no reason to be right here. Whenever the operating point
moves this far, the borrowed constants are the first place to look for free wins, because each one encodes an
assumption about a regime I've left behind. The softcap is the cleanest example because its regime dependence
is so direct — its whole job (bound runaway logits) is a function of how large logits get, which is a function
of how long and large the run is — but the general move is "re-examine every inherited number when the regime
shifts," and I expect it to keep paying off as the run gets shorter still.

The change itself is almost embarrassingly small — it's one literal edit, the number 30 becomes 15 in two
places: `logits = 30*torch.tanh(logits/30)` becomes `logits = 15*torch.tanh(logits/15)`. No new parameters,
no new tensors, no new compute to speak of — tanh costs the same regardless of the constant, so step_avg won't
move at all. It's purely a change in how much structure I'm imposing on the output. The one thing I have to be careful
about is applying the same constant everywhere the softcap appears — training *and* validation — so the model
is evaluated under exactly the output transform it trained with; a mismatch (train at 15, eval at 30, or vice
versa) would measure the model through a different final nonlinearity than it optimized against and corrupt the
val_loss comparison. Hence the edit lands in both places, changing 30 to 15 consistently. So the cost side is
exactly zero and the only question is whether the regime rewards the tighter bound. Given everything above —
that the cap at 30 is arithmetically nearly inert over the 10–15 range the run's logits occupy, that we're
firmly in the small-scale regime where imposed structure helps, and that 15 is a moderate halving rather than
a collapse onto the confident-prediction magnitude — I think it does.

If the mechanism is right, the falsifiable signature is unusual: step_avg dead flat (a tanh constant is free),
val_loss holding at or under the bar, and the step *count* dropping — because the imposed compactness does
work the steps were doing. That last part is the real claim, and it's the one that could fail: if val_loss
*rises* above 3.28, the cap is too tight and is clipping predictions the data wants (I'd go back toward 20);
if the step count *doesn't* fall while val_loss holds, then the structure wasn't load-bearing and the cap was
genuinely just insurance (harmless, but not a win) — in which case I've at least confirmed the softcap can be
tightened for free and can keep pushing it lower on a later rung. I'll lower the cap and expect to pull the step count down
from where the intervening tuning has it — around 1490 — to something like 1390, with val_loss staying under
3.28 at unchanged step time — a pure step-count win from a zero-cost, one-character edit.

```python
# before:
logits = 30 * torch.tanh(logits / 30)
# after (this record):
logits = 15 * torch.tanh(logits / 15)
```

The chain: every other knob on this model is learned or scheduled, so the un-revisited constants are where
stale assumptions hide, and the tanh softcap of 30 is one — arithmetically `30*tanh(logit/30)` reduces a
logit of 10 by only 3.6% and of 5 by 1%, so over the 10–15 band the run's logits actually occupy (a decisive
50304-way prediction needs a margin ~ln(25000) ≈ 10) it's a guardrail at a cliff edge I never reach, not a
regularizer; halving it to 15 makes the same logits bite hard (12.6% at logit 10) — imposing the structure
that the logit distribution stay compact; and in the small-scale, time-starved regime, free imposed structure
is work the network doesn't have to spend steps learning (while 20 barely turns the regularizer on and 10
would clip the confident predictions themselves), so the one-line, zero-cost change lets the step count fall
from ~1490 to ~1390 at flat step_avg while val_loss holds under 3.28.
