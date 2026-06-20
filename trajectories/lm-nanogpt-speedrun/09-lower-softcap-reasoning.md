Four point four one minutes now, 1530 steps, val_loss 3.2751 — the value embeddings paid for their extra
lookup time and then some. The architecture is dense with little learned knobs now: per-layer value
lambdas, embed-shortcut lambdas, U-net skip weights, a dedicated value table. With all of that in place I
find myself rereading the script for any *constant* I set once and never reconsidered, because a constant
chosen for one regime is exactly the kind of thing that's quietly wrong in another. And one jumps out: the
tanh logit softcap. I put it in to keep any single logit from blowing up — `logits = 30*torch.tanh(logits
/ 30)` — and I picked 30 the way you pick a safety bound, loose enough that it never interferes with normal
training. But "never interferes" should make me suspicious now. If a regularizer never bites, it isn't
regularizing; it's just sitting there.

Let me actually look at what a cap of 30 does in this regime. tanh(z) is essentially linear for small z —
tanh(z) ≈ z until |z| gets up near 1 — so `30*tanh(logits/30)` is essentially the identity until the raw
logit gets to a meaningful fraction of 30. The cap only starts to *do* anything once logits climb toward,
say, 15 or 20, and it only hard-saturates near 30. The question is: in a 1530-step run on this much data,
how big do the logits actually get? This is a small-scale regime — tiny by the standards of where logit
softcap was born. Gemma 2 is a multi-trillion-token model with a head over a huge vocabulary, and a cap of
30 there is genuinely protecting against runaway logits that *do* occur over long training. Here I'm
training for minutes, the head is zero-initialized and starts from uniform logits, and the run ends long
before logits would plausibly approach 30. So for almost the entire run, `30*tanh(logits/30)` ≈ `logits` —
the softcap is barely active. It's a guardrail at the edge of a cliff I never walk near.

So the honest reframing is: the cap isn't currently functioning as a regularizer at all, it's functioning
as the identity. And that reframes what *lowering* it would do. If I tighten the cap — say to 15 —
`15*tanh(logits/15)` is linear only until logits reach a fraction of 15, and it starts visibly bending
much earlier, becoming active over a far larger range of the logit values the run actually produces. A
tighter cap is a stronger constraint on the *shape* of the logit distribution: it forces the logits to be
more compact, squashing the large ones back toward the bounded range, which is precisely what a regularizer
does — it imposes structure the network would otherwise have to discover (or fail to discover) on its own.

And here's the thing about the small-scale regime specifically. When you have lots of data and lots of
steps, the network can afford to learn all the structure itself; imposing extra structure from outside just
gets in its way and you'd rather let it be flexible. But when you're training for *minutes*, the network
doesn't have time to learn everything from scratch, and any correct structure you can *hand* it — for free,
as a constraint — is structure it doesn't have to spend its scarce steps discovering. A tighter logit bound
is one such piece of structure: keeping the logit distribution compact is, empirically, a good inductive
bias for next-token prediction, and forcing it via the cap means the optimizer can spend its gradient
budget elsewhere. In the data-rich regime a tight cap would probably hurt — it would be a straitjacket. In
this data-starved, time-starved regime, the straitjacket is a gift. The regime determines the sign of the
trade, and I'm deep in the regime where extra imposed structure tends to win.

There's a tension I should name, because lowering the cap to 15 is not obviously safe: if I tighten it too
far I'll distort the actual probabilities the model needs to express. A token that genuinely deserves a
logit margin of 20 over its competitors can't get it if the cap is 15 — the softmax can never become as
confident as the data wants. That would *raise* the loss. So the right cap is the loosest one that still
bites meaningfully across the run, not the tightest possible. 15 is a halving — a real tightening, but not
a collapse to single digits — and it's the natural first step down to test whether the regime rewards more
structure. If val_loss holds at or below the bar with the cap at 15, that's direct evidence that the
network was leaving structure on the table that the cap can supply for free, and I'd expect to be able to
*drop steps* as a result — the imposed structure does some of the work the steps were doing.

The change itself is almost embarrassingly small — it's one literal edit, the number 30 becomes 15 in two
places: `logits = 30*torch.tanh(logits/30)` becomes `logits = 15*torch.tanh(logits/15)`. No new parameters,
no new tensors, no new compute to speak of — tanh costs the same regardless of the constant. It's purely a
change in how much structure I'm imposing on the output. So the cost side is zero and the only question is
whether the regime rewards the tighter bound. Given everything above — that the cap at 30 is currently
nearly inert, that we're firmly in the small-scale regime where imposed structure helps, and that 15 is a
moderate halving rather than a collapse — I think it does. I'll lower the cap and expect to pull the step
count down from where the intervening tuning has it (around 1490) to something like 1390, with val_loss
staying under 3.28.

So: the tanh softcap at 30 was chosen as a loose safety bound and is nearly inert in this short, small-scale
run — for almost the whole run `30*tanh(logits/30)` ≈ `logits`. Halve it to 15 so the cap becomes active
over the range of logits the run actually produces, turning an inert guardrail into a genuine regularizer
that keeps the logit distribution compact. In the data- and time-starved regime, that imposed structure is
work the network doesn't have to spend steps learning, so the step count can come down — from ~1490 to
~1390 — while val_loss stays under the bar.

```python
# before:
logits = 30 * torch.tanh(logits / 30)
# after (this record):
logits = 15 * torch.tanh(logits / 15)
```

The chain: a softcap of 30 is so loose that `30*tanh(logits/30)` ≈ `logits` for nearly the whole short
run — it's a guardrail at a cliff edge I never reach, so it isn't regularizing anything; halving it to 15
makes the cap active over the logit range the run actually produces, imposing the structure that the logit
distribution stay compact; and in the small-scale, time-starved regime, free imposed structure is work the
network doesn't have to spend steps learning, so the same one-line change lets the step count fall from
~1490 to ~1390 while val_loss holds under 3.28.
