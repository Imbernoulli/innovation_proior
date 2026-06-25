I want a Mixture-of-Experts layer to do what it promises: route each token to a couple of
specialized experts out of many, so that capacity grows with the number of experts while the
per-token compute stays fixed. The router is the small gate that, for each token, scores the
experts and sends the token to its top two. The first thing I have to decide is what trains the
router, and the honest answer at this starting point is: nothing but the language-model loss. There
is no separate signal telling the router to use all the experts; it sees only the cross-entropy at
the output, backpropagated through whichever experts the tokens happened to be sent to.

So let me think carefully about what that objective actually wants, because the whole problem of
this ladder is hiding in the answer. Suppose, early in training, the router by chance sends a few
more tokens to expert three than to the others. Expert three now receives more gradient, so it
trains faster, so its outputs become more useful, so the cross-entropy improves when tokens go to
it, so the router learns to send *even more* tokens to expert three. Let me make sure this is
actually a self-reinforcing loop and not just a one-step nudge that washes out. The router gate is a
softmax over expert scores; the gradient on the score for expert three has two pieces. One is the
usual softmax competition — raising score three lowers the others. The other is the chain through
the expert: the cross-entropy gradient that flows back into the gate is weighted by how good expert
three's prediction was relative to the alternatives. If expert three is currently *better* than the
others (because it has trained more), that weighting is positive, so the gate score for three is
pushed up, which routes more tokens to three next step, which trains three even more. The two pieces
point the same way, and the "expert three is better" condition is self-sustaining once it gets a head
start. So it is a feedback loop, not a transient. The objective is perfectly happy to pour almost
all the traffic into a handful of experts and let the rest starve. The starved experts receive
almost no gradient, never specialize, and become dead weight — parameters that cost memory and buy
nothing. That is router collapse, and on this reasoning it is not a rare pathology I have to provoke
but the default attractor of an unregularized MoE — though "default attractor" is a claim about the
end state of training, and I'll only really know its magnitude once I measure it.

I should be precise about why the cross-entropy cannot see this as a problem. The loss is a sum
over tokens of how well the model predicted each next token. A token does not care, for its own
loss, whether the expert it used is also used by a million other tokens or by ten; it cares only
that the expert it used produced a good prediction. There is no term anywhere that sums over
*experts* and asks whether the usage is spread out. Balance is a global property of the routing
distribution, and the per-token loss is blind to global properties of the routing distribution. So
the router will optimize what it is told to optimize — prediction — and balance will be whatever
falls out, which is to say, skewed.

Now I have to resist the temptation to fix this immediately, because the point of this first rung is
diagnostic, not corrective. If I jump straight to adding a balancing penalty, I will never know how
bad the disease actually is, and every later number — does the Switch loss cut the imbalance by half
or by a factor of ten? — will be uninterpretable because I have no floor to compare against. So this
rung is deliberately the control: I add *no* balancing loss at all. The editable slot returns a
literal zero. The model trains on cross-entropy alone, and I measure two things at the end.

The first is the obvious one: the cross-entropy itself, on held-out data, which I will also read as
perplexity. The second is the quantity this whole ladder is built around — the load imbalance. I
define it as the L1 deviation of the token allocation from uniform: take `f_i`, the fraction of all
the routed (token, slot) assignments that landed on expert `i`, and compute half the sum over
experts of the absolute gap between `f_i` and the uniform share `1/N`. Let me pin down the range of
this number so I know what a reading means before I see one. If every expert gets exactly `1/N`, the
sum is zero — perfect balance, as intended. The other end is the question. Put all the mass on a
single expert, `f = (1, 0, …, 0)`: then the one large term contributes `1 − 1/N` and the other seven
each contribute `1/N`, so the sum is `(1 − 1/N) + (N−1)·(1/N) = 2(1 − 1/N)`, and the half makes it
`1 − 1/N`. With `N=8` that ceiling is `7/8 = 0.875`. So the half is exactly the normalization that
puts the maximum at `1 − 1/N` rather than at two. But I should be careful: can the routing actually
reach that ceiling here? With top-`K=2`, every token contributes *two* slot-assignments to *two
distinct* experts, so no token can dump both of its slots on one expert. The most concentrated the
allocation can get is all tokens picking the *same two* experts, `f = (½, ½, 0, …, 0)`. Plugging that
in: the two heavy terms give `(½ − 1/8) = 3/8` each and the six empties give `1/8` each, half the
sum is `½·(2·3/8 + 6·1/8) = ½·(6/8 + 6/8) = 6/8 = 0.75`. So the *reachable* ceiling for a fully
collapsed top-2 router is `0.75`, not `0.875` — the distinctness constraint costs an eighth. That is
worth knowing: a reading near `0.75` would mean essentially total collapse onto two experts, not
"merely bad." It is a clean, bounded, model-agnostic readout of exactly the thing the cross-entropy
is blind to. And I will combine the two into the single fitness
the ShinkaEvolve evaluation uses — the negative of cross-entropy plus imbalance — so that later rungs
are judged on the *joint* point, not on imbalance alone, because driving imbalance to zero by
crushing the router would be a hollow victory if it wrecks the cross-entropy.

What do I expect this control to show? The cross-entropy should be reasonable — collapse is not a
catastrophe for the loss; a model that leans on a few well-trained experts still predicts
acceptably, which is exactly why the objective tolerates it. The imbalance I am less sure about in
magnitude, but I can bracket it with the numbers I just worked out. If the feedback argument is
right, the allocation should sit well above zero and somewhere in the upper part of the `[0, 0.75]`
band — not pinned at `0.75` (that would need *every* token to agree on the same two experts, which
the latent-topic structure should partly resist, since different topics ought to prefer different
experts) but clearly skewed, with a handful of experts carrying much more than `1/8` and several
sitting near empty. I would put my expectation roughly in the `0.4`–`0.6` range and treat anything
near zero as a sign the feedback loop is weaker than I argued and worth re-examining. The point is
that I now have a falsifiable prediction with computed endpoints rather than a vague "it'll be
unbalanced." Whatever the run returns, the gap between a tolerable cross-entropy and a non-trivial
imbalance is the opening: it would say the problem is real and that fixing it has to come from
*outside* the cross-entropy — from a term that explicitly looks at the routing distribution and
pushes it toward uniform. The only honest way to judge such a term later is against the imbalance
number this control is about to hand me, which is why this rung returns a literal zero and measures,
rather than fixing anything.
