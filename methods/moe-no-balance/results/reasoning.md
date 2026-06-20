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
it, so the router learns to send *even more* tokens to expert three. This is a positive feedback
loop, and nothing in the cross-entropy opposes it. The objective is perfectly happy to pour almost
all the traffic into a handful of experts and let the rest starve. The starved experts receive
almost no gradient, never specialize, and become dead weight — parameters that cost memory and buy
nothing. That is router collapse, and I am fairly sure it is not a rare pathology I have to provoke
but the default attractor of an unregularized MoE.

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
experts of the absolute gap between `f_i` and the uniform share `1/N`. The half is there so the
number runs from zero, when every expert gets exactly its uniform share, up toward one as the
routing concentrates onto fewer and fewer experts. It is a clean, bounded, model-agnostic readout of
exactly the thing the cross-entropy is blind to. And I will combine the two into the single fitness
the ShinkaEvolve evaluation uses — the negative of cross-entropy plus imbalance — so that later rungs
are judged on the *joint* point, not on imbalance alone, because driving imbalance to zero by
crushing the router would be a hollow victory if it wrecks the cross-entropy.

What do I expect this control to show? The cross-entropy should be reasonable — collapse is not a
catastrophe for the loss; a model that leans on a few well-trained experts still predicts
acceptably, which is exactly why the objective tolerates it. But the imbalance should be clearly
above zero: the token allocation should be visibly skewed away from uniform, with some experts
carrying well more than their share and others well under. That gap between a tolerable cross-entropy
and a bad imbalance is the entire opening. It says the problem is real and that fixing it has to come
from *outside* the cross-entropy — from a term I add by hand that explicitly looks at the routing
distribution and pushes it toward uniform. That added term is the next rung, and the only honest way
to judge it is against the imbalance number this control is about to hand me.
