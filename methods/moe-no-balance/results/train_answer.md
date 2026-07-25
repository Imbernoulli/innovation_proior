When I set out to understand load balancing in a Mixture-of-Experts model, I forced myself to start
with the case that has no load balancing at all, because I realized I had no way to judge any fix
without first measuring the disease. A MoE layer routes each token, through a small softmax router,
to its top two of eight experts, and the router is trained by nothing but the language-model
cross-entropy. The trouble is that the cross-entropy is a sum over tokens of how well each next
token was predicted, and a single token has no reason to care whether the expert it used is shared
by a million other tokens or by ten — it cares only that the expert it used predicted well. There is
no term anywhere in that loss that sums over experts and asks whether the usage is spread out. Load
balance is a global property of the routing distribution, and the per-token loss is structurally
blind to it.

That blindness has a direction. If, early in training, the router by chance sends a few extra tokens
to one expert, that expert gets more gradient, trains faster, becomes more useful, and so the router
learns to send it even more tokens. It is a positive feedback loop with nothing to oppose it, and
its attractor is collapse: a few experts soak up the traffic while the rest receive almost no
gradient, never specialize, and become dead weight. So the method here is deliberately to do nothing
— the load-balancing loss is a literal zero — and to measure what falls out.

I measure two things on held-out data. The first is the cross-entropy itself, $L_{CE}$, read also as
perplexity $\exp(L_{CE})$. The second is the load imbalance: let $f_i$ be the fraction of all routed
(token, slot) assignments that land on expert $i$ — a hard count taken over the top-$K$ choices
actually made — and define
$$L_{\mathrm{imb}} = \frac{1}{2}\sum_{i=1}^{N} \left| f_i - \frac{1}{N} \right|,$$
the L1 deviation of the allocation from uniform. This is zero for a perfectly balanced router and
climbs toward $1-1/N$ as routing concentrates onto fewer experts. I combine the two into the single
fitness used to score every later method,
$$r = -\left(L_{CE} + L_{\mathrm{imb}}\right),$$
so that fixes are judged on the joint point and not on imbalance alone, because crushing the
imbalance by wrecking the router would be a hollow win. What this control establishes is the honest
floor: a tolerable cross-entropy sitting on a clearly skewed allocation, which is exactly the
signature that says the cure must come from outside the cross-entropy, from a term added by hand
that looks at the routing distribution and pushes it toward uniform.

The method itself, then, is the deliberate absence of that term: the editable balancing slot — the
function later rungs will fill with an auxiliary loss built from the router's probabilities and its
hard top-$K$ choices — is fixed here to the constant map returning the scalar $0$, irrespective of
what those probabilities or choices are. The contract this control must satisfy is exactly as strict
as any later balancing loss: the map has to stay differentiable in the router's parameters —
trivially true, since a constant has zero gradient everywhere — and it must not drop a single token
or alter the architecture, so that every later rung's improvement over $r$ is attributable to the
loss term alone and not to a change in routing capacity. With $N=8$ experts, top-$K=2$ routing, two
MoE layers of width $d=64$, and $L_{CE}$, $L_{\mathrm{imb}}$ read off the same 20 held-out batches
used by every later rung, this control's protocol is the ruler against which every subsequent
balancing loss is measured: because it does nothing, whatever imbalance and fitness it produces are
intrinsic to unregularized MoE routing, not an artifact of a weak penalty.
