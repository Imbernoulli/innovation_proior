The residual-flow rule is the whole object I get to design, but a rule bolts onto an otherwise fixed
GPT, and with the rule left at its plainest the model *is* the floor — so the thing to start from is
just training this 24-layer GPT-2 Medium on FineWeb with the residual stream untouched. The scaffold
already hands me that: the default `Block.forward` is the standard Pre-LN block, `x = x + attn(ln_1(x))`
then `x = x + mlp(ln_2(x))`, and the default `GPT.forward` runs a bare loop `for block in h: x =
block(x)`. My step-1 edit is the trivial one — leave the loop exactly as the vanilla template writes it,
adding nothing. The `vanilla` baseline exists only so the rigorous-codebase harness has at least one
edit op per baseline, so the edit is an identity replacement of the block loop with itself; the model
that runs is plain additive Pre-LN.

I want to be precise about *why* this is the right floor and not an arbitrary one, because the entire
ladder is going to be measured against it. The Pre-LN choice is itself the resolution of the residual
lineage in the context, and I should walk that lineage carefully because each ancestor's gap is a clue
to where the floor itself is still weak. Start with plain stacked layers, no skip: compose `L`
width-preserving maps directly. In principle this is enough — a deep stack can represent anything a
shallow one can by idling the extra layers — but train it and the optimizer stalls past a few dozen
layers, and a deeper plain net reaches *higher* training error than a shallower one. That is not a
generalization gap; it is an optimization failure, and the mechanism is the per-layer gain. Perturb the
input by a little and ask how big the perturbation is at the output: each layer multiplies it by its
Jacobian, so across `L` layers the perturbation is multiplied by a product of `L` Jacobians, on average
a factor `r^L`. The backward pass is governed by the same product transposed. So if `r` is even slightly
above 1, gradients explode exponentially in depth; slightly below 1, they vanish. The only survivable
regime is `r ≈ 1` held at *every* layer, and a plain stack has no mechanism to hold it there.

The residual connection (`x_{l+1} = σ(x_l + F(x_l))`) was the escape: add an identity skip so the block
only has to learn a nudge to identity, and — this is the load-bearing part — the additive `1` in the
unrolled backward product gives the top gradient a route to every shallow layer that is *not* multiplied
by all the Jacobians above it. That made hundreds of layers trainable. But it does not pin `r = 1`: the
branch `F` fires at full strength at init, so the block is *not* the identity at step zero, and the
output variance of a residual block still compounds with depth. The skip tames the *worst* of `r^L`; it
does not set `r = 1`. Then the Transformer added normalization, and *where* it goes turns out to matter
enormously. Post-LN normalizes after the addition, `x ← LN(x + sublayer(x))`, which puts a LayerNorm
Jacobian *on* the residual highway — so the backward path becomes a product of normalization Jacobians,
exactly the multiplicative structure the skip was meant to avoid. Concretely, at init this gives large,
depth-imbalanced gradients near the output layer, which is why a full learning rate immediately
destabilizes a deep Post-LN Transformer and why the field bolts on learning-rate warm-up: ramp the rate
up slowly so the first updates are tiny enough to survive. Warm-up is a band-aid over bad init-time
signal propagation, and the tell that it is the *architecture* and not the optimizer is that the warm-up
dependence persists even when you swap Adam for plain SGD.

Pre-LN is the fix that the floor uses: move the norm *inside* the branch, `x ← x + sublayer(LN(x))`, so
each sublayer reads a normalized input but writes its raw output into an unnormalized stream whose
through-path is identity-plus-addition again. The leading `1` is restored, the backward highway is clean,
and the last-layer gradient now shrinks like `1/√L` instead of staying large and depth-independent. The
price is that the stream is no longer reset to a fixed scale between layers, so its expected squared norm
grows with depth — which is precisely why a single final LayerNorm before the head is needed, to soak up
that accumulated scale before the logits. So Pre-LN is not a strawman: it is the strongest *plain*
residual rule, the one the field actually trains with, and that is exactly what makes it the honest
floor. Any depth-flow redesign on this ladder has to beat the best simple thing, not a deliberately
crippled one. And note what Pre-LN did *not* fix even as it fixed the gradient highway — the very things
its own construction introduced — because those are the seams the rest of the climb will pull on.

Now I should name the one property of this floor that every later rung will attack, because it is
load-bearing for the whole climb. The Pre-LN residual stream is a *fixed unit-weight accumulator*. Read
the unrolled recurrence: with `x_{l+1} = x_l + F_l(LN(x_l))`, the state entering layer `l` is
`x_l = x_1 + Σ_{i<l} F_i(LN(x_i))` — the embedding plus every earlier branch output, each added with
coefficient exactly one. Three things follow, and they are the gaps the ladder will exploit. First, the
depth-mixing rule is *rigid*: the sequence axis gets full self-attention, the feed-forward gets a
learned nonlinearity, but the depth axis gets one unweighted running sum with no knob — no way to say
"layer 9 should lean on layer 3's output more than layer 8's," and the same mixing for every token.
Second, the stream variance *climbs with depth*: each branch reads a normalized input of fixed scale
and writes its raw output into an unnormalized stream, so the accumulated norm grows (linearly in the
benign case, faster in the bad one), and because the branch's LayerNorm divides by that growing scale,
the deep blocks' Jacobians shrink toward the identity — the deepest layers become near-identity maps
that contribute little, the redundancy that depth-pruning probes keep finding. Third, the branch is at
full strength from step zero: the block is *not* the identity at init, so the early, chaotic updates of
a freshly-random deep stack get written into the stream at full force, which is precisely the
instability that learning-rate warm-up is a band-aid over.

Let me press on each of those three gaps a little, because they are not equally important and I want to
know which one to attack first. The rigidity gap is the most fundamental: the depth axis is the only
axis of this network with *no* learned mixing. The sequence axis has self-attention, which lets every
position read a learned, content-dependent combination of every other position. The feature axis has the
MLP, a learned nonlinearity. The depth axis has a single unweighted running sum — no coefficients, no
content-dependence, the same accumulation for every token and every layer. That is a striking asymmetry
once I notice it: I have lavished learned, dynamic mixing on two of three axes and left the third frozen.
The variance gap is the most *mechanical*: it is a measurable, monotone drift that I can in principle
counteract with even a crude per-layer reweighting, and it is the proximate cause of the deep-layer
death. The init-strength gap is the most *temporal*: it is about *when* in training the branches are
allowed to write at full force, and it is the reason warm-up exists at all. These three are not
independent — a fix that holds the deep branches back early (temporal) also slows the variance climb
(mechanical) — which suggests the cheapest first move is one that touches *when and how much* each branch
writes, since that single lever bears on two of the three gaps at once without needing any new learned
machinery. That is the move I will reach for at step 2: not yet the full learned depth-mixing the
rigidity gap ultimately demands, but the smallest controlled perturbation of the accumulator that the
mechanical and temporal gaps invite, so I get a clean single-variable reading against whatever this floor
scores.

So the vanilla baseline is the weakest rung *by construction*: it has the cleanest backward highway the
plain residual can offer, and no mechanism at all for weighting, scheduling, or re-routing the depth
flow. It will train fine — Pre-LN GPT-2 Medium on 7B tokens is a well-trodden recipe, and I expect a
healthy validation loss in the low-2.2s, sensible WikiText-2 and LAMBADA perplexities, and downstream
accuracies in the mid-50s on ARC-Easy and low-30s on HellaSwag, since at this scale the model is barely
above chance on the harder zero-shot tasks. What it will *not* do is exploit the residual stream as a
designable object, because there is nothing in the default loop that touches it. That is the floor the
next rung has to clear: keep the clean Pre-LN highway, but stop treating the depth axis as a fixed
unit-weight sum — give the residual flow a knob. The first move I will reach for is the cheapest
possible one that changes *when and how much* each branch writes to the stream without adding real
parameters, so that whatever the vanilla numbers turn out to be, I have a controlled, single-variable
delta to read against them. The distilled module and the literal scaffold fill are in the answer.
