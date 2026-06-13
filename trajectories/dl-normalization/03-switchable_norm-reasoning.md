Group normalization did exactly what I bet it would, and the numbers say so in the place I most wanted to
see movement: ResNet-56 climbed from the blend's 66.06 to 67.90, ResNet-110 from 68.65 to 70.43 — clear
recovery on the two deep CIFAR-100 stacks, and the larger absolute gain on the deeper ResNet-110, just as I
expected from a stable per-image group statistic helping most across many blocks. The one place it did
*not* gain was MobileNetV2 / FashionMNIST, which slipped from 93.64 to 93.16 — and that slip is the tell I
need to read carefully, because it points directly at what group normalization left on the table. The
reason is built into the rung's own adaptive grouping rule. To stay legal on the small, awkwardly factored
channel counts in MobileNetV2's inverted residuals, the rule caps groups at the channel count and
decrements until divisibility, so on a 16-channel or a 32-channel layer it collapses to one channel per
group — which *is* instance normalization on that layer. So on exactly the network where many layers are
narrow, group normalization quietly reverts to the instance-equivalent boundary I was trying to escape, and
inherits its under-sharing. More generally, the rung made one structural commitment: a *single fixed*
grouping and a *single fixed* pooling rule, identical at every layer, chosen by a hand-written heuristic
rather than by the task. The deep ResNets rewarded that commitment; the narrow MobileNetV2 layers were
penalized by it. The headroom, then, is in the rigidity itself — different layers, and even different
channels, may want different statistics, and group normalization gives them no way to express that.

So the question for this rung is: instead of *fixing* which statistic each layer uses, can I let the layer
*learn* it? The family I have been walking is already three points in one space — batch, layer, instance,
each a different membership set `S_i`. Group normalization picked a fixed interior point on the
channel-sharing dial between layer and instance. But there is no reason a single layer should be forced to
commit to one of these; the most flexible thing is to compute *all three* standardizations and let the
layer learn a weighting over them, end to end from the classification loss. That is a strict
generalization of everything so far: if the learned weights collapse onto batch, the layer is batch
normalization; onto instance, instance; onto layer, layer; and any soft combination in between is a new
operating point that no single fixed rule could reach. Crucially, the *weighting* is the thing being
learned, not a hand-set hyperparameter like group count — which is precisely the rigidity that cost group
normalization on MobileNetV2. The decrement heuristic that collapsed narrow layers to instance-equivalent
would be replaced by a continuous knob the optimizer turns wherever it helps.

Let me build it carefully against the family notation, because the three statistics are *not* independent
quantities I have to recompute from scratch — they are nested reductions of the same feature map and I want
to be precise about which axes each one spans. Take `x` of shape `[B, C, H, W]`. The **instance** statistic
is per `(image, channel)`: `mean_IN = x.mean over (2,3)`, `var_IN = x.var over (2,3)`, the spatial map of
each channel alone. The **layer** statistic is per image: `mean_LN = x.mean over (1,2,3)`,
`var_LN = x.var over (1,2,3)`, all channels and all spatial of one image. The **batch** statistic is per
channel: `mean_BN = x.mean over (0,2,3)`, `var_BN = x.var over (0,2,3)`, that channel across the whole
batch and grid. All three with the biased (divide-by-`m`) variance, consistent with the standardization.
These are the three corners; the layer's job is to mix them.

Now the mix. I keep two small sets of learnable weights — one for the mean, one for the variance — each a
3-vector over `{IN, LN, BN}`, passed through a softmax so they form a convex combination that sums to one:
`mean_w = softmax(mean_weight)`, `var_w = softmax(var_weight)`, and the combined statistics are
`mean = mean_w[0]·mean_IN + mean_w[1]·mean_LN + mean_w[2]·mean_BN` and likewise for `var`. Standardize by
the *combined* mean and variance, `x̂ = (x − mean) / sqrt(var + ε)`, then one per-channel affine
`γ x̂ + β`. Two design points here are deliberate and worth pinning down, because they decide what this rung
*is* versus the more elaborate forms the idea can take. First, I learn **separate** weights for the mean
and the variance rather than one shared weighting. The reason is that the right *centering* and the right
*scaling* need not come from the same statistic: a channel might want batch-level centering (to preserve
between-image offset) while wanting instance-level scaling (to wash out per-image contrast magnitude), and
two weight vectors let the layer express that decomposition. Second — and this is the choice that pins this
rung to *this task's edit surface* — the weights are **per-layer, shared across all channels**, a single
3-vector each, not a per-channel weighting. The task's contract gives the layer `nn.Parameter(torch.ones(3))`
for the mean and the same for the variance: three numbers per layer, softmaxed, applied identically to
every channel. That is a much lighter footprint than a per-channel gate — the previous rung's blend
carried `C` gate parameters per layer, this one carries just `6` — and it makes the comparison clean: the
flexibility being added over group normalization is *which corner of the family this layer leans toward*,
decided globally for the layer, not a per-channel reshaping. The softmax is what makes the weights a true
selection: it forces them non-negative and summing to one, so the combination is a genuine convex
interpolation among the three normalizations rather than an arbitrary linear combination that could amplify
or cancel statistics in ways that destabilize training.

I need to handle the batch statistic the same honest way I did in the earlier rungs, and it is worth being
explicit because it carries the one residual risk on this rung. The textbook batch layer keeps a running
mean/variance frozen for inference; the task's contract computes the batch statistic *directly from the
current batch* in `forward`, in both train and eval, with no running buffer. At batch 128 in both phases
this is stable, and it keeps the layer a single consistent function with no hidden state to accumulate —
the same decision I made at rung one and the same one group normalization sidesteps by never touching the
batch axis. The residual risk is that the BN corner of the mix still inherits this on-the-fly batch
dependence: to whatever extent the learned `mean_w[2]` / `var_w[2]` lean on the batch corner, the layer's
eval behavior depends on the eval batch's composition. But the softmax mix can *learn its way out* of that
— if leaning on batch hurts, the optimizer can shift weight onto the layer or instance corners, which are
batch-free — and that adaptivity is exactly the point of this rung. So the construction is: compute the
three pairs of moments over their respective axes, softmax the two weight 3-vectors, form the convex-mixed
mean and variance, standardize, and apply the per-channel affine. No running buffers, `ε = 1e-5`, and the
affine `weight`/`bias` initialized to 1 and 0 as the family expects; the two weight 3-vectors initialized
to all-ones so the softmax starts uniform — every layer begins as an equal blend of the three corners and
gradient descent slides each layer toward whatever corner, or interior point, the loss prefers.

Why should this beat group normalization, and on which settings specifically? The argument is a
generalization argument: switchable normalization *contains* the previous rungs as special cases of its
learned weights, so in principle it can recover group normalization's good behavior on the deep ResNets by
leaning toward the layer/instance corners that group normalization interpolated, while *also* doing
something group normalization could not on the narrow MobileNetV2 layers — instead of being forced by the
decrement heuristic onto the instance-equivalent boundary, it can learn to lean on the **batch** corner
there, which group normalization never had access to (group normalization is strictly batch-free). On
MobileNetV2 / FashionMNIST, where the comfortable batch of 128 makes the batch statistic genuinely useful
and group normalization's narrow-layer collapse hurt it, recovering the batch corner per layer is exactly
the missing degree of freedom. So my falsifiable expectation against the prior numbers is concrete. On
MobileNetV2 / FashionMNIST — group normalization's weak point at 93.16 — I expect switchable normalization
to *recover and exceed*, because it can dial in the batch statistic that the narrow layers were denied;
this is the setting where I expect the clearest gain over the previous rung. On the two CIFAR-100 ResNets —
group normalization at 67.90 and 70.43 — I expect it to *match or modestly exceed*, because the learned mix
can reproduce a layer/instance-leaning blend close to what group normalization already found and add only
the marginal benefit of letting centering and scaling pick different corners; I would not expect a large
jump there, since group normalization was already well-suited to the deep stacks. The risk that would
falsify the bet is the on-the-fly batch dependence in the BN corner destabilizing the deep ResNets — if the
optimizer over-weights batch on a deep stack and the eval-batch coupling bites, ResNet-110 could come in
*at or below* group normalization rather than above. If switchable normalization clears group normalization
across the board — and especially if it reclaims the MobileNetV2 setting — it confirms that the productive
move at the top of this ladder is to stop fixing the statistic per layer and *learn* the convex combination
of the family's corners. The distilled module that lands this rung is in the answer.
