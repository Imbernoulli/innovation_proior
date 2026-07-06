The random forest came in at 0.9346 on random, 0.8536 on monotone, 0.9312 on sparse — geometric mean
0.9057, a clear gain over deep_dnf's 0.8532, and the gain landed exactly where I bet it would. The
random family jumped from 0.7605 to 0.9346, a swing of +0.174: the exact axis-aligned splits did not
suffer the noisy-OR's union-of-errors, and mixed polarity cost the tree nothing, just as predicted. In
earned-accuracy terms that is even starker — deep_dnf cleared the 0.5034 random base rate by 0.257, the
forest clears it by `0.9346 − 0.5034 = 0.431`, so the forest recovered nearly 87% of the gap to a perfect
classifier where deep_dnf recovered barely half. That vindicates the diagnosis — deep_dnf's random
weakness was the aggregation, not the representation. But the forest paid for it on monotone, which
*dropped* from deep_dnf's 0.9088 to 0.8536, the worst single number in the whole ladder so far, and a
loss of 0.055. That is the s=20 coverage effect I flagged: a wide target with 20 width-4 terms over 40
variables is a lot of conjunctions to cover, and with `sqrt(40) ≈ 6` random variables searched per node,
a specific term's four variables appear along a single path only rarely, so the union of 200 trees' paths
under-covers the term set and the ensemble leaves real terms unmodeled. The decorrelation that rescued the
random family is hurting the wide monotone family — exactly the `1 − (1−q)^{200}` coverage shortfall I
estimated, where two or three under-assembled terms cap the accuracy in the mid-0.80s. Sparse came in at
0.9312 — solid, and notably *higher* than monotone, the inversion I warned about (the easier-in-principle
junta beating the harder-looking wide DNF), but not the disaster the random-subset-misses-the-junta
argument feared; the 81%-coverage figure I computed was about right, the 19% of noise-splits cost a little
but not much. And it was cheap: the forest fits in about 0.67 seconds a family against deep_dnf's 12-to-38,
a 20-to-50x speedup, because there is no candidate mining and no gradient loop. So the forest is a
genuinely stronger learner than the bespoke DNF net, yet it has a structural ceiling on wide targets, and
its best number anywhere is 0.9346 — still well short of the near-perfect accuracy 20000 examples of a
width-4 concept ought to permit.

That gap is what sends me back to the model the differentiable approach abandoned in the first place: a
plain feed-forward network trained by backpropagation. The forest's limitation is that it bags
*independent* trees with *axis-aligned* splits and a *fixed* feature-subset rule; on a wide DNF it cannot
flexibly allocate capacity to cover all the terms, because the `sqrt(n)`-per-node rule is a hard
bottleneck that no amount of trees fully removes — it only unions many thin coverages. A fully-connected
MLP has none of those constraints — every hidden unit sees every input, the units jointly learn a
distributed re-coding of the inputs, and with enough hidden units there is *some* recoding that makes any
Boolean target linearly separable at the output. The reason I did not start here is the legibility worry
that drove deep_dnf: an MLP smears the learned function across thousands of real-valued weights with no
procedure to read off "which variables, which polarity." But this task does not score legibility — it
scores held-out accuracy. The only thing that matters is whether the network *fits and generalizes* the
hidden DNF from 20000 uniform examples, and on that axis the MLP's lack of structural commitment is an
asset, not a liability: it is free to carve the input space however the data demand, with no
`sqrt(n)`-per-node bottleneck and no union-of-soft-conjunctions accumulation.

Let me reconstruct why the MLP can represent these targets, because the whole bet rests on it, and I want
the construction exact rather than gestured at. A single linear-threshold unit fires on one side of a
hyperplane — it can do an AND or an OR of literals directly. For a conjunction, set the weight to +1 on
each positive literal's variable, −1 on each negated literal's variable, and choose the bias so the
pre-activation is positive only when every literal is satisfied. Take a width-4 all-positive term: weights
+1 on those four variables, bias −3.5, so the pre-activation is `x_a + x_b + x_c + x_d − 3.5`, which is
`4 − 3.5 = 0.5 > 0` when all four are on and `3 − 3.5 = −0.5 < 0` the instant any is off — the unit is
that term. Mixed polarity is just as cheap: for each negated literal `¬x_j` use weight −1 and add 1 to
the bias, so a width-4 term with `k` positive and `4−k` negative literals fires exactly when its `k`
positives are on and its `4−k` negatives are off, at pre-activation threshold `k − 0.5`. Then the output
unit OR's the term-units together — fire if *any* hidden unit fires — which is again a single
linear-threshold decision: sum the hidden activations, threshold at ≥ 1. So a two-layer net with one
hidden unit per DNF term represents the target *exactly*: hidden layer = the conjunctions, output = the
disjunction. With 256 hidden units and at most 20 terms, the network has `256/20 ≈ 12.8×` more units than
the monotone family needs, and `256/10 ≈ 25.6×` on the narrower random family — enormous slack, which
should give gradient descent room to find *a* fitting configuration rather than needing to discover the
exact minimal one. It is worth pricing the whole network too: the first layer is `n×256 + 256`, the second
`256×256 + 256 = 65792`, the head `256 + 1`, so on random `30·256 + 256 + 65792 + 257 = 73985` parameters,
on monotone `40·256 + 256 + 66049 = 76545`, on sparse `60·256 + 256 + 66049 = 81665` — about 74k-to-82k
weights, an order of magnitude more than deep_dnf's 3.6k-to-9.7k. The representational question is settled;
the only question is whether gradient descent *finds* such a configuration from random examples, and that
is what backpropagation is for.

There is a subtlety I should be honest about, because it is the crux of whether full connectivity actually
*helps* on the wide family. Representability is not findability. Knowing that some 20-unit configuration
computes the monotone target does not mean the optimizer lands on it: with 256 units and a random
initialization, many units start in flat ReLU regions or chase the same dominant terms, and the gradient
signal for a rarely-satisfied conjunction is weak. Put a number on "rarely": a width-4 term is satisfied
by `1/16 = 6.25%` of uniform inputs, so in a batch of 256 only about 16 examples satisfy any given term,
and those 16 are the *only* points carrying gradient information about that term's precise boundary. Over
an epoch of `20000/256 ≈ 78` batches, and 20 epochs, that is enough exposure in principle, but the loss at
any step is dominated by the frequently-firing structure and the easy negatives, so the thin 16-example
signal for a hard term is easily drowned. That tension — full capacity but diffuse gradient on rare terms —
is exactly what makes the wide monotone family the interesting test, and it is why I am only *moderately*
confident the MLP recovers it rather than certain. Against the forest, though, the comparison is favorable
on its face: the forest's monotone failure was a *coverage* bottleneck (the right variables never lined up
along a path), and the MLP has no such bottleneck at all — every one of its 256 units sees all 40
variables at once, so allocating 20 units to 20 terms is a matter of the optimizer *finding* them, not of
whether the architecture *can reach* them. I am trading a hard structural ceiling for a softer
optimization difficulty, and I expect that trade to help on monotone.

The training recipe is the scaffold default, and each piece is load-bearing. The unit is ReLU, not a hard
threshold: a step unit has zero gradient almost everywhere, so a gradient method has nothing to descend,
whereas ReLU is differentiable (piecewise) and lets a small weight change produce a measurable change in
the loss — and note the exact-representation construction above used hard thresholds, so training relies
on ReLU as the trainable surrogate that can *approach* those threshold units as weights grow. The
objective is binary cross-entropy on the output logit — the natural loss for a 0/1 target, far
better-conditioned for classification than squared error, and its gradient at the output is just the
signed probability error, which backpropagation then chains through the network: the error's sensitivity to
each weight is the upstream sensitivity times the local ReLU slope, computed in one backward sweep that
reuses the forward activations. The cross-entropy choice matters more than it looks on this task: squared
error on a saturated sigmoid has a gradient that scales with `σ'(z) = σ(z)(1−σ(z))`, which vanishes
precisely when the model is confidently wrong (output saturated at the wrong end), so it would stall
exactly on the rarely-satisfied terms I am most worried about; cross-entropy's gradient is the raw
probability error `σ(z) − y`, which stays order-1 on a confident mistake, so the optimizer keeps pushing on
the hard terms instead of giving up on them. That is not a cosmetic difference — on a family whose whole
difficulty is a thin gradient from rare terms, keeping that gradient from vanishing is the ballgame. Two
hidden layers of 256 give depth (so the network can compose features, not just take one linear recoding)
and width (so it has slack to find a good configuration rather than needing the exact minimal one). AdamW
adapts the per-parameter step and decouples weight decay (1e-4) as a clean L2 pull toward small weights —
mild regularization against overfitting the 20000 points, which matters because, like the forest, the MLP
sees a vanishing fraction of the Boolean cube. Twenty epochs over the 20000 examples at batch 256 and lr
1e-3 is enough passes for the loss to settle without overtraining. Threshold the output sigmoid at 0.5 for
the 0/1 prediction.

I should reason quantitatively about the sparse family in particular, because it is where the MLP's lack of
built-in variable selection is a real liability and I want to know whether weight decay closes the gap. On
sparse, 48 of the 60 input coordinates are pure noise, yet the first layer wires all 60 into every one of
the 256 hidden units — `60·256 = 15360` first-layer weights, of which `48/60 = 80%` connect to irrelevant
variables. A tree gets junta-irrelevance for free (it simply never finds a good split on a noise variable),
but the MLP must actively *learn* to zero those 12288 noise weights. The mechanism is weight decay: AdamW's
decoupled decay of 1e-4 applies a constant multiplicative pull `w ← w·(1 − lr·λ)` toward zero every step,
and a weight that receives no consistent gradient signal — which a noise-variable weight does not, since it
carries no label information — decays geometrically while the signal-carrying weights are held up by their
gradient. Over `78 batches × 20 epochs = 1560` steps the decay factor compounds to `(1 − 10^{-3}·10^{-4})`
per step, which is tiny per step but the point is directional: noise weights drift down, signal weights are
replenished, so the network gradually concentrates its first-layer mass on the 12 relevant coordinates. It
is not as clean as a tree's hard exclusion — some capacity leaks onto noise before decay suppresses it, and
the leak is what could cost the MLP a point or two versus the forest on sparse — but it is a real selection
pressure, not nothing. So my sparse prediction is "competitive, possibly a touch behind the forest," and the
mechanism is explicit: decay-driven soft selection racing against capacity wasted on noise.

The epoch budget is worth checking against the rare-term concern rather than taking 20 on faith. Each epoch
is one full pass, 78 batches; 20 epochs is 1560 gradient steps. A specific monotone term's boundary is
informed by its ~16 satisfying examples per batch, so across 1560 steps that term's positive region is
touched on the order of `1560 × 16 ≈ 25000` example-visits — plenty of raw exposure. The risk is not too
few visits, it is that those visits produce a *small* loss contribution once the easy structure is fit, so
the effective learning on the hard term stalls even though the data keeps arriving. That is precisely why
the cross-entropy-versus-squared-error argument matters: with cross-entropy the confident-mistake gradient
does not vanish, so those 25000 visits keep doing work rather than fading once the sigmoid saturates. If 20
epochs turns out to be too few, the symptom would be a monotone number that improves with more epochs — a
diagnosable knob — rather than a hard ceiling, which distinguishes an optimization shortfall from a
representational one and is worth remembering when I read the result.

There is a design detour worth naming and rejecting, because the obvious "improvement" would be to widen or
deepen the network to force the monotone terms in. If diffuse gradient on rare terms is the problem, why not
go to 1024 units, or four layers? I walk it: more units do not raise the per-term gradient — each hidden
unit still sees the same 16-in-256 positive rate for a given term — they only add more redundant units
chasing the same easy structure, and more depth risks the rare-term signal getting diluted through more
layers of ReLU gating before it reaches the units that could specialize. The capacity is already `12.8×`
over-provisioned; the bottleneck is *optimization*, not capacity, so throwing capacity at it is the wrong
lever and would mostly cost compute and overfitting room. Staying at the scaffold's 256×256 keeps the
comparison clean: this is the *generic* neural baseline, with no DNF-specific machinery at all, being asked
to beat both the hand-shaped DNF net and the tree ensemble. That it is also the cheapest edit on the
ladder — it is the template itself — is part of the finding. The full module is in the answer.

Now the falsifiable expectations against the forest's numbers, family by family. On **monotone** (rf 0.8536,
the forest's weak spot), I expect the MLP to *clearly beat* the forest. The forest's failure there was
structural — random feature subsets under-covering 20 wide terms — and the MLP has no such bottleneck:
every hidden unit sees all 40 variables, so allocating 20 of its 256 units to the 20 terms is trivially
within reach, and the distributed representation can even share structure across terms that overlap on
variables. This is the comparison that should most cleanly favor the MLP, and if it does not materialize my
coverage-bottleneck story for the forest is wrong. On **random** (rf 0.9346), I expect the MLP to be
*competitive or slightly better*: mixed polarity is as free for a hidden unit (negative weights) as it is
for a tree split, and 10 terms over 30 variables is a narrow enough target that 256 hidden units with a
`25.6×` capacity margin should fit it nearly exactly; I would expect the random number up near or above the
forest's 0.93, plausibly close to solved. On **sparse** (rf 0.9312), the MLP has to *learn* that 48 of the
60 variables are irrelevant rather than getting it for free the way a tree does by never splitting on them;
weight decay helps push the irrelevant-variable weights toward zero, but there is a real risk the MLP wastes
capacity on noise variables and lands a touch below the forest's sparse number. So sparse is the one family
where I would not be surprised to see the MLP trail — the mirror image of the forest, which got junta
irrelevance for free but paid on wide coverage.

It is worth seeing why monotone is the lever, mechanically, through the geometric mean. The forest's geomean
is `(0.9346 · 0.8536 · 0.9312)^{1/3} = 0.9057`, and the derivative of a geometric mean with respect to its
smallest factor is the largest — `∂G/∂x_i = G/(3 x_i)`, so the term with the smallest `x_i` moves `G` the
most per point gained. With monotone at 0.8536 that sensitivity is `0.9057/(3·0.8536) = 0.354` per unit,
against `0.323` for random at 0.9346; a point recovered on monotone buys more geomean than a point on
either near-ceiling family. Concretely, lifting monotone from 0.8536 to, say, the low-0.90s while holding
random and sparse flat would move the geomean by roughly `0.354 × 0.05 ≈ 0.018`, clearing the forest — so
the entire case for the MLP rests on the monotone recovery, and random/sparse only need to *not regress*.
That is why I am staking the prediction on the one family where full connectivity should most help, and why
a monotone that fails to move would sink the whole rung regardless of what happens elsewhere.

The prediction I am committing to: the geometric mean should rise above the forest's 0.9057, driven mainly
by recovering the monotone family the forest dropped, with random roughly flat-to-up and sparse the only
place the MLP might give ground. If instead the MLP fails to beat the forest on *monotone* — the family
where its full-connectivity advantage should be largest — then my account of the forest's weakness
(feature-subset under-coverage of wide DNF) is wrong and I need a different story, most likely that wide
monotone DNF is simply hard to fit for *any* flat one-pass learner. And if the MLP, despite full
connectivity and ample capacity, still cannot push any family to near-perfect accuracy, the remaining lever
is the one the forest left on the table: stop *bagging* independent learners and start *boosting* — fit each
new weak learner to the residual errors of the ensemble so far, so later learners explicitly correct
earlier mistakes. A gradient-boosted tree ensemble keeps the tree's exact conjunctive splits (which beat
the MLP's smeared representation on the random family) but adds the sequential error-correction the random
forest lacks, and that combination is the natural next rung.
