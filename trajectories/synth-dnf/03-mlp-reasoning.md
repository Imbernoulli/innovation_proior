The random forest came in at 0.9346 on random, 0.8536 on monotone, 0.9312 on sparse — geometric mean
0.9057, a clear gain over deep_dnf's 0.8532, and the gain landed exactly where I bet it would. The
random family jumped from 0.7605 to 0.9346: the exact axis-aligned splits did not suffer the noisy-OR's
union-of-errors, and mixed polarity cost the tree nothing, just as predicted. That vindicates the
diagnosis — deep_dnf's random weakness was the aggregation, not the representation. But the forest paid
for it on monotone, which *dropped* from deep_dnf's 0.9088 to 0.8536, the worst single number in the
whole ladder so far. That is the s=20 effect I flagged: a wide target with 20 width-4 terms over 40
variables is a lot of conjunctions to cover, and with `sqrt(40) ≈ 6` random variables searched per node,
many trees never get the right variable at the right depth, so the union of paths under-covers the term
set and the ensemble leaves real terms unmodeled. The decorrelation that rescued the random family is
hurting the wide monotone family. Sparse came in at 0.9312 — solid, and notably *higher* than monotone,
the inversion I warned about (the easier-in-principle junta beating the harder-looking wide DNF), but
not the disaster the random-subset-misses-the-junta argument feared. So the forest is a genuinely
stronger learner than the bespoke DNF net, yet it has a structural ceiling on wide targets, and its best
number anywhere is 0.9346 — still well short of the near-perfect accuracy 20000 examples of a width-4
concept ought to permit.

That gap is what sends me back to the model the differentiable approach abandoned in the first place: a
plain feed-forward network trained by backpropagation. The forest's limitation is that it bags
*independent* trees with *axis-aligned* splits and a *fixed* feature-subset rule; on a wide DNF it cannot
flexibly allocate capacity to cover all the terms. A fully-connected MLP has none of those constraints —
every hidden unit sees every input, the units jointly learn a distributed re-coding of the inputs, and
with enough hidden units there is *some* recoding that makes any Boolean target linearly separable at the
output. The reason I did not start here is the legibility worry that drove deep_dnf: an MLP smears the
learned function across thousands of real-valued weights with no procedure to read off "which variables,
which polarity." But this task does not score legibility — it scores held-out accuracy. The only thing
that matters is whether the network *fits and generalizes* the hidden DNF from 20000 uniform examples,
and on that axis the MLP's lack of structural commitment is an asset, not a liability: it is free to
carve the input space however the data demand, with no `sqrt(n)`-per-node bottleneck and no
union-of-soft-conjunctions accumulation.

Let me reconstruct why the MLP can represent these targets, because the whole bet rests on it. A single
linear-threshold unit fires on one side of a hyperplane — it can do an AND or an OR of literals directly
(weights ±1 on the relevant variables, a bias at the right count threshold), so it can already represent
*one* DNF term as a single hidden unit: set the weights to +1 on the positive literals and −1 on the
negated ones, bias so the unit fires only when all `w` literals are satisfied — for a width-4 positive
term, weights +1 on those four variables and a bias of −3.5 makes the pre-activation positive only when
all four are on. Then the output unit OR's the term-units together — fire if *any* hidden unit fires —
which is again a single linear-threshold decision (sum the hidden activations, threshold at ≥ 1). So a
two-layer net with one hidden unit per DNF term represents the target *exactly*: hidden layer = the
conjunctions, output = the disjunction. With 256 hidden units and at most 20 terms, the network has far
more than enough capacity to hold the formula — roughly a 12× over-provisioning on the monotone family,
which should give gradient descent ample slack to find a fitting configuration rather than needing to
discover the exact minimal one. The representational question is settled; the only question is whether
gradient descent *finds* such a configuration from random examples, and that is what backpropagation is
for.

There is a subtlety I should be honest about, because it is the crux of whether full connectivity
actually *helps* on the wide family. Representability is not findability. Knowing that some 20-unit
configuration computes the monotone target does not mean the optimizer lands on it: with 256 units and a
random initialization, many units start in flat ReLU regions or chase the same dominant terms, and the
gradient signal for a rarely-satisfied conjunction is weak — a width-4 term is satisfied by only 1 in 16
uniform inputs, so each term's positive examples are a small slice of the batch, and the loss is
dominated by the easy, frequently-firing structure. So even though the MLP can in principle cover all 20
monotone terms, the practical question is whether 20 epochs of AdamW concentrate enough gradient on the
harder-to-reach terms before the loss flattens. That tension — full capacity but diffuse gradient on rare
terms — is exactly what makes the wide monotone family the interesting test, and it is why I am only
*moderately* confident the MLP recovers it rather than certain.

The training recipe is the scaffold default, and each piece is load-bearing. The unit is ReLU, not a
hard threshold: a step unit has zero gradient almost everywhere, so a gradient method has nothing to
descend, whereas ReLU is differentiable (piecewise) and lets a small weight change produce a measurable
change in the loss. The objective is binary cross-entropy on the output logit — the natural loss for a
0/1 target, far better-conditioned for classification than squared error, and its gradient at the output
is just the signed probability error, which backpropagation then chains through the network: the error's
sensitivity to each weight is the upstream sensitivity times the local ReLU slope, computed in one
backward sweep that reuses the forward activations. The cross-entropy choice matters more than it looks
on this task: squared error on a saturated sigmoid has a gradient that vanishes precisely when the model
is confidently wrong, which would stall exactly on the rarely-satisfied terms I am most worried about,
whereas cross-entropy keeps a strong gradient on confident mistakes, so the optimizer keeps pushing on
the hard terms instead of giving up on them. Two hidden layers of 256 give depth (so the network
can compose features, not just take one linear recoding) and width (so it has slack to find a good
configuration rather than needing the exact minimal one). AdamW adapts the per-parameter step and
decouples weight decay (1e-4) as a clean L2 pull toward small weights — mild regularization against
overfitting the 20000 points, which matters because, like the forest, the MLP sees a vanishing fraction
of the Boolean cube. Twenty epochs over the 20000 examples at batch 256 and lr 1e-3 is enough passes for
the loss to settle without overtraining. Threshold the output sigmoid at 0.5 for the 0/1 prediction.

In the scaffold this is the *default* fill — `build_model` returns the `nn.Sequential` MLP, `make_dataset`
draws the uniform sample, `fit_and_predict` runs the AdamW + BCE loop. It is the cheapest edit on the
ladder (it is the template itself), which is part of the finding: the generic neural baseline, with no
DNF-specific machinery at all, is being asked to beat both the hand-shaped DNF net and the tree
ensemble. The full module is in the answer.

Now the falsifiable expectations against the forest's numbers, family by family. On **monotone** (rf
0.8536, the forest's weak spot), I expect the MLP to *clearly beat* the forest. The forest's failure
there was structural — random feature subsets under-covering 20 wide terms — and the MLP has no such
bottleneck: every hidden unit sees all 40 variables, so allocating 20 of its 256 units to the 20 terms
is trivially within reach, and the distributed representation can even share structure across terms.
This is the comparison that should most cleanly favor the MLP. On **random** (rf 0.9346), I expect the
MLP to be *competitive or slightly better*: mixed polarity is as free for a hidden unit (negative
weights) as it is for a tree split, and 10 terms over 30 variables is a narrow enough target that 256
hidden units should fit it nearly exactly; I would expect the random number up near or above the
forest's 0.93. On **sparse** (rf 0.9312), the MLP has to *learn* that 48 of the 60 variables are
irrelevant rather than getting it for free the way a tree does by never splitting on them; weight decay
helps push the irrelevant-variable weights toward zero, but there is a real risk the MLP wastes capacity
on noise variables and lands a touch below the forest's sparse number. So sparse is the one family where
I would not be surprised to see the MLP trail.

The prediction I am committing to: the geometric mean should rise above the forest's 0.9057, driven
mainly by recovering the monotone family the forest dropped, with random roughly flat-to-up and sparse
the only place the MLP might give ground. If instead the MLP fails to beat the forest on *monotone* —
the family where its full-connectivity advantage should be largest — then my account of the forest's
weakness (feature-subset under-coverage of wide DNF) is wrong and I need a different story. And if the
MLP, despite full connectivity and ample capacity, still cannot push any family to near-perfect
accuracy, the remaining lever is the one the forest left on the table: stop *bagging* independent learners
and start *boosting* — fit each new weak learner to the residual errors of the ensemble so far, so later
learners explicitly correct earlier mistakes. A gradient-boosted tree ensemble keeps the tree's exact
conjunctive splits (which beat the MLP's smeared representation on the random family) but adds the
sequential error-correction the random forest lacks, and that combination is the natural next rung.
