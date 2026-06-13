The reshape-CNN settled the first question and opened the second, and again the numbers are explicit.
Going nonlinear rescued ESTA_BACSU exactly as the diagnosis predicted: the stability assay went from
the linear head's −0.107 to 0.641, a swing of three-quarters of a correlation point, which is the
proof that the failure there was representational — a hyperplane that could not bend around the
folding cliff, fixed the moment the head could carve delta-space. And the nonlinear head did not pay
for that rescue elsewhere: BLAT_ECOLX climbed from 0.737 to 0.860 and RASH_HUMAN from 0.681 to 0.798,
so the curvature helped even on the assays that were already approximately linear. So nonlinearity is
not optional — it is worth roughly +0.12 on BLAT, +0.12 on RASH, and the difference between useless
and good on ESTA. That much is now established and I will not give it back.

But the reshape-CNN bundled the nonlinearity together with a second, separate bet — that imposing
*structured weight-sharing* over the embedding features, by reshaping the pooled vector into a fake
`(64, 40)` grid and convolving over it, is a better inductive bias than a plain dense head. And I
flagged going in that this bias has no physical justification: there is no real adjacency along the
embedding axis, the reshape just groups coordinates into arbitrary blocks and shares a filter across
them. The CNN's strong numbers prove the *nonlinearity* paid off, but they do not prove the
*weight-sharing* did — a flat dense head on the same signal might match or beat it with none of the
architectural overhead. That is precisely the question this rung answers: strip out the convolutional
structure, keep only a nonlinear dense head, and see whether the structure was carrying any weight or
was just along for the ride. If a plain MLP matches or beats the reshape-CNN across all three assays,
the lesson is that the fake-grid convolution was a bias the data did not want.

So let me derive the dense nonlinear head from scratch, because I want to understand *why* a hidden
layer is the right object and not just assert it. Start from what the linear head could and could not
do. A linear readout `x'w` answers with one side of a hyperplane in feature space — it can only
express a monotone gradient along a single direction of the delta. The ESTA −0.107 was that ceiling
made visible: stability fitness is a sharply curved, threshold-like function of the embedding shift
(tolerated until a cliff, then dead), and there is no single direction along which that is monotone,
so the linear fit ranked mutations roughly backwards. The general statement is that whenever the
similarity structure of the inputs and the required outputs disagree — nearby deltas demanding very
different fitness because one crossed the cliff and one did not — a single hyperplane cannot bridge
the gap. The cure is to change the space the decision is made in: insert a layer of hidden units
between the input and the readout, so the readout no longer sees the raw delta but whatever those
hidden units compute. With the right hidden features a curved mapping becomes a simple readout — the
hidden layer *recodes* the input into a space where the fitness function is easy to read off, and with
enough hidden units some such recoding always exists. That is the entire reason the reshape-CNN's
nonlinearity rescued ESTA, and a dense hidden layer delivers the same recoding without committing to
any spatial story about the coordinates.

Now the two requirements on the hidden unit's activation, because the choice decides whether the
thing even trains. First it must be **nonlinear**, and I want to nail that rather than assume it: if
the hidden units were linear, then `(delta · W1) · W2 = delta · (W1 W2)`, the two layers collapse to
a single equivalent linear map, and I am back to the hyperplane ceiling that gave me −0.107. The
nonlinearity is the only thing that makes the hidden layer more than decoration. Second it must be
**differentiable**, because the fixed loop trains by gradient descent (AdamW on the MSE loss), and a
step function has zero gradient almost everywhere — nothing to descend. So the unit has to be smooth
and nonlinear. The classical choice is the logistic, whose slope `y(1−y)` is cheap, but for a
regression head the rectified linear unit, ReLU, is the better modern default: it is nonlinear and
differentiable (almost everywhere), it does not saturate for large positive inputs the way the
logistic does, so gradients do not vanish in the active region, and it is exactly what carved the
delta-space inside the reshape-CNN's blocks. The hidden representation `ReLU(delta · W1 + b1)` is a
piecewise-linear recoding — each hidden unit is an oriented ramp in delta-space, and a layer of them
tiles the space into regions, which is precisely the machinery needed to place a stability cliff.

The learning is the part the textbook delta rule cannot do alone, and it is worth seeing why the loop
handles it. The output unit has a target — the fitness score — so its error signal is just
prediction-minus-target. The hidden units have *no* target; the data never says what a hidden unit
should output. The way out is that gradient descent does not need a target for a hidden unit, only the
derivative of the loss with respect to its weights, and the chain rule manufactures that by pushing
the output error backward through the same weights it flowed forward through, gated by each unit's
local slope — backpropagation. The fixed loop's autodiff does this for me; I do not hand-write it. So
the entire head reduces to: `delta → Linear → nonlinearity → Linear → scalar`, trained by the loop's
backpropagated MSE gradient. The point of deriving it is to be sure the dense head is the *minimal*
object that buys the curvature — one hidden layer with a nonlinearity, nothing structural bolted on.

Now the design decisions specific to this task's edit surface, and again the README's warning that a
same-named baseline is not the paper applies: this is not the manual-backprop, logistic-unit,
momentum-SGD MLP of the classic derivation, and it is not a deep residual MLP. It is one hidden layer,
trained by the loop's AdamW, and a few concrete choices matter. **Input.** I go back to
`delta_embedding` alone, not the concatenated `[embedding, delta]` the reshape-CNN used. The reason is
the same one that made the linear head read the delta: the raw embedding's bulk is the protein's
constant identity, which carries no within-assay ranking signal, and feeding it to a single hidden
layer mostly adds nuisance dimensions the layer has to learn to ignore. The delta already isolates
what the mutation did, and a single hidden layer on the delta is a clean, well-conditioned nonlinear
readout. If the dense delta-only head matches the both-inputs CNN, that is itself informative — it
says the raw embedding was not pulling weight either. **Width.** One hidden layer of 512 units —
comfortably wide enough to tile a 1280-d delta-space into enough regions to place a cliff, while
staying a single layer so I am testing "plain nonlinear head" and not sneaking in depth. **Dropout.**
A dropout of 0.1 on the hidden representation as a regularizer; with only ~2000–4800 mutants per assay
and a 512-wide layer, some regularization against memorizing the training fold is prudent, and the
loop already early-stops on validation Spearman, so dropout plus early stopping together guard
generalization across folds. One implementation quirk worth naming because it is the literal edit:
the dropout is applied to the *pre-activation* (`fc1` output) and then ReLU, i.e. `Linear → Dropout →
ReLU → Linear`, rather than the more common post-activation dropout. For ReLU the two orderings are
nearly equivalent up to which side of the threshold gets zeroed, and it is the order the scaffold fill
uses; I keep it as-is. **Optimizer knobs.** I leave `learning_rate` and `weight_decay` at the loop
defaults — a single hidden layer does not need the heavy weight decay the linear ridge head needed,
because the nonlinearity, the dropout, and early stopping are already controlling capacity. The
distilled module is in the answer.

So the head is deliberately the *least* structured nonlinear thing that could work:
`delta_embedding → Linear(1280, 512) → Dropout(0.1) → ReLU → Linear(512, 1)`, trained by the fixed
loop. Every difference from the reshape-CNN is a *removal*: no concatenation of the raw embedding, no
reshape into a fake grid, no convolutional weight-sharing, no residual blocks, no global pooling —
just a dense hidden layer. That is what makes it a clean controlled comparison: if it matches or beats
the CNN, the convolutional structure was inert.

Now the falsifiable expectations against the reshape-CNN's measured numbers, since that is the rung I
am being compared to. The CNN posted 0.860 on BLAT, 0.641 on ESTA, 0.798 on RASH. My prediction is
that the plain dense head will *match or slightly exceed* all three, because the only thing the CNN's
numbers proved was that the nonlinearity helps, and the dense head keeps the nonlinearity while
dropping a structural bias that had no physical grounding. Concretely I expect ESTA to stay rescued —
it should land in the same low-0.6s as the CNN, since both can carve the cliff and the dense head
carves it just as well — and I expect BLAT and RASH to edge *up* past the CNN, into the high-0.8s and
high-0.7s respectively, because removing the meaningless fake-grid convolution should remove a small
amount of inductive-bias noise and let the dense head fit the approximately-linear-plus-curvature
signal more directly. The risk, and the thing that would falsify this, is the opposite: if the dense
head comes back *below* the CNN on any assay, then the convolutional weight-sharing was actually
buying something — perhaps the reshape-and-share regularized the head in a way that helped
generalization across the small folds — and the structured bias would deserve to stay. The single
most diagnostic comparison is BLAT, where the CNN's 0.860 is the strongest baseline number on the
board: if the dense head clears it, the plain nonlinear head is the strongest of the three and the
lesson of the whole ladder is clean — go nonlinear (it rescues the cliff and lifts the linear
assays), but do not impose structure the embedding does not have. If the dense head falls short
there, the convolution earned its keep and the ladder ends on the structured head instead.
