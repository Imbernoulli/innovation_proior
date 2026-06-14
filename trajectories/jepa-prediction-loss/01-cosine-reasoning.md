I start here because the prediction loss is the one knob the task leaves open, and before I reach for
any robust or temporally-weighted refinement I want to fix the most basic ambiguity in matching two
latent feature maps: do I care about their *magnitude* or only their *direction*? The scaffold hands me
`state` and `predicted`, both `[B, C, T, H, W]`, where the channel axis `C` is the feature vector at
each spatial location and time step. The default fill returns a constant zero — no signal at all — so
the very first thing the predictor needs is *some* well-posed notion of "predicted is close to target."

The cleanest hypothesis to test first is that what matters in a learned latent space is the *pattern*
of activation across channels, not its overall scale. There is a real reason to suspect this. The
encoder and predictor are trained jointly with a Variance–Covariance regularizer bolted on by the
trainer; that regularizer already governs the *scale* of the features — the variance hinge keeps each
dimension's standard deviation above a floor and the covariance term decorrelates dimensions. If the
anti-collapse machinery is already pinning magnitude, then a prediction loss that *also* fights over
magnitude may be redundant or even at cross purposes with it. So let me try a loss that is blind to
magnitude and rewards only directional agreement: cosine similarity between the predicted and target
feature vectors at each spatial-temporal location.

The construction is direct. At every location, treat the length-`C` channel vector of `state` and of
`predicted` as vectors in feature space. L2-normalize each along the channel axis so it sits on the
unit sphere, take their dot product — which is now exactly the cosine of the angle between them, in
`[-1, 1]`, equal to `1` when they point the same way — and turn it into a loss by subtracting from one,
so perfect directional alignment costs zero and anti-alignment costs two. Then average that
per-location cost over the batch, channels-collapsed, time, and the spatial grid, so the scalar is a
mean angular disagreement comparable across the three model widths. The normalization is the whole
point: dividing each vector by its own norm before the dot product is what makes the loss invariant to
how *long* the predicted and target vectors are, so the predictor is graded purely on getting the
*direction* of the latent right and is free to let the regularizer set the scale.

Let me write the relationship out so I know exactly what gradient I am sending. With `s` the target
and `p` the prediction at one location, the normalized vectors are `ŝ = s/‖s‖` and `p̂ = p/‖p‖`, and
the per-location loss is `1 − ŝ·p̂`. Differentiating with respect to the predictor's output `p`, the
normalization contributes the projection operator `(I − p̂ p̂ᵀ)/‖p‖`: the gradient that reaches `p` is
the component of `ŝ` *orthogonal* to `p̂`, scaled by `1/‖p‖`. Two facts fall straight out of that. The
gradient lives entirely in the directions perpendicular to the current prediction — it rotates `p`
toward `s` and never pushes along `p` itself — which is the formal statement that the loss cannot move
magnitude. And the `1/‖p‖` prefactor means the rotational pull is *weaker* when `p` is short, so a
predictor whose features start small gets a faint directional signal early in training and a stronger
one only once magnitudes have grown — magnitudes this loss, by construction, is doing nothing to grow.
That coupling between a quantity I refuse to train and the strength of the signal I do send is the kind
of thing that can quietly stall a run, and it is worth holding in mind as I read the numbers.

I considered, and rejected for now, two nearby variants. One is to normalize over the whole feature
map (all of `C×H×W`) instead of per location, which would grade the predictor on one global direction
per frame; but that collapses every spatial descriptor into a single vector and throws away the
location-specific structure the detection probe needs, so it is strictly less informative for this
metric. The other is to keep the magnitude by *not* normalizing and instead penalize the negative dot
product directly; but an un-normalized dot product is unbounded and would let a few large-magnitude
locations dominate, which is a different failure (and one a later rung will have to confront on its own
terms). The clean, bounded, magnitude-blind choice is the per-location cosine, so that is the
hypothesis I commit to first.

Why normalize along the channel axis specifically, and not over the whole feature map? Because the
channel axis is where the *feature content* lives. The encoder produces a spatial map: at each pixel
location of the latent grid there is a `C`-dimensional descriptor of what the encoder saw there.
Directional agreement of those descriptors — "the predictor placed the right kind of feature at the
right location" — is the meaningful quantity. Normalizing across space or time would mix descriptors
from different locations into a single direction and destroy that per-location reading. So the
normalization axis is `dim=1`, the channel dimension, and the sum that forms the dot product is also
over `dim=1`; the remaining axes (batch, time, height, width) are averaged.

I should be honest with myself about the cost of throwing magnitude away, because that is the bet I am
making and it is the bet the next rung will test. A cosine loss is exactly invariant to a positive
rescaling of either argument: if the predictor outputs a feature vector pointing the right way but
half as long, this loss is perfectly happy, gradient zero. In a pure representation-matching setting
that invariance is a feature — it stops the predictor from wasting effort matching an arbitrary scale.
But there are two ways it can bite. First, the downstream metric is detection AP read out by a probe
head that sits on top of these latents, and a detection head generally *does* read magnitude —
activation strength at a location is part of how it scores "is there a digit here." If the prediction
loss is blind to magnitude, the predictor is never pushed to get the *strength* of the latent right,
only its direction, and the probe may inherit a representation whose magnitudes drift. Second, the
gradient of a normalized dot product is small precisely when the vectors are already short: the
1/‖v‖ factors that come out of differentiating the normalization shrink the directional gradient for
low-magnitude features, so early in training, when features are small, the directional signal can be
weak exactly where it is needed. Both of these are reasons cosine could *under*-train the predictor
relative to a loss that also pulls on magnitude — and that is the falsifiable thing I am putting on the
board for the next step.

There is also a subtle interaction with the VC regularizer I should name. The regularizer keeps the
variance of each feature dimension above a floor — it prevents collapse to zero — but it does *not*
guarantee that the predictor's outputs and the encoder's targets sit at the *same* scale; it only
prevents either from vanishing. So my premise — "the regularizer already governs magnitude, let the
loss handle direction" — is half true at best. It governs the *spread* of features, not the
*agreement of scale between predicted and target*. If that gap matters for the probe, cosine will
leave value on the table that an MSE-style magnitude-sensitive loss would capture. I expect this to
show up most at the largest model width, where there are more channels for magnitude information to
hide in and a larger predictor that could be matching scale if the loss asked it to.

So this is the floor I am establishing on purpose: the simplest well-posed feature-matching loss that
commits to one extreme of the magnitude-versus-direction question — pure direction. It is a legitimate
self-supervised choice with a clean motivation (don't fight the regularizer over scale), and it gives
the predictor a real learning signal where the default placeholder gave none. The whole hypothesis
rests on the claim that latent *direction* is what the detection probe ultimately needs, and that the
magnitude the cosine loss discards is either set adequately by the regularizer or irrelevant
downstream. The full scaffold module — channel-axis normalization, dot product, one-minus-cosine,
mean reduction — is in the answer.

What I will watch in the numbers: if direction is genuinely sufficient, cosine should land in a
healthy detection-AP range across all three widths and there is no obvious reason for it to fall apart
on the large model. If instead magnitude matters — if the probe needs activation strength the cosine
loss never trained — I expect the gap to open at the *large* width first, where the discarded magnitude
information is largest and the predictor most capable of matching scale had the loss demanded it. The
sharpest single number to fix is the large-model AP: a cosine loss that degrades there relative to
small/base would be direct evidence that throwing magnitude away costs the predictor exactly the scale
agreement the regularizer does not supply, and that the next rung must put magnitude back into the
loss. That is the test that decides whether the magnitude-blindness I am betting on is a feature or a
hole.
