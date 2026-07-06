Whitening and Dirac init have together pulled me to 18 epochs and 6.8 seconds, and the second of those
two wins is worth reading carefully before I decide where to go, because it confirms a prediction and
that tells me my mental model is calibrated. Dirac took me 21 → 18 epochs and 8.0 → 6.8 seconds. Epochs
fell by a factor 1.167, wall-clock by 1.176 — and, decisively, per-epoch cost went 8.0/21 = 0.381 s to
6.8/18 = 0.378 s, essentially flat. That flatness is exactly what I predicted for an initialization-only
change: Dirac adds nothing to any forward or backward step, so its entire effect had to show up as fewer
epochs and none of it as cheaper epochs, and that is precisely what happened. Whitening, by contrast, had
carried a 7% per-epoch dividend from freezing a layer. So the two initialization fixes behaved as
diagnosed: both attacked the *starting point*, both bought their speedup by removing epochs of
conditioning work, and together they cut the epoch count from 45 to 18 — a factor of 2.5 — while the
per-step cost barely moved.

It is worth placing 6.8 seconds against the prior art, because it tells me how much is left to fight for.
The baseline was 18.3 s; I am now at 6.8, a 2.69× total speedup, and that has drawn almost level with
hlb-CIFAR10's 6.3-second published record — the best public number I started below. So I am one good rung
from crossing under the prior record entirely, and everything from here is genuinely new ground rather
than catching up. That reframes the stakes: the easy structural wins (input conditioning, deep-layer
conditioning) are spent, and the next second has to come from somewhere subtler.

But that is the point. Both wins were *initialization* fixes: they conditioned where the optimizer
starts. What is left untouched is the *trajectory* — the ~18 epochs the optimizer still spends walking
the weights from that good starting point to their final values. At about 49 steps per epoch (50,000 /
1024 ≈ 48.8), 18 epochs is only ~880 optimizer steps, a genuinely tight budget, and I want to know
whether every parameter is actually *moving* at the rate it should over those 880 steps. The training
applies a single global learning-rate schedule — the triangular ramp — uniformly to every parameter. That
uniformity encodes an assumption I have not examined: that all parameters want to learn at roughly the
same speed. Let me question it, because the network mixes parameters of genuinely different *kinds*.

Do a census. Almost all of the ~1.97M trainable parameters are convolution weights — 3×3×C×C tensors,
thousands to hundreds of thousands of numbers each. Scattered among them are the BatchNorm biases: one
scalar per channel. Counting them across the six BatchNorms — channels 64, 64, 128, 128, 256, 256 — gives
64+64+128+128+256+256 = 896 scalar biases. That is about 896 / 1.97M ≈ 0.045% of the parameters, a
rounding error in the parameter count. And here is the structural fact that makes them special: in this
network the BatchNorm *weights* (scales) are frozen at 1, so the *only* thing each BatchNorm contributes
to learning is its per-channel bias — a single shift applied after normalization, before the GELU. So
these 896 numbers are not incidental; they are the entire learnable content of the normalization layers,
and they are doing something categorically different from the conv weights around them.

What does a BatchNorm bias actually do, mechanically? A conv weight learns a spatial feature — some
oriented edge or texture detector. A BatchNorm bias, sitting right before the GELU, learns *where to put
the threshold* for a whole channel: it slides the entire channel's normalized activation distribution
left or right relative to the GELU's kink at zero, setting how much of that feature map survives the
nonlinearity and how much gets suppressed. That is a low-dimensional, high-leverage decision. Moving one
bias by a small amount shifts an entire channel's operating point; moving one conv-weight entry nudges
one tap of one filter. So the two kinds of parameter live on very different loss geometries.

It is worth noticing that the substrate has already set this up for me: BatchNorm's scale is frozen at 1,
so the bias is the *sole* learnable knob of each normalization. That is not incidental to my argument — it
is what makes the argument clean. With a trainable scale, scale and bias are partly redundant (both can
rescale the effective threshold), and the "operating-point" role would be smeared across two coupled
parameters with a messier joint geometry. Freezing the scale collapses that to a single degree of freedom
per channel — a pure shift — which is exactly the low-dimensional, well-conditioned parameter that a large
learning rate is safe on. So the frozen-scale choice in the fixed substrate and the boosted-bias idea are
two halves of the same design: make the operating point a clean scalar, then let that scalar move fast.

That geometric difference is the whole argument, so let me make it precise. A convolution weight's effect
on the loss is diluted: it is one entry among 9C² sharing the job of representing a feature, and its
gradient is a small piece of a large, redundant ensemble spread across all spatial positions — a noisy,
low-magnitude signal, and the natural step size for it is modest. A BatchNorm bias is the opposite. It is
a single number that rigidly translates an entire channel, and its gradient ∂L/∂b_c aggregates the
per-position sensitivity over *every* spatial location and *every* image in the batch — for an 8×8 feature
map at batch 1024 that is 64 × 1024 ≈ 65,000 terms summed into one scalar. Averaging over ~65,000 samples
makes that gradient a clean, low-variance, high-confidence estimate of a global property of the channel:
the relative-noise on it scales like 1/√(65,000) ≈ 1/256. So the bias gradient is not just concentrated,
it is *statistically much better resolved* than any individual conv-weight gradient, and a well-resolved
gradient on a low-dimensional, well-conditioned parameter is exactly the thing that can absorb a large
step without overshooting. Under one shared learning rate, though, these biases are dragged along at the
conv weights' modest pace: by the time the schedule has let the conv weights settle over 18 epochs, the
biases — which could have snapped to their operating points in a fraction of that time — have been
crawling. The biases are *under*-stepped relative to what their clean gradient could support.

Let me put a crude budget on what "under-stepped" costs me in epochs, since epochs are the currency. A
freshly initialized bias starts near zero and needs to travel to its operating point, an O(1) shift on
the normalized scale. If each base-rate gradient step moves it by roughly δ, arriving takes on the order
of 1/δ steps. Suppose at the base rate that arrival consumes something like the first quarter of training
before the thresholds are set — a few epochs of the 18. At 64× the same journey takes ~1/(64δ) steps,
about 1.5% as long: the biases reach their operating points almost immediately and then spend the rest of
training being fine-tuned rather than transported. If the operating-point search really was eating even
three or four of the eighteen epochs, recovering most of them is the size of prize on offer. The number
is soft — I do not know δ precisely — but the shape is right: a parameter that is transport-limited rather
than precision-limited gains almost linearly from a larger step, and these biases look transport-limited.

So the move is to decouple the learning rate by parameter type, and I should weigh how against the
alternatives. One option is to switch the whole optimizer to something per-parameter adaptive — Adam or
similar — which would automatically give the well-conditioned biases effectively larger relative steps.
But that replaces the entire, well-tuned SGD trajectory, adds per-parameter state (doubling or tripling
optimizer memory), and changes the dynamics of the conv weights I have no complaint about — a large,
risky change to fix a small, specific problem. A second option is simply to raise the *global* learning
rate; but the global rate is already near the edge of stability for the conv weights, and pushing it
would diverge them long before the biases got the boost they can tolerate. The third option is surgical:
leave SGD and the global schedule exactly as they are for the conv weights, and give *only* the
BatchNorm-bias parameter group its own, much larger learning rate. That touches nothing I am happy with
and directly addresses the one thing I am not, so that is the move — a per-group learning-rate scaler.

How large a factor? The dimensional argument above says a big multiplier is *reasonable* for these
particular parameters — their gradient is resolved ~256× better than a typical weight's, so a
substantially larger step is well within what their signal supports. The specific value that the BN-bias
tuning in this lineage has converged on is 64×, and it sits comfortably inside that envelope: aggressive
enough to stop the biases crawling, far short of the ~256 that the SNR ceiling would nominally permit, so
there is real margin against overshoot. I will take 64× as the bias-lr scaler. It sounds alarming stated
baldly — a 64× learning rate is the kind of thing that blows up training — but the alarm is about
applying a huge rate to the *wrong* parameters (the noisy, ill-conditioned conv weights), and the entire
point of decoupling is that it lands only on the benign scalars.

One subtlety I have to get right, or the boost fights itself: weight decay. I want the biases to *move
fast*, but I do not want to *decay them 64× harder*, which is what would happen if I naively scaled the
group's whole update. The decoupled-weight-decay parametrization keeps learning rate and decay strength
separate, and I can use that to hold decay fixed while only the step grows. Concretely, the per-step decay
pull in this SGD is lr × (group weight_decay) × p, so if I set the bias group's weight_decay to wd /
lr_biases, the actual decay applied is lr_biases × (wd / lr_biases) × p = wd × p — identical to what every
other parameter feels, independent of the 64× on the learning rate. Let me confirm that lands the way I
claim: with lr_biases = 64·lr and weight_decay = wd/lr_biases, the decay term is lr_biases · wd/lr_biases
= wd, exactly, while the gradient term is scaled by the full lr_biases = 64·lr. So the biases take 64×
larger *gradient* steps and the *same* decay as before — the boost goes entirely into learning speed and
none of it into extra shrinkage, which is precisely the decomposition I want.

```python
lr_biases = lr * hyp['opt']['bias_scaler']          # bias_scaler = 64.0

norm_biases  = [p for k, p in model.named_parameters() if 'norm' in k and p.requires_grad]
other_params = [p for k, p in model.named_parameters() if 'norm' not in k and p.requires_grad]
param_configs = [dict(params=norm_biases,  lr=lr_biases, weight_decay=wd/lr_biases),
                 dict(params=other_params, lr=lr,        weight_decay=wd/lr)]
optimizer = torch.optim.SGD(param_configs, momentum=momentum, nesterov=True)
```

The `'norm' in k` filter is what routes only the BatchNorm biases into the boosted group; the whitening
conv's own learnable bias (its key does not contain 'norm') and everything else stay in the base group at
the ordinary rate — which is correct, because the argument I just made is specifically about the
per-channel normalization biases, not about every scalar in the network. Let me verify the partition is
clean, since a double-counted or dropped parameter in an optimizer is a silent bug. The two filters are
`'norm' in k` and `'norm' not in k` over the same `requires_grad` parameters — exact logical complements
— so every trainable parameter lands in exactly one group, none in both, none in neither. The boosted
group holds the 896 BatchNorm biases; the base group holds the ~1.97M conv weights plus the whitening
bias plus the linear head. Adds up, disjoint, complete — the SGD sees every trainable parameter exactly
once. And leaving the whitening bias *out* of the boost is deliberate, not an oversight: it is a scalar
too, but it is not a per-channel post-normalization operating-point knob in the same clean sense — it
shifts the frozen whitening output *before* any normalization, coupled to the eigenbasis rather than to a
unit-variance normalized distribution, so its curvature is not the tidy O(1) that licenses a 64× step.
The argument for the aggressive rate is specific to normalized biases, so the boost is scoped to them.

There is a reason the *operating point* matters so much in this particular network, and it is the GELU's
asymmetry. Because GELU passes positives roughly linearly and squashes negatives toward zero, where a
channel's bias sits relative to the kink decides that channel's *selectivity*: a bias that pushes the
distribution well positive makes the channel fire on almost everything (near-linear, low information), a
bias that pushes it negative makes the channel fire only on its strongest activations (sparse, selective).
For a small network with only a few hundred channels, getting that gating right — which channels are
broadly-on and which are sharply-selective — is a large fraction of what "learning good features" even
means here. So the biases are not a minor trim; they set the information-gating regime of every feature
map in the net. That also explains the natural ordering of my ladder: whitening and Dirac gave the
network a clean signal to work with, and now the biases decide how each channel *gates* that clean signal
through its nonlinearity. Conditioning the signal first and letting the gating catch up second is the
sensible sequence, and it is why a rate mismatch on the gates is worth fixing precisely now, once there
is a clean signal for them to gate.

There is a momentum interaction I should not gloss over, because it makes the effective step larger than
the bare 64× suggests. Both groups share the same Nesterov momentum, so every step is amplified by the
usual momentum factor ≈ 1/(1 − momentum) at steady state — the boosted bias group rides on top of that
just as the conv weights do. So the *effective* bias learning rate is 64× the base *and* momentum-inflated
on top, which is a second reason to prefer 64 over something nearer the ~256 SNR ceiling: I want headroom
against the momentum amplification, not to sit at the theoretical edge. It also means the biases, once
their gradient turns over near their operating point, will coast a little on accumulated momentum — which
is fine and even helpful for a scalar settling into a smooth minimum, but it is another reason the scaler
should be aggressive-but-not-reckless.

Let me also sanity-check the stability claim rather than just asserting the biases "can take it." For a
scalar parameter with local curvature h, a gradient step is stable when lr·h < 2 (beyond that the
quadratic model diverges). A BatchNorm bias sits on an especially benign curvature: it enters the loss
after normalization, so the activation it shifts is already unit-scaled, and its second derivative is
roughly O(1) rather than the wide, anisotropic spectrum a conv weight sees. So the stability ceiling
64·lr·h < 2 needs 64·lr to stay below ≈ 2/h with h = O(1) — comfortably satisfied for the small triangular
peak lr, which is exactly why 64× is safe *here* and would not be safe on the conv weights, whose h can be
large and whose direction-to-direction curvature varies wildly. The decoupling isolates the aggressive
rate to the one place the curvature permits it.

The prediction, stated to be falsifiable against the epoch column. If the BatchNorm biases really were the
bottleneck — under-stepped scalars whose clean, well-resolved gradient could support far larger steps —
then letting them move 64× faster should let the network find its right channel thresholds in far fewer
epochs, and the whole training should compress again with accuracy held at 94%. I expect this to be the
largest of the *optimization*-rate tricks, because it corrects a genuine rate mismatch on parameters that
set the operating point of every nonlinearity in the network — but not another whitening-scale jump,
because it is a refinement of the step size on 0.05% of the parameters, not a re-conditioning of the
entire signal path. So: a solid step down from 18 epochs, more than Dirac's ~3 but less than whitening's
~24, is my honest expectation. Per-epoch cost should again be flat — a per-group learning rate is free at
run time — so like Dirac, the whole effect should read out in the epoch column. The failure mode to watch
is divergence: if 64× turns out to be past the stability ceiling for these biases after all, I would see
the mean accuracy fall below the bar (or NaN out) rather than the epoch count drop, and I would have to
back the scaler off. The test is clean: if mean accuracy stays at 94% while epochs-to-94% fall, the
rate-mismatch diagnosis was right. The change is the two-group bias-lr scaler; the code is in the answer,
and the epochs table will tell me whether the biases really were crawling.
