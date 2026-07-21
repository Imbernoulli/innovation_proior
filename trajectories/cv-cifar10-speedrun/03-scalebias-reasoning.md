Whitening and Dirac init have together pulled me to 18 epochs and 6.8 seconds, and the Dirac win
confirms my mental model is calibrated. It took me 21 → 18 epochs and 8.0 → 6.8 seconds, and decisively
the per-epoch cost went 8.0/21 = 0.381 s to 6.8/18 = 0.378 s — flat. That flatness is exactly what I
predicted for an initialization-only change: Dirac adds nothing to any forward or backward step, so its
entire effect had to show up as fewer epochs, and it did. Whitening, by contrast, carried a 7% per-epoch
dividend from freezing a layer. So the two init fixes behaved as diagnosed — both attacked the *starting
point*, both bought their speedup by removing conditioning epochs — and together they cut 45 to 18, a
factor of 2.5, while per-step cost barely moved. And 6.8 s is a 2.69× total speedup that draws almost
level with hlb-CIFAR10's 6.3-second published record, so I am one good change from crossing under the prior
record entirely; everything from here is new ground.

Both wins conditioned where the optimizer *starts*. What is untouched is the *trajectory* — the ~18
epochs the optimizer still spends walking from that good start to the final weights. At ~49 steps per
epoch, 18 epochs is only ~880 optimizer steps, a tight budget, and the training applies a single global
learning-rate schedule — the triangular ramp — uniformly to every parameter. That uniformity encodes an
assumption: that all parameters want to learn at the same speed. But the network mixes parameters of
genuinely different *kinds*. Almost all of the ~1.97M trainable parameters are conv weights — thousands
to hundreds of thousands of numbers each. Scattered among them are the BatchNorm biases: one scalar per
channel, 64+64+128+128+256+256 = 896 across the six BatchNorms, about 0.045% of the parameters. And
because the BatchNorm *scales* are frozen at 1, each bias is the *entire* learnable content of its
normalization layer — a single shift applied after normalization, before the GELU.

Mechanically a BatchNorm bias does something categorically different from a conv weight. A conv weight
learns a spatial feature — an oriented edge or texture detector. A bias, sitting right before the GELU,
learns *where to put the threshold* for a whole channel: it slides the entire channel's normalized
distribution left or right relative to the GELU kink, setting how much of that feature map survives the
nonlinearity. That is a low-dimensional, high-leverage decision — move one bias and an entire channel's
operating point shifts; move one conv-weight entry and one tap of one filter nudges. The frozen scale is
what makes this clean: with a trainable scale, scale and bias are partly redundant and the
operating-point role smears across two coupled parameters; freezing the scale collapses it to a single
degree of freedom per channel, a pure shift, exactly the well-conditioned parameter a large learning
rate is safe on.

That geometric difference is the whole argument, so make it precise. A conv weight's effect on the loss
is diluted — one entry among 9C² sharing the job of representing a feature, its gradient a small piece
of a large redundant ensemble, noisy and low-magnitude, so the natural step size is modest. A BatchNorm
bias is the opposite: a single number that rigidly translates a channel, and its gradient aggregates the
per-position sensitivity over *every* spatial location and *every* image in the batch — for an 8×8 map
at batch 1024 that is 64 × 1024 ≈ 65,000 terms summed into one scalar. Its relative noise scales like
1/√65,000 ≈ 1/256, so the bias gradient is not just concentrated but *statistically much better
resolved* than any conv-weight gradient — and a well-resolved gradient on a low-dimensional,
well-conditioned parameter is exactly what can absorb a large step without overshooting. Under one shared
rate, though, these biases are dragged along at the conv weights' modest pace: by the time the schedule
lets the conv weights settle over 18 epochs, the biases — which could have snapped to their operating
points in a fraction of that — have been crawling. They are *under*-stepped relative to what their clean
gradient could support, and a bias that starts near zero and must travel an O(1) shift is
transport-limited, gaining almost linearly from a larger step.

So decouple the learning rate by parameter type. Switching the whole optimizer to Adam would
automatically give the well-conditioned biases larger relative steps, but it replaces the well-tuned SGD
trajectory, adds per-parameter state, and changes the dynamics of the conv weights I have no complaint
about. Raising the *global* rate would diverge the conv weights, which are already near the edge of
stability, long before the biases got their boost. The surgical move is to leave SGD and the global
schedule exactly as they are for the conv weights and give *only* the BatchNorm-bias group its own much
larger rate. How large? The dimensional argument says a big multiplier is reasonable — the gradient is
resolved ~256× better, so a substantially larger step is within what the signal supports — and 64× sits
comfortably inside that envelope: aggressive enough to stop the biases crawling, far short of the ~256
SNR ceiling, with real margin against overshoot (a margin I want because the shared Nesterov momentum
inflates the effective step further, and because the benign O(1) curvature of a post-normalization bias
keeps 64·lr·h well under the stability limit — which is precisely why 64× is safe *here* and would not
be on the wide, anisotropic curvature of the conv weights).

One subtlety, or the boost fights itself: weight decay. I want the biases to *move fast*, not to *decay
them 64× harder*, which is what naively scaling the whole update would do. The decoupled-weight-decay
parametrization keeps learning rate and decay separate: the per-step decay pull is lr × (group
weight_decay) × p, so if I set the bias group's weight_decay to wd / lr_biases, the actual decay applied
is lr_biases × (wd/lr_biases) × p = wd × p — identical to every other parameter, independent of the 64×
on the learning rate. So the biases take 64× larger *gradient* steps and the *same* decay as before; the
boost goes entirely into learning speed. Concretely a two-group SGD: the `'norm' in k` filter routes
only the BatchNorm biases into the boosted group at `lr·64`, everything else stays in the base group at
`lr`, each group's weight_decay set to keep decay strength constant (full optimizer in the answer).

Leaving the whitening conv's own learnable bias *out* of the boost is deliberate: it is a scalar too,
but it shifts the frozen whitening output *before* any normalization, coupled to the eigenbasis rather
than to a unit-variance normalized distribution, so its curvature is not the tidy O(1) that licenses a
64× step. The aggressive rate is specific to normalized biases, so the boost is scoped to them, and the
two filters `'norm' in k` / `'norm' not in k` are exact complements so every trainable parameter lands
in exactly one group.

This is also the natural place in the ladder for the fix: whitening and Dirac gave the network a clean
signal, and the biases decide how each channel *gates* that clean signal through its asymmetric GELU —
a bias pushed well positive makes a channel fire on almost everything (near-linear, low information),
one pushed negative makes it fire only on its strongest activations (sparse, selective). For a network
with only a few hundred channels, getting that gating right is a large fraction of what "learning good
features" means, so a rate mismatch on the gates is worth fixing precisely once there is a clean signal
for them to gate.

If the BatchNorm biases really were the bottleneck — under-stepped scalars whose clean gradient could
support far larger steps — then letting them move 64× faster should let the network find its channel
thresholds in far fewer epochs with accuracy held at 94%. I expect this to be the largest of the
optimization-rate tricks — more than Dirac's ~3 epochs but less than whitening's ~24, since it is a step-
size refinement on 0.05% of the parameters, not a re-conditioning of the whole signal path. Per-epoch
cost should again be flat (a per-group rate is free at run time). The failure mode is divergence: if 64×
is past the stability ceiling after all, I would see the mean fall below the bar or NaN out rather than
the epoch count drop, and I would back the scaler off. Mean holding at 94% while epochs-to-94% fall
confirms the rate-mismatch diagnosis. The code is in the answer.
