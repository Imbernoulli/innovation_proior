The orthogonal numbers came in almost exactly where the construction predicted, and the pattern is the
diagnosis. On VGG-16-BN — the plain stack orthogonality was derived for — it scored 72.83, its *best*
relative showing. On ResNet-56 it landed at 72.08, unremarkable, sitting right in the fan-scaling
neighborhood as I expected once I realized orthogonality fixes per-branch conditioning but not residual
accumulation. And on MobileNetV2/FashionMNIST it came in at 93.88, the *weakest* of the three runs — exactly
the depthwise-conv failure I flagged: those `(C, 9)` tall weight matrices cannot be made row-orthogonal, so
the isometry I paid for never materialized on the layer type that dominates that network. So the orthogonal
rung did not buy a uniform win; it bought a topology-dependent one, strong only where the topology matched the
theory and actively weakest where it didn't. That tells me the expensive part of the orthogonal scheme — the
full-spectrum control, every singular value pinned to one — is not what is doing the work, and on two of three
architectures it is either redundant or unattainable.

The redundancy is the key realization, and it reframes the whole problem. Every conv in all three networks is
followed by BatchNorm. BN takes whatever the conv produces and re-standardizes it to zero mean, unit variance
per channel before the next layer sees it. So the elaborate guarantee orthogonality offers — that the
*activation norm* is preserved layer to layer — is precisely the thing BN already enforces at every layer,
for free, from data statistics, *regardless* of how I initialized the conv. Pinning the whole singular-value
spectrum to control forward-norm propagation is solving a problem BN has already solved downstream. What BN
does *not* fix, and what initialization still genuinely owns, is the very first forward pass and the very
first backward pass before BN's running statistics have warmed up, and — more importantly — the *scale* of the
gradient that SGD sees at the start. If the conv weights are scaled wrong, the gradient that flows back
through them is scaled wrong, and the first few hundred SGD steps either crawl or thrash regardless of what BN
does to the forward activations. So the right target is not the full spectrum; it is getting the *second
moment* exactly right for ReLU, cheaply, uniformly, in the BN-friendly way — which is exactly what He scaling
was built to do and what I deliberately set aside to test the orthogonal hypothesis first.

So I go back to the variance question and do it properly for ReLU. Consider one conv layer with response
`y = Wx` (ignore bias, it is zero) feeding a ReLU. The forward-variance condition that keeps the signal from
shrinking or blowing as it passes the layer is the one He et al. (2015) derived: with ReLU killing half the
inputs, the per-output variance is `Var(y_out) = ½ · fan · Var(W) · Var(x)`, where the ½ is ReLU's leak and
`fan` is the number of input connections feeding each output. To hold `Var(y_out) = Var(x_in)` across the
layer you need `Var(W) = 2 / fan`. That factor of two is the entire content of He over Xavier — it pays back
exactly what ReLU takes. So I draw each conv weight from `N(0, √(2/fan))`. This matches the orthogonal-√2
target second moment, but with a plain Gaussian spectrum instead of a pinned one — and the orthogonal results
just told me the pinned spectrum wasn't the thing earning the points.

Now the one decision in this rung that is *not* the textbook default, and it is deliberate: which fan to use,
`fan_in` or `fan_out`. For a conv, `fan_in = in_channels · kh · kw` and `fan_out = out_channels · kh · kw`.
`fan_in` scaling keeps the *forward* activation variance stable; `fan_out` scaling keeps the *backward*
gradient variance stable. He's original paper showed that for a feed-forward ReLU net either choice works
because the ratio between consecutive layers' fan-in and fan-out telescopes — the depth-wise product of the
mismatch is bounded, so you don't get compounding decay either way. I choose **`fan_out` for the convs**, and
the reason is the residual and BN structure I am actually initializing. With `fan_out` the variance is set by
the number of *output* channels, which makes the backward gradient through each conv well-scaled — and since
BN already pins the forward activation variance regardless, controlling the *backward* scale is the higher-
value thing initialization can still do here. For the `Linear` classifier head I use **`fan_in`** instead:
the head has no BN after it and feeds directly into the softmax/cross-entropy, so I want its *forward*
pre-logit variance controlled, and `fan_in` is the choice that holds the logit scale sane at the start rather
than letting an over-large head saturate the softmax before training has moved. Splitting the mode by layer
role — `fan_out` on the BN-backed convs for gradient scale, `fan_in` on the bare head for logit scale — is
the considered version of "He init" for *this* substrate, not a blind application of one mode everywhere.

BatchNorm stays at `(γ=1, β=0)`, the same neutral identity affine I used for orthogonal, and for the same
reason: at this rung BN's job is to re-standardize, and the identity affine lets it do that without amplifying
or suppressing. I am explicitly *not* doing anything residual-aware yet — no per-block scaling, no zero-γ —
because the point of this rung is to isolate whether the simple, correct, BN-friendly second-moment target
already matches the orthogonal spectrum that cost so much to impose. If it does, that is strong evidence the
full-spectrum machinery was redundant, and it sets up the next rung cleanly: the residual *accumulation*
problem, which neither orthogonal nor plain He touches, becomes the obvious remaining lever.

Walk the three architectures and predict against the orthogonal numbers I now have in hand. **MobileNetV2** is
where I expect the clearest swing in my favor. Its depthwise convs — the ones whose tall `(C, 9)` matrices
broke orthogonality — are handled honestly by He scaling: for a depthwise conv `fan` counts the actual `kh·kw`
= 9 connections per output channel, so `N(0, √(2/fan_out))` gives each depthwise filter the *right per-filter
variance* without pretending to an isometry it cannot have. Where orthogonal had to degrade to "nine
well-conditioned directions," He just sets the correct second moment for those nine weights and moves on. So I
expect kaiming_normal to recover ground on MobileNetV2 relative to orthogonal's 93.88 — this is the
architecture where the orthogonal scheme was actively misapplied and the simpler target is strictly more
appropriate. **VGG-16-BN** is the closest call: it is the plain stack where orthogonality had its *best*
relative showing (72.83), because the full spectrum genuinely is well-defined and well-conditioned there. Here
I expect He to be competitive but I am not confident it beats orthogonal — on a deep plain ReLU stack the
pinned spectrum has a real, if small, conditioning advantage over a Gaussian one, so VGG is where orthogonal's
72.83 might hold up or even stay ahead. **ResNet-56** I expect to be roughly a wash with orthogonal's 72.08:
both schemes get the per-layer second moment right (orthogonal via √2 gain, He via `√(2/fan)`), and *neither*
addresses the residual accumulation that is the dominant effect in a 56-layer res-net — so they should land in
the same neighborhood, and whichever wins by tenths is noise rather than mechanism. That ResNet wash is
itself the most informative outcome I can hope for, because it isolates the lever the next rung must pull: if
two schemes that differ in their per-layer spectrum tie on the residual net, then per-layer spectrum is *not*
what limits the residual net — residual accumulation is.

So the step-2 edit is the literal He fill, with the role-split modes: iterate `model.modules()`; for every
`Conv2d` apply `kaiming_normal_(mode='fan_out', nonlinearity='relu')`; for every `Linear` apply
`kaiming_normal_(mode='fan_in', nonlinearity='relu')` with zero bias; for every `BatchNorm2d` set
`(weight=1, bias=0)`. One uniform pass, no architecture branching, no depth arithmetic. The distilled module
and its exact code live in the answer.

The falsifiable expectations against the orthogonal rung, stated plainly so the next rung has a clean
scoreboard to react to. (1) On MobileNetV2 I expect kaiming_normal to *beat* orthogonal's 93.88, because He
scaling treats depthwise convs correctly where orthogonality could not — if it does not, my whole "the
spectrum wasn't doing the work" diagnosis is wrong and I should reconsider whether depthwise layers want
something stranger. (2) On VGG-16-BN I expect a near-tie, with orthogonal's 72.83 possibly holding a slim edge
from its genuine spectral conditioning on the plain stack — if He clearly *wins* here too, then the spectrum
buys nothing even on its home turf and the orthogonal rung was strictly dominated. (3) On ResNet-56 I expect a
wash around 72.0–72.4, because both fix per-layer variance and neither fixes residual accumulation — and that
wash is the signal that the residual net's ceiling is set by accumulation, not by per-layer scale. If all
three land as expected, the lesson is sharp: the cheap, correct, BN-friendly second moment matches or beats
the expensive full-spectrum scheme everywhere except possibly the one plain stack, which means the next move
is not a better per-layer init at all — it is to finally attack the residual-accumulation problem that has sat
untouched under both rungs, by scaling the residual branches down and starting them near identity so that
adding depth stops inflating the main-path signal.
