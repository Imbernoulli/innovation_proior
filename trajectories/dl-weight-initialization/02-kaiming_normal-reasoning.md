The orthogonal numbers came in close to where the construction predicted, and I want to read the table
carefully before I move, because the pattern *is* the diagnosis for this rung. Three numbers: ResNet-56 on
CIFAR-100 at 72.08, VGG-16-BN on CIFAR-100 at 72.83, MobileNetV2 on FashionMNIST at 93.88. I have to be
disciplined about what these can and cannot be compared to. FashionMNIST is a different, easier task, so
93.88 is not on the same axis as the two CIFAR numbers — it is only an anchor the next method must clear on
that task. But ResNet-56 and VGG-16-BN are *both* on CIFAR-100, same data, same schedule, so their gap is a
comparison I am allowed to make: VGG beat ResNet by 0.75 points on identical data. I will not over-read it —
the two architectures have different capacity, so part of that 0.75 is VGG simply being a stronger model on
CIFAR-100, not the init — but the *sign* is exactly what rung 1 predicted: orthogonality should look best on
the plain stack it was built for and merely ordinary on the residual net whose accumulation it never touched.
So the qualitative story holds, and the three numbers become three per-task anchors that whatever I do next
has to beat. What the table does not do is tell me the expensive part of the orthogonal scheme earned its
keep. It bought a topology-dependent showing, strong exactly where the topology matched the theory and
weakest on the depthwise net where the isometry could not even be constructed. That is a hint — not yet proof
— that the full-spectrum control, every singular value pinned to one, is not the thing doing the work, and on
two of three architectures is either redundant or unattainable.

The redundancy is the realization that reframes the whole problem, and it is worth stating mechanically rather
than as a slogan. Every conv in all three networks is followed by BatchNorm. BN takes whatever the conv
produces and re-standardizes it to zero mean, unit variance per channel before the next layer sees it. So the
elaborate guarantee orthogonality offers — that the *per-channel activation variance* is preserved layer to
layer — is precisely the thing BN already enforces at every layer, for free, from data statistics,
*regardless* of how I initialized the conv. Pinning the whole singular-value spectrum to control forward-norm
propagation is, on the diagonal, solving a problem BN has already solved downstream. I flagged one crack in
that argument at the last rung — BN standardizes each channel independently, so it cannot fix the
*off-diagonal* conditioning of the weight map, which is why orthogonality might still hold a thin edge on the
plain stack — and the VGG-versus-ResNet gap is faintly consistent with that. But faint is the operative word:
whatever the spectrum bought, it did not lift the residual net, and it could not be built at all on the
depthwise net. So the question this rung exists to answer is sharp: does the *cheap* target — get the second
moment exactly right for ReLU and let BN own the rest — match the expensive pinned spectrum? If it does, the
spectrum was redundant; if it loses on VGG, the off-diagonal conditioning was real and I have to keep it. What
BN emphatically does *not* fix, and what initialization still genuinely owns, is the very first forward pass
and the very first backward pass before BN's running statistics have warmed up, and — more load-bearing — the
*scale of the gradient* that SGD sees at step zero. If the conv weights are scaled wrong, the gradient that
flows back through them is scaled wrong, and the first few hundred SGD steps either crawl or thrash regardless
of what BN does to the forward activations. So the right target is not the full spectrum; it is the second
moment, cheaply, uniformly, in the BN-friendly way — which is exactly what He scaling was built to do and what
I deliberately set aside to test the orthogonal hypothesis first.

So I go back to the variance question and do it properly for ReLU. Consider one conv layer with response
`y = Wx` (ignore bias, it is zero) feeding a ReLU. The forward-variance recursion is: `Var(y_out) = fan ·
Var(W) · Var(x_in)` for a bare linear layer, and the ReLU sitting on the *input* side (the previous layer's
activation is a rectified Gaussian with only half its variance surviving) contributes a factor of one half, so
across a `conv→relu` stage `Var(next) = ½ · fan · Var(W) · Var(prev)`, where `fan` is the number of input
connections feeding each output. To hold `Var(next) = Var(prev)` — the whole point, a fixed point at unit
gain — you need `Var(W) = 2/fan`. That factor of two is the entire content of He over Xavier: it pays back
exactly what ReLU takes. So I draw each conv weight from `N(0, √(2/fan))`. This matches the orthogonal-√2
target second moment (a row-orthonormal matrix scaled by √2 has per-entry RMS `√(2/fan_in)`, which I checked
last rung equals `√(2/576) ≈ 0.0589` on the 576-fan layer), but with a plain Gaussian spectrum instead of a
pinned one — and the orthogonal results give me no evidence the pinned spectrum was earning the points.

Now the one decision in this rung that is *not* the textbook default, and it is the decision worth spending
real thought on: which fan to use, `fan_in` or `fan_out`. For a conv, `fan_in = in_channels · kh · kw` and
`fan_out = out_channels · kh · kw`. `fan_in` scaling holds the *forward* activation variance at its fixed
point; `fan_out` scaling holds the *backward* gradient variance at its fixed point, because the backward pass
multiplies by `Wᵀ` whose second-moment map scales by `fan_out · Var(W)`. He's original paper showed that for a
feed-forward ReLU net *either* choice works, and I want to see *why* concretely, because the reason tells me
which to pick here. Suppose I use `fan_out` everywhere and track the forward variance. Layer `l` then
multiplies forward variance by `½ · fan_in_l · Var(W_l) = ½ · fan_in_l · 2/fan_out_l = fan_in_l/fan_out_l =
(in_l·k²)/(out_l·k²) = in_l/out_l`. Over the whole stack the forward variance is multiplied by the product
`Π_l in_l/out_l`. But channels chain — `in_{l+1} = out_l` — so the product telescopes: `in_1/out_1 ·
out_1/out_2 · out_2/out_3 · … = in_1/out_L`, a single bounded constant set by the first layer's input width
and the last layer's output width, *not* a factor that compounds with depth. So `fan_out` leaves the forward
variance off by one bounded constant, which BN then absorbs on its first pass, while keeping the backward
gradient variance *exactly* pinned. Symmetric statement holds for `fan_in` and the backward direction. That
telescoping is the real content of "either works," and it also tells me the decision is much smaller than it
looks: for the *square* convs — where `in_l = out_l`, so `fan_in = fan_out` identically — the two modes are
*literally the same numbers*. I checked one: a dense `64→64` 3×3 conv has `fan_in = fan_out = 576`. Most convs
inside a ResNet stage and most inside a VGG block keep the channel count fixed, so for the bulk of both of
those networks the mode choice changes nothing at all. It only bites where channels change (stage transitions,
VGG's width steps) and, most sharply, on the depthwise convs. So the decision is really "which direction do I
protect at the few layers where the two differ," and given BN already pins the forward per-channel variance
everywhere, I protect the **backward gradient scale** — I choose `fan_out` for the convs. Controlling the
backward scale is the higher-value thing initialization can still do here, because BN cannot do it for me on
the first step.

I want to see the telescoping on the actual VGG widths rather than trust the algebra abstractly, because if
the offset really compounded I would be choosing wrong. VGG-16-BN steps its channels `3 → 64 → 64 → 128 → 128
→ 256 → 256 → 256 → 512 → … → 512`. Walk the forward-variance factors under `fan_out`: the first conv
contributes `in_1/out_1 = 3/64`, the second `64/64 = 1`, the third `64/128 = 1/2`, and so on; every layer that
keeps its width contributes exactly `1`, and every widening layer contributes a fraction that the *next*
layer's numerator cancels. Multiply the whole chain and everything between the ends telescopes away, leaving
`in_1/out_L = 3/512 ≈ 0.006`. So the *entire* accumulated forward-variance error from using `fan_out` on
thirteen VGG convs is a single factor of about `1/170`, set only by the input being 3-channel and the last
conv being 512-wide — a one-time constant BN standardizes on its first pass, not a per-layer decay that eats
the signal. That is the concrete reassurance that `fan_out` is safe on the forward side; it also shows the
mode choice is almost entirely about the handful of widening layers, since every equal-width layer contributes
`1` in *both* modes. Where the modes genuinely part is the channel-changing convs — I checked a pointwise
`96 → 24` projection of the kind MobileNetV2 uses and it reports `fan_in = 96`, `fan_out = 24`, a 4× gap in
`Var(W)` between the two modes on that one layer — and that is exactly the parameter-dense, gradient-carrying
map where I want the backward scale pinned, which `fan_out` does.

It is worth being concrete about *why* the backward scale is the one initialization should spend its budget on,
because "BN owns the forward" only makes the case half. At step zero BN standardizes the forward activations
from the current batch's statistics, so whatever forward scale my conv produces is erased before the next
conv — the forward pass is, to first order, indifferent to my conv variance. The gradient is not. The gradient
that SGD applies to conv `l` is (schematically) the upstream gradient routed back through `W_{l+1}, W_{l+2},
…` via their transposes, and each of those transposes multiplies the gradient variance by `fan_out · Var(W)`.
If that product is far from one, the gradient reaching the early convs is either exponentially small — and SGD
with a fixed `lr = 0.1` crawls, moving those weights by nothing for many epochs — or exponentially large, and
SGD thrashes, taking steps so large the loss oscillates. Pinning `fan_out · Var(W) = 2 · (1/2) = 1` after the
ReLU half is exactly what keeps that backward product at unit gain layer over layer, so the gradient arrives at
every depth at the *same* scale and a single global learning rate is appropriate everywhere at once. That is a
property BN cannot hand me on the first step — BN's own backward correction depends on running statistics that
have not warmed up yet — and it is the concrete content of "initialization still owns the gradient scale."

I have to be honest about what `fan_out` does on the depthwise convs, because it is exactly the layer where
the two modes diverge most and it is easy to say something false about it. A depthwise conv's weight is
`(channels, 1, 3, 3)`, and PyTorch's fan bookkeeping — which counts input feature maps as `weight.size(1) = 1`
and does not divide by groups — gives `fan_in = 1·9 = 9` but `fan_out = channels·9`. I checked a concrete
one: a 96-channel depthwise layer reports `fan_in = 9`, `fan_out = 864`. The *physically correct* fan for a
depthwise filter is 9 — each output channel is fed by exactly nine weights, its own 3×3 tap — so `fan_in = 9`,
giving `Var(W) = 2/9 ≈ 0.222` (std `0.471`), is the variance that would hold a depthwise layer's forward
variance at its fixed point. My choice, `fan_out = 864`, gives `Var(W) = 2/864 ≈ 0.0023` (std `0.048`) — about
`√96 ≈ 10×` *smaller* in scale. So on the depthwise layers `fan_out` under-scales the filters by an order of
magnitude, and I should not pretend otherwise. Why do I accept it? Two reasons, both concrete. First, every
depthwise conv is immediately followed by BN, which re-standardizes its output to unit variance regardless of
how small I made the filter, so the forward under-scaling is invisible one layer later — this is the same "BN
owns the forward variance" fact I am leaning on everywhere. Second, the layers where the mode choice actually
carries the network's parameters and gradient are the *pointwise* 1×1 convs — MobileNetV2's expansions and
projections — which hold the overwhelming majority of its weights and are genuine dense (channel-mixing) maps
where `fan_out` correctly protects the backward scale. So `fan_out` is right for the layers that matter most
and merely-cushioned-by-BN on the cheap depthwise layers, and special-casing depthwise to `fan_in` would be
exactly the architecture branching I am deliberately refusing at this rung, whose entire job is to test one
uniform second-moment rule. The improvement over orthogonal on depthwise does not even depend on which fan I
pick: what broke orthogonality was that it orthonormalized a *fictitious* cross-channel axis and set no
per-filter scale at all; He, in either mode, treats the nine depthwise weights as nine real Gaussian draws
with a fan-derived variance and lets BN standardize the result — a coherent per-filter init where orthogonal
had none.

For the `Linear` classifier head I split the other way and use **`fan_in`**, and the reason is that the head
is the one place with no BN after it. It feeds directly into the softmax / cross-entropy, so there is no
downstream standardizer to rescue its forward scale — I have to control the *pre-logit* variance myself, and
`fan_in` is the mode that holds the forward variance at its fixed point. If I let the head over-scale, the
logits come out large, the softmax starts saturated, and the early gradient is distorted before training has
moved. VGG's hidden FC is the clearest case: it feeds a ReLU (then dropout, then the output FC), so `fan_in`
with the ReLU gain keeps its pre-activation variance at unit and the ReLU stage behaves like the conv stages.
There is one small imperfection I will accept rather than special-case: the *final* output FC is not followed
by a ReLU, so applying `nonlinearity='relu'` (the √2 gain) over-scales its logits by √2. On CIFAR-100 the head
maps a 64-wide feature (ResNet) or a 512-wide hidden (VGG) into 100 logits, so even with the spurious √2 the
logit standard deviation is `O(1)` — the softmax is not saturated — and the cost of the imperfection is a
constant factor the first few steps of SGD wash out. Keeping one uniform `Linear` rule is worth that; carving
out the last layer would be the same branching I am avoiding on the convs. Biases go to zero throughout, for
the same reason as before: no unit deserves an offset at init.

BatchNorm stays at `(γ=1, β=0)`, the same neutral identity affine I used for orthogonal, and for the same
reason: at this rung BN's job is to re-standardize, and the identity affine lets it do that without amplifying
or suppressing. I am explicitly *not* doing anything residual-aware yet — the residual accumulation problem
stays deliberately untouched at this rung — because the point of this rung is to isolate whether the simple,
correct, BN-friendly second-moment target
already matches the orthogonal spectrum that cost so much to impose. If it does, that is strong evidence the
full-spectrum machinery was redundant, and it sets up the next rung cleanly: the residual *accumulation*
problem, which neither orthogonal nor plain He touches, becomes the obvious remaining lever.

Walk the three architectures and predict against the orthogonal numbers I now have in hand. **MobileNetV2** is
where I expect the clearest swing in my favor, and I can name the mechanism exactly. Its depthwise convs — the
ones whose tall `(C, 9)` matrices broke orthogonality by orthonormalizing a cross-channel axis the operator is
indifferent to — are handled *coherently* by He: each depthwise filter gets a per-filter Gaussian with a
fan-derived variance, and BN standardizes the result, so where orthogonal had to degrade to "nine
well-conditioned directions of a fictitious matrix," He just draws nine real weights and moves on. So I expect
kaiming_normal to *beat* orthogonal's 93.88 on MobileNetV2 — this is the architecture where the orthogonal
scheme was actively misapplied and the simpler target is strictly more appropriate. If it does *not* beat
93.88, my whole "the spectrum wasn't doing the work" diagnosis is wrong and I should reconsider whether
depthwise layers want something stranger than either init. **VGG-16-BN** is the closest call, and it is the
one that tests my BN-crack argument directly. On the plain stack the full spectrum is well-defined and its
off-diagonal conditioning is the one thing BN's diagonal rescaling cannot supply, so orthogonal's 72.83 might
genuinely hold a slim edge here. I expect He to be competitive but I am *not* confident it beats orthogonal on
VGG — if it does, the off-diagonal conditioning bought nothing even on its home turf and the orthogonal rung
was strictly dominated; if orthogonal holds, the spectrum has a narrow, real, plain-stack-only value. I would
rather state that uncertainty honestly than predict a win I have not earned. **ResNet-56** I expect to be
roughly a wash with orthogonal's 72.08: both schemes get the per-layer second moment right (orthogonal via √2
gain, He via `√(2/fan)`), and *neither* addresses the residual accumulation — the additive growth of the main
path over 27 blocks — that is the dominant effect in a 56-layer res-net. So they should land in the same
neighborhood, and whichever wins by tenths is noise rather than mechanism. That ResNet wash is itself the most
informative outcome I can hope for, because it isolates the lever the next rung must pull: if two schemes that
differ *maximally* in their per-layer spectrum — one pinned to a single singular value, one scattered across
Marchenko–Pastur — tie on the residual net, then per-layer spectrum is provably *not* what limits the residual
net. Residual accumulation is.

So the step-2 edit is the literal He fill, with the role-split modes: iterate `model.modules()`; for every
`Conv2d` apply `kaiming_normal_(mode='fan_out', nonlinearity='relu')`; for every `Linear` apply
`kaiming_normal_(mode='fan_in', nonlinearity='relu')` with zero bias; for every `BatchNorm2d` set
`(weight=1, bias=0)`. One uniform pass, no architecture branching, no depth arithmetic. The distilled module
and its exact code live in the answer.

The falsifiable expectations against the orthogonal rung, stated plainly so the next rung has a clean
scoreboard to react to. (1) On MobileNetV2 I expect kaiming_normal to *beat* orthogonal's 93.88, because He
gives the depthwise filters a coherent per-filter init where orthogonality gave them a fictitious
orthonormalization — if it does not, my "the spectrum wasn't doing the work" reading is wrong. (2) On
VGG-16-BN I expect a near-tie, genuinely uncertain which way it falls, with orthogonal's 72.83 possibly
holding a slim edge from the off-diagonal spectral conditioning BN cannot supply on the plain stack — if He
clearly *wins* here too, the spectrum buys nothing even where it should and the orthogonal rung was strictly
dominated. (3) On ResNet-56 I expect a wash around 72.0–72.4, because both fix per-layer variance and neither
fixes residual accumulation — and that wash is the signal that the residual net's ceiling is set by
accumulation, not by per-layer scale. If all three land as expected, the lesson is sharp: the cheap, correct,
BN-friendly second moment matches or beats the expensive full-spectrum scheme everywhere except possibly the
one plain stack, which means the next move is not a better per-layer init at all — it is to finally attack the
residual-accumulation problem that has sat untouched under both rungs, the additive growth of the main-path
sum over 27 blocks that no per-layer second moment can reach and that inflates the signal with depth.
