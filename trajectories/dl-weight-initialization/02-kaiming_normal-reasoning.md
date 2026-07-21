The orthogonal numbers came in close to where the construction predicted, and the pattern *is* the diagnosis I
need. ResNet-56 on CIFAR-100 at 72.08, VGG-16-BN on CIFAR-100 at 72.83, MobileNetV2 on FashionMNIST
at 93.88. I have to be disciplined about what compares to what: FashionMNIST is a different, easier task, so
93.88 is only an anchor the next method must clear there, not on the CIFAR axis. But ResNet and VGG are both
CIFAR-100, same schedule, so their gap is fair — VGG beat ResNet by 0.75. Part of that is VGG simply being a
stronger model on CIFAR-100, not the init, so I won't over-read the magnitude; but the *sign* is exactly what
the orthogonal fill predicted — orthogonality best on the plain stack it was built for, ordinary on the residual net whose
accumulation it never touched. So the qualitative story holds, and the three numbers become per-task anchors
to beat. What the table does *not* show is that the expensive part of the orthogonal scheme earned its keep:
it bought a topology-dependent showing, strong where the topology matched and weakest on the depthwise net
where the isometry could not even be constructed. That hints — not proves — that the full-spectrum control was
not the thing doing the work, and on two of three architectures was either redundant or unattainable.

The redundancy is the realization that reframes the problem. Every conv in all three networks is followed by
BatchNorm, which re-standardizes the conv output to zero mean, unit variance per channel before the next layer
sees it. So orthogonality's guarantee that the *per-channel activation variance* is preserved layer to layer
is precisely what BN already enforces everywhere, for free, regardless of the conv init. I flagged one crack
last time — BN is diagonal, so it cannot fix the *off-diagonal* conditioning of the weight map, which is why
orthogonality might hold a thin edge on the plain stack — and the VGG-vs-ResNet gap is faintly consistent with
that. But faint is the word: the spectrum did not lift the residual net and could not be built on the depthwise
net. What BN emphatically does *not* fix, and what initialization still owns, is the very first forward and
backward pass before BN's running statistics have warmed up, and — more load-bearing — the *scale of the
gradient* SGD sees at step zero. If the conv weights are scaled wrong the returning gradient is scaled wrong,
and the first few hundred SGD steps crawl or thrash whatever BN does to the forward activations. So the right
target is not the full spectrum; it is the second moment, cheaply and uniformly — exactly what He was built
for and what I set aside to test orthogonality first.

Do the variance properly for ReLU. One conv `y = Wx` (bias zero) feeding a ReLU: the forward recursion across
a `conv→relu` stage is `Var(next) = ½ · fan · Var(W) · Var(prev)`, the one-half from the previous layer's
rectified-Gaussian activation keeping only half its variance. To hold `Var(next) = Var(prev)` you need
`Var(W) = 2/fan`. That factor of two is the whole content of He over Xavier — it pays back exactly what ReLU
takes. So each conv weight is drawn `N(0, √(2/fan))`. This matches the orthogonal-√2 target second moment
(row-orthonormal times √2 has per-entry RMS `√(2/fan_in)`, `≈ 0.0589` on the 576-fan layer) but with a plain
Gaussian spectrum instead of a pinned one — and the orthogonal results give no evidence the pinned spectrum
earned the points.

The one non-default decision here is worth real thought: `fan_in` or `fan_out`. For a conv `fan_in = in·k²`,
`fan_out = out·k²`; `fan_in` holds forward activation variance at its fixed point, `fan_out` holds backward
gradient variance, since the backward pass multiplies by `Wᵀ` whose second-moment map scales by
`fan_out · Var(W)`. He showed either works for a feed-forward ReLU net, and the reason tells me which to pick.
Track forward variance under `fan_out`: layer `l` multiplies it by `½ · fan_in_l · Var(W_l) = fan_in_l/fan_out_l
= in_l/out_l`. But channels chain, `in_{l+1} = out_l`, so over the stack the product telescopes:
`in_1/out_1 · out_1/out_2 · … = in_1/out_L`, a single bounded constant, *not* a factor that compounds with
depth. So `fan_out` leaves forward variance off by one constant BN absorbs on its first pass, while pinning the
backward gradient variance exactly (symmetrically for `fan_in`). That telescoping is the real content of
"either works," and it shows the decision is smaller than it looks: for *square* convs `fan_in = fan_out`
identically, so wherever the channel count is fixed — the bulk of both ResNet and VGG — the two modes are
literally the same numbers. The choice only bites where channels change (stage transitions, VGG's width steps,
and most sharply the depthwise convs). Since BN already pins forward per-channel variance everywhere, I protect
the **backward gradient scale** — `fan_out` for the convs — because controlling the backward scale is the
higher-value thing initialization can still do, and BN cannot do it on the first step. On the actual VGG widths
the accumulated forward-variance error from `fan_out` telescopes to `in_1/out_L = 3/512 ≈ 0.006`, a single
one-time constant BN standardizes, not a per-layer decay — concrete reassurance `fan_out` is safe forward.

I have to be honest about `fan_out` on the depthwise convs, the layer where the modes diverge most. A depthwise
weight is `(channels, 1, 3, 3)`, and PyTorch's fan bookkeeping counts input maps as `weight.size(1) = 1`
without dividing by groups, so `fan_in = 9` but `fan_out = channels·9` — a 96-channel depthwise layer reports
`fan_in = 9`, `fan_out = 864`. The physically correct fan is 9 (each output channel fed by its own nine taps),
giving `Var(W) = 2/9 ≈ 0.222`; my choice `fan_out = 864` gives `2/864 ≈ 0.0023`, about `√96 ≈ 10×` smaller. So
`fan_out` under-scales the depthwise filters by an order of magnitude, and I accept it for two concrete reasons.
First, every depthwise conv is immediately followed by BN, which re-standardizes its output regardless of how
small the filter is, so the under-scaling is invisible one layer later. Second, the layers that actually carry
MobileNetV2's parameters and gradient are the *pointwise* 1×1 convs — the expansions and projections — genuine
dense channel-mixing maps where `fan_out` correctly protects the backward scale (a `96 → 24` projection has
`fan_in = 96`, `fan_out = 24`, a 4× gap right where I want the backward scale pinned). So `fan_out` is right
for the layers that matter and merely BN-cushioned on the cheap ones, and special-casing depthwise to `fan_in`
would be exactly the architecture branching this pass refuses. The gain over orthogonal on depthwise doesn't
even depend on the mode: orthogonality orthonormalized a fictitious cross-channel axis and set no per-filter
scale; He, either mode, gives the nine depthwise weights a real fan-derived variance and lets BN standardize —
a coherent per-filter init where orthogonal had none.

For the `Linear` classifier head I split the other way and use **`fan_in`**, because the head is the one place
with no BN after it — it feeds straight into the softmax, with no downstream standardizer to rescue its forward
scale, so I have to control the pre-logit variance myself, and `fan_in` is the mode that holds forward variance
at its fixed point. Over-scale it and the logits come out large, the softmax starts saturated, the early
gradient distorted. One imperfection I accept rather than special-case: the *final* output FC is not followed by
a ReLU, so the `nonlinearity='relu'` √2 gain over-scales its logits by √2 — but mapping a 64-wide (ResNet) or
512-wide (VGG) feature into 100 logits, even with the spurious √2 the logit std is `O(1)`, softmax unsaturated,
and the constant washes out in a few steps. Keeping one uniform `Linear` rule is worth that; carving out the
last layer is the same branching I avoid on the convs. Biases zero throughout. BN stays `(γ=1, β=0)`, neutral,
and I am explicitly *not* doing anything residual-aware yet — this pass isolates whether the simple correct
BN-friendly second moment already matches the orthogonal spectrum, leaving the residual *accumulation* problem
as the obvious next lever if it does.

Predict against the orthogonal numbers in hand. On **MobileNetV2** I expect the clearest swing in my favor: its
depthwise convs, which broke orthogonality, get a coherent per-filter He init and BN standardizes the result,
so I expect to *beat* 93.88 — and if I don't, the "spectrum wasn't doing the work" reading is wrong and
depthwise layers want something stranger than either init. On **VGG-16-BN** it is a genuine near-tie: the plain
stack is where orthogonal's off-diagonal conditioning could hold a slim edge BN can't supply, so I am *not*
confident He beats 72.83 — if it clearly wins here too, the spectrum bought nothing even on its home turf. On
**ResNet-56** I expect a wash around orthogonal's 72.08, because both schemes fix the per-layer second moment
and *neither* touches residual accumulation over 27 blocks — and that wash is the most informative outcome,
because if two schemes that differ maximally in per-layer spectrum tie on the residual net, per-layer spectrum
is provably not what limits it. Residual accumulation is.

So the step-2 edit: iterate `model.modules()`, `kaiming_normal_(mode='fan_out', nonlinearity='relu')` on every
`Conv2d`, `kaiming_normal_(mode='fan_in', nonlinearity='relu')` with zero bias on every `Linear`, `(1, 0)` on
every `BatchNorm2d` — one uniform pass, no branching. If all three land as expected, the lesson is sharp: the
cheap correct second moment matches or beats the expensive spectrum everywhere except possibly the plain stack,
so the next move is not a better per-layer init but attacking the residual accumulation both attempts have left
untouched. The full module is in the answer.
