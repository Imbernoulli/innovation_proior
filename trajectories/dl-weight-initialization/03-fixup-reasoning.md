The Kaiming numbers landed almost exactly on the three predictions, and the one that matters most is the
ResNet wash. On MobileNetV2 He jumped to 94.49 from orthogonal's 93.88 — the depthwise convs got their
correct per-filter variance and the swing was real, confirming the spectrum was never the thing earning points
there. On VGG-16-BN He came in at 73.38, comfortably ahead of orthogonal's 72.83 — so the plain stack's
spectral conditioning, which I thought might hold a slim edge, did *not* survive contact; the cheap correct
second moment beat the expensive pinned spectrum even on its home turf. But ResNet-56 came in at 72.07,
essentially identical to orthogonal's 72.08 — a dead wash, two schemes that differ entirely in their per-layer
spectrum tying to within one part in seven thousand. That tie is the whole signal. When two initializers that
disagree about the per-layer map agree on the residual net, per-layer scale is provably *not* what limits the
residual net. The ResNet ceiling is set by something neither rung touched, and I now know what it is: residual
accumulation.

Make the accumulation problem precise, because the fix has to come out of its structure. A ResNet main path is
a running sum: `x_{l+1} = x_l + F_l(x_l)`, where `F_l` is a residual branch — for `BasicBlock`, two 3×3
convs each with BN. At initialization, both He and orthogonal give each branch an output `F_l(x_l)` whose
variance is roughly the same scale as `x_l` (that is what "preserve the per-layer variance" *means*). But the
branches *add*, and the branch outputs at init are roughly independent, so the variance of the running sum
grows additively: after `L` blocks the main-path variance is roughly `Var(x_0) + Σ_l Var(F_l) ≈ (1+L)·Var(x_0)`.
ResNet-56 on CIFAR has 27 `BasicBlock`s, so the signal entering the head is inflated by a factor on the order
of the depth. Three things follow at the start of training. The logits are over-scaled, so the softmax starts
saturated and the early gradient is distorted. BN does re-standardize the running sum each block — that is why
the network trains at all — but it is forced to do so by *dividing out* a large, depth-dependent variance,
which couples every block's BN to the depth and means the residual branches start out *competing* with the
identity path on equal footing rather than as small corrections. And the deepest blocks' branches, whose
contribution is buried in a large sum, get a poorly-scaled gradient. That is the accumulation tax, and no
per-layer second-moment choice can pay it, because the problem is *between* layers — in how the branches sum —
not *within* any one.

The remedy that the residual structure itself suggests is to make each branch *start small* so the running sum
does not inflate, and to start it from the identity rather than from a random function so SGD does not have to
first unlearn a bad random branch before it can use it. The cleanest way to say "start small" with the right
depth dependence comes from the residual-scaling analysis of Zhang, Dauphin & Ma (2019): if I want each
branch's contribution to the network function to be `Θ(1/L)` rather than `Θ(1)` — so that the sum over `L`
branches is `Θ(1)` again, depth-independent — and the branch has `m` weight layers, then scaling the branch's
weight layers by `L^{-1/(2m-2)}` achieves it. For the `BasicBlock` here, `m = 2` (two convs per branch), so
the exponent is `-1/(2·2-2) = -1/2`: scale by `L^{-1/2}`, i.e. `n_blocks^{-1/2}`. That is the load-bearing
move — controlling the *accumulation* directly, which is exactly the lever the ResNet wash exposed as untouched.

But here I have to be careful, because the substrate I am editing is *not* the normalization-free setting that
analysis was built for, and importing the wrong machinery would break things. The original residual-scaling
recipe was designed to train res-nets *without any normalization at all*: it removes BatchNorm entirely,
replaces it with learnable scalar biases inserted before every conv and activation, adds one learnable scalar
multiplier per branch, scales the *non-zero* branch weights by `L^{-1/(2m-2)}` while zero-initializing the
*last* conv of each branch, and zero-initializes the classifier. Every one of those pieces exists to
substitute for what BN would have done. But this task's networks **keep BatchNorm** — it is part of the frozen
model graph I am forbidden to touch — and I cannot add scalar biases or multipliers without altering the graph,
which the contract prohibits. So I must *not* import that story. I cannot add learnable scalars (graph change),
I cannot remove BN (graph change), and I should not zero the classifier (with BN present and the head's logits
already controlled by the `fan_in` scaling I kept, zeroing the head would only throw away a good start). What I
can do, and what is the *right* translation of the residual-scaling idea into a BN-equipped network, is two
edits, both inside the contract.

First, the branch down-scaling. I run the He pass first — every conv gets `N(0, √(2/fan_out))`, every Linear
`fan_in` He, every BN neutral — so I keep the step-2 second moment that just proved itself on VGG and Mobile.
Then, for ResNets only, I find the residual blocks and scale the *last* conv of each block (`conv2`) by
`n_blocks^{-1/2}`. This shrinks each branch's output by the depth-corrected factor so the running sum stops
inflating. I scale `conv2` specifically — the last weight layer in the branch — because that is the layer
whose output directly enters the additive sum, so scaling it is the most direct way to set the branch's
contribution scale; and I scale it *down from a good He init* rather than zeroing it, because with BN in the
loop I want the branch to be a small but live correction, not a dead one.

Second — and this is where the BN-equipped translation gets to do something the normalization-free recipe
cannot — I exploit BatchNorm itself to start the branch near identity, for free, with no graph change. In a
`BasicBlock` the branch ends `conv2 → bn2`, and the block output is `relu(bn2(conv2(...)) + shortcut(x))`. The
affine `weight` (γ) of `bn2` multiplies the *entire* branch output before it is added to the shortcut. So if I
set `bn2.weight = 0` at initialization, the branch output is multiplied by zero and the block computes exactly
`relu(shortcut(x))` — the residual branch starts as the *zero function* and the block starts as a clean
identity-plus-shortcut, precisely the "start from identity, don't unlearn a random branch" property I wanted,
achieved through a parameter BN already owns. This is the zero-γ trick (Goyal et al. 2017): zero the last BN's
scale in each residual block so residual branches begin inert and SGD grows them from zero. It is strictly
better than zero-initializing the *conv* would be here, because the conv weights stay at their He scale (ready
to contribute the moment γ lifts off zero) while the *block output* is what starts at identity — and crucially
γ is a single learnable scalar per channel that BN already exposes, so I change no graph and add no parameters.

So the task's residual init is a hybrid the normalization-free recipe never was: keep BN, keep the He second
moment, then for ResNets (a) down-scale the last branch conv by `n_blocks^{-1/2}` to kill accumulation and
(b) zero the last branch BN's γ to start each branch from identity. The branch-scaling exponent comes from the
`m=2`, `Θ(1/L)`-accumulation argument; the zero-γ start comes from BN's own affine. I am explicitly *not*
adding scalar biases or multipliers, *not* removing BN, *not* zeroing the classifier — those belong to the
no-normalization world and would violate the frozen-graph contract here.

For the non-residual architectures the answer is "do nothing extra," and it is worth saying why rather than
treating it as a default. **VGG-16-BN** has no shortcuts and no additive accumulation — it is a plain chain, so
there is no branch to scale and no last-branch-BN to zero; the He pass *is* the right init, and it already won
there at 73.38, so I leave it untouched. **MobileNetV2** does have additive shortcuts in its
`InvertedResidual` blocks, but its block structure is different (expand-1×1 → depthwise-3×3 → project-1×1,
three convs, with the residual added only when stride 1 and channels match), and the leaderboard's strongest
He result already lives here at 94.49; rather than guess a branch-scaling exponent for a block type the
`n_blocks`/`BasicBlock` accounting wasn't derived for, I gate the residual scaling on `arch.startswith('resnet')`
and let Mobile and VGG keep the plain He init that is already best for them. The honest scope of the residual
fix is the architecture whose accumulation I can actually count: ResNet.

So the step-3 edit, literally: phase one, the full He pass from step 2 (convs `fan_out`, Linear `fan_in`, BN
neutral, zero biases); phase two, *if* `arch` is a ResNet, count `n_blocks` = number of `BasicBlock`s,
multiply each block's `conv2.weight` by `n_blocks^{-1/2}`, and set each block's `bn2.weight` to zero. No graph
change, no added parameters, no data, no calibration. The distilled module and its exact code live in the
answer.

The falsifiable expectations against the Kaiming rung, where this either earns the top of the ladder or
doesn't. The decisive test is **ResNet-56**: it sat in a dead wash at 72.07/72.08 under the two per-layer
schemes precisely because neither addressed accumulation, and this rung is the *only* one that does — so if
the residual-scaling-plus-zero-γ story is right, ResNet-56 should finally move *up* off that 72.07 ceiling. If
it doesn't, then accumulation wasn't the binding constraint after all and my reading of the wash is wrong. On
**VGG-16-BN** I expect essentially no change from He's 73.38 in the worst case, because the residual phase
never fires there — any movement is plain seed-level noise, and a clean tie-or-better is the success
condition (a *drop* would mean my phase-one pass somehow diverged from step 2, which it must not). On
**MobileNetV2** the residual phase is gated off too, so I expect to hold He's 94.49 — again, holding is the
success condition, not improving. The shape of the win I am predicting is therefore narrow and specific: the
top of this ladder is won *on the residual net*, by paying the accumulation tax that the per-layer rungs left
on the table, while matching the best per-layer result everywhere the residual machinery correctly stays out
of the way. If ResNet-56 rises and the other two hold, this is the strongest rung; if ResNet-56 stays at 72.07,
the lesson would be that BN already neutralizes accumulation well enough that initialization cannot improve on
it, and the next move would have to look past initialization entirely.
