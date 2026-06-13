The block is the whole game, but every block is a variation on one fill of the scaffold, so the place to
start is the plainest non-trivial change I can make to the default and the one I most trust not to hurt:
put a single learnable scalar on the residual branch. Before I reach for anything structural — reordering
the normalization, dropping blocks, attention — I want to know what the *minimal* deviation from the
default block does, because that calibrates everything after it. The default is the post-activation basic
block: two 3×3 conv–BN layers with a ReLU between them, the shortcut added, and a final ReLU after the
addition, `H = ReLU(F(x) + shortcut(x))`. It already works; the question is whether the network is getting
the *amount* of residual right, block by block, and whether I can let it choose.

Let me be precise about what "the amount of residual" means and why it might matter. A residual block
commits to a fixed mixing rule: the branch output `F(x)` is added to the shortcut at full weight, 1·F(x).
That weight is a hard architectural constant, identical in the first block and the last, identical at depth
20 and depth 110. But the *right* contribution of a block is surely not constant. Early blocks, building
low-level features every later block depends on, may want to write strongly; deep blocks, refining an
already-rich representation, may want to write only a small correction — exactly the picture where the
optimal `F` is "near zero, a gentle nudge." With the weight pinned at 1, the only way the network can dial
a block down is to drive all of `F`'s conv weights small, fighting the same conditioning the residual
reformulation was supposed to relieve, and it has to do that with the *full* convolutional machinery rather
than one knob. So the minimal hypothesis is: give each block one extra degree of freedom — a scalar `alpha`
multiplying its residual branch, `H = ReLU(shortcut(x) + alpha · F(x))` — and let SGD learn how loud each
block should be. One parameter per block, negligible cost, and it sits exactly on the axis I suspect the
fixed block gets wrong.

Now the design decision that actually defines this rung, and it is *not* the one the obvious reference
would push me toward. The natural place to look for "scale the residual by a learnable scalar" is the line
of work on initializing that scalar to **zero** — start every block as a pure identity, `H = ReLU(x)`,
and let the gates open up during training so a freshly-initialized deep net begins life perfectly
signal-preserving. That story is about *trainability at extreme depth*: at zero-init the forward map is the
identity and the input-output Jacobian is exactly the identity, so gradients propagate without warm-up or
normalization, and people have used it to train networks thousands of layers deep. It is a real and
attractive idea. But I have to ask whether it fits *this* substrate, and when I trace it through, it does
not — and the place where it breaks is the load-bearing decision of this whole rung.

Here is the mismatch. That zero-init story assumes the residual branch is the *only* thing standing
between input and output — that with `alpha = 0` the block collapses to a clean identity. In this scaffold
the block is **post-activation**: there is a ReLU *after* the addition, and inside the branch there is BN
after the last conv. So even with `alpha = 0` the block is not the identity — it is `H = ReLU(shortcut(x))`,
and on the stride-1 blocks where the shortcut is a bare identity that is `ReLU(x)`, which is only the
identity because the block input happens to be non-negative (it is the output of a prior ReLU). More to the
point, the substrate already *has* its trainability solved: BN after every conv keeps the forward and
backward signals healthy at depth 110, and the schedule is a fixed cosine over 200 epochs with no warm-up
to remove. The disease that zero-init cures — an un-normalized deep net that can't start — is not present
here. If I zero-init `alpha`, I am instead *throwing away* the first stretch of training: every block
starts contributing nothing, the network is effectively shallow for as long as it takes the gates to lift
off zero, and against a fixed 200-epoch budget that is wasted depth, not bought depth. The zero-init move
trades early capacity for early stability I don't need.

So I init the gate at **one**, not zero. At `alpha = 1` the block is *bit-for-bit the default scaffold
block* — `H = ReLU(shortcut(x) + 1·F(x))` is exactly `H = ReLU(shortcut(x) + F(x))`. The network begins
training as the proven post-activation ResNet, full depth from step zero, and the gate is a *correction*
the optimizer is free to apply: it can pull a block down below 1 if that block is over-writing, or push it
above 1 if a block wants to write louder than the default allows. This is the right framing for the
substrate I have. I am not buying trainability — BN already gave me that. I am buying the ability to
*re-weight* each block's contribution around the default, starting from the default, so the worst case is
"the gates stay near 1 and I recover the baseline" and the upside is "some blocks find they should be
quieter or louder." Initializing at 1 makes the floor of this rung the baseline itself; initializing at 0
would make the floor strictly worse and bet the whole rung on the gates lifting fast enough. Given a fixed
budget and an already-trainable substrate, 1 is the only defensible choice — and it is the one decision
that separates this rung from the zero-init lineage it superficially resembles.

Where exactly does the scalar multiply? I want it on the residual branch and on nothing else — the
shortcut must stay un-gated, because the entire value of the identity path is that it is *always* open,
carrying signal and gradient untouched; gating the shortcut would reintroduce the Highway failure where a
drifting gate can close the highway. So the scalar multiplies `F(x)` after its final BN and *before* the
addition, leaving `shortcut(x)` at full strength. And it goes *inside* the post-activation structure: I
keep the block's own ordering exactly as the default has it — `out = ReLU(BN(conv1(x)))`, then
`out = BN(conv2(out))`, then `out = shortcut(x) + alpha · out`, then a final `ReLU` on the sum. I am
deliberately *not* touching the activation order on this rung; reordering BN/ReLU is a different lever I
want to hold constant so the only changed variable is the scalar gate. One scalar, one branch, the
post-activation block otherwise identical to default.

The implementation is a `nn.Parameter` of shape `(1,)` initialized to ones, registered so SGD trains it
alongside the convs (and so weight decay touches it — a mild pull of the gate toward zero, which is
benign here: it just gives the network a small standing incentive to quiet blocks it isn't using, the same
incentive that already acts on the conv weights). It is a single per-block scalar, broadcast across
channels and space — not a per-channel vector, because the hypothesis I'm testing is the coarse one,
"how loud is this *block*," and a per-channel gate would start to overlap with channel-recalibration ideas
I'd rather isolate to a later, dedicated rung. The full scaffold module is in the answer.

So the edit relative to the default is exactly: declare `self.alpha = nn.Parameter(torch.ones(1))` in the
constructor, and change the addition line from `out += self.shortcut(x)` to
`out = self.shortcut(x) + self.alpha * out`. Everything else — the two convs, the two BNs, the inter-conv
ReLU, the dimension-matching shortcut, the final ReLU — is the scaffold default, untouched. That is the
literal scaffold edit, and it is deliberately the smallest meaningful one.

Now reason about what this floor should do, because that is the point of running it first. At
initialization the network *is* the baseline post-activation ResNet, so anything the gates do is a
departure the optimizer chose. The honest expectation is modest. On the easy, shallow setting — ResNet-20
on CIFAR-10 — the baseline block is already near the ceiling of what 20 layers can extract from a 10-class
problem, and a per-block scalar has little structural slack to exploit; I'd expect the gates to settle
close to 1 and the accuracy to land right around a well-tuned vanilla ResNet-20, with any gain coming from
a handful of blocks discovering they should write a little more or a little less. On the deeper CIFAR-100
settings the scalar has *more* to do, because there are more blocks and more reason for their contributions
to differ — but the same property that makes this rung safe makes it weak: a single scalar per block is a
very low-bandwidth correction. It can rescale a block but it cannot change *what* the block computes, and
it cannot help where the bottleneck is the activation ordering or feature reuse rather than the residual
magnitude. So I expect this to roughly match or slightly edge the default block everywhere, strongest on
the shallow easy case where "match the baseline" is itself a high number, and to leave the deeper nets
wanting — precisely because the thing limiting a 56- or 110-layer net is more likely how cleanly gradients
and features pass *through* the identity path, which a residual-branch scalar does not touch.

That diagnosis is already the pointer to the next rung. If the gated block lands near the baseline and the
deep CIFAR-100 nets are where the room is, then the lever to reach for next is not *how much* residual each
block adds but *how* the block is wired around the identity path — the order of BN and ReLU relative to
the convolutions and the addition — because that is what governs whether the shortcut is a truly clean
highway at depth. The expectation I am setting, falsifiable against the numbers this rung returns: the
gated block should sit close to a strong vanilla ResNet on all three settings, best on ResNet-20/CIFAR-10,
with the deep CIFAR-100 nets showing the most unexploited headroom — the gap a re-ordering of the block's
internals, not a rescaling of its branch, would be the natural thing to attack.
