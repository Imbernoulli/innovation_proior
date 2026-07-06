Whitening the first conv cut me from 45 epochs to 21, from 18.3 to 8.0 A100-seconds — the input is now
well-conditioned, and the numbers tell me *how* the win arrived, which is what I want to read before
deciding where to attack next. The epoch count fell 45 → 21, a factor of 2.14. The wall-clock fell 18.3
→ 8.0, a factor of 2.29. Those are not quite the same number, and the small discrepancy is informative:
per-epoch cost went from 18.3/45 = 0.407 s to 8.0/21 = 0.381 s, about 6–7% cheaper per epoch. That
7% is exactly the flavour of saving I expected from freezing the first layer — once its weights and,
later, its bias stop taking gradient, the backward pass has slightly less to do. So the decomposition is
clean: roughly 93% of the whitening speedup was *removing epochs* (the conditioning win) and about 7% was
*cheaper epochs* (the frozen backward). The lesson for what comes next is that the remaining 8.0 seconds
is still overwhelmingly epoch-bound — it is 21 epochs at 0.381 s each — so if I want another big cut I
should again be hunting for *epochs I don't need to run*, not for a faster per-step kernel. At ~49 steps
per epoch (50,000 images / batch 1024 ≈ 48.8), 21 epochs is only about a thousand optimizer steps; every
epoch I can shave is worth a lot.

So where is the next block of wasted epochs? The whitening win was concentrated entirely at the input
layer, and it was concentrated there because that is where I looked. Everything *after* the first conv is
still randomly initialized, and I suspect the same kind of waste I just removed at layer one is hiding in
the deeper layers, only wearing a different disguise. Let me look at what the deeper convolutions are
being asked to do in the first few epochs.

The network is a stack: whitened first conv, then three blocks, each block two or more 3×3 convs with
BatchNorm and GELU — call it six trainable convs after the whitening layer. Standard initialization
(Kaiming/Xavier) sets each of those convs to small random weights, scaled so that activation variance is
preserved through the layer. Preserving variance is necessary — it is what keeps signals from exploding
or vanishing as they pass through a deep stack — but it is a statement about *magnitudes*, not about
*content*. A variance-preserving random init does not preserve the signal; it *rotates* it. Concretely, a
Kaiming-initialized 3×3 conv from C channels to C channels has 9C² weights drawn roughly iid with
variance 2/(9C); the operator it implements at step zero is close to a random orthogonal-ish map, which
sends the input into a random basis while keeping its norm. So at initialization every deep layer is a
random scrambling of its input. The clean, whitened signal that the first conv worked so hard to present
gets immediately mangled by the second layer's random rotation, then again by the third, then again —
six random rotations stacked before any of it reaches the head. And the network then has to *un-scramble*
them: it has to learn, layer by layer, to first approximate passing information *through* before it can
learn to transform that information usefully. A deep random init does not merely fail to help — it
actively destroys the input structure and forces the network to rebuild a path from scratch.

How far is a random init from a passthrough, quantitatively? The identity operation for a 3×3 conv is a
very specific, very *sparse* tensor: a kernel that is 1 at the center tap of each diagonal input→output
channel and 0 everywhere else — exactly C nonzero entries out of the 9C² weights. A Kaiming random init
has essentially none of its mass there; it is an O(1)-operator-norm distance away from that sparse target
in a 9C²-dimensional weight space. So each deep layer starts a full random rotation away from "do
nothing," and gradient descent must walk it back toward that sparse target before the layer can be useful
— across six layers, that is six independent walks-toward-identity happening in the early epochs, none of
which is about the labels. That is the deep-layer analogue of the input-decorrelation overhead whitening
just removed.

What I actually want at initialization is the opposite of a random scramble. I want each deep conv to
*pass its input through unchanged* at step zero — to be the identity — so the clean whitened signal
propagates all the way to the head untouched, and gradient descent only has to learn the *deviations*
from identity that help, rather than first learning identity and then learning the deviation on top of
it. This is the same logic as a residual connection (start near identity, learn the correction), and that
gives me a design choice worth walking, because there is more than one way to make a stack start near
identity. Option one is to add explicit *skip connections* — wrap each sub-block so its output is
x + F(x), which makes "do nothing" the structural default. Option two is a data-dependent init like LSUV
or an orthogonal init, which fixes the *scale* of activations layer by layer but still leaves each layer
a rotation, not a passthrough — so it does not actually solve the un-scrambling problem, only the
magnitude problem, which BatchNorm already handles here. Option three is to bake the identity directly
into the conv weights at initialization and change nothing about the architecture.

Let me weigh option one against option three, since option two clearly does not address the actual
complaint. A skip connection genuinely makes the block identity-friendly, but it changes the architecture
— it adds an add, a stored activation, and a second path through every block — and in a training that is
now only ~21 epochs and racing to get shorter, the weights never travel very far from their init, so the
*structural* insurance a skip provides against the weights wandering into a badly-conditioned
configuration is insurance I am unlikely to need. For a short run, the cheaper and more surgical move is
to fix the *starting point* and leave the architecture alone: if the layers begin as the identity and the
run is short enough that they stay near it, I get the benefit of "start near identity, learn the
correction" without paying for a permanent skip path. That is option three, and it costs exactly nothing
at run time — it is a one-line change to how the weights are initialized. So: Dirac init on the conv
weights, no architectural skip.

A 3×3 convolution can represent the identity exactly, which is what makes this possible: the Dirac-delta
kernel — 1 at the center tap of the diagonal input→output channel, 0 elsewhere — copies each input
channel straight to the corresponding output channel, and PyTorch's `torch.nn.init.dirac_` writes exactly
that. Let me verify the passthrough is exact and not merely approximate. A conv with a δ-kernel computes,
at each output position, the sum over the receptive field of kernel × input; with all kernel entries zero
except the center tap = 1 on the matching channel, that sum collapses to just the input value at that
same position and channel. With `padding='same'` the spatial size is preserved, so the output equals the
input, pixel for pixel, on the copied channels — a true identity, exactly, at step zero.

I should be honest that the *block* is not the identity even when its conv is, because there is a BN and a
GELU in the way, and I want to make sure that does not undercut the argument. After the Dirac conv comes
BatchNorm — with its scale frozen at 1, it subtracts the running mean and divides by the running std,
which is an affine rescaling of the passed-through signal, not a scramble; since the incoming signal is
already whitened and roughly unit-variance, BN leaves it close to itself. Then comes the GELU, which is a
genuine nonlinearity: GELU(x) ≈ x for large positive x and ≈ 0 for negative x, so the block as a whole is
*not* an identity map, and I am not claiming it is. What I am claiming is narrower and still exactly the
thing that matters: with a random conv, the signal arriving at each GELU is a *scrambled* version of the
whitened input, so the nonlinearity fires on garbage and the network must first learn to un-scramble;
with a Dirac conv, the signal arriving at each GELU is the *clean* whitened input, so the nonlinearity
fires on the real features from step one and the network only has to learn how to bend them. The GELU is
supposed to be there — it is the source of the network's expressive power — the point is only that it
should be applied to clean signal, not to noise, at initialization. Dirac guarantees that; random init
does not. The same holds across the `MaxPool2d(2)` that sits inside each block: an identity conv followed
by a pool still hands the pool the clean signal, and pooling a clean signal is fine.

Let me put a rough number on the scrambling I am removing, tracking one informative direction through the
stack. Say the whitened first conv has placed a class-discriminating edge feature into some channel at
unit strength. Pass it through a random 3×3 conv into a same-width layer: the output along any particular
channel is a random linear combination of all C input channels, so the fraction of that channel's output
variance that comes from *my* edge feature is on the order of 1/C — the feature is spread thinly across
all output channels and buried under the other C − 1 directions in each one. To recover it, downstream
layers must learn the specific combination that re-concentrates it. Stack two random layers and the
feature is spread across C² combinations; three, across C³. This is the compounding I described, made
concrete: the informative signal is not destroyed in an information-theoretic sense (a random rotation is
invertible), but it is *diffused*, and un-diffusing it is precisely the early-epoch work I want to
delete. A Dirac init keeps my edge feature sitting in its own channel at unit strength through every
layer — 1/1, not 1/C — so nothing has to be re-concentrated before learning can begin. The estimate is
crude (the layers are not exactly orthogonal and BN/GELU intervene) but the direction is unambiguous:
random init diffuses the signal by a factor that compounds with depth, Dirac init does not diffuse it at
all.

There is a second-order consequence worth naming, because it explains why "fewer epochs" should follow
rather than just "a nicer init." With random deep layers, the network's output at step zero is a
scrambled projection of a scrambled signal — the logits are essentially noise, the cross-entropy loss
sits near its maximum (ln 10 ≈ 2.30 for ten balanced classes, softened a little by the 0.2 label
smoothing), and the very first gradients are dominated by the need to un-scramble rather than by any
class signal. With Dirac init, the whitened input propagates cleanly to the flattened head, so the logits
at step zero are already a *structured* linear readout of real features; the starting loss is lower and,
more importantly, the starting gradient points at class-relevant directions from the first step instead
of at passthrough-reconstruction directions. That is the whole reason to expect *epochs* to fall and not
merely training to "look cleaner": the optimizer spends its ~1,000-step budget on the classification
problem from step one rather than burning the first few hundred steps re-establishing a signal path it
should never have lost.

There is a shape wrinkle, and checking it against the actual layer widths is the verification that the
scheme is even well-defined everywhere. The identity only makes literal sense when the number of output
channels is at least the number of input channels: with M inputs and N ≥ M outputs I copy each of the M
inputs to one of the N outputs (the first M filters get the identity) and leave the remaining N − M
outputs at their default random init — those are the "extra" feature detectors the layer is free to
learn. Do all my layers satisfy N ≥ M? Walking the stack after the whitening conv: 24 → 64 (M=24, N=64,
so 24 identity filters and 40 free), 64 → 64 (all 64 identity, none spare), 64 → 128 (64 identity, 64
free), 128 → 128 (all identity), 128 → 256 (128 identity, 128 free), 256 → 256 (all identity). Every
layer has N ≥ M, so the partial-identity init is well-defined at each one, with the widening layers
getting a mix of passthrough plus free capacity and the same-width layers getting a pure identity. Good —
no layer is left in an undefined state where I'd have more inputs than outputs to copy. The slice that
does this is `w[:w.size(1)]`, the first `in_channels` output filters; `dirac_` writes the identity into
them, and the rest of the weight tensor keeps its Kaiming values.

```python
class Conv(nn.Conv2d):
    def reset_parameters(self):
        super().reset_parameters()
        if self.bias is not None:
            self.bias.data.zero_()
        w = self.weight.data
        torch.nn.init.dirac_(w[:w.size(1)])
```

The first conv has to be excluded explicitly, and the reason is not arbitrary. It is the whitening layer,
and it is special in three ways that all say "leave it alone." It is frozen, so writing a Dirac into it
would be immediately overwritten by nothing — but more to the point, its *job* is decorrelation, not
passthrough: I want it to actively transform the input into a whitened basis, which is the opposite of
copying the input through unchanged. And it is a 2×2 stride-shrinking conv, not a 3×3 same-size conv, so
"identity via the center tap" is not even the right target for it — there is no center-of-nine tap and no
matched input/output channel structure (it goes 3 → 24). So the rule is precisely "Dirac on every
convolution *after* the first," and the whitening layer keeps the frozen eigenbasis I built for it in the
previous step. The two initializations are doing complementary jobs on adjacent layers: the first conv
transforms, everything after it passes through, and gradient descent learns the deviations from that
transform-then-passthrough starting configuration.

One more decision inside option three: pure identity on the first M filters, or a *blend* of identity and
random on all of them? Earlier schemes have tried mixing — initialize each filter as α·(identity) +
(1−α)·(random) — and I should say why I reject the blend here. The whole point is to make the deep layers
*transparent* at init so the whitened signal passes through clean. If I mix a fraction (1−α) of random
weight into the identity path, then the signal that comes through is α·(clean passthrough) + (1−α)·(random
scramble): I have re-injected exactly the noise I was trying to remove, just at reduced amplitude. With
α = 0.5, half the identity survives but a half-strength random rotation rides along on top of it, so the
signal-to-scramble ratio at the output of each layer is only about 1:1 — and stacked over six layers that
compounds. A pure Dirac on the first M filters keeps the passthrough at full strength (SNR effectively
infinite through the identity channels at step zero) while still leaving the extra N − M filters
completely free to learn new features. Pure passthrough on the first M, pure freedom on the rest — no
admixture — is strictly better for the transparency I am after. The `bias.data.zero_()` in the same
`reset_parameters` is part of the same idea: a nonzero bias would shift the passed-through signal off its
operating point before the GELU, so I zero it and let it be learned from a clean start.

I should be honest about how big a win to expect, and the honest expectation is "real but smaller than
whitening," for a reason I can argue from the mechanism. Whitening attacked the *input*, where the
redundancy was largest (κ in the hundreds) and the conditioning worst, so it bought the huge 45 → 21
jump. Dirac attacks the *deeper* layers, and by the time the signal reaches them it has already been
whitened once — so the random-scrambling waste is still real but it is operating on a signal that is no
longer pathologically ill-conditioned, and it is a smaller fraction of the total remaining work. So I
expect the epoch count to drop again below 21, but by a handful of epochs rather than another halving.
The two fixes should *compose* rather than fight: whitening conditions the input, Dirac keeps that
conditioning intact as the signal propagates downstream, so they are addressing sequential stages of the
same pipeline and their savings should stack roughly additively rather than one cannibalizing the other.
The falsifiable version: epochs-to-94% should come down from 21 with accuracy held at the bar, and if the
composition really is additive I should see a clean few-epoch reduction; if instead the extra-channel
random filters were quietly doing the heavy lifting anyway — if the network didn't actually need the
identity passthrough because the widening layers' free capacity covered for it — then the gain would be
slim and the epoch count would barely move. Per-epoch cost should be essentially unchanged, since Dirac
is an initialization and adds nothing to any forward or backward step — so unlike whitening, whose 7%
per-epoch dividend came from freezing a layer, this rung's whole effect should land in the epoch column
and none of it in the per-epoch column. That is a sharp diagnostic in its own right: if seconds were to
fall while epochs held, my accounting would be wrong; the two should move together this time, both driven
purely by needing fewer epochs.

I can set a rough magnitude even without the exact number. Whitening removed 24 epochs (45 → 21) by
fixing the input; Dirac removes deep-layer diffusion, which by the tracking argument above is real but
acts on an already-conditioned signal and only in the six convs after the first — so I would be surprised
by anything near another 24-epoch cut and unsurprised by a few epochs off 21. What the epochs-to-94%
table will actually teach me is whether the two initialization fixes compose additively (whitening's win
plus Dirac's win, cleanly stacked, because they condition sequential stages of one signal path) or
interact — Dirac mattering *more* now that there is a clean signal downstream worth preserving, or *less*
if whitening already delivered most of the conditioning the deep layers needed. The one-line change is
`dirac_` on the first M filters of every conv after the first (the frozen whitening layer keeps its
special init); the code is in the answer, and the epochs-to-94% table is what will settle whether the
deep-layer scrambling was really costing me epochs.
