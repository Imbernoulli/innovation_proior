Standard settled the placement question, and its numbers told me the ladder has converged on a
*local* optimum in a one-dimensional search. Standard came in at 18.60 small, 8.47 medium, 7.18
large — and crucially it beat full-attn at the two larger scales (8.47 < 8.82, 7.18 < 7.31) while
full-attn held the small scale (17.44 < 18.60). That is exactly the prediction from the last rung
confirmed: concentrating attention at 16×16 is the sweet spot per unit of quadratic cost at the
capacity-rich scales, and full-attn was over-provisioned at the fine maps — except at the small
scale, where the thin model's extra attention at 32×32 was substituting for missing convolutional
capacity. So the ladder so far has answered one question — *where to place a single attention
resolution* — and the answer is 16×16. But notice what the whole ladder has been varying: only the
`down_block_types`/`up_block_types` tuple, only attention placement, only a binary on/off at each of
four resolutions. Every rung kept the residual block internals, the head count, the timestep
injection, and the resolution-change mechanism exactly as the original DDPM denoiser fixed them. The
local optimum the ladder found is the best *attention-placement* under those frozen internals. The
question I have not asked is whether those internals are themselves leaving FID on the table.

There is a sharp tension in standard's numbers that says they are. Standard wins the large scale but
*loses* the small scale to full-attn. Read that carefully. At small scale the thin model benefits
from attention at *more* resolutions; at large scale it benefits from attention at *fewer*. A single
on/off attention tuple cannot satisfy both — whichever placement I pick is wrong at one end of the
budget range. That is the signature of a design that is under-parameterized in the wrong dimension:
I have been trading attention breadth against cost with a blunt instrument (which resolutions get a
block at all), when the real levers are finer. If I could make the attention itself *cheaper and more
expressive per resolution*, I could afford it at more resolutions without the full-attn cost, and get
the small-scale benefit of broad attention and the large-scale benefit of efficient attention at
once. So the next move is not another point in the placement search — it is to improve the things the
placement search held constant.

Let me reason about each frozen internal and whether it is actually optimal, because I refuse to
change them blindly. Start with the attention itself. Every rung used the harness default head
configuration — effectively a small fixed per-head width applied uniformly, and on the original DDPM
denoiser a *single* head at the one attention resolution. A single head forces every output position
to use one relevance pattern, one "which other positions matter" map, shared across all the relations
an image contains — the symmetric half of an object, its far edge, a repeated region. Those relations
are plural and a single softmax map is a compromise across them. Multi-head attention splits the
channel embedding into several heads, runs an independent scaled-dot-product attention in each
lower-dimensional subspace, and concatenates, so different heads specialize to different relations at
once — at nearly the cost of one full-width attention. So more heads should help. But *how* to set the
count? Two parameterizations: fix the number of heads, or fix the channels per head. The per-head
query/key dot product is a comparison in the head's subspace; too wide (few heads) and I have few
distinct relevance patterns; too narrow (many heads) and each comparison is in a space too cramped to
discriminate. There is a sweet spot in channels-per-head, and fixing *channels-per-head* (rather than
head count) is the cleaner scaling rule — a head always compares in a fixed-dimensional subspace
regardless of how wide the feature map is, and the head count grows automatically on the wider,
coarser maps where I want more relevance patterns. I will fix 64 channels per head: it sits at the
wall-clock sweet spot and matches the Transformer convention, so the comparison dimension is the one
already known to work. This is strictly more expressive than the single-head default at the same FLOP
budget, which is the kind of change that lets me afford attention more broadly.

Now the placement, revisited with cheaper-per-resolution attention in hand. The ladder's lesson was
that 16×16 is the single best resolution and 32×32 is over-provisioned — *under the frozen single-head
internals and a binary on/off choice*. But the small-scale loss to full-attn says broad attention
does help when the model is thin, and multi-head attention is cheaper-per-coherence than the
single-head version the ladder priced. So the right placement is not "16×16 only" and not "everywhere
including the maximal-cost 32×32 input map" — it is attention at the coarser feature tiers where
global structure is legible and the cost is bounded: the 32×32, 16×16, and 8×8 *feature* resolutions
(which on this UNet are the three finer-than-bottleneck tiers), leaving the very finest map's purely
local texture to convolution. Wait — on a 32×32 image the input-resolution feature map *is* 32×32, so
"attention at 32×32" here means the top feature tier. The cost-vs-structure argument still holds: I
want attention where features carry global meaning and N² is affordable, and I want to skip it only
where it is both expensive and useless. With multi-head 64-channel attention the cost at the top tier
is bounded enough to be worth the coherence the small-scale result says it buys, so spreading
attention across 32/16/8 — more than standard's single resolution, but disciplined unlike full-attn's
uniform everywhere — should give the small-scale breadth and the large-scale efficiency together.

Next frozen internal: how the timestep gets into the network. Every rung used additive injection —
the timestep embedding projected and *added* into each residual block. Addition is the weakest
conditioning: it shifts the activations by a t-dependent bias and nothing more. But the denoising
behavior should change in *gain*, not just offset, across the noise schedule — near a clean image the
network makes tiny corrections, near pure noise it must invent structure — and a bias cannot rescale
a channel as a function of t. What I want is for the embedding to control both a per-channel scale and
a per-channel shift of the normalized activations. That is the FiLM idea, and its normalization
incarnation is adaptive group normalization: after the group norm inside the residual block, apply a
t-dependent affine, scale·GroupNorm(h) + shift, where [scale, shift] is a linear projection of the
timestep embedding. A learned per-channel gain is strictly more expressive than a learned per-channel
offset, at trivial extra cost, and it lets the single shared network re-gain itself across the noise
levels rather than only re-bias itself. I make it a perturbation of identity — modulate by
(1 + scale) — so at init the block behaves like an unmodulated normalized block and the conditioning
grows in only as it earns its keep. This is the same identity-at-init discipline that makes adding
any new mechanism safe, and it should lower FID over the additive injection for the same reason FiLM
beats a bias in conditional vision generally.

I also considered the resolution-change mechanism and the residual scaling, and I should be honest
about which of those this edit surface lets me reach. Folding up/downsampling into a BigGAN-style
residual block — so the resampling itself gets a residual identity path and a clean gradient — is a
genuine improvement to the spine of the network, but it is not a knob `UNet2DModel` exposes; the
library does its resampling with the standard conv resample, so I cannot express the BigGAN-in-residual
variant through this `build_model` without abandoning `UNet2DModel` for a fully custom module. I note
this as a real piece the harness omits, not something I am choosing to skip. The 1/√2 residual rescale
is the other candidate, and there I have a reason *not* to want it: it is variance control, and with
group normalization already re-standardizing activations in every block, the √2 rescale is at best
redundant and can fight the normalization — so even if I could set it, I would leave it off. So the
changes I can faithfully land through this scaffold are the three that matter most for FID on this
harness: multi-resolution attention at 32/16/8, multi-head attention at 64 channels per head, and
adaptive group normalization for the timestep injection. Plus a small dose of dropout (0.1) inside
the residual blocks, the standard regularizer for a same-resolution image backbone that otherwise
overfits CIFAR with the kind of artifacts an unregularized model shows.

So the finale is a single `build_model` that keeps `UNet2DModel` but changes its internals along the
three reachable axes: `down_block_types` and `up_block_types` with attention-bearing blocks at the
three finer resolutions (32/16/8) and pure convolution only at the coarsest 4×4 tier;
`attention_head_dim=64` so each attention layer is multi-head with 64 channels per head;
`resnet_time_scale_shift="scale_shift"`, which the library maps to adaptive group normalization
(the FiLM-style per-channel scale-and-shift conditioning I derived); and `dropout=0.1`. Everything
outside the network — the ε-MSE loss, the linear 1000-step schedule, 50-step DDIM, AdamW, the EMA,
clean-fid — is the fixed substrate, untouched, so this is still purely an architecture change and the
comparison to the baselines stays honest.

What bar does this have to clear, and what would I validate? Standard, the strongest baseline, is
18.60 small, 8.47 medium, 7.18 large; full-attn is 17.44/8.82/7.31. The finale has to beat the *best*
baseline at each scale — so it must come in under 17.44 at small (where full-attn leads), under 8.47
at medium and under 7.18 at large (where standard leads). My falsifiable expectation, scale by scale:
at the small scale, the combination of broader multi-resolution attention (the breadth full-attn's
small-scale win showed helps a thin model) made cheaper by multi-head 64-channel attention, plus the
stronger AdaGN conditioning, should push below full-attn's 17.44 — this is the scale where I expect
the clearest improvement, because the baselines were forced to choose between breadth and efficiency
and the finale gets both. At medium and large, the targeted-but-broader placement should at least
match standard's disciplined 16×16-only placement, and AdaGN plus multi-head should edge it below
8.47 and 7.18 — though here the headroom is thinner, because standard was already near the
attention-placement optimum, so the win has to come from the conditioning and the head configuration
rather than the placement. The risk I would watch is that at large scale, where capacity is abundant
and standard is already strong, the added attention at the 32×32 tier reintroduces full-attn's
over-provisioning cost without enough coherence benefit to pay for it — in which case the finale would
match standard at large but not clearly beat it, and the real gains would concentrate at small and
medium. And the honest caveat on the comparison is the BigGAN-resampling piece the harness cannot
express: the published recipe's single biggest architectural win is the residual up/downsampling I
cannot reach through `UNet2DModel`, so this finale lands the FID-relevant subset that *is* reachable —
multi-resolution multi-head attention and AdaGN — and the bar it must clear is whether that subset,
alone, beats the best of the placement-only baselines at every scale.
