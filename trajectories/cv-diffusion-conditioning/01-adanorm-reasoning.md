The conditioning operator is the whole task, and before I can compare operators I need a floor — the
first concrete fill of `prepare_conditioning` plus `ClassConditioner` that turns the scaffold's
unconditional UNet into a class-conditional one. I want to start from the operator whose *cost and
structure are the tightest match to what the conditioning signal actually is*, so that whatever it leaves
on the table is a clean diagnosis of what the next rung must add. So let me reason carefully about what a
class label *is*, in this denoiser, and which mechanism fits it.

What I have is a noise-prediction UNet, $\epsilon_\theta(x_t, t)$: feed it a noised CIFAR image and the
timestep, it predicts the noise. The training loss is the friendly one — draw $x_0$, a timestep $t$, a
noise vector $\epsilon$, form $x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon$ in one
shot, regress $\lVert \epsilon - \epsilon_\theta(x_t,t)\rVert^2$. The third argument is what I'm adding:
a class index $c$, so the same noised latent at the same noise level denoises differently toward a
"horse" than toward a "ship." Crucially the backbone already conditions on *one* scalar side input — the
timestep $t$ — and it does so through the diffusers UNet's own machinery: $t$ becomes a sinusoidal
embedding, the time-embedding MLP lifts it to `time_embed_dim`, and inside every residual block that
vector drives the block's adaptive normalization (the diffusers `ResnetBlock2D` consumes `temb` and turns
it into a per-channel scale/shift on the block's GroupNorm). So the UNet has a *finished, tuned socket*
for a global per-channel side signal. The class label is exactly that shape — one global per-example
vector — so the question I must answer first is how that signal should be carried.

Let me characterize the class signal precisely, because the right operator follows from its structure. A
class index has no spatial layout — there is no "where" to a label, it applies to the whole image — and
no internal structure: it is one categorical value, which I embed into one vector. It is *global* (the
same for every spatial position), *per-channel* (its natural action is "turn some feature channels up,
others down"), and *content-blind* in the sense that it is drawn from the label alone, not from what is
currently forming in the feature map. That is the same profile as the timestep. Now let me actually walk
the operator families the prior art hands me and cost each against this profile, rather than asserting the
answer. Concatenation would broadcast the class vector into a constant $[B, d, H, W]$ feature slab and
stack it on the input before a conv — but a label has nothing image-shaped to concatenate; I would be
manufacturing $H\times W$ copies of one vector purely to feed a conv that then has to learn to ignore the
spatial dimension I invented. Wasteful, and it forces the signal in at the input only, where a
convolutional stack must then transport it through every resolution rather than re-injecting it. Bounded
gating, $F \mapsto \sigma(g(z))\odot F$ with $\sigma\in(0,1)$, can only *attenuate* channels — it cannot
add, cannot amplify past the current magnitude, and $\sigma$ saturates, so it is a strictly weaker affine
than one that also shifts. Attention would treat the class as a token to be read from, position by
position, with content-dependent weights — but reading from a *set* is what attention buys, and here the
set has one element, so its content-dependent weighting has nothing to weigh; most of that machinery would
sit idle for a structureless label, and it is the most expensive of the four. That leaves the operator
that takes the conditioning vector and *modulates every channel of every spatial location in the same
way*: an adaptive affine. It matches the profile exactly — global, per-channel, location-agnostic — and
nothing about it is idle for a single label.

This is the FiLM primitive, and the conditioning lineage converged on it from several directions, which
is why I trust it as the floor. FiLM states it generally: a conditioning input $z$ regresses a
per-channel scale and shift, $\mathrm{FiLM}(F) = \gamma(z)\cdot F + \beta(z)$, applied wherever there is a
feature map, with parameter count tied to channels (not resolution) so it is cheap to apply repeatedly.
Concatenation-then-linear reduces to the $\gamma = 1$ corner ($W[F;z] = W_F F + W_z z$, an additive
conditional bias); bounded gating is the $\beta = 0$, squashed-$\gamma$ corner. Conditional BatchNorm,
Conditional Instance Norm, and AdaIN are all the same affine attached to one normalization or another.
And most on-the-nose for a denoiser, ADM's adaptive Group Norm is exactly this affine living inside the
residual block, driven by the combined timestep-and-class embedding — and that recipe ablated it against
the older "add the embedding, then normalize" move and found the adaptive affine better. So the
established answer to "how do I inject a global side signal into a diffusion residual block" is "modulate
a normalization's affine from it." That is the family I will draw the floor from.

Now I have to make two design decisions that the FiLM family leaves open, and these are where the floor
earns or loses its keep. First: *where* does the class signal enter — the time path
(`prepare_conditioning`), the post-block path (`ClassConditioner`), or both? The scaffold gives me two
sockets. The path-of-least-resistance is to fold the class into the time embedding and let the UNet's own
in-block AdaGN carry it — that is the additive `time_emb + class_emb` move, and it is the road I mean to
try *next*, once I have a floor to measure it against, so I will deliberately not take it here. Instead I
want the floor to be a *self-contained, post-block adaptive-norm conditioner*: leave `prepare_conditioning`
an identity (`return time_emb`, so the timestep keeps its own tuned path untouched and the block AdaGN
stays a pure noise-level signal), and put *all* the class conditioning into the `ClassConditioner` module
applied after each block, as an explicit adaptive normalization on the `[B,C,H,W]` feature map. The
class never touches the time embedding; it acts only through a dedicated affine modulation of the
post-block features. This makes the floor a clean test of the post-block adaptive-norm operator on its
own — one thing conditioned in one place, so whatever it fails to buy is unambiguous.

Second decision, and the one I want to nail because it bites at initialization: the affine's
parameterization and its starting point. The naive form $\gamma(z)\cdot F + \beta(z)$ with a freshly
initialized regressor sits near $\gamma \approx 0, \beta \approx 0$, which *annihilates* the feature map
on step one. Let me actually check what that does to both the forward value and the gradient, because
"annihilate" needs to mean something. Forward: with $\gamma=\beta=0$ the modulated feature is $0\cdot F +
0 = 0$, so the whole feature map goes to zero at init — the block's output is destroyed before training
sees a single useful signal. Gradient: the local derivative of the modulated feature with respect to the
incoming $F$ is exactly $\gamma$, so at $\gamma\approx 0$ the gradient that flows back *through* the
feature is also near zero — the layer is not just wrong at init, it is nearly ungradient-able through its
main path. The identity affine is not $\gamma = 0$; it is $\gamma = 1, \beta = 0$. So I parameterize the
scale as a *deviation from one*: write the modulation as $(1+\text{scale})\cdot \hat F + \text{shift}$
where $\hat F$ is the normalized feature, so that $\text{scale}=\text{shift}=0$ leaves the normalized
feature unchanged and the regressor learns only how far to push *away* from "no modulation." Concretely:
if the useful gain for some channel turns out to be $1.5\times$, a bare-$\gamma$ regressor has to learn to
emit $1.5$ from a fresh init at $0$, climbing across the dead zone; the $(1+\text{scale})$ regressor has
to learn to emit $0.5$, and at init it emits $0$, which is the *neutral* pass-through $\hat F$ rather than
the annihilating $0$. Same expressible set of affines, a sane origin instead of a destructive one.

But there is a deeper initialization concern, because I am inserting a *brand-new* conditioning sublayer
into the middle of a backbone whose weights and learning-rate schedule were tuned for a denoiser. If I
drop a randomly-initialized affine modulator after each block, at step zero it injects a random
perturbation into a carefully balanced residual stream — I would be throwing away the good initialization
and forcing training to first undo the damage. The principled fix is the zero-init residual trick
(zeroing the last scale of a residual branch so the branch starts as the identity — Goyal et al. 2017's
last-BN-$\gamma$-zero, and the diffusion UNet's zero-last-conv). I want the inserted conditioner to be the
*exact identity at init* and to grow its effect only as training finds a use for it. The clean way is a
per-channel *gate* on the modulated-minus-original difference, regressed from the class like scale and
shift, positioned right at the residual add, and *zero-initialized*. With $\text{gate} = 0$ the module
returns its input bit-for-bit; the network is the original denoiser at init; and as the gate moves off
zero the class conditioning ramps up smoothly. This is adaptive normalization, zero-initialized —
adaLN-Zero — but rendered here on a convolutional feature map rather than on a transformer token stream.

Let me write the operation concretely against the substrate and trace it, because the gating interacts
with the parameterization in a way I want to be sure I understand before I trust it. The provided
`AdaLNBlock(channels, cond_dim)` is precisely this: a GroupNorm of the feature map, then a zero-initialized
`SiLU → Linear(cond_dim, 3·channels)` that emits scale, shift, and gate, applied as
$$x \;+\; \text{gate}\cdot\big((1+\text{scale})\cdot\mathrm{norm}(x) + \text{shift} - x\big).$$
Take the whole regressor zero-initialized, so at init the three vectors are $\text{scale}=\text{shift}=
\text{gate}=0$. The update becomes $x + 0\cdot((1+0)\hat x + 0 - x) = x + 0 = x$: the identity, exactly, at
every element, before any training — so the network is bit-for-bit the tuned denoiser and the good init is
preserved. Now the gradients, which is where I want to be careful, because a zero-init module could in
principle be a *dead* module. Write the branch as $o = x + g\,\big((1+s)\hat x + b - x\big)$ with $g, s, b$
the gate/scale/shift, each the output of the zero-init Linear applied to $\phi = \mathrm{SiLU}(\text{class
emb})$. The three local sensitivities are $\partial o/\partial g = (1+s)\hat x + b - x$, which at init
$(s{=}b{=}0)$ is $\hat x - x$ and is *nonzero* in general; $\partial o/\partial s = g\,\hat x$, which at
init is $0$; and $\partial o/\partial b = g$, which at init is $0$. Push these back to the Linear's weight
rows: $\partial o/\partial W_g = (\hat x - x)\,\phi^\top \neq 0$, but $\partial o/\partial W_s = 0$ and
$\partial o/\partial W_b = 0$. So the module is not dead — but only the *gate* row receives gradient at
init; scale and shift are frozen until the gate becomes nonzero. That is a genuine two-phase dynamic worth
naming: phase one, the gate learns "how much of the $\hat x - x$ direction is worth blending in at all,"
which is a class-dependent nudge toward the globally-normalized feature; phase two, once the gate is off
zero, scale and shift wake up and the modulation acquires real per-channel structure. It means the floor
does not condition instantly — it has to spend early training climbing the gate off zero before the affine
proper can even begin to learn. I will remember that when I read where this operator underperforms; the
climb-off-zero cost is real and it is paid per inserted conditioner.

Let me also pin the shapes and the normalization choice, since they carry an inductive-bias decision I
should be deliberate about. The feature map is $[B, C, H, W]$. The `AdaLNBlock`'s norm is `GroupNorm(1)` —
a single group over all $C$ channels — so the statistics are pooled across the *entire* channel-and-spatial
extent per sample, giving $\hat x$ a global normalization (this is LayerNorm-over-$(C,H,W)$ in GroupNorm
clothing, not the diffusers default of many groups). That reinforces exactly the profile I want: the class
affine acts on a globally-normalized feature and pushes it globally, matching the global, structureless
nature of the label. The `Linear(cond_dim, 3C)` consumes the class embedding at width `cond_dim` (which the
substrate sets to `time_embed_dim`) and emits $3C$ numbers, split into scale/shift/gate of shape $[B, C]$
each, broadcast to $[B, C, 1, 1]$ against the map — one number per channel, none per position, which is the
whole point: no spatial address. The residual add returns $[B, C, H, W]$, unchanged in shape, so the module
slots after each block invisibly to the rest of the wiring.

There is one interaction I should check before trusting the placement, because I am inserting a
normalization *right after* a block whose own AdaGN has just written a per-channel scale/shift into the
feature. Does my `GroupNorm(1)` undo the timestep's modulation? Partly, inside the branch: re-normalizing
$x$ to $\hat x$ does strip the magnitude the block's AdaGN just imposed, so $\hat x$ has lost the
timestep's per-channel gain. But the update is $x + \text{gate}\cdot((1+\text{scale})\hat x + \text{shift}
- x)$, and the leading $x$ — the block's output with the timestep modulation fully intact — is carried
untouched through the residual. Only the gated *correction* is computed on the re-normalized $\hat x$, and
at init the gate is zero so the correction is nil. So the timestep's work is never destroyed; the class
conditioner adds a class-dependent term on top of a preserved noise-level-modulated feature. That is the
behaviour I want, and it is only true because the operation is residual — a non-residual "normalize then
re-modulate" would indeed have clobbered the timestep signal. Good that the substrate's block is residual.

It is also worth being precise about *why* I call this operator low-bandwidth, because it is not a lack of
representational capacity. The affine offers $2C$ degrees of freedom per block (a scale and a shift per
channel), and $2C$ is enormous next to the $\log_2 10 \approx 3.3$ bits needed to name which CIFAR class I
am in — there is no shortage of parameters to *distinguish* the ten classes. The limit is not "can the
affine tell horse from ship" but "can the affine act *differently at different places in the picture*,"
and the answer is structurally no: the same scale and shift hit every spatial position, and neither reads
the local content. So the ceiling I am predicting is an *addressing* ceiling, not a capacity ceiling, and
that distinction is what tells me the fix has to change the operator's spatial/content access rather than
just give it more parameters.

I should also check the parameter cost, because the harness enforces a $1.05\times$-of-cross-attention-
reference budget and I do not want the floor to blow it. Per inserted conditioner the cost is the one
`Linear(cond_dim, 3C)`: about $3\,\text{cond\_dim}\cdot C$ weights. At Small, `cond_dim` is
$64\times4 = 256$, and the block channels run $\{64,128,128,128\}$, so the widest conditioner is roughly
$3\cdot256\cdot128 \approx 98\text{k}$ parameters, the narrowest ($C{=}64$) about $49\text{k}$; summed over
the handful of down/mid/up blocks the whole set of AdaLN regressors is under a million against a $\sim$9M
backbone — call it a few percent. Sanity against the block it lives beside: a resnet block's two $3\times3$
convs at $C{=}128$ are $\sim 2\cdot 9\cdot128^2 \approx 0.3$M, so the AdaLN regressor is a fraction of the
block it modulates, and it scales as $O(\text{cond\_dim}\cdot C)$ rather than the conv's $O(C^2)$, so its
share *shrinks* at the wider scales. And it is comfortably lighter than the cross-attention reference the
budget is measured against, whose per-block projections cost $\sim 2C^2 + 2\,\text{cond\_dim}\cdot C$. So
the floor fits the budget with room to spare; the budget will bind on a heavier operator, not on this one.

I should be explicit about what this *is not*, because the same name elsewhere means a much larger thing.
The canonical adaLN-Zero is the conditioning mechanism of a *transformer* diffusion backbone: it replaces
each LayerNorm's learned affine in a stack of ViT blocks, sums the timestep and class embeddings into one
conditioning vector $c = t_{\text{emb}} + c_{\text{emb}}$, regresses six modulation vectors per block
(scale/shift/gate for the attention sublayer and for the MLP sublayer) from a single `Linear(d, 6d)`,
adds a gateless adaptive-norm decode head, and zero-initializes all of it so the *entire block stack* is
the identity map at start, all to inherit transformer scaling. None of that applies here. The harness has
no transformer — the backbone is a fixed convolutional UNet I am forbidden to touch. There is no token
stream, no patchify, no final decode head to condition, and no $c = t+y$ sum (I am keeping the timestep
and class on *separate* paths by choice). What the harness exposes is exactly one thing: a post-block hook
that takes a `[B,C,H,W]` feature map and the class embedding. So I am porting *only* the core
adaLN-Zero operation — normalize, regress $(1+\text{scale})\cdot\hat x + \text{shift}$, gate it onto the
residual, zero-init the regressor so the block starts as the identity — onto a convolutional feature map
(GroupNorm in place of LayerNorm), inserted after each UNet block, carrying the class signal alone. The
transformer machinery the canonical method is famous for is simply not present, and importing its story
would mis-describe what runs here.

So the step-1 edit is settled: `prepare_conditioning` returns `time_emb` unchanged, and `ClassConditioner`
wraps one zero-initialized `AdaLNBlock` per block, modulating each post-block feature map from the class
embedding (the full scaffold module is in the answer). The training objective is unchanged in shape — the
plain noise-MSE — only the network now sees the class through these adaptive-norm gates; sampling feeds
the fixed class index at every DDIM step.

Now reason about what this floor should do and where it should run out, because that sets up the next
rung. The adaptive-norm conditioner gives, per channel, one scale and one shift, computed from the label
and applied identically at *every* spatial location. For steering toward one of ten classes that is real
but blunt: the class can say "globally emphasize channel 37, suppress channel 12," and through the gate it
can do so with progressively more strength as training proceeds — but it cannot say "near this corner of
the image emphasize this, over there emphasize that," because the modulation has no spatial address and
never looks at the local content of the feature it modulates. It is a global, content-agnostic per-channel
knob, the same profile as the timestep. For CIFAR-10's ten coarse classes that should already produce a
genuinely class-conditional model with a sensible FID at every scale — the floor *works*, it is not
broken. But there are two structural reasons I expect it to be the *weakest* of the conditioning operators
I will try. First, I deliberately routed the class only through the *post-block* path while leaving the
timestep alone, so the class signal reaches the residual blocks' own AdaGN not at all — it only gets to
nudge the feature *after* each block, which is a strictly thinner channel than letting the class ride the
block-internal modulation the way the timestep does, and (from the gradient trace above) that thin channel
even has to climb the gate off zero before it conditions at all. Second, even granting that, the operator
is spatially uniform: it is the lowest-bandwidth of the affine, attention, and combined options, exactly
the mechanism that the strongest pixel models found insufficient on its own. So my falsifiable expectation
is that this adaptive-norm-only conditioner lands a real but comparatively *high* FID at all three scales.
I cannot yet say *how* high — there is no prior number to anchor to — so I will state the shape of the
prediction instead: absolute FID should fall from Small to Large simply because bigger backbones denoise
better, and the floor should sit *above* whatever a better-routed or richer operator reaches at each scale.
The discriminating readout for the next step is already set up: if a cheaper operator that simply rides the
class on the timestep embedding beats this, that tells me the post-block-only routing was the bottleneck;
if a content-dependent operator beats it, that tells me the spatial uniformity was. Either way the
diagnosis for step 2 is already pointed: the question is whether the next gain comes from a *better route*
for the same affine, or from a *richer* operator — and the cheapest thing to try first is the better route.
